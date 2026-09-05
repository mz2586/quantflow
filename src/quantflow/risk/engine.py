"""The risk engine.

**Every** order in the system passes through :meth:`RiskEngine.approve`. There is no other
path from a strategy signal to an exchange — not in backtesting, not in paper trading, not
in live. That single invariant is what the rest of the risk design rests on.

The engine turns an unsized :class:`~quantflow.domain.signals.Signal` into either a sized,
protected, venue-legal :class:`~quantflow.domain.orders.OrderRequest`, or a documented
refusal. It never returns a partially-checked order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.clock import Clock, SystemClock, start_of_utc_day
from quantflow.core.config import RiskSettings
from quantflow.core.errors import RiskViolationError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO, round_price
from quantflow.domain.enums import OrderSide, OrderType, SignalDirection
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.persistence.database import Database
from quantflow.risk.conviction import allocation_fraction, percentile_of
from quantflow.risk.correlation import CorrelationMatrix
from quantflow.risk.killswitch import KillSwitch
from quantflow.risk.rules import RiskContext, RiskRule, RiskVerdict, build_default_rules
from quantflow.risk.sizing import PositionSizer, SizingRequest, SizingResult, build_sizer
from quantflow.risk.targets import cost_aware_target

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The engine's answer for one proposed trade."""

    approved: bool
    request: OrderRequest | None = None
    sizing: SizingResult | None = None
    verdicts: tuple[RiskVerdict, ...] = field(default_factory=tuple)
    reason: str = ""
    halted_trading: bool = False
    engaged_kill_switch: bool = False

    @property
    def denials(self) -> tuple[RiskVerdict, ...]:
        """Every rule that refused."""
        return tuple(verdict for verdict in self.verdicts if not verdict.allowed)

    @property
    def blocking_rule(self) -> str | None:
        """The first rule that refused, if any."""
        denials = self.denials
        return denials[0].rule if denials else None

    def raise_if_denied(self) -> OrderRequest:
        """Return the approved request, or raise.

        Raises:
            RiskViolationError: if the trade was refused.

        """
        if not self.approved or self.request is None:
            rule = self.blocking_rule or "unknown"
            raise RiskViolationError(self.reason or "risk engine denied the order", rule=rule)
        return self.request

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging, the API and the audit trail."""
        return {
            "approved": self.approved,
            "reason": self.reason,
            "blocking_rule": self.blocking_rule,
            "halted_trading": self.halted_trading,
            "engaged_kill_switch": self.engaged_kill_switch,
            "denials": [
                {
                    "rule": verdict.rule,
                    "message": verdict.message,
                    "observed": str(verdict.observed) if verdict.observed is not None else None,
                    "limit": str(verdict.limit) if verdict.limit is not None else None,
                }
                for verdict in self.denials
            ],
            "quantity": str(self.request.quantity) if self.request else None,
        }


#: How far inside the touch a passive entry is posted, as a fraction of price.
#:
#: The entry price is the bar's CLOSE, and the order reaches the venue after it. If the
#: market has moved even slightly against the passive side in that gap, a post-only order
#: priced at the close is now aggressive and the venue refuses it outright.
#:
#: Measured on this account 2026-08-17/18: **7 of 15 entry attempts were cancelled** that
#: way — fully qualified setups, past every gate, refused purely on price staleness. At the
#: session's +9.47 expectancy per trade that is roughly 66 USDT of foregone profit, and
#: recovering it requires no gate to be loosened.
#:
#: Two basis points is below any meaningful move on a 15m bar yet comfortably clear of the
#: spread on BTC and ETH, so the order rests as a maker instead of being rejected. It is
#: deliberately small: a wider offset would fill less often and, when it did fill, would
#: fill because the market ran past it.
PASSIVE_ENTRY_OFFSET_PCT = Decimal("0.0002")


def as_maker_entry(
    order_type: OrderType,
    *,
    limit_price: Decimal | None,
    reference_price: Decimal,
    enabled: bool,
    is_entry: bool = True,
    side: OrderSide | None = None,
    price_tick: Decimal | None = None,
    passive_offset_pct: Decimal = PASSIVE_ENTRY_OFFSET_PCT,
) -> tuple[OrderType, Decimal | None, bool]:
    """Convert an aggressive entry into a passive one, or leave it alone.

    Applied at the single point where a Signal becomes an OrderRequest, so all twenty-two
    strategies get maker pricing without any of them knowing about it, and there is exactly
    one place this can be wrong.

    Four things are deliberately NOT converted:

    * **Exits.** A reduce-only order waiting for a passive fill is not protection. The fee
      saved is bounded; the loss from a stop that never fills is not.
    * **Stop entries.** A stop is triggered *by* price moving away from you. Quoting
      passively at the same time contradicts the entry it is trying to make.
    * **A strategy's own limit price.** If a strategy chose a price, it knows something
      this layer does not; it becomes post-only but keeps its price.
    * **Anything at all when disabled**, so the default path is byte-identical to before.

    Returns:
        ``(order_type, price, post_only)`` for the request.

    """
    if not enabled or not is_entry:
        return order_type, limit_price, False
    if order_type.requires_trigger_price:
        return order_type, limit_price, False
    if order_type is OrderType.LIMIT and limit_price is not None:
        return order_type, limit_price, True
    # Just inside the touch, on the passive side. Resting exactly AT the close was the
    # defect: the close is already stale by the time the order lands, so any adverse tick
    # turns a passive order into an aggressive one and the venue cancels it rather than
    # charging taker. A small offset keeps it a genuine maker order without chasing.
    if side is None or passive_offset_pct <= ZERO:
        return OrderType.LIMIT, reference_price, True
    offset = reference_price * passive_offset_pct
    passive = reference_price - offset if side is OrderSide.BUY else reference_price + offset
    # Snapped to the venue's tick grid. An offset computed as a fraction of price almost
    # never lands on the grid, and the venue rejects the whole order outright: on
    # 2026-08-18 this produced "price 64296.63810 is not a multiple of tick 0.10" and
    # blocked eleven consecutive candidates. round_price rounds a buy DOWN and a sell UP,
    # which is the passive direction here, so snapping cannot make the order aggressive.
    if price_tick is not None and price_tick > ZERO:
        passive = round_price(passive, price_tick, side_is_buy=side is OrderSide.BUY)
    return OrderType.LIMIT, passive, True


def _expected_net_edge(
    *, side: OrderSide, entry: Decimal, target: Decimal | None, cost_rate: Decimal
) -> Decimal | None:
    """Expected return to target, less the round-trip cost, as a fraction of notional.

    ``None`` when there is no target to measure against — an unknown edge is not treated as
    a positive one, so it cannot unlock a larger position.
    """
    if target is None or entry <= ZERO:
        return None
    move = (target - entry) if side is OrderSide.BUY else (entry - target)
    return (move / entry) - cost_rate


def entry_has_expired(*, bars_resting: int, max_bars: int) -> bool:
    """Whether a passive entry has rested too long to still be acted on.

    A post-only entry is a bet on a price *and a moment*. The strategy's stop, target and
    size were all computed from the bar that produced the signal, so a fill several bars
    later is a position taken on analysis that no longer applies — entered at a price the
    market has already left behind, protected by levels chosen for different conditions.

    Missing the trade costs nothing. Taking an expired one costs whatever moved in between,
    which is why this bounds the wait rather than chasing the price.

    A limit of zero or less means "do not rest at all", and is deliberately distinguished
    from an absent limit: read through a falsy check, zero would mean rest forever, which
    is the opposite of what it says.
    """
    if max_bars <= 0:
        return bars_resting > 0
    return bars_resting > max_bars


class RiskEngine:
    """Sizes, protects and validates every order before it can reach a venue."""

    __slots__ = (
        "_clock",
        "_consecutive_losses",
        "_correlations",
        "_database",
        "_entry_scores",
        "_halted_until",
        "_kill_switch",
        "_last_loss_at",
        "_notifier",
        "_order_timestamps",
        "_rules",
        "_session_id",
        "_settings",
        "_sizer",
        "_thesis_failures",
        "_week_start_equity",
        "_week_started_at",
    )

    def __init__(
        self,
        settings: RiskSettings,
        *,
        sizer: PositionSizer | None = None,
        rules: list[RiskRule] | None = None,
        kill_switch: KillSwitch | None = None,
        database: Database | None = None,
        clock: Clock | None = None,
        session_id: str | None = None,
        notifier: Any | None = None,
    ) -> None:
        self._settings = settings
        #: Optional dispatcher. Alerting is best-effort and must never affect a decision.
        self._notifier = notifier
        self._clock = clock or SystemClock()
        self._sizer = sizer or build_sizer(settings)
        self._rules = rules if rules is not None else build_default_rules()
        self._kill_switch = kill_switch or KillSwitch(database, clock=self._clock)
        self._database = database
        self._session_id = session_id
        self._order_timestamps: list[datetime] = []
        #: UTC day for which trading is halted, if any.
        self._halted_until: datetime | None = None
        #: Equity at the start of the rolling seven-day window, and when it was taken.
        self._week_start_equity: Decimal | None = None
        self._week_started_at: datetime | None = None
        #: Losing trades closed back-to-back **per symbol**, and when the last one closed.
        #: Tracked per symbol deliberately: a portfolio-wide counter meant five losses
        #: spread across five unrelated markets paused entries on every symbol at once,
        #: including ones with no losing history. The cooldown is meant to step back from
        #: the market that is going against the strategy, not from all of them.
        self._consecutive_losses: dict[Symbol, int] = {}
        self._last_loss_at: dict[Symbol, datetime] = {}
        # (symbol, side) -> (when it was stopped out, what it scored going in). Keyed by
        # side because a long failing says nothing about a short in the same market.
        self._thesis_failures: dict[tuple[Symbol, OrderSide], tuple[datetime, Decimal | None]] = {}
        # What each symbol/side scored on the way in, so a stop-out can be compared against
        # the next candidate rather than needing the caller to remember it.
        self._entry_scores: dict[tuple[Symbol, OrderSide], Decimal | None] = {}
        #: Return correlations between traded symbols. Supplied from outside because the
        #: engine has no market data of its own and must never acquire any.
        self._correlations = CorrelationMatrix(values={})

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> RiskSettings:
        """The configured limits."""
        return self._settings

    @property
    def kill_switch(self) -> KillSwitch:
        """The kill switch."""
        return self._kill_switch

    @property
    def is_halted(self) -> bool:
        """Whether new entries are currently halted.

        The halt is scoped to a UTC day: it lifts automatically at midnight, unlike the
        kill switch, which requires an operator.
        """
        if self._halted_until is None:
            return False
        if self._clock.now() >= self._halted_until:
            self._halted_until = None
            return False
        return True

    async def start(self) -> None:
        """Restore latched state. Must be called before the first order."""
        await self._kill_switch.load()

    async def refresh_kill_switch(self) -> None:
        """Re-read the kill switch from storage.

        The switch is engaged from *other processes* — the CLI, and the dashboard through
        the API — but a long-running session loaded it once at startup and never looked
        again. So a halt was written, persisted, and correctly reported by every tool that
        read it fresh, while the bot it was meant to stop went on opening positions with a
        stale "clear" in memory.

        Called on the entry path, so the halt takes effect on the next decision rather than
        the next restart. A storage failure leaves the last known state in place rather
        than raising: losing the ability to read the switch must not become an inability
        to trade, and if it was already engaged it stays engaged.
        """
        try:
            await self._kill_switch.load()
        except Exception as exc:
            logger.warning("risk.kill_switch_refresh_failed", error=str(exc)[:200])

    def halt_for_the_day(self, reason: str) -> None:
        """Stop new entries until the next UTC day."""
        if self._halted_until is not None:
            return
        from datetime import timedelta

        self._halted_until = start_of_utc_day(self._clock.now()) + timedelta(days=1)
        logger.warning(
            "risk.trading_halted",
            reason=reason,
            until=self._halted_until.isoformat(),
        )

    def resume(self) -> None:
        """Lift a daily halt. Does not clear the kill switch."""
        self._halted_until = None
        logger.info("risk.trading_resumed")

    def record_order(self, at: datetime | None = None) -> None:
        """Register a submitted order for the rate limit."""
        moment = at or self._clock.now()
        self._order_timestamps.append(moment)
        self._prune_order_history(moment)

    def _prune_order_history(self, now: datetime) -> None:
        from datetime import timedelta

        cutoff = now - timedelta(minutes=1)
        self._order_timestamps = [stamp for stamp in self._order_timestamps if stamp > cutoff]

    def orders_in_last_minute(self, now: datetime | None = None) -> int:
        """How many orders were submitted in the trailing minute."""
        moment = now or self._clock.now()
        self._prune_order_history(moment)
        return len(self._order_timestamps)

    # ------------------------------------------------------------------ #
    # The gate
    # ------------------------------------------------------------------ #
    async def approve(
        self,
        request: OrderRequest,
        *,
        portfolio: PortfolioSnapshot,
        instrument: Instrument,
        reference_price: Decimal,
        resting_entry_notional: Mapping[str, Decimal] | None = None,
        candidate_score: Decimal | None = None,
    ) -> RiskDecision:
        """Run every rule against an already-sized order.

        Returns:
            A decision carrying either the approved request or the reasons for refusal.

        """
        # An operator halting from the CLI or the dashboard is a different process; without
        # this the running session would not notice until it restarted.
        await self.refresh_kill_switch()

        now = self._clock.now()
        thesis_failure = self._thesis_failures.get((request.symbol, request.side), (None, None))
        context = RiskContext(
            request=request,
            portfolio=portfolio,
            instrument=instrument,
            reference_price=reference_price,
            now=now,
            settings=self._settings,
            orders_last_minute=self.orders_in_last_minute(now),
            kill_switch_engaged=self._kill_switch.engaged,
            trading_halted=self.is_halted,
            week_start_equity=self._weekly_baseline(portfolio.equity, now),
            consecutive_losses=self._consecutive_losses.get(request.symbol, 0),
            last_loss_at=self._last_loss_at.get(request.symbol),
            last_thesis_failure_at=thesis_failure[0],
            last_thesis_failure_score=thesis_failure[1],
            correlated_open_symbols=self._correlated_open(request.symbol, portfolio),
            resting_entry_notional=resting_entry_notional or {},
            candidate_score=candidate_score,
        )

        verdicts = tuple(rule.evaluate(context) for rule in self._rules)
        denials = tuple(verdict for verdict in verdicts if not verdict.allowed)

        if not denials:
            # Remembered now so that if this trade later stops out, the cooldown can judge
            # the next candidate against what this one scored.
            self._entry_scores[(request.symbol, request.side)] = candidate_score
            return RiskDecision(approved=True, request=request, verdicts=verdicts)

        should_halt = any(verdict.halts_trading for verdict in denials)
        should_latch = any(verdict.engages_kill_switch for verdict in denials)
        reason = "; ".join(verdict.message for verdict in denials)

        # Act on the consequences before returning, so a caller that ignores the decision
        # object still cannot submit the next order.
        if should_latch:
            latch_reason = next(
                verdict.message for verdict in denials if verdict.engages_kill_switch
            )
            await self._kill_switch.engage(latch_reason)
        if should_halt and not self.is_halted:
            self.halt_for_the_day(reason)

        await self._persist(denials, context)
        await self._alert(denials, context, halted=should_halt)

        logger.warning(
            "risk.order_denied",
            symbol=str(request.symbol),
            side=request.side.value,
            quantity=str(request.quantity),
            rules=[verdict.rule for verdict in denials],
            reason=reason,
        )

        return RiskDecision(
            approved=False,
            verdicts=verdicts,
            reason=reason,
            halted_trading=should_halt,
            engaged_kill_switch=should_latch,
        )

    async def evaluate_signal(
        self,
        signal: Signal,
        *,
        portfolio: PortfolioSnapshot,
        instrument: Instrument,
        reference_price: Decimal,
        volatility: Decimal | None = None,
        resting_entry_notional: Mapping[str, Decimal] | None = None,
    ) -> RiskDecision:
        """Turn a strategy signal into an approved order, or refuse it.

        This is the only supported path from signal to order. It sizes the position,
        attaches a stop if the strategy did not supply one, then runs the full rule set.
        """
        if not signal.is_actionable:
            return RiskDecision(approved=False, reason="signal is not actionable")

        if signal.direction is SignalDirection.CLOSE:
            return await self._approve_exit(
                signal, portfolio=portfolio, instrument=instrument, reference_price=reference_price
            )

        side = OrderSide.BUY if signal.direction is SignalDirection.LONG else OrderSide.SELL
        stop_loss = signal.stop_loss_price or self._default_stop(side, reference_price)

        # Conviction, resolved before sizing so it can be applied inside the sizer's caps.
        # The orchestrator ranked this candidate against the whole field and attached its
        # score; this turns that ranking into a share of the allowed position. Every trade
        # before this was identically sized whatever the score.
        #
        # Gated on expected net edge: a candidate can top a weak field and still not be
        # worth more capital. Increases require the trade to be expected to pay for its own
        # execution; reductions do not, because trusting a marginal setup less is safe.
        raw_score = signal.metadata.get("orchestrator_score")
        score = Decimal(raw_score) if raw_score else None
        expected_edge = _expected_net_edge(
            side=side,
            entry=reference_price,
            target=signal.take_profit_price,
            cost_rate=self._settings.round_trip_cost_rate,
        )
        tier, allocation = allocation_fraction(score, expected_net_edge=expected_edge)

        try:
            sizing = self._sizer.size(
                SizingRequest(
                    equity=portfolio.equity,
                    price=reference_price,
                    instrument=instrument,
                    stop_loss_price=stop_loss,
                    # What this symbol already holds, so an additional leg is sized into the
                    # room left under the SAME cap rather than claiming the whole allowance
                    # a second time. Zero on a flat symbol, so the ordinary path is
                    # unchanged.
                    committed_notional=self._committed_on(
                        signal.symbol, portfolio, reference_price, resting_entry_notional
                    ),
                    # Conviction enters here — before the sizer's caps — because that is
                    # where the sizer applies it. Scaling after the caps produced sizes the
                    # next check rejected outright, so the two components contradicted each
                    # other and nothing was ever placed.
                    #
                    # Two independent attenuators, multiplied: the strategy's own
                    # confidence in this setup, and the orchestrator's ranking of it against
                    # the field. Both are fractions, so the product is one too and the cap
                    # still binds. Dropping either would discard information the other does
                    # not carry.
                    conviction=signal.conviction * allocation,
                    available_cash=portfolio.cash,
                    volatility=volatility,
                )
            )
        except ValidationError as exc:
            return RiskDecision(approved=False, reason=f"sizing failed: {exc.message}")

        if not sizing.is_tradable:
            # Name the rule *and* the numbers. "resolved to zero" alone sent an operator
            # back to the venue to work out which limit had bound, and a wrong-but-plausible
            # rejection then looks identical to a correct one.
            explanation = f"position size resolved to zero ({sizing.capped_by})"
            if sizing.detail:
                explanation = f"{explanation}: {sizing.detail}"
            return RiskDecision(approved=False, sizing=sizing, reason=explanation)

        entry_type, entry_price, post_only = as_maker_entry(
            signal.order_type,
            limit_price=signal.limit_price,
            reference_price=reference_price,
            enabled=self._settings.maker_first_entries,
            side=side,
            price_tick=instrument.price_tick,
        )
        # Applied at the same single point as maker conversion, so every strategy gets a
        # target that clears its own execution cost without any of them knowing about fees.
        # Widening only: a strategy that already chose a further level keeps it, and the
        # stop is never touched.
        target = cost_aware_target(
            side=side,
            entry=reference_price,
            target=signal.take_profit_price,
            atr=volatility,
            cost_rate=self._settings.round_trip_cost_rate,
        )
        if target is not None and target != signal.take_profit_price:
            logger.info(
                "risk.target_widened",
                symbol=str(signal.symbol),
                strategy=signal.strategy_id,
                requested=str(signal.take_profit_price),
                applied=str(target),
                reason="the requested target did not clear its round-trip cost",
            )
        logger.info(
            "risk.conviction_sized",
            symbol=str(signal.symbol),
            strategy=signal.strategy_id,
            score=str(score) if score is not None else None,
            percentile=percentile_of(score) if score is not None else None,
            tier=tier.value,
            allocation_fraction=str(allocation),
            final_quantity=str(sizing.quantity),
            capped_by=sizing.capped_by,
            notional=str(sizing.quantity * reference_price),
            stop_distance=str(abs(reference_price - stop_loss)),
            expected_net_edge=str(expected_edge) if expected_edge is not None else None,
        )
        request = OrderRequest(
            symbol=signal.symbol,
            side=side,
            order_type=entry_type,
            quantity=sizing.quantity,
            price=entry_price,
            post_only=post_only,
            stop_loss_price=stop_loss,
            take_profit_price=target,
            time_in_force=signal.time_in_force,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            metadata={"sizing_method": sizing.method, "reason": signal.reason[:200]},
        )

        decision = await self.approve(
            request,
            portfolio=portfolio,
            instrument=instrument,
            reference_price=reference_price,
            resting_entry_notional=resting_entry_notional,
            candidate_score=score,
        )
        if decision.approved:
            return RiskDecision(
                approved=True,
                request=decision.request,
                sizing=sizing,
                verdicts=decision.verdicts,
            )
        return RiskDecision(
            approved=False,
            sizing=sizing,
            verdicts=decision.verdicts,
            reason=decision.reason,
            halted_trading=decision.halted_trading,
            engaged_kill_switch=decision.engaged_kill_switch,
        )

    async def _approve_exit(
        self,
        signal: Signal,
        *,
        portfolio: PortfolioSnapshot,
        instrument: Instrument,
        reference_price: Decimal,
    ) -> RiskDecision:
        """Build a reduce-only order that flattens the position.

        Exits bypass sizing entirely — the size is whatever is open — and are exempt from
        exposure limits, because refusing an exit would trap the account in the very
        position the limits exist to avoid.
        """
        position = portfolio.position_for(signal.symbol)
        if position is None or position.is_flat:
            return RiskDecision(approved=False, reason="no open position to close")

        closing_side = position.closing_side()
        assert closing_side is not None
        request = OrderRequest(
            symbol=signal.symbol,
            side=closing_side,
            order_type=OrderType.MARKET,
            quantity=instrument.normalize_quantity(position.absolute_quantity),
            reduce_only=True,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            metadata={"reason": signal.reason[:200]},
        )
        if request.quantity <= ZERO:
            return RiskDecision(approved=False, reason="position is below the venue lot size")

        return await self.approve(
            request,
            portfolio=portfolio,
            instrument=instrument,
            reference_price=reference_price,
        )

    def set_correlations(self, matrix: CorrelationMatrix) -> None:
        """Supply the current return-correlation estimate.

        Pushed in rather than computed here: the risk engine owns decisions, not market
        data, and giving it a data feed would make it impossible to test a rule without
        also standing up a data source.
        """
        self._correlations = matrix

    def record_trade_result(
        self,
        net_pnl: Decimal,
        *,
        closed_at: datetime,
        symbol: Symbol,
        side: OrderSide | None = None,
        gross_pnl: Decimal | None = None,
    ) -> None:
        """Record a closed trade so the cooldowns can track it.

        A break-even trade counts as a loss: after fees it *is* one, and treating it as a
        reset would let a strategy grind through the streak limit indefinitely.

        The streak is kept per symbol, so a run of losses in one market pauses entries in
        that market only and leaves unrelated symbols tradable.
        """
        if net_pnl > ZERO:
            self._consecutive_losses.pop(symbol, None)
            self._last_loss_at.pop(symbol, None)
            if side is not None:
                # A winner clears the thesis mark: the case has been vindicated, and
                # holding a cooldown over it would penalise a strategy for being right.
                self._thesis_failures.pop((symbol, side), None)
            return
        self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        self._last_loss_at[symbol] = closed_at

        # The thesis itself is also marked failed, so the same symbol and side is not
        # re-entered on unchanged evidence. Recorded here rather than at the call sites so
        # paper and backtest cannot diverge: both reach this one method.
        #
        # Armed on GROSS loss, not net. A mechanical failure is not a failed thesis, and
        # judging by net cannot tell them apart: on 2026-08-17 a BTC entry filled and was
        # emergency-flattened 0.4 seconds later because its stop would not attach at the
        # venue. Gross was exactly 0.00 and the only loss was the 6.51 in fees — the market
        # never got to express an opinion — yet it armed the cooldown and locked BTC out
        # for an hour. An aborted entry, a rejected order and a fee-only scratch all reach
        # this method looking like small losses; only gross separates them from a thesis
        # the market actually refuted.
        if gross_pnl is not None and gross_pnl >= ZERO:
            return
        if side is not None:
            self._thesis_failures[(symbol, side)] = (
                closed_at,
                self._entry_scores.get((symbol, side)),
            )

    def export_cooldown_state(self) -> dict[str, Any]:
        """The thesis-cooldown state, in a form that survives a restart.

        Held in memory it does not. On 2026-08-17 the engine restarted between an entry
        being approved and its loss being recorded, so the entry score was gone and the
        cooldown refused a candidate with "no score was recorded to compare against" — the
        early-clear path, which exists precisely so a materially stronger signal can
        proceed, could not run at all. A cooldown that cannot be argued with is a harsher
        rule than the one that was designed.
        """
        return {
            "failures": [
                {
                    "symbol": str(symbol),
                    "side": side.value,
                    "at": moment.isoformat(),
                    "score": str(score) if score is not None else None,
                }
                for (symbol, side), (moment, score) in self._thesis_failures.items()
            ],
            "entry_scores": [
                {"symbol": str(symbol), "side": side.value, "score": str(score)}
                for (symbol, side), score in self._entry_scores.items()
                if score is not None
            ],
        }

    def restore_cooldown_state(self, payload: Any) -> int:
        """Rebuild cooldown state from :meth:`export_cooldown_state`.

        Never raises: a malformed or absent payload leaves the engine exactly as it started,
        which is the pre-existing behaviour rather than a new failure mode.

        Returns:
            How many failure records were restored.

        """
        if not isinstance(payload, dict):
            return 0
        restored = 0
        for row in payload.get("failures") or []:
            try:
                key = (Symbol.parse(row["symbol"]), OrderSide(row["side"]))
                score = row.get("score")
                self._thesis_failures[key] = (
                    datetime.fromisoformat(row["at"]),
                    Decimal(score) if score is not None else None,
                )
                restored += 1
            except (KeyError, TypeError, ValueError, ArithmeticError, ValidationError):
                continue
        for row in payload.get("entry_scores") or []:
            try:
                self._entry_scores[(Symbol.parse(row["symbol"]), OrderSide(row["side"]))] = Decimal(
                    row["score"]
                )
            except (KeyError, TypeError, ValueError, ArithmeticError, ValidationError):
                continue
        return restored

    @staticmethod
    def _committed_on(
        symbol: Symbol,
        portfolio: PortfolioSnapshot,
        reference_price: Decimal,
        resting: Mapping[str, Decimal] | None,
    ) -> Decimal:
        """Notional already committed on one symbol: open position plus resting entries.

        The same two components :class:`~quantflow.risk.rules.MaxPositionSizeRule` checks
        against, computed here so the sizer and the rule cannot disagree about how much
        room is left. A sizer that ignores this produces a leg the rule then refuses, which
        is how conviction and the position cap contradicted each other once already.
        """
        committed = ZERO
        position = portfolio.position_for(symbol)
        if position is not None:
            committed += position.notional(reference_price)
        if resting:
            committed += resting.get(str(symbol), ZERO)
        return committed

    def _weekly_baseline(self, equity: Decimal, now: datetime) -> Decimal | None:
        """Equity at the start of the current seven-day window.

        Rolls forward rather than resetting on a calendar boundary: a drawdown that
        straddles Sunday midnight is exactly as damaging as one that does not, and a
        calendar reset would forgive it for no reason.
        """
        if self._week_start_equity is None or self._week_started_at is None:
            self._week_start_equity = equity
            self._week_started_at = now
            return self._week_start_equity

        if now - self._week_started_at >= timedelta(days=7):
            self._week_start_equity = equity
            self._week_started_at = now
        return self._week_start_equity

    def _correlated_open(self, candidate: Symbol, portfolio: PortfolioSnapshot) -> tuple[str, ...]:
        """Open positions that move with the candidate beyond the threshold."""
        open_symbols = [
            position.symbol for position in portfolio.open_positions if position.symbol != candidate
        ]
        if not open_symbols:
            return ()
        hits = self._correlations.correlated_with(
            candidate, open_symbols, threshold=self._settings.correlation_threshold
        )
        return tuple(str(symbol) for symbol in hits)

    def _default_stop(self, side: OrderSide, price: Decimal) -> Decimal:
        """Apply the configured default stop when a strategy supplies none.

        The engine never lets an entry through unprotected: if the strategy has no opinion,
        the configured default applies rather than the position going out naked.
        """
        distance = price * self._settings.default_stop_loss_pct
        stop = price - distance if side is OrderSide.BUY else price + distance
        return max(stop, ZERO + price * Decimal("0.0001"))

    async def _alert(
        self,
        denials: tuple[RiskVerdict, ...],
        context: RiskContext,
        *,
        halted: bool,
    ) -> None:
        """Notify the operator about a refusal.

        Only the most severe denial is sent: an order blocked by five rules at once is one
        event to a human, not five. Failures are swallowed — an unreachable notification
        service must never change a risk outcome.
        """
        if self._notifier is None:
            return
        worst = max(denials, key=lambda verdict: verdict.severity.rank)
        try:
            await self._notifier.notify_risk(
                rule=worst.rule,
                message=worst.message,
                symbol=str(context.request.symbol),
                observed=worst.observed,
                limit=worst.limit,
                halted=halted,
            )
        except Exception as exc:
            logger.warning("risk.alert_failed", error=str(exc))

    async def _persist(self, denials: tuple[RiskVerdict, ...], context: RiskContext) -> None:
        """Write the refusals to the audit trail.

        Failures here are logged and swallowed: a database problem must not become a reason
        that an order slips through unchecked.
        """
        if self._database is None:
            return
        try:
            async with self._database.unit_of_work() as uow:
                for verdict in denials:
                    await uow.risk_events.record(
                        rule=verdict.rule,
                        message=verdict.message,
                        severity=verdict.severity.value,
                        symbol=context.request.symbol,
                        observed_value=verdict.observed,
                        limit_value=verdict.limit,
                        blocked_order=True,
                        halted_trading=verdict.halts_trading,
                        session_id=self._session_id,
                        context={key: str(value) for key, value in verdict.context.items()},
                    )
        except Exception as exc:
            logger.exception("risk.persist_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        """Current configuration and state, for the API and dashboard."""
        return {
            "sizer": self._sizer.name,
            "rules": [rule.name for rule in self._rules],
            "halted": self.is_halted,
            "kill_switch": {
                "engaged": self._kill_switch.engaged,
                "reason": self._kill_switch.state.reason,
            },
            "limits": {
                "max_position_pct": str(self._settings.max_position_pct),
                "max_total_exposure_pct": str(self._settings.max_total_exposure_pct),
                "max_concurrent_positions": self._settings.max_concurrent_positions,
                "max_daily_loss_pct": str(self._settings.max_daily_loss_pct),
                "max_drawdown_pct": str(self._settings.max_drawdown_pct),
                "max_leverage": str(self._settings.max_leverage),
                "require_stop_loss": self._settings.require_stop_loss,
                "max_order_notional": str(self._settings.max_order_notional),
                "max_orders_per_minute": self._settings.max_orders_per_minute,
            },
        }


def assert_protected(request: OrderRequest, settings: RiskSettings) -> None:
    """Last-line assertion that an entry carries a stop.

    Called by the execution engine immediately before submission. It is deliberately
    redundant with :class:`~quantflow.risk.rules.StopLossRequiredRule`: this is the check
    that catches a future refactor accidentally introducing a path around the engine.

    Raises:
        RiskViolationError: if an entry has no stop loss.

    """
    if not settings.require_stop_loss:
        return
    if request.reduce_only:
        return
    if request.stop_loss_price is None:
        raise RiskViolationError(
            f"refusing to submit an unprotected entry for {request.symbol}",
            rule="stop_loss_required",
            symbol=str(request.symbol),
        )


def daily_loss_headroom(portfolio: PortfolioSnapshot, settings: RiskSettings) -> Decimal:
    """Quote-currency loss still available today before the daily limit trips."""
    baseline = portfolio.day_start_equity
    if baseline <= ZERO:
        return settings.max_daily_loss_pct * portfolio.equity
    allowed = baseline * settings.max_daily_loss_pct
    used = max(ZERO, baseline - portfolio.equity)
    return max(ZERO, allowed - used)


def drawdown_headroom(portfolio: PortfolioSnapshot, settings: RiskSettings) -> Decimal:
    """Fractional drawdown still available before the kill switch trips."""
    return max(ZERO, settings.max_drawdown_pct - portfolio.drawdown_pct)


def exposure_headroom(portfolio: PortfolioSnapshot, settings: RiskSettings) -> Decimal:
    """Quote-currency notional that can still be added within the exposure limit."""
    equity = portfolio.equity
    if equity <= ZERO:
        return ZERO
    allowed = equity * settings.max_total_exposure_pct
    return max(ZERO, allowed - portfolio.gross_exposure)


def summarise_headroom(portfolio: PortfolioSnapshot, settings: RiskSettings) -> dict[str, str]:
    """All headroom measures, for the dashboard's risk panel."""
    return {
        "daily_loss": str(daily_loss_headroom(portfolio, settings)),
        "drawdown_pct": str(drawdown_headroom(portfolio, settings)),
        "exposure": str(exposure_headroom(portfolio, settings)),
        "positions": str(max(0, settings.max_concurrent_positions - portfolio.position_count)),
        "leverage_used": str(portfolio.leverage.quantize(Decimal("0.0001"))),
        "leverage_limit": str(settings.max_leverage),
        "utilisation": str(
            (portfolio.leverage / settings.max_total_exposure_pct).quantize(Decimal("0.0001"))
            if settings.max_total_exposure_pct > ZERO
            else ONE
        ),
    }
