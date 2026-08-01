"""The risk engine.

**Every** order in the system passes through :meth:`RiskEngine.approve`. There is no other
path from a strategy signal to an exchange — not in backtesting, not in paper trading, not
in live. That single invariant is what the rest of the risk design rests on.

The engine turns an unsized :class:`~quantflow.domain.signals.Signal` into either a sized,
protected, venue-legal :class:`~quantflow.domain.orders.OrderRequest`, or a documented
refusal. It never returns a partially-checked order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.clock import Clock, SystemClock, start_of_utc_day
from quantflow.core.config import RiskSettings
from quantflow.core.errors import RiskViolationError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import OrderSide, OrderType, SignalDirection
from quantflow.domain.instruments import Instrument
from quantflow.domain.orders import OrderRequest
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.persistence.database import Database
from quantflow.risk.killswitch import KillSwitch
from quantflow.risk.rules import RiskContext, RiskRule, RiskVerdict, build_default_rules
from quantflow.risk.sizing import PositionSizer, SizingRequest, SizingResult, build_sizer

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


class RiskEngine:
    """Sizes, protects and validates every order before it can reach a venue."""

    __slots__ = (
        "_clock",
        "_database",
        "_halted_until",
        "_kill_switch",
        "_notifier",
        "_order_timestamps",
        "_rules",
        "_session_id",
        "_settings",
        "_sizer",
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
    ) -> RiskDecision:
        """Run every rule against an already-sized order.

        Returns:
            A decision carrying either the approved request or the reasons for refusal.

        """
        now = self._clock.now()
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
        )

        verdicts = tuple(rule.evaluate(context) for rule in self._rules)
        denials = tuple(verdict for verdict in verdicts if not verdict.allowed)

        if not denials:
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

        try:
            sizing = self._sizer.size(
                SizingRequest(
                    equity=portfolio.equity,
                    price=reference_price,
                    instrument=instrument,
                    stop_loss_price=stop_loss,
                    conviction=signal.conviction,
                    available_cash=portfolio.cash,
                    volatility=volatility,
                )
            )
        except ValidationError as exc:
            return RiskDecision(approved=False, reason=f"sizing failed: {exc.message}")

        if not sizing.is_tradable:
            return RiskDecision(
                approved=False,
                sizing=sizing,
                reason=f"position size resolved to zero ({sizing.capped_by})",
            )

        request = OrderRequest(
            symbol=signal.symbol,
            side=side,
            order_type=signal.order_type,
            quantity=sizing.quantity,
            price=signal.limit_price,
            stop_loss_price=stop_loss,
            take_profit_price=signal.take_profit_price,
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
