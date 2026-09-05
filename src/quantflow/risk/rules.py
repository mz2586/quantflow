"""Risk rules.

Each rule is a small, independently testable predicate over the proposed order and the
current portfolio. They are composed by :class:`~quantflow.risk.engine.RiskEngine`, which
evaluates **all** of them and refuses the order if any denies it.

Rules never raise on a denial — they return a verdict. The engine decides what a denial
means (block the order, halt the day, latch the kill switch), which keeps policy in one
place and makes every rule trivial to unit test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.config import RiskSettings, Severity
from quantflow.core.precision import ONE, ZERO, safe_divide
from quantflow.domain.enums import OrderSide
from quantflow.domain.instruments import Instrument
from quantflow.domain.orders import OrderRequest
from quantflow.domain.portfolio import PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """A single rule's decision."""

    rule: str
    allowed: bool
    message: str = ""
    severity: Severity = Severity.WARNING
    observed: Decimal | None = None
    limit: Decimal | None = None
    halts_trading: bool = False
    """Whether a denial should stop *all* new entries, not just this order."""
    engages_kill_switch: bool = False
    """Whether a denial should latch the kill switch, requiring operator intervention."""
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, rule: str, message: str = "") -> RiskVerdict:
        """Build an approving verdict."""
        return cls(rule=rule, allowed=True, message=message, severity=Severity.DEBUG)

    @classmethod
    def deny(
        cls,
        rule: str,
        message: str,
        *,
        observed: Decimal | None = None,
        limit: Decimal | None = None,
        severity: Severity = Severity.WARNING,
        halts_trading: bool = False,
        engages_kill_switch: bool = False,
        **context: Any,
    ) -> RiskVerdict:
        """Build a denying verdict."""
        return cls(
            rule=rule,
            allowed=False,
            message=message,
            severity=severity,
            observed=observed,
            limit=limit,
            halts_trading=halts_trading,
            engages_kill_switch=engages_kill_switch,
            context=context,
        )


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Everything the rules need to judge one proposed order."""

    request: OrderRequest
    portfolio: PortfolioSnapshot
    instrument: Instrument
    reference_price: Decimal
    now: datetime
    settings: RiskSettings
    orders_last_minute: int = 0
    realized_pnl_today: Decimal = ZERO
    kill_switch_engaged: bool = False
    trading_halted: bool = False
    #: Equity at the start of the rolling seven-day window, or None when unknown.
    week_start_equity: Decimal | None = None
    #: Losing trades closed back-to-back with no winner between them.
    consecutive_losses: int = 0
    #: When the most recent losing trade closed, for the cooldown clock.
    last_loss_at: datetime | None = None
    #: Open positions that move with the candidate beyond the correlation threshold.
    #: Supplied by the engine, which owns the correlation estimate.
    correlated_open_symbols: tuple[str, ...] = ()
    #: Notional of *unfilled* entry orders already resting at the venue, per symbol.
    #:
    #: Counted as exposure because it becomes exposure without any further decision. A
    #: maker entry can rest for hours and fill on top of a position opened after it, and
    #: no rule ever re-examines it: on 2026-08-17 a WLD order placed at 01:00 while the
    #: symbol was flat passed the 20% check at 17.7%, rested 5.4 hours, and filled at
    #: 06:25 on top of a position opened at 01:30 — leaving one symbol at 35.4% of equity
    #: against a 20% cap, with every individual check having passed honestly.
    #:
    #: Reduce-only orders are excluded by the caller: a stop or target *removes* exposure.
    resting_entry_notional: Mapping[str, Decimal] = field(default_factory=dict)
    #: When this symbol/side was last stopped out, and what it scored going in.
    last_thesis_failure_at: datetime | None = None
    last_thesis_failure_score: Decimal | None = None
    #: The candidate's own score, so a re-entry can be judged against what just failed.
    candidate_score: Decimal | None = None

    @property
    def resting_for_symbol(self) -> Decimal:
        """Resting entry notional on the candidate's own symbol."""
        return self.resting_entry_notional.get(str(self.request.symbol), ZERO)

    @property
    def resting_total(self) -> Decimal:
        """Resting entry notional across every symbol."""
        return sum(self.resting_entry_notional.values(), ZERO)

    @property
    def notional(self) -> Decimal:
        """Quote-currency value of the proposed order."""
        return self.instrument.notional(self.request.quantity, self.reference_price)

    @property
    def is_entry(self) -> bool:
        """Whether the order opens or increases exposure."""
        return self.request.is_entry

    @property
    def is_reducing(self) -> bool:
        """Whether the order reduces or closes existing exposure.

        Reducing orders are exempt from most limits: refusing an exit because exposure is
        already too high would trap the account in exactly the position the limit exists
        to prevent.
        """
        if self.request.reduce_only:
            return True
        position = self.portfolio.position_for(self.request.symbol)
        if position is None or position.is_flat:
            return False
        return self.request.side is position.side.exit_side


class RiskRule(ABC):
    """A single risk constraint."""

    name: str = "rule"
    #: Whether the rule also applies to orders that reduce exposure.
    applies_to_exits: bool = False

    @abstractmethod
    def check(self, context: RiskContext) -> RiskVerdict:
        """Judge the proposed order."""

    def evaluate(self, context: RiskContext) -> RiskVerdict:
        """Apply the rule, short-circuiting for exits where appropriate."""
        if context.is_reducing and not self.applies_to_exits:
            return RiskVerdict.allow(self.name, "exempt: order reduces exposure")
        return self.check(context)


# --------------------------------------------------------------------------- #
# Hard stops
# --------------------------------------------------------------------------- #
class KillSwitchRule(RiskRule):
    """Blocks all new entries while the kill switch is latched."""

    name = "kill_switch"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny every entry when the switch is engaged."""
        if context.kill_switch_engaged:
            return RiskVerdict.deny(
                self.name,
                "kill switch is engaged; clear it explicitly before trading resumes",
                severity=Severity.CRITICAL,
                halts_trading=True,
            )
        return RiskVerdict.allow(self.name)


class TradingHaltedRule(RiskRule):
    """Blocks new entries while trading is halted for the day."""

    name = "trading_halted"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny every entry while halted."""
        if context.trading_halted:
            return RiskVerdict.deny(
                self.name,
                "trading is halted for the day",
                severity=Severity.WARNING,
                halts_trading=True,
            )
        return RiskVerdict.allow(self.name)


class StopLossRequiredRule(RiskRule):
    """Every entry must carry a stop loss.

    The single most important rule in the system. An entry without a defined exit has
    unbounded downside, and no amount of position sizing compensates for that — sizing
    assumes a known loss at a known price.
    """

    name = "stop_loss_required"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an entry that has no stop loss attached."""
        if not context.settings.require_stop_loss:
            return RiskVerdict.allow(self.name, "stop loss not required by configuration")
        if context.request.stop_loss_price is None:
            return RiskVerdict.deny(
                self.name,
                "entry rejected: no stop loss attached",
                severity=Severity.CRITICAL,
                symbol=str(context.request.symbol),
            )

        stop = context.request.stop_loss_price
        entry = context.reference_price
        if context.request.side is OrderSide.BUY and stop >= entry:
            return RiskVerdict.deny(
                self.name,
                f"long stop {stop} is not below entry {entry}",
                observed=stop,
                limit=entry,
                severity=Severity.CRITICAL,
            )
        if context.request.side is OrderSide.SELL and stop <= entry:
            return RiskVerdict.deny(
                self.name,
                f"short stop {stop} is not above entry {entry}",
                observed=stop,
                limit=entry,
                severity=Severity.CRITICAL,
            )

        distance_pct = safe_divide(abs(entry - stop), entry)
        if distance_pct > context.settings.max_stop_loss_pct:
            return RiskVerdict.deny(
                self.name,
                f"stop distance {distance_pct:.2%} exceeds the maximum "
                f"{context.settings.max_stop_loss_pct:.2%}",
                observed=distance_pct,
                limit=context.settings.max_stop_loss_pct,
            )
        return RiskVerdict.allow(self.name)


# --------------------------------------------------------------------------- #
# Exposure limits
# --------------------------------------------------------------------------- #
class MaxPositionSizeRule(RiskRule):
    """Caps any single position's notional as a fraction of equity."""

    name = "max_position_pct"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an order that would push one symbol past its share of equity."""
        equity = context.portfolio.equity
        if equity <= ZERO:
            return RiskVerdict.deny(self.name, "equity is not positive", severity=Severity.CRITICAL)

        existing = ZERO
        position = context.portfolio.position_for(context.request.symbol)
        if position is not None:
            existing = position.notional(context.reference_price)

        # Resting entries count. Without them this rule measured only what had already
        # filled, and a symbol could pass twice on its way to double the cap.
        resting = context.resting_for_symbol
        projected = safe_divide(existing + resting + context.notional, equity)
        limit = context.settings.max_position_pct
        if projected > limit:
            return RiskVerdict.deny(
                self.name,
                f"{context.request.symbol} would reach {projected:.2%} of equity, "
                f"above the {limit:.2%} limit "
                f"(open {existing}, resting {resting}, this order {context.notional})",
                observed=projected,
                limit=limit,
                symbol=str(context.request.symbol),
            )
        return RiskVerdict.allow(self.name)


class MaxTotalExposureRule(RiskRule):
    """Caps aggregate gross exposure across all positions."""

    name = "max_total_exposure_pct"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an order that would push gross exposure past the limit."""
        equity = context.portfolio.equity
        if equity <= ZERO:
            return RiskVerdict.deny(self.name, "equity is not positive", severity=Severity.CRITICAL)
        open_notional = context.portfolio.gross_exposure
        resting = context.resting_total
        projected = safe_divide(open_notional + resting + context.notional, equity)
        limit = context.settings.max_total_exposure_pct
        if projected > limit:
            return RiskVerdict.deny(
                self.name,
                f"gross exposure would reach {projected:.2%} of equity, "
                f"above the {limit:.2%} limit "
                f"(open {open_notional}, resting {resting}, this order {context.notional})",
                observed=projected,
                limit=limit,
            )
        return RiskVerdict.allow(self.name)


class MaxConcurrentPositionsRule(RiskRule):
    """Caps how many positions may be open at once."""

    name = "max_concurrent_positions"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny a *new* symbol once the position count is at its limit."""
        if context.portfolio.has_position(context.request.symbol):
            return RiskVerdict.allow(self.name, "adding to an existing position")
        current = context.portfolio.position_count
        limit = context.settings.max_concurrent_positions
        if current >= limit:
            return RiskVerdict.deny(
                self.name,
                f"{current} positions are already open, at the limit of {limit}",
                observed=Decimal(current),
                limit=Decimal(limit),
            )
        return RiskVerdict.allow(self.name)


class MaxLeverageRule(RiskRule):
    """Caps portfolio leverage."""

    name = "max_leverage"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an order that would exceed the configured leverage."""
        equity = context.portfolio.equity
        if equity <= ZERO:
            return RiskVerdict.deny(self.name, "equity is not positive", severity=Severity.CRITICAL)
        projected = safe_divide(context.portfolio.gross_exposure + context.notional, equity)
        limit = context.settings.max_leverage
        if projected > limit:
            return RiskVerdict.deny(
                self.name,
                f"leverage would reach {projected:.2f}x, above the {limit}x limit",
                observed=projected,
                limit=limit,
            )
        return RiskVerdict.allow(self.name)


class OrderNotionalRule(RiskRule):
    """Bounds a single order's notional, both above and below."""

    name = "order_notional"
    applies_to_exits = True

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an order that is too large, or too small to be worth the fees."""
        notional = context.notional
        if notional > context.settings.max_order_notional:
            return RiskVerdict.deny(
                self.name,
                f"order notional {notional} exceeds the maximum "
                f"{context.settings.max_order_notional}",
                observed=notional,
                limit=context.settings.max_order_notional,
            )
        # Exits are allowed to be small: closing a dust position is always permitted.
        if not context.is_reducing and notional < context.settings.min_order_notional:
            return RiskVerdict.deny(
                self.name,
                f"order notional {notional} is below the minimum "
                f"{context.settings.min_order_notional}",
                observed=notional,
                limit=context.settings.min_order_notional,
                severity=Severity.INFO,
            )
        return RiskVerdict.allow(self.name)


# --------------------------------------------------------------------------- #
# Loss limits
# --------------------------------------------------------------------------- #
class MaxDailyLossRule(RiskRule):
    """Halts new entries once the day's loss limit is reached.

    Halts rather than latches: a bad day is normal and should not require an operator to
    come and clear a switch before the next session.
    """

    name = "max_daily_loss"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny entries once today's loss exceeds the limit."""
        baseline = context.portfolio.day_start_equity
        if baseline <= ZERO:
            return RiskVerdict.allow(self.name, "no baseline equity for the day yet")

        daily_pnl = context.portfolio.equity - baseline
        if daily_pnl >= ZERO:
            return RiskVerdict.allow(self.name)

        loss_pct = safe_divide(-daily_pnl, baseline)
        limit = context.settings.max_daily_loss_pct
        if loss_pct >= limit:
            return RiskVerdict.deny(
                self.name,
                f"daily loss {loss_pct:.2%} has reached the {limit:.2%} limit; "
                "new entries are halted for the rest of the UTC day",
                observed=loss_pct,
                limit=limit,
                severity=Severity.CRITICAL,
                halts_trading=True,
            )
        return RiskVerdict.allow(self.name)


class MaxDrawdownRule(RiskRule):
    """Latches the kill switch once drawdown from the equity peak exceeds the limit.

    Unlike the daily-loss rule this *latches*: a drawdown that deep means the assumptions
    behind the strategy are in question, and resuming should be a deliberate human decision
    rather than something that happens automatically at midnight.
    """

    name = "max_drawdown"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny entries and latch the switch once drawdown exceeds the limit."""
        drawdown = context.portfolio.drawdown_pct
        limit = context.settings.max_drawdown_pct
        if drawdown >= limit:
            return RiskVerdict.deny(
                self.name,
                f"drawdown {drawdown:.2%} has reached the {limit:.2%} limit; "
                "the kill switch is being engaged",
                observed=drawdown,
                limit=limit,
                severity=Severity.CRITICAL,
                halts_trading=True,
                engages_kill_switch=True,
            )
        return RiskVerdict.allow(self.name)


class OrderRateRule(RiskRule):
    """Caps orders per minute.

    A runaway loop that submits an order every tick can breach every other limit before a
    human notices, and will get the account rate-limited or banned. This is the circuit
    breaker for a bug in our own code.
    """

    name = "order_rate"
    applies_to_exits = True

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny once the order rate exceeds the limit."""
        limit = context.settings.max_orders_per_minute
        if context.orders_last_minute >= limit:
            return RiskVerdict.deny(
                self.name,
                f"{context.orders_last_minute} orders in the last minute, at the limit of {limit}",
                observed=Decimal(context.orders_last_minute),
                limit=Decimal(limit),
                severity=Severity.CRITICAL,
            )
        return RiskVerdict.allow(self.name)


class InstrumentRule(RiskRule):
    """Validates the order against the venue's own trading rules."""

    name = "instrument_rules"
    applies_to_exits = True

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an order the venue would reject on lot, tick or notional grounds."""
        from quantflow.core.errors import ValidationError

        # A market order has no price of its own; the reference is a mark or last price
        # that need not sit on the tick grid. Only an order carrying an explicit limit or
        # trigger price is checked against it.
        explicit_price = context.request.price or context.request.trigger_price
        try:
            context.instrument.validate_order(
                context.request.quantity,
                explicit_price or context.reference_price,
                check_price_tick=explicit_price is not None,
            )
        except ValidationError as exc:
            return RiskVerdict.deny(
                self.name,
                f"venue rules rejected the order: {exc.message}",
                symbol=str(context.request.symbol),
                venue_rule=str(exc.details.get("rule", "unknown")),
            )
        return RiskVerdict.allow(self.name)


class SufficientCashRule(RiskRule):
    """Ensures the account can actually pay for a buy."""

    name = "sufficient_cash"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny a buy the account cannot fund."""
        if context.request.side is not OrderSide.BUY:
            return RiskVerdict.allow(self.name, "sells do not consume cash")
        if context.settings.max_leverage > ONE:
            return RiskVerdict.allow(self.name, "margin account")

        # Leave headroom for fees; an order that consumes the last cent gets rejected by
        # the venue for being unable to pay its own commission.
        required = context.notional * Decimal("1.005")
        if required > context.portfolio.cash:
            return RiskVerdict.deny(
                self.name,
                f"order needs {required:.2f} including fees but only "
                f"{context.portfolio.cash:.2f} cash is available",
                observed=required,
                limit=context.portfolio.cash,
            )
        return RiskVerdict.allow(self.name)


#: Evaluation order. Cheap, absolute blocks come first so an obviously-doomed order does
#: not incur the cost of the exposure calculations.
class MaxWeeklyLossRule(RiskRule):
    """Halts new entries once the rolling seven-day loss limit is reached.

    A daily limit alone is not enough: five consecutive days at 2.9% each pass the 3%
    daily rule every single time and still leave a 14% hole in the account. The weekly
    ceiling is what stops a slow bleed that never trips a daily check.
    """

    name = "max_weekly_loss"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny entries once the week's loss exceeds the limit."""
        baseline = context.week_start_equity
        if baseline is None or baseline <= ZERO:
            return RiskVerdict.allow(self.name, "no baseline equity for the week yet")

        weekly_pnl = context.portfolio.equity - baseline
        if weekly_pnl >= ZERO:
            return RiskVerdict.allow(self.name)

        loss_pct = safe_divide(-weekly_pnl, baseline)
        limit = context.settings.max_weekly_loss_pct
        if loss_pct >= limit:
            return RiskVerdict.deny(
                self.name,
                f"seven-day loss {loss_pct:.2%} has reached the {limit:.2%} limit; "
                "new entries are halted",
                observed=loss_pct,
                limit=limit,
                severity=Severity.CRITICAL,
                halts_trading=True,
            )
        return RiskVerdict.allow(self.name)


class CorrelationLimitRule(RiskRule):
    """Caps how many mutually correlated positions may be held at once.

    Position-count limits assume positions are independent bets. In crypto they are
    usually not: a basket of alts is one BTC beta expressed five ways, so a "diversified"
    five-position book can carry five times the exposure that was actually sized for.
    """

    name = "correlation_limit"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny an entry that would exceed the correlated-position cap."""
        correlated = context.correlated_open_symbols
        limit = context.settings.max_correlated_positions
        if len(correlated) < limit:
            return RiskVerdict.allow(self.name)

        return RiskVerdict.deny(
            self.name,
            f"already holding {len(correlated)} position(s) correlated above "
            f"{context.settings.correlation_threshold:.0%} with "
            f"{context.request.symbol} ({', '.join(correlated)}); "
            f"the cap is {limit}",
            observed=Decimal(len(correlated)),
            limit=Decimal(limit),
            severity=Severity.WARNING,
            symbol=str(context.request.symbol),
        )


class ThesisCooldownRule(RiskRule):
    """Refuses to re-enter a symbol and side straight after being stopped out of it.

    A stop-out is the market answering the question the entry asked. Re-entering the same
    symbol, the same direction, on evidence no stronger than the evidence that just failed
    is not a second opportunity — it is the same opinion paid for twice.

    Measured on 2026-08-17: ``momentum_roc`` was stopped out of WLD long for -235, then
    placed seven further WLD buys within four hours. Two filled, and WLD finished as the
    largest single loss of the session.

    Deliberately not a ban, and deliberately escapable two ways. Time alone clears it, so
    a genuine setup later in the session is not lost; and a materially better score clears
    it early, because sometimes the market really does turn immediately after taking
    someone out. What it refuses is the *unchanged* case, which is the one that has already
    been tested.
    """

    name = "thesis_cooldown"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny a same-side re-entry inside the cooldown unless the case has improved."""
        failed_at = context.last_thesis_failure_at
        if failed_at is None:
            return RiskVerdict.allow(self.name)

        elapsed = context.now - failed_at
        cooldown = timedelta(minutes=context.settings.thesis_cooldown_minutes)
        if elapsed >= cooldown:
            return RiskVerdict.allow(self.name, "the cooldown has expired")
        if elapsed < timedelta(0):
            # The failure is stamped after "now", so the two are not on one clock and the
            # elapsed time is not a real measurement. Enforcing a window computed from it
            # would hold a symbol for as long as the skew happened to be — seen as a
            # -960 minute elapsed producing a 1,020 minute block. A cooldown that cannot
            # be measured is not applied.
            return RiskVerdict.allow(
                self.name, "the recorded failure time is not comparable to the current time"
            )

        previous = context.last_thesis_failure_score
        current = context.candidate_score
        required = context.settings.thesis_score_improvement
        if previous is not None and current is not None and current - previous >= required:
            return RiskVerdict.allow(
                self.name,
                f"score improved from {previous} to {current}, clearing the cooldown early",
            )

        remaining = cooldown - elapsed
        minutes = Decimal(str(round(remaining.total_seconds() / 60, 1)))
        detail = (
            f"score {current} is not {required} better than the {previous} that failed"
            if previous is not None and current is not None
            else "no score was recorded to compare against"
        )
        return RiskVerdict.deny(
            self.name,
            f"{context.request.symbol} {context.request.side.value} was stopped out "
            f"{Decimal(str(round(elapsed.total_seconds() / 60, 1)))} minute(s) ago; "
            f"{detail}, so this re-enters a thesis that has already failed. "
            f"Clears in {minutes} minute(s) or on a better score",
            severity=Severity.WARNING,
        )


class ConsecutiveLossCooldownRule(RiskRule):
    """Pauses new entries after a run of losing trades.

    A losing streak is usually the market saying the regime the strategy was built for is
    no longer the regime being traded. Continuing to fire into it is how a bad day becomes
    a bad month. The pause expires on its own — this is a brake, not a latch, and it must
    not require an operator to come and clear it.
    """

    name = "consecutive_loss_cooldown"

    def check(self, context: RiskContext) -> RiskVerdict:
        """Deny entries while a post-streak cooldown is still running."""
        limit = context.settings.consecutive_loss_limit
        if context.consecutive_losses < limit or context.last_loss_at is None:
            return RiskVerdict.allow(self.name)

        elapsed = context.now - context.last_loss_at
        cooldown = timedelta(minutes=context.settings.loss_cooldown_minutes)
        if elapsed >= cooldown:
            return RiskVerdict.allow(self.name, "cooldown has expired")

        remaining = cooldown - elapsed
        minutes = Decimal(str(round(remaining.total_seconds() / 60, 1)))
        return RiskVerdict.deny(
            self.name,
            f"{context.consecutive_losses} consecutive losses hit the limit of {limit}; "
            f"new entries pause for another {minutes} minute(s)",
            observed=Decimal(context.consecutive_losses),
            limit=Decimal(limit),
            severity=Severity.WARNING,
        )


#: The standard rule set, ordered cheapest-and-hardest first: a latched kill switch or a
#: breached loss limit should short-circuit before anything computes a notional.
DEFAULT_RULES: tuple[type[RiskRule], ...] = (
    KillSwitchRule,
    TradingHaltedRule,
    MaxDrawdownRule,
    MaxWeeklyLossRule,
    MaxDailyLossRule,
    ConsecutiveLossCooldownRule,
    ThesisCooldownRule,
    OrderRateRule,
    StopLossRequiredRule,
    InstrumentRule,
    OrderNotionalRule,
    MaxConcurrentPositionsRule,
    CorrelationLimitRule,
    MaxPositionSizeRule,
    MaxTotalExposureRule,
    MaxLeverageRule,
    SufficientCashRule,
)


def build_default_rules() -> list[RiskRule]:
    """Instantiate the standard rule set."""
    return [rule_class() for rule_class in DEFAULT_RULES]
