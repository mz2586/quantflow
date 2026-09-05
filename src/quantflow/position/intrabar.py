"""Intrabar position management: what happens to an open position between candle closes.

A strategy that trades 5m or 15m bars only ever speaks at a close. Between those closes a
position is unattended: it can run +0.9% in favour, round-trip the whole move, and the
strategy will not have had a single opportunity to object. The exit that eventually fires
is priced off a bar that closed after the opportunity had already gone. On a 15m timeframe
that unattended window is 15 minutes of every 15 minutes.

This module is the reaction to every live price tick in that window. It is deliberately a
**pure function over state**:

* No IO, no network, no clock reads, no repository lookups. Every input arrives as an
  argument — the price, the ATR, the config, and the current time. A function that can
  fetch is a function that can fetch *the future*, and look-ahead never announces itself.
* No knowledge of the strategy. The module is never told whether the strategy currently
  says HOLD, and it must not be: profit protection that a strategy opinion can veto is not
  protection. The strategy's job is deciding what to own; this module's job is making sure
  a position that has already moved does not hand the move back.
* No knowledge of the venue. Symbols are plain string labels and quantities are **not**
  rounded to the exchange lot step — the caller owns tick/step rounding, because only the
  caller knows the instrument. See :class:`ManagementAction`.

Four things can end a position between candle closes, and they are evaluated in the
priority order declared below:

* a **loss rule** — the hard max loss the risk model sized the trade on, a supplied thesis
  invalidation level, or an adverse move abnormal enough (as a fraction of the stop
  distance, or as a multiple of ATR inside a short window) that waiting for the full stop
  is a choice rather than a plan;
* the **net-profit exit** — expected realised PnL *after* entry fee, exit fee, spread and
  slippage has cleared a configured buffer, so the trade is banked on the tick instead of
  waiting for a candle, a target, a strategy opinion or the last rung of the ladder;
* the **profit ladder and trail** — the original behaviour, unchanged;
* the **stale loser** — negative for longer than a configured holding period without ever
  having shown the edge it was opened for.

The two invariants that matter most, in order:

1. **The stop ratchets.** It moves toward profit or it does not move. A stop that can
   loosen is worse than no stop at all: it converts a bounded, known loss into an unbounded
   one at exactly the moment the market is proving the thesis wrong. Every stop change in
   this file goes through :func:`ratchet_stop`; there is no other path.
2. **A stage fires once.** Duplicate close orders are one of the few ways an automated
   system can lose money faster than the market can. ``stages_done`` is the record, and it
   is carried in the state so it survives a restart or a reconnect gap.

Everything here is off by default (``IntrabarConfig.enabled = False``). This changes how
*every* position in the system exits, which is not a change that should be inherited by an
older config file or arrive as a side effect of a deploy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final, Self

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ONE, ZERO, safe_divide, to_decimal
from quantflow.domain.enums import PositionSide

# --------------------------------------------------------------------------- #
# Exit priority
# --------------------------------------------------------------------------- #
# Several layers can all want to act on the same position on the same tick. Without a
# declared order the winner is whichever code path happens to run last, which is a race
# dressed up as a policy. Lower number = higher priority; ties resolve to the first action
# offered, so the caller's own ordering breaks a genuine draw rather than something
# arbitrary like a set iteration order.
#
# The full ladder, highest priority first:
#
# 1. emergency / account risk flatten      :data:`PRIORITY_RISK_FLATTEN`
# 2. hard protective stop (max loss)       :data:`PRIORITY_EXCHANGE_STOP`
# 3. thesis invalidation                   :data:`PRIORITY_THESIS_INVALIDATION`
# 4. loss acceleration / abnormal adverse  :data:`PRIORITY_LOSS_ACCELERATION`
# 5. net-positive profit exit              :data:`PRIORITY_NET_PROFIT_EXIT`
# 6. intrabar trailing / profit protection :data:`PRIORITY_INTRABAR`
# 7. strategy exit and fixed target        :data:`PRIORITY_STRATEGY_EXIT`
# 8. time exits (incl. the stale loser)    :data:`PRIORITY_TIME_EXIT`
#
# The ordering is *loss first, profit second, opinion last*. Everything that protects
# capital outranks everything that harvests it, and everything computed from realised price
# behaviour outranks anything inferred from a closed bar.

#: 1 — account-level protection: max drawdown, daily loss limit, kill switch. A flatten
#: order from here is not an opinion about this position; it is the account withdrawing
#: permission to hold any position at all, so nothing may outrank it.
PRIORITY_RISK_FLATTEN: Final = 1

#: 2 — the hard protective stop: the definitive max loss the risk model sized this trade
#: on, and the stop resting **at the exchange** that enforces it. The venue-side stop is
#: the only layer that keeps working when this process is dead, disconnected or wedged, so
#: nothing below this rank may countermand it; the in-process hard-loss check carries the
#: same rank because it is the same decision arriving a moment earlier.
PRIORITY_EXCHANGE_STOP: Final = 2

#: 3 — thesis invalidation: price has traded through the level at which the reason for
#: holding stopped being true. Only ever evaluated against a level someone supplied
#: (:attr:`PositionState.invalidation_price`); this module never infers a thesis.
PRIORITY_THESIS_INVALIDATION: Final = 3

#: 4 — loss acceleration: an adverse move far enough or fast enough to be abnormal rather
#: than noise. Ranked below invalidation because it is a statement about *velocity*, not
#: about whether the trade was right.
PRIORITY_LOSS_ACCELERATION: Final = 4

#: 5 — the net-positive profit exit: realised PnL after all exit costs has cleared the
#: configured buffer, so the trade can be banked *now*. Below every loss rule (a position
#: that is both profitable and blowing through its stop is not profitable, it is stale
#: data) and above every other harvesting rule, because a profit that exists after costs
#: beats a profit that a trail, a target or a strategy might still hand back.
PRIORITY_NET_PROFIT_EXIT: Final = 5

#: 6 — this module: intrabar trailing and profit protection. Above the strategy because it
#: acts on *realised price behaviour* (the position moved, the stop ratcheted, the price
#: came back through it), while the strategy acts on an inference from a closed bar.
PRIORITY_INTRABAR: Final = 6

#: 7 — the strategy's own exit signal and the fixed take-profit target.
PRIORITY_STRATEGY_EXIT: Final = 7

#: 8 — time-based exits: max holding period, session end, the stale loser. The weakest
#: reason to close, so it must never pre-empt a real one.
PRIORITY_TIME_EXIT: Final = 8

#: Sentinel priority for "do nothing". Not a real rank — it exists so that a NONE action
#: compares as the loser against every genuine action, and so an accidental comparison
#: never silently promotes inaction over a close.
PRIORITY_NONE: Final = 99


class StageAction(StrEnum):
    """What a profit stage does when its trigger is reached."""

    BREAKEVEN = "breakeven"
    """Move the stop to entry plus the estimated round-trip fee."""

    LOCK_PROFIT = "lock_profit"
    """Move the stop to a price that banks a specified net profit."""

    PARTIAL_EXIT = "partial_exit"
    """Close a fraction of the position and let the remainder run on a trail."""


class ActionKind(StrEnum):
    """What the caller must actually do with the exchange."""

    NONE = "none"
    MOVE_STOP = "move_stop"
    PARTIAL_CLOSE = "partial_close"
    FULL_CLOSE = "full_close"


@dataclass(frozen=True, slots=True)
class ProfitStage:
    """One rung of the profit-protection ladder.

    A stage is a promise: *once the position has shown me this much, I will never again let
    it show me less than that.* Stages are cumulative and monotonic by construction — the
    stop each one implies can only be better than the last, because every stop change goes
    through the ratchet.
    """

    #: Unrealised gain, as a fraction of entry price, that activates the stage.
    #: ``Decimal("0.0025")`` is +0.25%.
    trigger_pct: Decimal

    action: StageAction

    #: For :attr:`StageAction.LOCK_PROFIT`: the profit to bank, as a fraction of entry,
    #: **net of the round-trip fee estimate**. Locking 0.20% means 0.20% actually kept.
    lock_pct: Decimal | None = None

    #: For :attr:`StageAction.PARTIAL_EXIT`: the fraction of the **original** quantity to
    #: close. Of the original rather than the remaining, so that a ladder of partials is
    #: predictable: three 0.33 stages take roughly the whole position, instead of
    #: compounding down to a dust-sized residue that costs more in fees than it can earn.
    partial_fraction: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate the stage."""
        if self.trigger_pct <= ZERO:
            raise ValidationError(f"stage trigger must be positive, got {self.trigger_pct}")
        if self.action is StageAction.LOCK_PROFIT:
            if self.lock_pct is None or self.lock_pct <= ZERO:
                raise ValidationError("LOCK_PROFIT requires a positive lock_pct")
            if self.lock_pct >= self.trigger_pct:
                # Locking more than the move has produced would place the stop above the
                # price that triggered the stage, i.e. an instant stop-out.
                raise ValidationError(
                    f"lock_pct {self.lock_pct} must be below trigger_pct {self.trigger_pct}"
                )
        if self.action is StageAction.PARTIAL_EXIT and (
            self.partial_fraction is None
            or self.partial_fraction <= ZERO
            or self.partial_fraction > ONE
        ):
            raise ValidationError("PARTIAL_EXIT requires partial_fraction in (0, 1]")


#: The default ladder. Percentages are fractions of entry price, not percentage points.
#:
#: The spacing is the whole design. +0.25% is roughly two spreads plus the round trip on a
#: liquid perp — the first point at which the trade has genuinely stopped being an idea and
#: started being a position, and therefore the first point at which "give none of it back"
#: is a defensible promise rather than a way of getting stopped out by noise. Doubling to
#: +0.50% before banking anything, and 1.5x again before taking size off, keeps each rung
#: outside the band the previous one was fighting.
#:
#: These are not fitted values. They were not chosen by sweeping a backtest, because a
#: profit ladder tuned on the history it is measured against will always look excellent and
#: will always be measuring its own overfit.
DEFAULT_STAGES: Final[tuple[ProfitStage, ...]] = (
    ProfitStage(trigger_pct=Decimal("0.0025"), action=StageAction.BREAKEVEN),
    ProfitStage(
        trigger_pct=Decimal("0.0050"),
        action=StageAction.LOCK_PROFIT,
        lock_pct=Decimal("0.0020"),
    ),
    ProfitStage(
        trigger_pct=Decimal("0.0075"),
        action=StageAction.PARTIAL_EXIT,
        partial_fraction=Decimal("0.33"),
    ),
)


@dataclass(frozen=True, slots=True)
class IntrabarConfig:
    """Everything the intrabar layer is allowed to decide with.

    Every number is configurable and every default is documented, because a default that
    nobody can explain is a fitted constant with better manners.
    """

    #: The profit ladder, in ascending trigger order.
    stages: tuple[ProfitStage, ...] = DEFAULT_STAGES

    #: Trailing distance in ATR units. 1.5x is wide enough that ordinary retracement inside
    #: a live trend does not clip the position, and tight enough that a genuine reversal is
    #: caught well before the next 15m close would notice it.
    trail_atr_multiple: Decimal = Decimal("1.5")

    #: Absolute floor on the trailing distance, as a fraction of price. ATR collapses in a
    #: quiet market; without a floor the trail would sit inside the bid-ask noise and every
    #: position would be closed by the spread rather than by the market. 0.30% is
    #: comfortably outside a liquid perp's spread and a normal tick's jitter.
    min_trail_pct: Decimal = Decimal("0.003")

    #: Index of the stage whose firing switches trailing on; ``None`` disables trailing
    #: entirely. Defaults to the partial-exit rung: taking size off and trailing the
    #: remainder are one decision, not two. Before that rung the fixed stage stops are
    #: tighter and more predictable than a trail would be.
    trail_after_stage: int | None = 2

    #: Estimated **round-trip** fee, as a fraction of notional. Bybit taker is ~0.06% a
    #: side, so 0.12% both ways; maker-first entries bring it nearer 0.07%, which makes
    #: 0.12% the conservative choice. This exists because a "breakeven" stop placed at the
    #: entry price is not breakeven — it is a small guaranteed loss, repeated on every
    #: trade that reaches the first rung, which is most of them.
    fee_rate: Decimal = Decimal("0.0012")

    # ----------------------------------------------------------------- #
    # The cost model behind the net-profit exit
    # ----------------------------------------------------------------- #
    # :attr:`fee_rate` is the round trip as one number, which is all the ladder needs to
    # offset a stop by. Deciding whether a position is *actually* in profit needs the
    # round trip itemised, because the two fees are not the only cost of getting out and
    # a "profit" that ignores the spread is a profit that does not survive contact with
    # the order book. entry + exit below sum to the same 0.12% by default.

    #: Taker fee paid getting in, as a fraction of notional. Bybit taker is 0.06%.
    entry_fee_pct: Decimal = Decimal("0.0006")

    #: Taker fee paid getting out. The exit this module signals is a market order — it is
    #: an exit that must happen *now* — so the taker rate is the honest assumption.
    exit_fee_pct: Decimal = Decimal("0.0006")

    #: Half-spread paid crossing the book, as a fraction of price. 0.02% is a conservative
    #: figure for a liquid USDT perp; illiquid symbols should raise it. A net-profit rule
    #: that ignores the spread systematically closes trades that were never profitable.
    spread_pct: Decimal = Decimal("0.0002")

    #: Expected slippage on a market exit, as a fraction of price. Separate from the
    #: spread because they move for different reasons: the spread widens with the book,
    #: slippage grows with order size and speed.
    slippage_pct: Decimal = Decimal("0.0002")

    #: The buffer the net-of-costs profit must clear before the position is banked. 0.05%
    #: on top of ~0.16% of costs means the exit fires around +0.21% gross — small enough
    #: to be reachable inside a 15m bar, large enough that it is not paying fees to
    #: harvest the spread. Zero would mean closing at exactly break-even, which is a
    #: round trip's worth of risk taken for nothing.
    min_net_profit_pct: Decimal = Decimal("0.0005")

    #: The net-profit exit itself. Off in the dataclass default and **on** via
    #: :func:`quantflow.live.intrabar_manager.intrabar_config_from_env`, which is the
    #: same split :attr:`enabled` uses and for the same reason: this rule supersedes the
    #: profit ladder for most positions, so it may not arrive by inheritance in a config
    #: object written before it existed. The live engine gets it without setting anything.
    #:
    #: The split matters because this rule fires well below the first rung of the default
    #: ladder: a config that inherited it silently would have replaced staged profit
    #: protection with a scalp on every position in the system, without anybody choosing
    #: that. ``QF_NET_PROFIT_EXIT=false`` turns it off again for the live engine.
    net_profit_exit_enabled: bool = False

    # ----------------------------------------------------------------- #
    # Losing positions
    # ----------------------------------------------------------------- #
    # Every threshold here is derived per position from its own stop distance and its own
    # ATR. None of it is a flat percentage applied to every asset, because "1% adverse" is
    # noise on one symbol and a thesis failure on another.

    #: Close when price reaches the definitive max loss, expressed as a multiple of the
    #: **initial** risk (entry to the stop the position was sized on). 1.0 is exactly that
    #: stop: this rule does not tighten risk, it makes the risk model's own limit act on a
    #: tick instead of waiting for a bar. The exchange-side stop stays exactly where it is
    #: and remains authoritative — nothing here amends or cancels it.
    hard_max_loss_enabled: bool = True
    hard_max_loss_r: Decimal = Decimal("1")

    #: Close when price trades through :attr:`PositionState.invalidation_price`. Inactive
    #: unless a level was explicitly supplied: this module never infers a thesis from price
    #: action, and an invented invalidation level is a strategy opinion with no strategy
    #: behind it.
    invalidation_exit_enabled: bool = True

    #: Exit early when the adverse move is abnormal rather than ordinary. Two ways to
    #: qualify, and both are measured in the position's own units:
    #:
    #: * the excursion has eaten :attr:`loss_accel_stop_fraction` of the stop distance
    #:   *and* is at least :attr:`loss_accel_atr_multiple` × ATR — deep and abnormal, not
    #:   merely deep in a symbol whose ATR is large;
    #: * or the move **since the previous tick** is at least
    #:   :attr:`loss_accel_burst_atr_multiple` × ATR inside :attr:`loss_accel_window` —
    #:   the gap/liquidation cascade case, where the full stop is a price nobody will fill
    #:   at by the time it is reached.
    loss_accel_enabled: bool = True
    loss_accel_stop_fraction: Decimal = Decimal("0.6")
    loss_accel_atr_multiple: Decimal = Decimal("1.5")
    loss_accel_burst_atr_multiple: Decimal = Decimal("2")
    loss_accel_window: timedelta = timedelta(seconds=60)

    #: Close a position that has been negative for longer than :attr:`stale_loser_after`
    #: and never, at its best, showed enough favourable excursion to clear the net-profit
    #: buffer. Capital in a trade that has not worked in an hour is capital the next setup
    #: cannot use, and a position held past its thesis is held on hope. One hour is four
    #: 15m bars: long enough that a slow-starting trade is not cut on entry noise.
    stale_loser_enabled: bool = True
    stale_loser_after: timedelta = timedelta(hours=1)

    #: OFF by default. Switching this on changes the exit behaviour of every position in
    #: the system, so it must be a deliberate act rather than something an old config file
    #: inherits or a deploy turns on by omission.
    enabled: bool = False

    @property
    def round_trip_cost_pct(self) -> Decimal:
        """Everything getting in and back out costs, as a fraction of entry price.

        Both fees plus the spread plus expected slippage. This is the number a position has
        to beat before the word "profit" means anything.
        """
        return self.entry_fee_pct + self.exit_fee_pct + self.spread_pct + self.slippage_pct

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not self.stages:
            raise ValidationError("at least one profit stage is required")
        triggers = [stage.trigger_pct for stage in self.stages]
        if triggers != sorted(triggers):
            raise ValidationError(f"stages must be in ascending trigger order, got {triggers}")
        if self.trail_atr_multiple < ZERO:
            raise ValidationError(f"trail_atr_multiple must be >= 0, got {self.trail_atr_multiple}")
        if self.min_trail_pct <= ZERO:
            raise ValidationError(
                f"min_trail_pct must be positive — a zero floor lets a quiet market trail "
                f"inside the noise; got {self.min_trail_pct}"
            )
        if self.fee_rate < ZERO:
            raise ValidationError(f"fee_rate must be >= 0, got {self.fee_rate}")
        if self.trail_after_stage is not None and not (
            0 <= self.trail_after_stage < len(self.stages)
        ):
            raise ValidationError(
                f"trail_after_stage {self.trail_after_stage} is outside the "
                f"{len(self.stages)}-stage ladder"
            )
        self._validate_exit_rules()

    def _validate_exit_rules(self) -> None:
        """Validate the cost model and the loser thresholds.

        Split out from :meth:`__post_init__` only to keep each half readable; both run on
        every construction, because a config object that validated half of itself would be
        a config object nobody could trust.
        """
        for name in ("entry_fee_pct", "exit_fee_pct", "spread_pct", "slippage_pct"):
            value = getattr(self, name)
            if value < ZERO:
                # A negative cost is a rebate this module is not entitled to assume, and
                # it would let the net-profit exit fire on a position that is still red.
                raise ValidationError(f"{name} must be >= 0, got {value}")
        if self.min_net_profit_pct < ZERO:
            raise ValidationError(
                f"min_net_profit_pct must be >= 0 — a negative buffer would bank losses "
                f"as profits; got {self.min_net_profit_pct}"
            )
        if self.hard_max_loss_r <= ZERO:
            raise ValidationError(
                f"hard_max_loss_r must be positive — zero would close every position at "
                f"its entry price; got {self.hard_max_loss_r}"
            )
        if self.loss_accel_stop_fraction <= ZERO:
            raise ValidationError(
                f"loss_accel_stop_fraction must be positive, got {self.loss_accel_stop_fraction}"
            )
        if self.loss_accel_atr_multiple < ZERO or self.loss_accel_burst_atr_multiple <= ZERO:
            raise ValidationError("loss acceleration ATR multiples must be positive")
        if self.loss_accel_window <= timedelta(0):
            raise ValidationError(
                f"loss_accel_window must be positive, got {self.loss_accel_window}"
            )
        if self.stale_loser_after <= timedelta(0):
            raise ValidationError(
                f"stale_loser_after must be positive — a zero timeout closes every "
                f"position that is red on its first tick; got {self.stale_loser_after}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionState:
    """Everything this module knows about one open position.

    Keyword-only and frozen: this is a value that gets persisted, restored, and compared,
    and a positional constructor with this many ``Decimal`` fields is a silent
    argument-transposition bug waiting for a bad night to happen.

    Every tick returns a **new** state. The caller persists it; nothing here mutates in
    place, so a crash between the decision and the write leaves the previous state intact
    and merely repeats a tick rather than corrupting a ladder.
    """

    #: Plain label, e.g. ``"BTC/USDT"``. Deliberately not a ``Symbol``: this module never
    #: looks up an instrument, so carrying one would be coupling with no payoff.
    symbol: str

    side: PositionSide
    entry_price: Decimal

    #: Currently open quantity. Reaches zero when a full close is signalled, which is also
    #: how "this position is done, stop acting on it" is represented.
    quantity: Decimal

    #: Size at open. Partial fractions are taken against this, never against what is left.
    original_quantity: Decimal

    #: The live protective stop. Only ever moves through :func:`ratchet_stop`.
    current_stop: Decimal

    #: The stop as first placed. Kept separately because R multiple needs a *fixed*
    #: denominator: measured against a ratcheting stop, R climbs to infinity as the stop
    #: approaches entry, which would make it a number that flatters instead of informing.
    initial_stop: Decimal

    #: Fixed take-profit, if the strategy set one.
    target: Decimal | None

    #: The price at which the reason for holding this position stops being true, if the
    #: caller knows one. **Optional and never inferred.**
    #:
    #: This module is pure and is deliberately never handed strategy state (see
    #: :func:`on_price`), so a thesis can only arrive the way any other fact does: as a
    #: number on the state, set at entry or at adoption by whoever *does* know the
    #: strategy. Left ``None`` — which is what happens whenever no strategy published a
    #: level — the invalidation rule is simply inactive. Guessing a level from price
    #: action would be this module inventing a strategy opinion and then obeying it.
    invalidation_price: Decimal | None = None

    #: Highest price seen since entry — the reference a long's trail hangs below.
    high_water: Decimal

    #: Lowest price seen since entry — the reference a short's trail hangs above.
    low_water: Decimal

    opened_at: datetime

    #: Indices of stages already fired. The anti-duplicate record; persisted with the rest
    #: of the state so a restart cannot re-fire a partial that already executed.
    stages_done: frozenset[int] = frozenset()

    #: PnL from partials and closes this module has signalled, gross of fees, marked at the
    #: *tick* price rather than a fill price. It is the module's own running estimate so it
    #: can reason about what it has already given away; the FIFO ledger in
    #: ``quantflow.domain.positions`` remains the authority for accounting.
    realized_pnl: Decimal = ZERO

    #: Timestamp of the last tick applied. Feeds :func:`is_stale` without this module ever
    #: reading a clock itself.
    last_price_at: datetime | None = None

    #: The previous tick's price. Kept because *speed* is a fact about two ticks, not one:
    #: the loss-acceleration rule needs to know how far price moved since it last looked
    #: and how long that took, and a pure function cannot remember it any other way.
    last_price: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate the state."""
        if self.side not in (PositionSide.LONG, PositionSide.SHORT):
            raise ValidationError(
                f"intrabar management needs a directional position, got {self.side}"
            )
        if self.entry_price <= ZERO:
            raise ValidationError(f"entry_price must be positive, got {self.entry_price}")
        if self.original_quantity <= ZERO:
            raise ValidationError(
                f"original_quantity must be positive, got {self.original_quantity}"
            )
        if self.quantity < ZERO:
            raise ValidationError(f"quantity must be >= 0, got {self.quantity}")
        if self.quantity > self.original_quantity:
            raise ValidationError(
                f"quantity {self.quantity} exceeds original_quantity {self.original_quantity}"
            )
        if self.current_stop <= ZERO or self.initial_stop <= ZERO:
            raise ValidationError("stops must be positive prices")
        if self.high_water <= ZERO or self.low_water <= ZERO:
            raise ValidationError("water marks must be positive prices")
        if self.opened_at.tzinfo is None:
            raise ValidationError("opened_at must be timezone-aware")

    @classmethod
    def from_entry(
        cls,
        *,
        symbol: str,
        side: PositionSide,
        entry_price: Decimal,
        quantity: Decimal,
        stop: Decimal,
        opened_at: datetime,
        target: Decimal | None = None,
        invalidation_price: Decimal | None = None,
    ) -> Self:
        """Build the state a freshly opened position starts from.

        Both water marks begin at the entry price: before the first tick the best the
        position has done is exactly nothing, and seeding them from anywhere else would
        hand the trail a favourable excursion that never happened.
        """
        return cls(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            original_quantity=quantity,
            current_stop=stop,
            initial_stop=stop,
            target=target,
            invalidation_price=invalidation_price,
            high_water=entry_price,
            low_water=entry_price,
            opened_at=opened_at,
        )

    @property
    def is_long(self) -> bool:
        """Whether the position is long."""
        return self.side is PositionSide.LONG

    @property
    def is_closed(self) -> bool:
        """Whether nothing is left to manage."""
        return self.quantity <= ZERO

    @property
    def initial_risk(self) -> Decimal:
        """Distance from entry to the original stop — the ``1R`` this trade was sized on."""
        return abs(self.entry_price - self.initial_stop)

    @property
    def favourable_extreme(self) -> Decimal:
        """The water mark that matters for this side."""
        return self.high_water if self.is_long else self.low_water

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe primitives.

        ``Decimal`` goes out as a **string**, never a float: a restart that reloads
        ``0.1`` as ``0.1000000000000000055`` has silently moved a stop, and the whole point
        of persisting this state is that a restart changes nothing.
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": str(self.entry_price),
            "quantity": str(self.quantity),
            "original_quantity": str(self.original_quantity),
            "current_stop": str(self.current_stop),
            "initial_stop": str(self.initial_stop),
            "target": None if self.target is None else str(self.target),
            "invalidation_price": (
                None if self.invalidation_price is None else str(self.invalidation_price)
            ),
            "high_water": str(self.high_water),
            "low_water": str(self.low_water),
            "opened_at": self.opened_at.isoformat(),
            "stages_done": sorted(self.stages_done),
            "realized_pnl": str(self.realized_pnl),
            "last_price_at": None if self.last_price_at is None else self.last_price_at.isoformat(),
            "last_price": None if self.last_price is None else str(self.last_price),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Restore from :meth:`to_dict`. Round-tripping must be exact, including stages."""
        target = payload.get("target")
        invalidation = payload.get("invalidation_price")
        last_price_at = payload.get("last_price_at")
        last_price = payload.get("last_price")
        return cls(
            symbol=str(payload["symbol"]),
            side=PositionSide(payload["side"]),
            entry_price=to_decimal(payload["entry_price"]),
            quantity=to_decimal(payload["quantity"]),
            original_quantity=to_decimal(payload["original_quantity"]),
            current_stop=to_decimal(payload["current_stop"]),
            initial_stop=to_decimal(payload["initial_stop"]),
            target=None if target is None else to_decimal(target),
            invalidation_price=None if invalidation is None else to_decimal(invalidation),
            high_water=to_decimal(payload["high_water"]),
            low_water=to_decimal(payload["low_water"]),
            opened_at=datetime.fromisoformat(str(payload["opened_at"])),
            stages_done=frozenset(int(index) for index in payload.get("stages_done", ())),
            realized_pnl=to_decimal(payload.get("realized_pnl", ZERO)),
            last_price_at=(
                None if last_price_at is None else datetime.fromisoformat(str(last_price_at))
            ),
            last_price=None if last_price is None else to_decimal(last_price),
        )


@dataclass(frozen=True, slots=True)
class ManagementAction:
    """An instruction to the caller. This module never places an order itself.

    ``close_quantity`` is **unrounded**: this module has no instrument metadata and must
    not invent a lot step. The caller rounds down to the exchange step before sending —
    down, so an over-close can never turn a partial into an accidental reversal.

    A single action can carry both a stop move and a close: when a tick gaps far enough to
    trip the partial rung and the rungs below it at once, the stop change and the partial
    are one decision, and splitting them into two actions would let a caller apply one and
    drop the other.
    """

    kind: ActionKind
    new_stop: Decimal | None = None
    close_quantity: Decimal | None = None
    reason: str = ""
    priority: int = PRIORITY_INTRABAR

    @classmethod
    def none(cls, reason: str = "") -> Self:
        """Do nothing — carries :data:`PRIORITY_NONE` so it loses every comparison."""
        return cls(kind=ActionKind.NONE, reason=reason, priority=PRIORITY_NONE)

    @property
    def is_actionable(self) -> bool:
        """Whether the caller has to do anything at all."""
        return self.kind is not ActionKind.NONE


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def unrealized_pct(state: PositionState, price: Decimal) -> Decimal:
    """Unrealised gain as a fraction of entry, signed so positive is always favourable.

    Sign-normalising here is what lets every rule below be written once instead of twice,
    and a rule written twice is a rule that will eventually disagree with itself.
    """
    direction = Decimal(state.side.sign)
    return safe_divide((price - state.entry_price) * direction, state.entry_price)


def net_profit_pct(state: PositionState, price: Decimal, config: IntrabarConfig) -> Decimal:
    """What closing **right now** would actually keep, as a fraction of entry price.

    The gross move in the position's favour — ``(price - entry) / entry`` for a long,
    ``(entry - price) / entry`` for a short, which :func:`unrealized_pct` already
    sign-normalises — minus every cost of having been in and getting out:
    :attr:`IntrabarConfig.entry_fee_pct`, :attr:`~IntrabarConfig.exit_fee_pct`,
    :attr:`~IntrabarConfig.spread_pct` and :attr:`~IntrabarConfig.slippage_pct`.

    The entry fee is included even though it is already spent. It is not a decision cost —
    nothing can un-pay it — but this number answers *"what did the trade make"*, and a
    trade whose gross move only covers the exit is a trade that made nothing. Excluding it
    would produce a stream of exits that each look profitable and sum to a loss.

    Negative means the position is under water once the book is priced in, however green
    the unrealised PnL looks on a screen that quotes mid.
    """
    return unrealized_pct(state, price) - config.round_trip_cost_pct


def adverse_excursion(state: PositionState, price: Decimal) -> Decimal:
    """How far price sits **against** the position, in price units. Never negative.

    Zero for anything at or in front of the entry price, so every loser rule can be
    written once for both sides and read as "how much has this cost me so far".
    """
    return max(ZERO, adverse_move(state.side, state.entry_price, price))


def adverse_move(side: PositionSide, reference: Decimal, price: Decimal) -> Decimal:
    """Signed distance price has travelled against ``side`` since ``reference``.

    Positive is adverse for both sides — a long losing as price falls and a short losing
    as it rises produce the same positive number — which is what lets the loss rules be
    mirrored by construction instead of by a second branch that can drift out of step.
    """
    return (reference - price) * Decimal(side.sign)


def r_multiple(state: PositionState, price: Decimal) -> Decimal:
    """Unrealised gain in units of the original risk. Zero when the trade had no stop room."""
    direction = Decimal(state.side.sign)
    return safe_divide((price - state.entry_price) * direction, state.initial_risk)


def ratchet_stop(side: PositionSide, current: Decimal, candidate: Decimal) -> Decimal:
    """Return the tighter of two stops — the only way a stop is ever allowed to change.

    A long's stop may rise and a short's may fall; never the reverse. Expressing this as a
    ``max``/``min`` rather than an ``if candidate is better`` check means there is no
    branch in which a loosening value can slip through, including on the paths where the
    candidate was computed from a stale water mark or an out-of-order tick.
    """
    return max(current, candidate) if side is PositionSide.LONG else min(current, candidate)


def trail_distance(price: Decimal, atr: Decimal | None, config: IntrabarConfig) -> Decimal:
    """How far behind the water mark the trail sits.

    ``max(atr * multiple, price * min_trail_pct)``. The ATR term makes the trail respect
    the volatility actually present; the percentage floor stops a quiet tape — or a missing
    ATR — from collapsing the trail into the spread, where it would be closed out by noise
    rather than by any change in the market.
    """
    volatility_term = (atr or ZERO) * config.trail_atr_multiple
    floor = price * config.min_trail_pct
    return max(volatility_term, floor)


def is_stale(last_tick_age: timedelta, max_age: timedelta) -> bool:
    """Whether the price feed is too old to be trusted for **new** decisions.

    The rule, which the caller must implement and which is asymmetric on purpose:

    * Stale data blocks **new entries**. Opening a position on a price that may be minutes
      old is trading a fiction.
    * Stale data does **not** suspend position management. A position is already exposed;
      refusing to manage it because the feed is late leaves it exposed *and* unwatched,
      which is strictly worse than acting on the last known price.
    * Exchange-side protection is unaffected in either case — it lives on the venue and
      keeps working whether or not this process can see a price at all
      (:data:`PRIORITY_EXCHANGE_STOP`).
    """
    return last_tick_age > max_age


def resolve_actions(actions: Iterable[ManagementAction]) -> ManagementAction:
    """Pick the one action that wins when several layers speak at once.

    Lowest ``priority`` number wins; ties go to whichever was offered first, so the caller's
    own ordering is the tie-break rather than something incidental. ``NONE`` actions are
    discarded whenever any real action is present — inaction must never outrank a close,
    regardless of what priority a caller happened to stamp on it.

    The ranking is a **total order** over the eight levels declared at the top of this
    module — risk flatten, hard stop, thesis invalidation, loss acceleration, net-profit
    exit, intrabar protection, strategy exit, time exit — so any two actions from any two
    layers compare, and the comparison never depends on the order they were generated in
    except when they genuinely rank the same.
    """
    candidates = list(actions)
    if not candidates:
        return ManagementAction.none("no actions offered")
    actionable = [action for action in candidates if action.is_actionable]
    if not actionable:
        return candidates[0]
    winner = actionable[0]
    for action in actionable[1:]:
        if action.priority < winner.priority:
            winner = action
    return winner


# --------------------------------------------------------------------------- #
# The core
# --------------------------------------------------------------------------- #


def on_price(  # noqa: PLR0911 - one return per exit condition, in priority order,
    # reads far more clearly than nesting seven protection rules inside each other.
    state: PositionState,
    price: Decimal,
    *,
    atr: Decimal | None,
    config: IntrabarConfig,
    now: datetime,
) -> tuple[PositionState, ManagementAction]:
    """React to one live price tick. Pure: same inputs, same outputs, always.

    Order of evaluation is the exit priority ladder declared at the top of this module,
    and the reason for it:

    1. **Disabled or already flat** — return the state untouched. Disabled means inert, not
       "observe quietly"; a module that keeps updating water marks while switched off would
       act on a favourable excursion nobody was watching the moment it was switched on.
    2. **Water marks** — updated before anything else, so a tick that both extends the run
       and then trips a rule is measured against its own extreme.
    3. **The loss rules**, in their own order: the hard max loss the position was sized on,
       then a supplied thesis invalidation level, then loss acceleration. Capital
       protection is settled before profit is discussed, because a position that is through
       its max loss is not a candidate for anything else.
    4. **The net-profit exit** — if closing now would keep more than the configured buffer
       after every cost, close now. Ahead of the ladder, the target and the strategy on
       purpose: those all wait for something, and a profit that exists after costs is worth
       more than a larger one that may not survive the wait.
    5. **Has the stop or target already been crossed?** Checked before the ladder: if price
       is through the stop, the position needs closing, not a new stop. This is the case
       the whole module exists for — the crossing happened *between* closes and no candle
       will report it.
    6. **The ladder**, ascending, each rung at most once. A gap tick that clears three rungs
       at once fires all three in one pass rather than dribbling them out over three ticks
       the market may never deliver.
    7. **The trail**, once its rung has fired.
    8. **Ratchet** every candidate stop into one, then re-check the crossing: a stop that
       lands at or beyond the current price means the move has already been given back, and
       the honest response is to close now rather than to post a stop that triggers on
       arrival.
    9. **The stale loser**, last, because it is the weakest reason to close: a position
       still red after the configured holding period that never showed the edge it was
       opened for.

    ``now`` is injected rather than read so the function stays testable and deterministic;
    it is recorded on the state, feeds :func:`is_stale`, and is used only for the two rules
    that are explicitly about elapsed time (loss acceleration's window and the stale loser).

    The signature is the design. There is no parameter through which a strategy opinion,
    a repository or a clock could arrive, so profit protection cannot be vetoed by a
    strategy that still says HOLD. That is also why thesis invalidation reads a *level on
    the state* rather than a strategy object: see
    :attr:`PositionState.invalidation_price`.

    Returns:
        The updated state (persist it) and the single action to take.

    Raises:
        ValidationError: if the price is not a positive Decimal.

    """
    if price <= ZERO:
        raise ValidationError(f"tick price must be positive, got {price}")
    if not config.enabled:
        return state, ManagementAction.none("intrabar management disabled")
    if state.is_closed:
        # Requirement, not an optimisation: once a full close has been signalled every
        # later tick must be silent, or a slow fill turns into a second close order.
        return state, ManagementAction.none("position already flat")

    marked = _mark_water(state, price, now)

    # Velocity is a fact about two ticks, so the loss rules are handed the *pre-tick*
    # state: `marked` has already overwritten the previous price with this one.
    losing = _loss_exit(state, price, atr=atr, config=config, now=now)
    if losing is not None:
        return _close_all(marked, price), losing

    profitable = _net_profit_exit(marked, price, config)
    if profitable is not None:
        return _close_all(marked, price), profitable

    crossed = _crossing_exit(marked, price)
    if crossed is not None:
        return _close_all(marked, price), crossed

    rungs = _run_ladder(marked, price, config)
    done = marked.stages_done | rungs.fired
    stop_candidates = list(rungs.stop_candidates)
    if config.trail_after_stage is not None and config.trail_after_stage in done:
        stop_candidates.append(_trailing_stop(marked, price, atr, config))

    new_stop = marked.current_stop
    for candidate in stop_candidates:
        new_stop = ratchet_stop(marked.side, new_stop, candidate)

    close_quantity = rungs.close_quantity
    reasons = list(rungs.reasons)
    updated = replace(marked, stages_done=done, current_stop=new_stop)

    if _has_crossed(updated.side, price, new_stop):
        # The ratcheted stop is already through the price. Closing beats posting a stop
        # that would trigger the instant it arrived at the venue.
        return _close_all(updated, price), ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            reason=(
                f"protected stop {new_stop} reached at {price} (R {r_multiple(updated, price):.2f})"
            ),
            priority=PRIORITY_INTRABAR,
        )

    stop_moved = new_stop != marked.current_stop
    if close_quantity > ZERO:
        return _partial_close(updated, price, close_quantity, new_stop, reasons)
    if stop_moved:
        return updated, ManagementAction(
            kind=ActionKind.MOVE_STOP,
            new_stop=new_stop,
            reason="; ".join(reasons) if reasons else f"trail to {new_stop}",
            priority=PRIORITY_INTRABAR,
        )

    stale = _stale_loser_exit(updated, price, config, now)
    if stale is not None:
        return _close_all(updated, price), stale
    return updated, ManagementAction.none("no rung reached; stop unchanged")


@dataclass(frozen=True, slots=True)
class _LadderResult:
    """What one pass over the profit ladder produced on a single tick."""

    fired: frozenset[int]
    stop_candidates: tuple[Decimal, ...]
    close_quantity: Decimal
    reasons: tuple[str, ...]


def _run_ladder(state: PositionState, price: Decimal, config: IntrabarConfig) -> _LadderResult:
    """Fire every rung this tick has earned and not yet used, lowest first.

    All of them in one pass, not one per tick: a gap through three rungs is exactly the
    situation the module exists for, and rationing the response to one rung per tick would
    make the protection weakest precisely when the move was fastest.
    """
    fired: set[int] = set()
    stop_candidates: list[Decimal] = []
    close_quantity = ZERO
    reasons: list[str] = []

    gain = unrealized_pct(state, price)
    for index, stage in enumerate(config.stages):
        if index in state.stages_done:
            continue
        if gain < stage.trigger_pct:
            # Ascending order is enforced by IntrabarConfig, so the first unmet trigger
            # ends the ladder for this tick.
            break
        fired.add(index)
        stop_price, quantity, reason = _apply_stage(state, stage, index, config)
        if stop_price is not None:
            stop_candidates.append(stop_price)
        close_quantity += quantity
        reasons.append(reason)

    return _LadderResult(
        fired=frozenset(fired),
        stop_candidates=tuple(stop_candidates),
        close_quantity=close_quantity,
        reasons=tuple(reasons),
    )


def _mark_water(state: PositionState, price: Decimal, now: datetime) -> PositionState:
    """Extend the favourable water mark, and only ever in the favourable direction.

    Also records this tick as the previous one for the next call. Both are written in the
    same place so a rule can never be handed a price and a timestamp that disagree about
    which tick they came from.
    """
    if state.is_long:
        return replace(
            state,
            high_water=max(state.high_water, price),
            last_price_at=now,
            last_price=price,
        )
    return replace(
        state, low_water=min(state.low_water, price), last_price_at=now, last_price=price
    )


def _has_crossed(side: PositionSide, price: Decimal, level: Decimal) -> bool:
    """Whether price has reached a level lying **against** the position (a stop)."""
    return price <= level if side is PositionSide.LONG else price >= level


def _reached_target(side: PositionSide, price: Decimal, target: Decimal) -> bool:
    """Whether price has reached a level lying **in favour of** the position (a target).

    The mirror of :func:`_has_crossed`, spelled out separately rather than negated: ``not
    crossed`` would also be true one tick short of the target, and an exit that fires early
    is not a rounding error, it is a different trade.
    """
    return price >= target if side is PositionSide.LONG else price <= target


def hard_max_loss_price(state: PositionState, config: IntrabarConfig) -> Decimal | None:
    """The definitive max loss for this position, as a price. ``None`` if it has no risk.

    Derived from the position's **own** stop distance — the ``1R`` the risk engine sized
    it on — rather than from a flat percentage, because the same percentage is noise on
    one instrument and a thesis failure on another.

    ``None`` when entry and the initial stop coincide: there is no risk model to enforce,
    and a level equal to the entry price would close every position on its first red tick.
    """
    risk = state.initial_risk
    if risk <= ZERO:
        return None
    return state.entry_price - Decimal(state.side.sign) * config.hard_max_loss_r * risk


def _loss_exit(
    state: PositionState,
    price: Decimal,
    *,
    atr: Decimal | None,
    config: IntrabarConfig,
    now: datetime,
) -> ManagementAction | None:
    """The loss rules, hardest first. ``None`` means this position is not in trouble.

    ``state`` is the **pre-tick** state: the acceleration rule compares this price against
    the previous one, which the current tick is about to overwrite.

    Nothing here amends or cancels the exchange-side stop. Every one of these rules is an
    exit that arrives *earlier* than the venue stop would; the venue stop stays exactly
    where it is and keeps working if this process dies mid-decision, which is the only
    reason it is safe for any of this to live in a Python process at all.
    """
    if config.hard_max_loss_enabled:
        level = hard_max_loss_price(state, config)
        if level is not None and _has_crossed(state.side, price, level):
            return ManagementAction(
                kind=ActionKind.FULL_CLOSE,
                reason=(
                    f"hard max loss: price {price} reached {config.hard_max_loss_r}R "
                    f"({level}) from entry {state.entry_price}"
                ),
                priority=PRIORITY_EXCHANGE_STOP,
            )

    if (
        config.invalidation_exit_enabled
        and state.invalidation_price is not None
        and _has_crossed(state.side, price, state.invalidation_price)
    ):
        return ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            reason=(f"thesis invalidated: price {price} traded through {state.invalidation_price}"),
            priority=PRIORITY_THESIS_INVALIDATION,
        )

    accelerating = _loss_acceleration_reason(state, price, atr, config, now)
    if accelerating is not None:
        return ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            reason=accelerating,
            priority=PRIORITY_LOSS_ACCELERATION,
        )
    return None


def _loss_acceleration_reason(
    state: PositionState,
    price: Decimal,
    atr: Decimal | None,
    config: IntrabarConfig,
    now: datetime,
) -> str | None:
    """Whether the adverse move is abnormal enough to exit ahead of the full stop.

    Two qualifying shapes, both measured in the position's own units so that neither is a
    percentage borrowed from a different instrument:

    * **Deep and abnormal.** The excursion has eaten
      :attr:`~IntrabarConfig.loss_accel_stop_fraction` of the stop distance *and* is at
      least :attr:`~IntrabarConfig.loss_accel_atr_multiple` × ATR. Both clauses are
      required: the first alone is just a tighter stop, and cutting risk the engine
      deliberately sized is not this module's decision to make on a quiet tape.
    * **Fast.** The move **since the previous tick** is at least
      :attr:`~IntrabarConfig.loss_accel_burst_atr_multiple` × ATR inside
      :attr:`~IntrabarConfig.loss_accel_window`. This is the gap, the cascade and the
      liquidation wick — the cases where the stop is a price that will be printed through
      rather than filled at, and where waiting for it is how a 1R loss becomes a 3R one.

    Ordinary noise satisfies neither, which is the point: a position that is merely red is
    a position that is working, and closing it would be paying a round trip to convert a
    fluctuation into a realised loss.
    """
    if not config.loss_accel_enabled:
        return None
    adverse = adverse_excursion(state, price)
    if adverse <= ZERO:
        return None
    return _deep_loss_reason(state, price, adverse, atr, config) or _fast_loss_reason(
        state, price, atr, config, now
    )


def _deep_loss_reason(
    state: PositionState,
    price: Decimal,
    adverse: Decimal,
    atr: Decimal | None,
    config: IntrabarConfig,
) -> str | None:
    """The "deep and abnormal" half of loss acceleration. See the caller for the rationale."""
    risk = state.initial_risk
    if risk <= ZERO:
        return None
    fraction = safe_divide(adverse, risk)
    if fraction < config.loss_accel_stop_fraction:
        return None
    if atr is not None and atr > ZERO and adverse < atr * config.loss_accel_atr_multiple:
        # Deep, but not large for this instrument. Cutting inside the risk the engine
        # deliberately sized, on a move the symbol makes routinely, is a tighter stop
        # wearing an urgent name.
        return None
    return (
        f"loss acceleration: {fraction:.0%} of the {risk} stop distance given up at "
        f"{price} ({adverse} adverse, atr {atr})"
    )


def _fast_loss_reason(
    state: PositionState,
    price: Decimal,
    atr: Decimal | None,
    config: IntrabarConfig,
    now: datetime,
) -> str | None:
    """The "fast" half of loss acceleration: one tick to the next, inside the window."""
    if atr is None or atr <= ZERO or state.last_price is None or state.last_price_at is None:
        # No previous tick, or no volatility estimate to call a move large *relative to
        # what this instrument normally does*. Without both, "fast" has no meaning here.
        return None
    elapsed = now - state.last_price_at
    if elapsed < timedelta(0) or elapsed > config.loss_accel_window:
        return None
    step = adverse_move(state.side, state.last_price, price)
    if step < atr * config.loss_accel_burst_atr_multiple:
        return None
    return (
        f"loss acceleration: {step} against the position in {elapsed.total_seconds():.0f}s "
        f"({safe_divide(step, atr):.1f}x atr) at {price}"
    )


def _net_profit_exit(
    state: PositionState, price: Decimal, config: IntrabarConfig
) -> ManagementAction | None:
    """Close while the profit is real, or return ``None`` and let the ladder work.

    "Real" is the whole rule: net of the entry fee, the exit fee, the spread and expected
    slippage, and then clear of :attr:`~IntrabarConfig.min_net_profit_pct` on top. A
    position can be comfortably green on the screen and still fail this — the buffer exists
    precisely so that a move which only pays the exchange is left alone rather than
    harvested into a fee.

    It fires on the tick that qualifies, without waiting for a candle close, the strategy's
    opinion, the fixed target or the last rung of the ladder. That is the point: on a 15m
    timeframe every one of those is up to fifteen minutes away, and the move that qualified
    is under no obligation to still be there.
    """
    if not config.net_profit_exit_enabled:
        return None
    net = net_profit_pct(state, price, config)
    if net < config.min_net_profit_pct:
        return None
    return ManagementAction(
        kind=ActionKind.FULL_CLOSE,
        close_quantity=state.quantity,
        reason=(
            f"net profit exit: {net:.4%} net of {config.round_trip_cost_pct:.4%} costs at "
            f"{price} (gross {unrealized_pct(state, price):.4%}, buffer "
            f"{config.min_net_profit_pct:.4%})"
        ),
        priority=PRIORITY_NET_PROFIT_EXIT,
    )


def _stale_loser_exit(
    state: PositionState, price: Decimal, config: IntrabarConfig, now: datetime
) -> ManagementAction | None:
    """Close a position that has been red too long without ever showing its edge.

    Three conditions, all required:

    * it is **still negative** right now — a position that has come back is not stale, it
      is working;
    * it has been open longer than :attr:`~IntrabarConfig.stale_loser_after`;
    * and at its **best**, it never got far enough in front to clear the net-profit buffer
      after costs. That is the "no longer shows the required expected edge" test, measured
      against the favourable water mark rather than the current price, so a trade that did
      earn its edge and gave it back is judged by the profit rules and the trail instead.

    Elapsed time comes from ``now`` and :attr:`PositionState.opened_at`. This function
    never reads a clock; a clock inside a pure function is a clock that can be wrong in
    production and unfalsifiable in a test.
    """
    if not config.stale_loser_enabled:
        return None
    if unrealized_pct(state, price) >= ZERO:
        return None
    held = now - state.opened_at
    if held < config.stale_loser_after:
        return None
    best = net_profit_pct(state, state.favourable_extreme, config)
    if best >= config.min_net_profit_pct:
        return None
    return ManagementAction(
        kind=ActionKind.FULL_CLOSE,
        close_quantity=state.quantity,
        reason=(
            f"stale loser: {held} open, still {unrealized_pct(state, price):.4%} at {price}, "
            f"best net excursion {best:.4%} never cleared {config.min_net_profit_pct:.4%}"
        ),
        priority=PRIORITY_TIME_EXIT,
    )


def _crossing_exit(state: PositionState, price: Decimal) -> ManagementAction | None:
    """Close if the live price is already through the stop or the target.

    The stop is checked first and ranks above the target: if a single violent tick prints
    through both, the pessimistic reading is the safe one.
    """
    if _has_crossed(state.side, price, state.current_stop):
        return ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            reason=(
                f"price {price} crossed protective stop {state.current_stop} intrabar "
                f"(R {r_multiple(state, price):.2f})"
            ),
            priority=PRIORITY_INTRABAR,
        )
    if state.target is not None and _reached_target(state.side, price, state.target):
        return ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            reason=f"price {price} reached target {state.target} intrabar",
            priority=PRIORITY_STRATEGY_EXIT,
        )
    return None


def _apply_stage(
    state: PositionState,
    stage: ProfitStage,
    index: int,
    config: IntrabarConfig,
) -> tuple[Decimal | None, Decimal, str]:
    """Turn one triggered rung into ``(stop candidate, close quantity, reason)``."""
    direction = Decimal(state.side.sign)
    if stage.action is StageAction.BREAKEVEN:
        # Entry plus the round trip. A stop at the entry price exits for the cost of the
        # trade — a small, certain loss dressed up as "flat", taken on every trade that
        # gets this far.
        stop = state.entry_price * (ONE + direction * config.fee_rate)
        return stop, ZERO, f"stage {index} breakeven+fees at {stop} ({stage.trigger_pct:.2%})"
    if stage.action is StageAction.LOCK_PROFIT:
        assert stage.lock_pct is not None  # guaranteed by ProfitStage validation
        offset = stage.lock_pct + config.fee_rate
        stop = state.entry_price * (ONE + direction * offset)
        return stop, ZERO, f"stage {index} locks {stage.lock_pct:.2%} net at {stop}"
    assert stage.partial_fraction is not None  # guaranteed by ProfitStage validation
    quantity = min(state.original_quantity * stage.partial_fraction, state.quantity)
    return (
        None,
        quantity,
        f"stage {index} partial exit {stage.partial_fraction:.0%} "
        f"({quantity}) at {stage.trigger_pct:.2%}",
    )


def _trailing_stop(
    state: PositionState,
    price: Decimal,
    atr: Decimal | None,
    config: IntrabarConfig,
) -> Decimal:
    """The trail: a fixed distance behind the best price the position has reached."""
    distance = trail_distance(price, atr, config)
    if state.is_long:
        return state.high_water - distance
    return state.low_water + distance


def _position_pnl(state: PositionState, price: Decimal, quantity: Decimal) -> Decimal:
    """Gross PnL of closing ``quantity`` at ``price``. Fees are the ledger's business."""
    return (price - state.entry_price) * Decimal(state.side.sign) * quantity


def _close_all(state: PositionState, price: Decimal) -> PositionState:
    """Mark the position flat. Quantity zero is what silences every later tick."""
    return replace(
        state,
        quantity=ZERO,
        realized_pnl=state.realized_pnl + _position_pnl(state, price, state.quantity),
    )


def _partial_close(
    state: PositionState,
    price: Decimal,
    quantity: Decimal,
    new_stop: Decimal,
    reasons: list[str],
) -> tuple[PositionState, ManagementAction]:
    """Book a partial — or promote it to a full close when nothing meaningful would remain.

    ``entry_price`` is untouched: a partial changes how much is held, never what it cost.
    Recomputing an average on the way out is how a position's cost basis quietly drifts and
    every downstream PnL number stops reconciling.
    """
    quantity = min(quantity, state.quantity)
    if quantity >= state.quantity:
        return _close_all(state, price), ManagementAction(
            kind=ActionKind.FULL_CLOSE,
            close_quantity=state.quantity,
            reason="; ".join([*reasons, "partial would close the remainder"]),
            priority=PRIORITY_INTRABAR,
        )
    updated = replace(
        state,
        quantity=state.quantity - quantity,
        realized_pnl=state.realized_pnl + _position_pnl(state, price, quantity),
    )
    return updated, ManagementAction(
        kind=ActionKind.PARTIAL_CLOSE,
        new_stop=new_stop,
        close_quantity=quantity,
        reason="; ".join(reasons),
        priority=PRIORITY_INTRABAR,
    )


__all__ = [
    "DEFAULT_STAGES",
    "PRIORITY_EXCHANGE_STOP",
    "PRIORITY_INTRABAR",
    "PRIORITY_LOSS_ACCELERATION",
    "PRIORITY_NET_PROFIT_EXIT",
    "PRIORITY_NONE",
    "PRIORITY_RISK_FLATTEN",
    "PRIORITY_STRATEGY_EXIT",
    "PRIORITY_THESIS_INVALIDATION",
    "PRIORITY_TIME_EXIT",
    "ActionKind",
    "IntrabarConfig",
    "ManagementAction",
    "PositionState",
    "ProfitStage",
    "StageAction",
    "adverse_excursion",
    "adverse_move",
    "hard_max_loss_price",
    "is_stale",
    "net_profit_pct",
    "on_price",
    "r_multiple",
    "ratchet_stop",
    "resolve_actions",
    "trail_distance",
    "unrealized_pct",
]
