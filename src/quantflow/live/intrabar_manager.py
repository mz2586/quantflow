"""Drive the intrabar state machine from the live ticker stream.

`quantflow.position.intrabar` decides *what* should happen to a position on a given price;
it is pure and has no idea a venue exists. This module is the other half: it subscribes to
the live ticker for every held symbol, feeds each tick into that decision, and carries out
whatever comes back through the same gateway the rest of the system uses.

It runs as its own asyncio task, deliberately **independent of the candle loop**. That
independence is the entire point of the feature: the strategy layer keeps evaluating on
completed bars exactly as before, while protection reacts between them. A position that
runs favourably mid-bar no longer sits on the static stop it was opened with until the bar
closes.

There are **three** loops, and keeping them separate is the design:

* the **candle loop** (elsewhere) decides what to own;
* the **ticker loop** here reacts to price between bars;
* the **reconciliation loop** here keeps this module's idea of the book equal to the
  venue's.

The third one exists because the first version of this module adopted the venue's
positions once, at startup, and never looked again. Everything that can happen to a
position afterwards — a stop fill, a target fill, a partial, a fresh position on the same
symbol at a different price — was invisible. The observable failures were exactly the ones
you would predict: amendments retried forever against a position that had already closed
(`retCode 10001 "can not set tp/sl/ts for zero position"`), and profit stages computed
against an entry price that belonged to a trade that no longer existed. A ticker manager
that is its own source of truth is a ticker manager that is eventually wrong.

Four properties are load-bearing:

* **The venue stays authoritative.** In both directions. Local state is updated only
  *after* the exchange confirms a change we requested, and it is *overwritten* whenever
  the exchange reports something we did not expect. A stop that exists in memory and not
  at the venue is the worst of both worlds — it reports protection that would not survive
  this process dying.
* **Nothing is acted on without re-checking the venue first.** Every amendment and every
  close re-reads the position immediately beforehand. If it has gone, the symbol is
  untracked instead of amended, which is the difference between noticing a closed position
  and shouting at one twenty thousand times.
* **One action per position at a time.** An in-flight close is tracked so a second tick,
  arriving milliseconds later and seeing the same condition, cannot submit a duplicate.
* **Failure is contained.** A tick that raises is logged and skipped; the loop survives.
  Losing price management is bad, but a crashed manager takes protection down entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.logging import get_logger
from quantflow.domain.enums import OrderSide, OrderType, PositionSide, TimeInForce
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.bybit.mapping import from_ccxt_symbol
from quantflow.position.intrabar import (
    ActionKind,
    IntrabarConfig,
    ManagementAction,
    PositionState,
    ProfitStage,
    StageAction,
    on_price,
    ratchet_stop,
)

logger = get_logger(__name__)

#: How long a symbol may go without a tick before it is treated as stale. Management
#: continues on stale data — the exchange-side stop is still in force — but the operator
#: needs to see it, because a silently dead stream looks exactly like a quiet market.
STALE_AFTER = timedelta(seconds=30)

#: How often a tick is logged per symbol, after the first.
TICK_LOG_EVERY = 200

#: Default reconciliation cadence. Two seconds is chosen against the rate limit rather
#: than against the market: one ``fetch_positions`` returns the *whole* book, so the cost
#: is a single request every two seconds no matter how many symbols are traded. Polling
#: per symbol instead would multiply that by the universe size for no extra information.
DEFAULT_RECONCILE_SECONDS = 2.0

#: Floor on the cadence. A misconfigured zero would turn the loop into an unthrottled
#: request storm and get the account rate-limited out of its own protection.
MIN_RECONCILE_SECONDS = 0.1

#: Pause before re-subscribing after a ticker stream ends cleanly. A stream that returns
#: immediately would otherwise spin a CPU core; the reconnect is still effectively instant
#: from the market's point of view.
RECONNECT_DELAY_SECONDS = 1.0

#: Pause before re-subscribing after the ticker stream raised.
STREAM_ERROR_DELAY_SECONDS = 2.0

#: Minimum gap between stop-amendment attempts on a symbol after one has failed. An
#: amendment that failed will almost always fail again for the same reason, and a liquid
#: perp ticks several times a second: without a floor, one rejected amendment becomes
#: hundreds of identical rejected requests a minute.
STOP_RETRY_COOLDOWN_SECONDS = 5.0

#: How Bybit says "the stop you asked for is the stop I am already holding". Not an error:
#: this module owns no tick metadata by design, so the exchange snapping a request to its
#: own tick makes the *next* request look like a sub-tick improvement that never lands.
NOT_MODIFIED_MARKERS = ("34040", "not modified")

#: Smallest stop improvement worth a request, as a fraction of the stop price. One basis
#: point.
#:
#: This exists because of the interaction between two correct decisions. The venue snaps
#: every stop to its own tick, and this module deliberately holds no tick metadata (see the
#: module docstring in ``quantflow.position.intrabar``). Once reconciliation started pulling
#: the venue's *snapped* stop back into local state, a trail recomputed each tick produced a
#: candidate a few decimal places above it — a genuine improvement arithmetically, and no
#: change at all once the venue rounded it. Live, that was two amendment requests per second
#: against a stop that never moved.
#:
#: One basis point is below any tick a liquid perp uses relative to its own price, so a real
#: trail step always clears it, while sub-tick noise never does. It is a *floor on the
#: request*, not on the protection: the stop still ratchets on the very next tick that earns
#: a move worth making.
MIN_STOP_IMPROVEMENT_PCT = Decimal("0.0001")


# --------------------------------------------------------------------------- #
# The venue's view
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class VenuePosition:
    """One open position exactly as the exchange reports it.

    Deliberately not a :class:`PositionState`: this is what is *true*, with no water marks,
    no ladder history and no opinions. Keeping the two types distinct is what stops venue
    facts and local bookkeeping from being confused for one another at the point where the
    difference matters most.
    """

    symbol: Symbol
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    stop: Decimal | None
    target: Decimal | None


def _optional_price(value: Any) -> Decimal | None:
    """Parse a venue price field, treating absent / empty / ``"0"`` as "not set".

    Bybit reports an unset stop as the string ``"0"`` rather than omitting it, and a stop
    of zero is not a stop — it is the absence of one wearing a number.
    """
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_protection(
    symbol: Symbol, orders: Iterable[Any] | None
) -> tuple[Decimal | None, Decimal | None]:
    """A position's stop and target, read from the orders that actually protect it.

    Bybit keeps protection in one of two places. Under ``tpslMode: Full`` it sits on the
    position row. Under ``Partial`` — which is what a maker take-profit requires, and
    therefore what this engine uses — it is a pair of separate reduce-only trigger orders
    and the position row's own fields stay empty.

    Reading only the row is why a live ETH position with a venue stop at 1893.45 and a
    target at 1939.47 was reported as having neither, and was refused adoption with
    ``intrabar.adopt_skipped_no_stop``. The position was protected the whole time; the
    manager simply could not see it, so it declined to ratchet a winner it believed was
    naked.

    Which order is which is decided by its TYPE, not by where its trigger sits relative to
    entry. A take-profit is a limit order and carries a price; a stop is a market order and
    does not. That distinction is stable; position relative to entry is not.

    Deciding it by position was a real defect, live for 78 minutes on 2026-08-17. Once the
    trail ratchets a stop into profit — which is precisely what happens when a winner is
    running — that stop sits *above* entry on a long and was misread as the take-profit.
    A BTC position entered at 64,301.40 with a genuine target of 65,274.20 had its stop
    ratcheted to 64,378.50 at 18:23:21 and was closed two seconds later for "reaching
    target 64,378.5". The trail had been turned into a premature exit, cutting the winner
    at a fifth of its intended move.
    """
    stop = target = None
    for order in orders or []:
        if not getattr(order, "reduce_only", False):
            continue
        trigger = getattr(order, "trigger_price", None)
        if trigger is None:
            continue
        if str(getattr(order, "symbol", "")).split(":")[0] != str(symbol).split(":")[0]:
            continue
        if getattr(order, "price", None):
            target = trigger
        else:
            stop = trigger
    return stop, target


def parse_venue_positions(
    rows: Iterable[Mapping[str, Any]], protective_orders: Iterable[Any] | None = None
) -> dict[Symbol, VenuePosition]:
    """Turn a raw ``fetch_positions`` payload into the open book, keyed by symbol.

    Flat rows are dropped rather than represented: the venue reports a closed position as a
    row with zero contracts, and carrying that through as an entry would mean every caller
    has to remember to check. "Absent from the book" is the single, unambiguous way this
    module says *there is no position here*.

    ``protective_orders`` supplies the resting reduce-only orders, consulted when the
    position row carries no stop or target of its own. Without them a position protected in
    ``Partial`` mode reads as unprotected.
    """
    protective = list(protective_orders) if protective_orders is not None else None
    book: dict[Symbol, VenuePosition] = {}
    for raw in rows:
        try:
            quantity = Decimal(str(raw.get("contracts") or 0))
            entry = Decimal(str(raw.get("entryPrice") or 0))
        except (ArithmeticError, ValueError):
            logger.warning("intrabar.venue_row_unparsed", symbol=str(raw.get("symbol"))[:40])
            continue
        if quantity == 0 or entry <= 0:
            continue
        try:
            # The venue speaks CCXT's unified form ("ETH/USDT:USDT"); everything above this
            # boundary uses the plain pair. Converting here keeps that translation in the
            # one place that touches raw venue payloads.
            symbol = from_ccxt_symbol(str(raw.get("symbol")))
        except Exception:
            logger.warning("intrabar.venue_symbol_unparsed", symbol=str(raw.get("symbol"))[:40])
            continue
        info = raw.get("info") or {}
        side = (
            PositionSide.LONG
            if str(raw.get("side") or "").lower() == "long"
            else PositionSide.SHORT
        )
        stop = _optional_price(info.get("stopLoss"))
        target = _optional_price(info.get("takeProfit"))
        if stop is None or target is None:
            # Nothing on the row: look at what is actually resting against this position.
            resting_stop, resting_target = resolve_protection(symbol, protective)
            stop = stop if stop is not None else resting_stop
            target = target if target is not None else resting_target
        book[symbol] = VenuePosition(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            entry_price=entry,
            stop=stop,
            target=target,
        )
    return book


def improves_materially(
    side: PositionSide,
    current: Decimal,
    candidate: Decimal,
    *,
    min_pct: Decimal = MIN_STOP_IMPROVEMENT_PCT,
) -> bool:
    """Whether moving the stop from ``current`` to ``candidate`` is worth a request.

    Two ways to answer no, and they fail for different reasons:

    * ``candidate`` is not protective — the ratchet would reject it anyway;
    * ``candidate`` is protective but by less than :data:`MIN_STOP_IMPROVEMENT_PCT`, which
      on any real instrument is inside the venue's own tick. Sending it produces a stop
      identical to the one already resting there, and doing that on every tick is a request
      storm dressed as a trailing stop.
    """
    if ratchet_stop(side, current, candidate) != candidate:
        return False
    return abs(candidate - current) >= abs(current) * min_pct


def state_from_venue(
    position: VenuePosition, now: datetime, *, invalidation_price: Decimal | None = None
) -> PositionState | None:
    """Build fresh management state from a venue position, or ``None`` if it has no stop.

    Water marks start at the **entry price**, never at the current one. The favourable
    excursion that happened before this position came into view cannot be reconstructed,
    and inventing one would hand the trail a peak the position never actually reached.

    That seeding costs nothing in responsiveness: the profit ladder is evaluated against
    the *live tick price*, not against a water mark, so a position adopted while already
    past a rung fires that rung on its very first tick.

    ``invalidation_price`` is passed through untouched and is **never** derived from the
    venue payload. The exchange knows a stop and a take-profit; it has no idea what the
    strategy believed, so a thesis level can only come from a caller that does. Left
    ``None``, the invalidation rule stays inactive for this position.
    """
    if position.stop is None:
        # A position with no venue stop is not one this layer should start ratcheting:
        # there is nothing to move, and inventing a stop would be a risk decision the risk
        # engine never made.
        return None
    return PositionState.from_entry(
        symbol=position.symbol.slashed,
        side=position.side,
        entry_price=position.entry_price,
        quantity=position.quantity,
        stop=position.stop,
        opened_at=now,
        target=position.target,
        invalidation_price=invalidation_price,
    )


class IntrabarManager:
    """Runs the intrabar decision against a live ticker stream, one task per symbol."""

    __slots__ = (
        "_atr",
        "_clock",
        "_closing",
        "_config",
        "_gateway",
        "_invalidation",
        "_reconcile_seconds",
        "_reconcile_task",
        "_reconcile_wanted",
        "_states",
        "_stop_retry_after",
        "_stream",
        "_tasks",
        "_ticks",
        "_venue",
    )

    def __init__(
        self,
        gateway: Any,
        stream: Any,
        config: IntrabarConfig,
        *,
        reconcile_seconds: float = DEFAULT_RECONCILE_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._gateway = gateway
        self._stream = stream
        self._config = config
        self._states: dict[Symbol, PositionState] = {}
        #: Symbols with a close already submitted, mapped to the venue position the close
        #: was aimed at. A burst of ticks therefore cannot produce a burst of close orders,
        #: while a *different* position appearing on the same symbol — our close landed and
        #: the strategy re-entered — is still recognised as something new to manage.
        self._closing: dict[Symbol, VenuePosition] = {}
        self._atr: dict[Symbol, Decimal] = {}
        #: Thesis invalidation levels supplied from outside, by symbol. Survives adoption
        #: and re-adoption so a reconnect does not quietly drop the one input this module
        #: cannot reconstruct: the venue reports stops and targets, never a thesis.
        self._invalidation: dict[Symbol, Decimal] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._ticks: dict[Symbol, int] = {}
        self._reconcile_seconds = max(MIN_RECONCILE_SECONDS, float(reconcile_seconds))
        self._reconcile_task: asyncio.Task[None] | None = None
        #: Set to ask for a reconciliation before the next scheduled one.
        self._reconcile_wanted = asyncio.Event()
        #: The last book the venue reported. Diagnostics only — never a decision input.
        self._venue: dict[Symbol, VenuePosition] = {}
        #: Monotonic deadlines: a symbol whose stop amendment failed is not retried before
        #: this, so one rejection cannot become a per-tick request storm.
        self._stop_retry_after: dict[Symbol, float] = {}
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def enabled(self) -> bool:
        """Whether management is switched on at all."""
        return self._config.enabled

    @property
    def monitored(self) -> tuple[Symbol, ...]:
        """Symbols currently under intrabar management."""
        return tuple(self._states)

    @property
    def reconcile_seconds(self) -> float:
        """The reconciliation cadence actually in force."""
        return self._reconcile_seconds

    @property
    def venue_view(self) -> dict[Symbol, VenuePosition]:
        """The most recent venue book this manager saw."""
        return dict(self._venue)

    def state_for(self, symbol: Symbol) -> PositionState | None:
        """The tracked state for one symbol, if any."""
        return self._states.get(symbol)

    def track(self, state: PositionState) -> None:
        """Begin (or resume) managing a position."""
        symbol = Symbol.parse(state.symbol)
        assert isinstance(symbol, Symbol)
        self._states[symbol] = state
        self._closing.pop(symbol, None)
        if state.invalidation_price is not None:
            self._invalidation[symbol] = state.invalidation_price

    def set_atr(self, symbol: Symbol, atr: Decimal) -> None:
        """Supply the volatility used for trailing. Fed from the candle loop."""
        self._atr[symbol] = atr

    def set_invalidation(self, symbol: Symbol, price: Decimal | None) -> None:
        """Supply (or clear) the price at which the strategy's thesis is dead.

        The only way a thesis reaches this layer. Nothing is inferred: no level supplied
        means the invalidation rule is inactive for that symbol, which is the honest
        outcome when the strategy that opened the position never published one. Passing a
        level here applies it to the tracked state immediately and to every later adoption
        of the same symbol, so a reconnect does not silently drop it.
        """
        if price is None:
            self._invalidation.pop(symbol, None)
        else:
            self._invalidation[symbol] = price
        tracked = self._states.get(symbol)
        if tracked is not None:
            self._states[symbol] = replace(tracked, invalidation_price=price)

    def request_reconcile(self) -> None:
        """Ask for a reconciliation as soon as the loop can run one.

        Called after anything that could have changed the book — an order this module
        submitted, a stream reconnect — so the 2s cadence is a *ceiling* on how long a
        stale view can survive rather than the normal latency of noticing a change.
        """
        self._reconcile_wanted.set()

    async def start(self, symbols: list[Symbol]) -> None:
        """Reconcile against the venue, then launch the ticker and reconciliation loops."""
        if not self._config.enabled:
            logger.info("intrabar.disabled")
            return
        # Before the first tick, not after: a tick that arrives against an empty book is a
        # tick that does nothing, and the position it belonged to stays unmanaged until the
        # next poll.
        await self.reconcile()
        for symbol in symbols:
            self._tasks.append(asyncio.create_task(self._watch(symbol)))
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        logger.critical(
            "intrabar.started",
            symbols=[str(s) for s in symbols],
            stages=len(self._config.stages),
            trail_atr_multiple=str(self._config.trail_atr_multiple),
            reconcile_seconds=self._reconcile_seconds,
            adopted=[str(s) for s in self._states],
            # Logged in full because "is the net-profit exit actually on in the process
            # that is trading?" must be answerable from the log of the running engine, not
            # from the source of the version somebody believes is deployed.
            net_profit_exit=self._config.net_profit_exit_enabled,
            min_net_profit_pct=str(self._config.min_net_profit_pct),
            round_trip_cost_pct=str(self._config.round_trip_cost_pct),
            hard_max_loss=self._config.hard_max_loss_enabled,
            hard_max_loss_r=str(self._config.hard_max_loss_r),
            invalidation_exit=self._config.invalidation_exit_enabled,
            loss_accel=self._config.loss_accel_enabled,
            loss_accel_stop_fraction=str(self._config.loss_accel_stop_fraction),
            loss_accel_atr_multiple=str(self._config.loss_accel_atr_multiple),
            stale_loser=self._config.stale_loser_enabled,
            stale_loser_after_s=self._config.stale_loser_after.total_seconds(),
        )

    async def stop(self) -> None:
        """Cancel every task. Positions keep their exchange-side protection."""
        tasks = [*self._tasks]
        if self._reconcile_task is not None:
            tasks.append(self._reconcile_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        self._reconcile_task = None

    # ------------------------------------------------------------------ #
    # Loop 3: reconciliation
    # ------------------------------------------------------------------ #
    async def _reconcile_loop(self) -> None:
        """Poll the venue on a cadence, or sooner when something asks."""
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._reconcile_wanted.wait(), timeout=self._reconcile_seconds
                )
            self._reconcile_wanted.clear()
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A reconciliation that raises must not take protection down with it: the
                # ticker loop keeps working off the last known-good view, and the next pass
                # is two seconds away.
                logger.warning("intrabar.reconcile_failed", error=str(exc)[:200])

    async def reconcile(self) -> None:
        """Make local state equal the venue's. One request covers every symbol."""
        try:
            rows = await self._gateway.fetch_positions()
        except Exception as exc:
            logger.warning("intrabar.reconcile_fetch_failed", error=str(exc)[:200])
            return
        orders = await self._resting_protection()
        self._apply(parse_venue_positions(rows or [], orders))

    def _apply(self, venue: dict[Symbol, VenuePosition]) -> None:
        """Fold a venue snapshot into local state. The venue wins every disagreement."""
        self._venue = venue

        # Gone from the venue means gone, full stop. This is the branch whose absence
        # produced twenty thousand "can not set tp/sl/ts for zero position" rejections:
        # the position had closed and the manager had no way to find out.
        for symbol in [s for s in self._states if s not in venue]:
            self._untrack(symbol, reason="closed_at_venue")
        for symbol in [s for s in self._closing if s not in venue]:
            # The close we submitted has landed; the symbol is free to be managed again if
            # the strategy opens a new position on it.
            del self._closing[symbol]

        now = self._clock()
        for symbol, position in venue.items():
            in_flight = self._closing.get(symbol)
            if in_flight is not None:
                if (
                    in_flight.side is position.side
                    and in_flight.entry_price == position.entry_price
                ):
                    # Our close has not shown up at the venue yet. Re-adopting here would
                    # hand the next tick a position we have already decided to exit, and the
                    # second close order would be a real, fee-paying reversal risk.
                    continue
                # A different side or entry means this is not the position we closed: the
                # close landed and something new was opened on the same symbol.
                del self._closing[symbol]
            tracked = self._states.get(symbol)
            if tracked is None:
                self._adopt(position, now)
            else:
                self._resync(tracked, position, now)

    def _untrack(self, symbol: Symbol, *, reason: str) -> None:
        """Forget a symbol entirely — state, water marks, ladder and in-flight close."""
        state = self._states.pop(symbol, None)
        self._closing.pop(symbol, None)
        self._invalidation.pop(symbol, None)
        if state is not None:
            logger.critical(
                "intrabar.untracked",
                symbol=str(symbol),
                reason=reason,
                entry=str(state.entry_price),
                quantity=str(state.quantity),
            )

    async def _resting_protection(self) -> list[Any] | None:
        """Resting orders, for resolving protection the position row does not carry.

        Never raises: a failed order read must leave management as it was, not take the
        loop down. ``None`` means "not known", which is deliberately distinct from an empty
        list — an empty list would assert the position is unprotected.
        """
        fetch = getattr(self._gateway, "fetch_open_orders", None)
        if fetch is None:
            return None
        try:
            return list(await fetch())
        except Exception as exc:
            logger.warning("intrabar.protection_read_failed", error=str(exc)[:160])
            return None

    def _adopt(self, position: VenuePosition, now: datetime) -> None:
        """Start managing a position the venue holds and this manager does not."""
        state = state_from_venue(
            position, now, invalidation_price=self._invalidation.get(position.symbol)
        )
        if state is None:
            logger.warning("intrabar.adopt_skipped_no_stop", symbol=str(position.symbol))
            return
        self._states[position.symbol] = state
        logger.critical(
            "intrabar.adopted",
            symbol=str(position.symbol),
            side=position.side.value,
            entry=str(position.entry_price),
            quantity=str(position.quantity),
            stop=str(position.stop),
            target=None if position.target is None else str(position.target),
        )

    def _resync(self, tracked: PositionState, position: VenuePosition, now: datetime) -> None:
        """Reconcile one tracked position against the venue's version of it."""
        symbol = position.symbol
        if tracked.side is not position.side or tracked.entry_price != position.entry_price:
            # A different side or a different average entry is a **different position**,
            # whatever the symbol says. Nothing about the old one may survive: its water
            # marks describe an excursion this position never made, and its fired stages
            # describe profit this position never earned.
            # A different position on the same symbol keeps no thesis: the level that was
            # supplied described the trade that has just gone, and re-using it would apply
            # one strategy's invalidation to another strategy's entry.
            self._invalidation.pop(symbol, None)
            replacement = state_from_venue(position, now)
            if replacement is None:
                self._untrack(symbol, reason="replaced_without_venue_stop")
                return
            self._states[symbol] = replacement
            self._closing.pop(symbol, None)
            logger.critical(
                "intrabar.replaced",
                symbol=str(symbol),
                old_entry=str(tracked.entry_price),
                new_entry=str(position.entry_price),
                old_quantity=str(tracked.quantity),
                new_quantity=str(position.quantity),
                side=position.side.value,
            )
            return

        updated = tracked
        changes: list[str] = []
        if position.quantity != tracked.quantity:
            # A partial fill, or a scale-in at the same average price. ``original_quantity``
            # only ever grows, because the partial-exit rung is a fraction of the size the
            # position started at and must not shrink underneath itself.
            updated = replace(
                updated,
                quantity=position.quantity,
                original_quantity=max(tracked.original_quantity, position.quantity),
            )
            changes.append(f"quantity {tracked.quantity}->{position.quantity}")
        if position.stop is not None and position.stop != tracked.current_stop:
            # Including when the venue stop is *worse* than the one held locally. Believing
            # in a stop the exchange is not holding is precisely the failure this module
            # claims to prevent.
            updated = replace(updated, current_stop=position.stop)
            changes.append(f"stop {tracked.current_stop}->{position.stop}")
        if position.target != tracked.target:
            updated = replace(updated, target=position.target)
            changes.append(f"target {tracked.target}->{position.target}")

        if changes:
            self._states[symbol] = updated
            logger.info("intrabar.resynced", symbol=str(symbol), changes="; ".join(changes))

    # ------------------------------------------------------------------ #
    # Loop 2: the ticker
    # ------------------------------------------------------------------ #
    async def _watch(self, symbol: Symbol) -> None:
        """Consume the ticker stream for one symbol until cancelled.

        Reconnects are handled by looping: a dropped stream is retried rather than left
        dead, because an unmanaged position is precisely the state this exists to prevent.
        Every reconnect asks for a reconciliation, because a gap in the price feed is
        exactly the window in which a stop or target fills unseen.
        """
        while True:
            try:
                async for ticker in self._stream.watch_ticker(symbol):
                    await self._on_tick(symbol, ticker.last, ticker.timestamp)
                self.request_reconcile()
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("intrabar.stream_error", symbol=str(symbol), error=str(exc)[:200])
                self.request_reconcile()
                await asyncio.sleep(STREAM_ERROR_DELAY_SECONDS)

    async def _on_tick(self, symbol: Symbol, price: Decimal, now: Any) -> None:
        """One price update: decide, then act."""
        seen = self._ticks.get(symbol, 0) + 1
        self._ticks[symbol] = seen
        # A manager that takes no action looks identical whether ticks are flowing or the
        # stream is dead. The first tick and every TICK_LOG_EVERY after it are logged so
        # the difference is visible without turning on debug logging.
        if seen == 1 or seen % TICK_LOG_EVERY == 0:
            logger.info(
                "intrabar.tick",
                symbol=str(symbol),
                price=str(price),
                ticks=seen,
                managed=symbol in self._states,
            )

        state = self._states.get(symbol)
        if state is None or symbol in self._closing:
            return
        try:
            updated, action = on_price(
                state,
                price,
                atr=self._atr.get(symbol),
                config=self._config,
                now=now,
            )
        except Exception as exc:
            logger.warning("intrabar.decide_failed", symbol=str(symbol), error=str(exc)[:200])
            return

        if action.kind is ActionKind.NONE:
            if self._states.get(symbol) is state:
                self._states[symbol] = updated
            return

        if (
            action.kind is ActionKind.MOVE_STOP
            and action.new_stop is not None
            and not improves_materially(state.side, state.current_stop, action.new_stop)
        ):
            # Sub-tick noise. ``state.current_stop`` is the venue's own value — reconciliation
            # keeps it that way — so the honest record is that the stop stays where the venue
            # has it, while any rung that fired on this tick is still marked done and never
            # asks again. Checked before the pre-action venue read so a churning trail costs
            # no requests at all.
            logger.debug(
                "intrabar.stop_change_below_tick",
                symbol=str(symbol),
                venue_stop=str(state.current_stop),
                requested=str(action.new_stop),
            )
            self._states[symbol] = replace(updated, current_stop=state.current_stop)
            return

        if action.kind is ActionKind.MOVE_STOP and self._amendment_cooling_off(symbol):
            # An amendment that just failed will keep failing for the same reason, and a
            # 15m position ticks several times a second. Without a floor between attempts
            # one rejected amendment becomes hundreds of identical rejected requests.
            return

        # The venue is authoritative: local state advances only once the change is
        # confirmed, so a rejected amendment cannot leave us believing we are protected.
        #
        # The PRE-tick state is what gets executed against: a full close zeroes the
        # quantity in the updated state, so sizing the order from it would submit zero
        # contracts and silently leave the position open.
        outcome = await self._execute(symbol, state, updated, action)
        if outcome is not None and symbol in self._states:
            # ``symbol in self._states`` guards the case where executing untracked it — a
            # full close, or a position that turned out to have vanished. Re-inserting the
            # computed state there would resurrect a position that no longer exists.
            self._states[symbol] = outcome

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    async def _live_position(self, symbol: Symbol) -> VenuePosition | None:
        """Re-read the venue for one symbol immediately before acting on it.

        The book moves between the tick that produced a decision and the request that
        carries it out — that gap is where a stop fill lives. Paying one request to look
        again is the difference between noticing and being told twenty thousand times.
        """
        rows = await self._gateway.fetch_positions()
        orders = await self._resting_protection()
        self._venue = parse_venue_positions(rows or [], orders)
        return self._venue.get(symbol)

    def _amendment_cooling_off(self, symbol: Symbol) -> bool:
        """Whether a failed stop amendment on this symbol is still inside its retry floor."""
        until = self._stop_retry_after.get(symbol)
        return until is not None and time.monotonic() < until

    async def _amend_stop(
        self, symbol: Symbol, state: PositionState, requested: Decimal
    ) -> Decimal | None:
        """Amend the venue stop and return the level the **venue** ends up holding.

        Three outcomes, and the difference between them is the difference between a request
        storm and a working ratchet:

        * the venue confirms a level — use it, exactly as the venue spells it;
        * the venue refuses to modify a stop it already holds (``retCode 34040``) — that is
          the strongest possible confirmation, not a failure. It happens because the
          exchange snaps the requested price to its tick and this module deliberately owns
          no tick metadata, so a sub-tick improvement asks for a change the venue considers
          it has already made. Treating it as an error is what turned one rung into 504
          identical rejected requests in five minutes;
        * the venue accepts but its read-back has not caught up — look again rather than
          either believing an unconfirmed stop or discarding a real one.

        Returns ``None`` only when the venue cannot be shown to hold a better stop than
        before, which is the one case where local state must not advance.
        """
        try:
            confirmed = await self._gateway.set_trading_stop(symbol, stop_loss=requested)
        except Exception as exc:
            if not any(marker in str(exc).lower() for marker in NOT_MODIFIED_MARKERS):
                raise
            live = await self._live_position(symbol)
            venue_stop = None if live is None else live.stop
            logger.info(
                "intrabar.stop_already_at_venue",
                symbol=str(symbol),
                requested=str(requested),
                venue_stop=None if venue_stop is None else str(venue_stop),
            )
            return venue_stop
        if confirmed is not None:
            # The gateway is typed loosely at this boundary; normalise so a venue that
            # answers with a string can never end up compared against a Decimal.
            return Decimal(str(confirmed))
        live = await self._live_position(symbol)
        if live is None or live.stop is None:
            return None
        # Accept the re-read only if the venue is now holding something strictly more
        # protective than before. Anything else and the amendment cannot be shown to have
        # happened, which is exactly when believing it would be dangerous.
        moved = ratchet_stop(state.side, state.current_stop, live.stop)
        if moved == live.stop and live.stop != state.current_stop:
            return live.stop
        return None

    async def _execute(  # noqa: PLR0911 - one return per refusal, each with its own reason.
        # Collapsing them into a single exit would hide *which* guard declined to act, and
        # "the amendment was not sent" is only useful with the reason attached.
        self,
        symbol: Symbol,
        state: PositionState,
        updated: PositionState,
        action: ManagementAction,
    ) -> PositionState | None:
        """Carry out one action against the venue.

        Returns the state to persist, or ``None`` if nothing may advance. The returned state
        carries the stop **the venue reports**, not the one that was asked for: the exchange
        snaps prices to its own tick, and storing the unsnapped request is what makes the
        next tick believe the ratchet still has work to do.
        """
        try:
            live = await self._live_position(symbol)
            if live is None:
                # It closed between the tick and now. Untrack rather than amend: there is
                # no position to protect, and re-posting a stop against a flat symbol is
                # both rejected and meaningless.
                self._untrack(symbol, reason="vanished_before_action")
                return None
            if live.side is not state.side or live.entry_price != state.entry_price:
                # Not the position this decision was made about. Abort and let
                # reconciliation rebuild state from what is actually there.
                logger.warning(
                    "intrabar.action_aborted_position_changed",
                    symbol=str(symbol),
                    kind=action.kind.value,
                    decided_entry=str(state.entry_price),
                    venue_entry=str(live.entry_price),
                )
                self.request_reconcile()
                return None

            if action.kind is ActionKind.MOVE_STOP and action.new_stop is not None:
                confirmed = await self._amend_stop(symbol, state, action.new_stop)
                if confirmed is None:
                    logger.warning("intrabar.stop_not_confirmed", symbol=str(symbol))
                    self._stop_retry_after[symbol] = time.monotonic() + STOP_RETRY_COOLDOWN_SECONDS
                    return None
                self._stop_retry_after.pop(symbol, None)
                logger.critical(
                    "intrabar.stop_moved",
                    symbol=str(symbol),
                    new_stop=str(action.new_stop),
                    venue_stop=str(confirmed),
                    reason=action.reason,
                )
                self.request_reconcile()
                return replace(updated, current_stop=confirmed)

            if action.kind in (ActionKind.PARTIAL_CLOSE, ActionKind.FULL_CLOSE):
                # Size against what the venue is actually holding. A partial fill since the
                # decision means the intended quantity no longer exists, and an order for
                # more than is held is either rejected or — without reduce-only — a
                # reversal.
                quantity = min(action.close_quantity or state.quantity, live.quantity)
                if quantity <= 0:
                    logger.warning("intrabar.close_skipped_zero_quantity", symbol=str(symbol))
                    self.request_reconcile()
                    return None
                if action.kind is ActionKind.FULL_CLOSE:
                    self._closing[symbol] = live
                await self._gateway.submit_order(
                    OrderRequest(
                        symbol=symbol,
                        side=(OrderSide.SELL if state.side is PositionSide.LONG else OrderSide.BUY),
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        time_in_force=TimeInForce.GTC,
                        reduce_only=True,
                        metadata={"reason": action.reason[:200], "source": "intrabar"},
                    )
                )
                logger.critical(
                    "intrabar.closed",
                    symbol=str(symbol),
                    kind=action.kind.value,
                    quantity=str(quantity),
                    reason=action.reason,
                )
                if action.kind is ActionKind.FULL_CLOSE:
                    self._states.pop(symbol, None)
                self.request_reconcile()
                return updated
        except Exception as exc:
            self._closing.pop(symbol, None)
            logger.warning(
                "intrabar.execute_failed",
                symbol=str(symbol),
                kind=action.kind.value,
                error=str(exc)[:200],
            )
            if action.kind is ActionKind.MOVE_STOP:
                self._stop_retry_after[symbol] = time.monotonic() + STOP_RETRY_COOLDOWN_SECONDS
            self.request_reconcile()
            return None
        return None


def reconcile_seconds_from_env() -> float:
    """Read ``QF_INTRABAR_RECONCILE_SECONDS``, defaulting to :data:`DEFAULT_RECONCILE_SECONDS`.

    Clamped at :data:`MIN_RECONCILE_SECONDS`: a typo that set this to zero would replace a
    poll with a request storm, and the first thing the venue would do about it is rate-limit
    the account out of the protection this loop exists to provide.
    """
    raw = os.environ.get("QF_INTRABAR_RECONCILE_SECONDS", "").strip()
    if not raw:
        return DEFAULT_RECONCILE_SECONDS
    try:
        return max(MIN_RECONCILE_SECONDS, float(raw))
    except ValueError:
        logger.warning(
            "intrabar.bad_config", variable="QF_INTRABAR_RECONCILE_SECONDS", value=raw[:40]
        )
        return DEFAULT_RECONCILE_SECONDS


def intrabar_config_from_env() -> IntrabarConfig:
    """Build the config from the environment, defaulting to OFF.

    Off unless explicitly asked for: this changes how every position exits, and a feature
    that alters exit behaviour should never arrive by inheriting a config written before
    it existed.

    Every threshold is overridable so they can be tuned without a code change:

    * ``QF_INTRABAR_STAGE1_PCT`` / ``QF_INTRABAR_STAGE2_PCT`` / ``QF_INTRABAR_STAGE3_PCT``
    * ``QF_INTRABAR_LOCK_PCT`` — profit locked at stage 2
    * ``QF_INTRABAR_PARTIAL`` — fraction closed at stage 3
    * ``QF_INTRABAR_TRAIL_ATR`` / ``QF_INTRABAR_MIN_TRAIL_PCT``

    Raising stage 1 is the lever that matters most: it is the point at which a position
    stops being allowed to become a loser, and setting it too low converts ordinary noise
    into a stopped-out breakeven trade.

    The **net-profit exit** and its cost model:

    * ``QF_NET_PROFIT_EXIT`` — on unless set to ``false``. It defaults on *here* while the
      dataclass default stays off, which is the same split :attr:`IntrabarConfig.enabled`
      uses: the running engine gets the behaviour without an env file, and no library
      caller inherits it by writing ``IntrabarConfig(enabled=True)``.
    * ``QF_MIN_NET_PROFIT_PCT`` — buffer over costs, default 0.0005 (0.05%)
    * ``QF_ENTRY_FEE_PCT`` / ``QF_EXIT_FEE_PCT`` — default 0.0006 each (Bybit taker),
      summing to the 0.0012 round trip ``fee_rate`` already assumes
    * ``QF_SPREAD_PCT`` / ``QF_SLIPPAGE_PCT`` — default 0.0002 each

    The **loser rules**, all derived per position from its own stop distance and ATR:

    * ``QF_HARD_MAX_LOSS`` / ``QF_HARD_MAX_LOSS_R`` — default on at 1.0R, i.e. exactly the
      stop the risk engine sized the trade on
    * ``QF_INVALIDATION_EXIT`` — default on, but inert unless a level was supplied
    * ``QF_LOSS_ACCEL`` / ``QF_LOSS_ACCEL_STOP_FRACTION`` / ``QF_LOSS_ACCEL_ATR`` /
      ``QF_LOSS_ACCEL_BURST_ATR`` / ``QF_LOSS_ACCEL_WINDOW_S``
    * ``QF_STALE_LOSER`` / ``QF_STALE_LOSER_AFTER_S`` — default on at 3600s

    Every value falls back to its default with a warning rather than raising: a typo in an
    environment variable must not be able to take position protection offline.
    """

    def _decimal(name: str, default: Decimal) -> Decimal:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return Decimal(raw)
        except (ArithmeticError, ValueError):
            logger.warning("intrabar.bad_config", variable=name, value=raw[:40])
            return default

    def _flag(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        if raw in ("true", "1", "yes", "on"):
            return True
        if raw in ("false", "0", "no", "off"):
            return False
        logger.warning("intrabar.bad_config", variable=name, value=raw[:40])
        return default

    def _seconds(name: str, default: timedelta) -> timedelta:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            seconds = float(raw)
        except ValueError:
            logger.warning("intrabar.bad_config", variable=name, value=raw[:40])
            return default
        if seconds <= 0:
            logger.warning("intrabar.bad_config", variable=name, value=raw[:40])
            return default
        return timedelta(seconds=seconds)

    stages = (
        ProfitStage(
            trigger_pct=_decimal("QF_INTRABAR_STAGE1_PCT", Decimal("0.0025")),
            action=StageAction.BREAKEVEN,
        ),
        ProfitStage(
            trigger_pct=_decimal("QF_INTRABAR_STAGE2_PCT", Decimal("0.0050")),
            action=StageAction.LOCK_PROFIT,
            lock_pct=_decimal("QF_INTRABAR_LOCK_PCT", Decimal("0.0020")),
        ),
        ProfitStage(
            trigger_pct=_decimal("QF_INTRABAR_STAGE3_PCT", Decimal("0.0075")),
            action=StageAction.PARTIAL_EXIT,
            partial_fraction=_decimal("QF_INTRABAR_PARTIAL", Decimal("0.33")),
        ),
    )
    return IntrabarConfig(
        stages=stages,
        trail_atr_multiple=_decimal("QF_INTRABAR_TRAIL_ATR", Decimal("2")),
        min_trail_pct=_decimal("QF_INTRABAR_MIN_TRAIL_PCT", Decimal("0.002")),
        entry_fee_pct=_decimal("QF_ENTRY_FEE_PCT", Decimal("0.0006")),
        exit_fee_pct=_decimal("QF_EXIT_FEE_PCT", Decimal("0.0006")),
        spread_pct=_decimal("QF_SPREAD_PCT", Decimal("0.0002")),
        slippage_pct=_decimal("QF_SLIPPAGE_PCT", Decimal("0.0002")),
        min_net_profit_pct=_decimal("QF_MIN_NET_PROFIT_PCT", Decimal("0.0005")),
        net_profit_exit_enabled=_flag("QF_NET_PROFIT_EXIT", default=True),
        hard_max_loss_enabled=_flag("QF_HARD_MAX_LOSS", default=True),
        hard_max_loss_r=_decimal("QF_HARD_MAX_LOSS_R", Decimal("1")),
        invalidation_exit_enabled=_flag("QF_INVALIDATION_EXIT", default=True),
        loss_accel_enabled=_flag("QF_LOSS_ACCEL", default=True),
        loss_accel_stop_fraction=_decimal("QF_LOSS_ACCEL_STOP_FRACTION", Decimal("0.6")),
        loss_accel_atr_multiple=_decimal("QF_LOSS_ACCEL_ATR", Decimal("1.5")),
        loss_accel_burst_atr_multiple=_decimal("QF_LOSS_ACCEL_BURST_ATR", Decimal("2")),
        loss_accel_window=_seconds("QF_LOSS_ACCEL_WINDOW_S", timedelta(seconds=60)),
        stale_loser_enabled=_flag("QF_STALE_LOSER", default=True),
        stale_loser_after=_seconds("QF_STALE_LOSER_AFTER_S", timedelta(hours=1)),
        enabled=os.environ.get("QF_INTRABAR", "false").strip().lower() == "true",
    )


async def adopt_open_positions(gateway: Any, now: Any) -> list[PositionState]:
    """Build management state for positions the venue already holds.

    Kept as the one-shot form of what :meth:`IntrabarManager.reconcile` does continuously —
    useful anywhere a snapshot is wanted without a manager. The venue is the source of truth
    for what is open, so the state is rebuilt from it rather than from anything this process
    remembers.
    """
    adopted: list[PositionState] = []
    fetch_orders = getattr(gateway, "fetch_open_orders", None)
    orders = await fetch_orders() if fetch_orders is not None else None
    positions = parse_venue_positions(await gateway.fetch_positions() or [], orders)
    for position in positions.values():
        state = state_from_venue(position, now)
        if state is None:
            logger.warning("intrabar.adopt_skipped_no_stop", symbol=str(position.symbol))
            continue
        adopted.append(state)
    return adopted


__all__ = [
    "DEFAULT_RECONCILE_SECONDS",
    "MIN_RECONCILE_SECONDS",
    "MIN_STOP_IMPROVEMENT_PCT",
    "NOT_MODIFIED_MARKERS",
    "STALE_AFTER",
    "STOP_RETRY_COOLDOWN_SECONDS",
    "IntrabarManager",
    "VenuePosition",
    "adopt_open_positions",
    "improves_materially",
    "intrabar_config_from_env",
    "parse_venue_positions",
    "reconcile_seconds_from_env",
    "state_from_venue",
]
