"""Venue reconciliation.

Local state and the exchange drift: a missed fill, a manual close, a restart, a partial.
Whenever they disagree the venue is right — it holds the money. Anything the local book
believes is a claim, and a claim that contradicts the exchange is simply wrong.

Used at startup, before any signal is acted on. A position the system does not know it holds
cannot be sized against, cannot be stopped out by logic that never sees it, and will not
appear in any equity figure — so trading *around* it is trading blind.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill, Order, can_transition
from quantflow.domain.positions import ClosedTrade, Position
from quantflow.portfolio.manager import PortfolioManager

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """A position as the exchange reports it."""

    symbol: Symbol
    side: str
    quantity: Decimal
    entry_price: Decimal
    stop_loss_price: Decimal | None
    #: Leverage the VENUE reports for this position. Never assumed: if Bybit holds the
    #: symbol at 10x it reserves a tenth of the margin the bot thinks it has, and every
    #: equity-derived limit would then be measured against a reservation that is not real.
    leverage: Decimal = ONE
    #: Initial margin the venue reports it has reserved, where available.
    venue_margin: Decimal | None = None
    #: Take-profit the venue is holding, where it reports one.
    take_profit_price: Decimal | None = None

    @property
    def is_protected(self) -> bool:
        """Whether the venue is holding a stop for this position."""
        return self.stop_loss_price is not None and self.stop_loss_price > ZERO

    @property
    def position_side(self) -> PositionSide:
        """Direction, as the rest of the system spells it.

        Bybit says ``Buy``/``Sell`` in the raw payload and CCXT says ``long``/``short``;
        both reach this type through :func:`parse_venue_positions`, so both are accepted.
        """
        return PositionSide.SHORT if self.side in ("sell", "short") else PositionSide.LONG

    @property
    def signed_quantity(self) -> Decimal:
        """Quantity signed the way a local :class:`Position` signs it."""
        return -self.quantity if self.position_side is PositionSide.SHORT else self.quantity

    @property
    def margin_required(self) -> Decimal:
        """Margin reserved against this position.

        Prefers the venue's own figure. Falls back to notional / venue-reported leverage -
        still the venue's number, never the bot's assumption.
        """
        if self.venue_margin is not None and self.venue_margin > ZERO:
            return self.venue_margin
        leverage = self.leverage if self.leverage > ZERO else ONE
        return (self.quantity * self.entry_price) / leverage


@dataclass(slots=True)
class ReconciliationReport:
    """What the venue holds versus what was known locally."""

    venue_positions: list[VenuePosition] = field(default_factory=list)
    unknown_locally: list[VenuePosition] = field(default_factory=list)
    unprotected: list[VenuePosition] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Whether local state matches the venue and everything is protected."""
        return not self.unknown_locally and not self.unprotected

    @property
    def is_safe_to_trade(self) -> bool:
        """Whether new signals may be acted on.

        An unprotected live position is the disqualifying case: it is real, it is losing or
        winning right now, and nothing is guarding it.
        """
        return not self.unprotected

    def summary(self) -> str:
        """One line for logs and the harness table."""
        return (
            f"venue={len(self.venue_positions)} "
            f"unknown_locally={len(self.unknown_locally)} "
            f"unprotected={len(self.unprotected)}"
        )


def parse_venue_positions(
    raw_positions: list[dict[str, Any]], *, expected_leverage: Decimal = ONE
) -> list[VenuePosition]:
    """Extract live positions from a ccxt ``fetch_positions`` payload.

    Entries with zero size are skipped: Bybit reports flat symbols alongside real ones, and
    treating those as positions would manufacture drift that does not exist.
    """
    out: list[VenuePosition] = []
    for entry in raw_positions:
        info = entry.get("info", {}) if isinstance(entry, dict) else {}
        size_raw = info.get("size") if isinstance(info, dict) else None
        if size_raw in (None, "", "0", 0):
            continue
        try:
            quantity = Decimal(str(size_raw))
        except (ArithmeticError, ValueError):
            continue
        if quantity <= ZERO:
            continue

        raw_symbol = str(entry.get("symbol", "")).split(":")[0]
        try:
            symbol = Symbol.parse(raw_symbol)
        except Exception:  # pragma: no cover - a symbol we cannot parse is still a warning
            logger.warning("reconcile.unparseable_symbol", symbol=raw_symbol)
            continue

        stop_raw = info.get("stopLoss") if isinstance(info, dict) else None
        stop = None
        if stop_raw not in (None, "", "0", 0):
            try:
                stop = Decimal(str(stop_raw))
            except (ArithmeticError, ValueError):
                stop = None

        target_raw = info.get("takeProfit") if isinstance(info, dict) else None
        target = None
        if target_raw not in (None, "", "0", 0):
            try:
                target = Decimal(str(target_raw))
            except (ArithmeticError, ValueError):
                target = None

        entry_raw = info.get("avgPrice") or entry.get("entryPrice") or 0
        try:
            entry_price = Decimal(str(entry_raw))
        except (ArithmeticError, ValueError):
            entry_price = ZERO

        leverage = _decimal_or_none(info.get("leverage") or entry.get("leverage")) or ONE
        venue_margin = _decimal_or_none(
            info.get("positionIM") or info.get("initialMargin") or entry.get("initialMargin")
        )
        if leverage != expected_leverage:
            # Reconcile to the venue, never to the assumption - but say so loudly, because
            # it means the bot's margin view was about to be wrong.
            logger.warning(
                "reconcile.unexpected_leverage",
                symbol=str(symbol),
                venue_leverage=str(leverage),
                expected=str(expected_leverage),
                detail="using the venue value; the bot's margin assumption does not hold",
            )

        # CCXT normalises the direction to long/short at the top level; the raw V5 payload
        # says Buy/Sell. Prefer the normalised one and fall back, so a payload that carries
        # only one of the two is still read correctly rather than defaulting to long.
        side_raw = str(entry.get("side") or info.get("side") or "").lower()

        out.append(
            VenuePosition(
                symbol=symbol,
                side=side_raw,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss_price=stop,
                leverage=leverage,
                venue_margin=venue_margin,
                take_profit_price=target,
            )
        )
    return out


def reconcile(
    venue_positions: list[VenuePosition], known_symbols: set[Symbol]
) -> ReconciliationReport:
    """Compare the venue against what the local book knows."""
    report = ReconciliationReport(venue_positions=list(venue_positions))
    for position in venue_positions:
        if position.symbol not in known_symbols:
            report.unknown_locally.append(position)
        if not position.is_protected:
            report.unprotected.append(position)

    if report.unknown_locally:
        logger.critical(
            "reconcile.unknown_venue_positions",
            symbols=[str(p.symbol) for p in report.unknown_locally],
            detail="the venue holds positions the local book does not know about",
        )
    if report.unprotected:
        logger.critical(
            "reconcile.unprotected_venue_positions",
            symbols=[str(p.symbol) for p in report.unprotected],
            detail="live positions with no server-side stop",
        )
    return report


@dataclass(frozen=True, slots=True)
class VenueAccount:
    """Equity and margin as the exchange reports them.

    On demo or live this is the authoritative account state. A simulated book is a model of
    the account; this *is* the account, so every equity-derived limit should read it rather
    than a reconstruction from fills that may have drifted.
    """

    equity: Decimal
    available: Decimal
    margin_posted: Decimal
    unrealized_pnl: Decimal
    positions: tuple[VenuePosition, ...] = ()

    def matches(self, local_equity: Decimal, *, tolerance: Decimal) -> bool:
        """Whether a locally computed equity agrees with the venue within ``tolerance``."""
        return abs(self.equity - local_equity) <= tolerance


def parse_venue_account(
    balances: dict[str, Any], positions: list[VenuePosition], *, quote: str = "USDT"
) -> VenueAccount:
    """Build the account view from a ccxt balance payload plus parsed positions.

    Prefers the venue's own unified-account totals where present, because those already
    include unrealised PnL and cross-margin effects that a per-asset sum would miss.
    """
    info = balances.get("info", {}) if isinstance(balances, dict) else {}
    equity = _first_decimal(info, ("totalEquity", "totalWalletBalance", "equity"), default=None)
    available = _first_decimal(info, ("totalAvailableBalance", "availableBalance"), default=ZERO)
    margin = _first_decimal(info, ("totalInitialMargin", "totalPositionIM"), default=ZERO)

    unrealised = sum((ZERO for _ in positions), ZERO)
    derived_margin = sum((position.margin_required for position in positions), ZERO)
    if equity is None:
        entry = balances.get(quote) if isinstance(balances, dict) else None
        if isinstance(entry, dict):
            equity = _decimal_or_none(entry.get("total")) or ZERO
        else:
            equity = getattr(entry, "total", ZERO) if entry is not None else ZERO

    return VenueAccount(
        equity=equity or ZERO,
        available=available or ZERO,
        # The venue's own total wins; the per-position sum (itself venue-derived) stands in
        # when the payload omits it.
        margin_posted=margin if margin and margin > ZERO else derived_margin,
        unrealized_pnl=unrealised,
        positions=tuple(positions),
    )


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None


def _first_decimal(
    source: Any, keys: tuple[str, ...], *, default: Decimal | None
) -> Decimal | None:
    """The first parseable value among ``keys``, else ``default``."""
    if not isinstance(source, dict):
        return default
    for key in keys:
        parsed = _decimal_or_none(source.get(key))
        if parsed is not None:
            return parsed
    # Bybit nests the unified account under result.list[0].
    result = source.get("result")
    if isinstance(result, dict):
        entries = result.get("list")
        if isinstance(entries, list) and entries:
            return _first_decimal(entries[0], keys, default=default)
    return default


# --------------------------------------------------------------------------- #
# Live fill / order / position reconciliation
# --------------------------------------------------------------------------- #
#: How far back executions are read on the first pass of a session. A restart has to pick
#: up whatever filled while the process was down, and the venue's execution list is the
#: only record of it — but an unbounded window would replay a session's entire history on
#: every cold start. A day covers any realistic outage of an unattended bot.
DEFAULT_EXECUTION_LOOKBACK = timedelta(hours=24)

#: How many consecutive passes must agree that local state is wrong before it is *repaired*
#: from the position endpoint rather than from executions.
#:
#: Load-bearing. The two venue reads are not simultaneous: ``fetch_positions`` can show a
#: position closed a beat before ``fetch_my_trades`` lists the execution that closed it.
#: Repairing on the first disagreement would synthesise a close, and the real execution
#: would then arrive and be applied on top of a flat book — opening a position that does
#: not exist. Requiring the disagreement to persist gives the execution time to appear, and
#: executions always win when they do.
DEFAULT_REPAIR_CONFIRMATIONS = 2

#: How far a locally computed average entry may sit from the venue's before the two are
#: called different. FIFO lots and the venue's own average agree to the last place on a
#: single-fill position and to rounding on a scaled one; five basis points is far outside
#: that and far inside any real difference.
ENTRY_PRICE_TOLERANCE_PCT = Decimal("0.0005")

#: Most orders read back from the venue in a single pass. A session accumulates orders for
#: as long as it runs and a restart adopts every one that was still working, so an
#: unbounded read would fire dozens of requests before the first bar was processed and hit
#: the rate limiter on the way. The backlog drains over consecutive passes; a steady-state
#: session has a handful of working orders and never reaches the cap.
MAX_ORDER_READS_PER_PASS = 25

#: How many times an order that the venue will not describe is retried before it is left to
#: the execution stream. Without a ceiling, a row the exchange has forgotten becomes one
#: failed request per bar for the rest of the session.
MAX_ORDER_READ_FAILURES = 3


@dataclass(slots=True)
class LiveReconciliation:
    """Everything one reconciliation pass changed.

    Returned rather than acted on, so the component that owns persistence and the event bus
    stays the one that writes: the reconciler's job is to make the portfolio agree with the
    venue, not to decide what that means for the database.
    """

    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    adopted: list[Symbol] = field(default_factory=list)
    orphaned: list[Symbol] = field(default_factory=list)
    #: Wallet balance read from the venue, when this pass had a reason to ask.
    venue_cash: Decimal | None = None
    #: Whether local state had to be rebuilt from the position endpoint. True means the
    #: fill stream did not account for something the venue is holding.
    repaired: bool = False

    @property
    def changed(self) -> bool:
        """Whether anything at all moved."""
        return bool(self.orders or self.fills or self.positions or self.closed_trades)

    def summary(self) -> str:
        """One line for logs."""
        return (
            f"orders={len(self.orders)} fills={len(self.fills)} "
            f"positions={len(self.positions)} closed={len(self.closed_trades)} "
            f"adopted={len(self.adopted)} orphaned={len(self.orphaned)} "
            f"repaired={self.repaired}"
        )


def merge_venue_order(local: Order, venue: Order) -> Order:
    """Fold the venue's view of an order into the local record.

    The local record is kept as the base, not replaced: it carries the strategy that
    produced the order, the protective levels the risk engine attached and our own order id,
    none of which the venue knows. Only the facts the exchange owns — status, filled
    quantity, average price, fees — are taken from it.

    An illegal status transition is refused rather than forced. A venue payload that would
    move a FILLED order back to NEW is a parsing accident, and following it would corrupt
    the audit trail the OMS exists to provide.
    """
    filled = min(venue.filled_quantity, local.quantity)
    status = local.status
    if venue.status is not local.status and can_transition(local.status, venue.status):
        status = venue.status
    return replace(
        local,
        status=status,
        filled_quantity=max(local.filled_quantity, filled),
        average_fill_price=venue.average_fill_price or local.average_fill_price,
        fees_paid=max(local.fees_paid, venue.fees_paid),
        venue_order_id=local.venue_order_id or venue.venue_order_id,
        updated_at=max(local.updated_at, venue.updated_at),
    )


def attach_fills(order: Order, fills: Sequence[Fill]) -> Order:
    """Add executions to an order and recompute its aggregates from them.

    Deduplicated on ``fill_id``: a venue re-delivers the same execution on every poll, and
    counting one twice would report an order as more filled than it is.

    This is also the path that makes a *timed-out* submission correct itself. When the
    read-back of an order fails, the order sits at NEW while the venue has filled it; the
    executions still arrive here, and the status is promoted from what they add up to
    rather than from a response that never came.
    """
    known = {existing.fill_id for existing in order.fills}
    added = [fill for fill in fills if fill.fill_id and fill.fill_id not in known]
    if not added:
        return order

    combined = (*order.fills, *added)
    total = sum((fill.quantity for fill in combined), ZERO)
    gross = sum((fill.quantity * fill.price for fill in combined), ZERO)
    fees = sum((fill.fee for fill in combined), ZERO)
    # A venue that reports more filled than we ordered would break the Order invariant.
    filled = min(total, order.quantity)

    status = order.status
    target = (
        OrderStatus.FILLED
        if filled >= order.quantity
        else (OrderStatus.PARTIALLY_FILLED if filled > ZERO else status)
    )
    if target is not status and can_transition(status, target):
        status = target

    return replace(
        order,
        fills=combined,
        filled_quantity=filled,
        average_fill_price=(gross / total) if total > ZERO else order.average_fill_price,
        fees_paid=max(order.fees_paid, fees),
        status=status,
        updated_at=max(order.updated_at, *(fill.timestamp for fill in added)),
    )


def order_from_fills(order_id: str, fills: Sequence[Fill]) -> Order:
    """Build an order record for executions that have no local order.

    Not every fill on the account came from :meth:`OrderRouter.submit`. A venue-side stop or
    take-profit is created by the exchange alongside the entry, and the intrabar manager
    closes positions through the gateway directly. Those executions are real money moving,
    and without a record to hang them on they cannot be persisted at all — the fills table
    is keyed by order.
    """
    first = fills[0]
    quantity = sum((fill.quantity for fill in fills), ZERO)
    gross = sum((fill.quantity * fill.price for fill in fills), ZERO)
    stamps = [fill.timestamp for fill in fills]
    return Order(
        order_id=order_id[:36],
        client_order_id=order_id[:64],
        symbol=first.symbol,
        side=first.side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        status=OrderStatus.FILLED,
        created_at=min(stamps),
        updated_at=max(stamps),
        filled_quantity=quantity,
        average_fill_price=(gross / quantity) if quantity > ZERO else first.price,
        fees_paid=sum((fill.fee for fill in fills), ZERO),
        fills=tuple(fills),
        metadata={"source": "venue_reconciliation"},
    )


class LiveReconciler:
    """Makes the local portfolio equal what the venue actually holds.

    Three reads, in a deliberate order, because they answer different questions and only
    one of them is the whole truth:

    1. **Orders** (``fetch_order``) — what happened to each thing we sent. This is the only
       source for ``REJECTED`` and for a cancel, neither of which produces an execution.
    2. **Executions** (``fetch_my_trades``) — the fills themselves, with the quantity, the
       price and the fee the venue actually charged. This is what moves the portfolio, and
       it is the *only* thing allowed to, because a fill is the event that has a price.
    3. **Positions** (``fetch_positions``) — the venue's own statement of the book. Used as
       a check, and as a repair of last resort when the first two do not add up to it.

    The ordering matters as much as the reads. Executions are applied before positions are
    compared, so the normal case — an order fills, the execution appears, the position
    follows — never reaches the repair path at all. Repair exists for what executions cannot
    explain: a position opened before this fix existed, a fill older than the lookback, a
    manual intervention on the account.
    """

    __slots__ = (
        "_clock",
        "_confirmations",
        "_gateway",
        "_lookback",
        "_pending",
        "_portfolio",
        "_quote",
        "_read_failures",
        "_since",
        "_symbols",
    )

    def __init__(
        self,
        gateway: Any,
        portfolio: PortfolioManager,
        *,
        symbols: Iterable[Symbol],
        clock: Callable[[], datetime] | None = None,
        execution_lookback: timedelta = DEFAULT_EXECUTION_LOOKBACK,
        repair_confirmations: int = DEFAULT_REPAIR_CONFIRMATIONS,
        quote: str = "USDT",
    ) -> None:
        self._gateway = gateway
        self._portfolio = portfolio
        self._symbols: list[Symbol] = list(dict.fromkeys(symbols))
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lookback = execution_lookback
        self._confirmations = max(1, repair_confirmations)
        self._quote = quote
        #: Per-symbol high-water mark for the execution query. Never rewound: the venue
        #: re-delivers, and the portfolio's own fill-id set is what makes that harmless.
        self._since: dict[Symbol, datetime] = {}
        #: Unconfirmed disagreements, as ``symbol -> (signature, consecutive observations)``.
        self._pending: dict[Symbol, tuple[str, int]] = {}
        #: Consecutive failed read-backs per order. An order the venue will not describe is
        #: retried a few times and then left to the execution stream: a session accumulates
        #: orders for as long as it runs, and retrying every one of them forever would turn
        #: a handful of unresolvable rows into a permanent per-bar request storm.
        self._read_failures: dict[str, int] = {}

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """Symbols whose executions are polled."""
        return tuple(self._symbols)

    def track(self, symbol: Symbol) -> None:
        """Start polling executions for a symbol discovered after construction."""
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def register_venue_ids(self, orders: Iterable[Order]) -> None:
        """Re-teach the gateway the venue ids of orders restored from the database.

        ``fetch_order`` translates our id to the venue's through a map the gateway builds at
        submission. After a restart that map is empty, so every read-back of a pre-restart
        order would be sent with an id the exchange has never seen.
        """
        register = getattr(self._gateway, "register_venue_id", None)
        if not callable(register):
            return
        for order in orders:
            if order.venue_order_id:
                register(order.order_id, order.venue_order_id)

    # ------------------------------------------------------------------ #
    # The pass
    # ------------------------------------------------------------------ #
    async def reconcile(
        self,
        orders: Iterable[Order],
        *,
        initial: bool = False,
        only: Sequence[Symbol] | None = None,
    ) -> LiveReconciliation:
        """Run one reconciliation.

        ``initial`` is for startup: the confirmation delay is skipped, because at startup a
        disagreement is not a race between two endpoints, it is the state the process woke
        up in, and trading must not begin against a book that is known to be wrong.

        ``only`` narrows the pass to specific symbols, for the read straight after a
        submission. The **position pass is skipped** when it is set, deliberately: comparing
        the whole book against the venue while having polled executions for one symbol would
        judge every other symbol on a fill stream this pass never read, and the repair path
        would then close positions whose executions were simply not asked for.
        """
        outcome = LiveReconciliation()
        tracked = list(orders)
        scope = set(only) if only is not None else None
        if scope is not None:
            for symbol in scope:
                self.track(symbol)
            tracked = [order for order in tracked if order.symbol in scope]
        strategy_by_order = {order.order_id: order.strategy_id for order in tracked}

        executions = await self._collect_executions(scope)
        by_order: dict[str, list[Fill]] = {}
        for fill in executions:
            by_order.setdefault(fill.order_id, []).append(fill)

        await self._reconcile_orders(tracked, by_order, outcome)
        self._apply_executions(executions, strategy_by_order, outcome)
        if scope is None:
            await self._reconcile_positions(outcome, initial=initial)

        if initial or outcome.repaired:
            outcome.venue_cash = await self._venue_cash()

        if outcome.changed or outcome.repaired:
            logger.critical("reconcile.pass", detail=outcome.summary(), initial=initial)
        return outcome

    # ------------------------------------------------------------------ #
    # 1. Orders
    # ------------------------------------------------------------------ #
    async def _reconcile_orders(
        self,
        tracked: Sequence[Order],
        by_order: dict[str, list[Fill]],
        outcome: LiveReconciliation,
    ) -> None:
        # Oldest first, and capped. A backlog adopted from the database on restart drains
        # over a few passes instead of firing one read per stale row before the first bar is
        # processed; in steady state a session has a handful of working orders and the cap
        # never binds.
        working = sorted(
            (
                order
                for order in tracked
                if not order.status.is_terminal
                and self._read_failures.get(order.order_id, 0) < MAX_ORDER_READ_FAILURES
            ),
            key=lambda order: order.updated_at,
        )
        to_read = {order.order_id for order in working[:MAX_ORDER_READS_PER_PASS]}

        for order in tracked:
            extra = by_order.pop(order.order_id, [])
            merged = order
            if order.order_id in to_read:
                venue = await self._fetch_order(order)
                if venue is not None:
                    merged = merge_venue_order(merged, venue)
            if extra:
                merged = attach_fills(merged, extra)
            if merged != order:
                outcome.orders.append(merged)

        # Whatever is left belongs to an order this process never submitted through the
        # router: a venue-side stop or target, or an intrabar close.
        for order_id, fills in by_order.items():
            outcome.orders.append(order_from_fills(order_id, fills))

    async def _fetch_order(self, order: Order) -> Order | None:
        try:
            fetched: Order = await self._gateway.fetch_order(order.order_id, order.symbol)
        except Exception as exc:
            # Never fatal. An order the venue will not describe is still an order whose
            # executions arrive through `fetch_my_trades`, and that path alone is enough to
            # settle its status. Failing the pass here would give up the fills as well.
            failures = self._read_failures.get(order.order_id, 0) + 1
            self._read_failures[order.order_id] = failures
            logger.info(
                "reconcile.order_read_failed",
                order_id=order.order_id,
                symbol=str(order.symbol),
                attempts=failures,
                error=str(exc)[:160],
            )
            return None
        self._read_failures.pop(order.order_id, None)
        return fetched

    # ------------------------------------------------------------------ #
    # 2. Executions
    # ------------------------------------------------------------------ #
    async def _collect_executions(self, scope: set[Symbol] | None = None) -> list[Fill]:
        """Every execution the portfolio has not already seen, oldest first."""
        now = self._clock()
        known = self._portfolio.applied_fill_ids
        found: dict[str, Fill] = {}
        for symbol in self._symbols:
            if scope is not None and symbol not in scope:
                continue
            since = self._since.setdefault(symbol, now - self._lookback)
            try:
                rows = await self._gateway.fetch_my_trades(symbol, since=since)
            except Exception as exc:
                logger.warning(
                    "reconcile.executions_read_failed", symbol=str(symbol), error=str(exc)[:160]
                )
                continue
            newest = since
            for fill in rows or []:
                newest = max(newest, fill.timestamp)
                if not fill.fill_id or fill.fill_id in known or fill.fill_id in found:
                    continue
                found[fill.fill_id] = fill
            self._since[symbol] = newest
        return sorted(found.values(), key=lambda fill: (fill.timestamp, fill.fill_id))

    def _apply_executions(
        self,
        executions: Sequence[Fill],
        strategy_by_order: dict[str, str | None],
        outcome: LiveReconciliation,
    ) -> None:
        for fill in executions:
            if fill.fill_id in self._portfolio.applied_fill_ids:
                continue
            position, closed = self._portfolio.apply_fill(
                fill, strategy_id=strategy_by_order.get(fill.order_id)
            )
            outcome.fills.append(fill)
            outcome.positions.append(position)
            outcome.closed_trades.extend(closed)
            logger.critical(
                "reconcile.fill_applied",
                symbol=str(fill.symbol),
                side=fill.side.value,
                quantity=str(fill.quantity),
                price=str(fill.price),
                fee=str(fill.fee),
                realized_pnl=None if fill.realized_pnl is None else str(fill.realized_pnl),
                closed_trades=len(closed),
            )

    # ------------------------------------------------------------------ #
    # 3. Positions
    # ------------------------------------------------------------------ #
    async def _reconcile_positions(self, outcome: LiveReconciliation, *, initial: bool) -> None:
        try:
            rows = await self._gateway.fetch_positions()
        except Exception as exc:
            logger.warning("reconcile.positions_read_failed", error=str(exc)[:160])
            return

        venue = {position.symbol: position for position in parse_venue_positions(rows or [])}
        local = {position.symbol: position for position in self._portfolio.positions}
        now = self._clock()

        # A symbol the account holds but this session never configured still has executions,
        # and they are the only place its exit price and fee will ever be reported. Polling
        # it costs one request and is the difference between closing it from a real fill and
        # closing it from a stale mark.
        for symbol in venue:
            self.track(symbol)

        for symbol in sorted(set(venue) | set(local), key=str):
            target = venue.get(symbol)
            current = local.get(symbol)
            if _agrees(current, target):
                self._pending.pop(symbol, None)
                if target is not None:
                    # Take the venue's protective levels, not the ones that were asked for.
                    # The exchange snaps a stop to its own tick, so the requested price and
                    # the resting price differ in the last places — and the resting one is
                    # the only one that will actually fill. Reported as a change so the
                    # persisted row carries the level the venue is holding.
                    self._portfolio.set_protection(
                        symbol,
                        stop_loss_price=target.stop_loss_price,
                        take_profit_price=target.take_profit_price,
                    )
                    updated = self._portfolio.position_for(symbol)
                    if updated is not None and updated != current:
                        outcome.positions.append(updated)
                continue
            if not initial and not self._confirmed(symbol, target):
                continue
            self._repair(symbol, current, target, now, outcome)
            self._pending.pop(symbol, None)

    def _confirmed(self, symbol: Symbol, target: VenuePosition | None) -> bool:
        """Whether this exact disagreement has been seen often enough to act on."""
        signature = _signature(target)
        previous, count = self._pending.get(symbol, ("", 0))
        count = count + 1 if previous == signature else 1
        self._pending[symbol] = (signature, count)
        if count < self._confirmations:
            logger.info(
                "reconcile.drift_unconfirmed",
                symbol=str(symbol),
                observations=count,
                needed=self._confirmations,
                venue=signature,
            )
            return False
        return True

    def _repair(
        self,
        symbol: Symbol,
        current: Position | None,
        target: VenuePosition | None,
        now: datetime,
        outcome: LiveReconciliation,
    ) -> None:
        """Rebuild one symbol's local position from the venue's statement of it.

        Done with fills rather than by overwriting the object, so a position that is being
        removed still produces the round-trips it closed: a ``ClosedTrade`` with its realised
        PnL and its fees is the only durable record that a trade happened, and silently
        deleting the position would lose it.
        """
        outcome.repaired = True
        if current is not None and not current.is_flat:
            closing_side = current.closing_side()
            price = self._portfolio.mark_price(symbol) or current.average_entry_price
            if closing_side is not None and price > ZERO:
                fill = Fill(
                    fill_id=f"reconcile-close-{symbol.concatenated}-{uuid.uuid4().hex[:12]}",
                    order_id=f"reconcile-{uuid.uuid4().hex[:12]}",
                    symbol=symbol,
                    side=closing_side,
                    quantity=current.absolute_quantity,
                    price=price,
                    fee=ZERO,
                    fee_currency=symbol.quote,
                    timestamp=now,
                )
                position, closed = self._portfolio.apply_fill(fill, strategy_id=current.strategy_id)
                outcome.positions.append(position)
                outcome.closed_trades.extend(closed)
                if target is None:
                    outcome.orphaned.append(symbol)
                logger.critical(
                    "reconcile.local_position_closed",
                    symbol=str(symbol),
                    quantity=str(current.absolute_quantity),
                    price=str(price),
                    detail="the venue does not hold this position; local state was a claim",
                )

        if target is not None:
            fill = Fill(
                fill_id=f"reconcile-open-{symbol.concatenated}-{uuid.uuid4().hex[:12]}",
                order_id=f"reconcile-{uuid.uuid4().hex[:12]}",
                symbol=symbol,
                side=(
                    OrderSide.SELL if target.position_side is PositionSide.SHORT else OrderSide.BUY
                ),
                quantity=target.quantity,
                # The venue's own average entry, so the adopted position is the venue's
                # position and not an approximation of it.
                price=target.entry_price,
                fee=ZERO,
                fee_currency=symbol.quote,
                timestamp=now,
            )
            position, _ = self._portfolio.apply_fill(fill)
            self._portfolio.set_protection(
                symbol,
                stop_loss_price=target.stop_loss_price,
                take_profit_price=target.take_profit_price,
            )
            outcome.positions.append(self._portfolio.position_for(symbol) or position)
            outcome.adopted.append(symbol)
            logger.critical(
                "reconcile.venue_position_adopted",
                symbol=str(symbol),
                side=target.position_side.value,
                quantity=str(target.quantity),
                entry=str(target.entry_price),
                stop=None if target.stop_loss_price is None else str(target.stop_loss_price),
            )

        # Everything the venue executed on this symbol up to now is already accounted for by
        # the state just adopted. Replaying an execution from before the repair would count
        # the same trade twice, so the window starts again from here.
        self._since[symbol] = now

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    async def _venue_cash(self) -> Decimal | None:
        """The quote-currency wallet balance, or ``None`` if it cannot be read."""
        try:
            balances = await self._gateway.fetch_balances()
        except Exception as exc:
            logger.warning("reconcile.balance_read_failed", error=str(exc)[:160])
            return None
        balance = (balances or {}).get(self._quote)
        if balance is None:
            return None
        total = balance.free + balance.locked
        return total if total > ZERO else None


def _agrees(current: Position | None, target: VenuePosition | None) -> bool:
    """Whether the local position is the venue's position."""
    if target is None:
        return current is None or current.is_flat
    if current is None or current.is_flat:
        return False
    if current.quantity != target.signed_quantity:
        return False
    entry = current.average_entry_price
    if target.entry_price <= ZERO:
        return True
    return abs(entry - target.entry_price) <= target.entry_price * ENTRY_PRICE_TOLERANCE_PCT


def _signature(target: VenuePosition | None) -> str:
    """A stable description of a venue position, for confirming repeated observations."""
    if target is None:
        return "flat"
    return f"{target.position_side.value}:{target.quantity}:{target.entry_price}"


__all__ = [
    "DEFAULT_EXECUTION_LOOKBACK",
    "DEFAULT_REPAIR_CONFIRMATIONS",
    "ENTRY_PRICE_TOLERANCE_PCT",
    "LiveReconciler",
    "LiveReconciliation",
    "ReconciliationReport",
    "VenueAccount",
    "VenuePosition",
    "attach_fills",
    "merge_venue_order",
    "order_from_fills",
    "parse_venue_account",
    "parse_venue_positions",
    "reconcile",
]
