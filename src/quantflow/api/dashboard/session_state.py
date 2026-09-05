"""QuantFlow's own accounting for the running session.

This is one of the four sources the dashboard keeps strictly apart:

* **Venue account state** — what the exchange says the account holds. See
  :mod:`quantflow.api.dashboard.valuation`.
* **QuantFlow session state** — this module: what *this bot run* has done, reconstructed
  from reconciled rows in the database.
* **Portfolio valuation** — converting holdings to one unit at current prices.
* **Trading performance** — attribution and analytics over closed trades.

Conflating the first two is how a dashboard comes to report an account balance as though
it were strategy profit. They are computed separately here and labelled separately on the
wire, and no field mixes them.

Everything is computed with **SQL aggregates over indexed columns**, not by loading trades
into Python. The dashboard polls, and an endpoint that pulls ten thousand rows to sum a
column would make the operator's browser a load source against the database the trading
engine depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from quantflow.core.clock import utc_now
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import OPEN_ORDER_STATUSES, RunStatus
from quantflow.persistence.models import (
    ClosedTradeRecord,
    EquitySnapshotRecord,
    FillRecord,
    OrderRecord,
    PositionRecord,
    TradingSessionRecord,
)
from quantflow.universe.assets import ENERGY_ROOTS, INDEX_ETF_ROOTS, METAL_ROOTS

#: Ranges the equity and drawdown charts offer, and the window each covers.
RANGES: dict[str, timedelta | None] = {
    "1H": timedelta(hours=1),
    "6H": timedelta(hours=6),
    "24H": timedelta(hours=24),
    "7D": timedelta(days=7),
    "30D": timedelta(days=30),
    "ALL": None,
}

#: Most points returned for a chart. Beyond this the series is strided — never
#: interpolated, and never gap-filled: a fabricated point on an equity curve is a
#: fabricated account balance.
MAX_CHART_POINTS = 1_500

#: Below this many closed trades, per-group statistics are marked unreliable. A win rate
#: over two trades is noise, and rendering it identically to one over two hundred invites
#: the reader to act on it.
MIN_SAMPLE = 10


@dataclass(frozen=True, slots=True)
class SessionRef:
    """Identity of the session the dashboard is reporting on."""

    session_id: str
    mode: str
    status: str
    strategy_id: str
    timeframe: str
    base_currency: str
    symbols: tuple[str, ...]
    starting_equity: Decimal
    started_at: datetime | None
    created_at: datetime | None
    is_running: bool
    selection_basis: str

    def to_dict(self) -> dict[str, Any]:
        """Wire form."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "status": self.status,
            "strategy_id": self.strategy_id,
            "timeframe": self.timeframe,
            "base_currency": self.base_currency,
            "symbols": list(self.symbols),
            "starting_equity": str(self.starting_equity),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_running": self.is_running,
            "selection_basis": self.selection_basis,
        }


async def resolve_session(
    session: AsyncSession, *, session_id: str | None = None
) -> SessionRef | None:
    """Identify the session the dashboard should report on.

    The obvious implementation — "the most recently created session" — is wrong here, and
    was actively misleading on this deployment. Sessions are created once and *reused*
    across restarts: a restart re-stamps ``started_at`` but leaves ``created_at`` at the
    original insert. A stale session created later therefore outranks the one currently
    trading, and every trade, equity point and analytic on the page silently describes a
    run that stopped a day ago.

    Selection is therefore: the running session that started most recently; failing that,
    the most recently started session of any status. ``created_at`` is the last resort.

    Args:
        session: An open read session.
        session_id: Report on this session explicitly instead of choosing one.

    Returns:
        The chosen session, or ``None`` when none has ever run.

    """
    if session_id is not None:
        record = await session.get(TradingSessionRecord, session_id)
        return _to_ref(record, basis="requested explicitly") if record is not None else None

    running = (
        await session.execute(
            select(TradingSessionRecord)
            .where(TradingSessionRecord.status == RunStatus.RUNNING)
            .order_by(
                TradingSessionRecord.started_at.desc().nullslast(),
                TradingSessionRecord.created_at.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if running is not None:
        return _to_ref(running, basis="most recently started session with status running")

    latest = (
        await session.execute(
            select(TradingSessionRecord)
            .order_by(
                TradingSessionRecord.started_at.desc().nullslast(),
                TradingSessionRecord.created_at.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return None
    return _to_ref(latest, basis="no running session; most recently started session")


def _to_ref(record: TradingSessionRecord, *, basis: str) -> SessionRef:
    """Build a :class:`SessionRef` from an ORM row, inside its session."""
    symbols = record.symbols if isinstance(record.symbols, list) else []
    return SessionRef(
        session_id=record.id,
        mode=record.mode,
        status=str(getattr(record.status, "value", record.status)),
        strategy_id=record.strategy_id,
        timeframe=str(getattr(record.timeframe, "value", record.timeframe)),
        base_currency=record.base_currency,
        symbols=tuple(str(item) for item in symbols),
        starting_equity=record.starting_equity,
        started_at=record.started_at,
        created_at=record.created_at,
        is_running=record.status == RunStatus.RUNNING,
        selection_basis=basis,
    )


def _trade_aggregate(session_id: str | None) -> Select[Any]:
    """Aggregate over closed trades, computed by the database.

    ``session_id=None`` aggregates across every session. It is spelled as an explicit
    ``None`` rather than an omitted filter because ``== None`` renders as ``IS NULL`` in
    SQL, which would silently match nothing instead of matching everything.
    """
    aggregate = select(
        func.count().label("trades"),
        func.coalesce(func.sum(ClosedTradeRecord.gross_pnl), 0).label("gross"),
        func.coalesce(func.sum(ClosedTradeRecord.fees), 0).label("fees"),
        func.coalesce(func.sum(ClosedTradeRecord.net_pnl), 0).label("net"),
        func.count().filter(ClosedTradeRecord.net_pnl > 0).label("wins"),
        func.count().filter(ClosedTradeRecord.net_pnl < 0).label("losses"),
        func.coalesce(
            func.sum(ClosedTradeRecord.net_pnl).filter(ClosedTradeRecord.net_pnl > 0), 0
        ).label("gross_profit"),
        func.coalesce(
            func.sum(ClosedTradeRecord.net_pnl).filter(ClosedTradeRecord.net_pnl < 0), 0
        ).label("gross_loss"),
        func.max(ClosedTradeRecord.net_pnl).label("best"),
        func.min(ClosedTradeRecord.net_pnl).label("worst"),
        func.min(ClosedTradeRecord.exit_time).label("first_exit"),
        func.max(ClosedTradeRecord.exit_time).label("last_exit"),
        func.coalesce(func.avg(ClosedTradeRecord.holding_period_seconds), 0).label("avg_hold"),
    )
    if session_id is None:
        return aggregate
    return aggregate.where(ClosedTradeRecord.session_id == session_id)


async def performance(session: AsyncSession, reference: SessionRef) -> dict[str, Any]:
    """Everything in the dashboard header, drawn from reconciled session rows.

    Args:
        session: An open read session.
        reference: The session being reported on.

    Returns:
        A JSON-safe mapping with every money value as an exact decimal string.

    """
    session_id = reference.session_id
    totals = (await session.execute(_trade_aggregate(session_id))).one()

    midnight = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    today = (
        await session.execute(
            _trade_aggregate(session_id).where(ClosedTradeRecord.exit_time >= midnight)
        )
    ).one()

    equity = (
        await session.execute(
            select(
                func.max(EquitySnapshotRecord.equity).label("peak"),
                func.min(EquitySnapshotRecord.timestamp).label("first_at"),
                func.max(EquitySnapshotRecord.timestamp).label("last_at"),
                func.count().label("points"),
                func.max(EquitySnapshotRecord.drawdown_pct).label("max_drawdown"),
            ).where(EquitySnapshotRecord.session_id == session_id)
        )
    ).one()

    latest = (
        await session.execute(
            select(EquitySnapshotRecord)
            .where(EquitySnapshotRecord.session_id == session_id)
            .order_by(EquitySnapshotRecord.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    open_positions = (
        await session.execute(
            select(func.count())
            .select_from(PositionRecord)
            .where(
                PositionRecord.session_id == session_id,
                PositionRecord.closed_at.is_(None),
                PositionRecord.quantity != 0,
            )
        )
    ).scalar_one()

    open_orders = (
        await session.execute(
            select(func.count())
            .select_from(OrderRecord)
            .where(
                OrderRecord.session_id == session_id,
                OrderRecord.status.in_(OPEN_ORDER_STATUSES),
            )
        )
    ).scalar_one()

    net = _dec(totals.net)
    gross = _dec(totals.gross)
    fees = _dec(totals.fees)
    gross_profit = _dec(totals.gross_profit)
    gross_loss = abs(_dec(totals.gross_loss))
    trades = int(totals.trades)

    return {
        "trading_performance": {
            # Realised results, reconciled from closed round-trips. This is strategy
            # performance; it is never the account balance and never mixes with it.
            "closed_trades": trades,
            "gross_realized_pnl": str(gross),
            "total_fees": str(fees),
            "net_realized_pnl": str(net),
            "today_net_pnl": str(_dec(today.net)),
            "today_closed_trades": int(today.trades),
            "today_fees": str(_dec(today.fees)),
            "win_count": int(totals.wins),
            "loss_count": int(totals.losses),
            "win_rate": (
                str(safe_divide(Decimal(int(totals.wins)), Decimal(trades))) if trades else None
            ),
            # Profit factor is undefined without a losing trade; a division guarded to
            # zero would render as "0.00" and read as catastrophic rather than unknown.
            "profit_factor": (
                str(safe_divide(gross_profit, gross_loss)) if gross_loss > ZERO else None
            ),
            "gross_profit": str(gross_profit),
            "gross_loss": str(gross_loss),
            "average_net_pnl": str(safe_divide(net, Decimal(trades))) if trades else None,
            "best_trade": str(_dec(totals.best)) if totals.best is not None else None,
            "worst_trade": str(_dec(totals.worst)) if totals.worst is not None else None,
            "average_holding_seconds": str(_dec(totals.avg_hold)),
            "first_exit_at": _iso(totals.first_exit),
            "last_exit_at": _iso(totals.last_exit),
            "sample_is_thin": trades < MIN_SAMPLE,
        },
        "session_equity": {
            # QuantFlow's own equity accounting for this run, from persisted snapshots.
            "starting_equity": str(reference.starting_equity),
            # The base every percentage limit is computed against. Named for what it is
            # rather than as an "allocation": there is no ceiling below the wallet, and
            # calling it one was actively misleading — the dashboard reported a 10,000
            # allocation while sizing ran against the ~49,774 wallet, because the
            # reconciler re-anchors cash to the venue on its first pass.
            "capital_base": str(reference.starting_equity),
            "capital_base_source": (
                "authoritative Bybit USDT wallet equity, read at session start; only the "
                "USDT balance counts, since other assets are inventory on a USDT book"
            ),
            "latest_equity": str(latest.equity) if latest is not None else None,
            "latest_cash": str(latest.cash) if latest is not None else None,
            "latest_unrealized_pnl": str(latest.unrealized_pnl) if latest is not None else None,
            "latest_realized_pnl": str(latest.realized_pnl) if latest is not None else None,
            "latest_gross_exposure": str(latest.gross_exposure) if latest is not None else None,
            "latest_at": _iso(latest.timestamp) if latest is not None else None,
            "peak_equity": str(_dec(equity.peak)) if equity.peak is not None else None,
            "current_drawdown_pct": str(latest.drawdown_pct) if latest is not None else None,
            "max_drawdown_pct": (
                str(_dec(equity.max_drawdown)) if equity.max_drawdown is not None else None
            ),
            "return_pct": (
                str(safe_divide(net, reference.starting_equity))
                if reference.starting_equity > ZERO
                else None
            ),
            "return_basis": (
                "net realised PnL over the session's starting equity; the session row is "
                "reused across engine restarts, so starting equity is re-stamped on restart"
            ),
            "snapshot_count": int(equity.points),
            "history_from": _iso(equity.first_at),
            "history_to": _iso(equity.last_at),
        },
        "session_book": {
            # Counted from QuantFlow's own store. The venue is authoritative for what is
            # actually open; the dashboard shows both and flags a divergence rather than
            # silently preferring one.
            "open_positions": int(open_positions),
            "open_orders": int(open_orders),
            "source": "quantflow database, scoped to this session",
        },
    }


#: The periods the profit-and-loss panel offers, as lookbacks from now.
#:
#: ``SESSION`` and ``ALL`` are deliberately different questions and are both offered:
#: SESSION is this run of the engine, ALL is every trade ever recorded across every
#: session. On a bot that has been restarted 22 times, presenting only one of them would
#: answer "how much has it made" with a number that quietly excludes most of its life.
PNL_PERIODS: dict[str, timedelta | None] = {
    "TODAY": None,  # special-cased: since UTC midnight, not a rolling 24h
    "7D": timedelta(days=7),
    "30D": timedelta(days=30),
    "ALL": None,
    "SESSION": None,
}


def _summarise_aggregate(row: Any) -> dict[str, Any]:
    """Shape one aggregate row into the panel's figures."""
    trades = int(row.trades)
    gross_profit = _dec(row.gross_profit)
    gross_loss = abs(_dec(row.gross_loss))
    net = _dec(row.net)
    return {
        "closed_trades": trades,
        # Gross profit and gross loss are both net-of-fee sums, split by sign — the pair a
        # profit factor is actually computed from. The fee total is reported alongside
        # rather than folded in twice.
        "gross_profit": str(gross_profit),
        "gross_loss": str(gross_loss),
        "fees": str(_dec(row.fees)),
        "net_pnl": str(net),
        "win_count": int(row.wins),
        "loss_count": int(row.losses),
        "win_rate": str(safe_divide(Decimal(int(row.wins)), Decimal(trades))) if trades else None,
        # Undefined without a loser: a zero here would read as total ruin rather than as
        # "no losing trade yet".
        "profit_factor": str(safe_divide(gross_profit, gross_loss)) if gross_loss > ZERO else None,
        "sample_is_thin": trades < MIN_SAMPLE,
    }


async def attribution_coverage(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """How much of this session's history can actually be attributed.

    Reported rather than assumed, because the honest answer has been "almost none". Every
    closed trade in demo-10k-fresh carried a NULL strategy, which made every per-strategy
    question unanswerable while the dashboard happily rendered empty groupings as though
    they were findings.

    Fields the schema has no column for are named in ``not_recorded`` instead of being
    counted as missing data — absent by design and absent by accident deserve different
    responses, and neither should be filled in with a guess.
    """
    total, attributed, with_regime = (
        await session.execute(
            select(
                func.count(),
                func.count(ClosedTradeRecord.strategy_id),
                func.count().filter(ClosedTradeRecord.regime != "unknown"),
            ).where(ClosedTradeRecord.session_id == session_id)
        )
    ).one()

    trades = int(total)
    return {
        "closed_trades": trades,
        "attributed_to_strategy": int(attributed),
        "unattributed": trades - int(attributed),
        "attribution_pct": (
            str(safe_divide(Decimal(int(attributed)), Decimal(trades))) if trades else None
        ),
        "with_regime": int(with_regime),
        # Named explicitly so the panel renders NOT RECORDED rather than a zero that reads
        # as a measurement.
        "not_recorded": [
            "exit_reason — no column on closed_trades",
            "mfe / mae — no columns on closed_trades",
            "slippage — not captured at fill time",
            "entry_fee / exit_fee split — persisted as one total per trade",
            "conviction inputs — not persisted",
        ],
    }


async def execution_quality(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """How orders actually executed, split by liquidity role.

    ``Fill.role`` has always recorded whether an execution paid the maker or the taker
    rate, and nothing ever surfaced it. That mattered the moment maker-first was switched
    on: the entries went out as limit orders, looked correct in every log line, and were
    still charged the 0.055% taker rate because the post-only flag was being dropped before
    it reached the venue. A visible fee-per-role would have shown that in one glance.

    Fill rate is reported as the maker share of fills rather than of orders, because an
    order that never fills has no fee and does not belong in a fee comparison.
    """
    rows = (
        await session.execute(
            select(
                FillRecord.role,
                func.count().label("fills"),
                func.coalesce(func.sum(FillRecord.fee), 0).label("fees"),
                func.coalesce(func.sum(FillRecord.quantity * FillRecord.price), 0).label(
                    "notional"
                ),
            )
            .join(OrderRecord, FillRecord.order_id == OrderRecord.id)
            .where(OrderRecord.session_id == session_id)
            .group_by(FillRecord.role)
        )
    ).all()

    by_role: dict[str, Any] = {}
    total_fills = 0
    for row in rows:
        role = str(getattr(row.role, "value", row.role)).lower()
        fills = int(row.fills)
        total_fills += fills
        notional = _dec(row.notional)
        by_role[role] = {
            "fills": fills,
            "fees": str(_dec(row.fees)),
            "notional": str(notional),
            "average_fee_pct": (
                str(safe_divide(_dec(row.fees), notional) * Decimal("100"))
                if notional > ZERO
                else None
            ),
        }

    maker = by_role.get("maker", {}).get("fills", 0)
    return {
        "by_role": by_role,
        "total_fills": total_fills,
        # None rather than 0 when nothing has filled: "no maker fills yet" and "0% of fills
        # were maker" are different claims, and only the second is a measurement.
        "maker_fill_rate": (
            str(safe_divide(Decimal(maker), Decimal(total_fills))) if total_fills else None
        ),
        "source": "persisted fills, joined to this session's orders",
    }


async def pnl_by_period(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """Profit and loss over each offered period.

    ``TODAY``, ``7D`` and ``30D`` are scoped to this session, because that is the run whose
    equity curve sits beside them. ``ALL`` deliberately is not: it spans every session, and
    says so, so a restart cannot make the lifetime figure appear to reset.

    Args:
        session: An open read session.
        session_id: The session in scope for the session-scoped periods.

    Returns:
        A mapping of period name to figures, each declaring its own scope.

    """
    now = utc_now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    scoped = _trade_aggregate(session_id)
    queries: dict[str, tuple[Select[Any], str]] = {
        "TODAY": (
            scoped.where(ClosedTradeRecord.exit_time >= midnight),
            "this session, since 00:00 UTC",
        ),
        "7D": (
            scoped.where(ClosedTradeRecord.exit_time >= now - timedelta(days=7)),
            "this session, last 7 days",
        ),
        "30D": (
            scoped.where(ClosedTradeRecord.exit_time >= now - timedelta(days=30)),
            "this session, last 30 days",
        ),
        "SESSION": (scoped, "this session, all time"),
        "ALL": (_trade_aggregate(None), "every session ever recorded"),
    }

    periods: dict[str, Any] = {}
    for name, (query, scope) in queries.items():
        row = (await session.execute(query)).one()
        periods[name] = {**_summarise_aggregate(row), "scope": scope}
    return {
        "periods": periods,
        "order": ["TODAY", "7D", "30D", "SESSION", "ALL"],
        "generated_at": now.isoformat(),
    }


async def cumulative_pnl(
    session: AsyncSession, session_id: str, *, window: str = "ALL"
) -> dict[str, Any]:
    """Cumulative net PnL after each closed trade, for a single chart.

    Built from the trades themselves rather than from equity snapshots, so the line moves
    only when a round-trip closes and cannot be confused with mark-to-market drift on an
    open position.
    """
    delta = RANGES.get(window)
    query = (
        select(
            ClosedTradeRecord.exit_time,
            ClosedTradeRecord.net_pnl,
            ClosedTradeRecord.gross_pnl,
            ClosedTradeRecord.fees,
        )
        .where(ClosedTradeRecord.session_id == session_id)
        .order_by(ClosedTradeRecord.exit_time)
    )
    if delta is not None:
        query = query.where(ClosedTradeRecord.exit_time >= utc_now() - delta)

    rows = (await session.execute(query)).all()
    points: list[dict[str, Any]] = []
    net = gross = fees = ZERO
    for row in rows:
        net += _dec(row.net_pnl)
        gross += _dec(row.gross_pnl)
        fees += _dec(row.fees)
        points.append(
            {
                "at": _iso(row.exit_time),
                "cumulative_net": str(net),
                "cumulative_gross": str(gross),
                "cumulative_fees": str(fees),
            }
        )
    return {
        "window": window if window in RANGES else "ALL",
        "points": points[-MAX_CHART_POINTS:],
        "point_count": len(points),
        "truncated": len(points) > MAX_CHART_POINTS,
        "source": "closed trades in this session, accumulated in exit order",
    }


async def equity_series(
    session: AsyncSession, session_id: str, *, window: str = "ALL"
) -> dict[str, Any]:
    """The equity curve, its running peak and its drawdown, for one range.

    Missing points are never invented. When the requested window starts before the first
    stored snapshot the response says so, and the client renders the real extent of the
    history rather than a line that appears to begin at the window's edge.

    Args:
        session: An open read session.
        session_id: The session to chart.
        window: One of :data:`RANGES`.

    Returns:
        A JSON-safe mapping with the points, the true available extent and the largest
        drawdown episode within the window.

    """
    delta = RANGES.get(window)
    statement = (
        select(
            EquitySnapshotRecord.timestamp,
            EquitySnapshotRecord.equity,
            EquitySnapshotRecord.cash,
            EquitySnapshotRecord.realized_pnl,
            EquitySnapshotRecord.unrealized_pnl,
            EquitySnapshotRecord.drawdown_pct,
            EquitySnapshotRecord.position_count,
        )
        .where(EquitySnapshotRecord.session_id == session_id)
        .order_by(EquitySnapshotRecord.timestamp.asc())
    )
    if delta is not None:
        statement = statement.where(EquitySnapshotRecord.timestamp >= utc_now() - delta)

    rows = (await session.execute(statement)).all()

    extent = (
        await session.execute(
            select(
                func.min(EquitySnapshotRecord.timestamp),
                func.max(EquitySnapshotRecord.timestamp),
                func.count(),
            ).where(EquitySnapshotRecord.session_id == session_id)
        )
    ).one()

    stride = max(1, (len(rows) // MAX_CHART_POINTS) + 1)
    sampled = rows[::stride] if stride > 1 else rows
    if stride > 1 and rows and sampled[-1] is not rows[-1]:
        # The most recent point is always kept: an operator reads the right-hand end of an
        # equity curve as "now", and striding it away makes the chart lag reality.
        sampled = [*sampled, rows[-1]]

    points: list[dict[str, Any]] = []
    peak = ZERO
    largest = {"depth_pct": "0", "at": None, "peak_equity": None, "trough_equity": None}
    for row in sampled:
        peak = max(peak, row.equity)
        drawdown = safe_divide(peak - row.equity, peak)
        if Decimal(largest["depth_pct"] or "0") < drawdown:
            largest = {
                "depth_pct": str(drawdown),
                "at": _iso(row.timestamp),
                "peak_equity": str(peak),
                "trough_equity": str(row.equity),
            }
        points.append(
            {
                "timestamp": _iso(row.timestamp),
                "equity": str(row.equity),
                "cash": str(row.cash),
                "realized_pnl": str(row.realized_pnl),
                "unrealized_pnl": str(row.unrealized_pnl),
                # Recomputed from the series rather than read from the row, so the peak
                # line and the drawdown line on the chart cannot disagree with each other.
                "running_peak": str(peak),
                "drawdown_pct": str(drawdown),
                "recorded_drawdown_pct": str(row.drawdown_pct),
                "position_count": int(row.position_count),
            }
        )

    breaks = _discontinuities(points)
    return {
        "window": window if window in RANGES else "ALL",
        "points": points,
        "point_count": len(points),
        "stride": stride,
        "available_from": _iso(extent[0]),
        "available_to": _iso(extent[1]),
        "total_snapshots": int(extent[2]),
        "truncated": stride > 1,
        # A session row is reused across engine restarts, and a restart re-seeds equity
        # from the venue. When the capital base changed mid-session the curve is not one
        # continuous account, and reading a return across the break is meaningless. The
        # breaks are reported so the chart can mark them instead of drawing a smooth line
        # through a step change.
        "discontinuities": breaks,
        "continuous": not breaks,
        # Rendered verbatim by the client under the chart when the window predates the
        # first snapshot, instead of drawing a line that implies data that does not exist.
        "history_note": (
            f"history unavailable before {_iso(extent[0])}"
            if extent[0] is not None
            else "no equity history recorded for this session"
        ),
    }


#: Relative equity change between adjacent snapshots that indicates a re-seed rather than
#: a trading result. No strategy on a 15-minute bar moves an account by a quarter.
DISCONTINUITY_RATIO = Decimal("0.25")


def _discontinuities(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find step changes in an equity series that trading cannot explain."""
    breaks: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for point in points:
        current = Decimal(str(point["equity"]))
        if previous is not None and previous > ZERO:
            change = abs(current - previous) / previous
            if change > DISCONTINUITY_RATIO:
                breaks.append(
                    {
                        "at": point["timestamp"],
                        "from_equity": str(previous),
                        "to_equity": str(current),
                        "change_pct": str(change),
                        "likely_cause": (
                            "the session was restarted and its equity re-seeded from the "
                            "venue; the curve before this point describes a different "
                            "capital base"
                        ),
                    }
                )
        previous = current
    return breaks


async def trade_ledger(
    session: AsyncSession,
    session_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """Every closed round-trip, newest first, with the fields the schema actually holds.

    Fields the engine does not persist are reported once in ``not_recorded`` rather than
    being emitted per row as zeros or empty strings. A ledger that shows ``0.00`` for a
    quantity nobody measured is indistinguishable from one that measured zero.

    Args:
        session: An open read session.
        session_id: The session whose trades to list.
        limit: Page size.
        offset: Rows to skip.

    Returns:
        A JSON-safe mapping with the page of trades and the total count.

    """
    total = (
        await session.execute(
            select(func.count())
            .select_from(ClosedTradeRecord)
            .where(ClosedTradeRecord.session_id == session_id)
        )
    ).scalar_one()

    rows = (
        (
            await session.execute(
                select(ClosedTradeRecord)
                .where(ClosedTradeRecord.session_id == session_id)
                .order_by(ClosedTradeRecord.exit_time.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )

    trades: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        gross = row.gross_pnl
        fees = row.fees
        notional = row.quantity * row.entry_price
        trades.append(
            {
                # Numbered from the newest so the number is stable as the page scrolls,
                # and matches the count shown in the header.
                "trade_number": int(total) - offset - index,
                "trade_id": row.id,
                "symbol": row.symbol,
                "asset_class": _asset_class(row.symbol),
                "side": str(getattr(row.side, "value", row.side)),
                "quantity": str(row.quantity),
                "entry_time": _iso(row.entry_time),
                "exit_time": _iso(row.exit_time),
                "entry_price": str(row.entry_price),
                "exit_price": str(row.exit_price),
                "entry_notional": str(notional),
                "holding_seconds": int(row.holding_period_seconds),
                "gross_pnl": str(gross),
                "total_fees": str(fees),
                "net_pnl": str(row.net_pnl),
                "return_pct": str(row.return_pct),
                "fee_share_of_gross": (
                    str(safe_divide(fees, abs(gross))) if gross != ZERO else None
                ),
                "strategy_id": row.strategy_id,
                "regime": (
                    None
                    if str(getattr(row.regime, "value", row.regime)) == "unknown"
                    else str(getattr(row.regime, "value", row.regime))
                ),
                "notes": row.notes,
                # Persisted as a single figure per trade. The split is genuinely absent
                # from the schema, so it is declared absent rather than halved.
                "entry_fee": None,
                "exit_fee": None,
                "exit_reason": None,
                "mfe": None,
                "mae": None,
                "order_ids": None,
                "position_id": None,
                "venue_fill_ids": None,
            }
        )

    return {
        "trades": trades,
        "total": int(total),
        "limit": limit,
        "offset": offset,
        # Named explicitly so the client can render NOT RECORDED with a reason, and so a
        # future schema change makes these disappear from the list rather than silently
        # changing a column's meaning.
        "not_recorded": {
            "entry_fee": "closed_trades stores one combined fee per round-trip",
            "exit_fee": "closed_trades stores one combined fee per round-trip",
            "exit_reason": "no exit-reason column exists; notes is null for every trade",
            "mfe": "maximum favourable excursion is not measured by the engine",
            "mae": "maximum adverse excursion is not measured by the engine",
            "order_ids": "closed_trades holds no link back to the orders that made it",
            "position_id": "closed_trades holds no link back to the position",
            "venue_fill_ids": "fills are linked to orders, not to closed trades",
        },
    }


async def data_coverage(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """Where this session's own history is incomplete.

    A ledger that simply starts at its earliest row implies the history begins there. On
    this session it does not: fourteen entry orders were written before any fill was
    persisted, so the account traded for hours that the ledger cannot describe. Stating the
    boundary is the difference between a short history and a silently truncated one.

    Args:
        session: An open read session.
        session_id: The session to examine.

    Returns:
        A JSON-safe mapping describing the earliest reliable data and any gap before it.

    """
    from quantflow.persistence.models import FillRecord

    earliest_fill = (
        await session.execute(
            select(func.min(FillRecord.timestamp))
            .select_from(FillRecord)
            .join(OrderRecord, FillRecord.order_id == OrderRecord.id)
            .where(OrderRecord.session_id == session_id)
        )
    ).scalar_one_or_none()

    orphan = (
        await session.execute(
            select(
                func.count().label("orders"),
                func.min(OrderRecord.created_at).label("first"),
                func.max(OrderRecord.created_at).label("last"),
            ).where(
                OrderRecord.session_id == session_id,
                ~OrderRecord.fills.any(),
            )
        )
    ).one()

    orphan_count = int(orphan.orders)
    return {
        "earliest_fill_at": _iso(earliest_fill),
        "orders_without_fills": orphan_count,
        "gap_from": _iso(orphan.first),
        "gap_to": _iso(orphan.last),
        "has_gap": orphan_count > 0,
        "note": (
            f"historical data unavailable before {_iso(earliest_fill)} — {orphan_count} "
            f"entry order(s) between {_iso(orphan.first)} and {_iso(orphan.last)} have no "
            "fills persisted and produced no closed-trade rows; their per-fill fees, venue "
            "fill ids and liquidity roles are permanently lost"
            if orphan_count > 0
            else "every order in this session has at least one persisted fill"
        ),
    }


async def attribution(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """Performance grouped by strategy, symbol, side and exit reason.

    Args:
        session: An open read session.
        session_id: The session to analyse.

    Returns:
        A JSON-safe mapping of grouped statistics, each group carrying its own sample size
        and a reliability flag.

    """
    by_strategy = await _group(session, session_id, ClosedTradeRecord.strategy_id)
    by_symbol = await _group(session, session_id, ClosedTradeRecord.symbol)
    by_side = await _group(session, session_id, ClosedTradeRecord.side)

    for row in by_symbol:
        row["asset_class"] = _asset_class(str(row["key"]))

    unattributed = sum(1 for row in by_strategy if row["key"] is None for _ in (0,))
    return {
        "by_strategy": by_strategy,
        "by_symbol": by_symbol,
        "by_side": by_side,
        "by_exit_reason": [],
        "exit_reason_available": False,
        "exit_reason_note": (
            "the engine records no exit reason on a closed trade — there is no column for "
            "it and notes is null for every row, so this analysis cannot be produced"
        ),
        "strategy_attribution_note": (
            "trades written by venue reconciliation carry no strategy_id and appear under "
            "an unattributed group"
            if unattributed
            else None
        ),
        "min_sample": MIN_SAMPLE,
    }


async def _group(session: AsyncSession, session_id: str, column: Any) -> list[dict[str, Any]]:
    """Aggregate closed trades by one column, computed entirely in the database."""
    rows = (
        await session.execute(
            select(
                column.label("key"),
                func.count().label("trades"),
                func.coalesce(func.sum(ClosedTradeRecord.gross_pnl), 0).label("gross"),
                func.coalesce(func.sum(ClosedTradeRecord.fees), 0).label("fees"),
                func.coalesce(func.sum(ClosedTradeRecord.net_pnl), 0).label("net"),
                func.count().filter(ClosedTradeRecord.net_pnl > 0).label("wins"),
                func.coalesce(
                    func.sum(ClosedTradeRecord.net_pnl).filter(ClosedTradeRecord.net_pnl > 0), 0
                ).label("gross_profit"),
                func.coalesce(
                    func.sum(ClosedTradeRecord.net_pnl).filter(ClosedTradeRecord.net_pnl < 0), 0
                ).label("gross_loss"),
                func.max(ClosedTradeRecord.net_pnl).label("best"),
                func.min(ClosedTradeRecord.net_pnl).label("worst"),
            )
            .where(ClosedTradeRecord.session_id == session_id)
            .group_by(column)
            .order_by(func.coalesce(func.sum(ClosedTradeRecord.net_pnl), 0).desc())
        )
    ).all()

    results: list[dict[str, Any]] = []
    for row in rows:
        trades = int(row.trades)
        gross_loss = abs(_dec(row.gross_loss))
        gross_profit = _dec(row.gross_profit)
        net = _dec(row.net)
        gross = _dec(row.gross)
        fees = _dec(row.fees)
        results.append(
            {
                "key": str(getattr(row.key, "value", row.key)) if row.key is not None else None,
                "trades": trades,
                "gross_pnl": str(gross),
                "fees": str(fees),
                "net_pnl": str(net),
                "wins": int(row.wins),
                "win_rate": str(safe_divide(Decimal(int(row.wins)), Decimal(trades))),
                "average_net_pnl": str(safe_divide(net, Decimal(trades))),
                "profit_factor": (
                    str(safe_divide(gross_profit, gross_loss)) if gross_loss > ZERO else None
                ),
                "fee_share_of_gross": (
                    str(safe_divide(fees, abs(gross))) if gross != ZERO else None
                ),
                "best": str(_dec(row.best)) if row.best is not None else None,
                "worst": str(_dec(row.worst)) if row.worst is not None else None,
                # The client renders INSUFFICIENT SAMPLE from this rather than deciding a
                # threshold of its own, so the caveat cannot drift between panels.
                "reliable": trades >= MIN_SAMPLE,
            }
        )
    return results


async def fee_analysis(session: AsyncSession, session_id: str) -> dict[str, Any]:
    """What trading this session cost, beside what it earned.

    Args:
        session: An open read session.
        session_id: The session to analyse.

    Returns:
        A JSON-safe mapping of fee totals and ratios.

    """
    totals = (await session.execute(_trade_aggregate(session_id))).one()
    trades = int(totals.trades)
    fees = _dec(totals.fees)
    gross = _dec(totals.gross)
    net = _dec(totals.net)
    notional = (
        await session.execute(
            select(
                func.coalesce(
                    func.sum(ClosedTradeRecord.quantity * ClosedTradeRecord.entry_price), 0
                )
            ).where(ClosedTradeRecord.session_id == session_id)
        )
    ).scalar_one()

    return {
        "total_fees": str(fees),
        "gross_realized_pnl": str(gross),
        "net_realized_pnl": str(net),
        "closed_trades": trades,
        "average_fee_per_trade": str(safe_divide(fees, Decimal(trades))) if trades else None,
        "total_entry_notional": str(_dec(notional)),
        "average_fee_pct_of_notional": (
            str(safe_divide(fees, _dec(notional))) if _dec(notional) > ZERO else None
        ),
        # The headline ratio. Above 1.0 the strategy is gross-profitable and net-losing
        # purely on cost, which is a different problem from a strategy that is simply wrong.
        "fee_to_gross_ratio": str(safe_divide(fees, abs(gross))) if gross != ZERO else None,
        "fees_exceed_gross_profit": gross > ZERO and fees > gross,
        "entry_fees": None,
        "exit_fees": None,
        "not_recorded": {
            "entry_fees": "closed_trades stores one combined fee per round-trip",
            "exit_fees": "closed_trades stores one combined fee per round-trip",
        },
    }


def _asset_class(symbol: str) -> str:
    """Classify a symbol using the engine's own taxonomy.

    The engine has exactly six asset classes and this must agree with them, because a
    dashboard that invents its own grouping will disagree with the risk limits and the
    strategy-family gates that are applied per class. Meme is deliberately *not* inferred
    here: membership is decided at runtime from a curated list plus an eligibility scan,
    and only the engine's own log knows which coins passed. Callers that have those facts
    overlay them; this function's crypto default matches the engine's own fallback.
    """
    root = symbol.split("/", maxsplit=1)[0].upper().removeprefix("1000").removeprefix("10000")
    if root in ENERGY_ROOTS:
        return "energy"
    # Only the explicit metal codes. The engine also treats any X-prefixed root as a metal,
    # but it does so *within* the venue's ``commodity`` tag, which is not available here —
    # applying that rule unconditionally classifies XRP as a precious metal, and the
    # dashboard then reports a metals desk the engine is not trading.
    if root in METAL_ROOTS:
        return "metal"
    if root in INDEX_ETF_ROOTS:
        return "index"
    return "crypto"


def _dec(value: Any) -> Decimal:
    """Coerce a database aggregate to ``Decimal``."""
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 with an explicit UTC offset, or ``None``."""
    if value is None:
        return None
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).isoformat()


__all__ = [
    "MAX_CHART_POINTS",
    "MIN_SAMPLE",
    "PNL_PERIODS",
    "RANGES",
    "SessionRef",
    "attribution",
    "data_coverage",
    "equity_series",
    "fee_analysis",
    "performance",
    "resolve_session",
    "trade_ledger",
]
