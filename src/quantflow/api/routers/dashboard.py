"""Endpoints for the trading operations console.

Design rules, each of which exists because the previous dashboard broke one of them:

**Four sources, never blended.** Venue account state, QuantFlow session state, portfolio
valuation and trading performance appear in separate blocks with separate timestamps. A
figure derived from one is never presented beside a figure from another as though they were
comparable, and no field sums across them.

**No total without a unit.** The account endpoint this replaces added USDT, USDC, BTC and
ETH into a single "total balance". Nothing here produces a cross-asset total unless every
holding could be priced, and any such total is labelled in USDT with its method and the
timestamp of the prices used.

**Cheap and bounded.** Every figure is either a SQL aggregate over an indexed column or a
cached venue read with a hard deadline. The dashboard polls; it must never become a load
source against the database the trading engine depends on, and a slow venue must never
turn into a hanging request.

**Degrade, never disappear.** Each block reports its own availability, age and error. One
unreachable dependency costs one panel.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query

from quantflow.api.dashboard import status as status_module
from quantflow.api.dashboard.cache import (
    VENUE_DEADLINE_SECONDS,
    Cached,
    ResilientCache,
    freshness_block,
)
from quantflow.api.dashboard.decisions import (
    Decision,
    DecisionLog,
    EngineFacts,
    decision_feed_key,
    parse_feed,
    since,
    summarise,
)
from quantflow.api.dashboard.session_state import (
    RANGES,
    SessionRef,
    attribution,
    attribution_coverage,
    cumulative_pnl,
    data_coverage,
    equity_series,
    execution_quality,
    fee_analysis,
    performance,
    pnl_by_period,
    resolve_session,
    trade_ledger,
)
from quantflow.api.dashboard.valuation import order_rows, position_rows, value_balances
from quantflow.api.deps import DatabaseDep, StateDep
from quantflow.core.clock import utc_now
from quantflow.core.errors import NotFoundError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.live.heartbeat import Heartbeat, assess_engine, heartbeat_key
from quantflow.persistence.database import Database
from quantflow.risk.exposure import resting_entry_notional

logger = get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

#: Venue state is cached for slightly less than the dashboard's poll interval, so a refresh
#: normally sees a fresh read while several browser tabs share one venue round trip.
#: Bounds how fresh a pushed snapshot can be. Matched to the websocket's 2s broadcast so a
#: push always carries a reading no older than one interval; a longer TTL would make the
#: socket announce the same numbers twice and call it live.
VENUE_TTL_SECONDS = 2.0

#: Decision-log parsing is incremental, so this only bounds how often the file is stat-ed.
DECISION_TTL_SECONDS = 5.0

#: A venue reading older than this is reported as stale rather than presented as current.
VENUE_STALE_AFTER_SECONDS = 60.0

_venue_cache: ResilientCache[dict[str, Any]] = ResilientCache(VENUE_TTL_SECONDS, name="venue")
_decision_cache: ResilientCache[list[Decision]] = ResilientCache(
    DECISION_TTL_SECONDS, name="decisions"
)
_decision_log: DecisionLog | None = None
_engine_facts: EngineFacts | None = None


def _log_path() -> Path:
    """Where the engine writes its structured log.

    Overridable so a deployment that puts the log elsewhere does not need a code change,
    and so tests can point at a fixture.
    """
    return Path(os.environ.get("QF_DASHBOARD_BOT_LOG", "scratchpad/bot.log"))


class _MarkFromPositions:
    """Prices a symbol from the venue's own position rows.

    Only needed for an order carrying no limit price. A market order that is still resting
    is rare, and valuing it at the mark of a position in the same symbol is closer than
    dropping it from the exposure total entirely.
    """

    __slots__ = ("_marks",)

    def __init__(self, positions: list[dict[str, Any]]) -> None:
        self._marks = {
            str(row.get("symbol") or "").split(":")[0]: _decimal_or_zero(row.get("mark_price"))
            for row in positions
        }

    def mark_price(self, symbol: Any) -> Decimal | None:
        """Mark for ``symbol``, or ``None`` when no position prices it."""
        value = self._marks.get(str(symbol).split(":")[0])
        return value if value and value > ZERO else None


def _publish_facts(facts: EngineFacts) -> None:
    """Adopt the engine's own account of itself in place of the parsed-file one."""
    global _engine_facts  # noqa: PLW0603 — one published view per process.
    _engine_facts = facts


def _facts() -> EngineFacts:
    """What the engine reports about itself, preferring what it published."""
    return _engine_facts if _engine_facts is not None else _decisions_reader().facts()


def _decisions_reader() -> DecisionLog:
    """The process-wide incremental log reader."""
    global _decision_log  # noqa: PLW0603 — one reader per process holds the read offset.
    path = _log_path()
    if _decision_log is None or _decision_log.path != path:
        _decision_log = DecisionLog(path)
    return _decision_log


async def _read_decisions(state: Any, session_id: str | None) -> list[Decision]:
    """Decisions for this session, preferring the feed the engine publishes about itself.

    Reading the log here was the defect. This process is a container and the log is a host
    file reached through a bind mount, whose view of a large continuously-appended file goes
    stale: on 2026-08-16, with the log at 576 MB, the container was 145 lines and fifteen
    minutes behind — the engine had selected, sized and submitted an order at 12:45 while
    this endpoint reported "no decisions found in the log tail" and the console showed
    STARTING.

    So the engine parses its own log, on the host where its view is current, and publishes
    to Redis. An empty published feed is authoritative — it means the engine has decided
    nothing yet — and only a *missing* feed falls back to the file, which keeps an engine
    too old to publish readable rather than blank.
    """
    if session_id:
        try:
            raw = await state.cache.get(decision_feed_key(session_id))
        except Exception as exc:
            logger.info("dashboard.decision_feed_unavailable", error=str(exc)[:160])
        else:
            published = parse_feed(raw)
            if published is not None:
                decisions, facts = published
                if facts is not None:
                    # Facts describe the *running* engine, so they must come from the same
                    # source as the decisions. Left to the file they reported the previous
                    # run's start time beside this run's decisions.
                    _publish_facts(facts)
                return decisions

    # File reads are blocking. On the first refresh this touches tens of megabytes, which is
    # long enough to stall every other request on the loop if done inline.
    return await asyncio.to_thread(_decisions_reader().refresh)


async def _cached_decisions(state: Any, session_id: str | None) -> Cached[list[Decision]]:
    """Recent engine decisions, cached."""

    async def read() -> list[Decision]:
        return await _read_decisions(state, session_id)

    return await _decision_cache.get(read)


async def _read_venue(state: StateDep) -> dict[str, Any]:
    """Read the account, positions and working orders from the venue in one pass.

    Assembled together so the dashboard cannot show a balance from one instant beside
    positions from another — a mismatch that reads as a PnL error when it is a timing
    artefact.
    """
    # Reconnects on demand when the startup handshake failed, so a venue that was briefly
    # unreachable at boot does not leave this panel dead until somebody restarts the API.
    gateway = await state.ensure_gateway()
    if gateway is None:
        raise RuntimeError(
            "no exchange gateway is connected; the API could not authenticate to the venue"
        )

    balances = await gateway.fetch_balances()
    account = await value_balances(gateway, balances)

    positions: list[dict[str, Any]] = []
    unrealized = None
    position_error: str | None = None
    # Positions are not part of the gateway protocol — a spot venue has no such endpoint —
    # so the capability is probed rather than assumed. An account with no positions
    # endpoint reports that fact; it never renders as "no open positions", which would be
    # a claim the API is in no position to make.
    # Fetched before positions so each position can be matched to the resting orders that
    # protect it — with a maker take-profit the venue holds them as separate reduce-only
    # orders rather than on the position row.
    open_orders = await gateway.fetch_open_orders()

    fetch_positions = getattr(gateway, "fetch_positions", None)
    if fetch_positions is None:
        position_error = "this gateway exposes no positions endpoint (spot account)"
    else:
        try:
            positions, total = position_rows(await fetch_positions(), open_orders)
            unrealized = str(total)
        except Exception as exc:
            position_error = str(exc)
            logger.info("dashboard.positions_unavailable", error=str(exc))

    orders = order_rows(open_orders)

    # Positions, resting orders and their sum reported SEPARATELY. A resting order is
    # committed exposure — it becomes a position with no further decision — but it is not a
    # filled position, and collapsing the two into one number hides which is which. The
    # engine refuses orders on the total, so the operator has to be able to see the total
    # and its parts.
    resting = resting_entry_notional(open_orders, _MarkFromPositions(positions))
    resting_total = sum(resting.values(), ZERO)
    open_total = sum((_decimal_or_zero(row.get("notional_usdt")) for row in positions), ZERO)
    exposure = {
        "open_position_notional_usdt": str(open_total),
        "resting_order_notional_usdt": str(resting_total),
        "total_reserved_exposure_usdt": str(open_total + resting_total),
        "resting_by_symbol": {name: str(value) for name, value in resting.items()},
        "basis": (
            "open is the summed face value of filled positions; resting is unfilled entry "
            "orders, which the risk engine counts as committed because they become "
            "positions without a further decision; reduce-only protective orders are "
            "excluded because they remove exposure rather than add it"
        ),
    }

    return {
        "venue": gateway.name,
        "network": getattr(gateway, "network", "testnet" if gateway.is_testnet else "mainnet"),
        "authenticated": bool(getattr(gateway, "supports_trading", False)),
        "exposure": exposure,
        "account": account,
        "positions": positions,
        "position_count": len(positions),
        "position_error": position_error,
        # Marked-to-market by the venue itself, so it is the venue's number rather than
        # one this process derived from a stale mark.
        "unrealized_pnl": unrealized,
        "open_orders": orders,
        "open_order_count": len(orders),
    }


def _deployed(venue: dict[str, Any]) -> dict[str, Any]:
    """How much capital is actually in trades.

    Two different figures, both reported and each labelled, because they differ by the
    leverage multiple and confusing them misstates exposure by that factor:

    ``notional``
        The full face value of the open positions — what the account is exposed to.
    ``margin``
        The collateral the venue has actually set aside for them — what is unavailable to
        the next trade.

    Derived from the venue's own position rows, so an account with no positions reports
    zero deployed rather than an absent panel.
    """
    notional = ZERO
    margin = ZERO
    for position in venue.get("positions") or []:
        value = _decimal_or_zero(position.get("notional_usdt"))
        notional += abs(value)
        leverage = _decimal_or_zero(position.get("leverage"))
        # Initial margin is notional/leverage. A missing or zero leverage would divide by
        # zero, so the notional stands in — an overstatement of margin, never an
        # understatement of how much capital is committed.
        margin += abs(value / leverage) if leverage > ZERO else abs(value)
    return {
        "notional_usdt": str(notional),
        "margin_usdt": str(margin),
        "position_count": venue.get("position_count", 0),
        "basis": (
            "notional is the summed face value of open positions; margin is notional "
            "divided by each position's leverage, which is the collateral the venue has "
            "set aside. Both read from the venue's position rows."
        ),
    }


def _decimal_or_zero(value: Any) -> Decimal:
    """Parse a wire decimal string, treating anything unparseable as zero."""
    if value is None or value == "":
        return ZERO
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return ZERO


async def _venue(state: StateDep) -> Cached[dict[str, Any]]:
    """Cached venue state, bounded by a hard deadline."""
    return await _venue_cache.get(
        lambda: _read_venue(state), deadline_seconds=VENUE_DEADLINE_SECONDS
    )


async def _engine_health(state: StateDep, session_id: str) -> dict[str, Any]:
    """The engine's own report of whether it is running.

    Read from Redis rather than derived from the decision log, because the two answer
    different questions and only one of them is about *now*. A fully-invested book records
    no new entry decisions, and the log is seen through a bind mount that goes stale — both
    produce silence from a perfectly healthy engine, and that silence was twice published
    as ENGINE ERROR.

    A cache that cannot be read yields UNKNOWN, never STOPPED: not knowing whether the
    engine is alive is a different statement from knowing it is dead.
    """
    if state.cache is None:
        return assess_engine(None, now=utc_now()).to_dict()
    try:
        raw = await state.cache.get(heartbeat_key(session_id))
    except Exception as exc:
        logger.info("dashboard.heartbeat_unavailable", error=str(exc)[:160])
        raw = None
    return assess_engine(Heartbeat.from_dict(raw), now=utc_now()).to_dict()


async def _session_ref(database: Database, session_id: str | None) -> SessionRef:
    """Resolve the session to report on.

    Raises:
        NotFoundError: when no session has ever run.

    """
    async with database.read_session() as session:
        reference = await resolve_session(session, session_id=session_id)
    if reference is None:
        raise NotFoundError("no trading session has ever run")
    return reference


async def venue_snapshot(state: Any) -> dict[str, Any]:
    """The live venue block, for pushing over the websocket.

    Shares the cache with the REST endpoints, so a pushed snapshot and a polled one can
    never disagree — and every field carries the freshness of the read it came from.
    """
    cached = await _venue(state)
    venue = cached.value
    freshness = freshness_block(cached, label="bybit venue account read")
    if venue is None:
        return {"available": False, "freshness": freshness, "error": cached.error}
    return {
        "available": True,
        "account": venue["account"],
        "positions": venue["positions"],
        "position_count": venue["position_count"],
        "open_orders": venue["open_orders"],
        "open_order_count": venue["open_order_count"],
        "unrealized_pnl": venue["unrealized_pnl"],
        "deployed": _deployed(venue),
        "freshness": freshness,
    }


@router.get("/summary", summary="Everything the header and status bar need")
async def get_summary(
    state: StateDep,
    database: DatabaseDep,
    session_id: str | None = None,
) -> dict[str, Any]:
    """The polled endpoint: session identity, performance, venue state and engine status.

    Every block carries its own freshness. A block whose source failed reports
    ``available: false`` with the age of the last good reading, so the client can say
    "last successful update 14:32:07" rather than going blank.
    """
    reference = await _session_ref(database, session_id)

    venue_cached, decisions_cached, engine_health = await asyncio.gather(
        _venue(state),
        _cached_decisions(state, reference.session_id),
        _engine_health(state, reference.session_id),
    )
    venue = venue_cached.value
    # Scoped to this session. The log spans every run that has ever executed, so an
    # unfiltered tail reports a previous session's decisions as though they were current.
    decisions = since(decisions_cached.value or [], reference.started_at)
    facts = _facts()

    async with database.read_session() as session:
        metrics = await performance(session, reference)
        fees = await fee_analysis(session, reference.session_id)

    risk = state.risk
    kill_engaged = False
    halted = False
    if risk is not None:
        try:
            await risk.refresh_kill_switch()
            kill_engaged = risk.kill_switch.engaged
            halted = risk.is_halted
        except Exception as exc:
            logger.warning("dashboard.risk_unavailable", error=str(exc))

    last_snapshot = metrics["session_equity"]["latest_at"]
    derived = status_module.derive(
        venue_available=venue is not None,
        venue_error=venue_cached.error,
        kill_switch_engaged=kill_engaged,
        trading_halted=halted,
        session_running=reference.is_running,
        session_status=reference.status,
        open_position_count=venue["position_count"] if venue else None,
        last_snapshot_at=_parse_iso(last_snapshot),
        decisions=decisions,
        recent_order_rejections=sum(
            1 for item in decisions[-40:] if item.outcome == "RISK_BLOCKED"
        ),
    )

    venue_positions = venue["position_count"] if venue else None
    venue_orders = venue["open_order_count"] if venue else None
    book = metrics["session_book"]

    return {
        "generated_at": utc_now().isoformat(),
        "session": reference.to_dict(),
        "status": derived.to_dict(),
        # --- Source 1: the venue. Authoritative for balances and what is really open. ---
        "venue": {
            **(venue or {}),
            # "IN TRADES" on the executive header. Computed from the venue's own position
            # rows so it can never disagree with the position list rendered beside it.
            "deployed": _deployed(venue) if venue else None,
            "freshness": freshness_block(venue_cached, label="bybit venue account read"),
        },
        # --- Source 2 and 4: QuantFlow's own session state and realised performance. ---
        **metrics,
        "fees": fees,
        # The two stores disagree on this deployment, and the disagreement is itself the
        # finding: silently preferring one would hide a reconciliation failure.
        "book_reconciliation": {
            "venue_open_positions": venue_positions,
            "database_open_positions": book["open_positions"],
            "venue_open_orders": venue_orders,
            "database_open_orders": book["open_orders"],
            "positions_match": venue_positions == book["open_positions"],
            "orders_match": venue_orders == book["open_orders"],
            "authority": "the venue is authoritative; the database is QuantFlow's record",
        },
        # Engine health and trading status are deliberately separate fields answering
        # separate questions. "The engine is running" and "the engine is not entering
        # trades" are simultaneously true most of the time, and collapsing them into one
        # value is what reported a healthy engine as failed.
        "engine_health": engine_health,
        "risk": {
            "kill_switch_engaged": kill_engaged,
            "trading_halted": halted,
            "available": risk is not None,
        },
        "decisions": {
            **summarise(decisions),
            "freshness": freshness_block(decisions_cached, label=str(_log_path())),
        },
        # What the running process said about itself when it started. Taken from the
        # engine's own log rather than from configuration on disk: after a flag changes,
        # the file describes the next run and only the log describes this one.
        "engine": {
            **facts.to_dict(),
            # No process writes a pid file, so the API — which runs in a container and
            # cannot see host processes anyway — genuinely cannot report one.
            "pid": None,
            "pid_note": (
                "the engine writes no pid file and the API cannot observe host processes; "
                "identify it with: pgrep -f scripts/run_demo_bot.py"
            ),
            "supervisor": _supervisor_history(),
        },
    }


@router.get("/equity", summary="Equity curve, running peak and drawdown")
async def get_equity(
    database: DatabaseDep,
    window: Annotated[str, Query(pattern="^(1H|6H|24H|7D|30D|ALL)$")] = "ALL",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Persisted equity snapshots for one range, with the drawdown derived from them.

    Points are never fabricated: when the window reaches back further than the stored
    history the response reports the real extent instead of a line starting at the edge.
    """
    reference = await _session_ref(database, session_id)
    async with database.read_session() as session:
        series = await equity_series(session, reference.session_id, window=window)
    return {
        "session_id": reference.session_id,
        "ranges": list(RANGES),
        **series,
    }


@router.get("/trades", summary="Closed-trade ledger")
async def get_trades(
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Every closed round-trip this session, newest first.

    Columns the schema does not hold are declared in ``not_recorded`` rather than filled
    with zeros, so the client can render ``NOT RECORDED`` and mean it.
    """
    reference = await _session_ref(database, session_id)
    async with database.read_session() as session:
        ledger = await trade_ledger(session, reference.session_id, limit=limit, offset=offset)
        coverage = await data_coverage(session, reference.session_id)
    return {"session_id": reference.session_id, "coverage": coverage, **ledger}


@router.get("/pnl", summary="Profit and loss over each period, plus one cumulative series")
async def get_pnl(
    database: DatabaseDep,
    window: Annotated[str, Query(pattern="^(1H|6H|24H|7D|30D|ALL)$")] = "ALL",
    session_id: str | None = None,
) -> dict[str, Any]:
    """The one profit-and-loss panel: period figures and a single cumulative line.

    Gross profit, gross loss and fees are reported separately for every period rather than
    only as a net figure, because a book that is up on gross and down on net has a fee
    problem, and a single net number hides exactly that. On this deployment it is hiding a
    fee bill several times the gross edge.
    """
    reference = await _session_ref(database, session_id)
    async with database.read_session() as session:
        periods = await pnl_by_period(session, reference.session_id)
        series = await cumulative_pnl(session, reference.session_id, window=window)
    return {
        "session_id": reference.session_id,
        "ranges": list(RANGES),
        **periods,
        "cumulative": series,
    }


@router.get("/positions", summary="Open positions, as the venue reports them")
async def get_positions(state: StateDep) -> dict[str, Any]:
    """What is open right now, read from the venue rather than from our own book.

    The venue is authoritative: a position that exists there and not in the database is
    real exposure, and the opposite is a reconciliation failure. Rendering our own record
    here would show a position that had already been closed, which is the bug this panel
    exists to make impossible.
    """
    cached = await _venue(state)
    venue = cached.value
    if venue is None:
        return {
            "available": False,
            "positions": [],
            "position_count": 0,
            "error": cached.error,
            "freshness": freshness_block(cached, label="bybit venue account read"),
        }
    return {
        "available": True,
        "positions": venue["positions"],
        "position_count": venue["position_count"],
        "unrealized_pnl": venue["unrealized_pnl"],
        "deployed": _deployed(venue),
        "error": venue["position_error"],
        "freshness": freshness_block(cached, label="bybit venue account read"),
        "not_recorded": [
            # Named explicitly so the table renders NOT RECORDED instead of a blank that
            # reads as "no stage" or "no target".
            "profit_stage (held in the engine process; not exposed to the API)",
        ],
    }


@router.get("/orders", summary="Working orders, as the venue reports them")
async def get_orders(state: StateDep) -> dict[str, Any]:
    """Orders resting at the venue, including protective stops and targets.

    Each row carries ``purpose``, so a stop and a take-profit on the same position are
    distinguishable — without it a correct protective bracket reads as a duplicated exit.
    """
    cached = await _venue(state)
    venue = cached.value
    if venue is None:
        return {
            "available": False,
            "orders": [],
            "order_count": 0,
            "error": cached.error,
            "freshness": freshness_block(cached, label="bybit venue account read"),
        }
    return {
        "available": True,
        "orders": venue["open_orders"],
        "order_count": venue["open_order_count"],
        "freshness": freshness_block(cached, label="bybit venue account read"),
    }


@router.get("/analytics", summary="Attribution by strategy, symbol, side and exit reason")
async def get_analytics(
    database: DatabaseDep,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Grouped performance, each group carrying its own sample size.

    Groups below the reliability threshold are flagged rather than dropped: a strategy with
    two trades is worth seeing and is never worth acting on, and the client marks it
    ``INSUFFICIENT SAMPLE``.
    """
    reference = await _session_ref(database, session_id)
    async with database.read_session() as session:
        groups = await attribution(session, reference.session_id)
        execution = await execution_quality(session, reference.session_id)
        coverage = await attribution_coverage(session, reference.session_id)
    return {
        "session_id": reference.session_id,
        **groups,
        "execution_quality": execution,
        "attribution_coverage": coverage,
    }


@router.get("/decisions", summary="Recent decision-engine activity")
async def get_decisions(
    state: StateDep,
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Bars *this session* evaluated, and what it decided.

    This is the panel that answers "why is the bot flat". Source is the orchestrator's own
    structured log, read incrementally — and that log is one append-only file shared by
    every run, whose lines carry a redacted session id. Unscoped, it reports the tail of
    the file, which is whatever ran before. Two minutes into a fresh session it showed
    500 evaluated and 149 selected for a session that had decided nothing.

    Scoped by the session's start time, the only discriminator the log leaves available.
    """
    reference = await _session_ref(database, session_id)
    cached = await _cached_decisions(state, reference.session_id)
    decisions = since(cached.value or [], reference.started_at)
    recent = decisions[-limit:]
    return {
        "session_id": reference.session_id,
        "session_started_at": _iso_or_none(reference.started_at),
        "decisions": [item.to_dict() for item in reversed(recent)],
        "summary": summarise(decisions),
        "source": str(_log_path()),
        "freshness": freshness_block(cached, label=str(_log_path())),
        "not_recorded": {
            "expected_edge": (
                "the orchestrator logs a unit-free component score, not an expected edge in USDT"
            ),
            "estimated_cost": (
                "the orchestrator logs a unit-free cost score, not an estimated cost in USDT"
            ),
        },
    }


@router.get("/freshness", summary="How current every source on the page is")
async def get_freshness(
    state: StateDep,
    database: DatabaseDep,
    session_id: str | None = None,
) -> dict[str, Any]:
    """When each source last produced data, and whether that is recent enough.

    A dashboard whose numbers stopped updating looks exactly like one whose numbers are
    not changing. This endpoint is what tells them apart.
    """
    reference = await _session_ref(database, session_id)
    venue_cached, decisions_cached, engine_health = await asyncio.gather(
        _venue(state),
        _cached_decisions(state, reference.session_id),
        _engine_health(state, reference.session_id),
    )
    decisions = since(decisions_cached.value or [], reference.started_at)

    from sqlalchemy import func, select

    from quantflow.persistence.models import (
        CandleRecord,
        EquitySnapshotRecord,
        OrderRecord,
    )

    async with database.read_session() as session:
        last_snapshot = (
            await session.execute(
                select(func.max(EquitySnapshotRecord.timestamp)).where(
                    EquitySnapshotRecord.session_id == reference.session_id
                )
            )
        ).scalar_one_or_none()
        last_order = (
            await session.execute(
                select(func.max(OrderRecord.created_at)).where(
                    OrderRecord.session_id == reference.session_id
                )
            )
        ).scalar_one_or_none()
        # Restricted to this session's symbols and timeframe and covered by
        # ix_candles_symbol_tf_time. The unrestricted form of this question is a full
        # aggregate over a million rows, which is what previously took the API down.
        candle_rows = (
            await session.execute(
                select(CandleRecord.symbol, func.max(CandleRecord.open_time))
                .where(
                    CandleRecord.symbol.in_(list(reference.symbols) or [""]),
                    CandleRecord.timeframe == reference.timeframe,
                )
                .group_by(CandleRecord.symbol)
            )
        ).all()

    now = utc_now()
    candles = [
        {
            "symbol": symbol,
            "last_open_time": _iso_or_none(latest),
            "age_seconds": (now - latest).total_seconds() if latest is not None else None,
        }
        for symbol, latest in candle_rows
    ]
    newest_candle = max((row[1] for row in candle_rows if row[1] is not None), default=None)

    venue_age = venue_cached.age_seconds
    snapshot_age = (now - last_snapshot).total_seconds() if last_snapshot else None

    # A 15-minute engine writes a snapshot per bar, so "stale" has to be measured against
    # the bar, not against the poll interval.
    engine_stalled = (
        snapshot_age is not None and snapshot_age > status_module.SNAPSHOT_SILENCE.total_seconds()
    )
    venue_stalled = venue_age is not None and venue_age > VENUE_STALE_AFTER_SECONDS
    # Deliberately excludes the candle archive: see candle_note below.
    if venue_cached.value is None:
        state_label = "VENUE DISCONNECTED"
    elif engine_stalled or venue_stalled:
        state_label = "DATA STALE"
    else:
        state_label = "DATA FRESH"

    return {
        "generated_at": now.isoformat(),
        "state": state_label,
        # The engine's own report, kept beside the source ages it is meant to be read
        # against. Health is the engine's answer; the ages below are everyone else's.
        "engine_health": engine_health,
        "session_id": reference.session_id,
        "timeframe": reference.timeframe,
        "venue_sync": freshness_block(venue_cached, label="bybit venue account read"),
        "engine_log": freshness_block(decisions_cached, label=str(_log_path())),
        "last_decision_at": decisions[-1].timestamp.isoformat() if decisions else None,
        "last_equity_snapshot_at": _iso_or_none(last_snapshot),
        "last_equity_snapshot_age_seconds": snapshot_age,
        "last_order_at": _iso_or_none(last_order),
        "last_candle_at": _iso_or_none(newest_candle),
        "candles": sorted(candles, key=lambda row: str(row["symbol"])),
        # A vital distinction. The engine consumes candles from the venue's websocket and
        # does not write them back during a live run, so this table is a *download*
        # archive, not a picture of what the engine is receiving. Reported plainly because
        # an operator who reads a two-day-old bar as "the engine has no market data" will
        # go looking for a fault that is not there — and because the reverse mistake,
        # treating a genuinely dead feed as fine, is worse.
        "candle_note": (
            "these are stored candles from the download archive, not the engine's live "
            "stream; a live run reads bars from the venue websocket and does not persist "
            "them, so staleness here does not imply the engine has lost market data — "
            "judge that from the last decision instead"
        ),
        "reconciliation": {
            # The engine reconciles against the venue; the dashboard reports the venue read
            # it performed, which is the only reconciliation timestamp it can observe.
            "last_venue_read_at": (
                venue_cached.fetched_at.isoformat() if venue_cached.fetched_at else None
            ),
            "note": (
                "the API observes its own venue reads; the engine's internal reconciliation "
                "pass is not exposed as a timestamp"
            ),
        },
    }


#: Every asset class the engine can trade, with what is known about each independently of
#: whether this session enabled it. The names are the engine's own
#: :class:`quantflow.universe.assets.AssetClass` values — a dashboard that invented its own
#: grouping would disagree with the per-class risk limits and strategy-family gates.
ASSET_CLASS_CATALOGUE: tuple[tuple[str, str], ...] = (
    ("crypto", "Major and mid-cap digital assets"),
    ("meme", "Meme coins, selected at runtime by an eligibility scan"),
    ("metal", "Precious metals (XAU, XAG, XPT, XPD)"),
    ("energy", "Crude oil benchmarks and relatives"),
    ("equity", "Single-name equities"),
    ("index", "Index and sector ETFs"),
)


@router.get("/asset-classes", summary="What the running engine actually trades")
async def get_asset_classes(
    state: StateDep,
    database: DatabaseDep,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Per asset class: whether it is live, and if not, exactly why not.

    Three states are kept distinct, because they call for three different responses:

    ``ACTIVE``
        The running engine is subscribed to symbols in this class and evaluating them.
    ``IMPLEMENTED, BLOCKED``
        The engine tried and the venue refused — here, a product agreement nobody has
        signed. Enabling the flag again will not help.
    ``IMPLEMENTED, NOT ENABLED``
        Supported, switched off in this run's configuration.

    Marking a merely-supported class ACTIVE would tell an operator the account is
    diversified across markets it is not trading at all.
    """
    reference = await _session_ref(database, session_id)
    venue_cached, _ = await asyncio.gather(
        _venue(state), _cached_decisions(state, reference.session_id)
    )
    venue = venue_cached.value
    facts = _facts()

    from quantflow.api.dashboard.session_state import _asset_class

    # The engine's log is the authority on which symbols this process actually has, and on
    # which of them are memes — meme membership is decided by a runtime eligibility scan,
    # not by anything derivable from the ticker.
    live_symbols = list(facts.symbols) or list(reference.symbols)
    memes = set(facts.meme_symbols)
    blocked = set(facts.agreement_blocked)

    # The engine logs which class it assigned each symbol to, and that is the venue's own
    # `symbolType` answer rather than a second guess from the ticker — the only way to know
    # that SNDK is an equity and XRP is not a metal.
    declared: dict[str, str] = {
        symbol: name for name, items in facts.class_symbols.items() for symbol in items
    }

    def classify(symbol: str) -> str:
        if symbol in memes:
            return "meme"
        return declared.get(symbol) or _asset_class(symbol)

    by_class: dict[str, list[str]] = {name: [] for name, _ in ASSET_CLASS_CATALOGUE}
    for symbol in live_symbols:
        by_class.setdefault(classify(symbol), []).append(symbol)

    positions_by_class: dict[str, int] = {}
    if venue:
        for position in venue["positions"]:
            # Futures symbols arrive as BTC/USDT:USDT; the settlement suffix is not part of
            # the instrument for classification purposes.
            plain = str(position["symbol"]).split(":")[0]
            key = classify(plain)
            positions_by_class[key] = positions_by_class.get(key, 0) + 1

    # The engine names the class it quarantined, so that is used directly. Symbols are
    # kept as a fallback for refusals logged before the dedicated event existed, and
    # because a class can be inferred from a symbol but not the other way round.
    blocked_classes = {classify(symbol) for symbol in blocked}
    blocked_classes |= set(facts.agreement_blocked_classes)

    rows: list[dict[str, Any]] = []
    for name, description in ASSET_CLASS_CATALOGUE:
        symbols = sorted(by_class.get(name, []))
        seen = (
            facts.agreement_blocked_at.isoformat()
            if facts.agreement_blocked_at
            else "in this log window"
        )
        codes = ", ".join(facts.agreement_codes) or "110123/110125/110126"
        if symbols and name in blocked_classes:
            # Subscribed, streaming and evaluated, but the venue will not accept an order.
            # Reporting this as plain ACTIVE would claim the account is trading a market it
            # cannot; reporting it as NOT ENABLED would blame configuration for a refusal
            # that is entirely the venue's.
            state_label = "ACTIVE, ORDERS BLOCKED"
            reason = (
                "market data and strategy evaluation are live on these symbols, but the "
                f"venue refuses orders pending a signed product agreement (retCode {codes}, "
                f"seen {seen}). Sign it in the Bybit demo UI; no redeploy is needed."
            )
        elif symbols:
            state_label, reason = "ACTIVE", None
        elif name in blocked_classes:
            state_label = "IMPLEMENTED, BLOCKED"
            reason = (
                "the venue refused an order in this class pending a product agreement "
                f"(retCode {codes}, seen {seen})"
            )
        else:
            state_label = "IMPLEMENTED, NOT ENABLED"
            reason = "no symbols of this class are in the running engine's universe"
        rows.append(
            {
                "asset_class": name,
                "description": description,
                "state": state_label,
                "reason": reason,
                "symbols": symbols,
                "symbol_count": len(symbols),
                "open_positions": positions_by_class.get(name, 0),
                "data_live": bool(symbols) and venue is not None,
            }
        )

    # Forex is a real package in this codebase but is not part of the trading loop at all,
    # which is a different thing from a class the engine could trade today. Listing it as
    # merely "not enabled" beside the others would overstate how close it is to running.
    rows.append(
        {
            "asset_class": "forex",
            "description": "OANDA / MT5 brokers",
            "state": "IMPLEMENTED, NOT WIRED",
            "reason": (
                "the forex package is not imported by the live trading loop and no broker "
                "credentials are configured; it cannot trade in this deployment"
            ),
            "symbols": [],
            "symbol_count": 0,
            "open_positions": 0,
            "data_live": False,
        }
    )

    return {
        "session_id": reference.session_id,
        "timeframe": facts.timeframe or reference.timeframe,
        "asset_classes": rows,
        "symbol_count": len(live_symbols),
        "venue_available": venue is not None,
        "source": "the running engine's own startup log, plus live venue positions",
    }


def _supervisor_history(limit: int = 12) -> dict[str, Any]:
    """Recent restarts, read from the supervisor's log.

    The engine is run under a supervisor that restarts it, so "the engine is running" can
    be true of a process that started thirty seconds ago after being killed. A restart
    history turns an apparently healthy engine with an empty book into an explicable one.
    """
    path = Path(os.environ.get("QF_DASHBOARD_SUPERVISOR_LOG", "scratchpad/bot-supervisor.log"))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"available": False, "error": str(exc), "path": str(path), "events": []}

    events = [line.strip() for line in lines if line.strip().startswith("[")]
    exits = [line for line in events if "bot exited" in line]
    return {
        "available": True,
        "path": str(path),
        "events": events[-limit:],
        "restart_count": len([line for line in events if "starting bot" in line]),
        "exit_count": len(exits),
        # rc=137 is SIGKILL: on this host that has meant the OS reclaiming memory, not a
        # crash in the engine, and the two call for very different responses.
        "killed_count": len([line for line in exits if "rc=137" in line]),
    }


def _iso_or_none(value: Any) -> str | None:
    """ISO-8601 for a datetime, or ``None``."""
    return value.isoformat() if value is not None else None


def _parse_iso(value: Any) -> Any:
    """Parse an ISO timestamp emitted earlier in this request, tolerating ``None``."""
    if not isinstance(value, str):
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["router"]
