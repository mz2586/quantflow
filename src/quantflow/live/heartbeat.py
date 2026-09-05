"""What the engine is doing *now*, as distinct from what it last decided.

Two dashboard incidents produced this module, both the same mistake. The console inferred
"ENGINE ERROR" from the age of the newest strategy decision, and both times the engine was
alive and managing positions. The inference is wrong for two independent reasons:

* **A fully-invested book makes no entry decisions.** When every symbol already holds a
  position the orchestrator is never asked to choose one, so the decision feed goes quiet
  precisely when the engine is busiest.
* **The decision feed is not a reliable clock.** It is parsed from the engine's log file,
  which the API reads through a bind mount whose view of a large, rapidly-appended file
  goes stale within minutes — measured on this deployment, the same reader against the same
  path returned decisions on the host that the container never saw.

Liveness is therefore answered from a heartbeat the engine writes to Redis every loop.
Redis is a real network service reached over TCP, so a value read from it is current or the
read fails; there is no third state where it silently returns yesterday.

The heartbeat carries each loop's own timestamp rather than one overall "alive" flag,
because the loops fail independently. On 2026-08-14 the bar loop stopped at 19:00 while the
ticker loop ran for another nine hours: the process was alive, ticks were flowing, and no
bar was evaluated all night. Process liveness would have called that healthy. A per-loop
heartbeat calls it DEGRADED, which is the answer that would have woken somebody.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from quantflow.core.logging import get_logger

logger = get_logger(__name__)

#: Redis key prefix. One heartbeat per session, so two engines cannot overwrite each other.
HEARTBEAT_KEY_PREFIX = "heartbeat:engine"

#: How long a written heartbeat survives in Redis. Comfortably longer than
#: :data:`STALE_AFTER` so an expired key and a stale key are distinguishable: the first
#: means nobody has written for minutes, the second that the writer is falling behind.
HEARTBEAT_TTL_SECONDS = 900.0

#: How often the engine refreshes the heartbeat. Ticks arrive several times a second and
#: writing on each would make Redis a hot path for no gain.
HEARTBEAT_INTERVAL_SECONDS = 15.0

#: Beyond this with no heartbeat at all, the engine is not running its loops.
STALE_AFTER = timedelta(minutes=3)

#: Beyond this with no progress on one loop, that loop has stalled even though others run.
#:
#: Set against the trading timeframe rather than pulled from the air: on 15m bars a candle
#: loop that has not advanced in 45 minutes has missed three bars, which is past any
#: reconnect or backlog and into "this is not coming back on its own".
DEGRADED_AFTER = timedelta(minutes=45)


class EngineHealth(StrEnum):
    """Whether the engine is running. Never a statement about whether it is trading."""

    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


def heartbeat_key(session_id: str) -> str:
    """Redis key holding one session's heartbeat."""
    return f"{HEARTBEAT_KEY_PREFIX}:{session_id}"


#: Redis key prefix for the risk configuration the engine is *actually* enforcing.
#:
#: The API process loads its own settings, and they are not the engine's: the engine takes
#: its limits from the supervisor's environment. Measured 2026-08-17, ``/risk/status``
#: advertised ``max_position_pct 0.02`` and ``max_order_notional 5000`` while the running
#: engine enforced ``0.20`` and ``20000`` — a tenfold error on the two numbers an operator
#: is most likely to check before intervening.
#:
#: So the engine publishes what it enforces, and the dashboard reports that rather than its
#: own guess. Same reasoning as the heartbeat and the decision feed.
RISK_LIMITS_KEY_PREFIX = "risk_limits:engine"


#: Session-less alias for the limits the currently-running engine enforces.
#:
#: There is one live engine per deployment, and the dashboard wants "what is being enforced
#: right now" without first having to work out which session that is. The per-session key
#: remains the record; this is the pointer to the current one.
RISK_LIMITS_CURRENT_KEY = "risk_limits:current"

#: Redis key holding the thesis-cooldown state, so it survives a restart.
#:
#: The cooldown compares a new candidate against the score of the thesis that failed. Held
#: only in memory, that score is lost whenever the process restarts — and the engine then
#: enforces the cooldown with no way to clear it early, which is stricter than the rule as
#: designed. Seen live on 2026-08-17: BTC refused with "no score was recorded to compare
#: against" after a restart landed between the entry and the loss.
COOLDOWN_STATE_KEY = "cooldown_state:current"


def risk_limits_key(session_id: str) -> str:
    """Redis key holding the limits one session's engine is enforcing."""
    return f"{RISK_LIMITS_KEY_PREFIX}:{session_id}"


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """One engine's runtime state at the moment it last reported.

    Every loop reports its own last-progress timestamp. ``None`` means that loop has not
    run yet in this process, which is a normal state during startup and is deliberately not
    the same as "it ran a long time ago".
    """

    session_id: str
    pid: int
    written_at: datetime
    started_at: datetime | None = None
    last_tick_at: datetime | None = None
    last_candle_at: datetime | None = None
    last_reconcile_at: datetime | None = None
    last_decision_at: datetime | None = None
    open_positions: int | None = None
    symbols: tuple[str, ...] = ()
    #: Set when the engine is shutting down cleanly, so a planned stop is not read as a
    #: crash on the way out.
    stopped_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe wire form."""
        return {
            "session_id": self.session_id,
            "pid": self.pid,
            "written_at": self.written_at.isoformat(),
            "started_at": _iso(self.started_at),
            "last_tick_at": _iso(self.last_tick_at),
            "last_candle_at": _iso(self.last_candle_at),
            "last_reconcile_at": _iso(self.last_reconcile_at),
            "last_decision_at": _iso(self.last_decision_at),
            "open_positions": self.open_positions,
            "symbols": list(self.symbols),
            "stopped_at": _iso(self.stopped_at),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Heartbeat | None:
        """Parse a stored heartbeat, or ``None`` when it cannot be read.

        Never raises. A dashboard panel must not take the page down because a cache value
        is malformed, and a missing heartbeat is an ordinary condition rather than an error.
        """
        if not isinstance(payload, dict):
            return None
        try:
            written = _parse(payload.get("written_at"))
            if written is None:
                return None
            return cls(
                session_id=str(payload.get("session_id") or ""),
                pid=int(payload.get("pid") or 0),
                written_at=written,
                started_at=_parse(payload.get("started_at")),
                last_tick_at=_parse(payload.get("last_tick_at")),
                last_candle_at=_parse(payload.get("last_candle_at")),
                last_reconcile_at=_parse(payload.get("last_reconcile_at")),
                last_decision_at=_parse(payload.get("last_decision_at")),
                open_positions=(
                    int(payload["open_positions"])
                    if payload.get("open_positions") is not None
                    else None
                ),
                symbols=tuple(str(item) for item in (payload.get("symbols") or [])),
                stopped_at=_parse(payload.get("stopped_at")),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """The engine's health, with the timestamps it was derived from."""

    state: EngineHealth
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Wire form."""
        return {"state": self.state.value, "detail": self.detail, "evidence": self.evidence}


def assess_engine(heartbeat: Heartbeat | None, *, now: datetime) -> EngineStatus:
    """Decide whether the engine is running, from its heartbeat alone.

    Deliberately knows nothing about decisions, candidates, gates or positions. Whether the
    engine is *trading* is a separate question with a separate answer, and merging them is
    the defect this exists to prevent.

    Args:
        heartbeat: The engine's last report, or ``None`` if none could be read.
        now: The current time.

    Returns:
        The health state and the evidence behind it.

    """
    if heartbeat is None:
        return EngineStatus(
            EngineHealth.UNKNOWN,
            "no heartbeat has been published for this session; the engine may predate "
            "heartbeats, or the cache could not be read",
            {"heartbeat": None},
        )

    evidence: dict[str, Any] = {
        "pid": heartbeat.pid,
        "written_at": heartbeat.written_at.isoformat(),
        "started_at": _iso(heartbeat.started_at),
        "last_tick_at": _iso(heartbeat.last_tick_at),
        "last_candle_at": _iso(heartbeat.last_candle_at),
        "last_reconcile_at": _iso(heartbeat.last_reconcile_at),
        "last_decision_at": _iso(heartbeat.last_decision_at),
        "open_positions": heartbeat.open_positions,
        "heartbeat_age_seconds": (now - heartbeat.written_at).total_seconds(),
    }

    if heartbeat.stopped_at is not None:
        return EngineStatus(
            EngineHealth.STOPPED,
            f"the engine reported a clean stop at {heartbeat.stopped_at.isoformat()}",
            evidence,
        )

    age = now - heartbeat.written_at
    if age > STALE_AFTER:
        return EngineStatus(
            EngineHealth.STALE,
            f"the engine last reported {_age(age)} ago; it is not refreshing its heartbeat",
            evidence,
        )

    # Per-loop checks. The process is clearly alive to be writing at all, so anything here
    # is one loop having stopped while the others continue — the failure mode that looks
    # healthy from the outside and is the reason this is not a single alive flag.
    for label, moment in (
        ("candle loop", heartbeat.last_candle_at),
        ("reconciliation loop", heartbeat.last_reconcile_at),
    ):
        if moment is not None and now - moment > DEGRADED_AFTER:
            return EngineStatus(
                EngineHealth.DEGRADED,
                f"the {label} has not advanced in {_age(now - moment)} while the engine "
                "is otherwise alive",
                evidence,
            )

    return EngineStatus(
        EngineHealth.RUNNING,
        f"heartbeat {_age(age)} old; every loop is advancing",
        evidence,
    )


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _age(delta: timedelta) -> str:
    """Human-readable duration."""
    seconds = int(delta.total_seconds())
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s"
    if seconds < _SECONDS_PER_HOUR:
        return f"{seconds // _SECONDS_PER_MINUTE}m"
    hours, remainder = divmod(seconds, _SECONDS_PER_HOUR)
    return f"{hours}h{remainder // _SECONDS_PER_MINUTE:02d}m"


def _iso(value: datetime | None) -> str | None:
    """ISO-8601, or ``None``."""
    return value.isoformat() if value is not None else None


def _parse(value: Any) -> datetime | None:
    """Parse an ISO timestamp, or ``None`` when absent or malformed."""
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value)


__all__ = [
    "COOLDOWN_STATE_KEY",
    "DEGRADED_AFTER",
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_KEY_PREFIX",
    "HEARTBEAT_TTL_SECONDS",
    "RISK_LIMITS_CURRENT_KEY",
    "RISK_LIMITS_KEY_PREFIX",
    "STALE_AFTER",
    "EngineHealth",
    "EngineStatus",
    "Heartbeat",
    "assess_engine",
    "heartbeat_key",
    "risk_limits_key",
]
