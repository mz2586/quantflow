"""What the engine is actually doing, derived from runtime evidence.

A status derived from process existence answers the wrong question. "The process is
running" is true of an engine trading normally, an engine whose market-data stream died an
hour ago, and an engine looping on an exception it cannot recover from. The operator needs
to know which.

Every state below is therefore inferred from evidence the engine leaves behind — the
decisions it recorded, the equity snapshots it wrote, the kill switch, the venue's own view
of the account — and each carries the evidence that produced it, so a surprising status can
be argued with rather than merely believed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from quantflow.api.dashboard.decisions import Decision
from quantflow.core.clock import utc_now

#: States, most severe first. The first whose condition holds is the reported state.
TRADING = "TRADING"
WAITING = "WAITING FOR QUALIFIED SIGNAL"
FILTERING = "FILTERING"
RISK_BLOCKED = "RISK BLOCKED"
EXECUTION_BLOCKED = "EXECUTION BLOCKED"
ENGINE_ERROR = "ENGINE ERROR"
DISCONNECTED = "DISCONNECTED"
STARTING = "STARTING"

#: How long without any recorded decision before the engine is presumed not to be
#: evaluating. Generous relative to the 15-minute bar the engine runs on, because the
#: orchestrator logs per symbol per bar and a quiet gap is normal.
DECISION_SILENCE = timedelta(minutes=45)

#: Refusals in the recent window before execution is called broken rather than selective.
#:
#: Three, because one or two are the ordinary consequence of a risk limit binding — a
#: position-cap rejection is the system working. A sustained run of them while selection
#: keeps succeeding is a different thing: orders are being produced that cannot be placed.
EXECUTION_FAILURE_STREAK = 3

#: How long without a new equity snapshot before the engine is presumed stalled.
SNAPSHOT_SILENCE = timedelta(minutes=90)


@dataclass(frozen=True, slots=True)
class Status:
    """A derived engine state and the evidence behind it."""

    state: str
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Wire form."""
        return {"state": self.state, "detail": self.detail, "evidence": self.evidence}


def derive(  # noqa: PLR0911, PLR0912 — a flat ladder of guard clauses, read top to bottom.
    *,
    venue_available: bool,
    venue_error: str | None,
    kill_switch_engaged: bool,
    trading_halted: bool,
    session_running: bool,
    session_status: str | None,
    open_position_count: int | None,
    last_snapshot_at: datetime | None,
    decisions: list[Decision],
    recent_order_rejections: int,
) -> Status:
    """Decide what the engine is doing.

    Args:
        venue_available: Whether the venue answered this refresh.
        venue_error: Why it did not, when it did not.
        kill_switch_engaged: Whether the latched emergency halt is on.
        trading_halted: Whether a daily-loss halt is in force.
        session_running: Whether the session row says it is running.
        session_status: The session's recorded status.
        open_position_count: Open positions per the venue, or ``None`` when unknown.
        last_snapshot_at: When the engine last wrote an equity snapshot.
        decisions: Recent parsed decisions, oldest first.
        recent_order_rejections: Orders the venue or risk engine refused recently.

    Returns:
        The derived state with its supporting evidence.

    """
    now = utc_now()
    recent = [item for item in decisions if now - item.timestamp <= DECISION_SILENCE]
    outcomes = Counter(item.outcome for item in recent)
    last_decision_at = decisions[-1].timestamp if decisions else None

    evidence: dict[str, Any] = {
        "venue_available": venue_available,
        "open_positions_venue": open_position_count,
        "last_decision_at": last_decision_at.isoformat() if last_decision_at else None,
        "last_snapshot_at": last_snapshot_at.isoformat() if last_snapshot_at else None,
        "recent_decisions": len(recent),
        "recent_outcomes": dict(outcomes),
        "recent_order_rejections": recent_order_rejections,
        "kill_switch_engaged": kill_switch_engaged,
        "trading_halted": trading_halted,
        "session_status": session_status,
    }

    # Ordered by how much it changes what the operator should do next. A disconnected
    # venue outranks everything: no other reading on the page can be trusted while the
    # account cannot be read.
    if not venue_available:
        return Status(
            DISCONNECTED,
            venue_error or "the venue could not be reached on this refresh",
            evidence,
        )

    if kill_switch_engaged:
        return Status(RISK_BLOCKED, "the kill switch is engaged; no new entries", evidence)

    if session_status is not None and session_status.lower() in {"failed", "error"}:
        return Status(ENGINE_ERROR, f"the session is recorded as {session_status}", evidence)

    if not session_running:
        return Status(
            ENGINE_ERROR,
            f"no session is running (status {session_status or 'unknown'})",
            evidence,
        )

    if last_snapshot_at is not None and now - last_snapshot_at > SNAPSHOT_SILENCE:
        return Status(
            ENGINE_ERROR,
            f"no equity snapshot since {last_snapshot_at.isoformat()}; the engine "
            "appears to have stopped writing state",
            evidence,
        )

    if last_decision_at is None:
        return Status(
            STARTING,
            "no decisions found in the engine log tail; the engine may still be warming up",
            evidence,
        )

    # A silent decision log only means the engine is dead if it has also stopped writing
    # state. The two sources are not equally trustworthy: equity snapshots come from
    # Postgres, a real network service the engine writes to every bar, while decisions are
    # parsed from a log file the API sees through a bind mount — and on this deployment
    # that view goes stale within minutes while the host keeps appending. The same reader,
    # same path, run on the host returns decisions the container never saw.
    #
    # So a fresh snapshot beside a silent log is a log-reading fault, and saying
    # ENGINE ERROR there sends the operator hunting a dead bot that is in fact managing
    # positions. Both silent is the real failure and still reports as one.
    snapshot_is_fresh = last_snapshot_at is not None and now - last_snapshot_at <= SNAPSHOT_SILENCE
    if now - last_decision_at > DECISION_SILENCE and not snapshot_is_fresh:
        return Status(
            ENGINE_ERROR,
            f"the engine has recorded no decision since {last_decision_at.isoformat()}",
            evidence,
        )
    if now - last_decision_at > DECISION_SILENCE:
        # Surfaced rather than hidden: the panel still says the decision feed is behind,
        # it just no longer calls a healthy engine broken.
        evidence["decision_log_stale"] = True
        evidence["decision_log_note"] = (
            "the decision feed is behind; equity snapshots are current, so the engine is "
            "running and this is a log-reading lag, not an engine fault"
        )

    if trading_halted:
        return Status(RISK_BLOCKED, "trading is halted for the day", evidence)

    # One refusal is a fact about that order, not a state of the engine. A position-cap
    # rejection is the risk engine working, and latching the headline red on it sends the
    # operator hunting a fault that does not exist — observed live, where a single refusal
    # at 11:30 still read EXECUTION BLOCKED forty minutes and two evaluated bars later.
    #
    # Selection repeatedly succeeding while orders repeatedly fail is a real execution
    # fault, and that still reports as one. The rejection is always surfaced in evidence so
    # it is visible either way.
    evidence["last_order_refused"] = recent_order_rejections > 0
    if recent_order_rejections >= EXECUTION_FAILURE_STREAK and outcomes.get("SELECTED", 0) > 0:
        return Status(
            EXECUTION_BLOCKED,
            f"{recent_order_rejections} order(s) refused after selection succeeded",
            evidence,
        )

    if outcomes.get("RISK_BLOCKED", 0) > 0:
        return Status(
            RISK_BLOCKED,
            "the risk engine is refusing orders that selection produced",
            evidence,
        )

    if open_position_count:
        return Status(
            TRADING,
            f"{open_position_count} position(s) open on the venue",
            evidence,
        )

    if outcomes.get("GATED", 0) >= outcomes.get("DESELECTED", 0) and outcomes.get("GATED", 0):
        return Status(
            FILTERING,
            "candidates are being rejected by the economic gates before selection",
            evidence,
        )

    if outcomes:
        return Status(
            WAITING,
            "the engine is evaluating every bar and no candidate has qualified",
            evidence,
        )

    return Status(STARTING, "the engine is running but has not yet decided anything", evidence)


__all__ = [
    "DECISION_SILENCE",
    "DISCONNECTED",
    "ENGINE_ERROR",
    "EXECUTION_BLOCKED",
    "FILTERING",
    "RISK_BLOCKED",
    "SNAPSHOT_SILENCE",
    "STARTING",
    "TRADING",
    "WAITING",
    "Status",
    "derive",
]
