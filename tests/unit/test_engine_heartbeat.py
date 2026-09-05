"""Engine liveness, and its strict separation from trading status.

The dashboard twice told an operator the engine had failed while it was managing positions
perfectly well. Both times the reasoning was the same: the newest *decision* was old, so
the engine must be dead. It is not a valid inference. A fully-invested book emits no new
entry decisions at all, and the decision feed is read from a file whose contents the API
sees through a bind mount that goes stale within minutes.

So liveness is answered here from a heartbeat the engine writes to Redis — a real network
service, refreshed every loop — and *trading* status is answered separately from what the
engine is deciding. The two questions have different answers and must never be collapsed:
"the engine is running" and "the engine is not entering trades" are simultaneously true
most of the time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quantflow.live.heartbeat import (
    DEGRADED_AFTER,
    STALE_AFTER,
    EngineHealth,
    Heartbeat,
    assess_engine,
)

NOW = datetime(2026, 8, 15, 6, 30, tzinfo=UTC)


def beat(**overrides: object) -> Heartbeat:
    """A heartbeat written moments ago, with every loop current."""
    base: dict[str, object] = {
        "session_id": "demo-10k-live",
        "pid": 95878,
        "written_at": NOW - timedelta(seconds=5),
        "started_at": NOW - timedelta(hours=1),
        "last_tick_at": NOW - timedelta(seconds=5),
        "last_candle_at": NOW - timedelta(minutes=3),
        "last_reconcile_at": NOW - timedelta(minutes=3),
        "last_decision_at": NOW - timedelta(minutes=3),
        "open_positions": 2,
        "symbols": ("BTC/USDT", "ETH/USDT"),
    }
    base.update(overrides)
    return Heartbeat(**base)  # type: ignore[arg-type]


class TestEngineHealthFromRuntimeEvidence:
    def test_a_fresh_heartbeat_is_running_however_old_the_last_decision(self) -> None:
        """Condition A: heartbeat fresh, decision log ancient → RUNNING.

        This is the exact false alarm that started this: a decision from twelve hours ago
        beside a heartbeat from five seconds ago. The heartbeat wins, because it is the
        only one of the two that says anything about *now*.
        """
        health = assess_engine(beat(last_decision_at=NOW - timedelta(hours=12)), now=NOW)

        assert health.state is EngineHealth.RUNNING
        assert "12" not in health.detail  # the decision age is not the reason for anything

    def test_no_heartbeat_at_all_is_unknown_not_stopped(self) -> None:
        # Absence of evidence is not evidence of death: an engine started before this
        # feature existed, or a Redis that cannot be read, must not read as a crash.
        health = assess_engine(None, now=NOW)

        assert health.state is EngineHealth.UNKNOWN
        assert "no heartbeat" in health.detail.lower()

    def test_a_heartbeat_past_the_stale_threshold_is_stale(self) -> None:
        """Condition E: heartbeat stale → STALE, never RUNNING and never ERROR."""
        health = assess_engine(beat(written_at=NOW - STALE_AFTER - timedelta(seconds=1)), now=NOW)

        assert health.state is EngineHealth.STALE

    def test_a_stalled_candle_loop_is_degraded_while_ticks_continue(self) -> None:
        """The overnight failure, caught: ticks flowing, bars not.

        On 2026-08-14 the bar loop stopped at 19:00 and the ticker loop kept running for
        nine hours. Process liveness said healthy, and it was worse than useless. A loop
        that has stopped while the process lives is DEGRADED — a distinct state, because
        the response is different from both "fine" and "dead".
        """
        health = assess_engine(
            beat(last_candle_at=NOW - DEGRADED_AFTER - timedelta(minutes=1)), now=NOW
        )

        assert health.state is EngineHealth.DEGRADED
        assert "candle" in health.detail.lower()

    def test_a_stalled_reconciliation_loop_is_degraded(self) -> None:
        health = assess_engine(
            beat(last_reconcile_at=NOW - DEGRADED_AFTER - timedelta(minutes=1)), now=NOW
        )

        assert health.state is EngineHealth.DEGRADED
        assert "reconcil" in health.detail.lower()

    def test_an_explicitly_stopped_engine_is_stopped(self) -> None:
        """Condition D: the engine said it was shutting down."""
        health = assess_engine(beat(stopped_at=NOW - timedelta(seconds=30)), now=NOW)

        assert health.state is EngineHealth.STOPPED

    def test_health_carries_every_timestamp_for_display(self) -> None:
        # The operator asked to see these individually rather than as one verdict.
        health = assess_engine(beat(), now=NOW)

        for field in ("last_tick_at", "last_candle_at", "last_reconcile_at", "last_decision_at"):
            assert field in health.evidence


class TestHeartbeatSerialisation:
    def test_a_heartbeat_round_trips_through_redis_json(self) -> None:
        original = beat()

        restored = Heartbeat.from_dict(original.to_dict())

        assert restored == original

    def test_a_malformed_payload_yields_none_rather_than_raising(self) -> None:
        # A dashboard panel must not take the page down because a cache value is junk.
        assert Heartbeat.from_dict({"session_id": "x"}) is None
        assert Heartbeat.from_dict(None) is None
        assert Heartbeat.from_dict({"written_at": "not-a-timestamp"}) is None
