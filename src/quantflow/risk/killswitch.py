"""Emergency kill switch.

Latched, persistent state. Once engaged, no new entry can be submitted through any code
path until an operator explicitly clears it — a restart does **not** clear it, because the
condition that engaged it (a drawdown breach, a data fault, a manual panic) is still true
until a human says otherwise.

The in-memory flag is the fast path checked on every order; the database row is the durable
record that survives a process crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantflow.core.clock import Clock, SystemClock
from quantflow.core.errors import KillSwitchEngagedError
from quantflow.core.logging import get_logger
from quantflow.persistence.database import Database

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    """A snapshot of the switch."""

    engaged: bool
    reason: str | None = None
    engaged_at: datetime | None = None
    engaged_by: str | None = None

    @property
    def is_clear(self) -> bool:
        """Whether trading is permitted."""
        return not self.engaged


class KillSwitch:
    """Latched trading halt, backed by the database.

    Reads hit memory so the check costs nothing on the order path; writes go to both memory
    and the database so the latch survives a restart.
    """

    __slots__ = ("_clock", "_database", "_state")

    def __init__(self, database: Database | None = None, *, clock: Clock | None = None) -> None:
        self._database = database
        self._clock = clock or SystemClock()
        self._state = KillSwitchState(engaged=False)

    @property
    def state(self) -> KillSwitchState:
        """The current in-memory state."""
        return self._state

    @property
    def engaged(self) -> bool:
        """Whether the switch is latched."""
        return self._state.engaged

    async def load(self) -> KillSwitchState:
        """Restore the latch from the database.

        Called during startup. If it cannot be read, the switch **fails closed**: an
        unknown risk state is treated as unsafe, because resuming trading on the assumption
        that everything is fine is precisely the wrong bet to make after a crash.
        """
        if self._database is None:
            return self._state
        try:
            async with self._database.read_session() as session:
                from quantflow.persistence.repositories import RiskEventRepository

                record = await RiskEventRepository(session).get_kill_switch()
                self._state = KillSwitchState(
                    engaged=record.engaged,
                    reason=record.reason,
                    engaged_at=record.engaged_at,
                    engaged_by=record.engaged_by,
                )
        except Exception as exc:
            logger.exception("killswitch.load_failed", error=str(exc))
            self._state = KillSwitchState(
                engaged=True,
                reason=f"kill switch state could not be read ({exc}); failing closed",
                engaged_at=self._clock.now(),
                engaged_by="system",
            )
        if self._state.engaged:
            logger.critical(
                "killswitch.engaged_on_startup",
                reason=self._state.reason,
                engaged_at=self._state.engaged_at.isoformat() if self._state.engaged_at else None,
            )
        return self._state

    async def engage(self, reason: str, *, actor: str = "risk_engine") -> KillSwitchState:
        """Latch the switch. Idempotent: re-engaging keeps the original reason."""
        if self._state.engaged:
            return self._state

        self._state = KillSwitchState(
            engaged=True, reason=reason, engaged_at=self._clock.now(), engaged_by=actor
        )
        logger.critical("killswitch.engaged", reason=reason, actor=actor)

        if self._database is not None:
            try:
                async with self._database.unit_of_work() as uow:
                    await uow.risk_events.set_kill_switch(engaged=True, reason=reason, actor=actor)
                    await uow.risk_events.record(
                        rule="kill_switch",
                        message=reason,
                        severity="critical",
                        blocked_order=True,
                        halted_trading=True,
                    )
            except Exception as exc:
                logger.exception("killswitch.persist_failed", error=str(exc))
        return self._state

    async def clear(self, *, actor: str = "operator") -> KillSwitchState:
        """Clear the latch. Deliberately requires an explicit actor for the audit trail."""
        previous_reason = self._state.reason
        self._state = KillSwitchState(engaged=False)
        logger.warning("killswitch.cleared", actor=actor, previous_reason=previous_reason)

        if self._database is not None:
            async with self._database.unit_of_work() as uow:
                await uow.risk_events.set_kill_switch(engaged=False, actor=actor)
        return self._state

    def require_clear(self) -> None:
        """Assert trading is permitted.

        Raises:
            KillSwitchEngagedError: if the switch is latched.

        """
        if self._state.engaged:
            raise KillSwitchEngagedError(
                f"trading is halted: {self._state.reason or 'kill switch engaged'}",
                engaged_at=self._state.engaged_at.isoformat() if self._state.engaged_at else None,
            )
