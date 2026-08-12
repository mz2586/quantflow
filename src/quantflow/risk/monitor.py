"""Continuous loss monitoring.

The drawdown, daily-loss and weekly-loss limits lived only inside ``RiskEngine.approve``,
which runs when a *new order* is proposed. A position already open and going against you
produces no new signal, so nothing evaluated those limits and nothing acted. The account
could fall through every one of them and the only component that could have noticed was
never asked.

This monitor closes that. It runs on every equity sample - each candle - and on a timer, so
a breach is caught whether or not the strategy has anything to say. On breach it latches the
kill switch and flattens, in that order: latching first means that even if flattening fails
partway, no new entry can be opened on top of the loss.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal

from quantflow.core.config import RiskSettings
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.risk.engine import RiskEngine

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Breach:
    """A limit that has been exceeded."""

    rule: str
    observed: Decimal
    limit: Decimal
    message: str


def evaluate_limits(
    portfolio: PortfolioSnapshot,
    settings: RiskSettings,
    *,
    week_start_equity: Decimal | None = None,
) -> Breach | None:
    """Return the first breached limit, or ``None``.

    Pure: no engine, no IO, no side effects, so the decision to halt is testable on its own
    and cannot be confused with the act of halting.
    """
    drawdown = portfolio.drawdown_pct
    if drawdown >= settings.max_drawdown_pct:
        return Breach(
            rule="max_drawdown",
            observed=drawdown,
            limit=settings.max_drawdown_pct,
            message=(
                f"drawdown {drawdown:.2%} has reached the {settings.max_drawdown_pct:.2%} "
                "limit with a position open"
            ),
        )

    if portfolio.day_start_equity and portfolio.day_start_equity > ZERO:
        daily = (portfolio.day_start_equity - portfolio.equity) / portfolio.day_start_equity
        if daily >= settings.max_daily_loss_pct:
            return Breach(
                rule="max_daily_loss",
                observed=daily,
                limit=settings.max_daily_loss_pct,
                message=(
                    f"daily loss {daily:.2%} has reached the "
                    f"{settings.max_daily_loss_pct:.2%} limit"
                ),
            )

    if week_start_equity and week_start_equity > ZERO:
        weekly = (week_start_equity - portfolio.equity) / week_start_equity
        if weekly >= settings.max_weekly_loss_pct:
            return Breach(
                rule="max_weekly_loss",
                observed=weekly,
                limit=settings.max_weekly_loss_pct,
                message=(
                    f"weekly loss {weekly:.2%} has reached the "
                    f"{settings.max_weekly_loss_pct:.2%} limit"
                ),
            )
    return None


class LossMonitor:
    """Evaluates loss limits continuously and halts on breach.

    ``flatten`` is injected rather than reached for, so the same monitor serves paper and
    live: each supplies its own way of closing everything.
    """

    __slots__ = ("_flatten", "_risk", "_settings", "_tripped")

    def __init__(
        self,
        risk: RiskEngine,
        settings: RiskSettings,
        *,
        flatten: Callable[[str], Awaitable[object]],
    ) -> None:
        self._risk = risk
        self._settings = settings
        self._flatten = flatten
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """Whether this monitor has already fired."""
        return self._tripped

    async def check(
        self, portfolio: PortfolioSnapshot, *, week_start_equity: Decimal | None = None
    ) -> Breach | None:
        """Evaluate every limit; latch and flatten on the first breach.

        Idempotent. Once tripped it does nothing further, so a breach that persists across
        several bars does not issue a second round of closing orders on an already-flat book.
        """
        if self._tripped:
            return None

        breach = evaluate_limits(portfolio, self._settings, week_start_equity=week_start_equity)
        if breach is None:
            return None

        self._tripped = True
        logger.critical(
            "risk.loss_limit_breached",
            rule=breach.rule,
            observed=f"{breach.observed:.4%}",
            limit=f"{breach.limit:.4%}",
            equity=str(portfolio.equity),
            open_positions=len(portfolio.open_positions),
        )

        # Latch BEFORE flattening. If the close fails partway, the switch is already down
        # and nothing can add to the position while the operator investigates.
        await self._risk.kill_switch.engage(breach.message, actor="loss_monitor")
        try:
            await self._flatten(f"{breach.rule} breached")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("risk.flatten_after_breach_failed", rule=breach.rule, error=str(exc))
        return breach


__all__ = ["Breach", "LossMonitor", "evaluate_limits"]
