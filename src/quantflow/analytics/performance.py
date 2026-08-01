"""Performance analytics beyond the headline metrics.

The backtest metrics answer "how did it do". These answer "why", and "is that likely to
continue" — attribution by strategy, symbol and time, plus the diagnostics that separate a
real edge from a lucky run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade

#: Below this many trades a breakdown is reported but flagged as unreliable.
MIN_TRADES_FOR_ATTRIBUTION = 10

#: Above this share of total profit from one trade, the result is one lucky trade rather
#: than a demonstrated edge.
CONCENTRATION_THRESHOLD = Decimal("0.5")

#: A losing run at least this long is worth warning about: it is what an operator has to
#: sit through, and it predicts whether a strategy gets switched off far better than its
#: Sharpe ratio does.
NOTABLE_LOSS_STREAK = 8

#: A drawdown needs at least two equity samples to have a peak and a trough.
MIN_POINTS_FOR_DRAWDOWN = 2


@dataclass(frozen=True, slots=True)
class Attribution:
    """Performance for one slice of the trade population."""

    key: str
    trade_count: int
    net_pnl: Decimal
    gross_pnl: Decimal
    fees: Decimal
    win_count: int
    win_rate: Decimal
    average_pnl: Decimal
    best: Decimal
    worst: Decimal
    total_volume: Decimal

    @property
    def is_reliable(self) -> bool:
        """Whether this slice has enough trades to be worth acting on."""
        return self.trade_count >= MIN_TRADES_FOR_ATTRIBUTION

    @property
    def fee_drag_pct(self) -> Decimal:
        """Fees as a fraction of gross PnL.

        The number that reveals a strategy which is profitable before costs and
        unprofitable after — the single most common way a promising backtest dies.
        """
        return safe_divide(self.fees, abs(self.gross_pnl))

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API and reports."""
        return {
            "key": self.key,
            "trade_count": self.trade_count,
            "net_pnl": str(self.net_pnl),
            "gross_pnl": str(self.gross_pnl),
            "fees": str(self.fees),
            "win_count": self.win_count,
            "win_rate": str(self.win_rate),
            "average_pnl": str(self.average_pnl),
            "best": str(self.best),
            "worst": str(self.worst),
            "fee_drag_pct": str(self.fee_drag_pct),
            "reliable": self.is_reliable,
        }


def _attribute(key: str, trades: Sequence[ClosedTrade]) -> Attribution:
    wins = [trade for trade in trades if trade.is_win]
    net = sum((trade.net_pnl for trade in trades), ZERO)
    gross = sum((trade.gross_pnl for trade in trades), ZERO)
    fees = sum((trade.fees for trade in trades), ZERO)
    volume = sum((trade.quantity * trade.entry_price for trade in trades), ZERO)

    return Attribution(
        key=key,
        trade_count=len(trades),
        net_pnl=net,
        gross_pnl=gross,
        fees=fees,
        win_count=len(wins),
        win_rate=safe_divide(Decimal(len(wins)), Decimal(len(trades))),
        average_pnl=safe_divide(net, Decimal(len(trades))),
        best=max((trade.net_pnl for trade in trades), default=ZERO),
        worst=min((trade.net_pnl for trade in trades), default=ZERO),
        total_volume=volume,
    )


def by_strategy(trades: Sequence[ClosedTrade]) -> list[Attribution]:
    """Performance per strategy, best first."""
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.strategy_id or "unattributed"].append(trade)
    return sorted(
        (_attribute(key, group) for key, group in grouped.items()),
        key=lambda item: item.net_pnl,
        reverse=True,
    )


def by_symbol(trades: Sequence[ClosedTrade]) -> list[Attribution]:
    """Performance per symbol, best first."""
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol.slashed].append(trade)
    return sorted(
        (_attribute(key, group) for key, group in grouped.items()),
        key=lambda item: item.net_pnl,
        reverse=True,
    )


def by_side(trades: Sequence[ClosedTrade]) -> list[Attribution]:
    """Performance split long versus short.

    A strategy that only makes money on one side in a market that trended one way has not
    demonstrated an edge — it has demonstrated the trend.
    """
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.side.value].append(trade)
    return [_attribute(key, group) for key, group in sorted(grouped.items())]


def by_hour_of_day(trades: Sequence[ClosedTrade]) -> list[Attribution]:
    """Performance by UTC hour of entry.

    Crypto trades continuously but liquidity does not: spreads and volume shift with the
    Asian, European and US sessions, and a strategy can be quietly unprofitable in one.
    """
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[f"{trade.entry_time.hour:02d}"].append(trade)
    return [_attribute(key, group) for key, group in sorted(grouped.items())]


def by_month(trades: Sequence[ClosedTrade]) -> list[Attribution]:
    """Performance by calendar month of exit."""
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.exit_time.strftime("%Y-%m")].append(trade)
    return [_attribute(key, group) for key, group in sorted(grouped.items())]


@dataclass(frozen=True, slots=True)
class StreakAnalysis:
    """Consecutive-outcome statistics."""

    longest_win_streak: int
    longest_loss_streak: int
    current_streak: int
    """Positive for wins, negative for losses."""

    @property
    def is_on_a_losing_run(self) -> bool:
        """Whether the most recent trades were losses."""
        return self.current_streak < 0


def streaks(trades: Sequence[ClosedTrade]) -> StreakAnalysis:
    """Longest winning and losing runs.

    A long losing streak is what an operator actually has to sit through, and it is far
    more predictive of whether a strategy gets switched off than its Sharpe ratio is.
    """
    if not trades:
        return StreakAnalysis(0, 0, 0)

    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    longest_win = longest_loss = 0
    run = 0

    for trade in ordered:
        if trade.is_win:
            run = run + 1 if run > 0 else 1
            longest_win = max(longest_win, run)
        else:
            run = run - 1 if run < 0 else -1
            longest_loss = max(longest_loss, -run)

    return StreakAnalysis(
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        current_streak=run,
    )


@dataclass(frozen=True, slots=True)
class ConcentrationAnalysis:
    """How much of the result came from how few trades."""

    top_trade_share: Decimal
    """Share of total net profit contributed by the single best trade.

    Zero when the overall result is a loss — a "share of profit" is undefined then, which
    is why :attr:`total_net_pnl` is carried separately rather than being inferred.
    """
    top_five_share: Decimal
    profit_without_best: Decimal
    """Total net PnL excluding the single best trade."""
    total_net_pnl: Decimal = ZERO
    """Net PnL across every trade, reported even when negative."""

    @property
    def is_concentrated(self) -> bool:
        """Whether one trade dominates the result.

        A strategy whose profit is one lucky trade has not been demonstrated to work; it
        has been demonstrated to have got lucky once.
        """
        return self.top_trade_share > CONCENTRATION_THRESHOLD

    @property
    def survives_without_best_trade(self) -> bool:
        """Whether the strategy is still profitable with its best trade removed."""
        return self.profit_without_best > ZERO

    @property
    def rests_on_one_trade(self) -> bool:
        """Whether the entire result is one trade.

        True when the overall result is profitable but turns negative once the single best
        trade is removed. That is not a demonstrated edge; it is one trade that worked.
        """
        return self.total_net_pnl > ZERO and self.profit_without_best <= ZERO


def concentration(trades: Sequence[ClosedTrade]) -> ConcentrationAnalysis:
    """Measure how dependent the result is on a handful of trades."""
    if not trades:
        return ConcentrationAnalysis(ZERO, ZERO, ZERO, ZERO)

    pnls = sorted((trade.net_pnl for trade in trades), reverse=True)
    total = sum(pnls, ZERO)
    best = pnls[0]

    # A "share of profit" is undefined when the total is a loss, but the rest of the
    # analysis still matters — reporting nothing there would hide a result that is only
    # positive because of one outlier.
    shares = (
        (safe_divide(best, total), safe_divide(sum(pnls[:5], ZERO), total))
        if total > ZERO
        else (ZERO, ZERO)
    )
    return ConcentrationAnalysis(
        top_trade_share=shares[0],
        top_five_share=shares[1],
        profit_without_best=total - best,
        total_net_pnl=total,
    )


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    """One peak-to-recovery decline."""

    peak_at: datetime
    trough_at: datetime
    recovered_at: datetime | None
    peak_equity: Decimal
    trough_equity: Decimal
    depth_pct: Decimal
    duration_days: Decimal
    recovery_days: Decimal | None

    @property
    def recovered(self) -> bool:
        """Whether equity regained its prior peak."""
        return self.recovered_at is not None


def drawdown_episodes(
    curve: Sequence[EquityPoint], *, min_depth_pct: Decimal = Decimal("0.02")
) -> list[DrawdownEpisode]:
    """Every distinct drawdown deeper than ``min_depth_pct``.

    The maximum drawdown is one number; the *distribution* of drawdowns is what tells you
    whether the worst one was typical or an outlier.
    """
    if len(curve) < MIN_POINTS_FOR_DRAWDOWN:
        return []

    episodes: list[DrawdownEpisode] = []
    peak = curve[0].equity
    peak_at = curve[0].timestamp
    trough = peak
    trough_at = peak_at
    in_drawdown = False

    for point in curve:
        if point.equity >= peak:
            if in_drawdown:
                depth = safe_divide(peak - trough, peak)
                if depth >= min_depth_pct:
                    episodes.append(
                        DrawdownEpisode(
                            peak_at=peak_at,
                            trough_at=trough_at,
                            recovered_at=point.timestamp,
                            peak_equity=peak,
                            trough_equity=trough,
                            depth_pct=depth,
                            duration_days=_days_between(peak_at, trough_at),
                            recovery_days=_days_between(trough_at, point.timestamp),
                        )
                    )
                in_drawdown = False
            peak = point.equity
            peak_at = point.timestamp
            trough = point.equity
            trough_at = point.timestamp
            continue

        in_drawdown = True
        if point.equity < trough:
            trough = point.equity
            trough_at = point.timestamp

    # An unrecovered drawdown still counts — it is the one currently being lived through.
    if in_drawdown:
        depth = safe_divide(peak - trough, peak)
        if depth >= min_depth_pct:
            episodes.append(
                DrawdownEpisode(
                    peak_at=peak_at,
                    trough_at=trough_at,
                    recovered_at=None,
                    peak_equity=peak,
                    trough_equity=trough,
                    depth_pct=depth,
                    duration_days=_days_between(peak_at, trough_at),
                    recovery_days=None,
                )
            )

    return sorted(episodes, key=lambda episode: episode.depth_pct, reverse=True)


def _days_between(start: datetime, end: datetime) -> Decimal:
    return Decimal(str((end - start).total_seconds() / 86400))


@dataclass(frozen=True, slots=True)
class PerformanceReview:
    """The full analytical picture for a set of trades."""

    trade_count: int
    strategies: list[Attribution] = field(default_factory=list)
    symbols: list[Attribution] = field(default_factory=list)
    sides: list[Attribution] = field(default_factory=list)
    hours: list[Attribution] = field(default_factory=list)
    months: list[Attribution] = field(default_factory=list)
    streak: StreakAnalysis = field(default_factory=lambda: StreakAnalysis(0, 0, 0))
    concentration: ConcentrationAnalysis = field(
        default_factory=lambda: ConcentrationAnalysis(ZERO, ZERO, ZERO, ZERO)
    )
    drawdowns: list[DrawdownEpisode] = field(default_factory=list)

    def warnings(self) -> list[str]:
        """Plain-language caveats an operator should read before acting on this."""
        notes: list[str] = []

        if self.trade_count < MIN_TRADES_FOR_ATTRIBUTION:
            notes.append(
                f"Only {self.trade_count} trades — every breakdown below is noise at this "
                "sample size."
            )
        if self.concentration.is_concentrated:
            notes.append(
                f"The single best trade contributed "
                f"{self.concentration.top_trade_share:.0%} of total profit. Without it the "
                f"result is {self.concentration.profit_without_best:,.2f}."
            )
        if self.concentration.rests_on_one_trade:
            notes.append(
                "Removing the single best trade turns the result negative; this is one "
                "lucky trade rather than a demonstrated edge."
            )
        if self.streak.longest_loss_streak >= NOTABLE_LOSS_STREAK:
            notes.append(
                f"Longest losing streak was {self.streak.longest_loss_streak} trades — "
                "consider whether that is survivable in practice."
            )
        for attribution in self.strategies:
            if attribution.is_reliable and attribution.fee_drag_pct > Decimal("0.5"):
                notes.append(
                    f"{attribution.key}: fees consumed "
                    f"{attribution.fee_drag_pct:.0%} of gross profit."
                )
        return notes

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API."""
        return {
            "trade_count": self.trade_count,
            "by_strategy": [item.to_dict() for item in self.strategies],
            "by_symbol": [item.to_dict() for item in self.symbols],
            "by_side": [item.to_dict() for item in self.sides],
            "by_hour": [item.to_dict() for item in self.hours],
            "by_month": [item.to_dict() for item in self.months],
            "streaks": {
                "longest_win": self.streak.longest_win_streak,
                "longest_loss": self.streak.longest_loss_streak,
                "current": self.streak.current_streak,
            },
            "concentration": {
                "top_trade_share": str(self.concentration.top_trade_share),
                "top_five_share": str(self.concentration.top_five_share),
                "profit_without_best": str(self.concentration.profit_without_best),
                "total_net_pnl": str(self.concentration.total_net_pnl),
                "is_concentrated": self.concentration.is_concentrated,
                "rests_on_one_trade": self.concentration.rests_on_one_trade,
            },
            "drawdowns": [
                {
                    "depth_pct": str(episode.depth_pct),
                    "peak_at": episode.peak_at.isoformat(),
                    "trough_at": episode.trough_at.isoformat(),
                    "recovered": episode.recovered,
                    "duration_days": str(episode.duration_days),
                }
                for episode in self.drawdowns[:10]
            ],
            "warnings": self.warnings(),
        }


def review(trades: Sequence[ClosedTrade], curve: Sequence[EquityPoint] = ()) -> PerformanceReview:
    """Build the full analytical picture."""
    return PerformanceReview(
        trade_count=len(trades),
        strategies=by_strategy(trades),
        symbols=by_symbol(trades),
        sides=by_side(trades),
        hours=by_hour_of_day(trades),
        months=by_month(trades),
        streak=streaks(trades),
        concentration=concentration(trades),
        drawdowns=drawdown_episodes(curve),
    )


def rolling_win_rate(
    trades: Sequence[ClosedTrade], *, window: int = 20
) -> list[tuple[datetime, Decimal]]:
    """Win rate over a trailing window of trades.

    Reveals decay: a strategy whose win rate is drifting down over time is losing its edge,
    which a single aggregate figure hides completely.
    """
    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    if len(ordered) < window:
        return []
    series: list[tuple[datetime, Decimal]] = []
    for index in range(window, len(ordered) + 1):
        chunk = ordered[index - window : index]
        wins = sum(1 for trade in chunk if trade.is_win)
        series.append((chunk[-1].exit_time, Decimal(wins) / Decimal(window)))
    return series


def symbol_exposure(trades: Sequence[ClosedTrade]) -> dict[Symbol, Decimal]:
    """Total traded notional per symbol."""
    exposure: dict[Symbol, Decimal] = defaultdict(lambda: ZERO)
    for trade in trades:
        exposure[trade.symbol] += trade.quantity * trade.entry_price
    return dict(exposure)


def long_short_balance(trades: Sequence[ClosedTrade]) -> tuple[int, int]:
    """Count of long and short round-trips."""
    longs = sum(1 for trade in trades if trade.side is PositionSide.LONG)
    shorts = sum(1 for trade in trades if trade.side is PositionSide.SHORT)
    return longs, shorts
