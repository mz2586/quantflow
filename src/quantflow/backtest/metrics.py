"""Performance metrics.

Every metric states its convention explicitly. Two backtests are only comparable if they
agree on the risk-free rate, the annualisation factor and whether returns are simple or
logarithmic — and a Sharpe ratio quoted without those is close to meaningless.

Crypto trades continuously, so annualisation uses a **365-day** year, not the 252 trading
days used for equities. Using 252 here would overstate an annualised figure by ~20%.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from typing import Any

from quantflow.core.precision import ONE, ZERO, safe_divide
from quantflow.domain.enums import Timeframe
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade

SECONDS_PER_YEAR = 365.0 * 24 * 3600

#: A standard deviation needs at least two observations; below that every dispersion
#: metric is undefined rather than zero.
MIN_SAMPLES_FOR_DISPERSION = 2

#: Below this many observations a Sharpe confidence interval is not worth reporting.
MIN_SAMPLES_FOR_INTERVAL = 3

#: Below this many trades a result is statistical noise, however good it looks.
MIN_TRADES_FOR_SIGNIFICANCE = 30

#: Z-scores for the two confidence levels the interval helper supports.
Z_SCORE_95 = 1.96
Z_SCORE_90 = 1.645
CONFIDENCE_95 = 0.95


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """The full metric set for one run."""

    # -- returns --
    starting_equity: Decimal
    final_equity: Decimal
    total_return_pct: Decimal
    cagr: Decimal
    # -- risk --
    max_drawdown_pct: Decimal
    max_drawdown_duration_days: Decimal
    volatility_annual: Decimal
    downside_volatility_annual: Decimal
    # -- risk-adjusted --
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    # -- trades --
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: Decimal
    profit_factor: Decimal
    expectancy: Decimal
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_holding_hours: Decimal
    # -- costs and activity --
    total_fees: Decimal
    turnover: Decimal
    exposure_pct: Decimal
    # -- context --
    start: datetime | None = None
    end: datetime | None = None
    duration_days: Decimal = ZERO
    bars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_profitable(self) -> bool:
        """Whether the run finished above its starting equity."""
        return self.final_equity > self.starting_equity

    def to_dict(self) -> dict[str, Any]:
        """Serialise as JSON-safe floats for persistence and the API.

        Floats here, not Decimal: these are reporting figures, and nothing downstream
        converts them back into an order.
        """
        return {
            "starting_equity": float(self.starting_equity),
            "final_equity": float(self.final_equity),
            "total_return_pct": float(self.total_return_pct),
            "cagr": float(self.cagr),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "max_drawdown_duration_days": float(self.max_drawdown_duration_days),
            "volatility_annual": float(self.volatility_annual),
            "downside_volatility_annual": float(self.downside_volatility_annual),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "expectancy": float(self.expectancy),
            "average_win": float(self.average_win),
            "average_loss": float(self.average_loss),
            "largest_win": float(self.largest_win),
            "largest_loss": float(self.largest_loss),
            "average_holding_hours": float(self.average_holding_hours),
            "total_fees": float(self.total_fees),
            "turnover": float(self.turnover),
            "exposure_pct": float(self.exposure_pct),
            "duration_days": float(self.duration_days),
            "bars": self.bars,
        }

    def summary_lines(self) -> list[str]:
        """Human-readable summary for the CLI."""
        return [
            f"Return          {self.total_return_pct:>10.2%}",
            f"CAGR            {self.cagr:>10.2%}",
            f"Max drawdown    {self.max_drawdown_pct:>10.2%}",
            f"Sharpe          {self.sharpe_ratio:>10.2f}",
            f"Sortino         {self.sortino_ratio:>10.2f}",
            f"Calmar          {self.calmar_ratio:>10.2f}",
            f"Trades          {self.trade_count:>10d}",
            f"Win rate        {self.win_rate:>10.2%}",
            f"Profit factor   {self.profit_factor:>10.2f}",
            f"Fees            {self.total_fees:>10.2f}",
        ]


def _to_float(value: Decimal) -> float:
    return float(value)


def period_returns(curve: Sequence[EquityPoint]) -> list[float]:
    """Simple per-sample returns from an equity curve."""
    if len(curve) < MIN_SAMPLES_FOR_DISPERSION:
        return []
    returns: list[float] = []
    for previous, current in pairwise(curve):
        if previous.equity <= ZERO:
            returns.append(0.0)
            continue
        returns.append(_to_float((current.equity - previous.equity) / previous.equity))
    return returns


def annualisation_factor(timeframe: Timeframe) -> float:
    """Number of bars in a 365-day year for the given interval."""
    return timeframe.periods_per_year


def sharpe(
    returns: Sequence[float], *, periods_per_year: float, risk_free_rate: float = 0.0
) -> Decimal:
    """Annualised Sharpe ratio.

    ``risk_free_rate`` is an *annual* rate and is converted to a per-period rate before
    subtraction. Returns zero when there is no variance rather than dividing by zero — an
    "infinite Sharpe" is a bug indicator, not a result.
    """
    if len(returns) < MIN_SAMPLES_FOR_DISPERSION:
        return ZERO
    per_period_rf = risk_free_rate / periods_per_year
    excess = [value - per_period_rf for value in returns]
    mean = sum(excess) / len(excess)
    variance = sum((value - mean) ** 2 for value in excess) / (len(excess) - 1)
    if variance <= 0:
        return ZERO
    ratio = mean / math.sqrt(variance) * math.sqrt(periods_per_year)
    return Decimal(str(round(ratio, 6)))


def sortino(
    returns: Sequence[float], *, periods_per_year: float, risk_free_rate: float = 0.0
) -> Decimal:
    """Annualised Sortino ratio.

    Penalises only downside deviation. Upside volatility is not risk, and Sharpe's
    symmetric treatment of it under-rates strategies with occasional large gains.
    """
    if len(returns) < MIN_SAMPLES_FOR_DISPERSION:
        return ZERO
    per_period_rf = risk_free_rate / periods_per_year
    excess = [value - per_period_rf for value in returns]
    mean = sum(excess) / len(excess)
    downside = [value for value in excess if value < 0]
    if not downside:
        return ZERO
    downside_variance = sum(value**2 for value in downside) / len(excess)
    if downside_variance <= 0:
        return ZERO
    ratio = mean / math.sqrt(downside_variance) * math.sqrt(periods_per_year)
    return Decimal(str(round(ratio, 6)))


def volatility(returns: Sequence[float], *, periods_per_year: float) -> Decimal:
    """Annualised standard deviation of returns."""
    if len(returns) < MIN_SAMPLES_FOR_DISPERSION:
        return ZERO
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return Decimal(str(round(math.sqrt(variance) * math.sqrt(periods_per_year), 6)))


def downside_volatility(returns: Sequence[float], *, periods_per_year: float) -> Decimal:
    """Annualised standard deviation of negative returns only."""
    downside = [value for value in returns if value < 0]
    if len(downside) < MIN_SAMPLES_FOR_DISPERSION:
        return ZERO
    variance = sum(value**2 for value in downside) / len(returns)
    return Decimal(str(round(math.sqrt(variance) * math.sqrt(periods_per_year), 6)))


def max_drawdown(curve: Sequence[EquityPoint]) -> tuple[Decimal, Decimal]:
    """Deepest peak-to-trough decline and its duration.

    Returns:
        ``(max_drawdown_fraction, duration_days)``. Duration measures peak to *recovery*,
        or peak to the end of the run if equity never recovered — which is the number that
        actually matters to whoever has to sit through it.

    """
    if not curve:
        return ZERO, ZERO

    peak = curve[0].equity
    peak_time = curve[0].timestamp
    worst = ZERO
    longest = ZERO

    for point in curve:
        if point.equity >= peak:
            if peak_time is not None:
                span = Decimal(str((point.timestamp - peak_time).total_seconds() / 86400))
                longest = max(longest, span)
            peak = point.equity
            peak_time = point.timestamp
            continue
        if peak > ZERO:
            worst = max(worst, (peak - point.equity) / peak)

    # An unrecovered drawdown still counts, measured to the end of the run.
    if curve[-1].equity < peak and peak_time is not None:
        span = Decimal(str((curve[-1].timestamp - peak_time).total_seconds() / 86400))
        longest = max(longest, span)

    return worst, longest


def cagr(starting_equity: Decimal, final_equity: Decimal, *, duration_days: Decimal) -> Decimal:
    """Compound annual growth rate.

    Returns the simple total return for runs shorter than a week: annualising a three-day
    result produces a number that looks like a forecast and is not one.
    """
    if starting_equity <= ZERO or duration_days <= ZERO:
        return ZERO
    if final_equity <= ZERO:
        return Decimal("-1")
    if duration_days < Decimal("7"):
        return (final_equity - starting_equity) / starting_equity
    years = float(duration_days) / 365.0
    growth = float(final_equity / starting_equity)
    return Decimal(str(round(growth ** (1 / years) - 1, 6)))


def trade_statistics(trades: Sequence[ClosedTrade]) -> dict[str, Decimal | int]:
    """Win rate, profit factor, expectancy and related trade-level figures."""
    if not trades:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": ZERO,
            "profit_factor": ZERO,
            "expectancy": ZERO,
            "average_win": ZERO,
            "average_loss": ZERO,
            "largest_win": ZERO,
            "largest_loss": ZERO,
            "average_holding_hours": ZERO,
        }

    wins = [trade for trade in trades if trade.net_pnl > ZERO]
    losses = [trade for trade in trades if trade.net_pnl <= ZERO]
    gross_profit = sum((trade.net_pnl for trade in wins), ZERO)
    gross_loss = abs(sum((trade.net_pnl for trade in losses), ZERO))
    net = gross_profit - gross_loss

    return {
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": safe_divide(Decimal(len(wins)), Decimal(len(trades))),
        # An infinite profit factor (no losses) is reported as the gross profit rather
        # than infinity, which serialises and sorts sensibly.
        "profit_factor": safe_divide(gross_profit, gross_loss, default=gross_profit),
        "expectancy": safe_divide(net, Decimal(len(trades))),
        "average_win": safe_divide(gross_profit, Decimal(len(wins))) if wins else ZERO,
        "average_loss": safe_divide(gross_loss, Decimal(len(losses))) if losses else ZERO,
        "largest_win": max((trade.net_pnl for trade in wins), default=ZERO),
        "largest_loss": min((trade.net_pnl for trade in losses), default=ZERO),
        "average_holding_hours": safe_divide(
            sum((trade.holding_period for trade in trades), ZERO) / Decimal("3600"),
            Decimal(len(trades)),
        ),
    }


def exposure(curve: Sequence[EquityPoint]) -> Decimal:
    """Fraction of samples with at least one open position.

    A strategy that is only in the market 5% of the time and returns 10% is a very
    different proposition from one that is fully invested throughout for the same return.
    """
    if not curve:
        return ZERO
    invested = sum(1 for point in curve if point.position_count > 0)
    return safe_divide(Decimal(invested), Decimal(len(curve)))


def turnover(trades: Sequence[ClosedTrade], starting_equity: Decimal) -> Decimal:
    """Total traded notional as a multiple of starting equity."""
    if starting_equity <= ZERO:
        return ZERO
    traded = sum((trade.quantity * trade.entry_price for trade in trades), ZERO)
    return traded / starting_equity


def compute_metrics(
    *,
    curve: Sequence[EquityPoint],
    trades: Sequence[ClosedTrade],
    starting_equity: Decimal,
    timeframe: Timeframe,
    total_fees: Decimal = ZERO,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Assemble the full metric set."""
    final_equity = curve[-1].equity if curve else starting_equity
    start = curve[0].timestamp if curve else None
    end = curve[-1].timestamp if curve else None
    duration_days = Decimal(str((end - start).total_seconds() / 86400)) if start and end else ZERO

    returns = period_returns(curve)
    periods = annualisation_factor(timeframe)
    drawdown, drawdown_days = max_drawdown(curve)
    stats = trade_statistics(trades)
    growth = cagr(starting_equity, final_equity, duration_days=duration_days)

    return PerformanceMetrics(
        starting_equity=starting_equity,
        final_equity=final_equity,
        total_return_pct=safe_divide(final_equity - starting_equity, starting_equity),
        cagr=growth,
        max_drawdown_pct=drawdown,
        max_drawdown_duration_days=drawdown_days,
        volatility_annual=volatility(returns, periods_per_year=periods),
        downside_volatility_annual=downside_volatility(returns, periods_per_year=periods),
        sharpe_ratio=sharpe(returns, periods_per_year=periods, risk_free_rate=risk_free_rate),
        sortino_ratio=sortino(returns, periods_per_year=periods, risk_free_rate=risk_free_rate),
        # Calmar deliberately uses CAGR over max drawdown: return per unit of the worst
        # loss actually experienced, rather than per unit of volatility.
        calmar_ratio=safe_divide(growth, drawdown),
        trade_count=int(stats["trade_count"]),
        win_count=int(stats["win_count"]),
        loss_count=int(stats["loss_count"]),
        win_rate=Decimal(stats["win_rate"]),
        profit_factor=Decimal(stats["profit_factor"]),
        expectancy=Decimal(stats["expectancy"]),
        average_win=Decimal(stats["average_win"]),
        average_loss=Decimal(stats["average_loss"]),
        largest_win=Decimal(stats["largest_win"]),
        largest_loss=Decimal(stats["largest_loss"]),
        average_holding_hours=Decimal(stats["average_holding_hours"]),
        total_fees=total_fees,
        turnover=turnover(trades, starting_equity),
        exposure_pct=exposure(curve),
        start=start,
        end=end,
        duration_days=duration_days,
        bars=len(curve),
    )


def compare(first: PerformanceMetrics, second: PerformanceMetrics) -> dict[str, float]:
    """Difference between two metric sets, for walk-forward and optimisation reports."""
    return {
        "total_return_pct": float(first.total_return_pct - second.total_return_pct),
        "sharpe_ratio": float(first.sharpe_ratio - second.sharpe_ratio),
        "max_drawdown_pct": float(first.max_drawdown_pct - second.max_drawdown_pct),
        "win_rate": float(first.win_rate - second.win_rate),
        "profit_factor": float(first.profit_factor - second.profit_factor),
    }


def degradation_ratio(in_sample: PerformanceMetrics, out_of_sample: PerformanceMetrics) -> Decimal:
    """Out-of-sample Sharpe as a fraction of in-sample Sharpe.

    The single most useful overfitting signal from a walk-forward run: a ratio near 1
    suggests the edge generalises, and a ratio near 0 (or negative) says the parameters
    were fitted to noise.
    """
    if in_sample.sharpe_ratio <= ZERO:
        return ZERO
    return out_of_sample.sharpe_ratio / in_sample.sharpe_ratio


def is_statistically_thin(
    metrics: PerformanceMetrics, *, minimum_trades: int = MIN_TRADES_FOR_SIGNIFICANCE
) -> bool:
    """Whether a result rests on too few trades to mean anything.

    A 90% win rate over 5 trades is noise. Surfacing this stops an optimiser from
    selecting parameters that happened to produce three lucky trades.
    """
    return metrics.trade_count < minimum_trades


def sharpe_confidence_interval(
    returns: Sequence[float], *, periods_per_year: float, confidence: float = 0.95
) -> tuple[Decimal, Decimal]:
    """Approximate confidence interval for the Sharpe ratio.

    Uses the standard ``sqrt((1 + S^2/2) / n)`` standard error. A point estimate without
    an interval invites treating a Sharpe of 1.2 from 40 bars as comparable to a Sharpe of
    1.1 from 4000.
    """
    if len(returns) < MIN_SAMPLES_FOR_INTERVAL:
        return ZERO, ZERO
    point = float(sharpe(returns, periods_per_year=periods_per_year))
    n = len(returns)
    standard_error = math.sqrt((1 + (point**2) / 2) / n)
    # 1.96 for 95%, 1.645 for 90%.
    z = Z_SCORE_95 if confidence >= CONFIDENCE_95 else Z_SCORE_90
    margin = z * standard_error * math.sqrt(periods_per_year) / math.sqrt(periods_per_year)
    return (
        Decimal(str(round(point - margin, 4))),
        Decimal(str(round(point + margin, 4))),
    )


ONE_HUNDRED = Decimal("100")


def as_percentage(value: Decimal) -> Decimal:
    """Convert a fraction to a percentage."""
    return value * ONE_HUNDRED


def ratio_or_zero(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Safe division helper re-exported for report templates."""
    return safe_divide(numerator, denominator, default=ZERO)


def normalised_score(metrics: PerformanceMetrics) -> Decimal:
    """A single composite score for ranking optimisation trials.

    Rewards risk-adjusted return, penalises drawdown, and heavily penalises results built
    on too few trades — an optimiser left to maximise raw return will reliably find a
    parameter set that made one enormous lucky trade.
    """
    if metrics.trade_count == 0:
        return ZERO
    base = metrics.sharpe_ratio
    drawdown_penalty = ONE + metrics.max_drawdown_pct * Decimal("2")
    sample_penalty = (
        Decimal(str(min(1.0, metrics.trade_count / MIN_TRADES_FOR_SIGNIFICANCE)))
        if metrics.trade_count < MIN_TRADES_FOR_SIGNIFICANCE
        else ONE
    )
    return safe_divide(base, drawdown_penalty) * sample_penalty
