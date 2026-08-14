"""Acceptance thresholds: the gate that rejects a strategy before anyone gets attached.

The purpose of a threshold set fixed *in advance* is to remove the researcher from the
decision. Once results are on screen it is trivially easy to justify a strategy that
missed a bar — "the drawdown was one bad month", "the sample is short but the Sharpe is
good" — and that is how a losing strategy reaches production. These rules are declared
first, applied mechanically, and every failure is recorded with the number that caused it.

A rejected strategy is not deleted. It stays in the report with its reasons attached: a
record of what was tried and why it failed is the most reusable output a research process
produces, and re-testing an idea someone already killed is pure waste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quantflow.backtest.metrics import PerformanceMetrics


class RejectionCode(StrEnum):
    """Why a strategy failed the gate."""

    NEGATIVE_RETURN = "negative_return"
    INSUFFICIENT_RETURN = "insufficient_return"
    LOW_PROFIT_FACTOR = "low_profit_factor"
    LOW_SHARPE = "low_sharpe"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    LOW_WIN_RATE = "low_win_rate"
    TOO_FEW_TRADES = "too_few_trades"
    TOO_MANY_TRADES = "too_many_trades"
    FEE_DOMINATED = "fee_dominated"
    LOST_TO_BENCHMARK = "lost_to_benchmark"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class Rejection:
    """One failed criterion, with the numbers that failed it."""

    code: RejectionCode
    detail: str
    observed: Decimal | None = None
    threshold: Decimal | None = None

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    """Minimum standards a strategy must meet to be considered a candidate.

    Every default here is deliberately demanding. The cost of passing a bad strategy is
    real money; the cost of rejecting a good one is another research cycle.
    """

    #: Net return over the whole period must clear this, after costs.
    min_net_return: Decimal = Decimal("0.10")
    #: Gross profit / gross loss. Below ~1.3 there is no margin for the live-vs-backtest
    #: gap, and that gap is always worse than expected.
    min_profit_factor: Decimal = Decimal("1.30")
    #: Annualised Sharpe. Below 0.5 the equity curve is not distinguishable from luck at
    #: any sample size a retail researcher can obtain.
    min_sharpe: Decimal = Decimal("0.50")
    #: Peak-to-trough decline. A drawdown nobody can sit through is not a strategy.
    max_drawdown: Decimal = Decimal("0.35")
    #: Win rate floor. Set low on purpose — trend systems win rarely and win big, and a
    #: high floor would reject the entire family for the wrong reason.
    min_win_rate: Decimal = Decimal("0.25")
    #: Below this the result is a handful of lucky trades wearing a track record.
    min_trades: int = 30
    #: Above this the strategy is a fee-generation machine; the venue is the only winner.
    max_trades: int = 5_000
    #: Fees as a share of gross profit. Past this the venue takes most of the edge and
    #: any small worsening in fills flips the strategy negative.
    max_fee_share_of_gross_profit: Decimal = Decimal("0.50")
    #: Must beat buy-and-hold on the same data under the same costs. A strategy that
    #: takes on execution risk, parameter risk and operational risk to underperform
    #: holding the asset is worse than doing nothing.
    must_beat_benchmark: bool = True

    def evaluate(
        self,
        metrics: PerformanceMetrics,
        *,
        benchmark_return: Decimal | None = None,
    ) -> tuple[Rejection, ...]:
        """Every criterion this result fails.

        Returns all failures rather than the first: knowing a strategy missed on one
        metric is a tuning problem, knowing it missed on five is a dead end, and stopping
        at the first failure hides the difference.
        """
        failures: list[Rejection] = []

        if metrics.total_return_pct <= 0:
            failures.append(
                Rejection(
                    RejectionCode.NEGATIVE_RETURN,
                    f"lost money: {metrics.total_return_pct:.2%} net return",
                    metrics.total_return_pct,
                    Decimal("0"),
                )
            )
        elif metrics.total_return_pct < self.min_net_return:
            failures.append(
                Rejection(
                    RejectionCode.INSUFFICIENT_RETURN,
                    f"net return {metrics.total_return_pct:.2%} below the "
                    f"{self.min_net_return:.2%} minimum",
                    metrics.total_return_pct,
                    self.min_net_return,
                )
            )

        if metrics.profit_factor < self.min_profit_factor:
            failures.append(
                Rejection(
                    RejectionCode.LOW_PROFIT_FACTOR,
                    f"profit factor {metrics.profit_factor:.2f} below {self.min_profit_factor:.2f}",
                    metrics.profit_factor,
                    self.min_profit_factor,
                )
            )

        if metrics.sharpe_ratio < self.min_sharpe:
            failures.append(
                Rejection(
                    RejectionCode.LOW_SHARPE,
                    f"Sharpe {metrics.sharpe_ratio:.2f} below {self.min_sharpe:.2f}",
                    metrics.sharpe_ratio,
                    self.min_sharpe,
                )
            )

        if metrics.max_drawdown_pct > self.max_drawdown:
            failures.append(
                Rejection(
                    RejectionCode.EXCESSIVE_DRAWDOWN,
                    f"max drawdown {metrics.max_drawdown_pct:.2%} exceeds {self.max_drawdown:.2%}",
                    metrics.max_drawdown_pct,
                    self.max_drawdown,
                )
            )

        if metrics.trade_count < self.min_trades:
            failures.append(
                Rejection(
                    RejectionCode.TOO_FEW_TRADES,
                    f"{metrics.trade_count} trades is too few to distinguish skill from "
                    f"luck (minimum {self.min_trades})",
                    Decimal(metrics.trade_count),
                    Decimal(self.min_trades),
                )
            )
        elif metrics.trade_count > self.max_trades:
            failures.append(
                Rejection(
                    RejectionCode.TOO_MANY_TRADES,
                    f"{metrics.trade_count} trades exceeds {self.max_trades}; this is a "
                    "fee-generation machine",
                    Decimal(metrics.trade_count),
                    Decimal(self.max_trades),
                )
            )
        # Win rate is only meaningful once the sample is large enough to have one.
        elif metrics.win_rate < self.min_win_rate:
            failures.append(
                Rejection(
                    RejectionCode.LOW_WIN_RATE,
                    f"win rate {metrics.win_rate:.2%} below {self.min_win_rate:.2%}",
                    metrics.win_rate,
                    self.min_win_rate,
                )
            )

        fee_share = _fee_share_of_gross_profit(metrics)
        if fee_share is not None and fee_share > self.max_fee_share_of_gross_profit:
            failures.append(
                Rejection(
                    RejectionCode.FEE_DOMINATED,
                    f"fees consumed {fee_share:.1%} of gross profit, above the "
                    f"{self.max_fee_share_of_gross_profit:.0%} ceiling",
                    fee_share,
                    self.max_fee_share_of_gross_profit,
                )
            )

        if (
            self.must_beat_benchmark
            and benchmark_return is not None
            and metrics.total_return_pct <= benchmark_return
        ):
            failures.append(
                Rejection(
                    RejectionCode.LOST_TO_BENCHMARK,
                    f"net return {metrics.total_return_pct:.2%} did not beat buy-and-hold "
                    f"at {benchmark_return:.2%}",
                    metrics.total_return_pct,
                    benchmark_return,
                )
            )

        return tuple(failures)

    def describe(self) -> dict[str, str]:
        """The thresholds as display strings, for the report header."""
        return {
            "net return": f"≥ {self.min_net_return:.1%}",
            "profit factor": f"≥ {self.min_profit_factor:.2f}",
            "Sharpe ratio": f"≥ {self.min_sharpe:.2f}",
            "max drawdown": f"≤ {self.max_drawdown:.1%}",
            "win rate": f"≥ {self.min_win_rate:.1%}",
            "trades": f"{self.min_trades} to {self.max_trades}",
            "fees / gross profit": f"≤ {self.max_fee_share_of_gross_profit:.0%}",
            "beats buy-and-hold": "required" if self.must_beat_benchmark else "not required",
        }


def _fee_share_of_gross_profit(metrics: PerformanceMetrics) -> Decimal | None:
    """Fees as a fraction of gross profit, or ``None`` when there was no gross profit.

    Gross profit is reconstructed as net plus fees: a strategy that made 100 net having
    paid 900 in fees earned 1000 gross and gave 90% of it away, and the ratio is the
    cleanest single number for spotting that.
    """
    net = metrics.final_equity - metrics.starting_equity
    gross = net + metrics.total_fees
    if gross <= 0:
        return None
    return metrics.total_fees / gross


#: The default gate. Named so a report can state which standard was applied.
DEFAULT_THRESHOLDS: AcceptanceThresholds = AcceptanceThresholds()


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """The outcome of applying the gate to one strategy result."""

    accepted: bool
    rejections: tuple[Rejection, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        """One line explaining the verdict."""
        if self.accepted:
            return "accepted"
        return "; ".join(rejection.detail for rejection in self.rejections)


def screen(
    metrics: PerformanceMetrics,
    thresholds: AcceptanceThresholds = DEFAULT_THRESHOLDS,
    *,
    benchmark_return: Decimal | None = None,
) -> ScreenResult:
    """Apply the gate. Accepted only when nothing failed."""
    rejections = thresholds.evaluate(metrics, benchmark_return=benchmark_return)
    return ScreenResult(accepted=not rejections, rejections=rejections)
