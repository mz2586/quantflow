"""Why a strategy failed — the cause, not the symptom.

"Sharpe 0.42, rejected" tells a researcher nothing they can act on. The actionable
question is *what to change*, and the honest answers are few:

* **Costs** — the signal had an edge and the venue took it. Trade less often, or trade
  at maker prices. The strategy is not the problem.
* **Frequency** — profitable per trade before costs, but trading so often that even
  correct fills cannot pay for themselves. Same fix, different evidence.
* **Signal** — negative even with zero fees and zero slippage. No amount of execution
  work saves this; the idea is wrong.
* **Risk of ruin** — profitable but with a drawdown or a losing streak nobody could sit
  through. Size it down or discard it; the returns are not the binding constraint.
* **Sample** — too few trades to distinguish skill from luck either way. Not a verdict.

The distinction between the first three is made by *re-running the same strategy on the
same data with costs removed*. That comparison is the only way to separate "the signal is
worthless" from "the signal is fine and the fees ate it", and those two lead to opposite
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from quantflow.backtest.metrics import PerformanceMetrics
from quantflow.core.precision import ZERO, safe_divide


class FailureCause(StrEnum):
    """The primary reason a strategy failed, in the order they are tested."""

    NONE = "none"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    NO_SIGNAL = "no_signal"
    COSTS = "costs"
    OVER_TRADING = "over_trading"
    RISK_OF_RUIN = "risk_of_ruin"
    MARGINAL = "marginal"


#: Trades below which no verdict is attempted.
MIN_SAMPLE = 30

#: Share of gross profit consumed by costs past which costs are the story.
COST_DOMINANCE = Decimal("0.40")

#: Gross return per trade below which the edge is too thin to survive any friction.
THIN_EDGE_PER_TRADE = Decimal("0.001")


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """A machine-readable cause plus a sentence a person can act on."""

    cause: FailureCause
    explanation: str
    #: What to try next, or why there is nothing worth trying.
    recommendation: str
    #: Net return if the same strategy had paid nothing to trade. Present only when the
    #: cost-free comparison was run.
    frictionless_return: Decimal | None = None
    #: Costs as a share of gross profit, when gross profit was positive.
    cost_share: Decimal | None = None
    #: Average gross return per trade before costs.
    edge_per_trade: Decimal | None = None

    @property
    def is_fixable_by_execution(self) -> bool:
        """Whether better execution could plausibly rescue this.

        True only when the signal made money before costs. Everything else needs a
        different signal, not a different broker.
        """
        return self.cause in {FailureCause.COSTS, FailureCause.OVER_TRADING}

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "cause": str(self.cause),
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "fixable_by_execution": self.is_fixable_by_execution,
            "frictionless_return": (
                str(self.frictionless_return) if self.frictionless_return is not None else None
            ),
            "cost_share": str(self.cost_share) if self.cost_share is not None else None,
            "edge_per_trade": (
                str(self.edge_per_trade) if self.edge_per_trade is not None else None
            ),
        }


def diagnose(  # noqa: PLR0911 - one branch per cause is clearer than nesting
    metrics: PerformanceMetrics,
    *,
    frictionless: PerformanceMetrics | None = None,
    max_drawdown: Decimal = Decimal("0.35"),
) -> Diagnosis:
    """Determine the primary cause of failure.

    Args:
        metrics: The result under realistic costs.
        frictionless: The same strategy on the same data with no fees and no slippage.
            Without it the cost-versus-signal distinction cannot be made, and the
            diagnosis says so rather than guessing.
        max_drawdown: The drawdown ceiling the strategy was judged against.

    """
    if metrics.trade_count < MIN_SAMPLE:
        return Diagnosis(
            cause=FailureCause.INSUFFICIENT_SAMPLE,
            explanation=(
                f"{metrics.trade_count} trades is too few to tell skill from luck in "
                "either direction"
            ),
            recommendation=(
                "Test over a longer period or a faster timeframe before drawing any "
                "conclusion. This is not a rejection of the idea."
            ),
        )

    cost_share = _cost_share(metrics)
    edge = _edge_per_trade(metrics, frictionless)

    # Profitable but unholdable: the returns are not the binding constraint, so this is
    # checked before anything about signal quality.
    if metrics.total_return_pct > ZERO and metrics.max_drawdown_pct > max_drawdown:
        return Diagnosis(
            cause=FailureCause.RISK_OF_RUIN,
            explanation=(
                f"made {metrics.total_return_pct:.2%} but drew down "
                f"{metrics.max_drawdown_pct:.2%}, past the {max_drawdown:.0%} ceiling"
            ),
            recommendation=(
                "Reduce position size or tighten the stop. The edge may be real; the "
                "sizing is not survivable."
            ),
            cost_share=cost_share,
            edge_per_trade=edge,
        )

    if frictionless is None:
        return Diagnosis(
            cause=(
                FailureCause.MARGINAL if metrics.total_return_pct > ZERO else FailureCause.NO_SIGNAL
            ),
            explanation=(
                f"net {metrics.total_return_pct:.2%} with no cost-free comparison "
                "available, so costs and signal cannot be separated"
            ),
            recommendation=(
                "Re-run with the zero_cost model to establish whether the signal has any "
                "edge before costs."
            ),
            cost_share=cost_share,
            edge_per_trade=edge,
        )

    # The decisive comparison: did the signal make money before anyone was paid?
    if frictionless.total_return_pct <= ZERO:
        return Diagnosis(
            cause=FailureCause.NO_SIGNAL,
            explanation=(
                f"lost {frictionless.total_return_pct:.2%} even with zero fees and zero "
                "slippage; the signal itself has no edge"
            ),
            recommendation=(
                "Discard or redesign. No execution improvement can rescue a signal that "
                "loses money for free."
            ),
            frictionless_return=frictionless.total_return_pct,
            cost_share=cost_share,
            edge_per_trade=edge,
        )

    # The signal made money before costs. Whether the fix is "trade better" or "trade
    # less" comes down to how thin the per-trade edge was.
    if edge is not None and edge < THIN_EDGE_PER_TRADE:
        return Diagnosis(
            cause=FailureCause.OVER_TRADING,
            explanation=(
                f"gross edge of {edge:.4%} per trade over {metrics.trade_count} trades "
                f"cannot cover costs; frictionless return was "
                f"{frictionless.total_return_pct:.2%} against {metrics.total_return_pct:.2%} net"
            ),
            recommendation=(
                "The edge per trade is smaller than the cost of taking it. Trade a slower "
                "timeframe or filter entries harder; execution alone will not close this."
            ),
            frictionless_return=frictionless.total_return_pct,
            cost_share=cost_share,
            edge_per_trade=edge,
        )

    if cost_share is not None and cost_share >= COST_DOMINANCE:
        return Diagnosis(
            cause=FailureCause.COSTS,
            explanation=(
                f"costs took {cost_share:.1%} of gross profit; the same signal returned "
                f"{frictionless.total_return_pct:.2%} before costs against "
                f"{metrics.total_return_pct:.2%} after"
            ),
            recommendation=(
                "The signal works and the venue is taking it. Try maker-only entries or a "
                "slower timeframe before changing the logic."
            ),
            frictionless_return=frictionless.total_return_pct,
            cost_share=cost_share,
            edge_per_trade=edge,
        )

    return Diagnosis(
        cause=FailureCause.MARGINAL,
        explanation=(
            f"net {metrics.total_return_pct:.2%} against "
            f"{frictionless.total_return_pct:.2%} frictionless; positive before costs but "
            "not by enough to clear the thresholds"
        ),
        recommendation=(
            "Nothing single-cause to fix. Treat as a weak idea rather than a broken one."
        ),
        frictionless_return=frictionless.total_return_pct,
        cost_share=cost_share,
        edge_per_trade=edge,
    )


def _cost_share(metrics: PerformanceMetrics) -> Decimal | None:
    """Fees as a fraction of gross profit, or ``None`` when there was none.

    Gross is reconstructed as net plus fees: a strategy netting 100 having paid 900 in
    fees earned 1000 gross and handed 90% of it to the venue.
    """
    net = metrics.final_equity - metrics.starting_equity
    gross = net + metrics.total_fees
    if gross <= ZERO:
        return None
    return metrics.total_fees / gross


def _edge_per_trade(
    metrics: PerformanceMetrics, frictionless: PerformanceMetrics | None
) -> Decimal | None:
    """Average gross return per trade, from the cost-free run when available.

    Measured frictionless because a per-trade figure computed after costs already has the
    answer baked into it and cannot distinguish the two cases.
    """
    source = frictionless or metrics
    if source.trade_count == 0:
        return None
    return safe_divide(source.total_return_pct, Decimal(source.trade_count))


__all__ = [
    "COST_DOMINANCE",
    "MIN_SAMPLE",
    "THIN_EDGE_PER_TRADE",
    "Diagnosis",
    "FailureCause",
    "diagnose",
]
