"""How much capital a candidate earns, from how strongly the evidence supports it.

Every position this account has opened was the same size — ~2,485 notional on BNB, on SOL,
on XRP, regardless of how the candidate scored. A setup the engine rated 0.53 got exactly
the capital of one it rated 0.74.

**Thresholds are the observed distribution, not round numbers.** Measured over 371 selected
candidates from this engine's own decision log:

    min 0.532 · p25 0.650 · median 0.658 · p75 0.673 · p90 0.694 · max 0.736

Percentiles rather than absolutes because the score is a relative ranking, not a calibrated
probability. A fixed cut like "0.8 is strong" would classify every candidate this engine
has ever produced as weak and stop trading altogether.

**The gradient is deliberately shallow, and that is the point.** Two facts govern it. The
distribution is tight — half of all candidates fall within a 2.3% band — so the score
separates the top decile from the bottom quartile and very little in between. And live
expectancy is currently **−2.11 per trade**: sizing amplifies whatever edge the ranking
has, and amplifying a negative one just loses faster. So the spread starts near 1.0 and
widens only when live attribution shows high-conviction trades actually outperforming.

**Size never increases on a negative expected edge.** A candidate can be in the top decile
of a weak field and still not be worth more capital; conviction says *better than the
others*, not *profitable*. Only positive expected net edge unlocks the upper multipliers,
and reduction is always permitted because trusting a marginal setup less is safe in a way
that trusting it more is not.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from quantflow.core.precision import ONE, ZERO

#: Observed percentiles of the orchestrator's score for *selected* candidates, measured
#: 2026-08-15 over 371 decisions. Recorded here so the thresholds can be re-derived rather
#: than trusted: if the engine's scoring changes, these stop describing it.
OBSERVED_PERCENTILES: dict[str, Decimal] = {
    "min": Decimal("0.532"),
    "p25": Decimal("0.650"),
    "p50": Decimal("0.658"),
    "p75": Decimal("0.673"),
    "p90": Decimal("0.694"),
    "max": Decimal("0.736"),
}

WEAK_BELOW = OBSERVED_PERCENTILES["p25"]
NORMAL_BELOW = OBSERVED_PERCENTILES["p50"]
STRONG_BELOW = OBSERVED_PERCENTILES["p90"]


class Conviction(StrEnum):
    """How strongly the evidence backs a candidate, relative to what this engine produces."""

    WEAK = "WEAK"
    NORMAL = "NORMAL"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


#: Capital multipliers. Shallow on purpose — see the module docstring.
#:
#: NORMAL is exactly 1.0 so the median candidate is sized as it is today and the change is a
#: redistribution rather than a blanket increase in risk. VERY_STRONG at 1.30 puts ~25% more
#: on a top-decile setup, which is a real difference without letting one position dominate a
#: 10,000 allocation. WEAK at 0.80 is the only unconditional move, because sizing down a
#: marginal candidate cannot make things worse.
DEFAULT_MULTIPLIERS: dict[Conviction, Decimal] = {
    Conviction.WEAK: Decimal("0.80"),
    Conviction.NORMAL: ONE,
    Conviction.STRONG: Decimal("1.15"),
    Conviction.VERY_STRONG: Decimal("1.30"),
}


def classify(score: Decimal) -> Conviction:
    """Place a score in the distribution of what this engine selects."""
    if score < WEAK_BELOW:
        return Conviction.WEAK
    if score < NORMAL_BELOW:
        return Conviction.NORMAL
    if score < STRONG_BELOW:
        return Conviction.STRONG
    return Conviction.VERY_STRONG


def percentile_of(score: Decimal) -> str:
    """Where a score falls in the observed distribution, for the decision log."""
    if score < OBSERVED_PERCENTILES["p25"]:
        return "<p25"
    if score < OBSERVED_PERCENTILES["p50"]:
        return "p25-p50"
    if score < OBSERVED_PERCENTILES["p75"]:
        return "p50-p75"
    if score < OBSERVED_PERCENTILES["p90"]:
        return "p75-p90"
    return ">=p90"


def allocation_fraction(
    score: Decimal | None,
    *,
    expected_net_edge: Decimal | None = None,
    multipliers: dict[Conviction, Decimal] | None = None,
) -> tuple[Conviction, Decimal]:
    """The share of the maximum allowed position a candidate earns, in ``[0, 1]``.

    This is the form the sizer wants, and using it is what makes conviction and the
    position cap one model instead of two that contradict each other.

    The sizer's pipeline is ``raw -> conviction scaling -> hard caps -> rounding``: it
    attenuates by conviction and *then* clamps to ``max_position_pct``. Applying a
    multiplier after the sizer — as was done first — produces a size the very next layer
    rejects, because the sizer has already placed the position at the cap. That is not a
    limit doing its job, it is two components disagreeing: the live session selected three
    candidates and placed none, every order refused for "would reach 22.99% of equity,
    above the 20.00% limit".

    Expressed as a fraction the contradiction cannot occur. The strongest tier maps to
    1.0 — the full cap, never beyond — and weaker tiers take proportionally less. So
    conviction still decides how much capital a setup earns, and no tier can produce a
    size that fails the check that follows it.
    """
    # No ranking means no opinion, so nothing is attenuated. Scaling by the tier table
    # here would quietly shrink every trade that ran outside the orchestrator by the
    # distance between NORMAL and the top tier.
    if score is None or score <= ZERO:
        return Conviction.NORMAL, ONE

    tier, multiplier = size_multiplier(
        score, expected_net_edge=expected_net_edge, multipliers=multipliers
    )
    table = multipliers or DEFAULT_MULTIPLIERS
    ceiling = max(table.values())
    if ceiling <= ZERO:
        return tier, ONE
    return tier, min(ONE, multiplier / ceiling)


def size_multiplier(
    score: Decimal | None,
    *,
    expected_net_edge: Decimal | None = None,
    multipliers: dict[Conviction, Decimal] | None = None,
) -> tuple[Conviction, Decimal]:
    """The tier and capital multiplier for a candidate.

    Args:
        score: The orchestrator's score, or ``None`` when a strategy ran outside the
            orchestrator and produced no ranking.
        expected_net_edge: Expected edge after estimated costs. Anything at or below zero
            caps the multiplier at 1.0 — a candidate that is not expected to pay for its
            own execution does not earn extra capital however well it ranks.
        multipliers: Override the gradient. Exposed so it can be widened from live evidence
            without touching the strategy layer.

    Returns:
        ``(tier, multiplier)``. An absent score yields NORMAL at 1.0: unknown conviction is
        sized exactly as today rather than guessed in either direction.

    """
    if score is None or score <= ZERO:
        return Conviction.NORMAL, ONE

    table = multipliers or DEFAULT_MULTIPLIERS
    tier = classify(score)
    multiplier = table.get(tier, ONE)

    # Increases are earned by expected profitability, not by ranking alone. Reductions are
    # always allowed: sizing down a marginal candidate is safe in a way sizing up is not.
    if multiplier > ONE and (expected_net_edge is None or expected_net_edge <= ZERO):
        return tier, ONE
    return tier, multiplier


__all__ = [
    "DEFAULT_MULTIPLIERS",
    "NORMAL_BELOW",
    "OBSERVED_PERCENTILES",
    "STRONG_BELOW",
    "WEAK_BELOW",
    "Conviction",
    "classify",
    "percentile_of",
    "size_multiplier",
]
