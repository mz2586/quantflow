"""Trade selection: the layer that decides whether an opportunity is worth taking at all.

``scoring.gate_candidate`` already asks whether a candidate is economic in isolation —
reward:risk, edge after costs, position caps. This asks the questions that only make sense
across candidates and across history:

* Is anything **else** saying the same thing, from a genuinely different information source?
* Has this strategy shown anything in the regime that is actually prevailing right now?
* Is this a new position, or the one already held wearing a different ticker?
* Can the size be traded without the order becoming its own adverse price move?

Every threshold below is declared here, once, with the reasoning for it. None was chosen by
trying values and keeping the one that improved a backtest — that would be fitting a
selection layer to the same history it is supposed to generalise beyond.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from quantflow.core.precision import ZERO
from quantflow.orchestrator.scoring import SOLO_FAMILY_MIN_NET_EDGE

#: Independent information sources that must agree before an entry is allowed.
#:
#: Two, not one: a single indicator crossing a line is the weakest evidence the system can
#: produce, and it was previously sufficient. Not three: with five families and most
#: strategies clustered in trend and reversion, three would refuse nearly everything and
#: the layer would stop being selective and start being inert.
MIN_INDEPENDENT_FAMILIES = 2

#: Trades in a regime before that regime's expectancy is allowed to veto an entry.
#:
#: Below this a bad average is noise. The rule is to penalise persistent failure, not to
#: retire a strategy for a losing streak that any positive-expectancy system produces.
MIN_SAMPLES_FOR_EXPECTANCY = 20

#: Shrinkage constant: the sample size at which an observed expectancy is pulled halfway
#: toward neutral. Standard empirical-Bayes shrinkage, not a fitted value.
SHRINKAGE_PRIOR_WEIGHT = Decimal("30")

#: Maximum correlation with an existing position. Above this the "new" position is the old
#: one at double size, and the portfolio's diversification is imaginary.
MAX_CORRELATION_WITH_BOOK = Decimal("0.85")

#: Maximum share of a bar's traded volume one order may take. Far below the fill model's
#: 10% rejection threshold: the point is not to be *accepted*, it is to not be the reason
#: the price moved.
MAX_VOLUME_SHARE = Decimal("0.02")

#: Strategy families, by the information each one actually reads. Strategies inside a
#: family are variations on one observation: two moving-average crossovers do not become
#: two independent opinions by using different periods.
_FAMILIES: dict[str, str] = {
    # Trend: direction of a smoothed price path.
    "ema_cross": "trend",
    "triple_ma": "trend",
    "macd_trend": "trend",
    "adx_trend": "trend",
    "keltner_trend": "trend",
    "regime_adaptive": "trend",
    # Breakout: range expansion beyond a prior boundary.
    "donchian_breakout": "breakout",
    "dual_thrust": "breakout",
    "opening_range_breakout": "breakout",
    "bollinger_squeeze": "breakout",
    "atr_expansion": "breakout",
    # Mean reversion: distance from a central value.
    "rsi_reversion": "reversion",
    "bollinger_reversion": "reversion",
    "zscore_reversion": "reversion",
    "keltner_reversion": "reversion",
    "stochastic_reversion": "reversion",
    "vwap_reversion": "reversion",
    # Momentum: rate of change, not direction of a mean.
    "momentum_roc": "momentum",
    "vol_adjusted_momentum": "momentum",
    "vwap_momentum": "momentum",
    # Volume: participation, an input none of the above reads.
    "obv_trend": "volume",
    "volume_breakout": "volume",
    # --- added with the library expansion ---------------------------------
    # Trend: direction of a smoothed or banded price path.
    "supertrend": "trend",
    "ichimoku_trend": "trend",
    "parabolic_sar": "trend",
    "mtf_trend": "trend",
    "pullback_continuation": "trend",
    # Structure: swing pivots read directly, not through an indicator. A different
    # observation of the same tape, so it corroborates trend rather than echoing it.
    "swing_structure": "structure",
    # Momentum: rate of change and its derivatives.
    "momentum_acceleration": "momentum",
    "normalized_momentum": "momentum",
    "relative_momentum": "momentum",
    # Mean reversion: distance from a central value, by various metrics.
    "percentile_reversion": "reversion",
    "volatility_normalized_reversion": "reversion",
    "ma_deviation_reversion": "reversion",
    # Breakout: range expansion beyond a prior boundary.
    "breakout_retest": "breakout",
    "atr_breakout": "breakout",
    "range_expansion": "breakout",
    "support_resistance_breakout": "breakout",
    # Volume: participation, which no price-only strategy reads.
    "accumulation_distribution": "volume",
    "money_flow_index": "volume",
    "volume_price_divergence": "volume",
    # Volatility: the regime itself as the signal, not price direction.
    "volatility_regime": "volatility",
    "volatility_transition": "volatility",
}


def strategy_family(strategy_id: str) -> str:
    """The information source a strategy reads.

    An unmapped strategy gets a family of its own rather than a default one. Dropping a new
    strategy into an existing family would silently let it corroborate strategies it has
    nothing in common with, which is the exact failure this grouping exists to prevent.
    """
    return _FAMILIES.get(strategy_id, f"unmapped:{strategy_id}")


def shrunk_expectancy(observed: Decimal, *, samples: int) -> Decimal:
    """Pull an observed expectancy toward zero in proportion to how little it rests on.

    ``observed * n / (n + prior)``. At the prior weight the estimate is halved; by a few
    hundred trades it is essentially untouched. This is what stops a six-trade losing run
    from reading as a verdict, without ever flipping the sign of a real one.
    """
    if samples <= 0:
        return ZERO
    weight = Decimal(samples) / (Decimal(samples) + SHRINKAGE_PRIOR_WEIGHT)
    return observed * weight


@dataclass(frozen=True, slots=True)
class SelectionInputs:
    """Everything the selection layer needs, already measured elsewhere.

    Deliberately plain values rather than live objects: this module makes a judgement and
    must not be able to reach out for data, which is how look-ahead gets in.
    """

    strategy_id: str
    agreeing_families: int
    regime_expectancy: Decimal | None
    regime_samples: int
    max_correlation: Decimal
    volume_share: Decimal
    #: Distinct families the *pool* can produce at all. The confluence requirement is
    #: capped by this: demanding corroboration from sources that do not exist would not
    #: make the system selective, it would make it inert. A single-strategy pool can only
    #: ever offer one opinion, and the honest response is to judge that opinion on its own
    #: merits, not to refuse every trade forever.
    available_families: int = MIN_INDEPENDENT_FAMILIES
    #: Expected edge after round-trip costs, when it could be measured.
    #:
    #: Lets a strong lone opinion stand in for the missing second family — see
    #: :data:`~quantflow.orchestrator.scoring.SOLO_FAMILY_MIN_NET_EDGE`. ``None`` means the
    #: edge is unknown, and an unknown edge never buys a waiver.
    net_edge: Decimal | None = None


@dataclass(frozen=True, slots=True)
class SelectionVerdict:
    """Whether to take the trade, why not, and how good it is if so."""

    accepted: bool
    score: Decimal
    #: When refused, every objection. When accepted, any waiver that let it through — so a
    #: trade taken on one family's opinion says so in the decision log rather than being
    #: indistinguishable from a corroborated one.
    reasons: tuple[str, ...] = field(default_factory=tuple)


def assess_candidate(inputs: SelectionInputs) -> SelectionVerdict:
    """Accept or refuse one candidate, and score it if accepted.

    Every failing condition is collected rather than returning on the first: an operator
    reading a refusal should see the whole picture, not fix one objection and rediscover
    the next on the following bar.
    """
    reasons: list[str] = []
    waivers: list[str] = []

    required = min(MIN_INDEPENDENT_FAMILIES, max(1, inputs.available_families))
    if inputs.agreeing_families < required:
        # A lone family may still carry the trade, but only by paying for the corroboration
        # it lacks: twice the ordinary edge floor. Everything else about the candidate has
        # already been established upstream — reward:risk, a valid stop and target,
        # liquidity, and an edge past the floor — so what is missing here is a second
        # opinion, not soundness. An unmeasurable edge buys nothing.
        edge = inputs.net_edge
        if edge is not None and edge >= SOLO_FAMILY_MIN_NET_EDGE:
            # Recorded as a waiver, NOT as a reason: `reasons` is what refuses a candidate,
            # and appending here would reject the very trade being allowed through.
            waivers.append(
                f"confluence waived on {inputs.agreeing_families} family: net edge "
                f"{edge:.4%} clears the {SOLO_FAMILY_MIN_NET_EDGE:.4%} solo bar"
            )
        else:
            observed = f"{edge:.4%}" if edge is not None else "unmeasured"
            reasons.append(
                f"confluence {inputs.agreeing_families} below the "
                f"{required} independent families required, and net edge {observed} "
                f"does not reach the {SOLO_FAMILY_MIN_NET_EDGE:.4%} solo bar"
            )

    # Expectancy vetoes only on a sample large enough to mean something.
    expectancy = ZERO
    if inputs.regime_expectancy is not None and inputs.regime_samples > 0:
        expectancy = shrunk_expectancy(inputs.regime_expectancy, samples=inputs.regime_samples)
        if inputs.regime_samples >= MIN_SAMPLES_FOR_EXPECTANCY and expectancy < ZERO:
            reasons.append(
                f"regime expectancy {expectancy:.3f} over {inputs.regime_samples} trades "
                "is negative in the prevailing regime"
            )

    if inputs.max_correlation > MAX_CORRELATION_WITH_BOOK:
        reasons.append(
            f"correlation {inputs.max_correlation:.2f} with an open position exceeds "
            f"{MAX_CORRELATION_WITH_BOOK}"
        )

    if inputs.volume_share > MAX_VOLUME_SHARE:
        reasons.append(
            f"order is {inputs.volume_share:.2%} of bar volume, above the "
            f"{MAX_VOLUME_SHARE:.2%} liquidity ceiling"
        )

    if reasons:
        return SelectionVerdict(accepted=False, score=ZERO, reasons=tuple(reasons))

    # Score the survivors. Confluence beyond the minimum and demonstrated expectancy in
    # this regime both raise it; correlation with the book lowers it, so that between two
    # otherwise equal candidates the more diversifying one wins.
    # A waived candidate scores BELOW a corroborated one: the bonus is negative when only
    # one family agreed. It is allowed to compete, not promoted to the front of the queue.
    confluence_bonus = Decimal(inputs.agreeing_families - MIN_INDEPENDENT_FAMILIES)
    score = Decimal("1") + confluence_bonus + expectancy - inputs.max_correlation
    return SelectionVerdict(accepted=True, score=score, reasons=tuple(waivers))


__all__ = [
    "MAX_CORRELATION_WITH_BOOK",
    "MAX_VOLUME_SHARE",
    "MIN_INDEPENDENT_FAMILIES",
    "MIN_SAMPLES_FOR_EXPECTANCY",
    "SHRINKAGE_PRIOR_WEIGHT",
    "SelectionInputs",
    "SelectionVerdict",
    "assess_candidate",
    "shrunk_expectancy",
    "strategy_family",
]
