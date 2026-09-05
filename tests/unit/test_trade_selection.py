"""A selectivity layer above the strategies: fewer trades, better ones.

The existing gate already asks whether a *candidate* is economic — reward:risk, edge after
costs, per-strategy position caps. What it never asked is whether the candidate is
*corroborated*, whether its strategy has shown anything in the regime actually prevailing,
or whether it duplicates a position already held. So a single weak indicator could open a
trade on its own, in a regime where its strategy has never worked, alongside two other
positions expressing the same view.

Three additions, each answering one of those:

**Confluence.** Signals are grouped into families by information source — trend, momentum,
mean reversion, volatility, volume. Two strategies inside one family agreeing is one piece
of evidence counted twice: two moving-average crossovers are the same observation with
different periods. Agreement is only counted *across* families.

**Regime-conditional expectancy.** ``PerformanceMemory.for_regime`` already buckets results
by regime and was never consulted; the gate used the blended overall record. A trend
follower's lifetime average says little about what it does in chop.

**Correlation.** Two candidates on correlated symbols are one position in two names, at
twice the size and with none of the diversification the position count implies.

Small samples are handled by shrinking toward neutral rather than by disabling: an
expectancy from six trades is noise, and treating it as evidence in either direction is the
error to avoid. A strategy is penalised only once its sample is large enough for the
penalty to mean something, and it recovers as soon as its results do.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.orchestrator.scoring import SOLO_FAMILY_MIN_NET_EDGE
from quantflow.orchestrator.selection import (
    MIN_INDEPENDENT_FAMILIES,
    MIN_SAMPLES_FOR_EXPECTANCY,
    SelectionInputs,
    assess_candidate,
    shrunk_expectancy,
    strategy_family,
)


def inputs(**overrides: object) -> SelectionInputs:
    base: dict[str, object] = {
        "strategy_id": "ema_cross",
        "agreeing_families": 2,
        "regime_expectancy": Decimal("0.2"),
        "regime_samples": 40,
        "max_correlation": Decimal("0.1"),
        "volume_share": Decimal("0.01"),
    }
    base.update(overrides)
    return SelectionInputs(**base)  # type: ignore[arg-type]


class TestFamilies:
    def test_moving_average_strategies_share_a_family(self) -> None:
        """Two crossovers are one observation, not two."""
        assert strategy_family("ema_cross") == strategy_family("triple_ma")

    def test_trend_and_reversion_are_different_families(self) -> None:
        assert strategy_family("ema_cross") != strategy_family("rsi_reversion")

    def test_volume_is_its_own_information_source(self) -> None:
        assert strategy_family("obv_trend") != strategy_family("ema_cross")

    def test_unknown_strategy_gets_its_own_family(self) -> None:
        """An unmapped strategy must not silently join an existing family."""
        assert strategy_family("something_new") == "unmapped:something_new"


class TestConfluence:
    def test_a_lone_family_is_refused(self) -> None:
        """One weak indicator must not open a trade by itself."""
        verdict = assess_candidate(inputs(agreeing_families=1))

        assert not verdict.accepted

    def test_two_independent_families_are_enough(self) -> None:
        assert assess_candidate(inputs(agreeing_families=2)).accepted

    def test_the_refusal_says_why(self) -> None:
        verdict = assess_candidate(inputs(agreeing_families=1))

        assert "confluence" in " ".join(verdict.reasons).lower()

    def test_threshold_is_at_least_two(self) -> None:
        """A confluence requirement of one is not a confluence requirement."""
        assert MIN_INDEPENDENT_FAMILIES >= 2


class TestRegimeExpectancy:
    def test_negative_expectancy_on_a_real_sample_is_refused(self) -> None:
        verdict = assess_candidate(inputs(regime_expectancy=Decimal("-0.4"), regime_samples=60))

        assert not verdict.accepted

    def test_negative_expectancy_on_a_tiny_sample_is_not_disqualifying(self) -> None:
        """Six trades is noise. Do not retire a strategy on it."""
        verdict = assess_candidate(inputs(regime_expectancy=Decimal("-0.4"), regime_samples=6))

        assert verdict.accepted

    def test_no_history_in_this_regime_is_allowed_through(self) -> None:
        """Absence of evidence is not evidence of absence; it must be able to earn one."""
        verdict = assess_candidate(inputs(regime_expectancy=None, regime_samples=0))

        assert verdict.accepted

    def test_shrinkage_pulls_small_samples_toward_neutral(self) -> None:
        small = shrunk_expectancy(Decimal("-1.0"), samples=4)
        large = shrunk_expectancy(Decimal("-1.0"), samples=400)

        assert abs(small) < abs(large)

    def test_shrinkage_preserves_sign(self) -> None:
        assert shrunk_expectancy(Decimal("-1.0"), samples=50) < 0

    def test_large_samples_are_barely_shrunk(self) -> None:
        raw = Decimal("-1.0")
        assert abs(shrunk_expectancy(raw, samples=1000) - raw) < Decimal("0.05")

    def test_sample_floor_is_meaningful(self) -> None:
        assert MIN_SAMPLES_FOR_EXPECTANCY >= 20


class TestCorrelationAndLiquidity:
    def test_a_highly_correlated_duplicate_is_refused(self) -> None:
        """Two names, one position: the position count would overstate diversification."""
        verdict = assess_candidate(inputs(max_correlation=Decimal("0.95")))

        assert not verdict.accepted

    def test_uncorrelated_candidates_pass(self) -> None:
        assert assess_candidate(inputs(max_correlation=Decimal("0.1"))).accepted

    def test_an_illiquid_order_is_refused(self) -> None:
        """Size that moves the bar it trades in is a cost, not an opportunity."""
        verdict = assess_candidate(inputs(volume_share=Decimal("0.5")))

        assert not verdict.accepted

    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        """An operator should see all of it, not fix one and rediscover the next."""
        verdict = assess_candidate(inputs(agreeing_families=1, max_correlation=Decimal("0.99")))

        assert len(verdict.reasons) >= 2


class TestScore:
    def test_score_is_decimal(self) -> None:
        assert isinstance(assess_candidate(inputs()).score, Decimal)

    def test_more_confluence_scores_higher(self) -> None:
        two = assess_candidate(inputs(agreeing_families=2)).score
        three = assess_candidate(inputs(agreeing_families=3)).score

        assert three > two

    def test_proven_regime_expectancy_scores_higher(self) -> None:
        weak = assess_candidate(inputs(regime_expectancy=Decimal("0.05"))).score
        strong = assess_candidate(inputs(regime_expectancy=Decimal("0.8"))).score

        assert strong > weak


class TestConfluenceScalesWithThePool:
    """Corroboration cannot be demanded from sources the pool does not contain."""

    def test_a_single_family_pool_is_not_asked_for_two(self) -> None:
        verdict = assess_candidate(inputs(agreeing_families=1, available_families=1))

        assert verdict.accepted

    def test_a_rich_pool_still_requires_two(self) -> None:
        verdict = assess_candidate(inputs(agreeing_families=1, available_families=5))

        assert not verdict.accepted

    def test_default_assumes_a_rich_pool(self) -> None:
        """Omitting the field must not accidentally relax the requirement."""
        verdict = assess_candidate(inputs(agreeing_families=1))

        assert not verdict.accepted


class TestSoloFamilyConfluenceWaiver:
    """One family may carry a trade, but only by paying for the missing corroboration."""

    def test_a_strong_lone_family_is_accepted(self) -> None:
        verdict = assess_candidate(inputs(agreeing_families=1, net_edge=SOLO_FAMILY_MIN_NET_EDGE))
        assert verdict.accepted
        assert any("confluence waived" in reason for reason in verdict.reasons)

    def test_a_lone_family_just_below_the_bar_is_refused(self) -> None:
        edge = SOLO_FAMILY_MIN_NET_EDGE - Decimal("0.0001")
        verdict = assess_candidate(inputs(agreeing_families=1, net_edge=edge))
        assert not verdict.accepted
        assert any("does not reach" in reason for reason in verdict.reasons)

    def test_an_unmeasured_edge_never_buys_a_waiver(self) -> None:
        # An unknown edge is not a strong one. Waiving on None would turn the missing
        # measurement into a free pass, which is the opposite of what this gate is for.
        verdict = assess_candidate(inputs(agreeing_families=1, net_edge=None))
        assert not verdict.accepted

    def test_a_waived_candidate_scores_below_a_corroborated_one(self) -> None:
        # It competes; it does not jump the queue. Two agreeing families must still win a
        # tie against one, or the waiver would quietly prefer weaker evidence.
        waived = assess_candidate(
            inputs(agreeing_families=1, net_edge=SOLO_FAMILY_MIN_NET_EDGE * 2)
        )
        corroborated = assess_candidate(inputs(agreeing_families=2, net_edge=None))
        assert waived.accepted
        assert corroborated.accepted
        assert waived.score < corroborated.score

    def test_the_waiver_does_not_excuse_any_other_objection(self) -> None:
        # Confluence is the only thing it forgives. A correlation breach still refuses.
        verdict = assess_candidate(
            inputs(
                agreeing_families=1,
                net_edge=SOLO_FAMILY_MIN_NET_EDGE * 4,
                max_correlation=Decimal("0.99"),
            )
        )
        assert not verdict.accepted
        assert any("correlation" in reason for reason in verdict.reasons)
