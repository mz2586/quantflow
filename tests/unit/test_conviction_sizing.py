"""Conviction sizing: more capital on stronger evidence, and never on a negative edge.

Every position this account opened was identically sized (~2,485 notional) regardless of
score. This redistributes that capital — but carefully, because two measured facts limit
how far it should go: the score's spread is narrow (half of all selected candidates fall
inside 2.3%), and live expectancy is −2.11 per trade. Sizing amplifies whatever edge the
ranking carries; on a negative one it only loses faster.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.risk.conviction import (
    DEFAULT_MULTIPLIERS,
    Conviction,
    classify,
    percentile_of,
    size_multiplier,
)

POSITIVE = Decimal("0.004")
NEGATIVE = Decimal("-0.001")


class TestTiersFollowTheObservedDistribution:
    def test_the_bottom_quartile_is_weak(self) -> None:
        assert classify(Decimal("0.600")) is Conviction.WEAK

    def test_the_median_candidate_is_normal(self) -> None:
        assert classify(Decimal("0.652")) is Conviction.NORMAL

    def test_above_the_median_is_strong(self) -> None:
        assert classify(Decimal("0.670")) is Conviction.STRONG

    def test_the_top_decile_is_very_strong(self) -> None:
        assert classify(Decimal("0.700")) is Conviction.VERY_STRONG

    def test_percentile_labels_match_the_tiers(self) -> None:
        assert percentile_of(Decimal("0.600")) == "<p25"
        assert percentile_of(Decimal("0.700")) == ">=p90"


class TestSizingIsConservative:
    def test_the_median_candidate_is_sized_exactly_as_before(self) -> None:
        # The change must be a redistribution, not a blanket increase in risk.
        _, multiplier = size_multiplier(Decimal("0.652"), expected_net_edge=POSITIVE)
        assert multiplier == Decimal("1")

    def test_the_gradient_is_shallow_not_doubled(self) -> None:
        # A 2x multiplier on a −2.11 expectancy doubles the loss. The spread stays small
        # until live attribution shows high conviction actually outperforming.
        assert max(DEFAULT_MULTIPLIERS.values()) <= Decimal("1.5")

    def test_a_marginally_above_median_score_stays_near_baseline(self) -> None:
        _, multiplier = size_multiplier(Decimal("0.659"), expected_net_edge=POSITIVE)
        assert Decimal("1") <= multiplier <= Decimal("1.2")

    def test_a_top_decile_candidate_earns_more_capital(self) -> None:
        _, multiplier = size_multiplier(Decimal("0.720"), expected_net_edge=POSITIVE)
        assert multiplier > Decimal("1")


class TestNegativeEdgeNeverEarnsMoreCapital:
    def test_a_strong_score_with_negative_edge_is_capped_at_baseline(self) -> None:
        """Ranking well among poor candidates is not a reason to bet more.

        Conviction says "better than the others", never "profitable". With expectancy at
        −2.11 per trade this is the guard that stops the model amplifying a losing edge.
        """
        tier, multiplier = size_multiplier(Decimal("0.720"), expected_net_edge=NEGATIVE)

        assert tier is Conviction.VERY_STRONG
        assert multiplier == Decimal("1"), "a negative expected edge must not increase size"

    def test_an_unknown_edge_is_treated_as_not_earning_an_increase(self) -> None:
        _, multiplier = size_multiplier(Decimal("0.720"), expected_net_edge=None)
        assert multiplier == Decimal("1")

    def test_a_weak_score_is_still_reduced_on_a_negative_edge(self) -> None:
        # Reductions are always safe, so they are not gated on edge.
        _, multiplier = size_multiplier(Decimal("0.600"), expected_net_edge=NEGATIVE)
        assert multiplier < Decimal("1")

    def test_an_absent_score_uses_baseline(self) -> None:
        tier, multiplier = size_multiplier(None, expected_net_edge=POSITIVE)
        assert tier is Conviction.NORMAL
        assert multiplier == Decimal("1")


class TestGradientIsConfigurable:
    def test_an_override_replaces_the_defaults(self) -> None:
        # Widening the spread from live evidence must not require touching the strategy.
        wider = dict(DEFAULT_MULTIPLIERS) | {Conviction.VERY_STRONG: Decimal("1.75")}

        _, multiplier = size_multiplier(
            Decimal("0.720"), expected_net_edge=POSITIVE, multipliers=wider
        )

        assert multiplier == Decimal("1.75")


class TestCostAssumptionMatchesReality:
    """The cost gate must reject on what trading actually costs, not a stale guess.

    The orchestrator assumed a 0.2000% round trip. Measured over 18 live trades on this
    account: **0.0920%** — 2.29 in fees on 2,487 of notional. The gate was therefore
    subtracting more than twice the real cost from every candidate's edge before judging
    it, and rejecting trades that were genuinely profitable.

    Three rejections in one hour, all within a whisker of the floor::

        expected edge 0.3929% after 0.2000% costs is below the 0.4000% floor
        expected edge 0.3814% after 0.2000% costs is below the 0.4000% floor
        expected edge 0.3774% after 0.2000% costs is below the 0.4000% floor

    Add back the 0.108% the gate over-charged and every one of them clears 0.40%. This is
    not a lowered bar — the floor is untouched. It is the same bar applied to a true number.
    """

    def test_the_assumed_cost_is_not_wildly_above_what_is_paid(self) -> None:
        from quantflow.orchestrator.strategy import DEFAULT_COST_RATE

        measured_round_trip = Decimal("0.00092")

        assert (
            measured_round_trip * Decimal("1.5") >= DEFAULT_COST_RATE
        ), "the gate must not reject candidates using a cost far above the real one"

    def test_the_assumed_cost_is_not_optimistic(self) -> None:
        # Erring low would admit trades that cannot pay for themselves. The figure should
        # sit at or above what is actually charged.
        from quantflow.orchestrator.strategy import DEFAULT_COST_RATE

        assert Decimal("0.00092") <= DEFAULT_COST_RATE


class TestConvictionAndTheCapAreOneModel:
    """Conviction must never produce a size the next layer rejects.

    The sizer attenuates by conviction and *then* clamps to ``max_position_pct``. A
    multiplier applied after it is applied after the clamp, so anything above 1.0 breaches
    the cap by construction. Live consequence: three candidates selected, zero orders
    placed, each refused with "would reach 22.99% of equity, above the 20.00% limit" — two
    components disagreeing, not a limit working.

    As a fraction of the allowed maximum the contradiction is impossible.
    """

    def test_the_strongest_tier_asks_for_the_full_cap_and_no_more(self) -> None:
        from quantflow.risk.conviction import allocation_fraction

        _, fraction = allocation_fraction(Decimal("0.720"), expected_net_edge=POSITIVE)

        assert fraction == Decimal("1")

    def test_every_tier_stays_within_the_cap(self) -> None:
        from quantflow.risk.conviction import allocation_fraction

        for score in ("0.600", "0.652", "0.670", "0.720", "0.736"):
            _, fraction = allocation_fraction(Decimal(score), expected_net_edge=POSITIVE)
            assert Decimal("0") < fraction <= Decimal("1"), f"{score} escaped the cap"

    def test_a_weaker_tier_takes_proportionally_less(self) -> None:
        from quantflow.risk.conviction import allocation_fraction

        _, weak = allocation_fraction(Decimal("0.600"), expected_net_edge=POSITIVE)
        _, normal = allocation_fraction(Decimal("0.652"), expected_net_edge=POSITIVE)
        _, strong = allocation_fraction(Decimal("0.720"), expected_net_edge=POSITIVE)

        assert weak < normal < strong

    def test_a_negative_edge_still_cannot_earn_the_top_allocation(self) -> None:
        from quantflow.risk.conviction import allocation_fraction

        _, capped = allocation_fraction(Decimal("0.720"), expected_net_edge=NEGATIVE)
        _, earned = allocation_fraction(Decimal("0.720"), expected_net_edge=POSITIVE)

        assert capped < earned
