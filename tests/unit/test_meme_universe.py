"""Tests for the meme-coin universe and its eligibility gate.

The eligibility check has one property that matters more than any individual threshold:
each rejection rule must fire **on its own**, and all of them must be reported together.
So every test below starts from one known-good set of measurements and perturbs exactly one
of them. A rule that only fires when two things are wrong at once is a rule that will stay
silent on the day it is needed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.universe.meme import (
    DEFAULT_ELIGIBILITY_LIMITS,
    MAJOR_REFERENCE_VOLATILITY,
    MEME_BASE_ASSETS,
    MEME_SIZE_CEILING,
    EligibilityInputs,
    EligibilityLimits,
    MemeMarket,
    assess_eligibility,
    discover_meme_universe,
    is_meme,
    meme_size_factor,
    size_for_meme,
    strip_multiplier,
)

#: The meme markets Bybit actually lists as active USDT swaps, verified against the venue.
BYBIT_MEME_BASES = (
    "ACT",
    "BOME",
    "BRETT",
    "DOGE",
    "FARTCOIN",
    "GOAT",
    "MEME",
    "MEW",
    "MOODENG",
    "ORDI",
    "PENGU",
    "PNUT",
    "POPCAT",
    "SPX",
    "TRUMP",
    "WIF",
    "1000BONK",
    "1000BTT",
    "1000CAT",
    "1000FLOKI",
    "1000PEPE",
    "1000RATS",
    "1000TURBO",
    "10000SATS",
    "1000000BABYDOGE",
    "1000000MOG",
)


def instrument(
    base: str,
    quote: str = "USDT",
    *,
    active: bool = True,
    market_type: MarketType = MarketType.FUTURE,
    price_tick: str = "0.00001",
    quantity_step: str = "1",
    min_quantity: str = "1",
    min_notional: str = "5",
) -> Instrument:
    return Instrument(
        symbol=Symbol(base=base, quote=quote),
        market_type=market_type,
        price_tick=Decimal(price_tick),
        quantity_step=Decimal(quantity_step),
        min_quantity=Decimal(min_quantity),
        min_notional=Decimal(min_notional),
        active=active,
    )


def meme_market(
    base: str = "DOGE",
    *,
    active: bool = True,
    price_tick: str = "0.00001",
    min_quantity: str = "1",
    min_notional: str = "5",
) -> MemeMarket:
    root, multiplier = strip_multiplier(base)
    return MemeMarket(
        symbol=Symbol(base=base, quote="USDT"),
        base_root=root,
        multiplier=multiplier,
        price_tick=Decimal(price_tick),
        quantity_step=Decimal("1"),
        min_quantity=Decimal(min_quantity),
        min_notional=Decimal(min_notional),
        active=active,
        market_type=MarketType.FUTURE,
    )


def passing_inputs(market: MemeMarket | None = None) -> EligibilityInputs:
    """A DOGE/USDT snapshot that clears every rule with room to spare.

    Deliberately not marginal: every value sits clearly inside its limit, so a test that
    perturbs one field is unambiguously testing that field's rule.
    """
    return EligibilityInputs(
        market=market if market is not None else meme_market(),
        quote_volume_24h=Decimal("50000000"),
        bid=Decimal("0.40000"),
        ask=Decimal("0.40002"),
        ticker_age=timedelta(seconds=5),
        candle_age=timedelta(minutes=5),
        volatility=Decimal("0.02"),
        last_bar_range=Decimal("0.008"),
        typical_bar_range=Decimal("0.008"),
        last_bar_return=Decimal("0.01"),
        bar_quote_volume=Decimal("500000"),
        intended_quantity=Decimal("5000"),
        intended_price=Decimal("0.40"),
        stop_distance=Decimal("0.004"),
    )


def only_reason(inputs: EligibilityInputs) -> str:
    """Assert exactly one rule fired, and return it."""
    verdict = assess_eligibility(inputs)
    assert not verdict.eligible
    assert len(verdict.reasons) == 1, f"expected one reason, got {verdict.reasons}"
    return verdict.reasons[0]


class TestStripMultiplier:
    @pytest.mark.parametrize(
        ("base", "root", "multiplier"),
        [
            ("DOGE", "DOGE", "1"),
            ("WIF", "WIF", "1"),
            ("FARTCOIN", "FARTCOIN", "1"),
            ("1000PEPE", "PEPE", "1000"),
            ("1000BONK", "BONK", "1000"),
            ("1000BTT", "BTT", "1000"),
            ("10000SATS", "SATS", "10000"),
            ("1000000BABYDOGE", "BABYDOGE", "1000000"),
            ("1000000MOG", "MOG", "1000000"),
        ],
    )
    def test_splits_every_venue_prefix_form(self, base: str, root: str, multiplier: str) -> None:
        assert strip_multiplier(base) == (root, Decimal(multiplier))

    def test_longest_prefix_wins(self) -> None:
        # "1000000BABYDOGE" also starts with "1000"; a shortest-first scan would leave
        # "000BABYDOGE" and the coin would silently vanish from the universe.
        root, multiplier = strip_multiplier("1000000BABYDOGE")
        assert root == "BABYDOGE"
        assert multiplier == Decimal("1000000")

    def test_multiplier_is_decimal_never_float(self) -> None:
        _, multiplier = strip_multiplier("1000PEPE")
        assert isinstance(multiplier, Decimal)
        assert multiplier == Decimal("1000")

    def test_unprefixed_base_gets_unit_multiplier(self) -> None:
        root, multiplier = strip_multiplier("doge")
        assert root == "DOGE"
        assert multiplier == Decimal("1")

    def test_bare_digits_are_not_shredded(self) -> None:
        # Nothing legitimate remains behind the prefix, so leave the ticker intact rather
        # than reduce it to an empty root that matches nothing.
        assert strip_multiplier("1000") == ("1000", Decimal("1"))

    def test_prefix_is_only_stripped_before_a_letter(self) -> None:
        assert strip_multiplier("10009") == ("10009", Decimal("1"))


class TestIsMeme:
    @pytest.mark.parametrize("base", BYBIT_MEME_BASES)
    def test_every_listed_bybit_meme_is_recognised(self, base: str) -> None:
        assert is_meme(Symbol(base=base, quote="USDT"))

    @pytest.mark.parametrize("base", ["BTC", "ETH", "SOL", "LINK", "ARB", "1000XYZ"])
    def test_non_memes_are_rejected(self, base: str) -> None:
        assert not is_meme(Symbol(base=base, quote="USDT"))

    def test_curated_list_stores_roots_not_prefixed_tickers(self) -> None:
        assert "PEPE" in MEME_BASE_ASSETS
        assert "1000PEPE" not in MEME_BASE_ASSETS

    def test_classification_is_quote_agnostic(self) -> None:
        # is_meme judges the asset; the USDT restriction is a universe decision, not a
        # classification one.
        assert is_meme(Symbol(base="DOGE", quote="USDC"))


class TestDiscoverMemeUniverse:
    def test_keeps_active_usdt_memes_including_prefixed_ones(self) -> None:
        markets = discover_meme_universe([instrument("DOGE"), instrument("1000PEPE")])
        assert [str(market.symbol) for market in markets] == ["1000PEPE/USDT", "DOGE/USDT"]

    def test_excludes_non_meme_assets(self) -> None:
        markets = discover_meme_universe([instrument("BTC"), instrument("ETH")])
        assert markets == []

    def test_excludes_non_usdt_quotes(self) -> None:
        markets = discover_meme_universe([instrument("DOGE", "USDC"), instrument("PEPE", "BTC")])
        assert markets == []

    def test_excludes_inactive_instruments(self) -> None:
        markets = discover_meme_universe([instrument("DOGE", active=False)])
        assert markets == []

    def test_mixed_venue_snapshot(self) -> None:
        markets = discover_meme_universe(
            [
                instrument("BTC"),
                instrument("1000000MOG"),
                instrument("SHIB", "USDC"),
                instrument("WIF", active=False),
                instrument("DOGE"),
            ]
        )
        assert [market.base_root for market in markets] == ["MOG", "DOGE"]

    def test_result_is_sorted_deterministically(self) -> None:
        forward = discover_meme_universe(
            [instrument("WIF"), instrument("DOGE"), instrument("BOME")]
        )
        reverse = discover_meme_universe(
            [instrument("BOME"), instrument("WIF"), instrument("DOGE")]
        )
        assert [m.symbol for m in forward] == [m.symbol for m in reverse]
        assert [m.base_root for m in forward] == ["BOME", "DOGE", "WIF"]

    def test_carries_venue_rules_through_as_decimals(self) -> None:
        source = instrument(
            "1000PEPE",
            price_tick="0.0000001",
            quantity_step="10",
            min_quantity="10",
            min_notional="5",
        )
        (market,) = discover_meme_universe([source])
        assert market.base_root == "PEPE"
        assert market.multiplier == Decimal("1000")
        assert market.price_tick == Decimal("0.0000001")
        assert market.quantity_step == Decimal("10")
        assert market.min_quantity == Decimal("10")
        assert market.min_notional == Decimal("5")
        assert market.market_type is MarketType.FUTURE
        assert market.active
        for value in (
            market.multiplier,
            market.price_tick,
            market.quantity_step,
            market.min_quantity,
            market.min_notional,
        ):
            assert isinstance(value, Decimal)


class TestEligibilityAccepts:
    def test_a_healthy_market_passes_with_no_reasons(self) -> None:
        verdict = assess_eligibility(passing_inputs())
        assert verdict.eligible
        assert verdict.reasons == ()

    def test_boundary_values_are_inclusive_not_rejections(self) -> None:
        limits = DEFAULT_ELIGIBILITY_LIMITS
        inputs = replace(
            passing_inputs(),
            quote_volume_24h=limits.min_quote_volume_24h,
            ticker_age=limits.max_ticker_age,
            candle_age=limits.max_candle_age,
            volatility=limits.min_volatility,
        )
        assert assess_eligibility(inputs).eligible


class TestEligibilityRejections:
    def test_thin_24h_volume(self) -> None:
        reason = only_reason(replace(passing_inputs(), quote_volume_24h=Decimal("100000")))
        assert "24h quote volume" in reason
        assert "liquidity floor" in reason

    def test_wide_spread(self) -> None:
        # ~20bps: twice the limit, but still far narrower than the stop, so only the
        # spread rule can fire.
        reason = only_reason(replace(passing_inputs(), ask=Decimal("0.40080")))
        assert "spread" in reason
        assert "bps" in reason

    def test_crossed_or_absent_quote(self) -> None:
        verdict = assess_eligibility(
            replace(passing_inputs(), bid=Decimal("0.41"), ask=Decimal("0.40"))
        )
        assert not verdict.eligible
        assert any("two-sided quote" in reason for reason in verdict.reasons)

    def test_stale_ticker(self) -> None:
        reason = only_reason(replace(passing_inputs(), ticker_age=timedelta(minutes=5)))
        assert "quote is" in reason
        assert "staleness limit" in reason

    def test_stale_candle(self) -> None:
        reason = only_reason(replace(passing_inputs(), candle_age=timedelta(hours=3)))
        assert "last bar is" in reason
        assert "staleness limit" in reason

    def test_volatility_too_low(self) -> None:
        reason = only_reason(replace(passing_inputs(), volatility=Decimal("0.0005")))
        assert "below" in reason
        assert "round-trip cost" in reason

    def test_volatility_too_high(self) -> None:
        reason = only_reason(replace(passing_inputs(), volatility=Decimal("0.25")))
        assert "above" in reason
        assert "risk budget" in reason

    def test_flash_move_by_range_multiple(self) -> None:
        # 6.25x the typical range: a cascade, not a trend.
        reason = only_reason(replace(passing_inputs(), last_bar_range=Decimal("0.05")))
        assert "flash move" in reason
        assert "typical range" in reason

    def test_flash_move_by_absolute_return(self) -> None:
        # The relative test cannot fire here (range is normal); the absolute one must.
        reason = only_reason(replace(passing_inputs(), last_bar_return=Decimal("0.35")))
        assert "flash move" in reason
        assert "single-bar breaker" in reason

    def test_flash_return_breaker_is_symmetric(self) -> None:
        reason = only_reason(replace(passing_inputs(), last_bar_return=Decimal("-0.35")))
        assert "flash move" in reason

    def test_zero_typical_range_does_not_disable_the_breaker(self) -> None:
        inputs = replace(
            passing_inputs(),
            typical_bar_range=Decimal("0"),
            last_bar_range=Decimal("0.5"),
            last_bar_return=Decimal("0.4"),
        )
        verdict = assess_eligibility(inputs)
        assert not verdict.eligible
        assert any("single-bar breaker" in reason for reason in verdict.reasons)

    def test_notional_below_venue_minimum(self) -> None:
        # 2 contracts * 0.40 = 0.80 USDT, under the venue's 5 USDT floor, while the
        # quantity itself still clears min_quantity=1.
        reason = only_reason(replace(passing_inputs(), intended_quantity=Decimal("2")))
        assert "notional" in reason
        assert "venue minimum" in reason

    def test_quantity_below_venue_minimum(self) -> None:
        market = meme_market(min_quantity="100", min_notional="5")
        inputs = replace(
            passing_inputs(market), intended_quantity=Decimal("50")
        )  # 50 * 0.40 = 20 USDT, so only the quantity rule can fire
        reason = only_reason(inputs)
        assert "quantity" in reason
        assert "venue minimum" in reason

    def test_order_exceeds_share_of_bar_volume(self) -> None:
        reason = only_reason(replace(passing_inputs(), bar_quote_volume=Decimal("10000")))
        assert "last bar's volume" in reason
        assert "liquidity ceiling" in reason

    def test_stop_tighter_than_the_tick_floor(self) -> None:
        reason = only_reason(replace(passing_inputs(), stop_distance=Decimal("0.00005")))
        assert "stop distance" in reason

    def test_stop_inside_the_spread_is_not_a_stop(self) -> None:
        # Ten ticks is only 0.0001 here, but the live spread is 0.0006 — the wider of the
        # two governs, so a 0.0002 stop must still be refused.
        inputs = replace(passing_inputs(), ask=Decimal("0.40060"), stop_distance=Decimal("0.0002"))
        verdict = assess_eligibility(inputs)
        assert not verdict.eligible
        assert any("stop distance" in reason for reason in verdict.reasons)

    def test_inactive_market(self) -> None:
        reason = only_reason(passing_inputs(meme_market(active=False)))
        assert "not active" in reason

    def test_limits_are_configurable_without_editing_the_module(self) -> None:
        strict = EligibilityLimits(min_quote_volume_24h=Decimal("100000000"))
        verdict = assess_eligibility(passing_inputs(), strict)
        assert not verdict.eligible
        assert any("liquidity floor" in reason for reason in verdict.reasons)


class TestAllReasonsAreReported:
    def test_every_failing_rule_appears_not_just_the_first(self) -> None:
        inputs = replace(
            passing_inputs(),
            quote_volume_24h=Decimal("1000"),
            ticker_age=timedelta(minutes=10),
            candle_age=timedelta(hours=4),
            volatility=Decimal("0.0001"),
            stop_distance=Decimal("0"),
        )
        verdict = assess_eligibility(inputs)
        assert not verdict.eligible
        assert len(verdict.reasons) == 5
        joined = " | ".join(verdict.reasons)
        for fragment in (
            "24h quote volume",
            "quote is",
            "last bar is",
            "round-trip cost",
            "stop distance",
        ):
            assert fragment in joined

    def test_reasons_are_an_immutable_tuple(self) -> None:
        verdict = assess_eligibility(replace(passing_inputs(), stop_distance=Decimal("0")))
        assert isinstance(verdict.reasons, tuple)


class TestSizing:
    def test_always_strictly_smaller_than_the_major_equivalent(self) -> None:
        baseline = Decimal("1000")
        for volatility in ("0", "0.005", "0.01", "0.05", "0.2"):
            sized = size_for_meme(baseline, Decimal(volatility))
            assert sized < baseline
            assert sized > Decimal("0")

    def test_capped_by_the_ceiling_even_at_zero_volatility(self) -> None:
        assert size_for_meme(Decimal("1000"), Decimal("0")) == MEME_SIZE_CEILING * Decimal("1000")

    def test_shrinks_monotonically_as_volatility_rises(self) -> None:
        baseline = Decimal("1000")
        sizes = [
            size_for_meme(baseline, Decimal(volatility))
            for volatility in ("0.005", "0.01", "0.02", "0.04", "0.08")
        ]
        assert sizes == sorted(sizes, reverse=True)
        assert all(a > b for a, b in pairwise(sizes))

    def test_factor_halves_the_ceiling_at_the_reference_volatility(self) -> None:
        factor = meme_size_factor(MAJOR_REFERENCE_VOLATILITY)
        assert factor == MEME_SIZE_CEILING / Decimal("2")

    def test_factor_never_reaches_or_exceeds_one(self) -> None:
        for volatility in ("0", "0.0001", "0.5", "5"):
            assert meme_size_factor(Decimal(volatility)) < Decimal("1")

    def test_only_ever_reduces_regardless_of_reference(self) -> None:
        # Even given an absurdly generous reference the ceiling still binds.
        sized = size_for_meme(Decimal("1000"), Decimal("0.001"), reference_volatility=Decimal("10"))
        assert sized < Decimal("1000")
        assert sized <= MEME_SIZE_CEILING * Decimal("1000")

    def test_short_notional_keeps_its_sign(self) -> None:
        sized = size_for_meme(Decimal("-1000"), Decimal("0.01"))
        assert sized < Decimal("0")
        assert abs(sized) < Decimal("1000")

    def test_results_are_decimal_never_float(self) -> None:
        sized = size_for_meme(Decimal("1000"), Decimal("0.037"))
        factor = meme_size_factor(Decimal("0.037"))
        assert isinstance(sized, Decimal)
        assert isinstance(factor, Decimal)

    def test_rejects_nonsensical_arguments(self) -> None:
        with pytest.raises(ValidationError):
            meme_size_factor(Decimal("-0.01"))
        with pytest.raises(ValidationError):
            meme_size_factor(Decimal("0.01"), reference_volatility=Decimal("0"))
        with pytest.raises(ValidationError):
            meme_size_factor(Decimal("0.01"), ceiling=Decimal("1"))
