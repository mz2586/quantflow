"""Tests for the Market Intelligence module."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.intelligence.derivatives import (
    FundingSnapshot,
    OpenInterestSnapshot,
    funding_trend,
    has_perpetual,
    perpetual_symbol,
)
from quantflow.intelligence.measures import (
    MIN_BARS,
    measure_liquidity,
    measure_trend,
    measure_volatility,
    measure_volume,
)
from quantflow.intelligence.regime import Direction, Structure, VolatilityBand, classify
from quantflow.intelligence.snapshot import concentration_score, observe
from quantflow.risk.correlation import CorrelationMatrix, aligned_returns

BTC = Symbol(base="BTC", quote="USDT")
ETH = Symbol(base="ETH", quote="USDT")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def candle(
    index: int,
    close: Decimal,
    *,
    symbol: Symbol = BTC,
    volume: Decimal = Decimal("100"),
    quote_volume: Decimal = Decimal("0"),
    spread: Decimal = Decimal("1"),
) -> Candle:
    """One bar at hour ``index``."""
    return Candle(
        symbol=symbol,
        timeframe=Timeframe.H1,
        open_time=BASE + timedelta(hours=index),
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=volume,
        quote_volume=quote_volume,
        trades=10,
    )


def rising(count: int = 120, *, step: str = "1", symbol: Symbol = BTC) -> list[Candle]:
    """A clean uptrend."""
    return [
        candle(i, Decimal("1000") + Decimal(step) * Decimal(i), symbol=symbol) for i in range(count)
    ]


def choppy(count: int = 120, *, symbol: Symbol = BTC) -> list[Candle]:
    """Oscillation with no net progress."""
    return [
        candle(i, Decimal("1000") + (Decimal("10") if i % 2 else Decimal("-10")), symbol=symbol)
        for i in range(count)
    ]


class TestTrend:
    """Trend strength must separate progress from churn."""

    def test_a_clean_uptrend_is_trending(self) -> None:
        measure = measure_trend(rising())
        assert measure is not None
        assert measure.is_trending
        assert measure.direction > 0

    def test_a_clean_downtrend_is_trending_and_negative(self) -> None:
        measure = measure_trend(rising(step="-1"))
        assert measure is not None
        assert measure.is_trending
        assert measure.direction < 0

    def test_chop_is_not_trending_however_violent(self) -> None:
        # The point of efficiency: a market travelling 10 up and 10 back forever covers
        # enormous distance and gets nowhere. Range alone cannot tell them apart.
        measure = measure_trend(choppy())
        assert measure is not None
        assert not measure.is_trending
        assert measure.efficiency < Decimal("0.1")

    def test_too_little_history_is_unmeasurable(self) -> None:
        assert measure_trend(rising(MIN_BARS - 1)) is None


class TestVolatility:
    """Volatility must be judged against the market's own history."""

    def test_a_calm_market_is_measurable(self) -> None:
        measure = measure_volatility(rising())
        assert measure is not None
        assert measure.normalized_atr >= 0

    def test_an_expanding_market_reads_high(self) -> None:
        calm = [candle(i, Decimal("1000") + Decimal(i % 2)) for i in range(300)]
        violent = [
            candle(
                300 + i,
                Decimal("1000") + Decimal("400") * Decimal(i % 2),
                spread=Decimal("120"),
            )
            for i in range(20)
        ]
        measure = measure_volatility(calm + violent)
        assert measure is not None
        assert measure.relative_level > Decimal("1.5")
        assert measure.is_high

    def test_too_little_history_is_unmeasurable(self) -> None:
        assert measure_volatility(rising(MIN_BARS - 1)) is None


class TestVolume:
    """Volume expansion is relative to the market's own baseline."""

    def test_a_surge_reads_as_expansion(self) -> None:
        quiet = [candle(i, Decimal("1000"), volume=Decimal("100")) for i in range(120)]
        loud = [candle(120 + i, Decimal("1000"), volume=Decimal("500")) for i in range(20)]
        measure = measure_volume(quiet + loud)
        assert measure is not None
        assert measure.is_expanding

    def test_a_lull_reads_as_drying_up(self) -> None:
        loud = [candle(i, Decimal("1000"), volume=Decimal("500")) for i in range(120)]
        quiet = [candle(120 + i, Decimal("1000"), volume=Decimal("50")) for i in range(20)]
        measure = measure_volume(loud + quiet)
        assert measure is not None
        assert measure.is_drying_up


class TestLiquidity:
    """The proxy must work on backfilled candles, which carry no quote volume."""

    def test_observed_quote_volume_is_used_when_present(self) -> None:
        bars = [
            candle(i, Decimal("1000"), volume=Decimal("10"), quote_volume=Decimal("10000"))
            for i in range(60)
        ]
        measure = measure_liquidity(bars)
        assert measure is not None
        assert measure.quote_volume_observed
        assert measure.typical_quote_volume == Decimal("10000")

    def test_quote_volume_is_derived_when_the_venue_omits_it(self) -> None:
        # Every backfilled candle has quote_volume=0 because ccxt's fetch_ohlcv returns
        # six fields and quote volume is not one of them. Reading that zero literally
        # reported every market on the platform as having no liquidity at all.
        bars = [candle(i, Decimal("1000"), volume=Decimal("10")) for i in range(60)]
        measure = measure_liquidity(bars)
        assert measure is not None
        assert not measure.quote_volume_observed
        assert measure.typical_quote_volume > 0

    def test_absorption_is_judged_against_a_typical_bar(self) -> None:
        bars = [
            candle(i, Decimal("1000"), volume=Decimal("10"), quote_volume=Decimal("100000"))
            for i in range(60)
        ]
        measure = measure_liquidity(bars)
        assert measure is not None
        assert measure.can_absorb(Decimal("500"))
        assert not measure.can_absorb(Decimal("50000"))


class TestRegime:
    """Regime is three axes, and each must move independently."""

    def test_an_uptrend_classifies_bull_and_trending(self) -> None:
        profile = classify(rising())
        assert profile is not None
        assert profile.direction is Direction.BULL
        assert profile.structure is Structure.TRENDING

    def test_a_downtrend_classifies_bear(self) -> None:
        profile = classify(rising(step="-1"))
        assert profile is not None
        assert profile.direction is Direction.BEAR

    def test_chop_classifies_sideways_and_ranging(self) -> None:
        profile = classify(choppy())
        assert profile is not None
        assert profile.direction is Direction.SIDEWAYS
        assert profile.structure is Structure.RANGING

    def test_unmeasurable_history_is_none_not_unknown(self) -> None:
        # An UNKNOWN member gets treated as "calm" the first time somebody writes
        # `if regime.volatility is not HIGH`.
        assert classify(rising(MIN_BARS - 1)) is None

    def test_the_profile_explains_itself(self) -> None:
        profile = classify(rising())
        assert profile is not None
        assert "trend strength" in profile.explain()

    def test_the_single_label_projection_is_available(self) -> None:
        profile = classify(rising())
        assert profile is not None
        assert profile.as_market_regime is not None

    def test_axes_are_independent(self) -> None:
        profile = classify(rising())
        assert profile is not None
        assert profile.volatility in set(VolatilityBand)


class TestCorrelationAlignment:
    """Correlating by position rather than timestamp is silently wrong."""

    def test_misaligned_series_are_aligned_before_correlating(self) -> None:
        # The real defect this caught: live ingestion had given BTC and ETH 45 more bars
        # than SOL and BNB. Correlating the last N of each compared different weeks and
        # reported +0.02 between assets that actually move together at +0.84 — which
        # would have left the correlation risk rule permanently silent.
        stamps = [BASE + timedelta(hours=i) for i in range(80)]
        prices = [Decimal("100") + Decimal(i) for i in range(80)]

        full = list(zip(stamps, prices, strict=True))
        # Same series, but shifted: extra bars at the end that the peer does not have.
        extended = full + [
            (BASE + timedelta(hours=80 + i), Decimal("500") + Decimal(i * 37)) for i in range(20)
        ]

        aligned = aligned_returns({BTC: extended, ETH: full})
        matrix = CorrelationMatrix.from_returns(aligned)
        value = matrix.between(BTC, ETH)
        assert value is not None
        assert value > Decimal("0.99")

    def test_no_shared_timestamps_yields_no_returns(self) -> None:
        early = [(BASE + timedelta(hours=i), Decimal(100 + i)) for i in range(50)]
        late = [(BASE + timedelta(days=400, hours=i), Decimal(100 + i)) for i in range(50)]
        aligned = aligned_returns({BTC: early, ETH: late})
        assert aligned[BTC] == ()

    def test_an_empty_input_is_handled(self) -> None:
        assert aligned_returns({}) == {}


class TestDerivatives:
    """Funding and open interest are futures data and must say so."""

    def test_the_perpetual_ticker_is_derived(self) -> None:
        assert perpetual_symbol(BTC) == "BTC/USDT:USDT"

    def test_non_usdt_pairs_have_no_perpetual(self) -> None:
        assert not has_perpetual(Symbol(base="ETH", quote="BTC"))

    def test_funding_annualises_over_three_payments_a_day(self) -> None:
        snapshot = FundingSnapshot(BTC, "BTC/USDT:USDT", Decimal("0.0001"), BASE)
        assert snapshot.annualised == Decimal("0.0001") * Decimal("1095")

    def test_crowding_is_flagged_in_both_directions(self) -> None:
        long_side = FundingSnapshot(BTC, "BTC/USDT:USDT", Decimal("0.0005"), BASE)
        short_side = FundingSnapshot(BTC, "BTC/USDT:USDT", Decimal("-0.0005"), BASE)
        assert long_side.is_crowded_long
        assert short_side.is_crowded_short

    def test_the_payload_names_its_source(self) -> None:
        # A spot dashboard showing "open interest" without this would imply the spot
        # market has open interest. It does not.
        snapshot = OpenInterestSnapshot(BTC, "BTC/USDT:USDT", Decimal("1000"), None, BASE)
        assert "not spot" in snapshot.to_dict()["source"]

    def test_open_interest_change_needs_a_baseline(self) -> None:
        earlier = OpenInterestSnapshot(BTC, "BTC/USDT:USDT", Decimal("1000"), None, BASE)
        later = OpenInterestSnapshot(BTC, "BTC/USDT:USDT", Decimal("1200"), None, BASE)
        assert later.change_from(earlier) == Decimal("0.2")

    def test_funding_trend_of_nothing_is_none(self) -> None:
        assert funding_trend([]) is None


class TestSnapshot:
    """A reading must report what it could not measure."""

    @pytest.mark.asyncio
    async def test_a_full_reading_names_missing_derivatives(self) -> None:
        result = await observe(BTC, rising())
        assert result.trend is not None
        assert result.regime is not None
        assert any("funding" in item for item in result.unavailable)
        assert not result.is_complete

    @pytest.mark.asyncio
    async def test_an_empty_series_is_a_valid_but_empty_reading(self) -> None:
        result = await observe(BTC, [])
        assert result.bars_used == 0
        assert result.unavailable

    @pytest.mark.asyncio
    async def test_peers_produce_correlations(self) -> None:
        result = await observe(BTC, rising(), peers={ETH: rising(symbol=ETH)})
        assert "ETH/USDT" in result.correlations

    @pytest.mark.asyncio
    async def test_the_summary_is_one_readable_line(self) -> None:
        result = await observe(BTC, rising())
        assert str(BTC) in result.summary()

    def test_concentration_of_nothing_is_zero(self) -> None:
        assert concentration_score({}) == Decimal("0")

    def test_concentration_uses_absolute_correlation(self) -> None:
        # An inverted bet on the same driver is still a bet on that driver.
        score = concentration_score({"A": Decimal("-0.9"), "B": Decimal("0.9")})
        assert math.isclose(float(score), 0.9)
