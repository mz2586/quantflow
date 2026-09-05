"""Tests for the multi-asset universe: classification, eligibility, costs and gating.

The eligibility check has the same property that matters in
``tests/unit/test_meme_universe.py``: each rejection rule must fire **on its own**, and all
of them must be reported together. So every eligibility test starts from one known-good set
of measurements and perturbs exactly one of them. A rule that only fires when two things
are wrong at once is a rule that will stay silent on the day it is needed.

Classification has a second property worth as much: it must key off what the *venue* said,
not off how the ticker reads. ``XAU`` is gold because Bybit tagged it ``commodity``; a
classifier that pattern-matched the string would be inventing a fact, and would misread the
next listing whose name happens to look similar.

The venue metadata used throughout was read from the Bybit demo host on 2026-08-14.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.universe.assets import (
    ALLOWED_FAMILIES_BY_CLASS,
    ASSET_CLASS_METADATA_KEY,
    LIMITS_BY_CLASS,
    NON_CRYPTO_MIN_QUOTE_VOLUME_24H,
    AssetClass,
    AssetEligibilityInputs,
    AssetMarket,
    CostInputs,
    all_in_cost,
    assess_eligibility,
    asset_class_from_metadata,
    build_asset_market,
    classify_asset_class,
    clears_costs,
    discover_asset_universe,
    family_supports_class,
    limits_for,
    strategy_supports_class,
)

# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------

#: Every non-crypto market Bybit actually lists, with the tag it reports for each.
VENUE_TAGGED: tuple[tuple[str, str, AssetClass], ...] = (
    # commodity -> metals
    ("XAU", "commodity", AssetClass.METAL),
    ("XAG", "commodity", AssetClass.METAL),
    # commodity -> energy
    ("CL", "commodity", AssetClass.ENERGY),
    ("BZ", "commodity", AssetClass.ENERGY),
    # stock -> single names
    ("AAPL", "stock", AssetClass.EQUITY),
    ("MSFT", "stock", AssetClass.EQUITY),
    ("NVDA", "stock", AssetClass.EQUITY),
    ("TSLA", "stock", AssetClass.EQUITY),
    ("GOOGL", "stock", AssetClass.EQUITY),
    ("AMZN", "stock", AssetClass.EQUITY),
    ("META", "stock", AssetClass.EQUITY),
    ("ASML", "stock", AssetClass.EQUITY),
    ("TSM", "stock", AssetClass.EQUITY),
    ("SNDK", "stock", AssetClass.EQUITY),
    # stock -> index ETFs
    ("SPY", "stock", AssetClass.INDEX),
    ("QQQ", "stock", AssetClass.INDEX),
    ("TQQQ", "stock", AssetClass.INDEX),
    ("SQQQ", "stock", AssetClass.INDEX),
    ("SOXL", "stock", AssetClass.INDEX),
    ("SOXS", "stock", AssetClass.INDEX),
    ("KORU", "stock", AssetClass.INDEX),
    ("EWY", "stock", AssetClass.INDEX),
    # crypto
    ("BTC", "", AssetClass.CRYPTO),
    ("ETH", "", AssetClass.CRYPTO),
    ("SOL", "innovation", AssetClass.CRYPTO),
    ("LINK", "", AssetClass.CRYPTO),
    # memes, deferred to the meme module's curated roots
    ("DOGE", "", AssetClass.MEME),
    ("FARTCOIN", "", AssetClass.MEME),
    ("1000PEPE", "", AssetClass.MEME),
)


@pytest.mark.parametrize(
    ("base", "tag", "expected"), VENUE_TAGGED, ids=[r[0] for r in VENUE_TAGGED]
)
def test_classification_follows_the_venue_tag(base: str, tag: str, expected: AssetClass) -> None:
    """Every market Bybit lists lands in the class its ``symbolType`` implies."""
    assert classify_asset_class(Symbol(base, "USDT"), tag) == expected


def test_the_same_ticker_classifies_differently_under_a_different_venue_tag() -> None:
    """The tag is the key, not the name.

    This is the property that makes the classifier discovered rather than hardcoded. If
    Bybit tagged ``XAU`` as crypto tomorrow, this module would follow it — and if a token
    called ``CL`` were listed untagged, it would be crypto rather than crude.
    """
    assert classify_asset_class(Symbol("XAU", "USDT"), "commodity") is AssetClass.METAL
    assert classify_asset_class(Symbol("XAU", "USDT"), "") is AssetClass.CRYPTO
    assert classify_asset_class(Symbol("CL", "USDT"), "") is AssetClass.CRYPTO
    assert classify_asset_class(Symbol("AAPL", "USDT"), "") is AssetClass.CRYPTO


def test_an_unknown_commodity_root_falls_back_on_the_iso_metal_prefix() -> None:
    """ISO 4217 reserves ``X`` for precious metals, so an unlisted one still classifies.

    Platinum and palladium are not listed today. If they are listed tomorrow they must not
    arrive as energy, which is what an else-branch without this rule would do.
    """
    assert classify_asset_class(Symbol("XPT", "USDT"), "commodity") is AssetClass.METAL
    assert classify_asset_class(Symbol("XPD", "USDT"), "commodity") is AssetClass.METAL
    # A non-X commodity nobody has classified is energy, the larger of the two groups.
    assert classify_asset_class(Symbol("ZZZ", "USDT"), "commodity") is AssetClass.ENERGY


def test_an_unrecognised_venue_tag_is_treated_as_crypto() -> None:
    """A tag nobody has seen must not crash and must not invent a class.

    Crypto is the conservative landing place: its band admits the widest range of
    behaviour, so a market that does not belong there fails the measured checks rather than
    trading under thresholds chosen for something else.
    """
    assert classify_asset_class(Symbol("WAT", "USDT"), "something-new") is AssetClass.CRYPTO
    assert classify_asset_class(Symbol("WAT", "USDT"), "") is AssetClass.CRYPTO


def test_the_tag_is_read_case_and_whitespace_insensitively() -> None:
    """Venue payloads are strings; a stray space must not silently reclassify a market."""
    assert classify_asset_class(Symbol("XAU", "USDT"), " Commodity ") is AssetClass.METAL
    assert classify_asset_class(Symbol("SPY", "USDT"), "STOCK") is AssetClass.INDEX


def test_a_multiplier_prefixed_listing_classifies_on_its_root() -> None:
    """A ``1000``-basket listing must classify as the thing it is a basket of."""
    assert classify_asset_class(Symbol("1000PEPE", "USDT"), "") is AssetClass.MEME


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def instrument(
    base: str,
    *,
    tag: str = "",
    quote: str = "USDT",
    active: bool = True,
    tick: str = "0.01",
    step: str = "0.001",
    min_qty: str = "0.001",
    min_notional: str = "5",
    max_leverage: str = "100",
    funding: int | None = 480,
) -> Instrument:
    """A venue instrument with Bybit-shaped linear-perpetual rules."""
    return Instrument(
        symbol=Symbol(base, quote),
        market_type=MarketType.FUTURE,
        price_tick=Decimal(tick),
        quantity_step=Decimal(step),
        min_quantity=Decimal(min_qty),
        max_quantity=Decimal("500"),
        min_notional=Decimal(min_notional),
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0006"),
        max_leverage=Decimal(max_leverage),
        active=active,
        venue_symbol_type=tag,
        funding_interval_minutes=funding,
    )


#: A venue snapshot spanning every class, as the gateway would hand it over.
VENUE_SNAPSHOT = [
    instrument("XAU", tag="commodity", funding=240),
    instrument("XAG", tag="commodity", funding=240),
    instrument("CL", tag="commodity"),
    instrument("BZ", tag="commodity"),
    instrument("AAPL", tag="stock", max_leverage="50"),
    instrument("NVDA", tag="stock", max_leverage="50"),
    instrument("SPY", tag="stock", max_leverage="25"),
    instrument("SOXL", tag="stock", max_leverage="50"),
    instrument("BTC", tag=""),
    instrument("DOGE", tag=""),
]


def test_discovery_partitions_the_venue_into_classes() -> None:
    """Each class is discovered from the venue snapshot, with nothing lost or duplicated."""
    found: dict[AssetClass, set[str]] = {}
    for market in discover_asset_universe(VENUE_SNAPSHOT):
        found.setdefault(market.asset_class, set()).add(market.symbol.base)

    assert found[AssetClass.METAL] == {"XAU", "XAG"}
    assert found[AssetClass.ENERGY] == {"CL", "BZ"}
    assert found[AssetClass.EQUITY] == {"AAPL", "NVDA"}
    assert found[AssetClass.INDEX] == {"SPY", "SOXL"}
    assert found[AssetClass.CRYPTO] == {"BTC"}
    assert found[AssetClass.MEME] == {"DOGE"}
    assert sum(len(bases) for bases in found.values()) == len(VENUE_SNAPSHOT)


@pytest.mark.parametrize(
    ("asset_class", "expected"),
    [
        (AssetClass.METAL, {"XAU", "XAG"}),
        (AssetClass.ENERGY, {"CL", "BZ"}),
        (AssetClass.EQUITY, {"AAPL", "NVDA"}),
        (AssetClass.INDEX, {"SPY", "SOXL"}),
    ],
)
def test_discovery_can_be_restricted_to_one_class(
    asset_class: AssetClass, expected: set[str]
) -> None:
    """Asking for one class returns exactly that class."""
    markets = discover_asset_universe(VENUE_SNAPSHOT, classes=[asset_class])
    assert {market.symbol.base for market in markets} == expected
    assert all(market.asset_class is asset_class for market in markets)


def test_discovery_drops_inactive_and_wrongly_quoted_markets() -> None:
    """A delisted market still returns candles, so it must be excluded at discovery.

    A suspended instrument keeps serving a stale ticker and old bars, so a caller that
    only checks for the presence of data would size a position the venue then rejects.
    """
    snapshot = [
        instrument("XAU", tag="commodity"),
        instrument("XAG", tag="commodity", active=False),
        instrument("CL", tag="commodity", quote="USDC"),
    ]
    bases = {market.symbol.base for market in discover_asset_universe(snapshot)}
    assert bases == {"XAU"}


def test_discovery_is_deterministically_ordered() -> None:
    """Two runs over the same snapshot must produce the same order, or logs cannot diff."""
    forward = discover_asset_universe(VENUE_SNAPSHOT)
    backward = discover_asset_universe(list(reversed(VENUE_SNAPSHOT)))
    assert [m.symbol for m in forward] == [m.symbol for m in backward]
    assert forward == sorted(forward, key=lambda m: m.symbol)


def test_the_projection_carries_the_venue_rules_verbatim() -> None:
    """Fees, leverage and the funding interval must survive into the market projection."""
    market = build_asset_market(instrument("XAU", tag="commodity", funding=240))
    assert market.asset_class is AssetClass.METAL
    assert market.taker_fee == Decimal("0.0006")
    assert market.maker_fee == Decimal("0.0001")
    assert market.max_leverage == Decimal("100")
    assert market.funding_interval_minutes == 240
    assert market.min_notional == Decimal("5")


# --------------------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------------------

GOLD = build_asset_market(instrument("XAU", tag="commodity", funding=240))
CRUDE = build_asset_market(instrument("CL", tag="commodity", step="0.01", min_qty="0.01"))
APPLE = build_asset_market(instrument("AAPL", tag="stock", step="0.01", min_qty="0.01"))
INDEX = build_asset_market(instrument("SOXL", tag="stock", step="0.01", min_qty="0.01"))


def healthy(market: AssetMarket = GOLD) -> AssetEligibilityInputs:
    """A measurement set that passes every check, for one perturbation at a time.

    The numbers are gold's real ones: a 0.18% typical bar at ~3,400 with a 0.02bp spread
    and 177M of 24h turnover. A 500 USDT probe order against a bar carrying ~1.8M of
    volume is well inside the 2% ceiling.
    """
    return AssetEligibilityInputs(
        market=market,
        quote_volume_24h=Decimal("177000000"),
        bid=Decimal("3400.00"),
        ask=Decimal("3400.02"),
        ticker_age=timedelta(seconds=1),
        candle_age=timedelta(minutes=8),
        volatility=Decimal("0.0018"),
        last_bar_range=Decimal("6.12"),
        typical_bar_range=Decimal("6.12"),
        last_bar_return=Decimal("0.0009"),
        bar_quote_volume=Decimal("1800000"),
        intended_quantity=Decimal("0.147"),
        intended_price=Decimal("3400.02"),
        stop_distance=Decimal("6.12"),
    )


@pytest.mark.parametrize("market", [GOLD, CRUDE, APPLE, INDEX], ids=lambda m: str(m.symbol))
def test_a_healthy_market_is_eligible_on_every_class(market: AssetMarket) -> None:
    """The baseline must pass, or every perturbation below proves nothing."""
    verdict = assess_eligibility(replace(healthy(), market=market))
    assert verdict.eligible, verdict.reasons
    assert verdict.reasons == ()


def test_an_inactive_market_is_refused() -> None:
    dead = replace(GOLD, active=False)
    verdict = assess_eligibility(replace(healthy(), market=dead))
    assert not verdict.eligible
    assert any("not active" in reason for reason in verdict.reasons)


def test_thin_liquidity_is_refused() -> None:
    """The floor exists for the exit. Most of the venue's 193 equities fail here."""
    verdict = assess_eligibility(
        replace(healthy(), quote_volume_24h=NON_CRYPTO_MIN_QUOTE_VOLUME_24H - Decimal("1"))
    )
    assert not verdict.eligible
    assert any("liquidity floor" in reason for reason in verdict.reasons)


def test_a_wide_spread_is_refused() -> None:
    """A 20bp spread crossed twice eats more than the move the strategy is chasing."""
    verdict = assess_eligibility(replace(healthy(), bid=Decimal("3397"), ask=Decimal("3403")))
    assert not verdict.eligible
    assert any("spread" in reason for reason in verdict.reasons)


def test_a_one_sided_or_crossed_quote_is_refused() -> None:
    """No mid means no basis for any of the price arithmetic below it."""
    assert not assess_eligibility(replace(healthy(), bid=Decimal("0"))).eligible
    crossed = assess_eligibility(replace(healthy(), bid=Decimal("3401"), ask=Decimal("3399")))
    assert not crossed.eligible
    assert any("two-sided" in reason for reason in crossed.reasons)


def test_a_stale_quote_is_refused() -> None:
    verdict = assess_eligibility(replace(healthy(), ticker_age=timedelta(minutes=5)))
    assert not verdict.eligible
    assert any("staleness limit" in reason for reason in verdict.reasons)


def test_a_stale_bar_is_refused() -> None:
    """The check that catches an equity perpetual quoting through a closed cash session."""
    verdict = assess_eligibility(
        replace(healthy(market=APPLE), candle_age=timedelta(hours=3)), limits_for(AssetClass.EQUITY)
    )
    assert not verdict.eligible
    assert any("last bar is" in reason for reason in verdict.reasons)


def test_a_market_too_quiet_to_cover_its_costs_is_refused() -> None:
    """SPY's real 0.04% bar cannot pay a 0.12% round trip, and is refused for it."""
    verdict = assess_eligibility(replace(healthy(market=INDEX), volatility=Decimal("0.0004")))
    assert not verdict.eligible
    assert any("round-trip cost" in reason for reason in verdict.reasons)


def test_a_market_too_violent_to_control_is_refused() -> None:
    verdict = assess_eligibility(replace(healthy(), volatility=Decimal("0.09")))
    assert not verdict.eligible
    assert any("risk budget" in reason for reason in verdict.reasons)


def test_a_flash_move_relative_to_the_typical_range_is_refused() -> None:
    """Five times the normal bar is a shock, not a trend accelerating."""
    verdict = assess_eligibility(replace(healthy(), last_bar_range=Decimal("30.6")))
    assert not verdict.eligible
    assert any("flash move" in reason and "typical range" in reason for reason in verdict.reasons)


def test_a_flash_move_in_absolute_terms_is_refused() -> None:
    """The absolute breaker catches what the relative one cannot: a permanently wild market."""
    verdict = assess_eligibility(replace(healthy(), last_bar_return=Decimal("-0.08")))
    assert not verdict.eligible
    assert any("single-bar breaker" in reason for reason in verdict.reasons)


def test_a_flat_typical_range_does_not_disable_the_absolute_breaker() -> None:
    """No 'normal' to be a multiple of must not become a free pass."""
    verdict = assess_eligibility(
        replace(
            healthy(),
            typical_bar_range=Decimal("0"),
            last_bar_range=Decimal("500"),
            last_bar_return=Decimal("0.09"),
        )
    )
    assert not verdict.eligible
    assert any("single-bar breaker" in reason for reason in verdict.reasons)


def test_an_order_below_the_venue_lot_minimum_is_refused() -> None:
    verdict = assess_eligibility(replace(healthy(), intended_quantity=Decimal("0.0001")))
    assert not verdict.eligible
    assert any("venue minimum" in reason for reason in verdict.reasons)


def test_an_order_above_the_venue_lot_maximum_is_refused() -> None:
    verdict = assess_eligibility(replace(healthy(), intended_quantity=Decimal("9999")))
    assert not verdict.eligible
    assert any("venue maximum" in reason for reason in verdict.reasons)


def test_an_order_below_the_venue_notional_minimum_is_refused() -> None:
    """Gold's smallest lot is worth 3.40 against a 5 USDT floor — the two disagree."""
    tiny = replace(healthy(), intended_quantity=Decimal("0.001"))
    verdict = assess_eligibility(tiny)
    assert not verdict.eligible
    assert any("notional" in reason and "venue minimum" in reason for reason in verdict.reasons)


def test_an_order_too_large_for_the_bar_is_refused() -> None:
    """The ceiling that actually keeps the thin end of the equity list out."""
    verdict = assess_eligibility(replace(healthy(), bar_quote_volume=Decimal("10000")))
    assert not verdict.eligible
    assert any("liquidity ceiling" in reason for reason in verdict.reasons)


def test_a_stop_inside_the_tick_grid_is_refused() -> None:
    verdict = assess_eligibility(replace(healthy(), stop_distance=Decimal("0.05")))
    assert not verdict.eligible
    assert any("stop distance" in reason for reason in verdict.reasons)


def test_a_stop_inside_the_spread_is_refused() -> None:
    """A stop narrower than the spread measures the market existing, not an adverse move."""
    verdict = assess_eligibility(
        replace(
            healthy(),
            bid=Decimal("3399.00"),
            ask=Decimal("3400.00"),
            stop_distance=Decimal("0.5"),
        )
    )
    assert not verdict.eligible
    assert any("stop distance" in reason for reason in verdict.reasons)


def test_every_failure_is_reported_not_just_the_first() -> None:
    """An operator who fixes one objection must not rediscover the next a bar later."""
    verdict = assess_eligibility(
        replace(
            healthy(),
            quote_volume_24h=Decimal("1"),
            bid=Decimal("3300"),
            ask=Decimal("3500"),
            ticker_age=timedelta(minutes=10),
            volatility=Decimal("0.5"),
            stop_distance=Decimal("0"),
        )
    )
    assert not verdict.eligible
    assert len(verdict.reasons) >= 5, verdict.reasons


# --------------------------------------------------------------------------------------
# Per-class limits
# --------------------------------------------------------------------------------------


def test_every_class_has_declared_limits() -> None:
    """A class with no band would otherwise trade under someone else's numbers."""
    for asset_class in AssetClass:
        assert asset_class in LIMITS_BY_CLASS
        assert limits_for(asset_class) is LIMITS_BY_CLASS[asset_class]


def test_the_meme_volatility_floor_would_delete_the_non_crypto_classes() -> None:
    """The reason the bands are per class, stated as a test.

    Gold's typical 15m bar is 0.18%, crude's 0.32%, Apple's 0.13% and SPY's 0.04% — and
    BTC's is 0.20%. The meme floor of 0.4% rejects every one of them, so reusing it would
    not filter these markets, it would remove them.
    """
    meme_floor = LIMITS_BY_CLASS[AssetClass.MEME].min_volatility
    assert meme_floor == Decimal("0.004")
    measured = {
        AssetClass.METAL: Decimal("0.0018"),
        AssetClass.ENERGY: Decimal("0.0032"),
        AssetClass.EQUITY: Decimal("0.0013"),
        AssetClass.CRYPTO: Decimal("0.0020"),
    }
    for asset_class, volatility in measured.items():
        assert volatility < meme_floor, "the premise: these are all quieter than a meme"
        assert (
            volatility >= limits_for(asset_class).min_volatility
        ), f"{asset_class} must still be tradable under its own band"


def test_the_non_crypto_volatility_floor_is_the_round_trip_cost() -> None:
    """Derived, not picked: 2 x the venue's 6bp taker fee."""
    for asset_class in (AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX):
        assert limits_for(asset_class).min_volatility == Decimal("0.0012")


def test_non_crypto_flash_breakers_are_tighter_than_the_meme_one() -> None:
    """A 10% bar is an afternoon for a meme and a once-a-decade event in Brent."""
    meme = limits_for(AssetClass.MEME).max_abs_bar_return
    for asset_class in (AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX):
        assert limits_for(asset_class).max_abs_bar_return < meme


# --------------------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------------------


def cost_inputs(market: AssetMarket = GOLD, **overrides: object) -> CostInputs:
    kwargs: dict[str, object] = {
        "market": market,
        "bid": Decimal("3400.00"),
        "ask": Decimal("3400.02"),
    }
    kwargs.update(overrides)
    return CostInputs(**kwargs)  # type: ignore[arg-type]


def test_fees_come_from_the_venue_not_from_a_crypto_default() -> None:
    """Bybit charges 6bp taker on these perpetuals, against 10bp on spot.

    Blanket-applying a spot crypto fee would overstate the round trip by two thirds and
    refuse trades that were comfortably economic.
    """
    cost = all_in_cost(cost_inputs())
    assert cost.fees == Decimal("0.0012"), "2 x the venue's 6bp taker fee"

    maker = all_in_cost(cost_inputs(is_maker=True))
    assert maker.fees == Decimal("0.0002"), "2 x the venue's 1bp maker fee"


def test_a_venue_with_different_fees_produces_different_costs() -> None:
    """The fee is read per instrument, so a market on another schedule prices differently."""
    expensive = replace(GOLD, taker_fee=Decimal("0.001"))
    assert all_in_cost(cost_inputs(expensive)).fees == Decimal("0.002")


def test_the_spread_is_charged_once_per_round_trip_not_twice() -> None:
    """Half in and half out sums to one spread; double-counting refuses every trade."""
    cost = all_in_cost(cost_inputs(bid=Decimal("3400"), ask=Decimal("3404")))
    # 4 wide on a 3402 mid.
    assert cost.spread == pytest.approx(Decimal("4") / Decimal("3402"), rel=Decimal("0.001"))


def test_slippage_is_a_per_class_allowance_over_two_legs() -> None:
    """Single names get the widest allowance; metals the tightest."""
    metal = all_in_cost(cost_inputs(GOLD)).slippage
    equity = all_in_cost(cost_inputs(APPLE)).slippage
    assert metal == Decimal("2") / Decimal("10000"), "1bp a leg, two legs"
    assert equity == Decimal("6") / Decimal("10000"), "3bp a leg, two legs"
    assert equity > metal


def test_funding_scales_with_the_venues_own_interval() -> None:
    """Metals settle twice as often as energy, so the same hold costs twice as much.

    This is what an assumed 8-hour interval gets wrong, and it gets it wrong by exactly a
    factor of two on half the non-crypto universe.
    """
    rate = Decimal("0.0001")
    gold = all_in_cost(cost_inputs(GOLD, funding_rate=rate, holding_period=timedelta(hours=8)))
    crude = all_in_cost(cost_inputs(CRUDE, funding_rate=rate, holding_period=timedelta(hours=8)))

    assert GOLD.funding_interval_minutes == 240
    assert CRUDE.funding_interval_minutes == 480
    assert gold.funding == rate * Decimal("2"), "two 4-hour settlements in 8 hours"
    assert crude.funding == rate * Decimal("1"), "one 8-hour settlement in 8 hours"
    assert gold.funding == crude.funding * 2


def test_a_short_receives_the_funding_a_long_pays() -> None:
    """Positive rate means longs pay shorts; the sign is a real cash flow, not an absolute."""
    rate = Decimal("0.0001")
    long = all_in_cost(cost_inputs(GOLD, funding_rate=rate, side_is_long=True))
    short = all_in_cost(cost_inputs(GOLD, funding_rate=rate, side_is_long=False))
    assert long.funding > 0
    assert short.funding == -long.funding


def test_an_unmeasured_funding_rate_contributes_zero_and_says_so() -> None:
    """A guess would be worse than a gap — but a silent gap would be worse than both."""
    cost = all_in_cost(cost_inputs(GOLD, funding_rate=None))
    assert cost.funding == Decimal("0")
    assert any("funding excluded" in note for note in cost.assumptions)
    assert any("understates cost" in note for note in cost.assumptions)


def test_a_market_that_does_not_fund_is_charged_no_funding() -> None:
    """Spot has no funding interval, so a rate supplied for it must be ignored."""
    spot_like = replace(GOLD, funding_interval_minutes=None)
    cost = all_in_cost(cost_inputs(spot_like, funding_rate=Decimal("0.01")))
    assert cost.funding == Decimal("0")


def test_slippage_is_always_declared_as_an_assumption() -> None:
    """It is the one component the venue never reports, and that must stay visible."""
    cost = all_in_cost(cost_inputs(GOLD, funding_rate=Decimal("0")))
    assert any("slippage" in note and "not a measurement" in note for note in cost.assumptions)


def test_a_move_that_only_just_covers_costs_does_not_clear() -> None:
    """Breaking even is a reason to be indifferent, not a reason to consume a position slot."""
    inputs = cost_inputs(GOLD, funding_rate=Decimal("0"))
    total = all_in_cost(inputs).total
    verdict = clears_costs(total, inputs)
    assert not verdict.clears
    assert verdict.net_edge == Decimal("0")
    assert verdict.reason is not None
    assert "net-profit buffer" in verdict.reason


def test_a_move_that_clears_costs_and_the_buffer_is_accepted() -> None:
    inputs = cost_inputs(GOLD, funding_rate=Decimal("0"))
    total = all_in_cost(inputs).total
    verdict = clears_costs(total + Decimal("0.002"), inputs)
    assert verdict.clears
    assert verdict.reason is None
    assert verdict.net_edge >= verdict.buffer


def test_the_rejection_reason_itemises_every_component() -> None:
    """Naming only the total tells an operator it failed, not what to change."""
    inputs = cost_inputs(GOLD, funding_rate=Decimal("0.0002"))
    verdict = clears_costs(Decimal("0.0005"), inputs)
    assert not verdict.clears
    assert verdict.reason is not None
    for component in ("fees", "spread", "slippage", "funding"):
        assert component in verdict.reason


def test_a_costlier_class_needs_a_larger_move_for_the_same_verdict() -> None:
    """The whole point of per-class costs: the same edge is not economic everywhere."""
    move = Decimal("0.0035")
    gold = clears_costs(move, cost_inputs(GOLD, funding_rate=Decimal("0")))
    apple = clears_costs(move, cost_inputs(APPLE, funding_rate=Decimal("0")))
    assert gold.cost.total < apple.cost.total
    assert gold.clears
    assert not apple.clears


@pytest.mark.parametrize("bad", [Decimal("-0.01")])
def test_a_negative_expected_move_or_buffer_is_rejected(bad: Decimal) -> None:
    """Nonsense in must not become a verdict out."""
    with pytest.raises(ValidationError):
        clears_costs(bad, cost_inputs())
    with pytest.raises(ValidationError):
        clears_costs(Decimal("0.01"), cost_inputs(), buffer=bad)


# --------------------------------------------------------------------------------------
# Strategy gating
# --------------------------------------------------------------------------------------


def test_crypto_and_memes_admit_every_family() -> None:
    """The taxonomy was built on them and the library was validated against them."""
    for asset_class in (AssetClass.CRYPTO, AssetClass.MEME):
        assert ALLOWED_FAMILIES_BY_CLASS[asset_class] is None
        for family in (
            "trend",
            "momentum",
            "breakout",
            "reversion",
            "volatility",
            "volume",
            "structure",
        ):
            assert family_supports_class(family, asset_class)


@pytest.mark.parametrize(
    "asset_class",
    [AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX],
)
@pytest.mark.parametrize("family", ["trend", "momentum", "breakout", "reversion", "volatility"])
def test_the_requested_families_are_admitted_on_every_non_crypto_class(
    asset_class: AssetClass, family: str
) -> None:
    """Metals and energy get trend, momentum, breakout, reversion and volatility; so do
    equities and indices."""
    assert family_supports_class(family, asset_class)


@pytest.mark.parametrize(
    "asset_class",
    [AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX],
)
def test_volume_strategies_are_refused_outside_crypto(asset_class: AssetClass) -> None:
    """A synthetic perpetual's volume measures the derivative, not the underlying.

    AAPL's perpetual turns over a few million a day against many billions in the share
    itself, so OBV or MFI read on it are reading a shadow of participation.
    """
    assert not family_supports_class("volume", asset_class)


@pytest.mark.parametrize(
    "asset_class",
    [AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX],
)
def test_swing_structure_is_refused_outside_crypto(asset_class: AssetClass) -> None:
    """Pivots need a continuous tape; these underlyings gap over their own closed sessions."""
    assert not family_supports_class("structure", asset_class)


def test_an_unmapped_family_is_refused_outside_crypto() -> None:
    """A strategy nobody has classified reads a source nobody has checked here."""
    assert not family_supports_class("unmapped:something_new", AssetClass.METAL)
    assert family_supports_class("unmapped:something_new", AssetClass.CRYPTO)


def test_a_strategys_own_declaration_can_refuse_a_class_its_family_admits() -> None:
    """The two vetoes are ANDed, and either one is sufficient."""
    assert strategy_supports_class("trend", AssetClass.METAL)
    assert not strategy_supports_class(
        "trend", AssetClass.METAL, declared=frozenset({AssetClass.CRYPTO})
    )


def test_a_strategys_declaration_cannot_override_its_family() -> None:
    """Declaring itself universal must not buy a volume strategy an equity."""
    assert not strategy_supports_class(
        "volume", AssetClass.EQUITY, declared=frozenset({str(c) for c in AssetClass})
    )


def test_an_empty_declaration_means_every_class() -> None:
    """The default a strategy carries when it says nothing at all."""
    assert strategy_supports_class("trend", AssetClass.METAL, declared=frozenset())
    assert strategy_supports_class("trend", AssetClass.METAL, declared=None)


def test_declarations_may_be_plain_strings_or_enum_members() -> None:
    """``AssetClass`` is a ``StrEnum``, so the strategy layer need not import it."""
    assert strategy_supports_class("trend", AssetClass.METAL, declared=frozenset({"metal"}))
    assert not strategy_supports_class("trend", AssetClass.METAL, declared=frozenset({"equity"}))


# --------------------------------------------------------------------------------------
# Context metadata
# --------------------------------------------------------------------------------------


def test_the_asset_class_round_trips_through_context_metadata() -> None:
    """The engine records it; the orchestrator reads it back."""
    for asset_class in AssetClass:
        metadata = {ASSET_CLASS_METADATA_KEY: asset_class}
        assert asset_class_from_metadata(metadata) is asset_class
        # And as the plain string a serialised context would carry.
        as_string = {ASSET_CLASS_METADATA_KEY: asset_class.value}
        assert asset_class_from_metadata(as_string) is asset_class


@pytest.mark.parametrize("metadata", [{}, {"asset_class": "nonsense"}, {"asset_class": 7}])
def test_missing_or_unusable_metadata_falls_back_to_crypto(metadata: dict[str, object]) -> None:
    """An engine that predates the classification must lose no capability.

    Crypto admits every family, so an unpopulated context behaves exactly as it did before
    this module existed rather than acquiring a gate it was never designed for.
    """
    assert asset_class_from_metadata(metadata) is AssetClass.CRYPTO
