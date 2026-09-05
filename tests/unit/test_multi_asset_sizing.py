"""Sizing must produce venue-legal, non-zero quantities on every asset class.

``tests/unit/test_position_sizing_nonzero.py`` pins the same property for BTC after seven
live candidates were dropped with ``below_venue_min_quantity``. The non-crypto classes
reintroduce that failure through a different door, and the door is the one Bybit's gold
contract happens to sit in front of.

A venue states **two** minimums and satisfying one does not satisfy the other. ``XAUUSDT``
has a minimum quantity of 0.001 and a minimum notional of 5 USDT. With gold near 3,400 that
smallest legal lot is worth 3.40 — *under* the venue's own notional floor. Bumping a
rounded-down order up to ``min_quantity`` therefore produced a size that cleared the lot
check and failed the notional check on the very next line, and the sizer returned zero for
a market it could have traded perfectly well at 0.002.

Every number below is the venue's, read from the Bybit demo host on 2026-08-14: lot steps,
tick sizes, minimums, maximums and leverage caps are what ``instruments-info`` returns for
these contracts.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from quantflow.core.config import RiskSettings
from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Instrument
from quantflow.exchange.bybit.mapping import parse_instrument
from quantflow.risk.sizing import FixedFractionalSizer, SizingRequest

#: The demo account's actual balance while this was built.
VENUE_EQUITY = Decimal("49903.38")


def venue_market(
    symbol: str,
    *,
    qty_step: str,
    min_qty: str,
    tick: str,
    max_qty: str,
    symbol_type: str,
    max_leverage: str,
    funding_interval: int = 480,
    min_notional: str = "5",
) -> dict[str, Any]:
    """A CCXT market dict shaped exactly like Bybit's linear perpetual payload."""
    return {
        "symbol": f"{symbol}:USDT",
        "spot": False,
        "swap": True,
        "future": False,
        "option": False,
        "linear": True,
        "expiry": None,
        "active": True,
        # The venue's perpetual schedule: 1bp maker, 6bp taker. Not the 10bp/10bp of spot.
        "maker": 0.0001,
        "taker": 0.0006,
        "contractSize": 1.0,
        "precision": {"amount": float(qty_step), "price": float(tick)},
        "limits": {
            "amount": {"min": float(min_qty), "max": float(max_qty)},
            "cost": {"min": None},
            "leverage": {"max": float(max_leverage)},
        },
        "info": {
            "symbol": symbol.replace("/", ""),
            "symbolType": symbol_type,
            "status": "Trading",
            "fundingInterval": funding_interval,
            "lotSizeFilter": {
                "qtyStep": qty_step,
                "minOrderQty": min_qty,
                "maxOrderQty": max_qty,
                "minNotionalValue": min_notional,
            },
            "priceFilter": {"tickSize": tick},
            "leverageFilter": {"minLeverage": "1", "maxLeverage": max_leverage},
        },
    }


#: Gold. The contract that exposed the two-minimums bug: 0.001 lots at ~3,400 are worth
#: 3.40, which is below the venue's own 5 USDT floor.
XAU_PERP = venue_market(
    "XAU/USDT",
    qty_step="0.001",
    min_qty="0.001",
    tick="0.01",
    max_qty="500.000",
    symbol_type="commodity",
    max_leverage="100.00",
    funding_interval=240,
)
#: Silver. Same 0.001 grid; at ~38 the smallest lot is worth 0.038 — far under the floor,
#: so the notional-derived bump has to travel much further than one step.
XAG_PERP = venue_market(
    "XAG/USDT",
    qty_step="0.001",
    min_qty="0.001",
    tick="0.01",
    max_qty="20000.000",
    symbol_type="commodity",
    max_leverage="100.00",
    funding_interval=240,
)
#: WTI crude. Note the venue reports a three-decimal tick here and a two-decimal lot step.
CL_PERP = venue_market(
    "CL/USDT",
    qty_step="0.01",
    min_qty="0.01",
    tick="0.010",
    max_qty="3000.00",
    symbol_type="commodity",
    max_leverage="100.00",
)
#: Brent crude.
BZ_PERP = venue_market(
    "BZ/USDT",
    qty_step="0.01",
    min_qty="0.01",
    tick="0.01",
    max_qty="3400.00",
    symbol_type="commodity",
    max_leverage="100.00",
)
#: A single-name equity perpetual. Leverage caps at 50x rather than the metals' 100x.
AAPL_PERP = venue_market(
    "AAPL/USDT",
    qty_step="0.01",
    min_qty="0.01",
    tick="0.01",
    max_qty="2800.00",
    symbol_type="stock",
    max_leverage="50.00",
)
#: An index ETF perpetual. The venue caps these at 25x — the tightest of the three tiers.
SPY_PERP = venue_market(
    "SPY/USDT",
    qty_step="0.01",
    min_qty="0.01",
    tick="0.01",
    max_qty="960.00",
    symbol_type="stock",
    max_leverage="25.00",
)

#: Reference prices at the time the metadata above was read.
PRICES: dict[str, str] = {
    "XAU/USDT": "3400.00",
    "XAG/USDT": "38.00",
    "CL/USDT": "63.00",
    "BZ/USDT": "67.00",
    "AAPL/USDT": "230.00",
    "SPY/USDT": "640.00",
}

ALL_MARKETS = [XAU_PERP, XAG_PERP, CL_PERP, BZ_PERP, AAPL_PERP, SPY_PERP]


def instrument_for(market: dict[str, Any]) -> Instrument:
    """Parse a venue market, failing loudly if the parser rejects a tradable contract."""
    parsed = parse_instrument(market)
    assert parsed is not None, f"{market['symbol']} must parse into a tradable instrument"
    return parsed


def risk_settings(**overrides: object) -> RiskSettings:
    """The demo bot's live risk configuration."""
    kwargs: dict[str, object] = {
        "max_position_pct": Decimal("0.05"),
        "max_order_notional": Decimal("12000"),
        "max_total_exposure_pct": Decimal("1"),
        "max_leverage": Decimal("1"),
        "min_order_notional": Decimal("5"),
    }
    kwargs.update(overrides)
    return RiskSettings(**kwargs)  # type: ignore[arg-type]


def size_one(
    market: dict[str, Any],
    *,
    equity: Decimal = VENUE_EQUITY,
    stop_pct: Decimal = Decimal("0.02"),
    settings: RiskSettings | None = None,
    long: bool = True,
    risk_per_trade: Decimal = Decimal("0.01"),
) -> tuple[Instrument, Decimal, Any]:
    """Size one candidate the way the live path does; hand back instrument, price, result."""
    instrument = instrument_for(market)
    price = Decimal(PRICES[str(instrument.symbol)])
    stop = price * (Decimal("1") - stop_pct) if long else price * (Decimal("1") + stop_pct)
    sizer = FixedFractionalSizer(settings or risk_settings(), risk_per_trade=risk_per_trade)
    return (
        instrument,
        price,
        sizer.size(
            SizingRequest(
                equity=equity,
                price=price,
                instrument=instrument,
                stop_loss_price=stop,
                available_cash=equity,
            )
        ),
    )


def assert_venue_legal(instrument: Instrument, quantity: Decimal, price: Decimal) -> None:
    """A quantity is only non-zero in a useful sense if the venue would accept it."""
    assert quantity > ZERO, "a valid candidate must produce a non-zero quantity"
    assert quantity >= instrument.min_quantity, "below the venue lot minimum"
    assert quantity % instrument.quantity_step == ZERO, "must sit on the venue's lot grid"
    if instrument.max_quantity is not None:
        assert quantity <= instrument.max_quantity, "above the venue lot maximum"
    assert instrument.notional(quantity, price) >= instrument.min_notional, "under min notional"
    instrument.validate_order(quantity, price, check_price_tick=False)


# --------------------------------------------------------------------------------------
# The core regression: a valid size must not resolve to zero on any class
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("market", ALL_MARKETS, ids=lambda m: str(m["symbol"]))
@pytest.mark.parametrize("long", [True, False], ids=["long", "short"])
def test_every_asset_class_sizes_to_a_venue_legal_quantity(
    market: dict[str, Any], long: bool
) -> None:
    """Gold, silver, WTI, Brent, an equity and an index ETF all size to a real order.

    Long and short both, because a short is sized from a stop *above* the entry and the
    arithmetic is not symmetric — an error in the stop-distance sign shows up as a wildly
    different quantity on one side only, which is exactly the kind of defect that survives
    a one-sided test.
    """
    instrument, price, result = size_one(market, long=long)
    assert result.is_tradable, f"{instrument.symbol} produced no size: {result.detail}"
    assert_venue_legal(instrument, result.quantity, price)


@pytest.mark.parametrize("market", ALL_MARKETS, ids=lambda m: str(m["symbol"]))
def test_no_class_is_refused_for_being_under_a_venue_minimum(market: dict[str, Any]) -> None:
    """The specific rejection paths this change exists to eliminate never fire here."""
    _, _, result = size_one(market)
    assert result.capped_by not in {
        "below_venue_min_quantity",
        "below_min_notional",
        "violates_venue_rules",
    }, f"refused as {result.capped_by}: {result.detail}"


def test_gold_at_the_venue_floor_clears_the_notional_minimum_not_just_the_lot_minimum() -> None:
    """The exact bug, pinned at the size where the two minimums disagree.

    Squeezing the caps until the sizer is forced all the way down to the venue floor: the
    lot minimum alone (0.001, worth 3.40) is under the 5 USDT notional minimum, so a bump
    that stops at ``min_quantity`` returns zero. The correct floor is 0.002, worth 6.80.
    """
    instrument = instrument_for(XAU_PERP)
    price = Decimal(PRICES["XAU/USDT"])

    assert instrument.min_quantity == Decimal("0.001")
    assert (
        instrument.notional(instrument.min_quantity, price) < instrument.min_notional
    ), "this test is meaningless unless the lot minimum really is below the notional one"

    # A tiny risk budget drives the raw size under the floor, forcing the bump path.
    settings = risk_settings(max_position_pct=Decimal("0.0005"))
    sizer = FixedFractionalSizer(settings, risk_per_trade=Decimal("0.0000001"))
    result = sizer.size(
        SizingRequest(
            equity=VENUE_EQUITY,
            price=price,
            instrument=instrument,
            stop_loss_price=price * Decimal("0.98"),
            available_cash=VENUE_EQUITY,
        )
    )

    assert result.is_tradable, f"gold must still be tradable at the floor: {result.detail}"
    assert result.quantity == Decimal("0.002"), "the floor is notional-derived, not lot-derived"
    assert_venue_legal(instrument, result.quantity, price)


def test_a_venue_floor_that_would_breach_a_cap_is_still_refused() -> None:
    """The bump must not become a way to exceed a risk limit.

    The floor is honoured only when the resulting order still respects every cap. A
    max_order_notional below the smallest legal gold order has to refuse the trade, not
    round up through the ceiling — the whole point of the caps is that the venue's
    convenience does not override them.
    """
    instrument = instrument_for(XAU_PERP)
    price = Decimal(PRICES["XAU/USDT"])
    # The smallest legal order is 0.002 XAU = 6.80. Cap orders below that.
    settings = risk_settings(max_order_notional=Decimal("6"), min_order_notional=Decimal("1"))
    sizer = FixedFractionalSizer(settings, risk_per_trade=Decimal("0.0000001"))
    result = sizer.size(
        SizingRequest(
            equity=VENUE_EQUITY,
            price=price,
            instrument=instrument,
            stop_loss_price=price * Decimal("0.98"),
            available_cash=VENUE_EQUITY,
        )
    )

    assert not result.is_tradable
    assert result.capped_by == "below_venue_min_quantity"
    assert result.detail is not None
    assert "max_order_notional" in result.detail


# --------------------------------------------------------------------------------------
# The venue's own contract metadata must be what sizing reads
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("market", "tick", "step", "min_qty", "max_qty", "leverage"),
    [
        (XAU_PERP, "0.01", "0.001", "0.001", "500.000", "100.00"),
        (XAG_PERP, "0.01", "0.001", "0.001", "20000.000", "100.00"),
        (CL_PERP, "0.010", "0.01", "0.01", "3000.00", "100.00"),
        (BZ_PERP, "0.01", "0.01", "0.01", "3400.00", "100.00"),
        (AAPL_PERP, "0.01", "0.01", "0.01", "2800.00", "50.00"),
        (SPY_PERP, "0.01", "0.01", "0.01", "960.00", "25.00"),
    ],
    ids=lambda value: str(value)[:14],
)
def test_contract_rules_come_from_the_venue_not_from_a_default(
    market: dict[str, Any],
    tick: str,
    step: str,
    min_qty: str,
    max_qty: str,
    leverage: str,
) -> None:
    """Every rule sizing depends on is the venue's stated value, parsed from its filters."""
    instrument = instrument_for(market)
    assert instrument.price_tick == Decimal(tick)
    assert instrument.quantity_step == Decimal(step)
    assert instrument.min_quantity == Decimal(min_qty)
    assert instrument.max_quantity == Decimal(max_qty)
    assert instrument.min_notional == Decimal("5")
    assert instrument.max_leverage == Decimal(leverage)
    assert instrument.contract_size == Decimal("1")
    # The perpetual schedule, not spot's 10bp/10bp.
    assert instrument.taker_fee == Decimal("0.0006")
    assert instrument.maker_fee == Decimal("0.0001")


def test_the_configured_leverage_is_the_binding_one_at_real_venue_values() -> None:
    """With today's numbers, our own limit is always the tighter of the two.

    ``RiskSettings`` caps ``max_leverage`` at 20 and the venue's lowest ceiling on these
    contracts is 25x, so the venue's limit cannot bind in production. Pinning that here
    means the day it stops being true — a new class listed at 10x, or the configured cap
    raised — this test fails and says so, rather than the exchange rejecting the order.
    """
    for market in ALL_MARKETS:
        instrument = instrument_for(market)
        assert instrument.max_leverage >= Decimal("25"), (
            f"{instrument.symbol} now allows less leverage than RiskSettings can request; "
            "the venue ceiling has become the binding limit"
        )


def test_the_venue_leverage_ceiling_binds_when_it_is_the_tighter_one() -> None:
    """When a venue allows less than we would ask for, the venue wins.

    Constructed rather than parsed, because no contract the bot currently trades caps below
    the configured limit — see the test above. The behaviour still has to be correct for
    the case, since sizing against a leverage the venue does not grant produces an order it
    rejects outright.
    """
    from quantflow.core.config import MarketType
    from quantflow.domain.instruments import Symbol

    # Cash well below equity, so the leverage term (a multiple of *cash*) is the smallest
    # candidate rather than the position cap (a fraction of *equity*).
    equity = Decimal("10000")
    cash = Decimal("100")
    price = Decimal("100")
    constrained = Instrument(
        symbol=Symbol("TEST", "USDT"),
        market_type=MarketType.FUTURE,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.01"),
        min_quantity=Decimal("0.01"),
        min_notional=Decimal("5"),
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0006"),
        # The venue grants 2x; the configuration below asks for 20x.
        max_leverage=Decimal("2"),
    )
    sizer = FixedFractionalSizer(
        risk_settings(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("1"),
            max_order_notional=Decimal("1000000"),
        ),
        risk_per_trade=Decimal("1"),
    )
    result = sizer.size(
        SizingRequest(
            equity=equity,
            price=price,
            instrument=constrained,
            stop_loss_price=price * Decimal("0.999"),
            available_cash=cash,
        )
    )

    assert result.capped_by == "max_leverage"
    # 2x on 100 of cash is 200 of notional, not the 2,000 that 20x was configured to allow.
    assert result.notional <= cash * constrained.max_leverage
    assert result.notional > cash, "the venue's 2x, not collapsed to unlevered cash"


def test_funding_interval_is_read_per_instrument() -> None:
    """Metals settle every 4 hours and everything else every 8. Neither is assumed."""
    assert instrument_for(XAU_PERP).funding_interval_minutes == 240
    assert instrument_for(XAG_PERP).funding_interval_minutes == 240
    assert instrument_for(CL_PERP).funding_interval_minutes == 480
    assert instrument_for(AAPL_PERP).funding_interval_minutes == 480


def test_venue_symbol_type_survives_parsing() -> None:
    """The classification key must reach the instrument, or every class becomes crypto."""
    assert instrument_for(XAU_PERP).venue_symbol_type == "commodity"
    assert instrument_for(AAPL_PERP).venue_symbol_type == "stock"


def test_a_market_without_the_new_fields_still_parses() -> None:
    """A venue or fixture that reports neither field must not fail to load.

    The fields are additive: an exchange with no ``symbolType`` and no funding interval —
    spot, or any non-Bybit venue — has to keep producing a usable instrument rather than
    failing at the boundary over metadata nothing depends on.
    """
    market = venue_market(
        "XAU/USDT",
        qty_step="0.001",
        min_qty="0.001",
        tick="0.01",
        max_qty="500.000",
        symbol_type="commodity",
        max_leverage="100.00",
    )
    del market["info"]["symbolType"]
    del market["info"]["fundingInterval"]

    instrument = instrument_for(market)
    assert instrument.venue_symbol_type == ""
    assert instrument.funding_interval_minutes is None
    assert instrument.min_quantity == Decimal("0.001")


@pytest.mark.parametrize("bad", ["", "not-a-number", None, 0, -240])
def test_a_malformed_funding_interval_degrades_to_unknown(bad: object) -> None:
    """A malformed venue field must not crash the load, and must not become a fake number."""
    market = venue_market(
        "CL/USDT",
        qty_step="0.01",
        min_qty="0.01",
        tick="0.010",
        max_qty="3000.00",
        symbol_type="commodity",
        max_leverage="100.00",
    )
    market["info"]["fundingInterval"] = bad
    assert instrument_for(market).funding_interval_minutes is None


# --------------------------------------------------------------------------------------
# Existing behaviour must be unchanged
# --------------------------------------------------------------------------------------


def test_a_lot_dominated_instrument_is_unaffected_by_the_notional_floor() -> None:
    """Where ``min_quantity`` already dominates, the floor is exactly ``min_quantity``.

    Most instruments are in this case, and the change must be invisible to them. Brent's
    smallest lot is 0.01 at ~67, worth 0.67 — still under the 5 USDT floor, so it bumps to
    0.08. Contrast with an instrument priced high enough that one lot already clears it.
    """
    instrument = instrument_for(XAU_PERP)
    # One lot of gold at 6,000 is worth 6.00, already over the 5 USDT minimum.
    floor = FixedFractionalSizer._venue_floor_quantity(instrument, Decimal("6000"))
    assert floor == instrument.min_quantity

    # At 3,400 it is not, and the floor moves up one step to 0.002.
    assert FixedFractionalSizer._venue_floor_quantity(instrument, Decimal("3400")) == Decimal(
        "0.002"
    )
