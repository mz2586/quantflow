"""A selected candidate must never be dropped at sizing for a reason that is not true.

Seven live candidates were rejected with ``position size resolved to zero
(below_venue_min_quantity)`` on BTC/USDT while the demo account held ~49,940 USDT. Two
independent defects produced that rejection, and each is pinned here:

* **The instrument was not the contract being traded.** Bybit reports its USDT options as
  ``linear`` markets, and CCXT symbols such as ``BTC/USDT:USDT-260821-52000-C`` collapse
  onto ``BTC/USDT`` once the settlement suffix is stripped. Loading markets last-write-wins
  meant an option's rules — 0.01 lot minimum, 0.00001 price tick — stood in for the
  perpetual's 0.001 and 0.1. A 0.01 BTC "minimum" is ~630 USDT of notional, ten times the
  real floor.
* **The equity was not the account's.** The runner reads the venue balance at startup, then
  ``_restore`` overwrote it with the 10,000 the session's own equity curve had opened at,
  so ``max_position_pct`` of 5% meant 500 USDT. 630 > 500, and the venue floor "breached" a
  cap that described no real account.

Every number here is the venue's: lot steps, minimums and multipliers are the ones Bybit
returns for its linear perpetuals.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from quantflow.core.config import ExchangeEnv, ExchangeSettings, MarketType, RiskSettings
from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.exchange.bybit.mapping import parse_instrument
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.risk.engine import RiskEngine
from quantflow.risk.sizing import FixedFractionalSizer, SizingRequest
from tests.conftest import REFERENCE_TIME

#: The balance the demo account actually held while entries were being refused.
VENUE_EQUITY = Decimal("49940.48")

#: The stale figure the session's equity curve kept reinstating on every restart.
STALE_EQUITY = Decimal("10000")


def venue_market(
    symbol: str,
    *,
    qty_step: str,
    min_qty: str,
    tick: str,
    min_notional: str = "5",
    max_qty: float = 1500.0,
    contract_size: float = 1.0,
    precision_amount: float | None = None,
) -> dict[str, Any]:
    """A CCXT market dict shaped like Bybit's, filters and all."""
    return {
        "symbol": f"{symbol}:USDT",
        "spot": False,
        "swap": True,
        "future": False,
        "option": False,
        "linear": True,
        "expiry": None,
        "active": True,
        "maker": 0.0002,
        "taker": 0.00055,
        "contractSize": contract_size,
        # CCXT's derived view. It cannot express a step of 1 or 100 unambiguously, which is
        # why the raw filters below are the ones that must win.
        "precision": {"amount": precision_amount or float(qty_step), "price": float(tick)},
        "limits": {"amount": {"min": float(min_qty), "max": max_qty}, "cost": {"min": None}},
        "info": {
            "lotSizeFilter": {
                "qtyStep": qty_step,
                "minOrderQty": min_qty,
                "minNotionalValue": min_notional,
            },
            "priceFilter": {"tickSize": tick},
        },
    }


def option_market(symbol: str = "BTC/USDT") -> dict[str, Any]:
    """A Bybit USDT option: ``linear``, and it collapses onto the perpetual's symbol."""
    return {
        "symbol": f"{symbol}:USDT-260821-52000-C",
        "spot": False,
        "swap": False,
        "future": False,
        "option": True,
        "linear": True,
        "expiry": 1787299200000,
        "active": True,
        "contractSize": 1.0,
        "precision": {"amount": 0.01, "price": 5.0},
        "limits": {"amount": {"min": 0.01, "max": 500.0}, "cost": {"min": None}},
        "info": {
            "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01"},
            "priceFilter": {"tickSize": "5"},
        },
    }


def dated_future_market(symbol: str = "BTC/USDT") -> dict[str, Any]:
    """A dated linear future. Same underlying, different contract, same collapsed symbol."""
    market = venue_market(symbol, qty_step="0.001", min_qty="0.001", tick="0.1")
    market["symbol"] = f"{symbol}:USDT-260821"
    market["swap"] = False
    market["future"] = True
    market["expiry"] = 1787299200000
    return market


#: The linear perpetuals the bot actually trades, exactly as Bybit describes them.
BTC_PERP = venue_market("BTC/USDT", qty_step="0.001", min_qty="0.001", tick="0.1", max_qty=1500.0)
BNB_PERP = venue_market("BNB/USDT", qty_step="0.01", min_qty="0.01", tick="0.1", max_qty=6000.0)
SOL_PERP = venue_market("SOL/USDT", qty_step="0.1", min_qty="0.1", tick="0.01", max_qty=32000.0)
# The venue's XRP step is 0.1; 1 is used here as the coarser grid the brief specifies, so
# the rounding assertions bite on a whole-unit lot.
XRP_PERP = venue_market(
    "XRP/USDT", qty_step="1", min_qty="1", tick="0.0001", max_qty=7500000.0, precision_amount=1.0
)
FARTCOIN_PERP = venue_market(
    "FARTCOIN/USDT",
    qty_step="1",
    min_qty="1",
    tick="0.00001",
    max_qty=3000000.0,
    precision_amount=1.0,
)


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


def size(
    market: dict[str, Any],
    *,
    price: str,
    equity: Decimal = VENUE_EQUITY,
    stop_pct: Decimal = Decimal("0.02"),
    settings: RiskSettings | None = None,
    long: bool = True,
) -> tuple[Instrument, Any]:
    """Size one candidate the way the live path does, and hand back both halves."""
    instrument = instrument_for(market)
    reference = Decimal(price)
    stop = reference * (Decimal("1") - stop_pct) if long else reference * (Decimal("1") + stop_pct)
    resolved = settings or risk_settings()
    sizer = FixedFractionalSizer(resolved, risk_per_trade=Decimal("0.01"))
    return instrument, sizer.size(
        SizingRequest(
            equity=equity,
            price=reference,
            instrument=instrument,
            stop_loss_price=stop,
            available_cash=equity,
        )
    )


def assert_venue_legal(instrument: Instrument, quantity: Decimal, price: Decimal) -> None:
    """A quantity is only non-zero in a useful sense if the venue would accept it."""
    assert quantity > ZERO, "a valid candidate must produce a non-zero quantity"
    assert quantity >= instrument.min_quantity
    assert quantity % instrument.quantity_step == ZERO, "must sit on the venue's lot grid"
    assert instrument.notional(quantity, price) >= instrument.min_notional
    instrument.validate_order(quantity, price, check_price_tick=False)


def portfolio(cash: Decimal) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=REFERENCE_TIME,
        base_currency="USDT",
        cash=cash,
        peak_equity=cash,
        day_start_equity=cash,
    )


class TestVenueEquitySizesEveryTradedSymbol:
    """1-4: 5% of the real balance is thousands of USDT, not zero, on every symbol."""

    def test_bnb_sizes_a_non_zero_venue_legal_quantity(self) -> None:
        instrument, result = size(BNB_PERP, price="610.20")
        assert_venue_legal(instrument, result.quantity, Decimal("610.20"))
        assert result.notional <= VENUE_EQUITY * Decimal("0.05")
        assert result.notional > Decimal("2000"), "a 5% slice of ~49,940, not a rounding crumb"

    def test_sol_sizes_a_non_zero_venue_legal_quantity(self) -> None:
        instrument, result = size(SOL_PERP, price="141.37")
        assert_venue_legal(instrument, result.quantity, Decimal("141.37"))
        assert result.notional <= VENUE_EQUITY * Decimal("0.05")
        assert result.notional > Decimal("2000")

    def test_xrp_sizes_a_non_zero_venue_legal_quantity(self) -> None:
        instrument, result = size(XRP_PERP, price="2.4531")
        assert_venue_legal(instrument, result.quantity, Decimal("2.4531"))
        assert result.notional <= VENUE_EQUITY * Decimal("0.05")
        assert result.notional > Decimal("2000")

    def test_fartcoin_sizes_a_non_zero_venue_legal_quantity(self) -> None:
        instrument, result = size(FARTCOIN_PERP, price="0.58412")
        # The regression: CCXT's precision of 1.0 was read as "one decimal place", giving a
        # 0.1 lot grid on a contract the venue only trades in whole units.
        assert instrument.quantity_step == Decimal("1")
        assert_venue_legal(instrument, result.quantity, Decimal("0.58412"))
        assert result.notional > Decimal("2000")


class TestDirection:
    """5-6: sizing is direction-agnostic; a short is sized exactly like a long."""

    async def test_a_long_candidate_reaches_the_broker_with_a_non_zero_quantity(
        self, clock
    ) -> None:
        instrument = instrument_for(BTC_PERP)
        engine = RiskEngine(risk_settings(), clock=clock)
        decision = await engine.evaluate_signal(
            Signal(
                symbol=instrument.symbol,
                direction=SignalDirection.LONG,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
                reference_price=Decimal("62965.60"),
            ),
            portfolio=portfolio(VENUE_EQUITY),
            instrument=instrument,
            reference_price=Decimal("62965.60"),
        )
        assert decision.approved, decision.reason
        assert decision.request is not None
        assert_venue_legal(instrument, decision.request.quantity, Decimal("62965.60"))

    async def test_a_short_candidate_reaches_the_broker_with_a_non_zero_quantity(
        self, clock
    ) -> None:
        instrument = instrument_for(BTC_PERP)
        engine = RiskEngine(risk_settings(), clock=clock)
        decision = await engine.evaluate_signal(
            Signal(
                symbol=instrument.symbol,
                direction=SignalDirection.SHORT,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
                reference_price=Decimal("62965.60"),
            ),
            portfolio=portfolio(VENUE_EQUITY),
            instrument=instrument,
            reference_price=Decimal("62965.60"),
        )
        assert decision.approved, decision.reason
        assert decision.request is not None
        assert_venue_legal(instrument, decision.request.quantity, Decimal("62965.60"))


class TestVenueRounding:
    """7-10: the arithmetic that turns a target notional into a legal lot."""

    def test_the_quantity_is_snapped_down_onto_the_venue_lot_grid(self) -> None:
        # 5% of 49,940.48 is 2,497.024; at 141.37 that is 17.6631... SOL, and the venue
        # trades SOL in tenths. Rounding *down* is what keeps the cap intact.
        _, result = size(SOL_PERP, price="141.37")
        assert result.quantity == Decimal("17.6")
        assert result.quantity * Decimal("141.37") <= VENUE_EQUITY * Decimal("0.05")
        assert isinstance(result.quantity, Decimal)

    def test_a_sub_minimum_size_is_lifted_to_the_venue_minimum_when_the_caps_allow(self) -> None:
        # A 500 USDT cap cannot buy a 0.01 BTC option lot, but it comfortably buys the
        # perpetual's 0.001 - which is the lot the engine actually trades.
        instrument = instrument_for(BTC_PERP)
        assert instrument.min_quantity == Decimal("0.001"), "the perpetual's floor, not an option's"
        _, result = size(BTC_PERP, price="62965.60", equity=STALE_EQUITY, stop_pct=Decimal("0.30"))
        assert_venue_legal(instrument, result.quantity, Decimal("62965.60"))
        assert result.quantity >= instrument.min_quantity

    def test_the_venue_min_notional_is_read_from_the_venue_not_defaulted(self) -> None:
        # CCXT leaves limits.cost.min empty for Bybit perps; the 5 USDT floor lives in
        # lotSizeFilter.minNotionalValue, and defaulting to 1e-8 let sub-minimum orders pass.
        instrument = instrument_for(BTC_PERP)
        assert instrument.min_notional == Decimal("5")
        _, result = size(
            BTC_PERP,
            price="62965.60",
            equity=Decimal("60"),
            stop_pct=Decimal("0.90"),
            settings=risk_settings(max_position_pct=Decimal("0.05")),
        )
        assert result.quantity == ZERO
        assert result.capped_by in {"below_venue_min_quantity", "below_min_notional"}

    def test_a_contract_multiplier_is_carried_through_the_notional(self) -> None:
        # 1000-style contracts price one lot at 1000 units of the token. Ignoring the
        # multiplier understates the notional by three orders of magnitude.
        market = venue_market(
            "1000PEPE/USDT",
            qty_step="100",
            min_qty="100",
            tick="0.000001",
            contract_size=1000.0,
            precision_amount=100.0,
        )
        instrument = instrument_for(market)
        assert instrument.quantity_step == Decimal("100")
        assert instrument.contract_size == Decimal("1000")
        price = Decimal("0.0000102")
        _, result = size(market, price=str(price), stop_pct=Decimal("0.05"))
        assert_venue_legal(instrument, result.quantity, price)
        assert result.notional == result.quantity * price * Decimal("1000")
        assert result.notional <= VENUE_EQUITY * Decimal("0.05")


class TestStaleInputs:
    """11-12: neither a stale price nor a stale equity may be sized against."""

    def test_a_stale_or_absent_price_is_refused_rather_than_sized_against(self) -> None:
        instrument = instrument_for(BTC_PERP)
        for bad_price in (Decimal("0"), Decimal("-1")):
            with pytest.raises(ValidationError, match="price must be positive"):
                SizingRequest(
                    equity=VENUE_EQUITY,
                    price=bad_price,
                    instrument=instrument,
                    stop_loss_price=Decimal("61000"),
                )
        # A stale-but-positive price still sizes, and against that price only: the notional
        # must reconcile to the price handed in, never to a remembered one.
        _, result = size(BTC_PERP, price="62965.60")
        assert result.notional == result.quantity * Decimal("62965.60")

    async def test_stale_session_cash_is_re_anchored_to_the_authoritative_venue_balance(
        self, clock
    ) -> None:
        from quantflow.core.config import TradingMode
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.base import Strategy, StrategyContext

        instrument = instrument_for(BTC_PERP)

        class Quiet(Strategy):
            """Inert: this test is about restored state, not about signals."""

            strategy_id = "quiet"
            name = "quiet"

            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:  # pragma: no cover - inert
                return Signal.hold(instrument.symbol, REFERENCE_TIME, self.name)

        engine = PaperTradingEngine(
            Quiet(),
            PaperConfig(
                symbols=(instrument.symbol,),
                timeframe=Timeframe.M15,
                starting_equity=VENUE_EQUITY,
                equity_is_authoritative=True,
                risk=risk_settings(),
                persist=False,
                mode=TradingMode.LIVE,
            ),
            instruments={instrument.symbol: instrument},
            clock=clock,
        )
        # The persisted curve's opening balance is not evidence about a live account.
        assert engine._anchor_cash_to_venue(STALE_EQUITY, ()) == VENUE_EQUITY

        # A configured fallback is not authoritative, and must not overwrite session state.
        engine._config = PaperConfig(
            symbols=(instrument.symbol,),
            timeframe=engine._config.timeframe,
            starting_equity=VENUE_EQUITY,
            equity_is_authoritative=False,
            persist=False,
            mode=TradingMode.LIVE,
        )
        assert engine._anchor_cash_to_venue(STALE_EQUITY, ()) == STALE_EQUITY


class TestRefusalsRemainHonest:
    """13-15: small still trades, genuinely-too-small still does not, and zero never ships."""

    def test_a_small_but_valid_position_still_sizes_non_zero(self) -> None:
        # A 40 USDT account against a 62,965 BTC: one 0.001 lot is 63 USDT of notional,
        # which a 1x cash cap cannot fund. A 200 USDT account can, and must.
        instrument, result = size(
            BTC_PERP,
            price="62965.60",
            equity=Decimal("200"),
            stop_pct=Decimal("0.50"),
            settings=risk_settings(max_position_pct=Decimal("0.4")),
        )
        assert_venue_legal(instrument, result.quantity, Decimal("62965.60"))
        assert result.quantity == Decimal("0.001")

    def test_a_genuinely_below_minimum_position_is_rejected_with_a_specific_reason(self) -> None:
        _, result = size(
            BTC_PERP,
            price="62965.60",
            equity=Decimal("100"),
            settings=risk_settings(max_position_pct=Decimal("0.05")),
        )
        assert result.quantity == ZERO
        assert not result.is_tradable
        assert result.capped_by == "below_venue_min_quantity"
        assert result.detail is not None
        assert "0.001" in result.detail, "the refusal must name the venue lot it could not afford"
        assert "max_position_pct" in result.detail

    async def test_the_registry_hands_the_perpetuals_rules_to_the_broker_not_an_options(
        self,
    ) -> None:
        """The exact defect: one symbol, four venue markets, only one of them tradable."""
        settings = ExchangeSettings(
            api_key="demo-key",
            api_secret="demo-secret",
            env=ExchangeEnv.DEMO,
            market_type=MarketType.FUTURE,
        )
        gateway = BybitGateway(settings)
        markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "spot": True,
                "swap": False,
                "future": False,
                "option": False,
                "linear": False,
                "expiry": None,
                "active": True,
                "precision": {"amount": 1e-06, "price": 0.1},
                "limits": {"amount": {"min": 1e-06}, "cost": {"min": 5}},
            },
            "BTC/USDT:USDT": BTC_PERP,
            "BTC/USDT:USDT-260821": dated_future_market(),
            "BTC/USDT:USDT-260821-52000-C": option_market(),
        }

        async def load_markets(*_: object, **__: object) -> dict[str, Any]:
            return markets

        gateway._data_client = type("Stub", (), {"load_markets": staticmethod(load_markets)})()
        loaded = await gateway.load_instruments()

        btc = loaded[Symbol.parse("BTC/USDT")]
        assert btc.min_quantity == Decimal("0.001"), "the perpetual's lot, not the option's 0.01"
        assert btc.quantity_step == Decimal("0.001")
        assert btc.price_tick == Decimal("0.1"), "0.00001 came from reading a 5.0 tick as places"
        assert parse_instrument(option_market()) is None
        assert parse_instrument(dated_future_market()) is None

        # And the quantity handed to the broker off that instrument is non-zero and legal.
        _, result = size(BTC_PERP, price="62965.60")
        assert_venue_legal(btc, result.quantity, Decimal("62965.60"))
