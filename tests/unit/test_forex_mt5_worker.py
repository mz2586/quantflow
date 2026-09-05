"""MT5 worker: capability reporting, payload mapping, staleness, reconciliation.

The real ``MetaTrader5`` package is never imported here — it ships win_amd64 wheels
only. Every payload is a fake shaped like the terminal's structs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.errors import ForexCapabilityError, StaleMarketDataError
from quantflow.forex.instruments import TradeMode
from quantflow.forex.mt5_worker import (
    REQUIRED_ENV_VARS,
    Capabilities,
    MT5Credentials,
    MT5Worker,
    account_from_mt5,
    bar_from_mt5,
    build_app,
    capabilities,
    fill_from_mt5,
    instrument_from_symbol_info,
    mt5_order_type,
    order_from_mt5,
    position_from_mt5,
    side_and_type_from_mt5,
    tick_from_mt5,
)
from quantflow.forex.protocol import (
    ForexBroker,
    ForexOrderStatus,
    ForexOrderType,
    ForexPosition,
    ForexTick,
    ForexTimeframe,
    ensure_fresh,
    reconcile_positions,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)

GOOD_ENV = {
    "QF_MT5_LOGIN": "12345678",
    "QF_MT5_PASSWORD": "s3cret",
    "QF_MT5_SERVER": "Bybit-Demo",
}


def fake_symbol_info(**overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "name": "EURUSD+",
        "currency_base": "EUR",
        "currency_profit": "USD",
        "currency_margin": "EUR",
        "trade_contract_size": 100000.0,
        "volume_min": 0.01,
        "volume_max": 50.0,
        "volume_step": 0.01,
        "digits": 5,
        "point": 1e-05,
        "trade_tick_size": 1e-05,
        "trade_tick_value": 1.0,
        "margin_initial": 0.0,
        "trade_mode": 4,
        "spread": 12,
        "swap_long": -2.5,
        "swap_short": 0.8,
        "swap_rollover3days": 3,
        "visible": True,
        "select": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class TestCapabilities:
    def test_this_host_is_blocked_by_platform(self) -> None:
        caps = capabilities(env=GOOD_ENV, platform_system="Darwin")
        assert not caps.ready
        blockers = " ".join(caps.blockers)
        assert "Windows" in blockers
        assert "Darwin" in blockers

    def test_real_host_platform_is_reported_and_blocks_here(self) -> None:
        # No platform_system override: reads the actual host. This machine is Darwin.
        caps = capabilities(env=GOOD_ENV)
        assert caps.platform != ""
        if caps.platform != "Windows":
            assert not caps.ready
            assert any("Windows" in blocker for blocker in caps.blockers)

    def test_missing_package_is_reported_with_an_install_command(self) -> None:
        caps = capabilities(
            env=GOOD_ENV,
            platform_system="Windows",
            package_probe=lambda: (False, None),
        )
        assert not caps.ready
        assert any("pip install MetaTrader5" in blocker for blocker in caps.blockers)

    def test_missing_credentials_are_named_individually(self) -> None:
        caps = capabilities(
            env={"QF_MT5_SERVER": "Bybit-Demo"},
            platform_system="Windows",
            package_probe=lambda: (True, "5.0.45"),
        )
        assert not caps.ready
        assert caps.missing_env == ("QF_MT5_LOGIN", "QF_MT5_PASSWORD")
        assert any("QF_MT5_LOGIN" in blocker for blocker in caps.blockers)

    def test_everything_present_is_ready(self) -> None:
        caps = capabilities(
            env=GOOD_ENV,
            platform_system="Windows",
            package_probe=lambda: (True, "5.0.45"),
        )
        assert caps.ready
        assert caps.blockers == ()
        assert caps.package_version == "5.0.45"

    def test_required_env_vars_are_the_documented_three(self) -> None:
        assert REQUIRED_ENV_VARS == ("QF_MT5_LOGIN", "QF_MT5_PASSWORD", "QF_MT5_SERVER")

    def test_describe_lists_every_blocker(self) -> None:
        caps = capabilities(env={}, platform_system="Darwin", package_probe=lambda: (False, None))
        described = caps.describe()
        assert "Darwin" in described
        assert "MetaTrader5" in described
        assert "QF_MT5_LOGIN" in described

    def test_raise_if_not_ready_is_actionable(self) -> None:
        caps = capabilities(env={}, platform_system="Darwin", package_probe=lambda: (False, None))
        with pytest.raises(ForexCapabilityError) as excinfo:
            caps.raise_if_not_ready()
        assert "Windows" in str(excinfo.value)

    def test_raise_if_not_ready_is_silent_when_ready(self) -> None:
        caps = Capabilities(
            platform="Windows",
            python_version="3.12.0",
            package_available=True,
            package_version="5.0.45",
            missing_env=(),
            blockers=(),
        )
        caps.raise_if_not_ready()


class TestCredentials:
    def test_from_env(self) -> None:
        creds = MT5Credentials.from_env(GOOD_ENV)
        assert creds.login == 12345678
        assert creds.server == "Bybit-Demo"

    def test_password_is_not_in_the_repr(self) -> None:
        creds = MT5Credentials.from_env(GOOD_ENV)
        assert "s3cret" not in repr(creds)

    def test_missing_env_raises_with_the_variable_name(self) -> None:
        with pytest.raises(ForexCapabilityError) as excinfo:
            MT5Credentials.from_env({"QF_MT5_LOGIN": "1"})
        assert "QF_MT5_PASSWORD" in str(excinfo.value)

    def test_non_numeric_login_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MT5Credentials.from_env({**GOOD_ENV, "QF_MT5_LOGIN": "abc"})

    def test_demo_server_detected(self) -> None:
        assert MT5Credentials.from_env(GOOD_ENV).is_demo_server

    def test_live_server_detected_and_not_allowed_by_default(self) -> None:
        creds = MT5Credentials.from_env({**GOOD_ENV, "QF_MT5_SERVER": "Bybit-Live"})
        assert not creds.is_demo_server
        assert not creds.allow_live

    def test_live_requires_an_explicit_opt_in(self) -> None:
        creds = MT5Credentials.from_env(
            {**GOOD_ENV, "QF_MT5_SERVER": "Bybit-Live", "QF_MT5_ALLOW_LIVE": "1"}
        )
        assert creds.allow_live


class TestInstrumentParsing:
    def test_maps_an_mt5_shaped_payload(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info())
        assert instrument.symbol == "EURUSD+"
        assert instrument.base == "EUR"
        assert instrument.quote == "USD"
        assert instrument.contract_size == Decimal("100000")
        assert instrument.min_lot == Decimal("0.01")
        assert instrument.lot_step == Decimal("0.01")
        assert instrument.digits == 5
        assert instrument.trade_mode is TradeMode.FULL
        assert instrument.tradable

    def test_floats_become_exact_decimals(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info())
        assert instrument.point == Decimal("0.00001")
        assert instrument.tick_value == Decimal("1")

    def test_spread_is_carried_in_points(self) -> None:
        assert instrument_from_symbol_info(fake_symbol_info()).spread_points == Decimal("12")

    def test_swaps_are_carried_with_sign(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info())
        assert instrument.swap_long == Decimal("-2.5")
        assert instrument.swap_short == Decimal("0.8")

    def test_mt5_rollover_day_is_converted_to_python_weekday(self) -> None:
        # MT5 swap_rollover3days=3 is Wednesday; Python weekday for Wednesday is 2.
        assert instrument_from_symbol_info(fake_symbol_info()).triple_swap_weekday == 2

    def test_disabled_trade_mode_is_not_tradable(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info(trade_mode=0))
        assert instrument.trade_mode is TradeMode.DISABLED
        assert not instrument.can_trade(OrderSide.BUY)

    def test_invisible_symbol_is_not_tradable(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info(visible=False))
        assert not instrument.tradable

    def test_margin_initial_becomes_a_margin_rate_when_expressed_as_a_fraction(self) -> None:
        instrument = instrument_from_symbol_info(fake_symbol_info(margin_initial=0.02))
        assert instrument.margin_rate == Decimal("0.02")
        assert instrument.leverage == Decimal("50")

    def test_jpy_payload_keeps_three_digits(self) -> None:
        instrument = instrument_from_symbol_info(
            fake_symbol_info(
                name="USDJPY+",
                currency_base="USD",
                currency_profit="JPY",
                digits=3,
                point=0.001,
                trade_tick_size=0.001,
                trade_tick_value=0.65,
            )
        )
        assert instrument.digits == 3
        assert instrument.is_jpy_quoted
        assert instrument.pip_size == Decimal("0.01")

    def test_missing_field_raises_a_clear_error(self) -> None:
        payload = fake_symbol_info()
        del payload.volume_min
        with pytest.raises(ValidationError) as excinfo:
            instrument_from_symbol_info(payload)
        assert "volume_min" in str(excinfo.value)

    def test_mapping_payloads_are_accepted_too(self) -> None:
        payload = vars(fake_symbol_info())
        assert instrument_from_symbol_info(payload).symbol == "EURUSD+"


class TestTickAndBarParsing:
    def test_tick_mapping(self) -> None:
        raw = SimpleNamespace(bid=1.10001, ask=1.10013, last=0.0, volume=3, time=1786636800)
        tick = tick_from_mt5("EURUSD+", raw)
        assert tick.bid == Decimal("1.10001")
        assert tick.ask == Decimal("1.10013")
        assert tick.timestamp.tzinfo is UTC

    def test_tick_mid_and_spread(self) -> None:
        tick = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.10000"),
            ask=Decimal("1.10012"),
            timestamp=NOW,
        )
        assert tick.mid == Decimal("1.10006")
        assert tick.spread == Decimal("0.00012")

    def test_crossed_tick_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForexTick(
                symbol="EURUSD+",
                bid=Decimal("1.10020"),
                ask=Decimal("1.10000"),
                timestamp=NOW,
            )

    def test_bar_mapping(self) -> None:
        row = {
            "time": 1786636800,
            "open": 1.1,
            "high": 1.105,
            "low": 1.098,
            "close": 1.102,
            "tick_volume": 812,
            "spread": 12,
            "real_volume": 0,
        }
        bar = bar_from_mt5("EURUSD+", ForexTimeframe.M15, row)
        assert bar.close == Decimal("1.102")
        assert bar.tick_volume == 812
        assert bar.open_time.tzinfo is UTC

    def test_bar_rejects_inconsistent_high_low(self) -> None:
        row = {
            "time": 1786636800,
            "open": 1.1,
            "high": 1.05,
            "low": 1.098,
            "close": 1.102,
            "tick_volume": 1,
            "spread": 1,
            "real_volume": 0,
        }
        with pytest.raises(ValidationError):
            bar_from_mt5("EURUSD+", ForexTimeframe.M15, row)

    def test_timeframe_maps_to_the_mt5_constant_name(self) -> None:
        assert ForexTimeframe.M15.mt5_constant == "TIMEFRAME_M15"
        assert ForexTimeframe.H4.mt5_constant == "TIMEFRAME_H4"


class TestStaleData:
    def test_fresh_tick_passes(self) -> None:
        tick = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.1"),
            ask=Decimal("1.10012"),
            timestamp=NOW - timedelta(seconds=2),
        )
        ensure_fresh(tick, now=NOW, max_age=timedelta(seconds=10))

    def test_stale_tick_raises_with_the_age(self) -> None:
        tick = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.1"),
            ask=Decimal("1.10012"),
            timestamp=NOW - timedelta(minutes=5),
        )
        with pytest.raises(StaleMarketDataError) as excinfo:
            ensure_fresh(tick, now=NOW, max_age=timedelta(seconds=10))
        assert "EURUSD+" in str(excinfo.value)

    def test_a_tick_from_the_future_is_also_rejected(self) -> None:
        tick = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.1"),
            ask=Decimal("1.10012"),
            timestamp=NOW + timedelta(minutes=5),
        )
        with pytest.raises(StaleMarketDataError):
            ensure_fresh(tick, now=NOW, max_age=timedelta(seconds=10))

    def test_age_property(self) -> None:
        tick = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.1"),
            ask=Decimal("1.10012"),
            timestamp=NOW - timedelta(seconds=30),
        )
        assert tick.age(NOW) == timedelta(seconds=30)


class TestOrderTypeMapping:
    @pytest.mark.parametrize(
        ("side", "order_type", "code"),
        [
            (OrderSide.BUY, ForexOrderType.MARKET, 0),
            (OrderSide.SELL, ForexOrderType.MARKET, 1),
            (OrderSide.BUY, ForexOrderType.LIMIT, 2),
            (OrderSide.SELL, ForexOrderType.LIMIT, 3),
            (OrderSide.BUY, ForexOrderType.STOP, 4),
            (OrderSide.SELL, ForexOrderType.STOP, 5),
            (OrderSide.BUY, ForexOrderType.STOP_LIMIT, 6),
            (OrderSide.SELL, ForexOrderType.STOP_LIMIT, 7),
        ],
    )
    def test_round_trip(self, side: OrderSide, order_type: ForexOrderType, code: int) -> None:
        assert mt5_order_type(side, order_type) == code
        assert side_and_type_from_mt5(code) == (side, order_type)

    def test_unknown_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            side_and_type_from_mt5(42)


class TestPositionOrderFillMapping:
    def test_position_mapping(self) -> None:
        raw = SimpleNamespace(
            ticket=6001,
            symbol="EURUSD+",
            type=1,
            volume=0.5,
            price_open=1.10000,
            price_current=1.09900,
            sl=1.10200,
            tp=1.09500,
            swap=-2.5,
            profit=50.0,
            time=1786636800,
            magic=770,
            comment="qf",
        )
        position = position_from_mt5(raw)
        assert position.ticket == 6001
        assert position.side is OrderSide.SELL
        assert position.lots == Decimal("0.5")
        assert position.stop_loss == Decimal("1.10200")
        assert position.take_profit == Decimal("1.09500")

    def test_zero_stop_becomes_none(self) -> None:
        raw = SimpleNamespace(
            ticket=1,
            symbol="EURUSD+",
            type=0,
            volume=0.1,
            price_open=1.1,
            price_current=1.1,
            sl=0.0,
            tp=0.0,
            swap=0.0,
            profit=0.0,
            time=1786636800,
            magic=0,
            comment="",
        )
        position = position_from_mt5(raw)
        assert position.stop_loss is None
        assert position.take_profit is None

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (1, ForexOrderStatus.PLACED),
            (2, ForexOrderStatus.CANCELLED),
            (3, ForexOrderStatus.PARTIALLY_FILLED),
            (4, ForexOrderStatus.FILLED),
            (5, ForexOrderStatus.REJECTED),
            (6, ForexOrderStatus.EXPIRED),
        ],
    )
    def test_order_state_mapping(self, state: int, expected: ForexOrderStatus) -> None:
        raw = SimpleNamespace(
            ticket=9,
            symbol="EURUSD+",
            type=2,
            state=state,
            volume_initial=1.0,
            volume_current=0.4,
            price_open=1.09,
            sl=0.0,
            tp=0.0,
            time_setup=1786636800,
            magic=0,
            comment="",
        )
        order = order_from_mt5(raw)
        assert order.status is expected
        assert order.order_type is ForexOrderType.LIMIT
        assert order.side is OrderSide.BUY

    def test_fill_mapping(self) -> None:
        raw = SimpleNamespace(
            ticket=42,
            order=41,
            symbol="EURUSD+",
            type=0,
            entry=0,
            volume=0.2,
            price=1.10005,
            commission=-1.2,
            swap=0.0,
            profit=0.0,
            time=1786636800,
            magic=770,
            comment="",
        )
        fill = fill_from_mt5(raw)
        assert fill.ticket == 42
        assert fill.order_ticket == 41
        assert fill.side is OrderSide.BUY
        assert fill.lots == Decimal("0.2")
        assert fill.commission == Decimal("-1.2")
        assert fill.is_entry

    def test_account_mapping(self) -> None:
        raw = SimpleNamespace(
            login=12345678,
            server="Bybit-Demo",
            currency="USD",
            balance=10000.0,
            equity=10120.5,
            margin=220.0,
            margin_free=9900.5,
            margin_level=4600.0,
            leverage=100,
            trade_allowed=True,
            trade_mode=0,
            name="demo",
        )
        account = account_from_mt5(raw)
        assert account.login == 12345678
        assert account.currency == "USD"
        assert account.equity == Decimal("10120.5")
        assert account.is_demo
        assert account.trade_allowed

    def test_real_account_flagged_as_not_demo(self) -> None:
        raw = SimpleNamespace(
            login=1,
            server="Bybit-Live",
            currency="USD",
            balance=1.0,
            equity=1.0,
            margin=0.0,
            margin_free=1.0,
            margin_level=0.0,
            leverage=100,
            trade_allowed=True,
            trade_mode=2,
            name="real",
        )
        assert not account_from_mt5(raw).is_demo


class TestReconciliation:
    def position(self, ticket: int, lots: str, side: OrderSide = OrderSide.BUY) -> ForexPosition:
        return ForexPosition(
            ticket=ticket,
            symbol="EURUSD+",
            side=side,
            lots=Decimal(lots),
            entry_price=Decimal("1.1"),
            current_price=Decimal("1.1"),
            opened_at=NOW,
        )

    def test_clean_when_identical(self) -> None:
        expected = [self.position(1, "0.5")]
        report = reconcile_positions(expected, list(expected), now=NOW)
        assert report.is_clean
        assert report.matched == (1,)

    def test_detects_a_position_only_at_the_broker(self) -> None:
        report = reconcile_positions([], [self.position(1, "0.5")], now=NOW)
        assert not report.is_clean
        assert report.only_at_broker == (1,)

    def test_detects_a_position_only_known_locally(self) -> None:
        report = reconcile_positions([self.position(1, "0.5")], [], now=NOW)
        assert not report.is_clean
        assert report.only_local == (1,)

    def test_detects_a_lot_mismatch(self) -> None:
        report = reconcile_positions([self.position(1, "0.5")], [self.position(1, "0.3")], now=NOW)
        assert not report.is_clean
        assert len(report.lot_mismatches) == 1
        delta = report.lot_mismatches[0]
        assert delta.expected_lots == Decimal("0.5")
        assert delta.actual_lots == Decimal("0.3")

    def test_detects_a_side_mismatch(self) -> None:
        report = reconcile_positions(
            [self.position(1, "0.5", OrderSide.BUY)],
            [self.position(1, "0.5", OrderSide.SELL)],
            now=NOW,
        )
        assert not report.is_clean
        assert report.side_mismatches == (1,)

    def test_report_records_when_it_ran(self) -> None:
        assert reconcile_positions([], [], now=NOW).checked_at == NOW

    def test_empty_on_both_sides_is_clean(self) -> None:
        assert reconcile_positions([], [], now=NOW).is_clean


class TestWorkerRefusesToRunHere:
    def test_construction_is_allowed_without_a_terminal(self) -> None:
        MT5Worker(credentials=MT5Credentials.from_env(GOOD_ENV))

    def test_connect_raises_a_capability_error_on_this_platform(self) -> None:
        worker = MT5Worker(credentials=MT5Credentials.from_env(GOOD_ENV))
        with pytest.raises(ForexCapabilityError) as excinfo:
            worker.connect()
        message = str(excinfo.value)
        assert "MetaTrader5" in message or "Windows" in message

    def test_reading_the_account_without_a_connection_raises(self) -> None:
        worker = MT5Worker(credentials=MT5Credentials.from_env(GOOD_ENV))
        with pytest.raises(ForexCapabilityError):
            worker.get_account()

    def test_worker_structurally_satisfies_the_broker_protocol(self) -> None:
        broker: ForexBroker = MT5Worker(credentials=MT5Credentials.from_env(GOOD_ENV))
        assert broker is not None

    def test_live_server_is_refused_before_any_connection_attempt(self) -> None:
        creds = MT5Credentials.from_env({**GOOD_ENV, "QF_MT5_SERVER": "Bybit-Real"})
        with pytest.raises(ForexCapabilityError) as excinfo:
            MT5Worker(credentials=creds)
        assert "demo" in str(excinfo.value).lower()


class TestServiceBoundary:
    """The HTTP boundary must stay diagnosable on a host that cannot run the terminal."""

    async def request(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=build_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            return await client.get(path)

    async def test_capabilities_endpoint_works_without_a_terminal(self) -> None:
        response = await self.request("/capabilities")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["blockers"]

    async def test_health_endpoint_works_without_a_terminal(self) -> None:
        response = await self.request("/health")
        assert response.status_code == 200
        assert response.json()["worker_ready"] is False

    async def test_trading_endpoints_report_the_blocker_rather_than_crashing(self) -> None:
        response = await self.request("/account")
        assert response.status_code == 503
        assert "blockers" in response.json()["detail"]
