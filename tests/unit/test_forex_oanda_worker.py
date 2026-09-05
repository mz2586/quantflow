"""OANDA v20 transport — the Linux-native route.

No live endpoint is ever contacted. Every response is a fixture built from OANDA's
published v20 documentation and served through a fake HTTP transport.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.errors import (
    ForexAuthenticationError,
    ForexCapabilityError,
    ForexConnectionError,
    ForexOrderRejectedError,
    ForexSymbolError,
)
from quantflow.forex.instruments import ForexInstrument
from quantflow.forex.oanda_worker import (
    OANDA_LIVE_REST,
    OANDA_PRACTICE_REST,
    OANDA_PRACTICE_STREAM,
    OANDA_UNITS_PER_LOT,
    HttpTransport,
    OandaCredentials,
    OandaEnvironment,
    OandaWorker,
    account_from_oanda,
    bar_from_oanda,
    capabilities,
    fill_from_oanda,
    instrument_from_oanda,
    lots_from_units,
    order_from_oanda,
    parse_oanda_time,
    position_from_oanda_trade,
    raise_for_oanda_error,
    side_from_units,
    tick_from_oanda,
    units_from_lots,
)
from quantflow.forex.protocol import ForexBroker, ForexOrderRequest, ForexOrderType, ForexTimeframe

ACCOUNT_ID = "001-001-1234567-001"
GOOD_ENV = {
    "QF_OANDA_TOKEN": "abc-token-123",
    "QF_OANDA_ACCOUNT_ID": ACCOUNT_ID,
}

EUR_USD_PAYLOAD: dict[str, Any] = {
    "name": "EUR_USD",
    "type": "CURRENCY",
    "displayName": "EUR/USD",
    "pipLocation": -4,
    "displayPrecision": 5,
    "tradeUnitsPrecision": 0,
    "minimumTradeSize": "1",
    "maximumOrderUnits": "100000000",
    "maximumPositionSize": "0",
    "marginRate": "0.02",
}

USD_JPY_PAYLOAD: dict[str, Any] = {
    **EUR_USD_PAYLOAD,
    "name": "USD_JPY",
    "displayName": "USD/JPY",
    "pipLocation": -2,
    "displayPrecision": 3,
}


class FakeTransport:
    """Serves canned responses keyed by (method, path). Records every call."""

    def __init__(
        self,
        responses: Mapping[tuple[str, str], tuple[int, dict[str, Any]]] | None = None,
        lines: tuple[str, ...] = (),
    ) -> None:
        self.responses: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = dict(responses or {})
        self.lines = lines
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, Any]]:
        self.calls.append((method, url, dict(params or {}), json_body))
        key = (method, urlsplit(url).path)
        if key not in self.responses:
            return 404, {"errorMessage": f"no fixture for {key}"}
        return self.responses[key]

    def stream_lines(self, url: str, *, params: Mapping[str, str] | None = None) -> Iterator[str]:
        self.calls.append(("STREAM", url, dict(params or {}), None))
        yield from self.lines

    def close(self) -> None:
        self.closed = True

    def body_for(self, method: str, path: str) -> Any:
        for call_method, url, _params, body in self.calls:
            if call_method == method and urlsplit(url).path == path:
                return body
        raise AssertionError(f"no {method} {path} call was made")


def instruments_response(*payloads: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return 200, {"instruments": list(payloads), "lastTransactionID": "42"}


def make_worker(
    responses: Mapping[tuple[str, str], tuple[int, dict[str, Any]]] | None = None,
    lines: tuple[str, ...] = (),
) -> tuple[OandaWorker, FakeTransport]:
    base = {
        ("GET", f"/v3/accounts/{ACCOUNT_ID}/instruments"): instruments_response(
            EUR_USD_PAYLOAD, USD_JPY_PAYLOAD
        )
    }
    base.update(responses or {})
    transport = FakeTransport(base, lines)
    worker = OandaWorker(OandaCredentials.from_env(GOOD_ENV), transport=transport)
    return worker, transport


class TestCapabilities:
    def test_no_platform_blocker_this_is_the_linux_route(self) -> None:
        caps = capabilities(GOOD_ENV)
        assert caps.ready
        assert caps.linux_compatible
        assert caps.blockers == ()

    def test_missing_credentials_are_named(self) -> None:
        caps = capabilities({"QF_OANDA_TOKEN": "x"})
        assert not caps.ready
        assert caps.missing_env == ("QF_OANDA_ACCOUNT_ID",)
        assert any("QF_OANDA_ACCOUNT_ID" in blocker for blocker in caps.blockers)

    def test_blocker_explains_how_to_get_a_token(self) -> None:
        caps = capabilities({})
        assert any("Manage API Access" in blocker for blocker in caps.blockers)

    def test_live_environment_is_blocked_by_default(self) -> None:
        caps = capabilities({**GOOD_ENV, "QF_OANDA_ENVIRONMENT": "live"})
        assert not caps.ready
        assert any("real money" in blocker for blocker in caps.blockers)

    def test_live_can_be_opted_into(self) -> None:
        caps = capabilities(
            {**GOOD_ENV, "QF_OANDA_ENVIRONMENT": "live", "QF_OANDA_ALLOW_LIVE": "1"}
        )
        assert caps.ready

    def test_division_caveat_is_surfaced(self) -> None:
        assert any("division" in note for note in capabilities(GOOD_ENV).notes)

    def test_describe_mentions_every_blocker(self) -> None:
        assert "QF_OANDA_TOKEN" in capabilities({}).describe()

    def test_raise_if_not_ready(self) -> None:
        with pytest.raises(ForexCapabilityError):
            capabilities({}).raise_if_not_ready()


class TestCredentialsAndHosts:
    def test_from_env_defaults_to_practice(self) -> None:
        credentials = OandaCredentials.from_env(GOOD_ENV)
        assert credentials.is_practice
        assert credentials.environment.rest_host == OANDA_PRACTICE_REST
        assert credentials.environment.stream_host == OANDA_PRACTICE_STREAM

    def test_token_is_not_in_the_repr(self) -> None:
        assert "abc-token-123" not in repr(OandaCredentials.from_env(GOOD_ENV))

    def test_missing_env_raises_naming_the_variable(self) -> None:
        with pytest.raises(ForexCapabilityError) as excinfo:
            OandaCredentials.from_env({"QF_OANDA_TOKEN": "x"})
        assert "QF_OANDA_ACCOUNT_ID" in str(excinfo.value)

    def test_live_host_differs(self) -> None:
        assert OandaEnvironment.LIVE.rest_host == OANDA_LIVE_REST
        assert OandaEnvironment.PRACTICE.rest_host != OandaEnvironment.LIVE.rest_host

    def test_worker_refuses_live_without_an_explicit_override(self) -> None:
        credentials = OandaCredentials.from_env({**GOOD_ENV, "QF_OANDA_ENVIRONMENT": "live"})
        with pytest.raises(ForexCapabilityError) as excinfo:
            OandaWorker(credentials, transport=FakeTransport())
        assert "real money" in str(excinfo.value)

    def test_worker_allows_live_when_opted_in(self) -> None:
        credentials = OandaCredentials.from_env(
            {**GOOD_ENV, "QF_OANDA_ENVIRONMENT": "live", "QF_OANDA_ALLOW_LIVE": "1"}
        )
        OandaWorker(credentials, transport=FakeTransport())


class TestUnitsAndLots:
    def test_one_lot_is_one_hundred_thousand_units(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        assert units_from_lots(Decimal("1"), OrderSide.BUY, instrument) == OANDA_UNITS_PER_LOT

    def test_short_units_are_negative(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        assert units_from_lots(Decimal("0.5"), OrderSide.SELL, instrument) == Decimal("-50000")

    def test_round_trip(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        units = units_from_lots(Decimal("0.37"), OrderSide.SELL, instrument)
        assert lots_from_units(units, instrument) == Decimal("0.37")

    def test_side_reads_the_sign(self) -> None:
        assert side_from_units(Decimal("5000")) is OrderSide.BUY
        assert side_from_units(Decimal("-5000")) is OrderSide.SELL

    def test_zero_units_has_no_direction(self) -> None:
        with pytest.raises(ValidationError):
            side_from_units(Decimal("0"))

    def test_non_positive_lots_rejected(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        with pytest.raises(ValidationError):
            units_from_lots(Decimal("0"), OrderSide.BUY, instrument)


class TestInstrumentParsing:
    def test_eur_usd(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        assert instrument.symbol == "EUR_USD"
        assert instrument.base == "EUR"
        assert instrument.quote == "USD"
        assert instrument.contract_size == OANDA_UNITS_PER_LOT
        assert instrument.digits == 5
        assert instrument.pip_size == Decimal("0.0001")
        assert instrument.venue == "oanda"

    def test_minimum_trade_size_of_one_unit_becomes_a_micro_lot_floor(self) -> None:
        instrument = instrument_from_oanda(EUR_USD_PAYLOAD)
        assert instrument.min_lot == Decimal("0.00001")
        assert instrument.lot_step == Decimal("0.00001")

    def test_maximum_order_units_becomes_max_lot(self) -> None:
        assert instrument_from_oanda(EUR_USD_PAYLOAD).max_lot == Decimal("1000")

    def test_margin_rate_gives_leverage(self) -> None:
        assert instrument_from_oanda(EUR_USD_PAYLOAD).leverage == Decimal("50")

    def test_tick_value_for_a_usd_quoted_pair_on_a_usd_account(self) -> None:
        assert instrument_from_oanda(EUR_USD_PAYLOAD).value_per_point_per_lot == Decimal("1")

    def test_jpy_pair_keeps_three_digits(self) -> None:
        instrument = instrument_from_oanda(USD_JPY_PAYLOAD)
        assert instrument.digits == 3
        assert instrument.is_jpy_quoted
        assert instrument.pip_size == Decimal("0.01")

    def test_home_conversion_factor_converts_the_tick_value(self) -> None:
        instrument = instrument_from_oanda(
            USD_JPY_PAYLOAD, home_conversion_factor=Decimal("0.0065")
        )
        assert instrument.value_per_point_per_lot == Decimal("0.65")

    def test_pip_and_price_grid_disagreement_is_refused(self) -> None:
        payload = {**EUR_USD_PAYLOAD, "pipLocation": -5}
        with pytest.raises(ForexSymbolError) as excinfo:
            instrument_from_oanda(payload)
        assert "pip" in str(excinfo.value).lower()

    def test_unexpected_symbol_shape_is_refused(self) -> None:
        with pytest.raises(ForexSymbolError):
            instrument_from_oanda({**EUR_USD_PAYLOAD, "name": "EURUSD"})

    def test_missing_field_names_itself(self) -> None:
        payload = {key: value for key, value in EUR_USD_PAYLOAD.items() if key != "pipLocation"}
        with pytest.raises(ValidationError) as excinfo:
            instrument_from_oanda(payload)
        assert "pipLocation" in str(excinfo.value)

    def test_commission_block_is_scaled_to_a_lot(self) -> None:
        payload = {
            **EUR_USD_PAYLOAD,
            "commission": {"commission": "0.5", "unitsTraded": "10000", "minimumCommission": "0"},
        }
        assert instrument_from_oanda(payload).commission_per_lot == Decimal("5")

    def test_absent_commission_stays_zero_rather_than_invented(self) -> None:
        assert instrument_from_oanda(EUR_USD_PAYLOAD).commission_per_lot == Decimal("0")


class TestTimeParsing:
    def test_nanosecond_precision_is_truncated_not_rejected(self) -> None:
        parsed = parse_oanda_time("2026-08-12T13:45:30.123456789Z")
        assert parsed == datetime(2026, 8, 12, 13, 45, 30, 123456, tzinfo=UTC)

    def test_plain_second_precision(self) -> None:
        assert parse_oanda_time("2026-08-12T13:45:30Z").tzinfo is UTC

    def test_offset_timestamps_normalise_to_utc(self) -> None:
        assert parse_oanda_time("2026-08-12T15:45:30.000000000+02:00") == datetime(
            2026, 8, 12, 13, 45, 30, tzinfo=UTC
        )

    def test_garbage_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_oanda_time("not-a-time")


class TestPriceAndCandleParsing:
    def test_client_price(self) -> None:
        tick = tick_from_oanda(
            {
                "type": "PRICE",
                "instrument": "EUR_USD",
                "time": "2026-08-12T13:45:30.123456789Z",
                "tradeable": True,
                "bids": [{"price": "1.09990", "liquidity": 10000000}],
                "asks": [{"price": "1.10002", "liquidity": 10000000}],
                "closeoutBid": "1.09985",
                "closeoutAsk": "1.10007",
            }
        )
        assert tick.bid == Decimal("1.09990")
        assert tick.ask == Decimal("1.10002")
        assert tick.spread == Decimal("0.00012")

    def test_empty_ladders_fall_back_to_closeout_prices(self) -> None:
        tick = tick_from_oanda(
            {
                "instrument": "EUR_USD",
                "time": "2026-08-12T13:45:30Z",
                "bids": [],
                "asks": [],
                "closeoutBid": "1.09985",
                "closeoutAsk": "1.10007",
            }
        )
        assert tick.bid == Decimal("1.09985")

    def test_candle(self) -> None:
        bar = bar_from_oanda(
            "EUR_USD",
            ForexTimeframe.M15,
            {
                "complete": True,
                "volume": 812,
                "time": "2026-08-12T13:45:00.000000000Z",
                "mid": {"o": "1.10000", "h": "1.10120", "l": "1.09950", "c": "1.10080"},
            },
        )
        assert bar.close == Decimal("1.10080")
        assert bar.tick_volume == 812
        assert bar.timeframe is ForexTimeframe.M15

    def test_candle_falls_back_to_the_bid_ladder(self) -> None:
        bar = bar_from_oanda(
            "EUR_USD",
            ForexTimeframe.H1,
            {
                "complete": True,
                "volume": 5,
                "time": "2026-08-12T13:00:00Z",
                "bid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.15"},
            },
        )
        assert bar.high == Decimal("1.2")

    def test_candle_without_any_ladder_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            bar_from_oanda(
                "EUR_USD",
                ForexTimeframe.H1,
                {"complete": True, "volume": 5, "time": "2026-08-12T13:00:00Z"},
            )

    def test_granularity_codes(self) -> None:
        assert ForexTimeframe.M15.oanda_granularity == "M15"
        assert ForexTimeframe.D1.oanda_granularity == "D"
        assert ForexTimeframe.MN1.oanda_granularity == "M"


class TestAccountTradeAndFillParsing:
    def test_account_summary(self) -> None:
        account = account_from_oanda(
            {
                "id": ACCOUNT_ID,
                "alias": "Practice",
                "currency": "USD",
                "balance": "100000.0000",
                "NAV": "100120.5000",
                "marginUsed": "220.0000",
                "marginAvailable": "99900.5000",
                "marginCloseoutPercent": "0.00110",
                "marginRate": "0.02",
                "openTradeCount": 1,
                "lastTransactionID": "77",
            },
            is_practice=True,
        )
        assert account.currency == "USD"
        assert account.equity == Decimal("100120.5000")
        assert account.leverage == 50
        assert account.is_demo
        assert account.server == ACCOUNT_ID

    def test_live_account_is_not_flagged_demo(self) -> None:
        account = account_from_oanda(
            {"id": ACCOUNT_ID, "currency": "USD", "balance": "1", "NAV": "1"},
            is_practice=False,
        )
        assert not account.is_demo

    def test_open_trade_becomes_a_position(self) -> None:
        position = position_from_oanda_trade(
            {
                "id": "6001",
                "instrument": "EUR_USD",
                "price": "1.10000",
                "openTime": "2026-08-12T10:00:00.000000000Z",
                "initialUnits": "-50000",
                "currentUnits": "-50000",
                "realizedPL": "0.0000",
                "unrealizedPL": "12.5000",
                "financing": "-1.2000",
                "state": "OPEN",
                "stopLossOrder": {"id": "6002", "price": "1.10200"},
                "takeProfitOrder": {"id": "6003", "price": "1.09500"},
            },
            OANDA_UNITS_PER_LOT,
        )
        assert position.ticket == 6001
        assert position.side is OrderSide.SELL
        assert position.lots == Decimal("0.5")
        assert position.stop_loss == Decimal("1.10200")
        assert position.take_profit == Decimal("1.09500")
        assert position.swap == Decimal("-1.2000")

    def test_trade_without_protective_orders(self) -> None:
        position = position_from_oanda_trade(
            {
                "id": "1",
                "instrument": "EUR_USD",
                "price": "1.1",
                "openTime": "2026-08-12T10:00:00Z",
                "currentUnits": "10000",
            },
            OANDA_UNITS_PER_LOT,
        )
        assert position.stop_loss is None
        assert position.side is OrderSide.BUY
        assert position.lots == Decimal("0.1")

    def test_pending_limit_order(self) -> None:
        order = order_from_oanda(
            {
                "id": "7001",
                "type": "LIMIT",
                "instrument": "EUR_USD",
                "units": "25000",
                "price": "1.09500",
                "state": "PENDING",
                "createTime": "2026-08-12T09:00:00.000000000Z",
            },
            OANDA_UNITS_PER_LOT,
        )
        assert order.order_type is ForexOrderType.LIMIT
        assert order.side is OrderSide.BUY
        assert order.lots == Decimal("0.25")
        assert order.price == Decimal("1.09500")

    def test_unsupported_order_type_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            order_from_oanda(
                {"id": "1", "type": "FIXED_PRICE", "instrument": "EUR_USD", "units": "1"},
                OANDA_UNITS_PER_LOT,
            )

    def test_order_fill_transaction(self) -> None:
        fill = fill_from_oanda(
            {
                "id": "8001",
                "type": "ORDER_FILL",
                "orderID": "8000",
                "instrument": "EUR_USD",
                "units": "-50000",
                "price": "1.10005",
                "time": "2026-08-12T10:00:00.000000000Z",
                "commission": "0.0000",
                "financing": "0.0000",
                "pl": "0.0000",
            },
            OANDA_UNITS_PER_LOT,
        )
        assert fill.ticket == 8001
        assert fill.order_ticket == 8000
        assert fill.side is OrderSide.SELL
        assert fill.lots == Decimal("0.5")
        assert fill.is_entry

    def test_closing_fill_is_not_an_entry(self) -> None:
        fill = fill_from_oanda(
            {
                "id": "8002",
                "orderID": "8000",
                "instrument": "EUR_USD",
                "units": "50000",
                "price": "1.10105",
                "time": "2026-08-12T11:00:00Z",
                "pl": "50.0000",
            },
            OANDA_UNITS_PER_LOT,
        )
        assert not fill.is_entry
        assert fill.profit == Decimal("50.0000")


class TestErrorTranslation:
    def test_success_is_silent(self) -> None:
        raise_for_oanda_error(200, {})

    def test_unauthorised(self) -> None:
        with pytest.raises(ForexAuthenticationError) as excinfo:
            raise_for_oanda_error(401, {"errorMessage": "Insufficient authorization"})
        assert "token" in str(excinfo.value)

    def test_forbidden_is_also_an_auth_error(self) -> None:
        with pytest.raises(ForexAuthenticationError):
            raise_for_oanda_error(403, {"errorMessage": "not authorized"})

    def test_rate_limit_mentions_the_documented_ceiling(self) -> None:
        with pytest.raises(ForexConnectionError) as excinfo:
            raise_for_oanda_error(429, {"errorMessage": "too many"})
        assert "120" in str(excinfo.value)

    def test_server_error(self) -> None:
        with pytest.raises(ForexConnectionError):
            raise_for_oanda_error(503, {"errorMessage": "unavailable"})

    def test_reject_reason_is_surfaced(self) -> None:
        with pytest.raises(ForexOrderRejectedError) as excinfo:
            raise_for_oanda_error(
                400,
                {
                    "errorMessage": "The Order was rejected",
                    "orderRejectTransaction": {
                        "type": "MARKET_ORDER_REJECT",
                        "rejectReason": "STOP_LOSS_ON_FILL_PRICE_PRECISION_EXCEEDED",
                    },
                },
            )
        assert "PRECISION_EXCEEDED" in str(excinfo.value)

    def test_body_without_an_error_code_still_works(self) -> None:
        with pytest.raises(ForexOrderRejectedError):
            raise_for_oanda_error(404, {"errorMessage": "no such trade"})


class TestWorkerReads:
    def test_get_account_hits_the_summary_endpoint(self) -> None:
        worker, transport = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/summary"): (
                    200,
                    {
                        "account": {
                            "id": ACCOUNT_ID,
                            "currency": "USD",
                            "balance": "100000",
                            "NAV": "100000",
                            "marginRate": "0.02",
                        },
                        "lastTransactionID": "1",
                    },
                )
            }
        )
        account = worker.get_account()
        assert account.balance == Decimal("100000")
        assert transport.calls[0][1].startswith(OANDA_PRACTICE_REST)

    def test_get_symbols_prioritises_majors_without_inventing_any(self) -> None:
        worker, _ = make_worker()
        instruments = worker.get_symbols()
        assert [i.symbol for i in instruments] == ["EUR_USD", "USD_JPY"]

    def test_get_symbols_skips_an_instrument_with_inconsistent_pip_metadata(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/instruments"): instruments_response(
                    EUR_USD_PAYLOAD, {**USD_JPY_PAYLOAD, "pipLocation": -5}
                )
            }
        )
        assert [i.symbol for i in worker.get_symbols()] == ["EUR_USD"]

    def test_get_bars_requests_the_right_granularity_and_drops_incomplete_candles(self) -> None:
        worker, transport = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/instruments/EUR_USD/candles"): (
                    200,
                    {
                        "instrument": "EUR_USD",
                        "granularity": "M15",
                        "candles": [
                            {
                                "complete": True,
                                "volume": 10,
                                "time": "2026-08-12T13:00:00Z",
                                "mid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.15"},
                            },
                            {
                                "complete": False,
                                "volume": 3,
                                "time": "2026-08-12T13:15:00Z",
                                "mid": {"o": "1.15", "h": "1.16", "l": "1.14", "c": "1.155"},
                            },
                        ],
                    },
                )
            }
        )
        bars = worker.get_bars("EUR_USD", ForexTimeframe.M15, 2)
        assert len(bars) == 1
        assert transport.calls[0][2]["granularity"] == "M15"

    def test_get_bars_rejects_an_out_of_range_count(self) -> None:
        worker, _ = make_worker()
        with pytest.raises(ValidationError):
            worker.get_bars("EUR_USD", ForexTimeframe.M15, 99999)

    def test_get_positions_maps_open_trades(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/openTrades"): (
                    200,
                    {
                        "trades": [
                            {
                                "id": "6001",
                                "instrument": "EUR_USD",
                                "price": "1.1",
                                "openTime": "2026-08-12T10:00:00Z",
                                "currentUnits": "50000",
                            }
                        ],
                        "lastTransactionID": "9",
                    },
                )
            }
        )
        positions = worker.get_positions()
        assert len(positions) == 1
        assert positions[0].lots == Decimal("0.5")

    def test_get_positions_filters_by_symbol(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/openTrades"): (
                    200,
                    {
                        "trades": [
                            {
                                "id": "1",
                                "instrument": "EUR_USD",
                                "price": "1.1",
                                "openTime": "2026-08-12T10:00:00Z",
                                "currentUnits": "1000",
                            }
                        ]
                    },
                )
            }
        )
        assert worker.get_positions("USD_JPY") == ()

    def test_get_orders_excludes_protective_attachments(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/pendingOrders"): (
                    200,
                    {
                        "orders": [
                            {
                                "id": "7001",
                                "type": "LIMIT",
                                "instrument": "EUR_USD",
                                "units": "25000",
                                "price": "1.095",
                                "state": "PENDING",
                            },
                            {
                                "id": "7002",
                                "type": "STOP_LOSS",
                                "tradeID": "6001",
                                "price": "1.09",
                                "state": "PENDING",
                            },
                        ]
                    },
                )
            }
        )
        orders = worker.get_orders()
        assert len(orders) == 1
        assert orders[0].ticket == 7001

    def test_get_fills_since_id_returns_the_new_watermark(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/transactions/sinceid"): (
                    200,
                    {
                        "transactions": [
                            {
                                "id": "8001",
                                "type": "ORDER_FILL",
                                "orderID": "8000",
                                "instrument": "EUR_USD",
                                "units": "50000",
                                "price": "1.1",
                                "time": "2026-08-12T10:00:00Z",
                                "pl": "0",
                            },
                            {"id": "8002", "type": "DAILY_FINANCING"},
                        ],
                        "lastTransactionID": "8002",
                    },
                )
            }
        )
        fills, watermark = worker.get_fills_since_id("8000")
        assert len(fills) == 1
        assert watermark == "8002"

    def test_get_fills_walks_the_page_urls(self) -> None:
        page_url = f"{OANDA_PRACTICE_REST}/v3/accounts/{ACCOUNT_ID}/transactions/idrange"
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/transactions"): (
                    200,
                    {"pages": [page_url], "lastTransactionID": "9"},
                ),
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/transactions/idrange"): (
                    200,
                    {
                        "transactions": [
                            {
                                "id": "8001",
                                "type": "ORDER_FILL",
                                "orderID": "8000",
                                "instrument": "EUR_USD",
                                "units": "-50000",
                                "price": "1.1",
                                "time": "2026-08-12T10:00:00Z",
                                "pl": "0",
                            }
                        ]
                    },
                ),
            }
        )
        fills = worker.get_fills(datetime(2026, 8, 12, tzinfo=UTC))
        assert len(fills) == 1
        assert fills[0].side is OrderSide.SELL


class TestWorkerStreaming:
    def test_price_documents_become_ticks_and_heartbeats_are_dropped(self) -> None:
        lines = (
            json.dumps({"type": "HEARTBEAT", "time": "2026-08-12T13:45:00Z"}),
            json.dumps(
                {
                    "type": "PRICE",
                    "instrument": "EUR_USD",
                    "time": "2026-08-12T13:45:30Z",
                    "tradeable": True,
                    "bids": [{"price": "1.09990"}],
                    "asks": [{"price": "1.10002"}],
                }
            ),
            "",
        )
        worker, transport = make_worker(lines=lines)
        ticks = list(worker.subscribe_ticks(["EUR_USD"]))
        assert len(ticks) == 1
        assert ticks[0].bid == Decimal("1.09990")
        assert transport.calls[0][1].startswith(OANDA_PRACTICE_STREAM)

    def test_untradeable_prices_are_dropped(self) -> None:
        lines = (
            json.dumps(
                {
                    "type": "PRICE",
                    "instrument": "EUR_USD",
                    "time": "2026-08-12T13:45:30Z",
                    "tradeable": False,
                    "bids": [{"price": "1.09990"}],
                    "asks": [{"price": "1.10002"}],
                }
            ),
        )
        worker, _ = make_worker(lines=lines)
        assert list(worker.subscribe_ticks(["EUR_USD"])) == []

    def test_undecodable_lines_do_not_kill_the_stream(self) -> None:
        lines = (
            "{not json",
            json.dumps(
                {
                    "type": "PRICE",
                    "instrument": "EUR_USD",
                    "time": "2026-08-12T13:45:30Z",
                    "bids": [{"price": "1.09990"}],
                    "asks": [{"price": "1.10002"}],
                }
            ),
        )
        worker, _ = make_worker(lines=lines)
        assert len(list(worker.subscribe_ticks(["EUR_USD"]))) == 1

    def test_no_symbols_means_no_stream(self) -> None:
        worker, transport = make_worker()
        assert list(worker.subscribe_ticks([])) == []
        assert transport.calls == []


class TestWorkerWrites:
    def order_response(self, filled: bool = True) -> tuple[int, dict[str, Any]]:
        body: dict[str, Any] = {
            "orderCreateTransaction": {"id": "9000", "type": "MARKET_ORDER"},
            "relatedTransactionIDs": ["9000", "9001"],
            "lastTransactionID": "9001",
        }
        if filled:
            body["orderFillTransaction"] = {
                "id": "9001",
                "orderID": "9000",
                "instrument": "EUR_USD",
                "units": "50000",
                "price": "1.10005",
                "time": "2026-08-12T10:00:00Z",
                "pl": "0",
            }
        return 201, body

    def test_market_buy_sends_positive_units_and_fok(self) -> None:
        worker, transport = make_worker(
            {("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): self.order_response()}
        )
        ack = worker.submit_order(
            ForexOrderRequest(symbol="EUR_USD", side=OrderSide.BUY, lots=Decimal("0.5"))
        )
        sent = transport.body_for("POST", f"/v3/accounts/{ACCOUNT_ID}/orders")["order"]
        assert sent["units"] == "50000"
        assert sent["type"] == "MARKET"
        assert sent["timeInForce"] == "FOK"
        assert ack.accepted
        assert ack.filled_lots == Decimal("0.5")

    def test_market_sell_sends_negative_units(self) -> None:
        worker, transport = make_worker(
            {("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): self.order_response()}
        )
        worker.submit_order(
            ForexOrderRequest(symbol="EUR_USD", side=OrderSide.SELL, lots=Decimal("0.25"))
        )
        sent = transport.body_for("POST", f"/v3/accounts/{ACCOUNT_ID}/orders")["order"]
        assert sent["units"] == "-25000"

    def test_protective_levels_ride_along_on_the_fill(self) -> None:
        worker, transport = make_worker(
            {("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): self.order_response()}
        )
        worker.submit_order(
            ForexOrderRequest(
                symbol="EUR_USD",
                side=OrderSide.BUY,
                lots=Decimal("0.1"),
                stop_loss=Decimal("1.09800"),
                take_profit=Decimal("1.10400"),
            )
        )
        sent = transport.body_for("POST", f"/v3/accounts/{ACCOUNT_ID}/orders")["order"]
        assert sent["stopLossOnFill"]["price"] == "1.09800"
        assert sent["takeProfitOnFill"]["price"] == "1.10400"

    def test_a_201_without_a_fill_is_reported_as_placed_not_filled(self) -> None:
        worker, _ = make_worker(
            {("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): self.order_response(filled=False)}
        )
        ack = worker.submit_order(
            ForexOrderRequest(symbol="EUR_USD", side=OrderSide.BUY, lots=Decimal("0.1"))
        )
        assert ack.accepted
        assert ack.filled_lots == Decimal("0")
        assert ack.status.value == "placed"

    def test_limit_order_carries_a_price(self) -> None:
        worker, transport = make_worker(
            {("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): self.order_response(filled=False)}
        )
        worker.submit_order(
            ForexOrderRequest(
                symbol="EUR_USD",
                side=OrderSide.BUY,
                lots=Decimal("0.1"),
                order_type=ForexOrderType.LIMIT,
                price=Decimal("1.09500"),
            )
        )
        sent = transport.body_for("POST", f"/v3/accounts/{ACCOUNT_ID}/orders")["order"]
        assert sent["type"] == "LIMIT"
        assert sent["price"] == "1.09500"
        assert sent["timeInForce"] == "GTC"

    def test_stop_limit_is_refused_because_v20_has_no_such_type(self) -> None:
        worker, _ = make_worker()
        with pytest.raises(ValidationError):
            worker.submit_order(
                ForexOrderRequest(
                    symbol="EUR_USD",
                    side=OrderSide.BUY,
                    lots=Decimal("0.1"),
                    order_type=ForexOrderType.STOP_LIMIT,
                    price=Decimal("1.1"),
                )
            )

    def test_a_rejection_transaction_becomes_a_rejected_ack(self) -> None:
        worker, _ = make_worker(
            {
                ("POST", f"/v3/accounts/{ACCOUNT_ID}/orders"): (
                    201,
                    {
                        "orderRejectTransaction": {
                            "type": "MARKET_ORDER_REJECT",
                            "rejectReason": "INSUFFICIENT_MARGIN",
                        },
                        "lastTransactionID": "9001",
                    },
                )
            }
        )
        ack = worker.submit_order(
            ForexOrderRequest(symbol="EUR_USD", side=OrderSide.BUY, lots=Decimal("0.1"))
        )
        assert not ack.accepted
        assert ack.message == "INSUFFICIENT_MARGIN"

    def test_unknown_symbol_is_refused_before_any_order_is_sent(self) -> None:
        worker, transport = make_worker()
        with pytest.raises(ForexSymbolError):
            worker.submit_order(
                ForexOrderRequest(symbol="XXX_YYY", side=OrderSide.BUY, lots=Decimal("0.1"))
            )
        assert not any(method == "POST" for method, _url, _params, _body in transport.calls)

    def test_modify_stop_puts_to_the_trade_orders_endpoint(self) -> None:
        worker, transport = make_worker(
            {
                ("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/orders"): (
                    200,
                    {"lastTransactionID": "9100"},
                )
            }
        )
        ack = worker.modify_stop(6001, stop_loss=Decimal("1.10250"))
        sent = transport.body_for("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/orders")
        assert sent["stopLoss"]["price"] == "1.10250"
        assert ack.accepted

    def test_modify_stop_needs_at_least_one_level(self) -> None:
        worker, _ = make_worker()
        with pytest.raises(ValidationError):
            worker.modify_stop(6001)

    def test_full_close_sends_all(self) -> None:
        worker, transport = make_worker(
            {
                ("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/close"): (
                    200,
                    {
                        "orderFillTransaction": {
                            "id": "9200",
                            "instrument": "EUR_USD",
                            "units": "-50000",
                            "price": "1.10105",
                            "time": "2026-08-12T11:00:00Z",
                            "pl": "50",
                        }
                    },
                )
            }
        )
        ack = worker.close_position(6001)
        sent = transport.body_for("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/close")
        assert sent["units"] == "ALL"
        assert ack.filled_lots == Decimal("0.5")

    def test_partial_close_sends_units(self) -> None:
        worker, transport = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/trades/6001"): (
                    200,
                    {"trade": {"id": "6001", "instrument": "EUR_USD"}},
                ),
                ("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/close"): (200, {}),
            }
        )
        worker.close_position(6001, lots=Decimal("0.2"))
        sent = transport.body_for("PUT", f"/v3/accounts/{ACCOUNT_ID}/trades/6001/close")
        assert sent["units"] == "20000"

    def test_partial_close_rejects_a_non_positive_size(self) -> None:
        worker, _ = make_worker()
        with pytest.raises(ValidationError):
            worker.close_position(6001, lots=Decimal("0"))

    def test_auth_failure_surfaces_as_an_auth_error(self) -> None:
        worker, _ = make_worker(
            {
                ("GET", f"/v3/accounts/{ACCOUNT_ID}/summary"): (
                    401,
                    {"errorMessage": "Insufficient authorization to perform request."},
                )
            }
        )
        with pytest.raises(ForexAuthenticationError):
            worker.get_account()


class TestProtocolConformance:
    def test_worker_structurally_satisfies_the_broker_protocol(self) -> None:
        worker, _ = make_worker()
        broker: ForexBroker = worker
        assert broker is not None

    def test_disconnect_releases_the_connection_pool(self) -> None:
        worker, transport = make_worker()
        worker.disconnect()
        assert transport.closed

    def test_fake_transport_satisfies_the_http_protocol(self) -> None:
        transport: HttpTransport = FakeTransport()
        assert transport is not None

    def test_the_domain_layer_never_imports_a_venue(self) -> None:
        import quantflow.forex.costs as costs_module
        import quantflow.forex.instruments as instruments_module
        import quantflow.forex.plan as plan_module
        import quantflow.forex.protocol as protocol_module
        import quantflow.forex.sizing as sizing_module

        for module in (
            instruments_module,
            sizing_module,
            costs_module,
            protocol_module,
            plan_module,
        ):
            source = module.__doc__ or ""
            assert "import MetaTrader5" not in source
            names = dir(module)
            assert "MT5Worker" not in names
            assert "OandaWorker" not in names


class TestInstrumentCacheHelpers:
    def test_instrument_is_cached_after_the_first_lookup(self) -> None:
        worker, transport = make_worker()
        worker.get_symbols()
        calls_before = len(transport.calls)
        instrument: ForexInstrument = worker._instrument_or_raise("EUR_USD")
        assert instrument.symbol == "EUR_USD"
        assert len(transport.calls) == calls_before
