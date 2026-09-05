"""Tests for the operations-dashboard API layer.

The central case is the cross-asset balance defect: the previous account endpoint summed
USDT, USDC, BTC and ETH into one "total balance" and reported ``99,904.01`` for an account
whose actual trading capital was ``49,902.01`` USDT. The figure had no unit and was roughly
double the capital the engine could deploy. Every valuation test below exists to keep that
class of error out.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quantflow.api.dashboard import status as status_module
from quantflow.api.dashboard.cache import ResilientCache
from quantflow.api.dashboard.decisions import (
    Decision,
    DecisionLog,
    EngineFacts,
    build_feed,
    categorise,
    decision_feed_key,
    parse_feed,
    parse_line,
    parse_pairs,
    summarise,
)
from quantflow.api.dashboard.valuation import order_rows, position_rows, value_balances
from quantflow.domain.portfolio import Balance

# The live demo account at the time the dashboard was rebuilt.
LIVE_BALANCES = {
    "USDT": Balance(asset="USDT", free=Decimal("37470.99081325"), locked=Decimal("12428.35554076")),
    "USDC": Balance(asset="USDC", free=Decimal("50000.0"), locked=Decimal("0")),
    "BTC": Balance(asset="BTC", free=Decimal("1.0"), locked=Decimal("0")),
    "ETH": Balance(asset="ETH", free=Decimal("1.0"), locked=Decimal("0")),
}

PRICES = {
    "BTC": Decimal("62678.25"),
    "ETH": Decimal("1869.065"),
    "USDC": Decimal("1.00025"),
}


class FakeTicker:
    """Minimal ticker with the mid the valuation code reads."""

    def __init__(self, mid: Decimal) -> None:
        self.bid = mid
        self.ask = mid
        self.last = mid
        self.timestamp = datetime(2026, 8, 14, 13, 48, tzinfo=UTC)

    @property
    def mid(self) -> Decimal:
        """The mid price."""
        return self.bid


class FakeGateway:
    """A gateway that prices the assets it is told about and fails on the rest."""

    name = "bybit"

    def __init__(self, prices: dict[str, Decimal] | None = None) -> None:
        self._prices = prices if prices is not None else PRICES

    async def fetch_ticker(self, symbol: Any) -> FakeTicker:
        """Return a ticker, or raise for an asset with no price."""
        price = self._prices.get(symbol.base)
        if price is None:
            raise RuntimeError(f"no market for {symbol.base}")
        return FakeTicker(price)


def value(balances: dict[str, Balance], gateway: FakeGateway | None = None) -> dict[str, Any]:
    """Run the valuation synchronously for a test."""
    return asyncio.run(value_balances(gateway or FakeGateway(), balances))


class TestCrossAssetBalanceRegression:
    """The 99,904.01 defect must never return in any form."""

    def test_naive_sum_is_never_produced(self) -> None:
        result = value(LIVE_BALANCES)
        naive = sum((item.free + item.locked for item in LIVE_BALANCES.values()), Decimal("0"))

        assert naive == Decimal("99901.34635401")
        # No field anywhere in the payload may carry the unitless cross-asset sum.
        assert str(naive) not in _flatten(result)

    def test_trading_equity_is_usdt_only(self) -> None:
        result = value(LIVE_BALANCES)
        assert result["trading_equity_usdt"] == "49899.34635401"
        assert result["available_usdt"] == "37470.99081325"
        assert result["locked_usdt"] == "12428.35554076"

    def test_no_field_is_named_total_balance(self) -> None:
        result = value(LIVE_BALANCES)
        assert "total_balance" not in result
        assert "available_balance" not in result

    def test_other_assets_are_listed_separately(self) -> None:
        result = value(LIVE_BALANCES)
        assets = {item["asset"] for item in result["other_assets"]}
        assert assets == {"BTC", "ETH", "USDC"}
        assert "USDT" not in assets

    def test_usdc_is_valued_at_its_traded_rate_not_assumed_at_par(self) -> None:
        # Assuming a stablecoin equals the quote asset is the same unstated conversion the
        # defect made, merely with a smaller error.
        result = value(LIVE_BALANCES)
        usdc = next(item for item in result["other_assets"] if item["asset"] == "USDC")
        assert usdc["price_usdt"] == "1.00025"
        assert Decimal(usdc["value_usdt"]) == Decimal("50012.5")

    def test_total_is_labelled_in_usdt_with_method_and_timestamp(self) -> None:
        result = value(LIVE_BALANCES)
        assert result["total_portfolio_value_usdt"] is not None
        assert Decimal(result["total_portfolio_value_usdt"]) == Decimal("164459.16135401")
        assert "order-book mid" in result["valuation_method"]
        assert result["valued_at"] is not None

    def test_total_is_withheld_when_any_asset_cannot_be_priced(self) -> None:
        balances = {
            **LIVE_BALANCES,
            "WEIRD": Balance(asset="WEIRD", free=Decimal("5"), locked=Decimal("0")),
        }
        result = value(balances)

        assert result["total_portfolio_value_usdt"] is None
        assert result["valuation_method"] is None
        assert result["unpriced_assets"] == ["WEIRD"]
        weird = next(item for item in result["other_assets"] if item["asset"] == "WEIRD")
        assert weird["value_usdt"] is None
        assert "no current price" in weird["unpriced_reason"]

    def test_every_monetary_field_is_a_string(self) -> None:
        # Decimal on the wire as a JSON number becomes a float in the browser.
        result = value(LIVE_BALANCES)
        for key in ("trading_equity_usdt", "available_usdt", "total_portfolio_value_usdt"):
            assert isinstance(result[key], str)
        for item in result["other_assets"]:
            assert isinstance(item["quantity"], str)


class TestEmptyAndDegenerateAccounts:
    def test_empty_account_reports_zero_rather_than_failing(self) -> None:
        result = value({})
        assert result["trading_equity_usdt"] == "0"
        assert result["other_assets"] == []
        # Nothing is unpriced, so a total is still well defined — and it is zero.
        assert result["total_portfolio_value_usdt"] == "0"

    def test_usdt_only_account_has_no_other_assets(self) -> None:
        result = value({"USDT": LIVE_BALANCES["USDT"]})
        assert result["other_assets"] == []
        assert Decimal(result["total_portfolio_value_usdt"]) == Decimal("49899.34635401")

    def test_zero_balances_are_not_listed(self) -> None:
        result = value(
            {
                "USDT": LIVE_BALANCES["USDT"],
                "DUST": Balance(asset="DUST", free=Decimal("0"), locked=Decimal("0")),
            }
        )
        assert result["other_assets"] == []


class TestPositionRows:
    def test_zero_size_positions_are_dropped(self) -> None:
        rows, total = position_rows(
            [
                {"symbol": "SOL/USDT:USDT", "contracts": "33.1", "unrealizedPnl": "0.331"},
                {"symbol": "OLD/USDT:USDT", "contracts": "0", "unrealizedPnl": "0"},
            ]
        )
        assert len(rows) == 1
        assert total == Decimal("0.331")

    def test_venue_floats_do_not_become_binary_decimals(self) -> None:
        rows, _ = position_rows([{"symbol": "X/USDT", "contracts": 0.1, "entryPrice": 0.3}])
        assert rows[0]["quantity"] == "0.1"
        assert rows[0]["entry_price"] == "0.3"

    def test_absent_stop_is_none_not_zero(self) -> None:
        rows, _ = position_rows([{"symbol": "X/USDT", "contracts": "1", "info": {"stopLoss": "0"}}])
        assert rows[0]["venue_stop_loss"] is None

    def test_missing_fields_do_not_raise(self) -> None:
        rows, total = position_rows([{"contracts": "1"}])
        assert rows[0]["symbol"] == ""
        assert total == Decimal("0")


class TestOrderRows:
    def test_status_comes_from_the_venue(self) -> None:
        from quantflow.domain.enums import OrderSide, OrderStatus, OrderType, TimeInForce
        from quantflow.domain.instruments import Symbol
        from quantflow.domain.orders import Order

        order = Order(
            order_id="a",
            client_order_id="c",
            symbol=Symbol(base="SOL", quote="USDT"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("33.1"),
            status=OrderStatus.FILLED,
            created_at=datetime(2026, 8, 14, tzinfo=UTC),
            updated_at=datetime(2026, 8, 14, tzinfo=UTC),
            time_in_force=TimeInForce.GTC,
            metadata={"purpose": "take_profit"},
        )
        rows = order_rows([order])
        assert rows[0]["status"] == "filled"
        assert rows[0]["purpose"] == "take_profit"

    def test_empty_book_is_an_empty_list(self) -> None:
        assert order_rows([]) == []


class TestDecisionLogParsing:
    SELECTED = (
        "2026-08-14T13:45:37.970565Z [info     ] orchestrator.selected          "
        "[quantflow.orchestrator.strategy] candidates=9 components={'confidence': '1.00', "
        "'risk_reward': '0.67', 'cost': '0.84'} confidence=1.00 direction=long mode=live "
        "runner_up='macd_trend/SOL/USDT short conf=1.00 score=0.657' score=0.661 "
        "strategy=momentum_roc symbol=SOL/USDT version=0.1.0"
    )
    DESELECTED = (
        "2026-08-14T13:00:06.225320Z [info     ] orchestrator.all_deselected    "
        "[quantflow.orchestrator.strategy] candidates=5 first_reason='correlation 1.00 with "
        "an open position exceeds 0.85' mode=live regime=range symbol=ETH/USDT version=0.1.0"
    )
    DENIED = (
        "2026-08-14T10:15:34.966503Z [warning  ] risk.order_denied              "
        "[quantflow.risk.engine] mode=live quantity=4.11 reason='15 consecutive losses hit "
        "the limit of 4; new entries pause for another 235.3 minute(s)' "
        "rules=['consecutive_loss_cooldown'] side=sell symbol=BNB/USDT version=0.1.0"
    )

    def test_selection_is_parsed_with_its_component_scores(self) -> None:
        decision = parse_line(self.SELECTED)
        assert decision is not None
        assert decision.outcome == "SELECTED"
        assert decision.symbol == "SOL/USDT"
        assert decision.strategy == "momentum_roc"
        assert decision.direction == "long"
        assert decision.score == "0.661"
        assert decision.candidates == 9
        assert decision.components["cost"] == "0.84"

    def test_deselection_carries_the_reason_and_its_category(self) -> None:
        decision = parse_line(self.DESELECTED)
        assert decision is not None
        assert decision.outcome == "DESELECTED"
        assert decision.reason is not None
        assert "correlation 1.00" in decision.reason
        assert decision.category == "correlation"
        assert decision.regime == "range"

    def test_risk_denial_is_categorised_as_risk(self) -> None:
        decision = parse_line(self.DENIED)
        assert decision is not None
        assert decision.outcome == "RISK_BLOCKED"
        assert decision.category == "risk"

    def test_quoted_values_containing_spaces_survive(self) -> None:
        pairs = parse_pairs("reason='a b c' other=1")
        assert pairs["reason"] == "a b c"
        assert pairs["other"] == "1"

    def test_unrelated_lines_are_ignored(self) -> None:
        assert parse_line("2026-08-14T13:03:43Z [warning  ] stream.bad_ticker [x] a=1") is None
        assert parse_line("not a log line at all") is None
        assert parse_line("") is None

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("correlation 1.00 with an open position", "correlation"),
            ("failed the economic gates on cost", "cost"),
            ("insufficient confluence across the field", "confluence"),
            ("book depth too thin", "liquidity"),
            ("15 consecutive losses hit the limit of 4", "risk"),
            ("something nobody anticipated", "other"),
        ],
    )
    def test_categorisation(self, reason: str, expected: str) -> None:
        assert categorise(reason) == expected

    def test_summary_counts_outcomes_and_categories(self) -> None:
        decisions = [
            parse_line(self.SELECTED),
            parse_line(self.DESELECTED),
            parse_line(self.DESELECTED),
        ]
        summary = summarise([item for item in decisions if item is not None])
        assert summary["evaluated"] == 3
        assert summary["selected"] == 1
        assert summary["declined"] == 2
        assert summary["by_rejection_category"]["correlation"] == 2

    def test_empty_summary_is_well_formed(self) -> None:
        summary = summarise([])
        assert summary["evaluated"] == 0
        assert summary["first_at"] is None


class TestIncrementalLogReader:
    def test_only_new_bytes_are_read_on_refresh(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(TestDecisionLogParsing.DESELECTED + "\n", encoding="utf-8")

        reader = DecisionLog(log)
        assert len(reader.refresh()) == 1

        # Appending one line must add exactly one decision, not re-read the file.
        with log.open("a", encoding="utf-8") as handle:
            handle.write(TestDecisionLogParsing.SELECTED + "\n")
        assert len(reader.refresh()) == 2

    def test_a_partial_trailing_line_is_not_parsed_until_complete(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(TestDecisionLogParsing.SELECTED[:60], encoding="utf-8")
        reader = DecisionLog(log)
        assert reader.refresh() == []

        log.write_text(TestDecisionLogParsing.SELECTED + "\n", encoding="utf-8")
        assert len(reader.refresh()) == 1

    def test_truncation_resets_rather_than_reading_past_the_end(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(TestDecisionLogParsing.SELECTED + "\n", encoding="utf-8")
        reader = DecisionLog(log)
        reader.refresh()

        log.write_text(TestDecisionLogParsing.DESELECTED + "\n", encoding="utf-8")
        decisions = reader.refresh()
        assert len(decisions) == 1
        assert decisions[0].outcome == "DESELECTED"

    def test_a_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert DecisionLog(tmp_path / "absent.log").refresh() == []


class TestAgreementBlockDetection:
    """An unsigned-agreement block must be read from the venue's answer, not from digits.

    The venue codes are six-digit numbers, and six-digit numbers occur constantly in a
    trading log — inside prices, quantities, order ids and durations. Matching them as bare
    substrings marked the crypto book, which trades perfectly well, as blocked.
    """

    QUARANTINE = (
        "2026-08-14T15:00:00.000000Z [critical ] paper.class_quarantined "
        "[quantflow.paper.engine] symbol=XAU/USDT asset_class=metal venue_code=110123"
    )

    #: A perfectly ordinary line. `0.110125` is a price, not a venue code.
    INNOCENT = (
        "2026-08-14T15:00:01.000000Z [info     ] paper.order_placed "
        "[quantflow.paper.engine] symbol=BTC/USDT price=0.110125 quantity=110126"
    )

    def test_a_quarantine_event_records_the_class_the_engine_named(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(self.QUARANTINE + "\n", encoding="utf-8")

        reader = DecisionLog(log)
        reader.refresh()

        assert reader.facts().agreement_blocked_classes == ("metal",)
        assert "XAU/USDT" in reader.facts().agreement_blocked
        assert reader.facts().agreement_codes == ("110123",)

    def test_a_price_that_contains_a_code_is_not_a_block(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(self.INNOCENT + "\n", encoding="utf-8")

        reader = DecisionLog(log)
        reader.refresh()

        assert reader.facts().agreement_blocked == ()
        assert reader.facts().agreement_blocked_classes == ()
        assert reader.facts().agreement_codes == ()

    def test_engine_facts_are_read_from_the_startup_line(self, tmp_path: Path) -> None:
        log = tmp_path / "bot.log"
        log.write_text(
            "2026-08-14T13:42:36.906334Z [critical ] demo_bot.starting              "
            "[demo_bot] env=demo equity_source=venue max_concurrent=10 mode=live "
            "pool='full registry' starting_equity=49900.64518307 strategy=orchestrator "
            "symbols=['BTC/USDT', 'ETH/USDT'] timeframe=15m version=0.1.0\n",
            encoding="utf-8",
        )
        reader = DecisionLog(log)
        reader.refresh()
        facts = reader.facts()

        assert facts.timeframe == "15m"
        assert facts.symbols == ("BTC/USDT", "ETH/USDT")
        assert facts.pool == "full registry"
        assert facts.equity_source == "venue"


class TestStatusDerivation:
    def _derive(self, **overrides: Any) -> status_module.Status:
        base: dict[str, Any] = {
            "venue_available": True,
            "venue_error": None,
            "kill_switch_engaged": False,
            "trading_halted": False,
            "session_running": True,
            "session_status": "running",
            "open_position_count": 0,
            "last_snapshot_at": datetime.now(UTC),
            "decisions": [],
            "recent_order_rejections": 0,
        }
        base.update(overrides)
        return status_module.derive(**base)

    def test_unreachable_venue_outranks_everything(self) -> None:
        status = self._derive(venue_available=False, venue_error="connection refused")
        assert status.state == status_module.DISCONNECTED

    def test_kill_switch_reports_risk_blocked(self) -> None:
        assert self._derive(kill_switch_engaged=True).state == status_module.RISK_BLOCKED

    def test_a_stopped_session_is_an_engine_error(self) -> None:
        status = self._derive(session_running=False, session_status="completed")
        assert status.state == status_module.ENGINE_ERROR

    def test_a_long_snapshot_silence_is_an_engine_error(self) -> None:
        status = self._derive(last_snapshot_at=datetime.now(UTC) - timedelta(hours=4))
        assert status.state == status_module.ENGINE_ERROR

    def test_open_positions_mean_trading(self) -> None:
        decision = parse_line(TestDecisionLogParsing.SELECTED)
        assert decision is not None
        fresh = _at_now(decision)
        assert self._derive(open_position_count=3, decisions=[fresh]).state == status_module.TRADING

    def test_declining_every_bar_is_waiting_not_an_error(self) -> None:
        decision = parse_line(TestDecisionLogParsing.DESELECTED)
        assert decision is not None
        status = self._derive(decisions=[_at_now(decision)])
        assert status.state == status_module.WAITING

    def test_no_decisions_at_all_reads_as_starting(self) -> None:
        assert self._derive(decisions=[]).state == status_module.STARTING

    def test_a_stale_decision_log_is_not_an_engine_error_while_state_is_being_written(
        self,
    ) -> None:
        """A fresh equity snapshot outranks a silent decision log.

        The decision log is read from a file the API sees through a bind mount, and on this
        deployment that view goes stale within minutes of a container start while the host
        keeps appending — the same reader run against the same path on the host returns
        decisions the container never sees. The equity snapshot comes from Postgres, a real
        network service, and the engine writes one every bar.

        So a silent log beside a current snapshot means the log is not being read, not that
        the engine has stopped. Reporting ENGINE ERROR there sent the operator hunting a
        dead bot twice while it was managing positions perfectly well.
        """
        decision = parse_line(TestDecisionLogParsing.SELECTED)
        assert decision is not None
        stale = _shifted(decision, datetime.now(UTC) - timedelta(hours=3))

        status = self._derive(
            decisions=[stale],
            last_snapshot_at=datetime.now(UTC) - timedelta(minutes=2),
            open_position_count=2,
        )

        assert status.state != status_module.ENGINE_ERROR
        assert status.state == status_module.TRADING

    def test_a_stale_decision_log_with_a_stale_snapshot_is_still_an_engine_error(
        self,
    ) -> None:
        # Both sources silent is the real failure, and must keep reporting as one.
        decision = parse_line(TestDecisionLogParsing.SELECTED)
        assert decision is not None
        stale = _shifted(decision, datetime.now(UTC) - timedelta(hours=3))

        status = self._derive(
            decisions=[stale], last_snapshot_at=datetime.now(UTC) - timedelta(hours=3)
        )

        assert status.state == status_module.ENGINE_ERROR

    def test_evidence_is_always_attached(self) -> None:
        status = self._derive()
        assert "venue_available" in status.evidence
        assert "last_snapshot_at" in status.evidence


class TestResilientCache:
    def test_a_failed_refresh_serves_the_previous_value_marked_stale(self) -> None:
        cache: ResilientCache[str] = ResilientCache(0.0, name="test")

        async def scenario() -> None:
            good = await cache.get(_returning("first"))
            assert good.value == "first"
            assert good.stale is False

            bad = await cache.get(_raising(RuntimeError("venue down")))
            assert bad.value == "first"
            assert bad.stale is True
            assert bad.error == "venue down"

        asyncio.run(scenario())

    def test_a_timeout_does_not_propagate(self) -> None:
        cache: ResilientCache[str] = ResilientCache(0.0, name="test")

        async def slow() -> str:
            await asyncio.sleep(5)
            return "never"

        async def scenario() -> None:
            result = await cache.get(slow, deadline_seconds=0.01)
            assert result.value is None
            assert result.error is not None
            assert "timed out" in result.error

        asyncio.run(scenario())

    def test_a_value_within_its_ttl_is_not_recomputed(self) -> None:
        cache: ResilientCache[int] = ResilientCache(60.0, name="test")
        calls = 0

        async def factory() -> int:
            nonlocal calls
            calls += 1
            return calls

        async def scenario() -> None:
            await cache.get(factory)
            await cache.get(factory)
            assert calls == 1

        asyncio.run(scenario())

    def test_a_failure_does_not_reset_the_clock_on_the_data(self) -> None:
        """The lie that showed a 2h40m-old venue read as live.

        A failed refresh pushes the TTL forward so the dependency is not hammered while it
        is down. The bug was that the *next* caller then passed the freshness check, and
        was handed the old value marked ``stale=False`` — because freshness was judged on
        the backoff window rather than on the age of the data.

        Observed live on 2026-08-15: every Bybit read had failed for 2h40m on an
        InvalidNonce, and the dashboard reported ``age_seconds: 9615`` beside
        ``stale: False, available: True``. It showed zero open positions while the venue
        held three.
        """
        # A long TTL so a failure's backoff window is wide, and a short max age so the
        # payload ages out inside it — the exact shape of the live failure.
        cache: ResilientCache[str] = ResilientCache(5.0, name="test", max_age_seconds=0.3)

        async def scenario() -> None:
            first = await cache.get(_returning("first"))
            assert first.stale is False

            # Let the payload age out, then fail the refresh. The failure pushes the
            # backoff window five seconds into the future.
            await asyncio.sleep(0.4)
            failed = await cache.get(_raising(RuntimeError("InvalidNonce")))
            assert failed.stale is True

            # This caller lands squarely inside that backoff window, which is where the
            # old code handed back the ageing value marked fresh.
            served = await cache.get(_raising(RuntimeError("InvalidNonce")))
            assert served.value == "first"
            assert served.stale is True, "a stale payload must never be reported as fresh"
            assert served.age_seconds is not None
            assert served.age_seconds > 0.3

        asyncio.run(scenario())

    def test_a_value_inside_its_max_age_is_still_fresh(self) -> None:
        # The guard must not flip everything to stale: a recent value is genuinely fresh.
        cache: ResilientCache[str] = ResilientCache(60.0, name="test", max_age_seconds=30.0)

        async def scenario() -> None:
            await cache.get(_returning("first"))
            again = await cache.get(_returning("second"))
            assert again.stale is False

        asyncio.run(scenario())

    def test_a_never_fetched_value_is_absent_rather_than_stale(self) -> None:
        cache: ResilientCache[str] = ResilientCache(0.0, name="test")

        async def scenario() -> None:
            result = await cache.get(_raising(RuntimeError("down")))
            assert result.value is None
            assert result.stale is False

        asyncio.run(scenario())


def _at_now(decision: Any) -> Any:
    """Move a parsed decision to the present so recency filters see it."""
    from dataclasses import replace

    return replace(decision, timestamp=datetime.now(UTC))


def _shifted(decision: Any, moment: datetime) -> Any:
    """Move a parsed decision to a specific time."""
    from dataclasses import replace

    return replace(decision, timestamp=moment)


def _returning(value: str) -> Any:
    """A factory that returns ``value``."""

    async def factory() -> str:
        return value

    return factory


def _raising(error: Exception) -> Any:
    """A factory that raises."""

    async def factory() -> str:
        raise error

    return factory


def _flatten(payload: Any) -> str:
    """Render a payload to one string, for 'this number appears nowhere' assertions."""
    return repr(payload)


class TestAssetClassification:
    """Classification must agree with the engine's own taxonomy."""

    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("BTC/USDT", "crypto"),
            ("ETH/USDT", "crypto"),
            # XRP is X-prefixed and three characters. The engine's X-prefix rule applies
            # only inside the venue's `commodity` tag; applied blindly it makes XRP a
            # precious metal, which then reports a metals desk that does not exist.
            ("XRP/USDT", "crypto"),
            ("FARTCOIN/USDT", "crypto"),
            ("XAU/USDT", "metal"),
            ("XAG/USDT", "metal"),
            ("CL/USDT", "energy"),
            ("BZ/USDT", "energy"),
            ("SOXL/USDT", "index"),
            ("1000PEPE/USDT", "crypto"),
        ],
    )
    def test_classification(self, symbol: str, expected: str) -> None:
        from quantflow.api.dashboard.session_state import _asset_class

        assert _asset_class(symbol) == expected


class TestDecisionsAreScopedToTheCurrentSession:
    """The Overview page must never show a previous run's decisions as current.

    Decisions are parsed from the engine's log, and that log is a single append-only file
    shared by every run — 576 MB spanning 57 engine starts on this deployment. The lines
    carry ``session_id=***redacted***``, so the session cannot be recovered from them at
    all. Without a boundary the panel simply reports the tail of the file.

    Observed two minutes into a fresh session: **500 evaluated · 149 selected · 351
    declined · 8 orders refused · last decision 3h42m ago**, on a session that had made
    exactly zero decisions. Every number belonged to runs that had already ended.

    The session's own ``started_at`` is the only usable discriminator, and it is exact: an
    engine cannot have decided anything before it started.
    """

    @staticmethod
    def _decision_at(moment: datetime) -> Any:
        parsed = parse_line(TestDecisionLogParsing.SELECTED)
        assert parsed is not None
        return _shifted(parsed, moment)

    def test_decisions_from_before_the_session_started_are_excluded(self) -> None:
        from quantflow.api.dashboard.decisions import since

        started = datetime(2026, 8, 16, 11, 25, 40, tzinfo=UTC)
        older = self._decision_at(started - timedelta(hours=3))
        current = self._decision_at(started + timedelta(minutes=5))

        kept = since([older, current], started)

        assert kept == [current]

    def test_a_session_with_no_decisions_yet_reports_an_empty_list(self) -> None:
        # Not the previous run's tail. Zero is the truthful answer here.
        from quantflow.api.dashboard.decisions import since

        started = datetime(2026, 8, 16, 11, 25, 40, tzinfo=UTC)
        older = self._decision_at(started - timedelta(minutes=1))

        assert since([older], started) == []

    def test_no_start_time_keeps_everything(self) -> None:
        # A session with no recorded start cannot be filtered, and silently showing
        # nothing would be its own kind of lie.
        from quantflow.api.dashboard.decisions import since

        older = self._decision_at(datetime(2026, 8, 16, 8, 0, tzinfo=UTC))

        assert since([older], None) == [older]

    def test_a_decision_exactly_at_the_start_is_kept(self) -> None:
        from quantflow.api.dashboard.decisions import since

        started = datetime(2026, 8, 16, 11, 25, 40, tzinfo=UTC)
        boundary = self._decision_at(started)

        assert since([boundary], started) == [boundary]


class TestOneRejectionIsNotAnEngineState:
    """A single refused order is a fact about that order, not about the engine.

    Observed live: one order was refused at 11:30 for breaching the position cap — the risk
    engine working exactly as configured — and forty minutes later the headline still read
    EXECUTION BLOCKED in red, while the engine had gone on to evaluate two more bars and
    select three candidates.

    "The engine is running", "the bot is waiting for a setup" and "the last order was
    refused" are three separate facts. Collapsing them into one red box tells the operator
    to go hunting for a fault that does not exist.
    """

    @staticmethod
    def _decision(outcome: str, moment: datetime) -> Any:
        raw = (
            TestDecisionLogParsing.SELECTED
            if outcome == "SELECTED"
            else TestDecisionLogParsing.DESELECTED
        )
        parsed = parse_line(raw)
        assert parsed is not None
        return _shifted(parsed, moment)

    def test_a_single_stale_rejection_does_not_block_the_headline(self) -> None:
        now = datetime.now(UTC)
        status = status_module.derive(
            venue_available=True,
            venue_error=None,
            kill_switch_engaged=False,
            trading_halted=False,
            session_running=True,
            session_status="running",
            open_position_count=0,
            last_snapshot_at=now - timedelta(minutes=1),
            decisions=[self._decision("SELECTED", now - timedelta(minutes=1))],
            recent_order_rejections=1,
        )

        assert status.state != status_module.EXECUTION_BLOCKED

    def test_repeated_rejections_still_block(self) -> None:
        # Selection succeeding and orders repeatedly failing is a genuine execution fault
        # and must keep reporting as one.
        now = datetime.now(UTC)
        status = status_module.derive(
            venue_available=True,
            venue_error=None,
            kill_switch_engaged=False,
            trading_halted=False,
            session_running=True,
            session_status="running",
            open_position_count=0,
            last_snapshot_at=now - timedelta(minutes=1),
            decisions=[self._decision("SELECTED", now - timedelta(minutes=1))],
            recent_order_rejections=4,
        )

        assert status.state == status_module.EXECUTION_BLOCKED


class TestDecisionFeed:
    """Publishing decisions through Redis rather than re-reading a bind-mounted file."""

    def _decision(self) -> Decision:
        return Decision(
            timestamp=datetime(2026, 8, 16, 12, 45, 11, tzinfo=UTC),
            event="orchestrator.selected",
            symbol="SNDK/USDT",
            outcome="SELECTED",
            strategy="momentum_roc",
            direction="long",
            score="0.665",
            confidence="1.00",
            candidates=2,
            regime="range",
            reason=None,
            category=None,
            components={"confidence": "1.00", "cost": "0.88"},
            runner_up="keltner_trend/SNDK/USDT short",
        )

    def test_a_decision_survives_the_round_trip(self) -> None:
        assert Decision.from_dict(self._decision().to_dict()) == self._decision()

    def test_the_key_is_scoped_to_one_session(self) -> None:
        assert decision_feed_key("a") != decision_feed_key("b")

    def test_an_empty_feed_is_not_a_missing_feed(self) -> None:
        # The distinction is what stops an idle engine falling back to the stale file: an
        # engine that has published nothing yet has genuinely decided nothing.
        assert parse_feed({"decisions": [], "facts": {}}) == ([], EngineFacts())
        assert parse_feed(None) is None

    def test_a_malformed_entry_is_dropped_rather_than_raising(self) -> None:
        payload = {"decisions": [self._decision().to_dict(), {"timestamp": "x"}, {"n": 1}]}
        decisions, _ = parse_feed(payload)  # type: ignore[misc]
        assert decisions == [self._decision()]

    def test_facts_survive_the_round_trip(self) -> None:
        # The engine's start time must travel with its decisions. Read from the file it
        # reported the *previous* run's start beside the current run's decisions.
        facts = EngineFacts(
            started_at=datetime(2026, 8, 16, 13, 3, 39, tzinfo=UTC),
            mode="live",
            symbols=("BTC/USDT",),
            agreement_blocked_classes=("equity",),
        )
        restored = parse_feed(build_feed([], facts))
        assert restored is not None
        assert restored[1] == facts

    def test_a_bare_list_still_reads_as_decisions(self) -> None:
        # An engine published before facts were carried must not be discarded whole.
        assert parse_feed([self._decision().to_dict()]) == ([self._decision()], None)
