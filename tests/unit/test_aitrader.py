"""Tests for the AI trading service.

Weighted heavily toward the validation boundary, because that is where untrusted model
output becomes an order, and toward the guarantee that nothing here reaches a venue
without passing the risk engine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.aitrader.client import Completion, NullClient
from quantflow.aitrader.context import (
    build_indicators,
    describe_positions,
    summarise_order_book,
)
from quantflow.aitrader.decision import Action, AIDecision, DecisionError, parse_decision
from quantflow.aitrader.journal import CycleRecord, DecisionJournal
from quantflow.aitrader.prompt import SYSTEM_PROMPT, build_user_prompt
from quantflow.aitrader.service import AITradingService
from quantflow.core.config import LLMSettings, RiskSettings, TradingMode
from quantflow.domain.enums import OrderSide, OrderStatus, OrderType, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, OrderBookLevel
from quantflow.domain.orders import Order
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.execution.engine import ExecutionEngine
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine

BTC = Symbol(base="BTC", quote="USDT")
ETH = Symbol(base="ETH", quote="USDT")
NOW = datetime(2026, 6, 1, tzinfo=UTC)


class TestDecisionParsing:
    """Untrusted text becoming an order is the highest-risk boundary here."""

    def test_a_clean_decision_parses(self) -> None:
        text = '{"action":"BUY","symbol":"BTCUSDT","confidence":0.8,"reason":"trend up"}'
        result = parse_decision(text, allowed=(BTC,))
        assert isinstance(result, AIDecision)
        assert result.action is Action.BUY
        assert result.confidence == Decimal("0.8")

    def test_a_fenced_block_is_unwrapped(self) -> None:
        text = (
            "Here:\n```json\n"
            '{"action":"HOLD","symbol":"BTCUSDT","confidence":0.1,"reason":"flat"}'
            "\n```"
        )
        assert isinstance(parse_decision(text, allowed=(BTC,)), AIDecision)

    def test_prose_around_one_object_is_tolerated(self) -> None:
        text = 'I think: {"action":"HOLD","symbol":"BTCUSDT","confidence":0.2,"reason":"chop"} ok?'
        assert isinstance(parse_decision(text, allowed=(BTC,)), AIDecision)

    def test_two_fenced_objects_are_refused(self) -> None:
        # Picking one would be a guess, and a guess here opens a position nobody chose.
        text = (
            '```json\n{"action":"BUY","symbol":"BTCUSDT","confidence":0.9,"reason":"a"}\n```\n'
            '```json\n{"action":"SELL","symbol":"BTCUSDT","confidence":0.9,"reason":"b"}\n```'
        )
        assert isinstance(parse_decision(text, allowed=(BTC,)), DecisionError)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "no json at all",
            "{not valid json}",
            "[1,2,3]",
            '{"action":"MAYBE","symbol":"BTCUSDT","confidence":0.5,"reason":"x"}',
            '{"action":"BUY","confidence":0.5,"reason":"x"}',
            '{"action":"BUY","symbol":"BTCUSDT","reason":"x"}',
            '{"action":"BUY","symbol":"BTCUSDT","confidence":1.5,"reason":"x"}',
            '{"action":"BUY","symbol":"BTCUSDT","confidence":-0.1,"reason":"x"}',
            '{"action":"BUY","symbol":"BTCUSDT","confidence":"high","reason":"x"}',
            '{"action":"BUY","symbol":"BTCUSDT","confidence":0.5,"reason":""}',
            '{"action":"BUY","symbol":"BTCUSDT","confidence":0.5}',
        ],
    )
    def test_malformed_payloads_fail_closed(self, text: str) -> None:
        assert isinstance(parse_decision(text, allowed=(BTC,)), DecisionError)

    def test_an_unpermitted_symbol_is_refused(self) -> None:
        # A model inventing a ticker must never reach an exchange. "The model asked for
        # it" is not authorisation.
        text = '{"action":"BUY","symbol":"DOGEUSDT","confidence":0.9,"reason":"moon"}'
        result = parse_decision(text, allowed=(BTC, ETH))
        assert isinstance(result, DecisionError)
        assert "not one of the permitted" in result.reason

    def test_boolean_confidence_is_refused(self) -> None:
        # True is an int in Python and would otherwise pass as 1.0 - the most dangerous
        # value in the range.
        text = '{"action":"BUY","symbol":"BTCUSDT","confidence":true,"reason":"x"}'
        assert isinstance(parse_decision(text, allowed=(BTC,)), DecisionError)

    @pytest.mark.parametrize("spelling", ["BTCUSDT", "BTC/USDT", "btcusdt", "BTC-USDT"])
    def test_symbol_notation_is_accepted_in_any_common_form(self, spelling: str) -> None:
        text = f'{{"action":"HOLD","symbol":"{spelling}","confidence":0.1,"reason":"x"}}'
        assert isinstance(parse_decision(text, allowed=(BTC,)), AIDecision)

    def test_the_confidence_floor_is_a_separate_decision(self) -> None:
        text = '{"action":"BUY","symbol":"BTCUSDT","confidence":0.3,"reason":"weak"}'
        result = parse_decision(text, allowed=(BTC,))
        assert isinstance(result, AIDecision)
        assert not result.meets(Decimal("0.65"))


def candles(count: int = 250) -> list[Candle]:
    """A rising series."""
    return [
        Candle(
            symbol=BTC,
            timeframe=Timeframe.H1,
            open_time=NOW + timedelta(hours=i),
            open=Decimal(1000 + i),
            high=Decimal(1002 + i),
            low=Decimal(998 + i),
            close=Decimal(1000 + i),
            volume=Decimal("100"),
            quote_volume=Decimal("100000"),
            trades=10,
        )
        for i in range(count)
    ]


class TestContext:
    """The model must be told what it is not being given."""

    def test_indicators_are_computed(self) -> None:
        values, missing = build_indicators(candles())
        assert "rsi_14" in values
        assert "ema_200" in values
        assert not missing

    def test_short_history_reports_what_is_missing(self) -> None:
        _, missing = build_indicators(candles(30))
        assert missing
        assert "ema_200" in missing

    def test_no_candles_is_reported_not_silently_empty(self) -> None:
        values, missing = build_indicators([])
        assert values == {}
        assert missing

    def test_the_order_book_is_reduced_to_what_matters(self) -> None:
        book = OrderBook(
            symbol=BTC,
            timestamp=NOW,
            bids=(OrderBookLevel(price=Decimal("100"), quantity=Decimal("2")),),
            asks=(OrderBookLevel(price=Decimal("101"), quantity=Decimal("1")),),
        )
        summary = summarise_order_book(book)
        assert summary is not None
        assert summary["spread"] == "1.00"
        # More resting size on the bid than the ask.
        assert Decimal(summary["bid_share"]) > Decimal("0.5")

    def test_a_missing_book_is_none_not_a_fabricated_summary(self) -> None:
        assert summarise_order_book(None) is None

    def test_positions_render_without_a_mark_price(self) -> None:
        snapshot = PortfolioSnapshot(timestamp=NOW, base_currency="USDT", cash=Decimal("1000"))
        assert describe_positions(snapshot) == ()


class TestPrompt:
    """The prompt must state the model's limits and the data's gaps."""

    def test_the_system_prompt_states_what_the_model_cannot_do(self) -> None:
        assert "do NOT choose position size" in SYSTEM_PROMPT
        assert "do NOT set stop losses" in SYSTEM_PROMPT
        assert "Prefer HOLD" in SYSTEM_PROMPT

    def test_the_system_prompt_names_the_cost_of_trading(self) -> None:
        # Without this the model has no basis for judging whether a move is worth taking.
        assert "0.20%" in SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_the_user_prompt_lists_unavailable_data(self) -> None:
        service = build_service()
        context = await service._build_context(NOW)
        rendered = build_user_prompt(context)
        assert "UNAVAILABLE" in rendered
        assert "PERMITTED SYMBOLS" in rendered


class StubClient:
    """Returns a scripted response."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    @property
    def model(self) -> str:
        return "stub"

    async def complete(self, *, system: str, user: str) -> Completion:
        del system, user
        self.calls += 1
        return Completion(text=self._text, model="stub")

    async def aclose(self) -> None:
        return None


class StubMarket:
    """Serves candles. Read-only: the service can observe but not trade through it."""

    def __init__(self, bars: list[Candle] | None = None) -> None:
        self._bars = bars if bars is not None else candles()

    async def fetch_candles(self, symbol: Symbol, timeframe: object, *, limit: int) -> list[Candle]:
        del symbol, timeframe
        return self._bars[-limit:]


class StubGateway(StubMarket):
    """A venue that accepts whatever the execution layer sends it.

    Separate from StubMarket on purpose: the service is handed the read-only one, so a
    test would fail loudly if it ever tried to reach a venue directly instead of going
    through the risk-gated execution path.
    """

    supports_trading = True
    is_testnet = True

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[object] = []

    async def submit_order(self, request: object) -> Order:
        self.submitted.append(request)
        quantity = getattr(request, "quantity", Decimal("1"))
        symbol = getattr(request, "symbol", BTC)
        side = getattr(request, "side", OrderSide.BUY)
        price = Decimal("1249")
        return Order(
            order_id="stub-1",
            client_order_id="c-stub-1",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            status=OrderStatus.FILLED,
            created_at=NOW,
            updated_at=NOW,
            filled_quantity=quantity,
            average_fill_price=price,
        )


def build_service(response: str | None = None) -> AITradingService:
    """A service wired to stubs, in paper mode."""
    settings = RiskSettings(
        max_position_pct=Decimal("0.5"),
        max_total_exposure_pct=Decimal("0.9"),
        max_order_notional=Decimal("100000"),
        consecutive_loss_limit=100,
        max_correlated_positions=50,
    )
    gateway = StubGateway()
    execution = ExecutionEngine(
        gateway=gateway,  # type: ignore[arg-type]
        risk=RiskEngine(settings),
        portfolio=PortfolioManager(starting_equity=Decimal("10000")),
        settings=settings,
        mode=TradingMode.PAPER,
        instruments={BTC: Instrument(symbol=BTC)},
    )
    client = NullClient() if response is None else StubClient(response)
    return AITradingService(
        client=client,
        execution=execution,
        market=StubMarket(),
        symbols=(BTC,),
        settings=LLMSettings(),
        timeframe=Timeframe.H1,
    )


class TestService:
    """The loop must fail closed and must never bypass the risk engine."""

    @pytest.mark.asyncio
    async def test_the_null_client_produces_a_hold_and_no_order(self) -> None:
        # The default configuration must decline to trade rather than do something
        # arbitrary.
        service = build_service()
        outcome = await service.run_cycle()
        assert outcome.decision is not None
        assert outcome.decision.action is Action.HOLD
        assert not outcome.traded

    @pytest.mark.asyncio
    async def test_an_unparseable_response_becomes_a_hold(self) -> None:
        service = build_service("I think you should buy some bitcoin!")
        outcome = await service.run_cycle()
        assert outcome.error is not None
        assert not outcome.traded
        assert service.state.parse_failures == 1

    @pytest.mark.asyncio
    async def test_low_confidence_does_not_trade(self) -> None:
        service = build_service(
            '{"action":"BUY","symbol":"BTCUSDT","confidence":0.2,"reason":"hunch"}'
        )
        outcome = await service.run_cycle()
        assert not outcome.traded
        assert service.state.below_confidence == 1
        assert outcome.skipped_reason is not None
        assert "below the" in outcome.skipped_reason

    @pytest.mark.asyncio
    async def test_a_confident_buy_reaches_the_execution_layer(self) -> None:
        service = build_service(
            '{"action":"BUY","symbol":"BTCUSDT","confidence":0.9,"reason":"strong trend"}'
        )
        outcome = await service.run_cycle()
        # It went through execution, which means it went through the risk engine. Whether
        # risk approved it is a separate question and either answer is valid here.
        assert outcome.execution is not None
        assert service.state.decisions == 1

    @pytest.mark.asyncio
    async def test_the_service_defaults_to_paper(self) -> None:
        assert build_service().mode is TradingMode.PAPER

    @pytest.mark.asyncio
    async def test_live_is_not_permitted_without_the_env_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
        assert build_service().describe()["live_trading_permitted"] is False

    @pytest.mark.asyncio
    async def test_every_cycle_is_journalled(self) -> None:
        service = build_service()
        await service.run_cycle()
        assert len(service.journal) == 1
        record = service.journal.records[0]
        assert record.system_prompt
        assert record.user_prompt
        assert record.response

    @pytest.mark.asyncio
    async def test_the_service_exposes_no_direct_order_method(self) -> None:
        # The structural guarantee: the only route out is execute_signal, which is
        # risk-gated.
        public = {name for name in dir(build_service()) if not name.startswith("_")}
        assert not (public & {"place_order", "submit_order", "buy", "sell"})


class TestJournal:
    """PnL is written back on close, not guessed at decision time."""

    def _record(self) -> CycleRecord:
        return CycleRecord(
            started_at=NOW,
            model="stub",
            mode="paper",
            system_prompt="s",
            user_prompt="u",
            response="r",
            context={},
            outcome={"traded": True},
        )

    def test_pnl_starts_unknown_not_zero(self) -> None:
        # None and 0.0 mean very different things about a trade.
        record = self._record()
        assert record.realized_pnl is None
        assert record.to_dict()["realized_pnl"] is None

    def test_pnl_is_attached_to_the_originating_cycle(self) -> None:
        journal = DecisionJournal()
        journal.append(self._record())
        journal.note_order("order-1")
        assert journal.attach_pnl("order-1", Decimal("12.50"))
        assert journal.records[0].realized_pnl == Decimal("12.50")

    def test_attaching_to_an_unknown_order_reports_failure(self) -> None:
        assert not DecisionJournal().attach_pnl("missing", Decimal("1"))

    def test_the_in_memory_tail_is_bounded(self) -> None:
        journal = DecisionJournal(max_in_memory=3)
        for _ in range(10):
            journal.append(self._record())
        assert len(journal) == 3

    def test_the_summary_counts_only_settled_trades(self) -> None:
        journal = DecisionJournal()
        journal.append(self._record())
        summary = journal.summary()
        assert summary["cycles_that_traded"] == 1
        assert summary["settled_trades"] == 0
