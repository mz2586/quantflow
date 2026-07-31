"""Paper-trading engine driven by a scripted feed.

The key assertion is that paper and backtest agree: given the same bars, the same
strategy and the same fill model, both engines must produce the same trades. If they
diverge, paper results say nothing about the backtest and vice versa.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.backtest.engine import BacktestConfig, BacktestEngine
from quantflow.core.clock import FrozenClock
from quantflow.core.config import TradingMode
from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import RunStatus, SignalDirection, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, Ticker, Trade
from quantflow.exchange.simulator import FeeModel, FixedSlippage
from quantflow.paper.engine import PaperConfig, PaperTradingEngine
from tests.conftest import REFERENCE_TIME
from tests.unit.test_backtest import (
    ScriptedStrategy,
    flat_candles,
    instrument,
    permissive_risk,
)


class HistoryGateway:
    """A market-data gateway that serves a fixed set of closed bars."""

    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles
        self.requests = 0

    async def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        del symbol, timeframe, since
        self.requests += 1
        return self.candles[-limit:]

    async def load_instruments(self) -> dict[Symbol, Instrument]:  # pragma: no cover
        return {}

    async def get_instrument(self, symbol: Symbol) -> Instrument:  # pragma: no cover
        return Instrument(symbol=symbol)

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:  # pragma: no cover
        raise NotImplementedError

    async def fetch_order_book(
        self, symbol: Symbol, *, depth: int = 20
    ) -> OrderBook:  # pragma: no cover
        raise NotImplementedError

    async def fetch_recent_trades(
        self, symbol: Symbol, *, limit: int = 100
    ) -> list[Trade]:  # pragma: no cover
        return []

    async def server_time(self) -> datetime:  # pragma: no cover
        return REFERENCE_TIME


async def feed_from(candles: Sequence[Candle]) -> AsyncIterator[Candle]:
    """Yield bars as a live feed would."""
    for candle in candles:
        yield candle


def paper_config(btc: Symbol, **overrides: object) -> PaperConfig:
    kwargs: dict[str, object] = {
        "symbols": (btc,),
        "timeframe": Timeframe.H1,
        "starting_equity": Decimal("10000"),
        "risk": permissive_risk(),
        "slippage": FixedSlippage(ZERO),
        "fees": FeeModel(maker_rate=ZERO, taker_rate=ZERO),
        "history_bars": 100,
        "persist": False,
    }
    kwargs.update(overrides)
    return PaperConfig(**kwargs)  # type: ignore[arg-type]


class TestPreparation:
    async def test_history_is_preloaded(self, btc: Symbol, clock: FrozenClock) -> None:
        # Without seeding, a 200-bar strategy would produce nothing for over a week.
        history = flat_candles(btc, ["100"] * 50)
        clock.set(history[-1].close_time)
        engine = PaperTradingEngine(
            ScriptedStrategy({}, warmup=1),
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        gateway = HistoryGateway(history)
        await engine.prepare(gateway)

        assert gateway.requests == 1
        assert engine.state.status is RunStatus.RUNNING
        assert engine.portfolio.mark_price(btc) == Decimal("100")

    async def test_forming_bars_are_excluded_from_history(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        # A forming bar's close is still moving; storing it would seed a fiction.
        history = flat_candles(btc, ["100"] * 10)
        clock.set(history[-1].open_time + timedelta(minutes=30))
        engine = PaperTradingEngine(
            ScriptedStrategy({}, warmup=1),
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        await engine.prepare(HistoryGateway(history))
        # The last bar has not closed, so only nine are retained.
        assert engine.portfolio.mark_price(btc) == Decimal("100")

    async def test_missing_instrument_is_rejected(self, btc: Symbol, clock: FrozenClock) -> None:
        engine = PaperTradingEngine(
            ScriptedStrategy({}), paper_config(btc), instruments={}, clock=clock
        )
        with pytest.raises(ValidationError, match="no instrument metadata"):
            await engine.prepare(HistoryGateway(flat_candles(btc, ["100"] * 10)))

    async def test_no_closed_history_is_rejected(self, btc: Symbol, clock: FrozenClock) -> None:
        engine = PaperTradingEngine(
            ScriptedStrategy({}),
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        with pytest.raises(MarketDataError, match="no closed history"):
            await engine.prepare(HistoryGateway([]))


class TestLiveLoop:
    async def _prepared(
        self, btc: Symbol, clock: FrozenClock, strategy: ScriptedStrategy, history: list[Candle]
    ) -> PaperTradingEngine:
        clock.set(history[-1].close_time)
        engine = PaperTradingEngine(
            strategy,
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        await engine.prepare(HistoryGateway(history))
        return engine

    async def test_a_full_session_trades_and_completes(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        history = flat_candles(btc, ["100"] * 10)
        live = [
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=10 + index),
                open=Decimal(price),
                high=Decimal(price),
                low=Decimal(price),
                close=Decimal(price),
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            )
            for index, price in enumerate(["100", "100", "110", "110", "110"])
        ]
        # History supplies 10 bars, so live bar 0 lands at index 10.
        strategy = ScriptedStrategy({10: SignalDirection.LONG, 12: SignalDirection.CLOSE}, warmup=1)
        engine = await self._prepared(btc, clock, strategy, history)

        state = await engine.run(feed_from(live))

        assert state.status is RunStatus.COMPLETED
        assert state.bars_seen == 5
        assert state.signals == 2
        assert state.orders == 2
        assert state.fills == 2
        assert len(engine.portfolio.closed_trades) == 1
        trade = engine.portfolio.closed_trades[0]
        assert trade.entry_price == Decimal("100")
        assert trade.exit_price == Decimal("110")

    async def test_out_of_order_bars_are_dropped(self, btc: Symbol, clock: FrozenClock) -> None:
        # A reconnect can replay a bar already processed; applying it twice would
        # double-count volume and re-run the strategy on stale state.
        history = flat_candles(btc, ["100"] * 10)
        engine = await self._prepared(btc, clock, ScriptedStrategy({}, warmup=1), history)
        replayed = history[-1]
        await engine.on_candle(replayed)
        assert engine.state.bars_seen == 0

    async def test_bars_for_unknown_symbols_are_ignored(
        self, btc: Symbol, eth: Symbol, clock: FrozenClock
    ) -> None:
        history = flat_candles(btc, ["100"] * 10)
        engine = await self._prepared(btc, clock, ScriptedStrategy({}, warmup=1), history)
        await engine.on_candle(flat_candles(eth, ["50"] * 1)[0])
        assert engine.state.bars_seen == 0

    async def test_a_stop_closes_the_position(self, btc: Symbol, clock: FrozenClock) -> None:
        history = flat_candles(btc, ["100"] * 10)
        live = [
            *flat_candles(btc, ["100"])[:1],
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=11),
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            ),
            # Dips to 80 intrabar, closes back at 100.
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=12),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("80"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            ),
        ]
        live[0] = Candle(
            symbol=btc,
            timeframe=Timeframe.H1,
            open_time=REFERENCE_TIME + timedelta(hours=10),
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("1000"),
            quote_volume=Decimal("100000"),
        )
        strategy = ScriptedStrategy({10: SignalDirection.LONG}, warmup=1, stop_pct=Decimal("0.1"))
        engine = await self._prepared(btc, clock, strategy, history)

        await engine.run(feed_from(live))

        assert len(engine.portfolio.closed_trades) == 1
        assert engine.portfolio.closed_trades[0].exit_price == Decimal("90")

    async def test_stop_flattens_when_asked(self, btc: Symbol, clock: FrozenClock) -> None:
        history = flat_candles(btc, ["100"] * 10)
        live = [
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=10 + index),
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            )
            for index in range(3)
        ]
        strategy = ScriptedStrategy({10: SignalDirection.LONG}, warmup=1)
        engine = await self._prepared(btc, clock, strategy, history)
        await engine.run(feed_from(live))
        assert engine.portfolio.position_for(btc) is not None

        fills = await engine.flatten_all(reason="test")
        assert len(fills) == 1
        assert engine.portfolio.position_for(btc) is None

    async def test_risk_rejections_are_counted(self, btc: Symbol, clock: FrozenClock) -> None:
        history = flat_candles(btc, ["100"] * 10)
        clock.set(history[-1].close_time)
        engine = PaperTradingEngine(
            ScriptedStrategy({10: SignalDirection.LONG}, warmup=1),
            paper_config(
                btc,
                starting_equity=Decimal("20"),
                risk=permissive_risk(min_order_notional=Decimal("500")),
            ),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        await engine.prepare(HistoryGateway(history))
        live = flat_candles(btc, ["100"] * 3)
        live = [
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=10 + index),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                quote_volume=candle.quote_volume,
            )
            for index, candle in enumerate(live)
        ]
        await engine.run(feed_from(live))
        assert engine.state.rejections >= 1
        assert engine.state.orders == 0

    async def test_snapshot_shape(self, btc: Symbol, clock: FrozenClock) -> None:
        history = flat_candles(btc, ["100"] * 10)
        engine = await self._prepared(btc, clock, ScriptedStrategy({}, warmup=1), history)
        snapshot = engine.snapshot()
        assert set(snapshot) == {"session", "portfolio", "positions", "risk"}
        assert snapshot["risk"]["kill_switch"]["engaged"] is False

    async def test_mode_is_always_paper(self, btc: Symbol, clock: FrozenClock) -> None:
        engine = PaperTradingEngine(
            ScriptedStrategy({}),
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        assert engine.mode is TradingMode.PAPER


class TestPaperMatchesBacktest:
    async def test_identical_bars_produce_identical_trades(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        """The property that makes paper results meaningful evidence.

        Same bars, same strategy, same fill model — the two engines must agree. Any
        divergence means one of them is wrong and neither can be trusted.
        """
        prices = [str(100 + (index % 11) * 2) for index in range(40)]
        candles = flat_candles(btc, prices)
        script = {5: SignalDirection.LONG, 15: SignalDirection.CLOSE, 25: SignalDirection.LONG}

        backtest = await BacktestEngine(
            ScriptedStrategy(dict(script), warmup=1),
            BacktestConfig(
                symbols=(btc,),
                timeframe=Timeframe.H1,
                starting_equity=Decimal("10000"),
                risk=permissive_risk(),
                slippage=FixedSlippage(ZERO),
                fees=FeeModel(maker_rate=ZERO, taker_rate=ZERO),
            ),
            {btc: instrument(btc)},
        ).run({btc: candles})

        # The paper engine seeds one bar of history then consumes the rest live, so the
        # scripted indices line up with the backtest's.
        clock.set(candles[0].close_time)
        paper = PaperTradingEngine(
            ScriptedStrategy(dict(script), warmup=1),
            paper_config(btc),
            instruments={btc: instrument(btc)},
            clock=clock,
        )
        await paper.prepare(HistoryGateway(candles[:1]))
        await paper.run(feed_from(candles[1:]))

        assert len(paper.portfolio.closed_trades) == len(backtest.closed_trades)
        for live_trade, historical in zip(
            paper.portfolio.closed_trades, backtest.closed_trades, strict=True
        ):
            assert live_trade.entry_price == historical.entry_price
            assert live_trade.exit_price == historical.exit_price
            assert live_trade.quantity == historical.quantity
            assert live_trade.gross_pnl == historical.gross_pnl

        assert paper.portfolio.equity() == backtest.final_equity
