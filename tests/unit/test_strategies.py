"""Strategy contract, registry and the three reference strategies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.core.errors import InsufficientDataError, NotFoundError, StrategyError
from quantflow.domain.enums import OrderSide, SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import CandleSeries
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import Position
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.library import (
    DonchianBreakoutParams,
    DonchianBreakoutStrategy,
    EmaCrossParams,
    EmaCrossStrategy,
    RsiReversionParams,
    RsiReversionStrategy,
)
from quantflow.strategy.registry import StrategyRegistry, load_builtin_strategies
from tests.conftest import REFERENCE_TIME, make_candles


def make_context(
    symbol: Symbol,
    closes: Sequence[float | int],
    *,
    position: Position | None = None,
    cash: Decimal = Decimal("10000"),
) -> StrategyContext:
    candles = make_candles(symbol, [Decimal(str(value)) for value in closes])
    series = CandleSeries(candles)
    return StrategyContext(
        symbol=symbol,
        timeframe=Timeframe.H1,
        history=series,
        now=series.end + Timeframe.H1.delta,
        portfolio=PortfolioSnapshot(
            timestamp=series.end,
            base_currency="USDT",
            cash=cash,
            positions=(position,) if position else (),
            mark_prices={symbol: candles[-1].close},
        ),
        position=position,
    )


def open_long(symbol: Symbol, quantity: str = "1", price: str = "100") -> Position:
    position, _ = Position(symbol=symbol).apply_fill(
        Fill(
            fill_id="f",
            order_id="o",
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal(quantity),
            price=Decimal(price),
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp=REFERENCE_TIME,
        )
    )
    return position


class TestStrategyContract:
    def test_missing_strategy_id_is_rejected(self) -> None:
        class Nameless(Strategy):
            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:
                return context.hold("", "")

        with pytest.raises(StrategyError, match="must declare a strategy_id"):
            Nameless()

    def test_wrong_params_type_is_rejected(self) -> None:
        class Other(StrategyParams):
            pass

        with pytest.raises(StrategyError, match="expects"):
            EmaCrossStrategy(Other())

    def test_params_accept_a_dict(self) -> None:
        strategy = EmaCrossStrategy({"fast_period": 5, "slow_period": 20})
        assert strategy.params.fast_period == 5

    def test_unknown_parameters_are_rejected(self) -> None:
        # A typo'd parameter that silently uses the default is a whole class of
        # "why did the backtest change" debugging.
        with pytest.raises(PydanticValidationError):
            EmaCrossStrategy({"fastt_period": 5})

    def test_warmup_produces_hold_not_an_error(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy()
        signal = strategy.evaluate(make_context(btc, [100, 101, 102]))
        assert signal.direction is SignalDirection.HOLD
        assert "warming up" in signal.reason

    def test_a_raising_strategy_is_contained(self, btc: Symbol) -> None:
        # One misbehaving strategy must not abandon positions managed by the others.
        class Exploding(Strategy):
            strategy_id = "exploding"
            description = "always raises"

            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:
                raise RuntimeError("kaboom")

        signal = Exploding().evaluate(make_context(btc, [100, 101, 102]))
        assert signal.direction is SignalDirection.HOLD
        assert "kaboom" in signal.reason

    def test_insufficient_data_produces_hold(self, btc: Symbol) -> None:
        class Hungry(Strategy):
            strategy_id = "hungry"

            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:
                context.require_history(1000)
                return context.hold("unreachable", self.strategy_id)

        signal = Hungry().evaluate(make_context(btc, [100, 101, 102]))
        assert signal.direction is SignalDirection.HOLD
        assert "insufficient data" in signal.reason

    def test_misattributed_signal_is_rejected(self, btc: Symbol) -> None:
        class Liar(Strategy):
            strategy_id = "liar"

            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:
                return Signal.hold(context.symbol, context.now, "someone_else")

        with pytest.raises(StrategyError, match="attribution must match"):
            Liar().evaluate(make_context(btc, [100, 101]))

    def test_context_exposes_only_closed_history(self, btc: Symbol) -> None:
        # There is no field here that could hold a future price — that is the design.
        context = make_context(btc, [100, 101, 102])
        assert context.candle.close == Decimal("102")
        assert len(context.history) == 3
        assert context.index == 2
        assert context.now > context.candle.open_time

    def test_context_position_helpers(self, btc: Symbol) -> None:
        flat = make_context(btc, [100, 101])
        assert not flat.has_position
        assert not flat.is_long

        held = make_context(btc, [100, 101], position=open_long(btc))
        assert held.has_position
        assert held.is_long
        assert not held.is_short

    def test_require_history(self, btc: Symbol) -> None:
        context = make_context(btc, [100, 101])
        context.require_history(2)
        with pytest.raises(InsufficientDataError, match="need 10"):
            context.require_history(10)

    def test_describe(self) -> None:
        described = EmaCrossStrategy().describe()
        assert described["strategy_id"] == "ema_cross"
        assert "fast_period" in described["params"]


class TestRegistry:
    def test_builtins_self_register(self) -> None:
        registry = load_builtin_strategies()
        assert {"ema_cross", "rsi_reversion", "donchian_breakout"} <= set(registry.names())

    def test_create_by_name(self) -> None:
        strategy = load_builtin_strategies().create("ema_cross", {"fast_period": 8})
        assert isinstance(strategy, EmaCrossStrategy)
        assert strategy.params.fast_period == 8

    def test_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(NotFoundError, match="available:"):
            load_builtin_strategies().get("does_not_exist")

    def test_duplicate_id_is_rejected(self) -> None:
        registry = StrategyRegistry()

        class First(Strategy):
            strategy_id = "dupe"

            @property
            def warmup_bars(self) -> int:
                return 1

            def generate(self, context: StrategyContext) -> Signal:
                return context.hold("", self.strategy_id)

        class Second(First):
            strategy_id = "dupe"

        registry.register(First)
        with pytest.raises(StrategyError, match="already registered"):
            registry.register(Second)

    def test_reregistering_the_same_class_is_a_noop(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossStrategy)
        registry.register(EmaCrossStrategy)
        assert len(registry) == 1

    def test_json_schema_is_available_for_forms(self) -> None:
        schema = load_builtin_strategies().json_schema("ema_cross")
        assert "fast_period" in schema["properties"]

    def test_describe_all(self) -> None:
        described = load_builtin_strategies().describe_all()
        assert len(described) >= 3
        assert all("defaults" in entry for entry in described)


class TestEmaCross:
    def test_parameter_validation(self) -> None:
        with pytest.raises(PydanticValidationError, match="must be below"):
            EmaCrossParams(fast_period=30, slow_period=10)
        with pytest.raises(PydanticValidationError, match="atr_target_multiple"):
            EmaCrossParams(atr_stop_multiple=Decimal("5"), atr_target_multiple=Decimal("2"))

    def test_golden_cross_opens_a_long(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8, "atr_period": 3})
        # A long downtrend then a sharp reversal forces a genuine crossover.
        closes = [100 - index for index in range(40)] + [60 + index * 4 for index in range(20)]
        signals = [
            strategy.evaluate(make_context(btc, closes[: index + 1]))
            for index in range(strategy.warmup_bars, len(closes))
        ]
        assert any(signal.direction is SignalDirection.LONG for signal in signals)

    def test_entry_carries_a_stop_and_target(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8, "atr_period": 3})
        closes = [100 - index for index in range(40)] + [60 + index * 4 for index in range(20)]
        for index in range(strategy.warmup_bars, len(closes)):
            signal = strategy.evaluate(make_context(btc, closes[: index + 1]))
            if signal.direction is SignalDirection.LONG:
                assert signal.stop_loss_price is not None
                assert signal.stop_loss_price < signal.reference_price  # type: ignore[operator]
                assert signal.take_profit_price is not None
                assert signal.take_profit_price > signal.reference_price  # type: ignore[operator]
                return
        pytest.fail("no long entry was produced")

    def test_holds_while_long_without_a_reverse_cross(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8})
        closes = [100 + index * 2 for index in range(40)]
        signal = strategy.evaluate(make_context(btc, closes, position=open_long(btc, price="100")))
        assert signal.direction is SignalDirection.HOLD

    def test_death_cross_closes_a_long(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8, "atr_period": 3})
        closes = [100 + index * 3 for index in range(30)] + [190 - index * 8 for index in range(20)]
        found = False
        for index in range(strategy.warmup_bars, len(closes)):
            signal = strategy.evaluate(
                make_context(btc, closes[: index + 1], position=open_long(btc, price="100"))
            )
            if signal.direction is SignalDirection.CLOSE:
                found = True
                break
        assert found

    def test_shorts_are_disabled_by_default(self, btc: Symbol) -> None:
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8})
        closes = [100 + index * 3 for index in range(30)] + [190 - index * 8 for index in range(20)]
        signals = [
            strategy.evaluate(make_context(btc, closes[: index + 1]))
            for index in range(strategy.warmup_bars, len(closes))
        ]
        assert all(signal.direction is not SignalDirection.SHORT for signal in signals)

    def test_determinism(self, btc: Symbol) -> None:
        # A strategy that is not reproducible makes its own backtest meaningless.
        strategy = EmaCrossStrategy({"fast_period": 3, "slow_period": 8})
        closes = [100 + (index % 17) * 2 for index in range(80)]
        context = make_context(btc, closes)
        first = strategy.evaluate(context)
        second = strategy.evaluate(context)
        assert first.direction is second.direction
        assert first.reference_price == second.reference_price


class TestRsiReversion:
    def test_parameter_validation(self) -> None:
        with pytest.raises(PydanticValidationError, match="oversold must be below"):
            RsiReversionParams(oversold=Decimal("70"), overbought=Decimal("30"))
        with pytest.raises(PydanticValidationError, match="exit_level"):
            RsiReversionParams(exit_level=Decimal("95"))

    def test_trend_filter_blocks_buying_a_downtrend(self, btc: Symbol) -> None:
        # Mean reversion bought blindly in a downtrend is how a smooth equity curve
        # becomes a single catastrophic loss.
        strategy = RsiReversionStrategy(
            {"rsi_period": 5, "trend_period": 20, "use_trend_filter": True}
        )
        closes = [100 - index for index in range(60)]
        signals = [
            strategy.evaluate(make_context(btc, closes[: index + 1]))
            for index in range(strategy.warmup_bars, len(closes))
        ]
        assert all(signal.direction is not SignalDirection.LONG for signal in signals)
        assert any("trend filter" in signal.reason for signal in signals)

    def test_buys_a_dip_within_an_uptrend(self, btc: Symbol) -> None:
        strategy = RsiReversionStrategy(
            {"rsi_period": 5, "trend_period": 30, "use_trend_filter": True, "atr_period": 5}
        )
        # A long uptrend, then a dip sharp enough to reach oversold while price is still
        # comfortably above the trend line.
        closes = [100 + index * 3 for index in range(60)] + [
            277 - index * 8 for index in range(1, 6)
        ]
        signals = [
            strategy.evaluate(make_context(btc, closes[: index + 1]))
            for index in range(strategy.warmup_bars, len(closes))
        ]
        assert any(signal.direction is SignalDirection.LONG for signal in signals)

    def test_exits_at_the_midline(self, btc: Symbol) -> None:
        strategy = RsiReversionStrategy({"rsi_period": 5, "use_trend_filter": False})
        closes = [100 + index * 2 for index in range(40)]
        signal = strategy.evaluate(make_context(btc, closes, position=open_long(btc, price="100")))
        assert signal.direction is SignalDirection.CLOSE

    def test_conviction_scales_with_depth(self, btc: Symbol) -> None:
        strategy = RsiReversionStrategy(
            {"rsi_period": 5, "use_trend_filter": False, "atr_period": 5}
        )
        closes = [100] * 20 + [100 - index * 5 for index in range(1, 8)]
        for index in range(strategy.warmup_bars, len(closes)):
            signal = strategy.evaluate(make_context(btc, closes[: index + 1]))
            if signal.direction is SignalDirection.LONG:
                assert Decimal("0.25") <= signal.conviction <= Decimal("1")
                return
        pytest.fail("no long entry was produced")


class TestDonchianBreakout:
    def test_symmetric_channel_is_rejected(self) -> None:
        # A symmetric channel exits every trend the moment it starts.
        with pytest.raises(PydanticValidationError, match="must be below"):
            DonchianBreakoutParams(entry_period=20, exit_period=20)

    def test_breakout_opens_a_long(self, btc: Symbol) -> None:
        strategy = DonchianBreakoutStrategy({"entry_period": 10, "exit_period": 5, "atr_period": 5})
        closes = [100] * 20 + [150]
        signal = strategy.evaluate(make_context(btc, closes))
        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None

    def test_matching_the_prior_high_is_not_a_breakout(self, btc: Symbol) -> None:
        # The channel must be breached, not merely equalled. This is the boundary the
        # "exclude the current bar" rule exists to keep well-defined.
        strategy = DonchianBreakoutStrategy({"entry_period": 10, "exit_period": 5})
        equalled = [100] * 10 + [120] + [110] * 9 + [120]
        exceeded = [100] * 10 + [120] + [110] * 9 + [121]
        assert strategy.evaluate(make_context(btc, equalled)).direction is SignalDirection.HOLD
        assert strategy.evaluate(make_context(btc, exceeded)).direction is SignalDirection.LONG

    def test_channel_exit_closes_a_long(self, btc: Symbol) -> None:
        strategy = DonchianBreakoutStrategy({"entry_period": 10, "exit_period": 5})
        closes = [100] * 20 + [50]
        signal = strategy.evaluate(make_context(btc, closes, position=open_long(btc, price="100")))
        assert signal.direction is SignalDirection.CLOSE

    def test_inside_the_channel_holds(self, btc: Symbol) -> None:
        strategy = DonchianBreakoutStrategy({"entry_period": 10, "exit_period": 5})
        closes = [100 + (index % 3) for index in range(30)]
        signal = strategy.evaluate(make_context(btc, closes))
        assert signal.direction is SignalDirection.HOLD

    def test_buffer_suppresses_a_marginal_breakout(self, btc: Symbol) -> None:
        closes = [100] * 20 + [100.5]
        without = DonchianBreakoutStrategy({"entry_period": 10, "exit_period": 5})
        with_buffer = DonchianBreakoutStrategy(
            {"entry_period": 10, "exit_period": 5, "breakout_buffer_pct": Decimal("0.02")}
        )
        assert without.evaluate(make_context(btc, closes)).direction is SignalDirection.LONG
        assert with_buffer.evaluate(make_context(btc, closes)).direction is SignalDirection.HOLD


class TestAllStrategiesShareTheContract:
    @pytest.mark.parametrize("strategy_id", ["ema_cross", "rsi_reversion", "donchian_breakout"])
    def test_never_emits_an_unprotected_entry_when_atr_is_available(
        self, strategy_id: str, btc: Symbol
    ) -> None:
        strategy = load_builtin_strategies().create(strategy_id)
        closes = [100 + (index % 23) * 1.5 for index in range(400)]
        for index in range(strategy.warmup_bars, len(closes), 7):
            signal = strategy.evaluate(make_context(btc, closes[: index + 1]))
            if signal.is_entry:
                assert signal.stop_loss_price is not None, (
                    f"{strategy_id} produced an entry with no stop"
                )

    @pytest.mark.parametrize("strategy_id", ["ema_cross", "rsi_reversion", "donchian_breakout"])
    def test_signals_are_attributed_correctly(self, strategy_id: str, btc: Symbol) -> None:
        strategy = load_builtin_strategies().create(strategy_id)
        closes = [100 + (index % 11) for index in range(400)]
        signal = strategy.evaluate(make_context(btc, closes))
        assert signal.strategy_id == strategy_id

    @pytest.mark.parametrize("strategy_id", ["ema_cross", "rsi_reversion", "donchian_breakout"])
    def test_short_history_never_raises(self, strategy_id: str, btc: Symbol) -> None:
        strategy = load_builtin_strategies().create(strategy_id)
        for length in range(2, 12):
            signal = strategy.evaluate(make_context(btc, [100 + i for i in range(length)]))
            assert signal.direction is SignalDirection.HOLD

    @pytest.mark.parametrize("strategy_id", ["ema_cross", "rsi_reversion", "donchian_breakout"])
    def test_timestamps_come_from_the_context(self, strategy_id: str, btc: Symbol) -> None:
        # A strategy that read the wall clock would not be reproducible.
        strategy = load_builtin_strategies().create(strategy_id)
        context = make_context(btc, [100 + (index % 13) for index in range(400)])
        assert strategy.evaluate(context).timestamp == context.now
        assert context.now == REFERENCE_TIME + timedelta(hours=400)
