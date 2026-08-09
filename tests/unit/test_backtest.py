"""Backtest engine and metrics.

The two tests that matter most here are:

- :class:`TestAnalyticFixture` — a scripted strategy on a scripted price series where the
  exact PnL can be computed by hand. If the engine's arithmetic drifts, this fails with a
  specific number rather than a vague "looks different".
- :class:`TestNoLookAhead` — proves an order generated on bar *i* cannot fill before bar
  *i+1*. A backtester that fills on the signal bar manufactures returns that do not exist,
  and it does so invisibly.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    assert_no_lookahead,
    entry_and_exit_counts,
    rejection_reasons,
    signal_summary,
)
from quantflow.backtest.metrics import (
    cagr,
    compute_metrics,
    degradation_ratio,
    exposure,
    is_statistically_thin,
    max_drawdown,
    normalised_score,
    period_returns,
    sharpe,
    sortino,
    trade_statistics,
    turnover,
    volatility,
)
from quantflow.core.config import RiskSettings
from quantflow.core.errors import BacktestError, InsufficientDataError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import PositionSide, RunStatus, SignalDirection, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade
from quantflow.domain.signals import Signal
from quantflow.exchange.simulator import FeeModel, FixedSlippage
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from tests.conftest import REFERENCE_TIME


def instrument(symbol: Symbol) -> Instrument:
    return Instrument(
        symbol=symbol,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.00001"),
        min_quantity=Decimal("0.00001"),
        min_notional=Decimal("1"),
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
    )


def permissive_risk(**overrides: object) -> RiskSettings:
    kwargs: dict[str, object] = {
        "max_position_pct": Decimal("0.9"),
        "max_total_exposure_pct": Decimal("0.95"),
        "max_order_notional": Decimal("1000000"),
        "min_order_notional": Decimal("1"),
        "max_concurrent_positions": 10,
        "max_orders_per_minute": 600,
        "max_daily_loss_pct": Decimal("0.99"),
        # The capital-preservation rules are disabled here for the same reason as the
        # rest: a test of the engine must measure the engine, not the caps. Their own
        # behaviour is covered in test_risk_capital_preservation.py.
        "max_weekly_loss_pct": Decimal("0.99"),
        "max_drawdown_pct": Decimal("0.99"),
        "consecutive_loss_limit": 100,
        "max_correlated_positions": 50,
        "max_stop_loss_pct": Decimal("0.9"),
    }
    kwargs.update(overrides)
    return RiskSettings(**kwargs)  # type: ignore[arg-type]


class ScriptedStrategy(Strategy):
    """Emits a fixed sequence of directions, one per bar after warm-up.

    Removes the strategy from the equation entirely so a test measures only the engine.
    """

    strategy_id = "scripted"
    description = "emits a scripted sequence of signals"
    params_model = StrategyParams

    def __init__(
        self,
        script: dict[int, SignalDirection],
        *,
        warmup: int = 1,
        stop_pct: Decimal | None = Decimal("0.5"),
    ) -> None:
        super().__init__(None)
        self.script = script
        self._warmup = warmup
        self.stop_pct = stop_pct

    @property
    def warmup_bars(self) -> int:
        return self._warmup

    def generate(self, context: StrategyContext) -> Signal:
        direction = self.script.get(context.index)
        if direction is None or direction is SignalDirection.HOLD:
            return context.hold("not scripted", self.strategy_id)
        stop = (
            context.price * (Decimal("1") - self.stop_pct)
            if self.stop_pct is not None and direction is SignalDirection.LONG
            else None
        )
        return Signal(
            symbol=context.symbol,
            direction=direction,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            stop_loss_price=stop,
            reason="scripted",
        )


def flat_candles(symbol: Symbol, prices: list[str], *, volume: str = "1000") -> list[Candle]:
    """Candles whose OHLC are all the same price, so fills are perfectly predictable."""
    return [
        Candle(
            symbol=symbol,
            timeframe=Timeframe.H1,
            open_time=REFERENCE_TIME + timedelta(hours=index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            volume=Decimal(volume),
            quote_volume=Decimal(volume) * Decimal(price),
        )
        for index, price in enumerate(prices)
    ]


class TestAnalyticFixture:
    """A run whose PnL is computable by hand."""

    async def test_exact_pnl_of_a_single_round_trip(self, btc: Symbol) -> None:
        # Flat bars at 100, 100, 110, 110. Buy is signalled on bar 1 (price 100) and
        # therefore fills at bar 2's open (110)... which would be a loss-making entry, so
        # the script signals on bar 0 to fill at bar 1's open of 100.
        prices = ["100", "100", "110", "110"]
        candles = flat_candles(btc, prices)
        strategy = ScriptedStrategy(
            {0: SignalDirection.LONG, 2: SignalDirection.CLOSE},
            warmup=1,
            stop_pct=Decimal("0.5"),
        )
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            risk=permissive_risk(),
            slippage=FixedSlippage(ZERO),
            fees=FeeModel(maker_rate=ZERO, taker_rate=ZERO),
        )
        engine = BacktestEngine(strategy, config, {btc: instrument(btc)})
        result = await engine.run({btc: candles})

        assert result.succeeded
        assert len(result.closed_trades) == 1
        trade = result.closed_trades[0]
        # Entry fills at bar 1's open (100), exit fills at bar 3's open (110).
        assert trade.entry_price == Decimal("100")
        assert trade.exit_price == Decimal("110")
        assert trade.side is PositionSide.LONG
        # With zero fees and zero slippage, PnL is exactly quantity * 10.
        assert trade.gross_pnl == trade.quantity * Decimal("10")
        assert trade.fees == ZERO
        assert result.final_equity == Decimal("10000") + trade.gross_pnl

    async def test_fees_reduce_pnl_by_exactly_the_fee_rate(self, btc: Symbol) -> None:
        candles = flat_candles(btc, ["100", "100", "110", "110"])
        strategy = ScriptedStrategy({0: SignalDirection.LONG, 2: SignalDirection.CLOSE})
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            risk=permissive_risk(),
            slippage=FixedSlippage(ZERO),
            fees=FeeModel(maker_rate=Decimal("0.001"), taker_rate=Decimal("0.001")),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})

        trade = result.closed_trades[0]
        expected_fees = trade.quantity * Decimal("100") * Decimal(
            "0.001"
        ) + trade.quantity * Decimal("110") * Decimal("0.001")
        assert trade.fees == pytest.approx(expected_fees, abs=Decimal("0.0000001"))

    async def test_a_flat_market_leaves_equity_unchanged(self, btc: Symbol) -> None:
        candles = flat_candles(btc, ["100"] * 10)
        strategy = ScriptedStrategy({}, warmup=1)
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            risk=permissive_risk(),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        assert result.final_equity == Decimal("10000")
        assert result.closed_trades == ()


class TestNoLookAhead:
    async def test_an_order_never_fills_on_its_signal_bar(self, btc: Symbol) -> None:
        # Bar 2 jumps to 200. If the engine filled on the signal bar, the entry would be
        # at 100 and the run would show a fictional profit.
        candles = flat_candles(btc, ["100", "100", "200", "200"])
        strategy = ScriptedStrategy({1: SignalDirection.LONG}, warmup=1)
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            risk=permissive_risk(),
            slippage=FixedSlippage(ZERO),
            fees=FeeModel(maker_rate=ZERO, taker_rate=ZERO),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.fills
        # Signalled on bar 1 (close 100), filled at bar 2's open of 200.
        assert order.fills[0].price == Decimal("200")

    async def test_every_fill_follows_its_signal(self, btc: Symbol) -> None:
        candles = flat_candles(btc, [str(100 + index) for index in range(30)])
        strategy = ScriptedStrategy(
            dict.fromkeys((2, 10), SignalDirection.LONG)
            | dict.fromkeys((6, 14), SignalDirection.CLOSE),
            warmup=1,
        )
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            risk=permissive_risk(),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        # The regression guard itself.
        assert_no_lookahead(result)

    async def test_the_lookahead_guard_actually_catches_a_violation(self, btc: Symbol) -> None:
        # Prove the guard is not vacuous: hand it a deliberately corrupted result.
        from dataclasses import replace

        candles = flat_candles(btc, ["100"] * 10)
        strategy = ScriptedStrategy({1: SignalDirection.LONG}, warmup=1)
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        assert result.orders
        assert result.orders[0].fills

        order = result.orders[0]
        corrupted_fill = replace(order.fills[0], timestamp=REFERENCE_TIME - timedelta(days=1))
        corrupted = replace(order, fills=(corrupted_fill,))
        broken = replace(result, orders=(corrupted,))

        with pytest.raises(BacktestError, match="look-ahead detected"):
            assert_no_lookahead(broken)


class TestRiskIntegration:
    async def test_signals_are_routed_through_the_risk_engine(self, btc: Symbol) -> None:
        # A strategy emitting entries with no stop must still be blocked, exactly as live.
        candles = flat_candles(btc, ["100"] * 20)
        strategy = ScriptedStrategy(
            dict.fromkeys(range(1, 10), SignalDirection.LONG),
            warmup=1,
            stop_pct=None,
        )
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            risk=permissive_risk(default_stop_loss_pct=Decimal("0.02")),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        # The risk engine attaches its default stop rather than refusing outright.
        for order in result.orders:
            if not order.reduce_only:
                assert order.stop_loss_price is not None

    async def test_a_blocking_risk_config_produces_no_orders(self, btc: Symbol) -> None:
        # An account too small to clear the venue minimum: every signal sizes to zero.
        candles = flat_candles(btc, ["100"] * 20)
        strategy = ScriptedStrategy(dict.fromkeys(range(1, 10), SignalDirection.LONG), warmup=1)
        config = BacktestConfig(
            symbols=(btc,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("20"),
            risk=permissive_risk(min_order_notional=Decimal("500")),
        )
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        assert result.orders == ()
        assert result.rejected_signals
        # A run with signals but no orders is a misconfiguration; make it diagnosable.
        assert rejection_reasons(result)
        assert signal_summary(result)["orders"] == 0

    async def test_a_stop_closes_the_position_intrabar(self, btc: Symbol) -> None:
        symbol = btc
        candles = [
            *flat_candles(symbol, ["100", "100"]),
            # A bar that dips to 80 intrabar but closes back at 100.
            Candle(
                symbol=symbol,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + timedelta(hours=2),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("80"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            ),
            *[
                Candle(
                    symbol=symbol,
                    timeframe=Timeframe.H1,
                    open_time=REFERENCE_TIME + timedelta(hours=3 + offset),
                    open=Decimal("100"),
                    high=Decimal("100"),
                    low=Decimal("100"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                    quote_volume=Decimal("100000"),
                )
                for offset in range(2)
            ],
        ]
        strategy = ScriptedStrategy({0: SignalDirection.LONG}, warmup=1, stop_pct=Decimal("0.1"))
        config = BacktestConfig(
            symbols=(symbol,),
            timeframe=Timeframe.H1,
            risk=permissive_risk(),
            slippage=FixedSlippage(ZERO),
            fees=FeeModel(maker_rate=ZERO, taker_rate=ZERO),
        )
        result = await BacktestEngine(strategy, config, {symbol: instrument(symbol)}).run(
            {symbol: candles}
        )
        # Only checking the close would have missed this entirely.
        assert len(result.closed_trades) == 1
        assert result.closed_trades[0].exit_price == Decimal("90")


class TestValidation:
    async def test_empty_data_is_rejected(self, btc: Symbol) -> None:
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1)
        with pytest.raises(BacktestError, match="no market data"):
            await BacktestEngine(ScriptedStrategy({}), config, {btc: instrument(btc)}).run({})

    async def test_missing_instrument_is_rejected(self, btc: Symbol) -> None:
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1)
        with pytest.raises(BacktestError, match="no instrument metadata"):
            await BacktestEngine(ScriptedStrategy({}), config, {}).run(
                {btc: flat_candles(btc, ["100"] * 5)}
            )

    async def test_too_little_history_is_rejected(self, btc: Symbol) -> None:
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1)
        strategy = ScriptedStrategy({}, warmup=100)
        with pytest.raises(InsufficientDataError, match="needs"):
            await BacktestEngine(strategy, config, {btc: instrument(btc)}).run(
                {btc: flat_candles(btc, ["100"] * 5)}
            )

    async def test_mismatched_timeframe_is_rejected(self, btc: Symbol) -> None:
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.D1)
        with pytest.raises(BacktestError, match="not 1d"):
            await BacktestEngine(ScriptedStrategy({}), config, {btc: instrument(btc)}).run(
                {btc: flat_candles(btc, ["100"] * 5)}
            )

    async def test_a_raising_strategy_fails_the_run_not_the_process(self, btc: Symbol) -> None:
        class Exploding(ScriptedStrategy):
            def generate(self, context: StrategyContext) -> Signal:
                raise RuntimeError("kaboom")

        # The Strategy base class contains the error, so the run completes with holds.
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(Exploding({}), config, {btc: instrument(btc)}).run(
            {btc: flat_candles(btc, ["100"] * 10)}
        )
        assert result.status is RunStatus.COMPLETED
        assert result.orders == ()


class TestMultiSymbol:
    async def test_symbols_are_replayed_on_a_shared_timeline(
        self, btc: Symbol, eth: Symbol
    ) -> None:
        config = BacktestConfig(symbols=(btc, eth), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(
            ScriptedStrategy({2: SignalDirection.LONG}, warmup=1),
            config,
            {btc: instrument(btc), eth: instrument(eth)},
        ).run({btc: flat_candles(btc, ["100"] * 10), eth: flat_candles(eth, ["50"] * 10)})

        assert result.succeeded
        assert result.bars_processed == 10
        traded = {order.symbol for order in result.orders}
        assert traded == {btc, eth}

    async def test_a_gap_in_one_symbol_does_not_drop_the_others_bars(
        self, btc: Symbol, eth: Symbol
    ) -> None:
        # A union timeline, not an intersection: dropping a bar because an unrelated
        # symbol lacks it would silently change the other symbol's result.
        btc_candles = flat_candles(btc, ["100"] * 10)
        eth_candles = [
            candle
            for index, candle in enumerate(flat_candles(eth, ["50"] * 10))
            if index not in (4, 5)
        ]
        config = BacktestConfig(symbols=(btc, eth), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(
            ScriptedStrategy({}, warmup=1),
            config,
            {btc: instrument(btc), eth: instrument(eth)},
        ).run({btc: btc_candles, eth: eth_candles})
        assert result.bars_processed == 10


class TestDeterminism:
    async def test_the_same_inputs_produce_the_same_result(self, btc: Symbol) -> None:
        candles = flat_candles(btc, [str(100 + (index % 7)) for index in range(60)])
        script = dict.fromkeys((5, 20, 40), SignalDirection.LONG) | dict.fromkeys(
            (10, 30, 50), SignalDirection.CLOSE
        )

        async def run_once() -> Decimal:
            config = BacktestConfig(
                symbols=(btc,),
                timeframe=Timeframe.H1,
                risk=permissive_risk(),
                slippage=FixedSlippage(Decimal("0.0005")),
            )
            result = await BacktestEngine(
                ScriptedStrategy(script, warmup=1), config, {btc: instrument(btc)}
            ).run({btc: candles})
            return result.final_equity

        assert await run_once() == await run_once()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def curve(values: list[str], *, positions: int = 0) -> list[EquityPoint]:
    return [
        EquityPoint(
            timestamp=REFERENCE_TIME + timedelta(hours=index),
            equity=Decimal(value),
            cash=Decimal(value),
            position_count=positions,
        )
        for index, value in enumerate(values)
    ]


def trade(pnl: str, *, fees: str = "0", quantity: str = "1", entry: str = "100") -> ClosedTrade:
    return ClosedTrade(
        symbol=Symbol(base="BTC", quote="USDT"),
        side=PositionSide.LONG,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry),
        exit_price=Decimal(entry) + Decimal(pnl),
        entry_time=REFERENCE_TIME,
        exit_time=REFERENCE_TIME + timedelta(hours=4),
        gross_pnl=Decimal(pnl),
        fees=Decimal(fees),
    )


class TestMetrics:
    def test_period_returns(self) -> None:
        returns = period_returns(curve(["100", "110", "99"]))
        assert returns[0] == pytest.approx(0.1)
        assert returns[1] == pytest.approx(-0.1)

    def test_sharpe_of_a_constant_series_is_zero(self) -> None:
        # Zero variance means an undefined ratio; "infinite Sharpe" is a bug indicator.
        assert sharpe([0.01] * 20, periods_per_year=8760) == ZERO

    def test_sharpe_is_positive_for_a_rising_series(self) -> None:
        returns = [0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012]
        assert sharpe(returns, periods_per_year=8760) > ZERO

    def test_sharpe_uses_a_365_day_year(self) -> None:
        # A 252-day convention would overstate the annualised figure by ~20%.
        returns = [0.01, -0.005, 0.02, 0.001, -0.01]
        crypto = sharpe(returns, periods_per_year=365)
        equities = sharpe(returns, periods_per_year=252)
        assert crypto != equities

    def test_sortino_ignores_upside_volatility(self) -> None:
        steady = [0.01, 0.01, -0.01, 0.01, 0.01, -0.01]
        spiky = [0.01, 0.30, -0.01, 0.01, 0.25, -0.01]
        assert sortino(spiky, periods_per_year=365) > sortino(steady, periods_per_year=365)

    def test_sortino_without_losses_is_zero(self) -> None:
        assert sortino([0.01, 0.02, 0.03], periods_per_year=365) == ZERO

    def test_volatility(self) -> None:
        assert volatility([0.0] * 10, periods_per_year=365) == ZERO
        assert volatility([0.01, -0.01, 0.02, -0.02], periods_per_year=365) > ZERO

    def test_max_drawdown_measures_peak_to_trough(self) -> None:
        drawdown, _ = max_drawdown(curve(["100", "120", "90", "110"]))
        assert drawdown == Decimal("30") / Decimal("120")

    def test_max_drawdown_of_a_rising_curve_is_zero(self) -> None:
        drawdown, _ = max_drawdown(curve(["100", "110", "120"]))
        assert drawdown == ZERO

    def test_unrecovered_drawdown_duration_runs_to_the_end(self) -> None:
        # The number that actually matters to whoever has to sit through it.
        _, duration = max_drawdown(curve(["100", "120", "90", "85", "80"]))
        assert duration > ZERO

    def test_cagr_of_a_short_run_is_the_simple_return(self) -> None:
        # Annualising a three-day result produces a number that looks like a forecast.
        result = cagr(Decimal("100"), Decimal("110"), duration_days=Decimal("3"))
        assert result == Decimal("0.1")

    def test_cagr_annualises_a_long_run(self) -> None:
        result = cagr(Decimal("100"), Decimal("200"), duration_days=Decimal("365"))
        assert result == pytest.approx(Decimal("1"), abs=Decimal("0.01"))

    def test_cagr_of_a_wiped_out_account(self) -> None:
        assert cagr(Decimal("100"), ZERO, duration_days=Decimal("100")) == Decimal("-1")

    def test_trade_statistics(self) -> None:
        stats = trade_statistics([trade("10"), trade("20"), trade("-5"), trade("-5")])
        assert stats["trade_count"] == 4
        assert stats["win_count"] == 2
        assert stats["win_rate"] == Decimal("0.5")
        assert stats["profit_factor"] == Decimal("30") / Decimal("10")
        assert stats["expectancy"] == Decimal("5")
        assert stats["largest_win"] == Decimal("20")
        assert stats["largest_loss"] == Decimal("-5")

    def test_profit_factor_without_losses_reports_gross_profit(self) -> None:
        # Infinity does not serialise or sort; the gross profit does.
        stats = trade_statistics([trade("10"), trade("20")])
        assert stats["profit_factor"] == Decimal("30")

    def test_trade_statistics_of_an_empty_run(self) -> None:
        stats = trade_statistics([])
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == ZERO

    def test_exposure(self) -> None:
        mixed = curve(["100", "100"], positions=1) + curve(["100", "100"], positions=0)
        assert exposure(mixed) == Decimal("0.5")

    def test_turnover(self) -> None:
        assert turnover([trade("10", quantity="2", entry="100")], Decimal("1000")) == Decimal("0.2")

    def test_compute_metrics_assembles_everything(self) -> None:
        metrics = compute_metrics(
            curve=curve(["10000", "10500", "10200", "11000"], positions=1),
            trades=[trade("500"), trade("-300"), trade("800")],
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
            total_fees=Decimal("12.5"),
        )
        assert metrics.final_equity == Decimal("11000")
        assert metrics.total_return_pct == Decimal("0.1")
        assert metrics.trade_count == 3
        assert metrics.total_fees == Decimal("12.5")
        assert metrics.is_profitable
        assert "sharpe_ratio" in metrics.to_dict()
        assert len(metrics.summary_lines()) == 10

    def test_metrics_of_an_empty_run(self) -> None:
        metrics = compute_metrics(
            curve=[], trades=[], starting_equity=Decimal("10000"), timeframe=Timeframe.H1
        )
        assert metrics.final_equity == Decimal("10000")
        assert metrics.trade_count == 0
        assert metrics.sharpe_ratio == ZERO

    def test_thin_results_are_flagged(self) -> None:
        # A 90% win rate over 5 trades is noise; the optimiser must not chase it.
        rising = curve(["10000", "10200", "10150", "10400", "10600", "10550", "11000"])
        thin = compute_metrics(
            curve=rising,
            trades=[trade("100")] * 5,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
        )
        substantial = compute_metrics(
            curve=rising,
            trades=[trade("100")] * 50,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
        )
        assert is_statistically_thin(thin)
        assert not is_statistically_thin(substantial)
        # Identical curve, identical Sharpe — only the sample-size penalty differs.
        assert thin.sharpe_ratio == substantial.sharpe_ratio
        assert normalised_score(thin) < normalised_score(substantial)

    def test_normalised_score_of_a_run_with_no_trades_is_zero(self) -> None:
        metrics = compute_metrics(
            curve=curve(["10000"]),
            trades=[],
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
        )
        assert normalised_score(metrics) == ZERO

    def test_degradation_ratio(self) -> None:
        strong = compute_metrics(
            curve=curve(["10000", "10500", "11000", "11500"]),
            trades=[trade("100")] * 40,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
        )
        weak = compute_metrics(
            curve=curve(["10000", "9900", "9800", "9700"]),
            trades=[trade("-100")] * 40,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
        )
        # Out-of-sample far below in-sample is the classic overfitting signature.
        assert degradation_ratio(strong, weak) < Decimal("0.5")


class TestReporting:
    async def test_signal_summary_and_counts(self, btc: Symbol) -> None:
        candles = flat_candles(btc, [str(100 + index) for index in range(20)])
        strategy = ScriptedStrategy({2: SignalDirection.LONG, 8: SignalDirection.CLOSE}, warmup=1)
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})
        summary = signal_summary(result)
        assert summary["signals"] == 2
        entries, exits = entry_and_exit_counts(result)
        assert entries == 1
        assert exits == 1
