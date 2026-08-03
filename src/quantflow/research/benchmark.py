"""Buy-and-hold, computed directly from the price series.

Running buy-and-hold *through the trading engine* does not produce buy-and-hold. The risk
engine refuses to let any entry go out unprotected — correctly, that is the platform's
most important invariant — so it attaches the configured default stop. In crypto that stop
is hit within days, the position re-enters, and the "benchmark" turns into a stopped-out
churn machine: measured on BTC it produced 27 trades, a 0% win rate and a **negative**
return over a period in which the asset itself rose. A benchmark that broken makes every
strategy look good, which is the exact opposite of what a benchmark is for.

So the benchmark is measured here instead of traded. Holding an asset is a property of the
*market*, not of a trading system: there is no stop, no re-entry and no risk engine,
because a person holding an asset has none of those things either. It still pays the full
entry and exit cost — fees and slippage on both legs — so the comparison against a strategy
that pays the same costs stays honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from quantflow.backtest.metrics import PerformanceMetrics, compute_metrics
from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide, PositionSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.domain.portfolio import EquityPoint, build_equity_curve
from quantflow.domain.positions import ClosedTrade
from quantflow.research.costs import CostModel

#: Reported as the benchmark's strategy id so it is unmistakable in a report.
BENCHMARK_LABEL = "buy_and_hold"


def buy_and_hold_metrics(
    symbol: Symbol,
    candles: Sequence[Candle],
    *,
    starting_equity: Decimal,
    timeframe: Timeframe,
    costs: CostModel,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Metrics for buying at the first bar and selling at the last.

    Marked to market on every bar, so the resulting Sharpe and drawdown describe the
    experience of actually holding — including the part where it falls 70% and the holder
    has to sit through it.

    Raises:
        ValidationError: if there are no candles, or the first bar has no usable price.

    """
    if not candles:
        raise ValidationError("cannot benchmark an empty series", field="candles")

    entry_reference = candles[0].open
    exit_reference = candles[-1].close
    if entry_reference <= ZERO:
        raise ValidationError("first bar has no usable price", field="candles")

    # Both legs cross the spread and pay the taker fee, exactly as a strategy's market
    # orders do. Buying and holding is cheap, not free.
    entry_price = costs.slippage.apply(
        reference_price=entry_reference,
        side=OrderSide.BUY,
        quantity=ZERO,
        candle=candles[0],
    )
    exit_price = costs.slippage.apply(
        reference_price=exit_reference,
        side=OrderSide.SELL,
        quantity=ZERO,
        candle=candles[-1],
    )
    taker = costs.fees.taker_rate if costs.fees.taker_rate is not None else Decimal("0.001")

    # Size so that the entry notional plus its fee consumes the whole starting balance:
    # a holder puts everything in, and sizing on notional alone would quietly assume a
    # small cash buffer that never gets invested.
    quantity = starting_equity / (entry_price * (Decimal("1") + taker))
    if quantity <= ZERO:
        raise ValidationError("starting equity is too small to buy any quantity", field="equity")

    entry_fee = entry_price * quantity * taker
    exit_fee = exit_price * quantity * taker
    cash_after_entry = starting_equity - entry_price * quantity - entry_fee

    points: list[EquityPoint] = []
    for candle in candles:
        points.append(
            EquityPoint(
                timestamp=candle.close_time,
                equity=cash_after_entry + candle.close * quantity,
                cash=cash_after_entry,
                position_count=1,
                unrealized_pnl=(candle.close - entry_price) * quantity,
            )
        )

    # The final point is the position liquidated, so the curve ends where a holder who
    # actually sold would end rather than at an unrealisable mark.
    final_equity = cash_after_entry + exit_price * quantity - exit_fee
    points[-1] = EquityPoint(
        timestamp=candles[-1].close_time,
        equity=final_equity,
        cash=final_equity,
        position_count=0,
        realized_pnl=final_equity - starting_equity,
    )

    trade = ClosedTrade(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=candles[0].close_time,
        exit_time=candles[-1].close_time,
        gross_pnl=(exit_price - entry_price) * quantity,
        fees=entry_fee + exit_fee,
        strategy_id=BENCHMARK_LABEL,
    )

    return compute_metrics(
        curve=build_equity_curve(points),
        trades=(trade,),
        starting_equity=starting_equity,
        timeframe=timeframe,
        total_fees=entry_fee + exit_fee,
        risk_free_rate=risk_free_rate,
    )


__all__ = ["BENCHMARK_LABEL", "buy_and_hold_metrics"]
