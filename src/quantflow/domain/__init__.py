"""Pure domain layer.

Value objects and invariants only — no IO, no framework imports, no globals. Everything here
is safe to construct in a test without a database, a network or a clock.
"""

from __future__ import annotations

from quantflow.domain.enums import (
    LiquidityRole,
    MarketRegime,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RunStatus,
    SignalDirection,
    Timeframe,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import (
    Candle,
    CandleSeries,
    DataIntegrityReport,
    OrderBook,
    OrderBookLevel,
    Ticker,
    Trade,
)
from quantflow.domain.orders import Fill, Order, OrderRequest, can_transition, new_client_order_id
from quantflow.domain.portfolio import Balance, EquityPoint, PortfolioSnapshot, build_equity_curve
from quantflow.domain.positions import ClosedTrade, Lot, Position
from quantflow.domain.signals import Signal

__all__ = [
    "Balance",
    "Candle",
    "CandleSeries",
    "ClosedTrade",
    "DataIntegrityReport",
    "EquityPoint",
    "Fill",
    "Instrument",
    "LiquidityRole",
    "Lot",
    "MarketRegime",
    "Order",
    "OrderBook",
    "OrderBookLevel",
    "OrderRequest",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PortfolioSnapshot",
    "Position",
    "PositionSide",
    "RunStatus",
    "Signal",
    "SignalDirection",
    "Symbol",
    "Ticker",
    "TimeInForce",
    "Timeframe",
    "Trade",
    "build_equity_curve",
    "can_transition",
    "new_client_order_id",
]
