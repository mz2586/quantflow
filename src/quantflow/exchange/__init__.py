"""Exchange integration: the gateway protocol, Binance connector and simulated venue."""

from __future__ import annotations

from quantflow.exchange.base import (
    ExchangeGateway,
    InstrumentCache,
    MarketDataGateway,
    StreamingGateway,
    TradingGateway,
    normalize_order,
)
from quantflow.exchange.ratelimit import RateLimiter, TokenBucket, retry_async
from quantflow.exchange.simulator import (
    FeeModel,
    FixedSlippage,
    SimulatedBroker,
    SlippageModel,
    SpreadSlippage,
    VolumeShareSlippage,
    match_against_candle,
)

__all__ = [
    "ExchangeGateway",
    "FeeModel",
    "FixedSlippage",
    "InstrumentCache",
    "MarketDataGateway",
    "RateLimiter",
    "SimulatedBroker",
    "SlippageModel",
    "SpreadSlippage",
    "StreamingGateway",
    "TokenBucket",
    "TradingGateway",
    "VolumeShareSlippage",
    "match_against_candle",
    "normalize_order",
    "retry_async",
]
