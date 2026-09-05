"""Forex execution — a transport-agnostic FX domain layer plus pluggable venue adapters.

.. warning::

   **EXPERIMENTAL. This package has never placed an order.**

   Nothing in ``quantflow.forex`` has been run against a live or a demo FX account. No FX
   credentials exist in this project. Every adapter is written to its venue's published
   API and tested against fakes; the first real order will be the first real order. It is
   not covered by the demo-venue validation the crypto path has, and it is excluded from
   the supported surface of the v0.1.0 release. Treat it as a design under review, not as
   working software.

FX is not crypto with different tickers. Size is quoted in lots against a venue-defined
contract size, the money value of a price move comes from a tick value rather than the
price itself, the market shuts at the weekend, and a held position keeps paying swap —
with a triple charge on one weekday. Every one of those differences is modelled here so
that no strategy has to rediscover it.

Layout:

* :mod:`~quantflow.forex.instruments`, :mod:`~quantflow.forex.sizing`,
  :mod:`~quantflow.forex.costs`, :mod:`~quantflow.forex.sessions`,
  :mod:`~quantflow.forex.exits` and :mod:`~quantflow.forex.plan` are the **domain layer**.
  They import no venue SDK and hold no connection.
* :mod:`~quantflow.forex.protocol` is the **interface** — :class:`ForexBroker`.
* :mod:`~quantflow.forex.mt5_worker` (Windows-only) and
  :mod:`~quantflow.forex.oanda_worker` (Linux-friendly REST) are **transports**. Both are
  optional; importing this package pulls in neither venue SDK.
"""

from __future__ import annotations

from quantflow.forex.costs import ForexCostModel, TradeCosts, expected_net_edge, swap_nights
from quantflow.forex.errors import (
    ForexCapabilityError,
    ForexConnectionError,
    ForexError,
    ForexOrderRejectedError,
    MarketClosedError,
    StaleMarketDataError,
)
from quantflow.forex.exits import IntrabarExit, IntrabarOutcome, evaluate_intrabar_exit
from quantflow.forex.instruments import MAJORS, ForexInstrument, TradeMode, prioritise_symbols
from quantflow.forex.plan import PlanRejection, TradeDirection, TradePlan, plan_trade
from quantflow.forex.protocol import (
    AccountInfo,
    ForexBar,
    ForexBroker,
    ForexFill,
    ForexOrder,
    ForexOrderRequest,
    ForexOrderStatus,
    ForexOrderType,
    ForexPosition,
    ForexTick,
    ForexTimeframe,
    OrderAck,
    ReconciliationReport,
    ensure_fresh,
    reconcile_positions,
)
from quantflow.forex.sessions import SessionClock, SessionWindow, TradingSession
from quantflow.forex.sizing import (
    LotSizingResult,
    SizingRejection,
    lots_for_risk,
    lots_for_risk_from_prices,
    stop_distance_points,
)

__all__ = [
    "MAJORS",
    "AccountInfo",
    "ForexBar",
    "ForexBroker",
    "ForexCapabilityError",
    "ForexConnectionError",
    "ForexCostModel",
    "ForexError",
    "ForexFill",
    "ForexInstrument",
    "ForexOrder",
    "ForexOrderRejectedError",
    "ForexOrderRequest",
    "ForexOrderStatus",
    "ForexOrderType",
    "ForexPosition",
    "ForexTick",
    "ForexTimeframe",
    "IntrabarExit",
    "IntrabarOutcome",
    "LotSizingResult",
    "MarketClosedError",
    "OrderAck",
    "PlanRejection",
    "ReconciliationReport",
    "SessionClock",
    "SessionWindow",
    "SizingRejection",
    "StaleMarketDataError",
    "TradeCosts",
    "TradeDirection",
    "TradeMode",
    "TradePlan",
    "TradingSession",
    "ensure_fresh",
    "evaluate_intrabar_exit",
    "expected_net_edge",
    "lots_for_risk",
    "lots_for_risk_from_prices",
    "plan_trade",
    "prioritise_symbols",
    "reconcile_positions",
    "stop_distance_points",
    "swap_nights",
]
