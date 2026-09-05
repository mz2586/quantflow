"""The LONG / SHORT / NO-TRADE decision.

Everything else in this package answers one question each — is the market open, how big
should this be, what will it cost. This module runs them in the order that makes a refusal
cheap and puts the answer in one auditable object.

The gates run before sizing on purpose: there is no point sizing a trade into a closed
market or a stale quote, and a rejected plan carries the specific gate that stopped it
rather than a bare ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide
from quantflow.forex.costs import ForexCostModel, TradeCosts
from quantflow.forex.errors import StaleMarketDataError
from quantflow.forex.instruments import ForexInstrument
from quantflow.forex.protocol import DEFAULT_MAX_TICK_AGE, ForexTick, ensure_fresh
from quantflow.forex.sessions import SessionClock, TradingSession
from quantflow.forex.sizing import LotSizingResult, lots_for_risk_from_prices


class TradeDirection(StrEnum):
    """The decision itself."""

    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class PlanRejection(StrEnum):
    """Which gate stopped the trade."""

    MARKET_CLOSED = "market_closed"
    WEEKLY_CLOSE_APPROACHING = "weekly_close_approaching"
    STALE_QUOTE = "stale_quote"
    SIZING_REJECTED = "sizing_rejected"
    NEGATIVE_NET_EDGE = "negative_net_edge"


@dataclass(frozen=True, slots=True)
class TradePlan:
    """A fully-costed trading decision, or an explained refusal."""

    direction: TradeDirection
    symbol: str
    session: TradingSession
    side: OrderSide | None = None
    lots: Decimal = ZERO
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    sizing: LotSizingResult | None = None
    costs: TradeCosts | None = None
    net_edge: Decimal | None = None
    reason: PlanRejection | None = None
    message: str = ""

    def __bool__(self) -> bool:
        """Truthy when a trade should actually be placed."""
        return self.direction is not TradeDirection.NO_TRADE


def plan_trade(
    *,
    instrument: ForexInstrument,
    side: OrderSide,
    entry_price: Decimal,
    stop_loss: Decimal,
    account_risk: Decimal,
    clock: SessionClock,
    now: datetime,
    cost_model: ForexCostModel | None = None,
    take_profit: Decimal | None = None,
    gross_edge: Decimal | None = None,
    tick: ForexTick | None = None,
    max_tick_age: timedelta = DEFAULT_MAX_TICK_AGE,
    block_before_weekly_close: bool = True,
    expected_close: datetime | None = None,
) -> TradePlan:
    """Decide whether to trade, in which direction and how big.

    Args:
        instrument: The symbol under consideration.
        side: The direction the strategy wants.
        entry_price: Intended entry.
        stop_loss: Protective stop. Must sit on the correct side of ``entry_price``.
        account_risk: Money the trade may lose, in account currency.
        clock: The session calendar to gate against.
        now: Decision time; must be timezone-aware.
        cost_model: Cost parameters. Costs are skipped when omitted.
        take_profit: Optional target, carried through to the plan.
        gross_edge: Expected gross profit. When given, a trade whose net edge after costs
            is not positive is refused.
        tick: Latest quote, checked for staleness when supplied.
        max_tick_age: Freshness budget for ``tick``.
        block_before_weekly_close: Refuse new positions in the run-up to the weekly close.
        expected_close: Anticipated exit time, used to price the swap.

    Returns:
        A :class:`TradePlan`. ``NO_TRADE`` plans always carry a ``reason``.

    """
    session = clock.classify(now)

    if not clock.is_open(now, instrument.sessions):
        return _no_trade(
            instrument,
            session,
            PlanRejection.MARKET_CLOSED,
            f"{instrument.symbol} is outside its trading session at {now.isoformat()}",
        )

    if block_before_weekly_close and clock.in_friday_close_window(now):
        return _no_trade(
            instrument,
            session,
            PlanRejection.WEEKLY_CLOSE_APPROACHING,
            f"within {clock.friday_close_buffer} of the weekly close; "
            "a new position would be carried over the weekend gap",
        )

    if tick is not None:
        try:
            ensure_fresh(tick, now, max_tick_age)
        except StaleMarketDataError as exc:
            return _no_trade(instrument, session, PlanRejection.STALE_QUOTE, exc.message)

    sizing = lots_for_risk_from_prices(account_risk, entry_price, stop_loss, instrument, side)
    if not sizing.accepted:
        return _no_trade(
            instrument,
            session,
            PlanRejection.SIZING_REJECTED,
            sizing.message or "sizing produced no tradable size",
            sizing=sizing,
        )

    costs: TradeCosts | None = None
    net_edge: Decimal | None = None
    if cost_model is not None:
        costs = cost_model.estimate(
            instrument, sizing.lots, side, opened_at=now, closed_at=expected_close
        )
        if gross_edge is not None:
            net_edge = costs.net_edge(gross_edge)
            if net_edge <= ZERO:
                return _no_trade(
                    instrument,
                    session,
                    PlanRejection.NEGATIVE_NET_EDGE,
                    f"gross edge {gross_edge} does not survive {costs.total} of costs",
                    sizing=sizing,
                    costs=costs,
                    net_edge=net_edge,
                )

    return TradePlan(
        direction=TradeDirection.LONG if side is OrderSide.BUY else TradeDirection.SHORT,
        symbol=instrument.symbol,
        session=session,
        side=side,
        lots=sizing.lots,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sizing=sizing,
        costs=costs,
        net_edge=net_edge,
    )


def _no_trade(
    instrument: ForexInstrument,
    session: TradingSession,
    reason: PlanRejection,
    message: str,
    *,
    sizing: LotSizingResult | None = None,
    costs: TradeCosts | None = None,
    net_edge: Decimal | None = None,
) -> TradePlan:
    """Build a refused plan carrying the gate that stopped it."""
    return TradePlan(
        direction=TradeDirection.NO_TRADE,
        symbol=instrument.symbol,
        session=session,
        sizing=sizing,
        costs=costs,
        net_edge=net_edge,
        reason=reason,
        message=message,
    )
