"""The exchange boundary.

Everything above this line — strategies, risk, execution, backtesting — depends only on
:class:`ExchangeGateway`. That is what lets the *same* engine code run against Bybit, the
paper broker and the backtest simulator without a single conditional on trading mode — and
it is why replacing Binance with Bybit V5 changed no strategy, risk or execution code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, Ticker, Trade
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import Balance

#: Bybit V5 returns at most 1000 klines per request; the downloader paginates on this.
MAX_CANDLES_PER_REQUEST = 1000


@runtime_checkable
class MarketDataGateway(Protocol):
    """Read-only market data access."""

    async def load_instruments(self) -> dict[Symbol, Instrument]:
        """Fetch tradability rules for every symbol on the venue."""
        ...

    async def get_instrument(self, symbol: Symbol) -> Instrument:
        """Fetch tradability rules for one symbol.

        Raises:
            InvalidSymbolError: if the venue does not list the symbol.

        """
        ...

    async def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        since: datetime | None = None,
        limit: int = MAX_CANDLES_PER_REQUEST,
    ) -> list[Candle]:
        """Fetch OHLCV bars starting at ``since``, oldest first.

        The final bar may still be forming; callers that require closed bars must filter
        on :meth:`Candle.is_closed`.
        """
        ...

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        """Fetch the current best bid/ask and last price."""
        ...

    async def fetch_order_book(self, symbol: Symbol, *, depth: int = 20) -> OrderBook:
        """Fetch an L2 order-book snapshot."""
        ...

    async def fetch_recent_trades(self, symbol: Symbol, *, limit: int = 100) -> list[Trade]:
        """Fetch recent public trades."""
        ...

    async def server_time(self) -> datetime:
        """The venue's clock. Used to detect local clock drift before signing requests."""
        ...


@runtime_checkable
class TradingGateway(Protocol):
    """Order placement and account state."""

    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit an order and return it in its acknowledged state.

        Raises:
            OrderRejectedError: if the venue rejects the order.
            InsufficientFundsError: if the balance cannot support it.
            RateLimitError: if the venue rate-limits the request.

        """
        ...

    async def cancel_order(self, order_id: str, symbol: Symbol) -> Order:
        """Cancel a working order and return its resulting state."""
        ...

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:
        """Fetch the current state of one order."""
        ...

    async def fetch_open_orders(self, symbol: Symbol | None = None) -> list[Order]:
        """Fetch every order still working on the venue."""
        ...

    async def fetch_my_trades(
        self, symbol: Symbol, *, since: datetime | None = None, limit: int = 100
    ) -> list[Fill]:
        """Fetch our own executions — the source of truth for reconciliation."""
        ...

    async def fetch_balances(self) -> dict[str, Balance]:
        """Fetch account balances keyed by asset."""
        ...


@runtime_checkable
class StreamingGateway(Protocol):
    """Real-time market data over websockets."""

    def watch_candles(self, symbol: Symbol, timeframe: Timeframe) -> AsyncIterator[Candle]:
        """Stream candle updates, including in-progress bars."""
        ...

    def watch_ticker(self, symbol: Symbol) -> AsyncIterator[Ticker]:
        """Stream ticker updates."""
        ...

    def watch_trades(self, symbol: Symbol) -> AsyncIterator[Trade]:
        """Stream the public trade tape."""
        ...


@runtime_checkable
class ExchangeGateway(MarketDataGateway, TradingGateway, Protocol):
    """The full venue interface: market data plus trading."""

    @property
    def name(self) -> str:
        """Venue identifier, e.g. ``bybit``."""
        ...

    @property
    def is_testnet(self) -> bool:
        """Whether this gateway points at a sandbox rather than production."""
        ...

    @property
    def supports_trading(self) -> bool:
        """Whether credentials are configured for order placement."""
        ...

    async def connect(self) -> None:
        """Establish connections and load instrument metadata."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


class InstrumentCache:
    """In-memory instrument lookup shared by gateways.

    Trading rules change rarely but are needed on every order, so they are loaded once at
    connect time rather than fetched per request.
    """

    __slots__ = ("_instruments",)

    def __init__(self, instruments: dict[Symbol, Instrument] | None = None) -> None:
        self._instruments: dict[Symbol, Instrument] = dict(instruments or {})

    def put(self, instrument: Instrument) -> None:
        """Store or replace an instrument."""
        self._instruments[instrument.symbol] = instrument

    def put_many(self, instruments: Sequence[Instrument]) -> None:
        """Store or replace many instruments."""
        for instrument in instruments:
            self.put(instrument)

    def get(self, symbol: Symbol) -> Instrument | None:
        """Look up an instrument, or ``None``."""
        return self._instruments.get(symbol)

    def require(self, symbol: Symbol) -> Instrument:
        """Look up an instrument, raising if it is unknown.

        Raises:
            InvalidSymbolError: if the symbol has not been loaded.

        """
        from quantflow.core.errors import InvalidSymbolError

        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise InvalidSymbolError(
                f"instrument {symbol} is not loaded; call connect() first",
                symbol=str(symbol),
            )
        return instrument

    def all(self) -> dict[Symbol, Instrument]:
        """A copy of every loaded instrument."""
        return dict(self._instruments)

    def __len__(self) -> int:
        return len(self._instruments)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._instruments


def normalize_order(request: OrderRequest, instrument: Instrument) -> OrderRequest:
    """Snap an order request onto the venue's price and lot grids.

    Applied immediately before submission. Doing it here rather than in the strategy or the
    risk engine means those layers can work in clean numbers and never see a rejection for
    a precision violation.

    Raises:
        ValidationError: if the normalised order still violates a venue rule.

    """
    from dataclasses import replace

    from quantflow.domain.enums import OrderSide

    side_is_buy = request.side is OrderSide.BUY
    quantity = instrument.normalize_quantity(request.quantity)
    price = (
        instrument.normalize_price(request.price, side_is_buy=side_is_buy)
        if request.price is not None
        else None
    )
    trigger = (
        instrument.normalize_price(request.trigger_price, side_is_buy=side_is_buy)
        if request.trigger_price is not None
        else None
    )
    stop_loss = (
        instrument.normalize_price(request.stop_loss_price, side_is_buy=not side_is_buy)
        if request.stop_loss_price is not None
        else None
    )
    take_profit = (
        instrument.normalize_price(request.take_profit_price, side_is_buy=not side_is_buy)
        if request.take_profit_price is not None
        else None
    )

    return replace(
        request,
        quantity=quantity,
        price=price,
        trigger_price=trigger,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
    )


def estimate_fee(
    instrument: Instrument, quantity: Decimal, price: Decimal, *, is_maker: bool
) -> Decimal:
    """Estimate the fee on a fill, in quote currency."""
    return instrument.notional(quantity, price) * instrument.fee_rate(is_maker=is_maker)
