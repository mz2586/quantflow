"""API request and response schemas.

Separate from the domain objects on purpose. The domain is free to change shape; the wire
contract is not, and coupling the two means a refactor silently breaks every client.

Money crosses the wire as **strings**, never as JSON numbers. A `Decimal` serialised as a
float loses precision the moment it reaches a JavaScript client, where `0.1 + 0.2` is
famously not `0.3` — and a dashboard that renders a position size wrong is worse than one
that renders nothing.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from quantflow.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RunStatus,
    SignalDirection,
    Timeframe,
)


class ApiModel(BaseModel):
    """Base for every schema: strict, immutable, and Decimal-safe on the wire."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    @field_serializer("*", when_used="json", check_fields=False)
    def _serialize_decimals(self, value: Any) -> Any:
        """Render Decimal as a string so no precision is lost in transit."""
        if isinstance(value, Decimal):
            return str(value)
        return value


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
class HealthResponse(ApiModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"
    version: str
    environment: str


class ComponentHealth(ApiModel):
    """One dependency's health."""

    name: str
    healthy: bool
    detail: str | None = None
    latency_ms: float | None = None


class ReadinessResponse(ApiModel):
    """Readiness response, including each dependency."""

    ready: bool
    components: tuple[ComponentHealth, ...]
    trading_mode: str
    kill_switch_engaged: bool


class ErrorDetail(ApiModel):
    """The error envelope every failure returns.

    A single shape for every error means a client writes one handler, not one per endpoint.
    """

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(ApiModel):
    """Top-level error body."""

    error: ErrorDetail


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class CandleResponse(ApiModel):
    """One OHLCV bar."""

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal = Decimal("0")
    trades: int = 0


class CandlesResponse(ApiModel):
    """A candle series with its integrity status."""

    symbol: str
    timeframe: Timeframe
    count: int
    candles: tuple[CandleResponse, ...]
    gaps: int = 0
    """Number of missing bars detected. Non-zero means the series is not contiguous and
    any metric computed from it should be treated with suspicion."""


class SymbolSummary(ApiModel):
    """A stored series and its coverage."""

    symbol: str
    timeframe: Timeframe
    bars: int
    start: datetime | None = None
    end: datetime | None = None


class TickerResponse(ApiModel):
    """Current best bid/ask."""

    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    spread_pct: Decimal


class InstrumentResponse(ApiModel):
    """A venue's trading rules for one symbol."""

    symbol: str
    market_type: str
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    min_notional: Decimal
    maker_fee: Decimal
    taker_fee: Decimal
    active: bool


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class StrategyDescription(ApiModel):
    """A registered strategy and its parameter schema."""

    strategy_id: str
    description: str
    warmup_bars: int
    defaults: dict[str, Any]
    #: `schema` shadows a BaseModel attribute, so the field is named `parameter_schema`
    #: internally and serialised as `schema` for the client.
    parameter_schema: dict[str, Any] = Field(serialization_alias="schema")


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
class PositionResponse(ApiModel):
    """An open position."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    average_entry_price: Decimal
    mark_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_pct: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    opened_at: datetime | None = None
    strategy_id: str | None = None


class PortfolioResponse(ApiModel):
    """Current portfolio state."""

    base_currency: str
    equity: Decimal
    cash: Decimal
    starting_equity: Decimal
    total_return_pct: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    gross_exposure: Decimal
    leverage: Decimal
    drawdown_pct: Decimal
    daily_pnl: Decimal
    position_count: int
    positions: tuple[PositionResponse, ...] = ()


class EquityPointResponse(ApiModel):
    """One sample on the equity curve."""

    timestamp: datetime
    equity: Decimal
    cash: Decimal
    drawdown_pct: Decimal
    position_count: int


class OrderResponse(ApiModel):
    """An order and its fill state."""

    order_id: str
    client_order_id: str
    venue_order_id: str | None = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    price: Decimal | None = None
    filled_quantity: Decimal
    average_fill_price: Decimal
    fees_paid: Decimal
    stop_loss_price: Decimal | None = None
    strategy_id: str | None = None
    created_at: datetime
    updated_at: datetime
    reject_reason: str | None = None


class TradeResponse(ApiModel):
    """A completed round-trip."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    return_pct: Decimal
    holding_hours: Decimal
    strategy_id: str | None = None


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
class RiskLimits(ApiModel):
    """The configured hard limits."""

    max_position_pct: Decimal
    max_total_exposure_pct: Decimal
    max_concurrent_positions: int
    max_daily_loss_pct: Decimal
    max_drawdown_pct: Decimal
    max_leverage: Decimal
    require_stop_loss: bool
    max_order_notional: Decimal
    max_orders_per_minute: int


class KillSwitchResponse(ApiModel):
    """Kill-switch state."""

    engaged: bool
    reason: str | None = None
    engaged_at: datetime | None = None
    engaged_by: str | None = None


class RiskStatusResponse(ApiModel):
    """Everything the risk panel needs."""

    trading_halted: bool
    kill_switch: KillSwitchResponse
    limits: RiskLimits
    headroom: dict[str, str]
    sizer: str
    rules: tuple[str, ...]
    #: Where the reported limits came from: the running engine, or this API process's own
    #: configuration when the engine has published nothing. Stated rather than implied,
    #: because the two have differed by a factor of ten and the panel gave no hint which
    #: it was showing.
    limits_source: str = "this API process"


class KillSwitchRequest(ApiModel):
    """Engage or clear the kill switch.

    ``reason`` is mandatory when engaging: a halt with no recorded cause is close to
    useless during the post-mortem that follows it.
    """

    engaged: bool
    reason: str | None = Field(default=None, max_length=500)
    actor: str = Field(default="operator", max_length=64)


class RiskEventResponse(ApiModel):
    """One entry from the risk audit trail."""

    rule: str
    severity: str
    message: str
    symbol: str | None = None
    observed_value: Decimal | None = None
    limit_value: Decimal | None = None
    blocked_order: bool
    halted_trading: bool
    created_at: datetime


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
class BacktestRequest(ApiModel):
    """Request to run a backtest."""

    strategy_id: str = Field(min_length=1, max_length=64)
    symbols: Annotated[list[str], Field(min_length=1, max_length=20)]
    timeframe: Timeframe = Timeframe.H1
    start: datetime
    end: datetime
    starting_equity: Decimal = Field(default=Decimal("10000"), gt=0)
    params: dict[str, Any] = Field(default_factory=dict)
    generate_report: bool = False


class BacktestMetricsResponse(ApiModel):
    """Headline metrics for a completed run."""

    starting_equity: Decimal
    final_equity: Decimal
    total_return_pct: Decimal
    cagr: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    trade_count: int
    win_rate: Decimal
    profit_factor: Decimal
    total_fees: Decimal
    exposure_pct: Decimal
    statistically_thin: bool
    """True when the run has too few trades for the metrics to mean anything."""


class BacktestResponse(ApiModel):
    """A backtest run and its results."""

    run_id: str
    status: RunStatus
    strategy_id: str
    symbols: tuple[str, ...]
    timeframe: Timeframe
    bars: int
    duration_seconds: float
    metrics: BacktestMetricsResponse | None = None
    signals: int = 0
    orders: int = 0
    rejections: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    report_path: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Sessions and signals
# --------------------------------------------------------------------------- #
class SessionResponse(ApiModel):
    """A trading session."""

    session_id: str
    mode: str
    status: RunStatus
    strategy_id: str
    symbols: tuple[str, ...]
    timeframe: Timeframe
    starting_equity: Decimal
    final_equity: Decimal | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class SignalResponse(ApiModel):
    """A strategy signal."""

    signal_id: str
    symbol: str
    direction: SignalDirection
    strategy_id: str
    conviction: Decimal
    timestamp: datetime
    reference_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    reason: str = ""


class PaginatedResponse(ApiModel):
    """Envelope for list endpoints."""

    total: int
    limit: int
    offset: int


class MessageResponse(ApiModel):
    """A simple acknowledgement."""

    message: str
