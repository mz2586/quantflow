"""ORM models.

Deliberately separate from the domain layer. The domain classes are frozen, validated value
objects; these are mutable persistence records. Mapping between them is explicit in the
repositories, which means a schema change cannot silently alter trading semantics — and the
domain stays importable without SQLAlchemy.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from quantflow.domain.enums import (
    LiquidityRole,
    MarketRegime,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RunStatus,
    Timeframe,
    TimeInForce,
)
from quantflow.persistence.base import Base, TimestampMixin, UuidPkMixin


def _enum(python_enum: type, name: str) -> SqlEnum:
    """Build a native Postgres enum that stores the *values*, not the member names."""
    return SqlEnum(
        python_enum,
        name=name,
        native_enum=True,
        values_callable=lambda enum_cls: [member.value for member in enum_cls],
        validate_strings=True,
    )


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class CandleRecord(Base):
    """An OHLCV bar.

    The natural key ``(symbol, timeframe, open_time)`` is the primary key: it makes
    re-downloading a range an idempotent ``ON CONFLICT DO UPDATE`` rather than a
    delete-then-insert that can lose data if it fails halfway.
    """

    __tablename__ = "candles"
    __table_args__ = (
        CheckConstraint("high >= low", name="high_ge_low"),
        CheckConstraint("low >= 0", name="low_non_negative"),
        CheckConstraint("volume >= 0", name="volume_non_negative"),
        CheckConstraint('"open" BETWEEN low AND high', name="open_within_range"),
        CheckConstraint('"close" BETWEEN low AND high', name="close_within_range"),
        Index("ix_candles_symbol_tf_time", "symbol", "timeframe", "open_time"),
    )

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timeframe: Mapped[Timeframe] = mapped_column(
        _enum(Timeframe, "timeframe_enum"), primary_key=True
    )
    open_time: Mapped[datetime] = mapped_column(primary_key=True)
    open: Mapped[Decimal] = mapped_column(nullable=False)
    high: Mapped[Decimal] = mapped_column(nullable=False)
    low: Mapped[Decimal] = mapped_column(nullable=False)
    close: Mapped[Decimal] = mapped_column(nullable=False)
    volume: Mapped[Decimal] = mapped_column(nullable=False)
    quote_volume: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class InstrumentRecord(Base, TimestampMixin):
    """Cached exchange trading rules for a symbol."""

    __tablename__ = "instruments"
    __table_args__ = (UniqueConstraint("exchange", "symbol", "market_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="binance")
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False, default="spot")
    base_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    price_tick: Mapped[Decimal] = mapped_column(nullable=False)
    quantity_step: Mapped[Decimal] = mapped_column(nullable=False)
    min_quantity: Mapped[Decimal] = mapped_column(nullable=False)
    max_quantity: Mapped[Decimal | None] = mapped_column(nullable=True)
    min_notional: Mapped[Decimal] = mapped_column(nullable=False)
    max_notional: Mapped[Decimal | None] = mapped_column(nullable=True)
    maker_fee: Mapped[Decimal] = mapped_column(nullable=False)
    taker_fee: Mapped[Decimal] = mapped_column(nullable=False)
    max_leverage: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("1"))
    contract_size: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("1"))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# --------------------------------------------------------------------------- #
# Trading
# --------------------------------------------------------------------------- #
class OrderRecord(Base, TimestampMixin):
    """A submitted order and its aggregate fill state."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("filled_quantity >= 0", name="filled_non_negative"),
        CheckConstraint("filled_quantity <= quantity", name="filled_within_quantity"),
        UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        Index("ix_orders_session_status", "session_id", "status"),
        Index("ix_orders_symbol_created", "symbol", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    venue_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="SET NULL"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "order_side_enum"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(
        _enum(OrderType, "order_type_enum"), nullable=False
    )
    status: Mapped[OrderStatus] = mapped_column(
        _enum(OrderStatus, "order_status_enum"), nullable=False, index=True
    )
    time_in_force: Mapped[TimeInForce] = mapped_column(
        _enum(TimeInForce, "time_in_force_enum"), nullable=False, default=TimeInForce.GTC
    )
    quantity: Mapped[Decimal] = mapped_column(nullable=False)
    price: Mapped[Decimal | None] = mapped_column(nullable=True)
    trigger_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    filled_quantity: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    average_fill_price: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    fees_paid: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    stop_loss_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    signal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(nullable=True)
    meta: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)

    fills: Mapped[list[FillRecord]] = relationship(
        back_populates="order", cascade="all, delete-orphan", lazy="selectin"
    )


class FillRecord(Base):
    """A single execution against an order."""

    __tablename__ = "fills"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("price > 0", name="price_positive"),
        CheckConstraint("fee >= 0", name="fee_non_negative"),
        UniqueConstraint("venue_fill_id", "order_id", name="uq_fills_venue_fill_id_order_id"),
        Index("ix_fills_order_time", "order_id", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    venue_fill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "order_side_enum"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(nullable=False)
    price: Mapped[Decimal] = mapped_column(nullable=False)
    fee: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    fee_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    role: Mapped[LiquidityRole] = mapped_column(
        _enum(LiquidityRole, "liquidity_role_enum"), nullable=False, default=LiquidityRole.TAKER
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)

    order: Mapped[OrderRecord] = relationship(back_populates="fills")


class PositionRecord(Base, TimestampMixin):
    """Current or historical exposure in a symbol."""

    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_session_symbol", "session_id", "symbol"),
        Index("ix_positions_open", "session_id", "closed_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[PositionSide] = mapped_column(
        _enum(PositionSide, "position_side_enum"), nullable=False
    )
    quantity: Mapped[Decimal] = mapped_column(nullable=False)
    average_entry_price: Mapped[Decimal] = mapped_column(nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    fees_paid: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    stop_loss_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    opened_at: Mapped[datetime | None] = mapped_column(nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    lots: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list)


class ClosedTradeRecord(Base):
    """A completed round-trip, the unit analytics and the AI journal consume."""

    __tablename__ = "closed_trades"
    __table_args__ = (
        Index("ix_closed_trades_session_exit", "session_id", "exit_time"),
        Index("ix_closed_trades_strategy_exit", "strategy_id", "exit_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[PositionSide] = mapped_column(
        _enum(PositionSide, "position_side_enum"), nullable=False
    )
    quantity: Mapped[Decimal] = mapped_column(nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(nullable=False)
    entry_time: Mapped[datetime] = mapped_column(nullable=False)
    exit_time: Mapped[datetime] = mapped_column(nullable=False, index=True)
    gross_pnl: Mapped[Decimal] = mapped_column(nullable=False)
    fees: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    net_pnl: Mapped[Decimal] = mapped_column(nullable=False)
    return_pct: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    holding_period_seconds: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    regime: Mapped[MarketRegime] = mapped_column(
        _enum(MarketRegime, "market_regime_enum"), nullable=False, default=MarketRegime.UNKNOWN
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class SignalRecord(Base):
    """A strategy signal, retained so live behaviour can be replayed and audited."""

    __tablename__ = "signals"
    __table_args__ = (Index("ix_signals_strategy_time", "strategy_id", "timestamp"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=True
    )
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    conviction: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)
    reference_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    stop_loss_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    regime: Mapped[MarketRegime] = mapped_column(
        _enum(MarketRegime, "market_regime_enum"), nullable=False, default=MarketRegime.UNKNOWN
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    acted_on: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejection_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)


# --------------------------------------------------------------------------- #
# Sessions, equity and risk
# --------------------------------------------------------------------------- #
class TradingSessionRecord(Base, TimestampMixin):
    """One run of the trading engine (backtest, paper or live)."""

    __tablename__ = "trading_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status_enum"), nullable=False, default=RunStatus.PENDING
    )
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    strategy_params: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    symbols: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    timeframe: Mapped[Timeframe] = mapped_column(_enum(Timeframe, "timeframe_enum"), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="USDT")
    starting_equity: Mapped[Decimal] = mapped_column(nullable=False)
    final_equity: Mapped[Decimal | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    risk_config: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class EquitySnapshotRecord(Base):
    """A sample on a session's equity curve."""

    __tablename__ = "equity_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "timestamp", name="uq_equity_snapshots_session_id_timestamp"
        ),
        Index("ix_equity_snapshots_session_time", "session_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    equity: Mapped[Decimal] = mapped_column(nullable=False)
    cash: Mapped[Decimal] = mapped_column(nullable=False)
    position_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gross_exposure: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    unrealized_pnl: Mapped[Decimal] = mapped_column(nullable=False, default=Decimal("0"))
    drawdown_pct: Mapped[Decimal] = mapped_column(
        Numeric(10, 8), nullable=False, default=Decimal("0")
    )


class RiskEventRecord(Base, TimestampMixin):
    """An audit record of every risk decision that blocked or halted trading.

    Persisted unconditionally: if the engine refuses an order, there must be a durable
    record of which rule fired and on what numbers.
    """

    __tablename__ = "risk_events"
    __table_args__ = (Index("ix_risk_events_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_sessions.id", ondelete="CASCADE"), nullable=True
    )
    rule: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warning")
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    observed_value: Mapped[Decimal | None] = mapped_column(nullable=True)
    limit_value: Mapped[Decimal | None] = mapped_column(nullable=True)
    blocked_order: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    halted_trading: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    context: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)


class KillSwitchRecord(Base, TimestampMixin):
    """Latched kill-switch state.

    A single row (``id = 1``). Persisting it means a restart cannot silently resume trading
    after an emergency halt — the operator has to clear it explicitly.
    """

    __tablename__ = "kill_switch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    engaged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    engaged_at: Mapped[datetime | None] = mapped_column(nullable=True)
    engaged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cleared_at: Mapped[datetime | None] = mapped_column(nullable=True)
    cleared_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


# --------------------------------------------------------------------------- #
# Backtests, optimisation and AI
# --------------------------------------------------------------------------- #
class BacktestRunRecord(Base, TimestampMixin, UuidPkMixin):
    """A completed backtest and its headline metrics."""

    __tablename__ = "backtest_runs"
    __table_args__ = (Index("ix_backtest_runs_strategy_created", "strategy_id", "created_at"),)

    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    strategy_params: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    symbols: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    timeframe: Mapped[Timeframe] = mapped_column(_enum(Timeframe, "timeframe_enum"), nullable=False)
    start: Mapped[datetime] = mapped_column(nullable=False)
    end: Mapped[datetime] = mapped_column(nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status_enum"), nullable=False, default=RunStatus.PENDING
    )
    starting_equity: Mapped[Decimal] = mapped_column(nullable=False)
    final_equity: Mapped[Decimal | None] = mapped_column(nullable=True)
    total_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    sortino_ratio: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 8), nullable=True)
    profit_factor: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    duration_seconds: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class OptimizationRunRecord(Base, TimestampMixin, UuidPkMixin):
    """An Optuna parameter search."""

    __tablename__ = "optimization_runs"

    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    objective: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status_enum"), nullable=False, default=RunStatus.PENDING
    )
    search_space: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    trials_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trials_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_params: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    best_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    in_sample_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    out_of_sample_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class RegimeObservationRecord(Base):
    """A market-regime classification produced by the AI engine."""

    __tablename__ = "regime_observations"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "timeframe",
            "timestamp",
            name="uq_regime_observations_symbol_timeframe_timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[Timeframe] = mapped_column(_enum(Timeframe, "timeframe_enum"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)
    regime: Mapped[MarketRegime] = mapped_column(
        _enum(MarketRegime, "market_regime_enum"), nullable=False
    )
    confidence: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    features: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")


class JournalEntryRecord(Base, TimestampMixin, UuidPkMixin):
    """An LLM-generated review of recent trading activity."""

    __tablename__ = "journal_entries"

    period_start: Mapped[datetime] = mapped_column(nullable=False)
    period_end: Mapped[datetime] = mapped_column(nullable=False, index=True)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    weaknesses: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class SentimentRecord(Base):
    """A news/social sentiment observation for a symbol."""

    __tablename__ = "sentiment_observations"
    __table_args__ = (Index("ix_sentiment_symbol_time", "symbol", "observed_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(7, 6), nullable=False)
    magnitude: Mapped[Decimal] = mapped_column(Numeric(7, 6), nullable=False, default=Decimal("0"))
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    headline: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
