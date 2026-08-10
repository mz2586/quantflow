"""Application configuration.

Settings are loaded from (in decreasing precedence): explicit constructor kwargs,
process environment, `.env` file, field defaults. Nested sections use a ``__``
delimiter, e.g. ``QF_DATABASE__HOST``.

Secrets use :class:`pydantic.SecretStr` so they cannot be leaked by an accidental
``repr()`` or by structured-log serialisation.
"""

from __future__ import annotations

import functools
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from quantflow.core.errors import ConfigurationError

#: Not a credential: an intent token the operator must set to arm live order flow.
LIVE_CONFIRMATION_TOKEN = "I_UNDERSTAND_THE_RISK"  # noqa: S105


def _empty_secret_to_none(value: Any) -> Any:
    """Treat a blank secret as absent.

    `.env` files carry empty placeholders (`QF_EXCHANGE__API_KEY=`), which Pydantic would
    otherwise turn into a present-but-empty SecretStr. The system would then believe it
    has credentials, attempt to sign requests with an empty key, and fail at the venue
    with an error that points nowhere near the real cause.
    """
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, SecretStr) and not value.get_secret_value().strip():
        return None
    return value


OptionalSecret = Annotated[SecretStr | None, BeforeValidator(_empty_secret_to_none)]

Fraction = Annotated[Decimal, Field(gt=Decimal("0"), le=Decimal("1"))]
PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production_like(self) -> bool:
        """Whether stricter validation and JSON logging should apply."""
        return self in (Environment.STAGING, Environment.PRODUCTION)


class TradingMode(StrEnum):
    """How order flow is handled."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class MarketType(StrEnum):
    """Exchange market segment."""

    SPOT = "spot"
    FUTURE = "future"


class Severity(StrEnum):
    """Notification severity, ordered."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric ordering for threshold comparisons."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.DEBUG: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}


class DatabaseSettings(BaseModel):
    """PostgreSQL connection settings."""

    model_config = {"frozen": True}

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "quantflow"
    password: SecretStr = SecretStr("quantflow")
    name: str = "quantflow"
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=200)
    pool_timeout_seconds: float = Field(default=30.0, gt=0)
    pool_recycle_seconds: int = Field(default=1800, ge=60)
    statement_timeout_ms: int = Field(default=30_000, ge=1_000)
    echo: bool = False

    def dsn(self, *, driver: str = "postgresql+asyncpg", hide_password: bool = False) -> str:
        """Build a SQLAlchemy DSN.

        Args:
            driver: SQLAlchemy dialect+driver string. Alembic needs a sync driver.
            hide_password: Replace the password with ``***`` for safe logging.

        """
        password = "***" if hide_password else self.password.get_secret_value()
        return f"{driver}://{self.user}:{password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_dsn(self) -> str:
        """DSN for the async application engine."""
        return self.dsn(driver="postgresql+asyncpg")

    @property
    def sync_dsn(self) -> str:
        """DSN for Alembic and other synchronous tooling."""
        return self.dsn(driver="postgresql+psycopg")

    @property
    def safe_dsn(self) -> str:
        """Password-redacted DSN, safe to log."""
        return self.dsn(hide_password=True)


class RedisSettings(BaseModel):
    """Redis connection settings."""

    model_config = {"frozen": True}

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: OptionalSecret = None
    max_connections: int = Field(default=50, ge=1)
    socket_timeout_seconds: float = Field(default=5.0, gt=0)
    key_prefix: str = "qf"

    @property
    def url(self) -> str:
        """Redis connection URL including credentials."""
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def safe_url(self) -> str:
        """Credential-redacted Redis URL, safe to log."""
        auth = ":***@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class ExchangeSettings(BaseModel):
    """Binance (or other CCXT venue) connection settings."""

    model_config = {"frozen": True}

    name: str = "binance"
    api_key: OptionalSecret = None
    api_secret: OptionalSecret = None
    testnet: bool = True
    #: Read public market data from production even when trading on testnet. Binance's
    #: testnet carries almost no history and its prices are synthetic, so backtesting or
    #: warming up a strategy on it produces results that mean nothing. Public endpoints
    #: need no credentials, so this costs nothing and is on by default.
    market_data_from_production: bool = True
    market_type: MarketType = MarketType.SPOT
    rate_limit_per_second: float = Field(default=10.0, gt=0, le=100)
    rate_limit_burst: int = Field(default=20, ge=1)
    request_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    max_retries: int = Field(default=4, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=0.5, gt=0)
    recv_window_ms: int = Field(default=5_000, ge=1_000, le=60_000)
    ws_reconnect_max_seconds: float = Field(default=60.0, gt=0)

    @property
    def has_credentials(self) -> bool:
        """Whether both an API key and secret are configured."""
        return self.api_key is not None and self.api_secret is not None

    @property
    def use_production_market_data(self) -> bool:
        """Whether public market data should come from the production venue."""
        return self.testnet and self.market_data_from_production


class TradingSettings(BaseModel):
    """Trading mode and account baseline."""

    model_config = {"frozen": True}

    mode: TradingMode = TradingMode.PAPER
    base_currency: str = Field(default="USDT", min_length=2, max_length=10)
    starting_equity: PositiveDecimal = Decimal("10000")
    live_confirmation: OptionalSecret = None
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    default_timeframe: str = "1h"

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def is_live_armed(self) -> bool:
        """True only when mode is live *and* the confirmation token matches."""
        if self.mode is not TradingMode.LIVE:
            return False
        token = self.live_confirmation
        return token is not None and token.get_secret_value() == LIVE_CONFIRMATION_TOKEN

    @model_validator(mode="after")
    def _validate_live(self) -> Self:
        if self.mode is TradingMode.LIVE and not self.is_live_armed:
            raise ValueError(
                "live mode requires QF_TRADING__LIVE_CONFIRMATION=" f"{LIVE_CONFIRMATION_TOKEN!r}"
            )
        return self


class RiskSettings(BaseModel):
    """Hard risk limits.

    Every field is mandatory by construction — defaults are conservative, never
    permissive — and the risk engine refuses to arm without them.
    """

    model_config = {"frozen": True}

    max_position_pct: Fraction = Decimal("0.10")
    max_total_exposure_pct: Annotated[Decimal, Field(gt=Decimal("0"), le=Decimal("10"))] = Decimal(
        "0.60"
    )
    max_concurrent_positions: int = Field(default=5, ge=1, le=100)
    max_daily_loss_pct: Fraction = Decimal("0.03")
    #: Rolling seven-day loss ceiling. A daily limit alone permits five consecutive days
    #: at 2.9% each — inside the daily rule every time, and a 14% hole in a week.
    max_weekly_loss_pct: Fraction = Decimal("0.08")
    max_drawdown_pct: Fraction = Decimal("0.15")
    default_stop_loss_pct: Fraction = Decimal("0.02")
    max_stop_loss_pct: Fraction = Decimal("0.20")
    max_leverage: Annotated[Decimal, Field(ge=Decimal("1"), le=Decimal("20"))] = Decimal("1")
    require_stop_loss: bool = True
    max_order_notional: PositiveDecimal = Decimal("5000")
    min_order_notional: PositiveDecimal = Decimal("10")
    max_orders_per_minute: int = Field(default=10, ge=1, le=600)
    max_slippage_pct: Fraction = Decimal("0.01")
    #: Absolute return correlation at or above which two positions count as one bet.
    correlation_threshold: Fraction = Decimal("0.80")
    #: How many mutually correlated positions may be held at once. Crypto is not a
    #: diversified universe; five alt positions are usually one BTC position in costume.
    max_correlated_positions: int = Field(default=2, ge=1, le=50)
    #: Consecutive losing trades that trigger a cooldown.
    consecutive_loss_limit: int = Field(default=4, ge=1, le=100)
    #: Minutes of no new entries after the limit is hit. A losing streak is usually the
    #: market telling you the regime changed; continuing to fire into it is how a bad day
    #: becomes a bad month.
    loss_cooldown_minutes: int = Field(default=240, ge=1, le=10_080)

    @model_validator(mode="after")
    def _validate_coherence(self) -> Self:
        if self.max_position_pct > self.max_total_exposure_pct:
            raise ValueError("max_position_pct cannot exceed max_total_exposure_pct")
        if self.default_stop_loss_pct > self.max_stop_loss_pct:
            raise ValueError("default_stop_loss_pct cannot exceed max_stop_loss_pct")
        if self.max_daily_loss_pct > self.max_weekly_loss_pct:
            raise ValueError("max_daily_loss_pct cannot exceed max_weekly_loss_pct")
        if self.max_weekly_loss_pct > self.max_drawdown_pct:
            raise ValueError("max_weekly_loss_pct cannot exceed max_drawdown_pct")
        if self.min_order_notional >= self.max_order_notional:
            raise ValueError("min_order_notional must be below max_order_notional")
        return self


class NotificationSettings(BaseModel):
    """Outbound alerting configuration."""

    model_config = {"frozen": True}

    telegram_enabled: bool = False
    telegram_bot_token: OptionalSecret = None
    telegram_chat_id: str | None = None
    telegram_timeout_seconds: float = Field(default=10.0, gt=0)
    min_severity: Severity = Severity.INFO
    rate_limit_per_minute: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _validate_telegram(self) -> Self:
        if self.telegram_enabled and not (self.telegram_bot_token and self.telegram_chat_id):
            raise ValueError("telegram_enabled requires telegram_bot_token and telegram_chat_id")
        return self


class AISettings(BaseModel):
    """AI/LLM provider configuration."""

    model_config = {"frozen": True}

    provider: Literal["anthropic", "none"] = "none"
    anthropic_api_key: OptionalSecret = None
    model: str = "claude-sonnet-5"
    max_tokens: int = Field(default=4096, ge=256, le=64_000)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=90.0, gt=0)
    news_provider: Literal["none", "cryptopanic"] = "none"
    news_api_key: OptionalSecret = None
    optuna_trials: int = Field(default=100, ge=1, le=10_000)
    optuna_seed: int = 42

    @property
    def llm_enabled(self) -> bool:
        """Whether an LLM provider is configured and usable."""
        return self.provider == "anthropic" and self.anthropic_api_key is not None

    @model_validator(mode="after")
    def _validate_provider(self) -> Self:
        if self.provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("provider='anthropic' requires ai.anthropic_api_key")
        if self.news_provider != "none" and self.news_api_key is None:
            raise ValueError(f"news_provider={self.news_provider!r} requires ai.news_api_key")
        return self


class StorageSettings(BaseModel):
    """Filesystem locations for datasets and generated reports."""

    model_config = {"frozen": True}

    data_dir: Path = Path("./data")
    report_dir: Path = Path("./reports")

    @property
    def candles_dir(self) -> Path:
        """Parquet dataset root for OHLCV data."""
        return self.data_dir / "candles"

    def ensure_directories(self) -> None:
        """Create the configured directories if they do not exist."""
        for path in (self.data_dir, self.report_dir, self.candles_dir):
            path.mkdir(parents=True, exist_ok=True)


class LLMSettings(BaseModel):
    """Configuration for the AI trading service's language-model client.

    Provider-agnostic on purpose: the service depends on a protocol, not a vendor, so a
    model can be swapped without touching the decision loop.
    """

    model_config = {"frozen": True}

    #: Which client to build. "null" returns a deterministic HOLD and needs no
    #: credentials - it is the default so that nothing accidentally makes paid API calls,
    #: and so the service is testable with no network at all.
    provider: Literal["null", "anthropic", "openai"] = "null"
    model: str = "claude-sonnet-5"
    api_key: OptionalSecret = None
    base_url: str | None = None
    #: Hard ceiling on response size. A model that rambles cannot be parsed anyway.
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    #: Zero by default. A trading decision that changes between identical inputs cannot
    #: be reasoned about, reproduced, or audited after a loss.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    #: Seconds between decision cycles.
    interval_seconds: float = Field(default=300.0, ge=5.0)
    #: Candles handed to the model per symbol. Enough for context, few enough that the
    #: prompt stays inside a sane budget.
    candles_in_prompt: int = Field(default=100, ge=20, le=500)
    #: Confidence below which the service holds regardless of the action returned.
    min_confidence: Decimal = Field(default=Decimal("0.65"), ge=0, le=1)

    @model_validator(mode="after")
    def _validate_credentials(self) -> Self:
        if self.provider != "null" and self.api_key is None:
            raise ValueError(f"provider {self.provider!r} requires QF_LLM__API_KEY")
        return self


class Settings(BaseSettings):
    """Root settings object. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="QF_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # --- application ---
    env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    app_name: str = "QuantFlow"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- api ---
    api_host: str = "0.0.0.0"  # noqa: S104 — bound inside a container network
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=32)
    api_prefix: str = "/api/v1"
    api_key: OptionalSecret = None
    secret_key: OptionalSecret = None
    # NoDecode: pydantic-settings would otherwise try to JSON-decode the env value
    # before our comma-splitting validator runs.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ("http://localhost:5173",)

    # --- sections ---
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ai: AISettings = Field(default_factory=AISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator("log_format")
    @classmethod
    def _force_json_in_prod(cls, value: str, info: ValidationInfo) -> str:
        env = info.data.get("env")
        if isinstance(env, Environment) and env.is_production_like:
            return "json"
        return value

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> Self:
        if not self.env.is_production_like:
            return self
        problems: list[str] = []
        if self.debug:
            problems.append("debug must be false")
        if self.api_key is None:
            problems.append("api_key is required")
        if self.secret_key is None:
            problems.append("secret_key is required")
        if self.database.password.get_secret_value() in ("quantflow", "postgres", ""):
            problems.append("database.password must not be a default value")
        if self.trading.mode is TradingMode.LIVE and self.exchange.testnet:
            problems.append("live trading cannot run against the exchange testnet")
        if problems:
            raise ValueError(f"invalid {self.env} configuration: {'; '.join(problems)}")
        return self

    @property
    def is_live(self) -> bool:
        """Whether live order submission is armed."""
        return self.trading.is_live_armed


@functools.cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Raises:
        ConfigurationError: if the environment does not validate.

    """
    try:
        return Settings()
    except Exception as exc:  # pragma: no cover - exercised via monkeypatched env
        raise ConfigurationError(f"failed to load settings: {exc}") from exc


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests only."""
    get_settings.cache_clear()
