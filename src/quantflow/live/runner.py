"""Trading runners: the long-lived processes that actually trade.

Both paper and live sessions run through :class:`TradingRunner`. The **only** difference is
which gateway fills the orders — everything else (strategy, risk, portfolio, notifications,
persistence, shutdown) is identical code. That is deliberate: a live-only code path is a
path that has never been exercised until the moment real money is on it.

Live trading is disabled by default and requires **three** independent affirmations:

1. ``ENABLE_LIVE_TRADING=true`` in the environment,
2. ``QF_TRADING__MODE=live``,
3. ``QF_TRADING__LIVE_CONFIRMATION=I_UNDERSTAND_THE_RISK``.

Any one missing and the runner refuses to arm. Three separate gates is not paranoia: each
is easy to set accidentally in isolation (a copied `.env`, a stale shell export, a
container inheriting a variable), and requiring all three means an accident has to happen
three times in three different places.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from quantflow.cache.redis import Cache, EventBus
from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import Settings, TradingMode
from quantflow.core.errors import (
    ConfigurationError,
    LiveTradingNotArmedError,
    ValidationError,
)
from quantflow.core.logging import get_logger, log_context
from quantflow.domain.enums import RunStatus, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.exchange.bybit.ws import BybitStream
from quantflow.notifications.dispatcher import NotificationDispatcher, build_dispatcher
from quantflow.paper.engine import PaperConfig, PaperSessionState, PaperTradingEngine
from quantflow.persistence.database import Database
from quantflow.risk.engine import RiskEngine
from quantflow.strategy.base import Strategy
from quantflow.strategy.registry import load_builtin_strategies

logger = get_logger(__name__)

#: The environment variable that must be exactly "true" before live order submission is
#: even considered. Deliberately NOT prefixed with QF_ and NOT part of the Settings model:
#: it must be impossible to set it by editing a config file that something else copies.
LIVE_TRADING_ENV_VAR: Final = "ENABLE_LIVE_TRADING"

#: How long the runner waits for in-flight work during a graceful shutdown.
SHUTDOWN_GRACE_SECONDS: Final = 10.0


def live_trading_env_enabled() -> bool:
    """Whether ``ENABLE_LIVE_TRADING`` is set to exactly ``true``.

    Case-insensitive, whitespace-trimmed, but nothing else counts: ``1``, ``yes`` and
    ``TRUE-ish`` values are all rejected. A gate this consequential should require the
    operator to have typed the intended word.
    """
    return os.environ.get(LIVE_TRADING_ENV_VAR, "").strip().lower() == "true"


@dataclass(frozen=True, slots=True)
class LiveArmingCheck:
    """The result of evaluating every live-trading gate."""

    env_flag: bool
    mode_is_live: bool
    confirmation_token: bool
    has_credentials: bool
    not_testnet: bool

    @property
    def armed(self) -> bool:
        """Whether live order submission is permitted.

        Every gate must pass. Credentials and a production endpoint are included because
        an "armed" live session that cannot actually reach the venue is a worse state than
        a refusal: it looks like it is trading and is not.
        """
        return (
            self.env_flag
            and self.mode_is_live
            and self.confirmation_token
            and self.has_credentials
            and self.not_testnet
        )

    def blockers(self) -> list[str]:
        """Every reason live trading is not armed, in the order to fix them."""
        reasons: list[str] = []
        if not self.env_flag:
            reasons.append(f"{LIVE_TRADING_ENV_VAR} is not set to 'true'")
        if not self.mode_is_live:
            reasons.append("QF_TRADING__MODE is not 'live'")
        if not self.confirmation_token:
            reasons.append("QF_TRADING__LIVE_CONFIRMATION is missing or wrong")
        if not self.has_credentials:
            reasons.append("exchange API credentials are not configured")
        if not self.not_testnet:
            reasons.append("exchange is pointed at the testnet")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API and for logging."""
        return {
            "armed": self.armed,
            "env_flag": self.env_flag,
            "mode_is_live": self.mode_is_live,
            "confirmation_token": self.confirmation_token,
            "has_credentials": self.has_credentials,
            "not_testnet": self.not_testnet,
            "blockers": self.blockers(),
        }


def check_live_arming(settings: Settings) -> LiveArmingCheck:
    """Evaluate every live-trading gate without side effects."""
    return LiveArmingCheck(
        env_flag=live_trading_env_enabled(),
        mode_is_live=settings.trading.mode is TradingMode.LIVE,
        confirmation_token=settings.trading.is_live_armed,
        has_credentials=settings.exchange.has_credentials,
        not_testnet=not settings.exchange.testnet,
    )


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Configuration for one trading session."""

    strategy_id: str
    symbols: tuple[Symbol, ...]
    timeframe: Timeframe
    mode: TradingMode = TradingMode.PAPER
    starting_equity: Decimal = Decimal("10000")
    strategy_params: dict[str, Any] = field(default_factory=dict)
    history_bars: int = 500
    persist: bool = True
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not self.symbols:
            raise ValidationError("a trading session needs at least one symbol")
        if self.mode is TradingMode.BACKTEST:
            raise ValidationError("use the backtest engine for backtest mode")


class TradingRunner:
    """Owns one trading session end to end.

    Handles startup, the live feed, graceful shutdown on SIGINT/SIGTERM, and the live
    arming gate. Trading itself is delegated to :class:`PaperTradingEngine`, which is
    shared by both modes.
    """

    __slots__ = (
        "_cache",
        "_clock",
        "_config",
        "_database",
        "_dispatcher",
        "_engine",
        "_gateway",
        "_settings",
        "_stopping",
        "_strategy",
        "_stream",
    )

    def __init__(
        self,
        settings: Settings,
        config: RunnerConfig,
        *,
        clock: Clock | None = None,
        strategy: Strategy | None = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self._clock = clock or SystemClock()
        self._strategy = strategy or load_builtin_strategies().create(
            config.strategy_id, config.strategy_params
        )
        self._database: Database | None = None
        self._cache: Cache | None = None
        self._dispatcher: NotificationDispatcher | None = None
        self._gateway: BybitGateway | None = None
        self._stream: BybitStream | None = None
        self._engine: PaperTradingEngine | None = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Arming
    # ------------------------------------------------------------------ #
    def arming(self) -> LiveArmingCheck:
        """Evaluate the live-trading gates."""
        return check_live_arming(self._settings)

    def _assert_mode_permitted(self) -> None:
        """Refuse to start a live session unless every gate passes.

        Raises:
            LiveTradingNotArmedError: listing every unmet requirement, so the operator can
                fix them in one pass rather than discovering them one restart at a time.

        """
        if self._config.mode is not TradingMode.LIVE:
            return
        check = self.arming()
        if not check.armed:
            raise LiveTradingNotArmedError(
                "live trading is not armed: " + "; ".join(check.blockers()),
                blockers=check.blockers(),
            )
        logger.critical(
            "runner.live_trading_armed",
            session_id=self._config.session_id,
            symbols=[str(symbol) for symbol in self._config.symbols],
            strategy_id=self._config.strategy_id,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> PaperSessionState:
        """Run the session until the feed ends or a stop is requested."""
        self._assert_mode_permitted()

        with log_context(
            session_id=self._config.session_id,
            mode=self._config.mode.value,
            strategy_id=self._config.strategy_id,
        ):
            await self._build()
            assert self._engine is not None
            assert self._stream is not None

            await self._announce(started=True)
            try:
                state = await self._engine.run(self._feed())
            except asyncio.CancelledError:
                logger.info("runner.cancelled")
                raise
            finally:
                await self._announce(started=False)
                await self.aclose()
            return state

    async def _build(self) -> None:
        """Construct every dependency for the session."""
        settings = self._settings

        self._database = Database.from_settings(settings) if self._config.persist else None
        self._dispatcher = build_dispatcher(settings.notifications, clock=self._clock)

        cache: Cache | None = None
        event_bus: EventBus | None = None
        with contextlib.suppress(Exception):
            cache = Cache.from_settings(settings)
            if await cache.ping():
                event_bus = EventBus(cache)
            else:
                await cache.aclose()
                cache = None
        self._cache = cache

        self._gateway = BybitGateway(settings.exchange, clock=self._clock)
        await self._gateway.connect()

        instruments: dict[Symbol, Instrument] = {}
        for symbol in self._config.symbols:
            instruments[symbol] = await self._gateway.get_instrument(symbol)

        risk = RiskEngine(
            settings.risk,
            clock=self._clock,
            database=self._database,
            session_id=self._config.session_id,
            notifier=self._dispatcher,
        )
        await risk.start()
        if risk.kill_switch.engaged:
            raise ConfigurationError(
                "refusing to start: the kill switch is engaged "
                f"({risk.kill_switch.state.reason}). Clear it explicitly first."
            )

        self._engine = PaperTradingEngine(
            self._strategy,
            PaperConfig(
                symbols=self._config.symbols,
                timeframe=self._config.timeframe,
                starting_equity=self._config.starting_equity,
                base_currency=settings.trading.base_currency,
                risk=settings.risk,
                history_bars=self._config.history_bars,
                persist=self._config.persist,
                session_id=self._config.session_id,
            ),
            instruments=instruments,
            database=self._database,
            event_bus=event_bus,
            clock=self._clock,
        )
        # The engine builds its own risk engine; replace it with the one already started
        # and wired to notifications, so alerts and the loaded kill-switch state apply.
        self._engine._risk = risk

        await self._engine.prepare(self._gateway)
        self._stream = BybitStream(settings.exchange, clock=self._clock)

    async def _feed(self) -> AsyncIterator[Candle]:
        """Yield closed bars until a stop is requested."""
        assert self._stream is not None
        async for candle in self._stream.watch_many_candles(
            list(self._config.symbols), self._config.timeframe, closed_only=True
        ):
            if self._stopping.is_set():
                return
            yield candle

    async def stop(self, *, flatten: bool = False) -> None:
        """Request a graceful stop.

        ``flatten`` is **off** by default: closing every position on shutdown converts a
        planned restart into a set of real, fee-paying market orders, and an operator
        restarting a process rarely intends to liquidate the book.
        """
        self._stopping.set()
        if self._engine is not None:
            await self._engine.stop(flatten=flatten)
        logger.info("runner.stop_requested", flatten=flatten)

    async def _announce(self, *, started: bool) -> None:
        """Notify that the session started or stopped."""
        if self._dispatcher is None:
            return
        with contextlib.suppress(Exception):
            await self._dispatcher.notify_session(
                session_id=self._config.session_id,
                mode=self._config.mode.value,
                strategy_id=self._config.strategy_id,
                started=started,
            )

    async def aclose(self) -> None:
        """Release every resource, in reverse order of construction."""
        for closer in (
            self._gateway.aclose if self._gateway else None,
            self._cache.aclose if self._cache else None,
            self._dispatcher.aclose if self._dispatcher else None,
            self._database.aclose if self._database else None,
        ):
            if closer is None:
                continue
            with contextlib.suppress(Exception):
                await closer()
        logger.info("runner.closed", session_id=self._config.session_id)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> PaperTradingEngine | None:
        """The trading engine, once built."""
        return self._engine

    def snapshot(self) -> dict[str, Any]:
        """Session state for the API and dashboard."""
        if self._engine is None:
            return {
                "session_id": self._config.session_id,
                "mode": self._config.mode.value,
                "status": RunStatus.PENDING.value,
                "arming": self.arming().to_dict(),
            }
        return {
            **self._engine.snapshot(),
            "mode": self._config.mode.value,
            "arming": self.arming().to_dict(),
        }


async def run_session(
    settings: Settings, config: RunnerConfig, *, clock: Clock | None = None
) -> PaperSessionState:
    """Run one session with signal handling wired up."""
    runner = TradingRunner(settings, config, clock=clock)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(runner.stop()))

    return await runner.start()


def describe_arming(settings: Settings) -> str:
    """Human-readable live-arming status, for the CLI."""
    check = check_live_arming(settings)
    if check.armed:
        return "LIVE TRADING IS ARMED — real orders will be sent"
    lines = ["live trading is DISABLED. Unmet requirements:"]
    lines.extend(f"  - {reason}" for reason in check.blockers())
    return "\n".join(lines)


def utc_now(clock: Clock | None = None) -> datetime:
    """Current time from the given clock."""
    return (clock or SystemClock()).now()
