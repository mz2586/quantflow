"""The strategy contract.

A strategy is a **pure decision function**: bars and portfolio state in, signals out. It
cannot place orders, cannot size positions, and cannot touch the exchange. That is not a
stylistic preference — it is what makes it structurally impossible for a strategy to route
around the risk engine, and it is why the identical strategy object runs unmodified in
backtest, paper and live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from quantflow.core.errors import InsufficientDataError, StrategyError
from quantflow.core.logging import get_logger
from quantflow.domain.enums import MarketRegime, PositionSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import ClosedTrade, Position
from quantflow.domain.signals import Signal

logger = get_logger(__name__)


class StrategyParams(BaseModel):
    """Base class for a strategy's parameter schema.

    Pydantic rather than a plain dataclass so parameters validate at construction, render
    to JSON for the API and the optimiser, and reject unknown keys — a typo'd parameter
    name that silently uses the default is a whole class of "why did the backtest change"
    debugging.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for persistence and reporting."""
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to see at one decision point.

    ``history`` contains **only closed bars up to and including** the decision bar. The
    engine constructs it that way so a strategy physically cannot read a future price:
    there is no field here that holds one.
    """

    symbol: Symbol
    timeframe: Timeframe
    history: CandleSeries
    now: datetime
    portfolio: PortfolioSnapshot
    position: Position | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candle(self) -> Candle:
        """The bar the decision is being made on (the most recent closed bar)."""
        return self.history[-1]

    @property
    def price(self) -> Decimal:
        """Close of the decision bar."""
        return self.candle.close

    @property
    def index(self) -> int:
        """Index of the decision bar within ``history``."""
        return len(self.history) - 1

    @property
    def closes(self) -> tuple[Decimal, ...]:
        """Close prices of the visible history."""
        return self.history.closes()

    @property
    def candles(self) -> tuple[Candle, ...]:
        """The visible history."""
        return self.history.candles

    @property
    def has_position(self) -> bool:
        """Whether there is open exposure in this symbol."""
        return self.position is not None and not self.position.is_flat

    @property
    def position_side(self) -> PositionSide:
        """Direction of the open position, or ``FLAT``."""
        return self.position.side if self.position else PositionSide.FLAT

    @property
    def is_long(self) -> bool:
        """Whether the position is long."""
        return self.position_side is PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Whether the position is short."""
        return self.position_side is PositionSide.SHORT

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """Open PnL as a fraction of cost basis, or zero when flat."""
        if self.position is None or self.position.is_flat:
            return Decimal("0")
        return self.position.unrealized_pnl_pct(self.price)

    def require_history(self, bars: int) -> None:
        """Assert enough history exists for an indicator warm-up.

        Raises:
            InsufficientDataError: if fewer than ``bars`` candles are visible.

        """
        if len(self.history) < bars:
            raise InsufficientDataError(
                f"{self.symbol} has {len(self.history)} bars, need {bars}",
                symbol=str(self.symbol),
                available=len(self.history),
                required=bars,
            )

    def hold(self, reason: str, strategy_id: str) -> Signal:
        """Build a no-action signal for this context."""
        return Signal.hold(self.symbol, self.now, strategy_id, reason)


class Strategy(ABC):
    """Base class for all strategies.

    Subclasses implement :meth:`generate` and declare ``params_model`` and
    ``warmup_bars``. Everything else — parameter validation, warm-up enforcement, error
    containment — is handled here so every strategy behaves consistently.
    """

    #: Stable identifier used in persistence, the registry and reports.
    strategy_id: ClassVar[str] = ""
    #: Human-readable description, surfaced in the API and dashboard.
    description: ClassVar[str] = ""
    #: Parameter schema.
    params_model: ClassVar[type[StrategyParams]] = StrategyParams

    def __init__(self, params: StrategyParams | dict[str, Any] | None = None) -> None:
        if not self.strategy_id:
            raise StrategyError(f"{type(self).__name__} must declare a strategy_id")
        if isinstance(params, dict):
            self.params = self.params_model(**params)
        elif params is None:
            self.params = self.params_model()
        elif not isinstance(params, self.params_model):
            raise StrategyError(
                f"{self.strategy_id} expects {self.params_model.__name__}, "
                f"got {type(params).__name__}"
            )
        else:
            self.params = params

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Bars of history required before the strategy can decide anything.

        The engine withholds the strategy entirely until this many closed bars exist, so
        an indicator never reads a half-formed value.
        """

    @abstractmethod
    def generate(self, context: StrategyContext) -> Signal:
        """Decide what to do on this bar.

        Must be **pure and deterministic**: the same context must always produce the same
        signal. Any IO, randomness or wall-clock read here breaks reproducibility and makes
        a backtest result meaningless as evidence about live behaviour.
        """

    # ------------------------------------------------------------------ #
    # Engine entry point
    # ------------------------------------------------------------------ #
    def evaluate(self, context: StrategyContext) -> Signal:
        """Run the strategy with warm-up and error containment.

        A strategy that raises produces a HOLD rather than taking the engine down: one
        misbehaving strategy must not abandon open positions managed by the others.
        """
        if len(context.history) < self.warmup_bars:
            return context.hold(
                f"warming up ({len(context.history)}/{self.warmup_bars} bars)",
                self.strategy_id,
            )
        try:
            signal = self.generate(context)
        except InsufficientDataError as exc:
            return context.hold(f"insufficient data: {exc.message}", self.strategy_id)
        except Exception as exc:
            logger.exception(
                "strategy.failed",
                strategy_id=self.strategy_id,
                symbol=str(context.symbol),
                at=context.now.isoformat(),
                error=str(exc),
            )
            return context.hold(f"strategy error: {exc}", self.strategy_id)

        if signal.strategy_id != self.strategy_id:
            raise StrategyError(
                f"{self.strategy_id} emitted a signal attributed to "
                f"{signal.strategy_id!r}; attribution must match",
                strategy_id=self.strategy_id,
            )
        return signal

    # ------------------------------------------------------------------ #
    # Optional hooks
    # ------------------------------------------------------------------ #
    def on_start(self, symbols: Sequence[Symbol]) -> None:  # noqa: B027 - optional hook
        """Called once before the first bar. Default: no-op."""

    def on_fill(self, symbol: Symbol, position: Position) -> None:  # noqa: B027 - optional hook
        """Called after a fill updates a position. Default: no-op."""

    def on_trade_closed(self, trade: ClosedTrade) -> None:  # noqa: B027 - optional hook
        """Called when a round-trip completes. Default: no-op.

        A composite strategy uses this to keep a realised record of its members; a plain
        strategy has no use for it and ignores it.
        """

    def on_restore(self, positions: Sequence[Position]) -> None:  # noqa: B027 - optional hook
        """Called after state is rebuilt from the database. Default: no-op.

        Lets a composite strategy re-adopt positions opened before a restart, so an
        existing trade is still managed by the member that opened it.
        """

    def on_finish(self) -> None:  # noqa: B027 - optional hook
        """Called once after the last bar. Default: no-op."""

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        """Serialise the strategy's identity and configuration."""
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "warmup_bars": self.warmup_bars,
            "params": self.params.to_dict(),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.strategy_id!r} params={self.params!r}>"
