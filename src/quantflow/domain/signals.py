"""Strategy signals — the contract between the strategy engine and everything downstream.

A signal expresses *intent*, never a sized order. Sizing is the risk engine's job: keeping
that boundary sharp is what makes it impossible for a strategy to route around position
limits. Strategies may express *conviction* (0–1), which the risk engine may use to scale
within its own limits, but never beyond them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import MarketRegime, OrderType, SignalDirection, TimeInForce
from quantflow.domain.instruments import Symbol


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's intent for one symbol at one point in time."""

    symbol: Symbol
    direction: SignalDirection
    timestamp: datetime
    strategy_id: str
    conviction: Decimal = ONE
    """Strength in ``[0, 1]``. The risk engine may scale size by this, never above its cap."""
    reference_price: Decimal | None = None
    """The price the decision was made at — used to detect stale signals."""
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    regime: MarketRegime = MarketRegime.UNKNOWN
    reason: str = ""
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the signal."""
        if self.timestamp.tzinfo is None:
            raise ValidationError("signal timestamp must be timezone-aware UTC")
        if not (ZERO <= self.conviction <= ONE):
            raise ValidationError(
                f"conviction must be in [0, 1], got {self.conviction}",
                strategy_id=self.strategy_id,
            )
        if not self.strategy_id:
            raise ValidationError("signal requires a strategy_id")
        if self.order_type.requires_price and self.limit_price is None:
            raise ValidationError(
                f"signal with order_type={self.order_type} requires a limit_price",
                strategy_id=self.strategy_id,
            )
        for name in ("reference_price", "limit_price", "stop_loss_price", "take_profit_price"):
            value: Decimal | None = getattr(self, name)
            if value is not None and value <= ZERO:
                raise ValidationError(f"signal {name} must be positive, got {value}")

    @property
    def is_actionable(self) -> bool:
        """Whether the signal should produce an order."""
        return self.direction.is_actionable and self.conviction > ZERO

    @property
    def is_entry(self) -> bool:
        """Whether the signal opens exposure."""
        return self.direction in (SignalDirection.LONG, SignalDirection.SHORT)

    @property
    def is_exit(self) -> bool:
        """Whether the signal closes exposure."""
        return self.direction is SignalDirection.CLOSE

    def is_stale(self, now: datetime, *, max_age_seconds: float) -> bool:
        """Whether the signal is older than ``max_age_seconds``.

        Acting on a stale signal is one of the more expensive live-trading mistakes; the
        execution engine checks this before every submission.
        """
        return (now - self.timestamp).total_seconds() > max_age_seconds

    @classmethod
    def hold(
        cls, symbol: Symbol, timestamp: datetime, strategy_id: str, reason: str = ""
    ) -> Signal:
        """Build an explicit no-action signal.

        Strategies return this rather than ``None`` so that "the strategy ran and decided
        nothing" is distinguishable from "the strategy did not run".
        """
        return cls(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            timestamp=timestamp,
            strategy_id=strategy_id,
            conviction=ZERO,
            reason=reason,
        )
