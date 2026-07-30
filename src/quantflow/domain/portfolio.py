"""Portfolio state: balances, equity, drawdown and exposure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import Position, gross_exposure, net_exposure


@dataclass(frozen=True, slots=True)
class Balance:
    """Free and locked amounts of a single asset."""

    asset: str
    free: Decimal
    locked: Decimal = ZERO

    def __post_init__(self) -> None:
        """Validate the balance."""
        if self.free < ZERO or self.locked < ZERO:
            raise ValidationError(f"negative balance for {self.asset}")

    @property
    def total(self) -> Decimal:
        """Free plus locked."""
        return self.free + self.locked


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state.

    ``equity`` is cash plus the mark-to-market value of open positions. ``peak_equity`` is
    carried forward so drawdown can be computed without re-reading history — critical for
    the max-drawdown risk rule, which must be evaluated on every order.
    """

    timestamp: datetime
    base_currency: str
    cash: Decimal
    positions: tuple[Position, ...] = field(default_factory=tuple)
    mark_prices: Mapping[Symbol, Decimal] = field(default_factory=dict)
    peak_equity: Decimal = ZERO
    day_start_equity: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO

    def __post_init__(self) -> None:
        """Validate the snapshot."""
        if self.timestamp.tzinfo is None:
            raise ValidationError("portfolio timestamp must be timezone-aware UTC")
        if self.cash < ZERO:
            raise ValidationError(f"cash cannot be negative, got {self.cash}")

    # ------------------------------------------------------------------ #
    # Valuation
    # ------------------------------------------------------------------ #
    @property
    def open_positions(self) -> tuple[Position, ...]:
        """Positions with non-zero exposure."""
        return tuple(position for position in self.positions if not position.is_flat)

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.open_positions)

    @property
    def positions_value(self) -> Decimal:
        """Signed mark-to-market value of open positions."""
        return net_exposure(self.open_positions, dict(self.mark_prices))

    @property
    def gross_exposure(self) -> Decimal:
        """Sum of absolute position notionals."""
        return gross_exposure(self.open_positions, dict(self.mark_prices))

    @property
    def equity(self) -> Decimal:
        """Total account value: cash plus position value."""
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Aggregate mark-to-market PnL on open positions."""
        total = ZERO
        for position in self.open_positions:
            price = self.mark_prices.get(position.symbol)
            if price is None:
                raise ValidationError(
                    f"missing mark price for {position.symbol}", symbol=str(position.symbol)
                )
            total += position.unrealized_pnl(price)
        return total

    @property
    def total_pnl(self) -> Decimal:
        """Realised plus unrealised PnL."""
        return self.realized_pnl + self.unrealized_pnl

    # ------------------------------------------------------------------ #
    # Risk measures
    # ------------------------------------------------------------------ #
    @property
    def leverage(self) -> Decimal:
        """Gross exposure divided by equity."""
        return safe_divide(self.gross_exposure, self.equity)

    @property
    def exposure_pct(self) -> Decimal:
        """Gross exposure as a fraction of equity — same as leverage, named for the rule."""
        return self.leverage

    @property
    def drawdown_pct(self) -> Decimal:
        """Current drawdown from the equity peak, as a positive fraction."""
        peak = max(self.peak_equity, self.equity)
        if peak <= ZERO:
            return ZERO
        return max(ZERO, safe_divide(peak - self.equity, peak))

    @property
    def daily_pnl(self) -> Decimal:
        """PnL since the start of the current UTC day."""
        if self.day_start_equity == ZERO:
            return ZERO
        return self.equity - self.day_start_equity

    @property
    def daily_pnl_pct(self) -> Decimal:
        """Daily PnL as a fraction of the day's opening equity."""
        return safe_divide(self.daily_pnl, self.day_start_equity)

    @property
    def free_cash(self) -> Decimal:
        """Cash available for new positions."""
        return self.cash

    def position_for(self, symbol: Symbol) -> Position | None:
        """The open position in ``symbol``, if any."""
        for position in self.positions:
            if position.symbol == symbol and not position.is_flat:
                return position
        return None

    def has_position(self, symbol: Symbol) -> bool:
        """Whether there is open exposure in ``symbol``."""
        return self.position_for(symbol) is not None

    def position_pct(self, symbol: Symbol) -> Decimal:
        """A symbol's notional as a fraction of equity."""
        position = self.position_for(symbol)
        if position is None:
            return ZERO
        price = self.mark_prices.get(symbol)
        if price is None:
            raise ValidationError(f"missing mark price for {symbol}", symbol=str(symbol))
        return safe_divide(position.notional(price), self.equity)


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """A single sample on the equity curve."""

    timestamp: datetime
    equity: Decimal
    cash: Decimal
    position_count: int
    drawdown_pct: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO


def build_equity_curve(points: Sequence[EquityPoint]) -> tuple[EquityPoint, ...]:
    """Sort samples chronologically and backfill running drawdown.

    Drawdown is recomputed from the series rather than trusted from each sample, so a curve
    stitched together from multiple sources is still internally consistent.
    """
    ordered = sorted(points, key=lambda point: point.timestamp)
    peak = ZERO
    rebuilt: list[EquityPoint] = []
    for point in ordered:
        peak = max(peak, point.equity)
        drawdown = ZERO if peak <= ZERO else max(ZERO, safe_divide(peak - point.equity, peak))
        rebuilt.append(
            EquityPoint(
                timestamp=point.timestamp,
                equity=point.equity,
                cash=point.cash,
                position_count=point.position_count,
                drawdown_pct=drawdown,
                realized_pnl=point.realized_pnl,
                unrealized_pnl=point.unrealized_pnl,
            )
        )
    return tuple(rebuilt)
