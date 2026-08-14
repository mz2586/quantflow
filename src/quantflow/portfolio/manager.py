"""Portfolio manager: the single source of truth for cash, positions and equity.

Every fill in the system is applied here, exactly once. Cash and positions move together in
one operation so the two can never disagree — a portfolio where cash has been debited but
the position not yet credited is a portfolio whose equity is wrong, and equity is what every
risk limit is measured against.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal

from quantflow.core.clock import Clock, SystemClock, start_of_utc_day
from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import EquityPoint, PortfolioSnapshot
from quantflow.domain.positions import ClosedTrade, Position
from quantflow.portfolio.funding import (
    FundingCharge,
    charge_for,
    funding_stamps,
    total_funding,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class PortfolioManager:
    """Mutable portfolio state, updated fill by fill.

    Deliberately the one mutable object in the trading path: every other layer works with
    immutable snapshots taken from it, so there is exactly one place where state changes.
    """

    base_currency: str = "USDT"
    starting_equity: Decimal = Decimal("10000")
    clock: Clock = field(default_factory=SystemClock)
    #: Spot subtracts the full notional to buy and credits it to sell. A linear perp does
    #: neither: you post margin and settle PnL, so a short cannot create cash. Defaulting to
    #: SPOT keeps every existing caller unchanged.
    market_type: MarketType = MarketType.SPOT
    #: Leverage assumed when reserving initial margin.
    leverage: Decimal = Decimal("1")

    _cash: Decimal = field(init=False)
    _positions: dict[Symbol, Position] = field(default_factory=dict, init=False)
    _mark_prices: dict[Symbol, Decimal] = field(default_factory=dict, init=False)
    _closed_trades: list[ClosedTrade] = field(default_factory=list, init=False)
    _equity_curve: list[EquityPoint] = field(default_factory=list, init=False)
    _applied_fill_ids: set[str] = field(default_factory=set, init=False)
    _peak_equity: Decimal = field(init=False)
    _day_start_equity: Decimal = field(init=False)
    _current_day: datetime | None = field(default=None, init=False)
    _realized_pnl: Decimal = field(default=ZERO, init=False)
    _fees_paid: Decimal = field(default=ZERO, init=False)
    _margin_by_symbol: dict[Symbol, Decimal] = field(default_factory=dict, init=False)
    _funding_paid: Decimal = field(default=ZERO, init=False)
    _last_funding_at: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialise cash and the equity high-water mark."""
        if self.starting_equity <= ZERO:
            raise ValidationError(f"starting equity must be positive, got {self.starting_equity}")
        self._cash = self.starting_equity
        self._peak_equity = self.starting_equity
        self._day_start_equity = self.starting_equity

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    @property
    def cash(self) -> Decimal:
        """Free quote-currency balance."""
        return self._cash

    @property
    def positions(self) -> tuple[Position, ...]:
        """Every open position."""
        return tuple(position for position in self._positions.values() if not position.is_flat)

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        """Completed round-trips, oldest first."""
        return tuple(self._closed_trades)

    @property
    def equity_curve(self) -> tuple[EquityPoint, ...]:
        """The recorded equity curve."""
        return tuple(self._equity_curve)

    @property
    def realized_pnl(self) -> Decimal:
        """Gross realised PnL, before fees."""
        return self._realized_pnl

    @property
    def fees_paid(self) -> Decimal:
        """Total fees paid."""
        return self._fees_paid

    @property
    def peak_equity(self) -> Decimal:
        """Highest equity seen, the reference for drawdown."""
        return self._peak_equity

    @property
    def applied_fill_ids(self) -> frozenset[str]:
        """Every fill id already folded in.

        Exposed because a venue reconciler has to know what it has *already* applied before
        it replays an execution report: the exchange re-delivers the same execution on every
        poll, and only this set makes a poll idempotent. It is also what must be persisted
        and restored, or a restart re-applies the whole window.
        """
        return frozenset(self._applied_fill_ids)

    def position_for(self, symbol: Symbol) -> Position | None:
        """The open position in ``symbol``, if any."""
        position = self._positions.get(symbol)
        return position if position is not None and not position.is_flat else None

    def mark_price(self, symbol: Symbol) -> Decimal | None:
        """The latest mark price for ``symbol``."""
        return self._mark_prices.get(symbol)

    def equity(self) -> Decimal:
        """Account value under the configured market's accounting.

        Spot: cash plus what the held assets are worth. Futures: the wallet balance plus
        unrealised PnL. Adding a perp's notional to cash would count the same money twice
        and inflate every equity-derived limit.
        """
        total = self._cash
        for position in self.positions:
            price = self._mark_prices.get(position.symbol)
            if price is None:
                raise ValidationError(
                    f"cannot value the portfolio: no mark price for {position.symbol}",
                    symbol=str(position.symbol),
                )
            if self.market_type is MarketType.FUTURE:
                total += position.unrealized_pnl(price)
            else:
                total += position.market_value(price)
        return total

    @property
    def funding_paid(self) -> Decimal:
        """Net funding settled so far. Negative means the account has paid out."""
        return self._funding_paid

    def settle_funding(
        self,
        now: datetime,
        *,
        rate_for: Callable[[Symbol, datetime], Decimal | None],
    ) -> tuple[FundingCharge, ...]:
        """Charge every funding stamp that has elapsed since the last settlement.

        Shared by backtest and paper deliberately: funding is a real cash movement, and a
        backtest that models it differently from paper is describing a different account.
        Spot has no funding and is skipped entirely.

        ``rate_for`` returns ``None`` where no rate is known, and that stamp is skipped -
        inventing a rate would fabricate a cost, which is no better than ignoring a real one.
        """
        if self.market_type is not MarketType.FUTURE:
            return ()

        since = self._last_funding_at or now
        self._last_funding_at = now
        charges: list[FundingCharge] = []
        for stamp in funding_stamps(since, now):
            for position in self.positions:
                price = self._mark_prices.get(position.symbol)
                if price is None:
                    continue
                rate = rate_for(position.symbol, stamp)
                if rate is None:
                    continue
                charge = charge_for(position, price=price, rate=rate, settled_at=stamp)
                if charge is None:
                    continue
                # Funding settles in cash, exactly as the venue does it.
                self._cash += charge.amount
                self._funding_paid += charge.amount
                charges.append(charge)

        if charges:
            logger.info(
                "portfolio.funding_settled",
                stamps=len({c.settled_at for c in charges}),
                charges=len(charges),
                net=str(total_funding(charges)),
            )
        return tuple(charges)

    @property
    def margin_posted(self) -> Decimal:
        """Margin currently reserved against open positions."""
        return sum(self._margin_by_symbol.values(), ZERO)

    def _apply_margin_cash(
        self,
        before: Position,
        after: Position,
        fill: Fill,
        closed: tuple[ClosedTrade, ...],
    ) -> None:
        """Move cash the way a linear perp actually does.

        The wallet changes on exactly two things: fees, and realised PnL when a position
        closes. Opening reserves margin - a reservation against the same wallet, not a
        spend - so a short credits nothing and cannot inflate cash. That inflation was the
        defect: on spot maths a short looked like income, and every limit read from it.
        """
        previous = abs(before.quantity)
        current = abs(after.quantity)
        reserved = self._margin_by_symbol.get(fill.symbol, ZERO)

        if current > previous:
            added_notional = (current - previous) * fill.price
            self._margin_by_symbol[fill.symbol] = reserved + added_notional / self.leverage
        elif previous > ZERO:
            released_fraction = (previous - current) / previous
            self._margin_by_symbol[fill.symbol] = reserved - reserved * released_fraction

        if after.is_flat:
            self._margin_by_symbol.pop(fill.symbol, None)

        self._cash -= fill.fee
        self._cash += sum((trade.gross_pnl for trade in closed), ZERO)

    def snapshot(self, at: datetime | None = None) -> PortfolioSnapshot:
        """An immutable view for the risk engine and the strategies."""
        return PortfolioSnapshot(
            timestamp=at or self.clock.now(),
            base_currency=self.base_currency,
            cash=self._cash,
            market_type=self.market_type,
            margin_posted=self.margin_posted,
            positions=self.positions,
            mark_prices=dict(self._mark_prices),
            peak_equity=self._peak_equity,
            day_start_equity=self._day_start_equity,
            realized_pnl=self._realized_pnl,
            fees_paid=self._fees_paid,
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def update_mark_price(self, symbol: Symbol, price: Decimal) -> None:
        """Record the latest price used to value a position."""
        if price <= ZERO:
            raise ValidationError(f"mark price must be positive, got {price}")
        self._mark_prices[symbol] = price

    def update_mark_prices(self, prices: Mapping[Symbol, Decimal]) -> None:
        """Record several mark prices at once."""
        for symbol, price in prices.items():
            self.update_mark_price(symbol, price)

    def apply_fill(
        self, fill: Fill, *, strategy_id: str | None = None
    ) -> tuple[Position, tuple[ClosedTrade, ...]]:
        """Apply a fill to cash and positions in one atomic step.

        Idempotent on ``fill_id``: exchanges re-deliver execution reports on reconnect, and
        double-counting one would corrupt both the position and the cash balance.

        Args:
            fill: The execution to apply.
            strategy_id: Which strategy opened the position. Carried onto the position and
                therefore onto every ClosedTrade it produces — without it, all performance
                attribution collapses into a single "unattributed" bucket, which makes
                per-strategy analysis impossible after the fact.

        Returns:
            The updated position and any round-trips the fill closed.

        """
        if fill.fill_id in self._applied_fill_ids:
            logger.debug("portfolio.duplicate_fill_ignored", fill_id=fill.fill_id)
            existing = self._positions.get(fill.symbol, Position(symbol=fill.symbol))
            return existing, ()

        position = self._positions.get(fill.symbol)
        if position is None:
            position = Position(symbol=fill.symbol, strategy_id=strategy_id)
        elif position.is_flat and strategy_id is not None:
            # Re-opening a previously flat position: adopt the new owner rather than
            # keeping whichever strategy happened to trade it last.
            position = replace(position, strategy_id=strategy_id)
        updated, closed = position.apply_fill(fill)

        if self.market_type is MarketType.FUTURE:
            self._apply_margin_cash(position, updated, fill, closed)
        # Spot: a buy spends notional plus fee, a sell receives notional minus fee.
        elif fill.side is OrderSide.BUY:
            self._cash -= fill.notional + fill.fee
        else:
            self._cash += fill.notional - fill.fee

        self._positions[fill.symbol] = updated
        self._applied_fill_ids.add(fill.fill_id)
        self._fees_paid += fill.fee
        self._closed_trades.extend(closed)
        self._realized_pnl += sum((trade.gross_pnl for trade in closed), ZERO)
        self._mark_prices[fill.symbol] = fill.price

        logger.debug(
            "portfolio.fill_applied",
            symbol=str(fill.symbol),
            side=fill.side.value,
            quantity=str(fill.quantity),
            price=str(fill.price),
            cash=str(self._cash),
            closed_trades=len(closed),
        )
        return updated, closed

    def apply_fills(
        self, fills: Iterable[Fill], *, strategy_id: str | None = None
    ) -> tuple[ClosedTrade, ...]:
        """Apply several fills in timestamp order."""
        closed: list[ClosedTrade] = []
        for fill in sorted(fills, key=lambda item: item.timestamp):
            _, trades = self.apply_fill(fill, strategy_id=strategy_id)
            closed.extend(trades)
        return tuple(closed)

    def set_protection(
        self,
        symbol: Symbol,
        *,
        stop_loss_price: Decimal | None = None,
        take_profit_price: Decimal | None = None,
    ) -> None:
        """Attach protective levels to an open position."""
        position = self._positions.get(symbol)
        if position is None or position.is_flat:
            return
        self._positions[symbol] = position.with_protection(
            stop_loss_price=stop_loss_price, take_profit_price=take_profit_price
        )

    def record_equity(self, at: datetime | None = None) -> EquityPoint:
        """Sample the equity curve and roll the drawdown and daily baselines.

        Called once per bar. Rolling the UTC-day baseline here rather than on a timer keeps
        the daily-loss limit aligned with the bars the engine actually processed, so a
        backtest and a live session compute it identically.
        """
        moment = at or self.clock.now()
        self._roll_day(moment)

        equity = self.equity()
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (
            (self._peak_equity - equity) / self._peak_equity if self._peak_equity > ZERO else ZERO
        )

        unrealized = sum(
            (
                position.unrealized_pnl(self._mark_prices[position.symbol])
                for position in self.positions
                if position.symbol in self._mark_prices
            ),
            ZERO,
        )

        # Positions without a mark are skipped rather than raising: a sample is a
        # measurement, and one unmarked symbol must not stop the curve advancing.
        exposure = sum(
            (
                position.notional(self._mark_prices[position.symbol])
                for position in self.positions
                if position.symbol in self._mark_prices
            ),
            ZERO,
        )

        point = EquityPoint(
            timestamp=moment,
            equity=equity,
            cash=self._cash,
            position_count=len(self.positions),
            drawdown_pct=max(ZERO, drawdown),
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
            gross_exposure=exposure,
        )
        self._equity_curve.append(point)
        return point

    def _roll_day(self, moment: datetime) -> None:
        """Reset the daily baseline when the UTC day changes."""
        day = start_of_utc_day(moment)
        if self._current_day is None:
            self._current_day = day
            self._day_start_equity = self.equity()
            return
        if day > self._current_day:
            self._current_day = day
            self._day_start_equity = self.equity()
            logger.debug(
                "portfolio.new_trading_day",
                day=day.date().isoformat(),
                opening_equity=str(self._day_start_equity),
            )

    # ------------------------------------------------------------------ #
    # Recovery
    # ------------------------------------------------------------------ #
    def restore(
        self,
        *,
        cash: Decimal,
        positions: Iterable[Position],
        peak_equity: Decimal | None = None,
        day_start_equity: Decimal | None = None,
        realized_pnl: Decimal = ZERO,
        fees_paid: Decimal = ZERO,
        applied_fill_ids: Iterable[str] = (),
    ) -> None:
        """Rebuild state after a restart.

        ``applied_fill_ids`` must be restored too: without it, a fill already applied
        before the crash would be reapplied when the exchange re-delivers it.
        """
        self._cash = cash
        self._positions = {position.symbol: position for position in positions}
        self._peak_equity = peak_equity if peak_equity is not None else cash
        self._day_start_equity = day_start_equity if day_start_equity is not None else cash
        self._realized_pnl = realized_pnl
        self._fees_paid = fees_paid
        self._applied_fill_ids = set(applied_fill_ids)
        logger.info(
            "portfolio.restored",
            cash=str(cash),
            positions=len(self._positions),
            known_fills=len(self._applied_fill_ids),
        )

    def anchor_cash(self, cash: Decimal, *, reason: str) -> bool:
        """Set the wallet balance to a figure read from the venue.

        Only for a live account, and only from an authoritative read. Cash on a linear-perp
        account is the venue's wallet balance — not a quantity this process derives — so once
        the exchange has been asked, its answer replaces whatever the local fold produced.
        Reconstructing it from fills is a *model* of the wallet, and a model that has been
        replaying a bounded execution window will differ from the wallet by every fee and
        every realised PnL that fell outside it.

        Returns whether the balance moved. Non-positive figures are refused: they are a
        failed read rather than an empty account, and sizing against zero would stop the
        session as surely as sizing against a fiction.
        """
        if cash <= ZERO:
            logger.warning("portfolio.cash_anchor_rejected", value=str(cash), reason=reason)
            return False
        if cash == self._cash:
            return False
        logger.warning(
            "portfolio.cash_anchored",
            previous=str(self._cash),
            venue_cash=str(cash),
            reason=reason,
        )
        self._cash = cash
        self._peak_equity = max(self._peak_equity, cash)
        return True

    def mark_fill_applied(self, fill_id: str) -> None:
        """Record a fill id as already folded in without applying it.

        Used when a venue execution has been superseded by a state repair taken from the
        same venue: applying it afterwards would count the same trade twice.
        """
        self._applied_fill_ids.add(fill_id)

    def reconcile(self, venue_positions: Mapping[Symbol, Decimal]) -> dict[Symbol, Decimal]:
        """Compare local positions against the venue's.

        Returns:
            ``{symbol: venue_quantity - local_quantity}`` for every mismatch. A non-empty
            result means the local view is wrong and trading should stop until it is
            resolved — continuing would size orders against a fictional position.

        """
        discrepancies: dict[Symbol, Decimal] = {}
        symbols = set(venue_positions) | set(self._positions)
        for symbol in symbols:
            local = self._positions.get(symbol)
            local_quantity = local.quantity if local else ZERO
            venue_quantity = venue_positions.get(symbol, ZERO)
            if local_quantity != venue_quantity:
                discrepancies[symbol] = venue_quantity - local_quantity
        if discrepancies:
            logger.error(
                "portfolio.reconciliation_mismatch",
                mismatches={str(key): str(value) for key, value in discrepancies.items()},
            )
        return discrepancies

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, str | int]:
        """Headline figures for logs, the API and reports."""
        equity = self.equity()
        wins = [trade for trade in self._closed_trades if trade.is_win]
        return {
            "equity": str(equity),
            "cash": str(self._cash),
            "starting_equity": str(self.starting_equity),
            "total_return_pct": str(
                (equity - self.starting_equity) / self.starting_equity
                if self.starting_equity > ZERO
                else ZERO
            ),
            "realized_pnl": str(self._realized_pnl),
            "fees_paid": str(self._fees_paid),
            "open_positions": len(self.positions),
            "closed_trades": len(self._closed_trades),
            "wins": len(wins),
            "peak_equity": str(self._peak_equity),
        }
