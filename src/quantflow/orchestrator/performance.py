"""Per-strategy performance memory.

What the orchestrator knows about how each of its members has actually done, kept
separately per strategy, per symbol and per regime so a strategy is never credited with
another's result.

Three properties matter more than the arithmetic:

- **Sample thresholds.** Below :data:`MIN_TRADES_FOR_EVIDENCE` a record does not move a
  score at all. Three lucky trades promoting a strategy is precisely how a decision engine
  ends up chasing whatever ran hottest an hour ago.
- **Recency weighting.** Older trades count for less via an exponential half-life, so the
  system adapts without discarding the longer record wholesale.
- **Recoverability.** A penalty is a function of current evidence, not a latch. A strategy
  that starts performing regains its weight on the next trades; nothing is blacklisted.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import MarketRegime
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import ClosedTrade

#: Closed trades a bucket needs before it is allowed to move a score.
MIN_TRADES_FOR_EVIDENCE = 20

#: Smaller threshold for the finer-grained buckets. Per-regime and per-symbol slices fill
#: far more slowly than the overall record, and requiring twenty in each would mean they
#: never activate in a session of realistic length.
MIN_TRADES_FOR_SLICE = 8

#: Trades after which a result carries half its original weight.
RECENCY_HALF_LIFE = 25

#: How many recent trades the short-window statistics look at.
RECENT_WINDOW = 10


def _recency_weights(count: int) -> list[Decimal]:
    """Weights for ``count`` trades, oldest first, halving every :data:`RECENCY_HALF_LIFE`."""
    weights: list[Decimal] = []
    for age in range(count - 1, -1, -1):
        weights.append(Decimal(2) ** Decimal(-age / RECENCY_HALF_LIFE))
    return weights


@dataclass(frozen=True, slots=True)
class Record:
    """Summary of one bucket of closed trades."""

    key: str
    trades: int = 0
    wins: int = 0
    net_pnl: Decimal = ZERO
    gross_win: Decimal = ZERO
    gross_loss: Decimal = ZERO
    weighted_pnl: Decimal = ZERO
    recent_trades: int = 0
    recent_wins: int = 0
    recent_pnl: Decimal = ZERO
    max_drawdown: Decimal = ZERO
    loss_streak: int = 0

    @property
    def is_meaningful(self) -> bool:
        """Whether the sample is large enough to move a score."""
        return self.trades >= MIN_TRADES_FOR_EVIDENCE

    @property
    def has_slice_evidence(self) -> bool:
        """Whether a finer-grained slice has enough trades to say anything."""
        return self.trades >= MIN_TRADES_FOR_SLICE

    @property
    def win_rate(self) -> Decimal:
        """Wins over trades."""
        return Decimal(self.wins) / Decimal(self.trades) if self.trades else ZERO

    @property
    def recent_win_rate(self) -> Decimal:
        """Wins over trades in the recent window."""
        return (
            Decimal(self.recent_wins) / Decimal(self.recent_trades) if self.recent_trades else ZERO
        )

    @property
    def profit_factor(self) -> Decimal | None:
        """Gross win over gross loss. ``None`` when there are no losses to divide by."""
        if self.gross_loss <= ZERO:
            return None
        return self.gross_win / self.gross_loss

    @property
    def average_win(self) -> Decimal:
        """Mean winning trade."""
        return self.gross_win / Decimal(self.wins) if self.wins else ZERO

    @property
    def average_loss(self) -> Decimal:
        """Mean losing trade, as a positive number."""
        losses = self.trades - self.wins
        return self.gross_loss / Decimal(losses) if losses else ZERO

    def to_dict(self) -> dict[str, str | int]:
        """Serialisable summary for logs and the API."""
        return {
            "key": self.key,
            "trades": self.trades,
            "wins": self.wins,
            "net_pnl": str(self.net_pnl),
            "weighted_pnl": str(self.weighted_pnl),
            "recent_trades": self.recent_trades,
            "recent_pnl": str(self.recent_pnl),
            "loss_streak": self.loss_streak,
            "max_drawdown": str(self.max_drawdown),
        }


def summarise(key: str, trades: Sequence[ClosedTrade]) -> Record:
    """Build a :class:`Record` from a bucket's trades, oldest first."""
    if not trades:
        return Record(key=key)

    wins = sum(1 for trade in trades if trade.net_pnl > ZERO)
    gross_win = sum((t.net_pnl for t in trades if t.net_pnl > ZERO), ZERO)
    gross_loss = abs(sum((t.net_pnl for t in trades if t.net_pnl <= ZERO), ZERO))

    weights = _recency_weights(len(trades))
    weighted = sum(
        (trade.net_pnl * weight for trade, weight in zip(trades, weights, strict=True)), ZERO
    )

    recent = trades[-RECENT_WINDOW:]
    # Drawdown on the strategy's own equity curve, not the portfolio's: it measures how
    # badly this strategy alone has been running.
    running = ZERO
    peak = ZERO
    drawdown = ZERO
    streak = 0
    worst_streak = 0
    for trade in trades:
        running += trade.net_pnl
        peak = max(peak, running)
        drawdown = max(drawdown, peak - running)
        if trade.net_pnl > ZERO:
            streak = 0
        else:
            streak += 1
            worst_streak = max(worst_streak, streak)

    return Record(
        key=key,
        trades=len(trades),
        wins=wins,
        net_pnl=sum((t.net_pnl for t in trades), ZERO),
        gross_win=gross_win,
        gross_loss=gross_loss,
        weighted_pnl=weighted,
        recent_trades=len(recent),
        recent_wins=sum(1 for t in recent if t.net_pnl > ZERO),
        recent_pnl=sum((t.net_pnl for t in recent), ZERO),
        max_drawdown=drawdown,
        # The *current* streak, which is what a penalty should react to.
        loss_streak=streak,
    )


@dataclass(slots=True)
class PerformanceMemory:
    """Closed trades bucketed by strategy, and by strategy within symbol and regime."""

    _by_strategy: dict[str, deque[ClosedTrade]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=500))
    )
    _by_strategy_symbol: dict[tuple[str, Symbol], deque[ClosedTrade]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=500))
    )
    _by_strategy_regime: dict[tuple[str, MarketRegime], deque[ClosedTrade]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=500))
    )

    def record(self, trade: ClosedTrade, *, regime: MarketRegime = MarketRegime.UNKNOWN) -> None:
        """File one completed round-trip into every bucket it belongs to."""
        key = trade.strategy_id or "unknown"
        self._by_strategy[key].append(trade)
        self._by_strategy_symbol[(key, trade.symbol)].append(trade)
        # `trade.regime` is set by the engine where known; the argument is the fallback for
        # callers that have the regime to hand but the trade does not carry it.
        observed = getattr(trade, "regime", None) or regime
        if isinstance(observed, MarketRegime):
            self._by_strategy_regime[(key, observed)].append(trade)

    def load(self, trades: Iterable[ClosedTrade]) -> None:
        """Rebuild memory from persisted trades, oldest first."""
        for trade in trades:
            self.record(trade)

    def overall(self, strategy_id: str) -> Record:
        """The strategy's whole record."""
        return summarise(strategy_id, list(self._by_strategy.get(strategy_id, ())))

    def for_symbol(self, strategy_id: str, symbol: Symbol) -> Record:
        """The strategy's record on one symbol."""
        key = (strategy_id, symbol)
        return summarise(
            f"{strategy_id}@{symbol.slashed}", list(self._by_strategy_symbol.get(key, ()))
        )

    def for_regime(self, strategy_id: str, regime: MarketRegime) -> Record:
        """The strategy's record in one regime."""
        key = (strategy_id, regime)
        return summarise(
            f"{strategy_id}#{regime.value}", list(self._by_strategy_regime.get(key, ()))
        )

    def known_strategies(self) -> list[str]:
        """Every strategy with at least one recorded trade."""
        return sorted(self._by_strategy)

    def total_trades(self) -> int:
        """Trades recorded across all strategies."""
        return sum(len(bucket) for bucket in self._by_strategy.values())


def evidence_score(record: Record) -> Decimal:
    """Map a record to ``[0,1]``, or neutral 0.5 when the sample is too small.

    Blends risk-adjusted quality (profit factor) with recency-weighted profitability, then
    applies penalties for an active losing streak and for drawdown. Every term is a
    function of *current* evidence, so a strategy recovers as soon as its results do —
    there is no latch and no permanent exclusion.
    """
    if not record.is_meaningful:
        return Decimal("0.5")

    factor = record.profit_factor
    # No losses at all in a meaningful sample is genuinely good. Otherwise profit factor
    # 1.0 is break-even and maps to the midpoint; 2.0 and above saturates.
    quality = ONE if factor is None else min(max(factor / Decimal("2"), ZERO), ONE)

    direction = ONE if record.weighted_pnl > ZERO else ZERO
    base = quality * Decimal("0.7") + direction * Decimal("0.3")

    # An active losing streak is the most current evidence there is.
    streak_penalty = min(Decimal(record.loss_streak) / Decimal("10"), Decimal("0.3"))
    return min(max(base - streak_penalty, ZERO), ONE)


#: Profit factor below which a strategy with a meaningful sample is treated as having
#: demonstrated negative expectancy. Set below 1.0 deliberately: a strategy hovering at
#: break-even is unproven, not disproven, and only a clear shortfall should stop it.
NEGATIVE_EXPECTANCY_PF = Decimal("0.8")


def has_negative_expectancy(record: Record) -> bool:
    """Whether the record is a large enough sample to conclude the strategy loses money.

    Three conditions together, so no single bad patch is enough: a meaningful sample, a
    profit factor clearly below break-even, and a recency-weighted PnL that is still
    negative — meaning the losses are not merely old history the strategy has since grown
    out of.

    Recomputed from current evidence on every bar, so a strategy that starts winning stops
    being blocked without anything having to reset it. Nothing here is a latch.
    """
    if not record.is_meaningful:
        return False
    factor = record.profit_factor
    if factor is None:
        return False
    return factor < NEGATIVE_EXPECTANCY_PF and record.weighted_pnl < ZERO


__all__ = [
    "MIN_TRADES_FOR_EVIDENCE",
    "MIN_TRADES_FOR_SLICE",
    "NEGATIVE_EXPECTANCY_PF",
    "RECENCY_HALF_LIFE",
    "RECENT_WINDOW",
    "PerformanceMemory",
    "Record",
    "evidence_score",
    "has_negative_expectancy",
    "summarise",
]
