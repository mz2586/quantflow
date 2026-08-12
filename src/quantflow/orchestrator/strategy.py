"""The Strategy Orchestrator.

Sits above the individual strategies and below the risk engine. It implements ``Strategy``
itself, which is the whole integration: the paper engine already calls
``strategy.evaluate(context)`` once per symbol per closed bar, so an orchestrator that
satisfies that interface drops in with no change to the engine, the broker, the risk
engine or persistence.

What it adds is *selection*. Every enabled member is evaluated on every bar; the actionable
ones become candidates; the candidates are scored by a transparent model and the best one
is returned — or nothing is, when none clears the bar.

Two properties are load-bearing:

- **Attribution survives.** The winning member's own ``Signal`` is returned unchanged, so
  the order, position and closed trade all record the member that actually decided, not
  the string "orchestrator". Existing per-strategy dashboards keep working.
- **Positions keep their owner.** Once a position exists, only the strategy that opened it
  is consulted for the exit. A different member cannot take over a trade it did not open,
  which would leave the position managed by rules it was never sized under.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydantic import Field

from quantflow.ai.regime import RuleBasedRegimeDetector
from quantflow.core.errors import InsufficientDataError, StrategyError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import MarketRegime, SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import ClosedTrade, Position
from quantflow.domain.signals import Signal
from quantflow.orchestrator.performance import (
    MIN_TRADES_FOR_EVIDENCE,
    PerformanceMemory,
    evidence_score,
    has_negative_expectancy,
)
from quantflow.orchestrator.scoring import (
    MIN_SCORE_TO_TRADE,
    Candidate,
    StrategyRecord,
    gate_candidate,
    rank,
    score_candidate,
)
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import load_builtin_strategies, register_strategy

logger = get_logger(__name__)

#: Members excluded by default. `buy_and_hold` exists as a research benchmark: it emits one
#: entry and never exits, so inside a live orchestrator it would occupy a slot forever.
#: `orchestrator` excludes *itself* — it is registered like any other strategy, and without
#: this the default roster would contain an orchestrator containing an orchestrator.
DEFAULT_EXCLUDED = frozenset({"buy_and_hold", "orchestrator"})

#: Round-trip cost assumption used by the cost component, as a fraction of notional.
DEFAULT_COST_RATE = Decimal("0.002")


class OrchestratorParams(StrategyParams):
    """Parameters for :class:`StrategyOrchestrator`."""

    min_score: Decimal = Field(default=MIN_SCORE_TO_TRADE, ge=0, le=1)
    cost_rate: Decimal = Field(default=DEFAULT_COST_RATE, ge=0, le=1)


@dataclass(slots=True)
class Decision:
    """What the orchestrator concluded on one bar, for logging and the dashboard."""

    symbol: Symbol
    evaluated: int
    candidates: list[Candidate] = field(default_factory=list)
    selected: Candidate | None = None
    reason: str = ""
    regime: MarketRegime = MarketRegime.UNKNOWN
    gated: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "symbol": self.symbol.slashed,
            "evaluated": self.evaluated,
            "candidates": len(self.candidates),
            "regime": self.regime.value,
            "gated": len(self.gated),
            "selected": self.selected.strategy_id if self.selected else None,
            "score": str(self.selected.score) if self.selected else None,
            "reason": self.reason,
        }


@register_strategy
class StrategyOrchestrator(Strategy):
    """Evaluates every member each bar and executes the strongest valid candidate."""

    strategy_id = "orchestrator"
    description = "Evaluates all strategies each bar and selects the highest-scoring candidate"
    params_model = OrchestratorParams

    params: OrchestratorParams

    def __init__(
        self,
        params: OrchestratorParams | dict[str, Any] | None = None,
        *,
        members: Sequence[Strategy] | None = None,
    ) -> None:
        # `params` comes first and positionally because `StrategyRegistry.create` calls
        # `cls(params)`; making `members` keyword-only keeps the registry path working while
        # still allowing an explicit roster in tests and callers that want one.
        super().__init__(params)
        # `None` means "use the whole registry"; an explicit empty sequence is a mistake and
        # is rejected rather than silently replaced with every strategy there is.
        self._members: tuple[Strategy, ...] = (
            tuple(members) if members is not None else _default_members()
        )
        if not self._members:
            from quantflow.core.errors import ValidationError

            raise ValidationError("orchestrator needs at least one member", field="members")
        #: symbol -> strategy id that opened the live position, so exits stay with the owner.
        self._owners: dict[Symbol, str] = {}
        self._records: dict[str, StrategyRecord] = {}
        self._last_decision: Decision | None = None
        #: Realised results, bucketed per strategy / symbol / regime.
        self._memory = PerformanceMemory()
        #: Reused rather than reimplemented - this classifier already exists and is tested.
        self._detector = RuleBasedRegimeDetector()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def members(self) -> tuple[Strategy, ...]:
        """The member strategies, in evaluation order."""
        return self._members

    @property
    def owners(self) -> dict[Symbol, str]:
        """Which strategy owns each open position."""
        return dict(self._owners)

    @property
    def memory(self) -> PerformanceMemory:
        """Realised per-strategy performance."""
        return self._memory

    @property
    def records(self) -> dict[str, StrategyRecord]:
        """Realised per-strategy record for this session."""
        return dict(self._records)

    @property
    def last_decision(self) -> Decision | None:
        """The most recent decision, for logging and explanation."""
        return self._last_decision

    @property
    def warmup_bars(self) -> int:
        """The shortest member's warm-up.

        Deliberately the minimum, not the maximum: gating the whole orchestrator on its
        slowest member would silence fifteen ready strategies while one warms up. Each
        member still enforces its own warm-up inside ``evaluate``, so a cold member simply
        abstains.
        """
        return min(member.warmup_bars for member in self._members)

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    def evaluate(self, context: StrategyContext) -> Signal:
        """Run the orchestrator with warm-up and error containment.

        Overridden for one reason: the base implementation rejects a signal whose
        ``strategy_id`` differs from its own, and this class deliberately returns the
        winning *member's* signal so that orders, positions and closed trades record the
        strategy that actually decided. Re-stamping it as "orchestrator" would satisfy the
        check and destroy per-strategy attribution everywhere downstream.

        The guard still holds where it matters — each member is run through its own
        ``evaluate``, so a member that mis-attributes its signal is still caught.
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
                "orchestrator.failed",
                symbol=str(context.symbol),
                at=context.now.isoformat(),
                error=str(exc),
            )
            return context.hold(f"orchestrator error: {exc}", self.strategy_id)

        known = {member.strategy_id for member in self._members} | {self.strategy_id}
        if signal.strategy_id not in known:
            raise StrategyError(
                f"orchestrator returned a signal attributed to {signal.strategy_id!r}, "
                "which is not one of its members",
                strategy_id=self.strategy_id,
            )
        return signal

    def generate(self, context: StrategyContext) -> Signal:
        """Poll every member, score the actionable ones, return the winner."""
        if context.has_position:
            return self._manage_open_position(context)

        # The engine hands every strategy `MarketRegime.UNKNOWN`, so the orchestrator
        # classifies for itself from the same closed bars the members see. Nothing here
        # reads beyond the decision bar.
        regime = self._classify(context)

        candidates: list[Candidate] = []
        rejected: list[tuple[str, str]] = []
        for member in self._members:
            signal = member.evaluate(context)
            if not signal.is_actionable:
                rejected.append((member.strategy_id, signal.reason))
                continue
            if signal.direction is SignalDirection.CLOSE:
                # A close with no position is meaningless; ignore rather than execute.
                rejected.append((member.strategy_id, "close signal with no open position"))
                continue
            candidates.append(_to_candidate(signal, context))

        decision = Decision(symbol=context.symbol, evaluated=len(self._members))
        decision.regime = regime
        if not candidates:
            decision.reason = "no actionable candidates"
            self._last_decision = decision
            return context.hold("no actionable candidates", self.strategy_id)

        # Economic gates come before ranking: ranking a set of uneconomic candidates only
        # identifies the least bad one and then trades it.
        counts = self._positions_per_strategy()
        viable: list[Candidate] = []
        for candidate in candidates:
            # A strategy that has demonstrated negative expectancy on a real sample is
            # refused an entry outright rather than merely down-weighted. Down-weighting
            # only reorders a field; if every rival is also weak the proven loser still
            # wins the ranking and trades. Re-evaluated from current evidence each bar, so
            # it recovers as soon as its results do.
            proven_loser = self._memory.overall(candidate.strategy_id)
            if has_negative_expectancy(proven_loser):
                factor = proven_loser.profit_factor
                rejected.append(
                    (
                        candidate.strategy_id,
                        (
                            f"negative expectancy over {proven_loser.trades} trades "
                            f"(profit factor {factor:.2f})"
                            if factor is not None
                            else "negative expectancy"
                        ),
                    )
                )
                continue
            reason = gate_candidate(
                candidate,
                cost_rate=self.params.cost_rate,
                strategy_position_counts=counts,
            )
            if reason is None:
                viable.append(candidate)
            else:
                rejected.append((candidate.strategy_id, reason))
        decision.gated = [(name, reason) for name, reason in rejected]

        if not viable:
            decision.reason = f"all {len(candidates)} candidates failed the economic gates"
            self._last_decision = decision
            logger.info(
                "orchestrator.all_gated",
                symbol=str(context.symbol),
                regime=regime.value,
                candidates=len(candidates),
                first_reason=rejected[-1][1] if rejected else "",
            )
            return context.hold(decision.reason, self.strategy_id)

        open_symbols = frozenset(self._owners)
        scored = [
            score_candidate(
                candidate,
                regime=regime,
                records=self._evidence_records(regime, context.symbol),
                open_symbols=open_symbols,
                cost_rate=self.params.cost_rate,
            )
            for candidate in viable
        ]
        ordered = rank(scored)
        decision.candidates = ordered
        best = ordered[0]

        if best.score < self.params.min_score:
            decision.reason = f"best score {best.score:.3f} below floor {self.params.min_score}"
            self._last_decision = decision
            logger.info(
                "orchestrator.no_trade",
                symbol=str(context.symbol),
                candidates=len(ordered),
                best=best.strategy_id,
                best_score=f"{best.score:.3f}",
                floor=str(self.params.min_score),
            )
            return context.hold(decision.reason, self.strategy_id)

        decision.selected = best
        decision.reason = f"highest score {best.score:.3f} of {len(ordered)} candidates"
        self._last_decision = decision
        self._owners[context.symbol] = best.strategy_id

        logger.info(
            "orchestrator.selected",
            symbol=str(context.symbol),
            strategy=best.strategy_id,
            direction=best.direction.value,
            confidence=f"{best.confidence:.2f}",
            score=f"{best.score:.3f}",
            candidates=len(ordered),
            runner_up=ordered[1].describe() if len(ordered) > 1 else None,
            components={name: f"{value:.2f}" for name, value in best.components.items()},
        )
        # The member's own signal, unchanged: attribution downstream must name the strategy
        # that actually decided.
        return best.signal

    def _manage_open_position(self, context: StrategyContext) -> Signal:
        """Consult only the strategy that opened this position."""
        owner_id = self._owners.get(context.symbol)
        owner = next((member for member in self._members if member.strategy_id == owner_id), None)
        if owner is None:
            # No recorded owner - after a restart, for instance. Hold rather than let an
            # arbitrary member close a position it did not open; the risk engine's stop and
            # target still protect it.
            return context.hold(
                f"position in {context.symbol.slashed} has no recorded owner", self.strategy_id
            )
        signal = owner.evaluate(context)
        if signal.direction is SignalDirection.CLOSE:
            self._owners.pop(context.symbol, None)
        return signal

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def adopt(self, symbol: Symbol, strategy_id: str) -> None:
        """Record an existing position's owner, used when restoring after a restart."""
        self._owners[symbol] = strategy_id

    # ------------------------------------------------------------------ #
    # Regime and evidence
    # ------------------------------------------------------------------ #
    def _classify(self, context: StrategyContext) -> MarketRegime:
        """Classify the market from bars up to and including the decision bar."""
        try:
            observation = self._detector.detect(context.candles)
        except Exception:  # pragma: no cover - a detector fault must not stop trading
            return MarketRegime.UNKNOWN
        # An unconfident classification is worse than no classification: it would move a
        # fifth of every score on a label the detector itself does not stand behind.
        return observation.regime if observation.is_confident else MarketRegime.UNKNOWN

    def _evidence_records(
        self,
        regime: MarketRegime,
        symbol: Symbol,  # noqa: ARG002 - reserved for the per-symbol slice
    ) -> dict[str, StrategyRecord]:
        """Per-strategy evidence, preferring the most specific bucket with enough trades.

        Regime-specific evidence answers the question that actually matters — "does this
        strategy work in *this* kind of market" — but fills slowly, so it is used only once
        the slice itself has a usable sample. Otherwise the overall record stands in, and
        below the overall threshold the score falls back to neutral.
        """
        out: dict[str, StrategyRecord] = {}
        for member in self._members:
            name = member.strategy_id
            sliced = self._memory.for_regime(name, regime)
            source = sliced if sliced.has_slice_evidence else self._memory.overall(name)
            trades = source.trades
            if sliced.has_slice_evidence:
                # A qualifying slice speaks for itself, so present it as meaningful even
                # though it holds fewer trades than the overall threshold.
                trades = max(trades, MIN_TRADES_FOR_EVIDENCE)
            out[name] = StrategyRecord(
                strategy_id=name,
                trades=trades,
                wins=int(evidence_score(source) * Decimal(trades)) if trades else 0,
                net_pnl=source.net_pnl,
            )
        return out

    def _positions_per_strategy(self) -> dict[str, int]:
        """How many open positions each strategy currently owns."""
        counts: dict[str, int] = {}
        for owner in self._owners.values():
            counts[owner] = counts.get(owner, 0) + 1
        return counts

    def on_restore(self, positions: Sequence[Position]) -> None:
        """Re-adopt positions rebuilt from the database after a restart.

        Each persisted position carries the strategy that opened it, so ownership survives
        a crash and an existing trade keeps being managed by its own author's exit rules.
        """
        for position in positions:
            if position.strategy_id:
                self._owners[position.symbol] = position.strategy_id

    def on_trade_closed(self, trade: ClosedTrade) -> None:
        """Engine hook: fold a completed round-trip into its strategy's record."""
        self.record_trade(trade)

    def record_trade(self, trade: ClosedTrade) -> None:
        """Fold a completed trade into the owning strategy's record and memory."""
        self._memory.record(trade)
        key = trade.strategy_id or "unknown"
        current = self._records.get(key, StrategyRecord(strategy_id=key))
        self._records[key] = StrategyRecord(
            strategy_id=key,
            trades=current.trades + 1,
            wins=current.wins + (1 if trade.net_pnl > ZERO else 0),
            net_pnl=current.net_pnl + trade.net_pnl,
        )

    def on_start(self, symbols: Sequence[Symbol]) -> None:
        """Forward to every member."""
        for member in self._members:
            member.on_start(symbols)

    def on_finish(self) -> None:
        """Forward to every member."""
        for member in self._members:
            member.on_finish()

    def describe(self) -> dict[str, Any]:
        """Include the member roster alongside the base description."""
        base = super().describe()
        base["members"] = [member.strategy_id for member in self._members]
        return base


def _to_candidate(signal: Signal, context: StrategyContext) -> Candidate:
    """Build a candidate from an actionable member signal."""
    return Candidate(
        symbol=signal.symbol,
        strategy_id=signal.strategy_id,
        direction=signal.direction,
        confidence=signal.conviction,
        entry=signal.reference_price or context.price,
        stop_loss=signal.stop_loss_price,
        take_profit=signal.take_profit_price,
        timestamp=signal.timestamp,
        signal=signal,
        regime=context.regime,
    )


def _default_members(excluded: frozenset[str] = DEFAULT_EXCLUDED) -> tuple[Strategy, ...]:
    """Every registered strategy except the excluded ones, in stable order."""
    registry = load_builtin_strategies()
    return tuple(registry.create(name) for name in sorted(registry.names()) if name not in excluded)


__all__ = [
    "DEFAULT_EXCLUDED",
    "Decision",
    "OrchestratorParams",
    "StrategyOrchestrator",
]
