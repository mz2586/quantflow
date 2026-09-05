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

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from pydantic import Field

from quantflow.ai.regime import RuleBasedRegimeDetector
from quantflow.core.errors import InsufficientDataError, StrategyError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO
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
from quantflow.orchestrator.pyramid import EntryThesis, pyramid_verdict
from quantflow.orchestrator.scoring import (
    MAX_LEGS_PER_SYMBOL,
    MIN_SCORE_TO_TRADE,
    Candidate,
    StrategyRecord,
    gate_candidate,
    net_edge_of,
    rank,
    score_candidate,
)
from quantflow.orchestrator.selection import (
    SelectionInputs,
    assess_candidate,
    strategy_family,
)
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import (
    StrategyRegistry,
    load_builtin_strategies,
    register_strategy,
)
from quantflow.universe.assets import (
    asset_class_from_metadata,
    gating_reason,
    strategy_supports_class,
)

logger = get_logger(__name__)

#: Members excluded by default. `buy_and_hold` exists as a research benchmark: it emits one
#: entry and never exits, so inside a live orchestrator it would occupy a slot forever.
#: `orchestrator` excludes *itself* — it is registered like any other strategy, and without
#: this the default roster would contain an orchestrator containing an orchestrator.
DEFAULT_EXCLUDED = frozenset({"buy_and_hold", "orchestrator"})

#: Round-trip cost assumption used by the cost component, as a fraction of notional.
#:
#: Measured, not assumed. Across 18 live trades on this account the real round trip was
#: **0.0920%** — 2.29 in fees against 2,487 of notional. The previous 0.2000% was more than
#: double that, and the gate subtracted the difference from every candidate's edge before
#: judging it. Three candidates were refused in a single hour at 0.3929%, 0.3814% and
#: 0.3774% against a 0.4000% floor; each clears it once the over-charge is removed.
#:
#: Set to 0.0011 rather than the measured 0.00092: that is taker on both sides, the worst
#: case the account can actually pay. Erring high by a hair keeps the gate honest, while
#: erring low would admit trades that cannot cover their own execution.
#:
#: The floor itself is untouched. This is the same bar applied to a true number.
DEFAULT_COST_RATE = Decimal("0.0011")


class OrchestratorParams(StrategyParams):
    """Parameters for :class:`StrategyOrchestrator`."""

    min_score: Decimal = Field(default=MIN_SCORE_TO_TRADE, ge=0, le=1)
    cost_rate: Decimal = Field(default=DEFAULT_COST_RATE, ge=0, le=1)
    #: Restrict the candidate pool to these strategy ids. ``None`` means the whole
    #: registry; an explicit empty list is a mistake and is rejected, matching how
    #: ``members`` already behaves. A live session builds its strategy through the
    #: registry, which cannot reach the keyword-only ``members`` argument, so retiring
    #: strategies from a live rotation has to travel as a parameter.
    pool: list[str] | None = Field(default=None)


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
        pool: list[str] | None = getattr(self.params, "pool", None)
        if members is not None:
            self._members: tuple[Strategy, ...] = tuple(members)
        elif pool is not None:
            self._members = _members_from_pool(pool)
        else:
            self._members = _default_members()
        if not self._members:
            from quantflow.core.errors import ValidationError

            raise ValidationError("orchestrator needs at least one member", field="members")
        #: symbol -> strategy id that opened the live position, so exits stay with the owner.
        self._owners: dict[Symbol, str] = {}
        #: The case that opened each symbol's position, so a later candidate can be judged
        #: against it rather than merely counted.
        self._thesis: dict[Symbol, EntryThesis] = {}
        #: Entry legs currently open per symbol. The venue nets them into one position;
        #: this is what bounds how many times the engine may add.
        self._legs: dict[Symbol, int] = {}
        self._records: dict[str, StrategyRecord] = {}
        self._last_decision: Decision | None = None
        #: Realised results, bucketed per strategy / symbol / regime.
        self._memory = PerformanceMemory()
        #: Reused rather than reimplemented - this classifier already exists and is tested.
        self._detector = RuleBasedRegimeDetector()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def _select(
        self,
        candidates: list[Candidate],
        *,
        regime: MarketRegime,
        field: list[Candidate] | None = None,
    ) -> tuple[list[Candidate], list[tuple[str, str]]]:
        """Apply the selection layer, returning survivors and why the rest were refused.

        The measurements handed to :func:`assess_candidate` all come from state the
        orchestrator already holds — the candidate field for confluence, per-regime memory
        for expectancy, open positions for duplication. Nothing is fetched, so nothing here
        can see a bar the strategies did not.
        """
        survivors: list[Candidate] = []
        refused: list[tuple[str, str]] = []
        # Corroboration is counted over every candidate that FIRED, not over the ones that
        # survived the economic gate. A strategy whose own reward:risk was too thin to
        # trade is still an independent opinion about direction, and deleting it before
        # counting would let the gate quietly destroy the evidence this layer weighs.
        opinions = field if field is not None else candidates
        # What the pool could corroborate with, at best. A roster that cannot produce two
        # independent families is not asked to.
        available = len({strategy_family(member.strategy_id) for member in self._members})

        for candidate in candidates:
            record = self._memory.for_regime(candidate.strategy_id, regime)
            # Expectancy per trade, from the record's own totals. Record exposes counts and
            # PnL rather than a ratio, so it is derived here instead of assumed to exist.
            expectancy = (record.net_pnl / Decimal(record.trades)) if record.trades > 0 else None
            # A candidate whose symbol is already held is the clearest possible duplicate;
            # anything else is treated as uncorrelated because this class holds no return
            # series of its own, and inventing a correlation would be worse than omitting
            # one. The portfolio-level correlation rule in the risk engine still applies.
            duplicate = ONE if candidate.symbol in self._owners else ZERO
            verdict = assess_candidate(
                SelectionInputs(
                    strategy_id=candidate.strategy_id,
                    agreeing_families=_count_agreeing_families(opinions, candidate),
                    regime_expectancy=expectancy,
                    regime_samples=record.trades,
                    max_correlation=duplicate,
                    volume_share=ZERO,
                    available_families=available,
                    # Lets a strong lone family stand in for the missing second opinion.
                    # Same arithmetic the economic gate used, so the two cannot disagree.
                    net_edge=net_edge_of(candidate, cost_rate=self.params.cost_rate),
                )
            )
            if verdict.accepted:
                survivors.append(candidate)
            else:
                refused.append((candidate.strategy_id, "; ".join(verdict.reasons)))

        return survivors, refused

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

    def _deselected(
        self,
        context: StrategyContext,
        decision: Decision,
        candidates: list[Candidate],
        rejects: list[tuple[str, str]],
        regime: MarketRegime,
    ) -> Signal:
        """Record and explain a bar where every candidate failed selection.

        Not trading is the expected outcome most of the time now, so it is logged with the
        first reason attached: a silent hold and a hold caused by a broken rule look
        identical otherwise.
        """
        decision.reason = (
            f"all {len(candidates)} candidates failed selection "
            "(confluence, regime expectancy or correlation)"
        )
        self._last_decision = decision
        logger.info(
            "orchestrator.all_deselected",
            symbol=str(context.symbol),
            regime=regime.value,
            candidates=len(candidates),
            first_reason=rejects[0][1] if rejects else "",
        )
        return context.hold(decision.reason, self.strategy_id)

    def _admit_for_asset_class(
        self, context: StrategyContext
    ) -> tuple[list[Strategy], list[tuple[str, str]]]:
        """Split the roster into members this instrument admits and members it does not.

        A strategy whose information source is meaningless on this instrument is refused
        *before* it is evaluated, not after. Running it and discarding the signal would
        still let a volume strategy reading a synthetic equity perpetual's thin,
        venue-local tape count toward the confluence requirement in the selection layer —
        the orchestrator would refuse to act on it while treating it as corroboration,
        which is exactly what the family taxonomy exists to prevent.

        The class is resolved once per bar rather than once per member, and is only ever
        used to refuse: a member that clears the gate is judged on precisely the same
        evidence it always was. Refusals are returned rather than logged away so they land
        in ``Decision.gated``, because a member that vanished without explanation is
        indistinguishable from a broken one.
        """
        asset_class = asset_class_from_metadata(context.metadata)
        admitted: list[Strategy] = []
        refused: list[tuple[str, str]] = []
        for member in self._members:
            family = strategy_family(member.strategy_id)
            if strategy_supports_class(
                family, asset_class, declared=member.supported_asset_classes
            ):
                admitted.append(member)
            else:
                refused.append(
                    (member.strategy_id, gating_reason(member.strategy_id, family, asset_class))
                )
        return admitted, refused

    def generate(  # noqa: PLR0911, PLR0912 - each return is a distinct refusal reason
        self, context: StrategyContext
    ) -> Signal:
        """Poll every member, score the actionable ones, return the winner.

        The branch count is high because each exit is a *different* answer to "why not" —
        no candidates, all gated, all deselected, below the score floor, pyramid refused.
        Collapsing them would trade a legible decision log for a tidier function, and the
        decision log is what makes "the bot is not trading" answerable.
        """
        pyramiding = False
        if context.has_position:
            managed, pyramiding = self._open_position_route(context)
            if managed is not None:
                return managed

        # The engine hands every strategy `MarketRegime.UNKNOWN`, so the orchestrator
        # classifies for itself from the same closed bars the members see. Nothing here
        # reads beyond the decision bar.
        regime = self._classify(context)

        candidates: list[Candidate] = []
        admitted, rejected = self._admit_for_asset_class(context)
        for member in admitted:
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

        # Selection layer: corroboration, regime-conditional evidence and duplication.
        # These are questions about the field and about history, so they come after the
        # per-candidate economics and before ranking — ranking a set of uncorroborated
        # candidates just finds the most confident one.
        viable, selection_rejects = self._select(viable, regime=regime, field=candidates)
        rejected.extend(selection_rejects)
        decision.gated = [(name, reason) for name, reason in rejected]
        if not viable:
            return self._deselected(context, decision, candidates, selection_rejects, regime)

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

        if pyramiding:
            refusal = self._pyramid_admits(context, best, regime)
            if refusal is not None:
                return refusal

        decision.selected = best
        decision.reason = f"highest score {best.score:.3f} of {len(ordered)} candidates"
        self._last_decision = decision
        self._owners[context.symbol] = best.strategy_id
        if pyramiding:
            self._legs[context.symbol] = self._legs.get(context.symbol, 1) + 1
        else:
            self._legs[context.symbol] = 1
            self._thesis[context.symbol] = EntryThesis(
                strategy_id=best.strategy_id,
                direction=best.direction,
                regime=regime,
                score=best.score,
            )

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
        # The member's own signal, with the orchestrator's ranking attached. Attribution
        # downstream must still name the strategy that actually decided, so only metadata
        # is added — direction, prices and conviction are untouched.
        #
        # The score has to travel because the risk engine sizes on it and cannot recompute
        # it: ranking a candidate needs the whole field, which only exists here.
        return replace(
            best.signal,
            metadata={**best.signal.metadata, "orchestrator_score": str(best.score)},
        )

    def _pyramid_slot(self, context: StrategyContext) -> tuple[bool, str]:
        """Whether this symbol may take another entry leg.

        Only counts legs and the pyramiding switch. Whether a *particular* candidate earns
        the slot is :func:`~quantflow.orchestrator.pyramid.pyramid_verdict`, and every risk
        and exposure rule still runs after that — a slot is permission to be considered,
        never permission to trade.
        """
        if MAX_LEGS_PER_SYMBOL <= 1:
            return False, "pyramiding is disabled (MAX_LEGS_PER_SYMBOL <= 1)"
        legs = self._legs.get(context.symbol, 1)
        if legs >= MAX_LEGS_PER_SYMBOL:
            return False, f"{legs} leg(s) already open, limit {MAX_LEGS_PER_SYMBOL}"
        return True, f"{legs} of {MAX_LEGS_PER_SYMBOL} legs used"

    def _open_position_route(self, context: StrategyContext) -> tuple[Signal | None, bool]:
        """Decide what a symbol that already holds a position should do this bar.

        Returns ``(signal, pyramiding)``. A signal means the answer is settled — the owner
        wants out, or no second leg is available. ``(None, True)`` means the symbol earns a
        full evaluation for an additional leg, which then runs every gate an empty symbol
        would face plus the materially-different test.
        """
        managed = self._manage_open_position(context)
        if managed.direction is SignalDirection.CLOSE:
            return managed, False
        allowed, why = self._pyramid_slot(context)
        if not allowed:
            logger.info("orchestrator.pyramid_declined", symbol=str(context.symbol), reason=why)
            return managed, False
        return None, True

    def _pyramid_admits(
        self, context: StrategyContext, best: Any, regime: MarketRegime
    ) -> Signal | None:
        """Judge a second leg against the thesis already running.

        Returns ``None`` when the leg may proceed, or the hold signal to return when it may
        not. Every decision is logged either way — a refusal is as worth reading as an
        admission, because "why did it not pyramid" is the question this will be asked.
        """
        existing = self._thesis.get(context.symbol)
        position = context.position
        unrealized = position.unrealized_pnl(context.price) if position is not None else ZERO

        if existing is None:
            logger.info(
                "orchestrator.pyramid_declined",
                symbol=str(context.symbol),
                reason="no recorded thesis for the open position to compare against",
            )
            return context.hold("pyramid: no thesis on record", self.strategy_id)

        allowed, why = pyramid_verdict(
            existing,
            strategy_id=best.strategy_id,
            direction=best.direction,
            regime=regime,
            score=best.score,
            unrealized_pnl=unrealized,
        )
        logger.critical(
            "orchestrator.pyramid_decision",
            symbol=str(context.symbol),
            decision="ALLOW" if allowed else "REFUSE",
            reason=why,
            existing_strategy=existing.strategy_id,
            new_strategy=best.strategy_id,
            existing_score=str(existing.score),
            new_score=str(best.score),
            existing_regime=existing.regime.value,
            new_regime=regime.value,
            existing_legs=self._legs.get(context.symbol, 1),
            unrealized_pnl=str(unrealized),
        )
        if not allowed:
            return context.hold(f"pyramid refused: {why}", self.strategy_id)
        return None

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
            self._thesis.pop(context.symbol, None)
            self._legs.pop(context.symbol, None)
        return signal

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def adopt(self, symbol: Symbol, strategy_id: str) -> None:
        """Record an existing position's owner, used when restoring after a restart."""
        self._owners[symbol] = strategy_id

    def sync_owners(self, open_symbols: Collection[Symbol]) -> tuple[Symbol, ...]:
        """Drop ownership of every symbol the venue no longer holds a position in.

        The venue is authoritative about what is open; this map is only a record of which
        member is responsible for each of those positions. Before this existed, ownership
        was released in exactly one place — the owning member returning ``CLOSE`` on a bar
        where the engine still saw the position — so every other way a position can end
        left the symbol owned forever: a venue stop, a take-profit, an intrabar exit, a
        manual close, a liquidation.

        Because an owned symbol counts as an open position in the duplicate guard, the
        effect compounded until nothing could be entered at all. Observed live on
        2026-08-14: 52 of 52 candidates declined for correlation with open positions, with
        zero positions open at the venue.

        Deliberately one-directional. A venue position with no owner is **not** adopted
        here, because this function cannot know which member should manage it, and guessing
        would hand a live position to a strategy that did not open it — the restore path
        (:meth:`on_restore`) has the persisted ``strategy_id`` and is the right place.

        Args:
            open_symbols: Symbols the venue currently holds a position in.

        Returns:
            The symbols whose ownership was released, for logging.

        """
        held = frozenset(open_symbols)
        released = tuple(symbol for symbol in self._owners if symbol not in held)
        for symbol in released:
            del self._owners[symbol]
        if released:
            logger.info(
                "orchestrator.owners_released",
                symbols=[str(symbol) for symbol in released],
                remaining=len(self._owners),
                reason="venue no longer holds a position in these symbols",
            )
        return released

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
        """Engine hook: fold a completed round-trip into its strategy's record.

        Also releases the symbol. A completed round-trip is proof the position is gone, and
        acting on it here frees the symbol immediately rather than leaving it blocked until
        the next venue sync — which matters on a 15m timeframe, where that wait is a whole
        bar of missed entries.
        """
        self._owners.pop(trade.symbol, None)
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


def _count_agreeing_families(candidates: list[Candidate], candidate: Candidate) -> int:
    """How many distinct information sources back this candidate's direction.

    Counted over *families*, not strategies: five moving-average variants agreeing is one
    observation restated five times, and treating it as five was how a single weak
    indicator came to justify an entry.
    """
    return len(
        {
            strategy_family(other.strategy_id)
            for other in candidates
            if other.direction is candidate.direction and other.symbol == candidate.symbol
        }
    )


#: Env switch for short generation. Defaults ON: a long-only book cannot be right in a
#: falling market, and every strategy below already implements its own short path — they
#: were simply never asked for one.
SHORTS_ENABLED_VAR = "QF_ALLOW_SHORT"


def _short_enabled() -> bool:
    """Whether members should be built with short generation allowed."""
    import os

    return os.environ.get(SHORTS_ENABLED_VAR, "true").strip().lower() == "true"


def _create_member(registry: StrategyRegistry, name: str) -> Strategy:
    """Build one member, enabling shorts only where the strategy actually supports them.

    ``allow_short`` is passed ONLY to strategies whose params model declares it. The five
    long-only strategies have no short path, and their params models forbid extra fields —
    passing the flag would either raise or, worse, imply a capability that does not exist.
    Enabling a direction a strategy cannot compute is how a system starts taking trades
    nobody designed.
    """
    if _short_enabled() and "allow_short" in registry.params_model(name).model_fields:
        return registry.create(name, {"allow_short": True})
    return registry.create(name)


def _default_members(excluded: frozenset[str] = DEFAULT_EXCLUDED) -> tuple[Strategy, ...]:
    """Every registered strategy except the excluded ones, in stable order."""
    registry = load_builtin_strategies()
    return tuple(
        _create_member(registry, name) for name in sorted(registry.names()) if name not in excluded
    )


def _members_from_pool(pool: Sequence[str]) -> tuple[Strategy, ...]:
    """Build exactly the named strategies, refusing anything that cannot be honoured.

    Every rejection here is a case where carrying on would produce a roster the operator
    did not ask for: an unknown id would quietly shrink the pool, and self-inclusion would
    recurse. Both are worse than a startup failure.
    """
    from quantflow.core.errors import ValidationError

    registry = load_builtin_strategies()
    known = set(registry.names())
    requested = list(dict.fromkeys(pool))  # de-duplicate, keep the caller's order

    if not requested:
        raise ValidationError("orchestrator pool cannot be empty", field="pool")

    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValidationError(
            f"unknown strategies in orchestrator pool: {', '.join(sorted(unknown))}",
            field="pool",
        )

    excluded = [name for name in requested if name in DEFAULT_EXCLUDED]
    if excluded:
        raise ValidationError(
            f"these cannot be orchestrator members: {', '.join(sorted(excluded))}",
            field="pool",
        )

    return tuple(_create_member(registry, name) for name in sorted(requested))


__all__ = [
    "DEFAULT_EXCLUDED",
    "Decision",
    "OrchestratorParams",
    "StrategyOrchestrator",
]
