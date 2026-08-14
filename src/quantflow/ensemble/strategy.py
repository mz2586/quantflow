"""An ensemble that trades several strategies together, or nothing at all.

Members vote; votes are weighted by each member's historical risk-adjusted performance;
and the ensemble only acts when the weighted agreement clears a floor. Below that floor
it holds — which is the entire point, because "the members disagree" is information, and
acting on a split decision means taking a position nobody actually believed in.

The ensemble is itself a `Strategy`. It emits signals and cannot place orders, size
positions or reach the exchange, exactly like every member. Combining strategies does not
earn a shortcut around the risk engine, and structurally it does not get one: the
composite signal goes through the same gate as any other.

Disagreement is not averaged away. A member voting to go long and another voting to go
short do not produce a small long — they produce no trade, because there is no such thing
as being slightly right about direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import Field

from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.ensemble.weights import WeightSet
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams

logger = get_logger(__name__)

ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class Vote:
    """One member's contribution to a decision."""

    strategy_id: str
    direction: SignalDirection
    weight: Decimal
    conviction: Decimal
    reason: str

    @property
    def influence(self) -> Decimal:
        """Weight scaled by the member's own conviction."""
        return self.weight * self.conviction


@dataclass(frozen=True, slots=True)
class EnsembleDecision:
    """What the ensemble concluded, and from what.

    Kept whole rather than collapsed to a signal so a decision — including a decision not
    to trade — can be explained afterwards. "Why did nothing happen at 14:00" is the
    hardest question to answer about an ensemble, and this is the answer.
    """

    direction: SignalDirection
    confidence: Decimal
    votes: tuple[Vote, ...]
    participating: int
    abstaining: int
    reason: str

    @property
    def is_actionable(self) -> bool:
        """Whether the ensemble decided to do anything."""
        return self.direction is not SignalDirection.HOLD

    def explain(self) -> str:
        """One line describing the decision and the split behind it."""
        breakdown = ", ".join(
            f"{vote.strategy_id}={vote.direction.value}@{vote.weight:.2f}"
            for vote in self.votes
            if vote.direction is not SignalDirection.HOLD
        )
        return (
            f"{self.direction.value} at {self.confidence:.0%} confidence "
            f"({self.participating} voting, {self.abstaining} abstaining)"
            + (f": {breakdown}" if breakdown else "")
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the journal and the API."""
        return {
            "direction": self.direction.value,
            "confidence": str(self.confidence),
            "participating": self.participating,
            "abstaining": self.abstaining,
            "reason": self.reason,
            "votes": [
                {
                    "strategy_id": vote.strategy_id,
                    "direction": vote.direction.value,
                    "weight": str(vote.weight),
                    "conviction": str(vote.conviction),
                }
                for vote in self.votes
            ],
        }


class EnsembleParams(StrategyParams):
    """Parameters for :class:`EnsembleStrategy`."""

    #: Weighted agreement required before the ensemble will act. Below this it holds.
    min_confidence: Decimal = Field(default=Decimal("0.55"), gt=0, le=1)
    #: Members that must produce a non-hold vote before a decision is considered at all.
    #: One member voting while four abstain is not a consensus, whatever its weight.
    min_participants: int = Field(default=2, ge=1, le=50)
    #: Whether an opposing vote cancels agreement outright.
    require_no_dissent: bool = False


class EnsembleStrategy(Strategy):
    """Weighted vote across member strategies, holding when confidence is low."""

    strategy_id = "ensemble"
    description = "Weighted vote across member strategies, with a no-trade confidence floor"
    params_model = EnsembleParams

    params: EnsembleParams

    def __init__(
        self,
        members: Sequence[Strategy],
        weights: WeightSet,
        params: EnsembleParams | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(params)
        if not members:
            from quantflow.core.errors import ValidationError

            raise ValidationError("an ensemble needs at least one member", field="members")
        self._members = tuple(members)
        self._weights = weights
        self._last_decision: EnsembleDecision | None = None

    @property
    def members(self) -> tuple[Strategy, ...]:
        """The member strategies."""
        return self._members

    @property
    def weights(self) -> WeightSet:
        """The current allocation across members."""
        return self._weights

    @property
    def last_decision(self) -> EnsembleDecision | None:
        """The most recent decision, for journalling and explanation."""
        return self._last_decision

    @property
    def warmup_bars(self) -> int:
        """The longest warm-up among members.

        The ensemble cannot decide until every member can: acting while some are still
        warming would silently weight the ensemble toward whichever happened to be ready.
        """
        return max(member.warmup_bars for member in self._members)

    def generate(self, context: StrategyContext) -> Signal:
        """Poll every member, weigh the votes, and act only on real agreement."""
        decision = self.decide(context)
        self._last_decision = decision

        if not decision.is_actionable:
            return context.hold(decision.reason, self.strategy_id)

        # Protective levels come from the highest-influence member that supplied them.
        # Averaging stop prices across members would invent a level none of them chose.
        anchor = self._anchor_signal(context, decision)
        return Signal(
            symbol=context.symbol,
            direction=decision.direction,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            conviction=decision.confidence,
            reference_price=context.price,
            stop_loss_price=anchor.stop_loss_price if anchor else None,
            take_profit_price=anchor.take_profit_price if anchor else None,
            reason=decision.explain(),
        )

    def decide(self, context: StrategyContext) -> EnsembleDecision:
        """Collect and weigh member votes without emitting a signal."""
        votes: list[Vote] = []
        for member in self._members:
            weight = self._weights.weight_for(member.strategy_id)
            if weight <= ZERO:
                continue
            signal = member.evaluate(context)
            votes.append(
                Vote(
                    strategy_id=member.strategy_id,
                    direction=signal.direction,
                    weight=weight,
                    conviction=signal.conviction,
                    reason=signal.reason,
                )
            )

        active = [vote for vote in votes if vote.direction is not SignalDirection.HOLD]
        abstaining = len(votes) - len(active)

        if not active:
            return EnsembleDecision(
                direction=SignalDirection.HOLD,
                confidence=ZERO,
                votes=tuple(votes),
                participating=0,
                abstaining=abstaining,
                reason="no member voted to act",
            )

        if len(active) < self.params.min_participants:
            return EnsembleDecision(
                direction=SignalDirection.HOLD,
                confidence=ZERO,
                votes=tuple(votes),
                participating=len(active),
                abstaining=abstaining,
                reason=(
                    f"only {len(active)} member(s) voted; {self.params.min_participants} required"
                ),
            )

        tally: dict[SignalDirection, Decimal] = {}
        for vote in active:
            tally[vote.direction] = tally.get(vote.direction, ZERO) + vote.influence

        total = sum(tally.values(), ZERO)
        winner, winning = max(tally.items(), key=lambda pair: pair[1])
        confidence = safe_divide(winning, total)

        if self.params.require_no_dissent and len(tally) > 1:
            return EnsembleDecision(
                direction=SignalDirection.HOLD,
                confidence=confidence,
                votes=tuple(votes),
                participating=len(active),
                abstaining=abstaining,
                reason=(
                    f"members disagree across {len(tally)} directions and dissent is not allowed"
                ),
            )

        if confidence < self.params.min_confidence:
            return EnsembleDecision(
                direction=SignalDirection.HOLD,
                confidence=confidence,
                votes=tuple(votes),
                participating=len(active),
                abstaining=abstaining,
                reason=(
                    f"agreement {confidence:.0%} is below the "
                    f"{self.params.min_confidence:.0%} floor; no trade"
                ),
            )

        return EnsembleDecision(
            direction=winner,
            confidence=confidence,
            votes=tuple(votes),
            participating=len(active),
            abstaining=abstaining,
            reason=f"{confidence:.0%} weighted agreement on {winner.value}",
        )

    def _anchor_signal(self, context: StrategyContext, decision: EnsembleDecision) -> Signal | None:
        """Re-poll the most influential agreeing member for its protective levels."""
        agreeing = [vote for vote in decision.votes if vote.direction is decision.direction]
        if not agreeing:
            return None
        best = max(agreeing, key=lambda vote: vote.influence)
        for member in self._members:
            if member.strategy_id == best.strategy_id:
                return member.evaluate(context)
        return None

    def on_start(self, symbols: Sequence[Any]) -> None:
        """Forward to every member."""
        for member in self._members:
            member.on_start(symbols)

    def on_finish(self) -> None:
        """Forward to every member."""
        for member in self._members:
            member.on_finish()


__all__ = ["EnsembleDecision", "EnsembleParams", "EnsembleStrategy", "Vote"]
