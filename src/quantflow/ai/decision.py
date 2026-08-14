"""The AI decision engine.

**The AI advises. It never trades.**

This is the single most important property in the module. The engine produces an
:class:`AIAdvice` object that can *veto* a signal or *reduce* its conviction — it cannot
create a signal, cannot increase conviction beyond what the strategy asked for, and cannot
place an order. Every decision it touches still flows through the strategy contract and
then through the risk engine, unchanged.

The reasoning is straightforward. A model that can only reduce exposure has a bounded
worst case: the worst a broken model can do is stop the system trading. A model that can
*increase* exposure has an unbounded worst case, and no amount of validation makes that
acceptable when the failure mode is losing money that belongs to someone.

So the interface is deliberately asymmetric:

- ``veto`` → the signal is dropped.
- ``conviction_multiplier`` ∈ [0, 1] → conviction can only shrink.
- There is no field that can add a position, flip a side, or widen a stop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from quantflow.ai.regime import (
    RegimeDetector,
    RegimeObservation,
    RuleBasedRegimeDetector,
)
from quantflow.core.clock import Clock, SystemClock
from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import MarketRegime, SignalDirection
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal

logger = get_logger(__name__)

#: Regimes each signal direction is considered compatible with. A long in a bear trend is
#: not forbidden — it is discounted, because counter-trend entries are lower-probability
#: rather than invalid.
FAVOURABLE_REGIMES: dict[SignalDirection, frozenset[MarketRegime]] = {
    SignalDirection.LONG: frozenset({MarketRegime.BULL_TREND, MarketRegime.RANGE}),
    SignalDirection.SHORT: frozenset({MarketRegime.BEAR_TREND, MarketRegime.RANGE}),
}

#: Conviction multiplier applied when a signal opposes the prevailing trend.
COUNTER_TREND_MULTIPLIER = Decimal("0.5")

#: Conviction multiplier applied in a high-volatility regime, regardless of direction.
HIGH_VOLATILITY_MULTIPLIER = Decimal("0.5")

#: Below this regime confidence, the regime is treated as unknown and left alone rather
#: than acted on. A low-confidence label is worse than no label: it invites a decision.
MIN_ACTIONABLE_CONFIDENCE = Decimal("0.55")


@dataclass(frozen=True, slots=True)
class AIAdvice:
    """The engine's opinion on one signal.

    Structurally incapable of increasing risk: there is no field here that can create a
    position, flip a direction, widen a stop or raise conviction above 1.
    """

    veto: bool = False
    conviction_multiplier: Decimal = ONE
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: Decimal = ZERO
    reasons: tuple[str, ...] = field(default_factory=tuple)
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce the risk-reducing-only invariant.

        Raises:
            ValidationError: if the multiplier falls outside ``[0, 1]``. A value above 1
                would mean the AI had increased exposure, which this design forbids
                outright rather than merely discouraging.

        """
        if not (ZERO <= self.conviction_multiplier <= ONE):
            raise ValidationError(
                "AI conviction_multiplier must be in [0, 1]; the AI may only reduce risk, "
                f"never increase it (got {self.conviction_multiplier})",
                multiplier=str(self.conviction_multiplier),
            )

    @property
    def is_neutral(self) -> bool:
        """Whether the advice changes nothing."""
        return not self.veto and self.conviction_multiplier == ONE

    @property
    def summary(self) -> str:
        """One-line explanation, carried onto the signal for the audit trail."""
        if self.veto:
            return "AI veto: " + "; ".join(self.reasons)
        if self.conviction_multiplier < ONE:
            return f"AI scaled conviction x{self.conviction_multiplier}: " + "; ".join(self.reasons)
        return "AI: no adjustment"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging and the API."""
        return {
            "veto": self.veto,
            "conviction_multiplier": str(self.conviction_multiplier),
            "regime": self.regime.value,
            "regime_confidence": str(self.regime_confidence),
            "reasons": list(self.reasons),
        }


@runtime_checkable
class AdvisorProtocol(Protocol):
    """Something that can advise on a signal."""

    @property
    def name(self) -> str:
        """Advisor identifier."""
        ...

    def advise(self, signal: Signal, candles: Sequence[Candle]) -> AIAdvice:
        """Produce advice for one signal."""
        ...


class RegimeAdvisor:
    """Advises based on the detected market regime.

    Two adjustments, both risk-reducing:

    - A signal against the prevailing trend has its conviction halved.
    - Any signal in a high-volatility regime has its conviction halved.

    Vetoes are reserved for the case where the regime is confidently and directly opposed
    to the trade — a long in a confident bear trend. Everything else is a discount, because
    a strategy that is never allowed to trade against the trend is a strategy that cannot
    catch a reversal.
    """

    __slots__ = ("_detector", "_veto_counter_trend")

    def __init__(
        self,
        detector: RegimeDetector | None = None,
        *,
        veto_counter_trend: bool = False,
    ) -> None:
        self._detector = detector or RuleBasedRegimeDetector()
        self._veto_counter_trend = veto_counter_trend

    @property
    def name(self) -> str:
        """Advisor identifier."""
        return "regime"

    def advise(self, signal: Signal, candles: Sequence[Candle]) -> AIAdvice:
        """Adjust a signal for the prevailing regime."""
        try:
            observation = self._detector.detect(candles)
        except InsufficientDataError:
            # Not enough history to have an opinion. Silence, not a guess.
            return AIAdvice(reasons=("insufficient history for regime detection",))

        if observation.confidence < MIN_ACTIONABLE_CONFIDENCE:
            return AIAdvice(
                regime=observation.regime,
                regime_confidence=observation.confidence,
                reasons=(
                    f"regime {observation.regime.value} at confidence "
                    f"{observation.confidence:.2f} is below the actionable threshold",
                ),
                features=observation.features.to_dict(),
            )

        return self._evaluate(signal, observation)

    def _evaluate(self, signal: Signal, observation: RegimeObservation) -> AIAdvice:
        multiplier = ONE
        reasons: list[str] = []
        veto = False

        if observation.regime is MarketRegime.HIGH_VOLATILITY:
            multiplier *= HIGH_VOLATILITY_MULTIPLIER
            reasons.append(f"high-volatility regime ({observation.reason}); halving conviction")

        favourable = FAVOURABLE_REGIMES.get(signal.direction, frozenset())
        counter_trend = (
            signal.direction in (SignalDirection.LONG, SignalDirection.SHORT)
            and observation.regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)
            and observation.regime not in favourable
        )

        if counter_trend:
            if self._veto_counter_trend:
                veto = True
                reasons.append(
                    f"{signal.direction.value} entry opposes a confident {observation.regime.value}"
                )
            else:
                multiplier *= COUNTER_TREND_MULTIPLIER
                reasons.append(
                    f"{signal.direction.value} entry opposes the "
                    f"{observation.regime.value}; halving conviction"
                )

        return AIAdvice(
            veto=veto,
            conviction_multiplier=multiplier,
            regime=observation.regime,
            regime_confidence=observation.confidence,
            reasons=tuple(reasons),
            features=observation.features.to_dict(),
        )


@dataclass(slots=True)
class AIDecisionEngine:
    """Combines advisors into a single, always risk-reducing opinion.

    When advisors disagree, the **most conservative** answer wins: any veto is a veto, and
    multipliers compound rather than average. That is the correct combination rule for a
    system whose advisors can only reduce risk — averaging would let one optimistic advisor
    dilute another's warning.
    """

    advisors: Sequence[AdvisorProtocol] = field(default_factory=tuple)
    clock: Clock = field(default_factory=SystemClock)
    enabled: bool = True
    #: Floor on the combined multiplier. Without it, three independent halvings produce a
    #: position too small to clear the venue minimum, which reads as "the AI broke the
    #: system" rather than "the AI was cautious".
    min_multiplier: Decimal = Decimal("0.1")

    def advise(self, signal: Signal, candles: Sequence[Candle]) -> AIAdvice:
        """Produce combined advice for one signal."""
        if not self.enabled or not self.advisors or not signal.is_actionable:
            return AIAdvice()

        multiplier = ONE
        reasons: list[str] = []
        features: dict[str, float] = {}
        regime = MarketRegime.UNKNOWN
        confidence = ZERO
        veto = False

        for advisor in self.advisors:
            try:
                advice = advisor.advise(signal, candles)
            except Exception as exc:
                # Failing open (ignoring the advisor) is the right call *because* advisors
                # can only reduce risk: losing one is a lost safety check, not a lost
                # trading decision.
                logger.exception("ai.advisor_failed", advisor=advisor.name, error=str(exc))
                continue

            veto = veto or advice.veto
            multiplier *= advice.conviction_multiplier
            reasons.extend(f"[{advisor.name}] {reason}" for reason in advice.reasons)
            features.update(advice.features)
            if advice.regime is not MarketRegime.UNKNOWN:
                regime = advice.regime
                confidence = advice.regime_confidence

        return AIAdvice(
            veto=veto,
            conviction_multiplier=max(self.min_multiplier, multiplier),
            regime=regime,
            regime_confidence=confidence,
            reasons=tuple(reasons),
            features=features,
        )

    def apply(self, signal: Signal, candles: Sequence[Candle]) -> tuple[Signal | None, AIAdvice]:
        """Apply advice to a signal.

        Returns:
            ``(adjusted_signal, advice)``. The signal is ``None`` when vetoed. The adjusted
            signal is otherwise identical except for a **reduced** conviction and an
            appended reason — direction, stop, target and symbol are never modified, so the
            risk engine receives exactly what the strategy intended, only smaller.

        """
        advice = self.advise(signal, candles)

        if advice.veto:
            logger.info(
                "ai.signal_vetoed",
                symbol=str(signal.symbol),
                direction=signal.direction.value,
                strategy_id=signal.strategy_id,
                reasons=list(advice.reasons),
            )
            return None, advice

        if advice.is_neutral:
            return signal, advice

        from dataclasses import replace

        adjusted = replace(
            signal,
            conviction=signal.conviction * advice.conviction_multiplier,
            regime=advice.regime,
            reason=f"{signal.reason} | {advice.summary}"[:500],
        )
        logger.debug(
            "ai.conviction_scaled",
            symbol=str(signal.symbol),
            multiplier=str(advice.conviction_multiplier),
            before=str(signal.conviction),
            after=str(adjusted.conviction),
        )
        return adjusted, advice

    def describe(self) -> dict[str, Any]:
        """Configuration summary for the API."""
        return {
            "enabled": self.enabled,
            "advisors": [advisor.name for advisor in self.advisors],
            "min_multiplier": str(self.min_multiplier),
            "can_increase_risk": False,
        }


def build_engine(
    *,
    enabled: bool = True,
    detector: RegimeDetector | None = None,
    veto_counter_trend: bool = False,
    clock: Clock | None = None,
) -> AIDecisionEngine:
    """Construct the default engine: regime advice only."""
    return AIDecisionEngine(
        advisors=(RegimeAdvisor(detector, veto_counter_trend=veto_counter_trend),),
        clock=clock or SystemClock(),
        enabled=enabled,
    )


def assert_risk_reducing(original: Signal, adjusted: Signal | None) -> None:
    """Assert the AI only ever reduced risk.

    Called after every adjustment. Deliberately redundant with :class:`AIAdvice`'s own
    validation: this is the check that catches a future refactor introducing a path where
    the AI can enlarge a position.

    Raises:
        ValidationError: if the adjusted signal is riskier than the original in any way.

    """
    if adjusted is None:
        return
    if adjusted.conviction > original.conviction:
        raise ValidationError(
            "AI increased conviction, which is forbidden: "
            f"{original.conviction} -> {adjusted.conviction}",
            symbol=str(original.symbol),
        )
    if adjusted.direction is not original.direction:
        raise ValidationError(
            "AI changed the signal direction, which is forbidden: "
            f"{original.direction.value} -> {adjusted.direction.value}",
            symbol=str(original.symbol),
        )
    if adjusted.symbol != original.symbol:
        raise ValidationError("AI changed the signal symbol, which is forbidden")
    if original.stop_loss_price is not None and adjusted.stop_loss_price != (
        original.stop_loss_price
    ):
        raise ValidationError(
            "AI modified the stop loss, which is forbidden", symbol=str(original.symbol)
        )


def observation_timestamp(candles: Sequence[Candle]) -> datetime | None:
    """Close time of the most recent bar, if any."""
    return candles[-1].close_time if candles else None
