"""AI-augmented strategy wrapper.

Wraps any :class:`~quantflow.strategy.base.Strategy` so its signals pass through the AI
decision engine before reaching the engine — and therefore, unchanged, before reaching the
risk engine.

The composition order is the whole design:

    strategy.generate → AI (may veto or shrink) → risk engine (may refuse) → venue

The AI sits **between** the strategy and the risk engine, never around it. It cannot see
the risk engine, cannot call it, and cannot produce anything the risk engine will not
subsequently inspect. A vetoed signal becomes a HOLD; a discounted signal becomes a smaller
version of the same signal. Neither can produce an order the strategy did not ask for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from quantflow.ai.decision import AIAdvice, AIDecisionEngine, assert_risk_reducing
from quantflow.core.logging import get_logger
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import Position
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams

logger = get_logger(__name__)


class AIAugmentedStrategy(Strategy):
    """Applies AI advice to another strategy's signals.

    Transparent by design: it delegates identity, warm-up and parameters to the wrapped
    strategy, so nothing downstream — persistence, attribution, the dashboard — needs to
    know the AI is involved. The signal's ``reason`` records what the AI did, so an
    operator can always reconstruct why a position was smaller than expected.
    """

    strategy_id: ClassVar[str] = "ai_augmented"
    description: ClassVar[str] = "wraps a strategy with AI regime filtering"
    params_model: ClassVar[type[StrategyParams]] = StrategyParams

    def __init__(
        self,
        inner: Strategy,
        engine: AIDecisionEngine,
        *,
        record_advice: bool = True,
    ) -> None:
        # Deliberately bypasses Strategy.__init__: identity comes from the wrapped
        # strategy, so attribution and persisted records keep pointing at the real one.
        self.inner = inner
        self.engine = engine
        self.params = inner.params
        self.record_advice = record_advice
        self.advice_log: list[tuple[Signal, AIAdvice]] = []

    # ------------------------------------------------------------------ #
    # Identity delegates to the wrapped strategy
    # ------------------------------------------------------------------ #
    @property
    def warmup_bars(self) -> int:
        """The larger of the strategy's warm-up and the AI's feature window.

        The AI needs its own history before it can classify a regime; starting to trade
        before then would mean the first trades run with no AI oversight at all, silently.
        """
        from quantflow.ai.regime import MIN_BARS

        return max(self.inner.warmup_bars, MIN_BARS)

    def generate(self, context: StrategyContext) -> Signal:
        """Run the wrapped strategy, then apply AI advice."""
        signal = self.inner.generate(context)
        if not signal.is_actionable:
            return signal

        adjusted, advice = self.engine.apply(signal, context.candles)

        # Redundant with AIAdvice's own validation, deliberately: this is the assertion
        # that catches a future refactor introducing a path where the AI enlarges a trade.
        assert_risk_reducing(signal, adjusted)

        if self.record_advice:
            self.advice_log.append((signal, advice))

        if adjusted is None:
            return context.hold(advice.summary, signal.strategy_id)
        return adjusted

    def evaluate(self, context: StrategyContext) -> Signal:
        """Warm-up and error containment, then generate.

        Overridden so the wrapper's own (longer) warm-up applies and so a failure inside
        the AI is contained exactly as a strategy failure is.
        """
        if len(context.history) < self.warmup_bars:
            return context.hold(
                f"warming up ({len(context.history)}/{self.warmup_bars} bars)",
                self.inner.strategy_id,
            )
        try:
            return self.generate(context)
        except Exception as exc:
            logger.exception(
                "ai_strategy.failed",
                strategy_id=self.inner.strategy_id,
                symbol=str(context.symbol),
                error=str(exc),
            )
            return context.hold(f"AI strategy error: {exc}", self.inner.strategy_id)

    # ------------------------------------------------------------------ #
    # Delegated hooks
    # ------------------------------------------------------------------ #
    def on_start(self, symbols: Sequence[Symbol]) -> None:
        """Forward to the wrapped strategy."""
        self.inner.on_start(symbols)

    def on_fill(self, symbol: Symbol, position: Position) -> None:
        """Forward to the wrapped strategy."""
        self.inner.on_fill(symbol, position)

    def on_finish(self) -> None:
        """Forward to the wrapped strategy."""
        self.inner.on_finish()

    def describe(self) -> dict[str, Any]:
        """Identity of the wrapped strategy, plus the AI configuration."""
        return {
            **self.inner.describe(),
            "ai": self.engine.describe(),
            "wrapped_by": "ai_augmented",
        }

    def advice_summary(self) -> dict[str, Any]:
        """What the AI actually did over the session.

        Surfaced so its effect is measurable rather than assumed: an AI layer that vetoed
        nothing and scaled nothing is a layer that is not earning its complexity.
        """
        if not self.advice_log:
            return {"signals_seen": 0, "vetoed": 0, "scaled": 0, "neutral": 0}
        vetoed = sum(1 for _, advice in self.advice_log if advice.veto)
        scaled = sum(
            1
            for _, advice in self.advice_log
            if not advice.veto and advice.conviction_multiplier < 1
        )
        regimes: dict[str, int] = {}
        for _, advice in self.advice_log:
            regimes[advice.regime.value] = regimes.get(advice.regime.value, 0) + 1
        return {
            "signals_seen": len(self.advice_log),
            "vetoed": vetoed,
            "scaled": scaled,
            "neutral": len(self.advice_log) - vetoed - scaled,
            "regimes": regimes,
        }


def wrap(
    strategy: Strategy,
    *,
    enabled: bool = True,
    veto_counter_trend: bool = False,
) -> Strategy:
    """Wrap a strategy with the default AI engine.

    Returns the strategy unchanged when ``enabled`` is false, so a caller never has to
    branch on whether the AI is switched on.
    """
    if not enabled:
        return strategy
    from quantflow.ai.decision import build_engine

    return AIAugmentedStrategy(
        strategy, build_engine(enabled=True, veto_counter_trend=veto_counter_trend)
    )
