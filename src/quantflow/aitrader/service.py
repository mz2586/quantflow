"""The AI trading service: one decision cycle, end to end.

Build context → prompt → model → validate → risk engine → execution → journal.

The model is an *advisor at the front of an existing pipeline*, not a replacement for it.
Its output becomes a `Signal`, and a Signal goes through `ExecutionEngine.execute_signal`,
which is risk-gated. There is no path from here to a venue that skips that. The AI does
not size the position and does not set the stop; both are decided downstream by code that
does not consult it.

Live execution stays behind the same five-condition interlock as everything else. This
service adds no new way to arm it, and paper is the default.

Every cycle is journalled whole — prompt, raw response, decision, execution result — for
the same reason the rest of the platform records its reasoning: a decision that cannot be
reviewed after a loss is not auditable, and an unauditable trading bot is one you cannot
debug and should not run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.aitrader.client import LLMClient, LLMError
from quantflow.aitrader.context import (
    DecisionContext,
    build_symbol_context,
    describe_positions,
)
from quantflow.aitrader.decision import Action, AIDecision, DecisionError, parse_decision
from quantflow.aitrader.journal import CycleRecord, DecisionJournal
from quantflow.aitrader.prompt import SYSTEM_PROMPT, build_user_prompt
from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import LLMSettings, TradingMode
from quantflow.core.errors import QuantFlowError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal
from quantflow.execution.engine import ExecutionEngine, ExecutionResult
from quantflow.intelligence.derivatives import DerivativesSource
from quantflow.live.runner import live_trading_env_enabled

logger = get_logger(__name__)

#: What the service reads from a market-data source. Narrow on purpose: the service must
#: not be able to place an order through this, only observe.
MARKET_METHODS = ("fetch_candles", "fetch_order_book")


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """What one decision cycle produced."""

    decision: AIDecision | None
    error: DecisionError | None
    execution: ExecutionResult | None
    skipped_reason: str | None = None

    @property
    def traded(self) -> bool:
        """Whether an order actually reached the execution layer."""
        return self.execution is not None and self.execution.submitted

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the journal and the API."""
        return {
            "decision": self.decision.to_dict() if self.decision else None,
            "error": self.error.to_dict() if self.error else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "skipped_reason": self.skipped_reason,
            "traded": self.traded,
        }


@dataclass(slots=True)
class ServiceState:
    """Counters for one run of the service."""

    cycles: int = 0
    decisions: int = 0
    holds: int = 0
    orders: int = 0
    parse_failures: int = 0
    model_failures: int = 0
    risk_rejections: int = 0
    below_confidence: int = 0
    started_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API."""
        return {
            "cycles": self.cycles,
            "decisions": self.decisions,
            "holds": self.holds,
            "orders": self.orders,
            "parse_failures": self.parse_failures,
            "model_failures": self.model_failures,
            "risk_rejections": self.risk_rejections,
            "below_confidence": self.below_confidence,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


class AITradingService:
    """Runs the AI decision loop against the existing execution stack."""

    def __init__(
        self,
        *,
        client: LLMClient,
        execution: ExecutionEngine,
        market: Any,
        symbols: Sequence[Symbol],
        settings: LLMSettings,
        journal: DecisionJournal | None = None,
        derivatives: DerivativesSource | None = None,
        clock: Clock | None = None,
        timeframe: Any = None,
    ) -> None:
        self._client = client
        self._execution = execution
        self._market = market
        self._symbols = tuple(symbols)
        self._settings = settings
        self._journal = journal or DecisionJournal()
        self._derivatives = derivatives
        self._clock = clock or SystemClock()
        self._timeframe = timeframe
        self._state = ServiceState()
        self._stop = asyncio.Event()

    @property
    def state(self) -> ServiceState:
        """Counters for this run."""
        return self._state

    @property
    def journal(self) -> DecisionJournal:
        """The cycle journal."""
        return self._journal

    @property
    def mode(self) -> TradingMode:
        """The execution mode. Paper unless explicitly armed elsewhere."""
        return self._execution.mode

    def describe(self) -> dict[str, Any]:
        """Service status, including why live trading is or is not permitted."""
        return {
            "mode": self.mode.value,
            "model": self._client.model,
            "symbols": [str(symbol) for symbol in self._symbols],
            "min_confidence": str(self._settings.min_confidence),
            "interval_seconds": self._settings.interval_seconds,
            "live_env_flag": live_trading_env_enabled(),
            "live_trading_permitted": self.mode is TradingMode.LIVE and live_trading_env_enabled(),
            "state": self._state.to_dict(),
        }

    async def run_cycle(self) -> CycleOutcome:
        """Run one full decision cycle."""
        started = self._clock.now()
        self._state.cycles += 1

        context = await self._build_context(started)
        system = SYSTEM_PROMPT
        user = build_user_prompt(context)

        try:
            completion = await self._client.complete(system=system, user=user)
        except LLMError as exc:
            self._state.model_failures += 1
            logger.warning("aitrader.model_failed", error=str(exc))
            outcome = CycleOutcome(
                decision=None,
                error=DecisionError(f"model call failed: {exc}"),
                execution=None,
                skipped_reason="model call failed",
            )
            self._record(started, context, system, user, "", outcome, None)
            return outcome

        parsed = parse_decision(completion.text, allowed=self._symbols)
        if isinstance(parsed, DecisionError):
            # A malformed decision is a HOLD, not a retry: guessing at intent is how an
            # unintended position gets opened.
            self._state.parse_failures += 1
            logger.warning("aitrader.unparseable", reason=parsed.reason)
            outcome = CycleOutcome(
                decision=None,
                error=parsed,
                execution=None,
                skipped_reason=f"unparseable response: {parsed.reason}",
            )
            self._record(started, context, system, user, completion.text, outcome, completion)
            return outcome

        self._state.decisions += 1
        outcome = await self._act_on(parsed, context)
        self._record(started, context, system, user, completion.text, outcome, completion)
        return outcome

    async def run_forever(self) -> ServiceState:
        """Run cycles until stopped."""
        self._state.started_at = self._clock.now()
        logger.info(
            "aitrader.started",
            mode=self.mode.value,
            model=self._client.model,
            symbols=[str(symbol) for symbol in self._symbols],
            interval_seconds=self._settings.interval_seconds,
        )
        while not self._stop.is_set():
            try:
                await self.run_cycle()
            except QuantFlowError as exc:
                # One bad cycle must not end the service; the next one may succeed.
                logger.exception("aitrader.cycle_failed", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._settings.interval_seconds)
            except TimeoutError:
                continue
        logger.info("aitrader.stopped", **self._state.to_dict())
        return self._state

    def stop(self) -> None:
        """Ask the loop to finish after the current cycle."""
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _build_context(self, now: datetime) -> DecisionContext:
        """Gather account state and market data for every permitted symbol."""
        snapshot = self._execution.portfolio.snapshot(now)
        contexts = []
        for symbol in self._symbols:
            candles = await self._fetch_candles(symbol)
            book = await self._fetch_order_book(symbol)
            funding = None
            open_interest = None
            if self._derivatives is not None:
                funding = await self._derivatives.fetch_funding_rate(str(symbol))
                open_interest = await self._derivatives.fetch_open_interest(str(symbol))
            contexts.append(
                build_symbol_context(
                    symbol,
                    candles,
                    order_book=book,
                    funding=funding,
                    open_interest=open_interest,
                )
            )

        return DecisionContext(
            observed_at=now,
            base_currency=snapshot.base_currency,
            equity=snapshot.equity,
            cash=snapshot.cash,
            positions=describe_positions(snapshot),
            symbols=tuple(contexts),
            mode=self.mode.value,
        )

    async def _fetch_candles(self, symbol: Symbol) -> list[Candle]:
        """Recent candles, or an empty list when the feed fails."""
        try:
            candles = await self._market.fetch_candles(
                symbol, self._timeframe, limit=self._settings.candles_in_prompt
            )
        except QuantFlowError as exc:
            logger.warning("aitrader.candles_failed", symbol=str(symbol), error=str(exc))
            return []
        return list(candles)

    async def _fetch_order_book(self, symbol: Symbol) -> Any:
        """Order book, or ``None`` when unavailable.

        A missing book is reported to the model rather than hidden, so the absence is
        something it can weigh instead of something it silently reasons past.
        """
        fetch = getattr(self._market, "fetch_order_book", None)
        if fetch is None:
            return None
        try:
            return await fetch(symbol)
        except QuantFlowError as exc:
            logger.warning("aitrader.order_book_failed", symbol=str(symbol), error=str(exc))
            return None

    async def _act_on(self, decision: AIDecision, context: DecisionContext) -> CycleOutcome:
        """Turn a validated decision into an execution attempt, or explain the refusal."""
        if decision.action is Action.HOLD:
            self._state.holds += 1
            return CycleOutcome(decision, None, None, skipped_reason="model chose HOLD")

        if not decision.meets(self._settings.min_confidence):
            self._state.below_confidence += 1
            return CycleOutcome(
                decision,
                None,
                None,
                skipped_reason=(
                    f"confidence {decision.confidence} below the "
                    f"{self._settings.min_confidence} floor"
                ),
            )

        reference = self._reference_price(decision.symbol, context)
        if reference <= ZERO:
            return CycleOutcome(decision, None, None, skipped_reason="no reference price available")

        signal = Signal(
            symbol=decision.symbol,
            direction=(
                SignalDirection.LONG if decision.action is Action.BUY else SignalDirection.CLOSE
            ),
            timestamp=self._clock.now(),
            strategy_id="ai_trader",
            conviction=decision.confidence,
            reference_price=reference,
            reason=decision.reason,
            metadata={"model": self._client.model, "action": decision.action.value},
        )

        # The only route to a venue, and it is risk-gated. Sizing and the protective stop
        # are applied inside; nothing the model returned can influence either.
        result = await self._execution.execute_signal(signal, reference_price=reference)
        if result.submitted:
            self._state.orders += 1
        elif result.rejected_by_risk:
            self._state.risk_rejections += 1
        return CycleOutcome(decision, None, result)

    def _reference_price(self, symbol: Symbol, context: DecisionContext) -> Decimal:
        """Last traded price for the decided symbol."""
        for item in context.symbols:
            if item.symbol == symbol:
                return item.last_price
        return ZERO

    def _record(
        self,
        started: datetime,
        context: DecisionContext,
        system: str,
        user: str,
        response: str,
        outcome: CycleOutcome,
        completion: Any,
    ) -> None:
        """Journal the whole cycle."""
        self._journal.append(
            CycleRecord(
                started_at=started,
                model=self._client.model,
                mode=self.mode.value,
                system_prompt=system,
                user_prompt=user,
                response=response,
                context=context.to_dict(),
                outcome=outcome.to_dict(),
                input_tokens=getattr(completion, "input_tokens", None),
                output_tokens=getattr(completion, "output_tokens", None),
            )
        )


__all__ = ["AITradingService", "CycleOutcome", "ServiceState"]
