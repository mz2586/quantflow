"""AI trading service: a language model advising the existing execution stack.

The model is an advisor at the front of a pipeline, not a replacement for it. Its output
becomes a `Signal`, and a Signal goes through `ExecutionEngine.execute_signal`, which is
risk-gated. There is no path from here to a venue that skips that gate. The model does not
size positions and does not set stops; both are decided downstream by code that never
consults it.

Live execution stays behind the same five-condition interlock as everything else — this
service adds no new way to arm it, and paper is the default. The default LLM provider is
`null`, which returns a deterministic HOLD, so an unconfigured deployment declines to
trade rather than doing something arbitrary.
"""

from __future__ import annotations

from quantflow.aitrader.client import (
    AnthropicClient,
    Completion,
    LLMClient,
    LLMError,
    NullClient,
    OpenAICompatibleClient,
    build_client,
)
from quantflow.aitrader.context import (
    DecisionContext,
    SymbolContext,
    build_symbol_context,
    describe_positions,
    summarise_order_book,
)
from quantflow.aitrader.decision import Action, AIDecision, DecisionError, parse_decision
from quantflow.aitrader.journal import CycleRecord, DecisionJournal
from quantflow.aitrader.prompt import SYSTEM_PROMPT, build_user_prompt
from quantflow.aitrader.service import AITradingService, CycleOutcome, ServiceState

__all__ = [
    "SYSTEM_PROMPT",
    "AIDecision",
    "AITradingService",
    "Action",
    "AnthropicClient",
    "Completion",
    "CycleOutcome",
    "CycleRecord",
    "DecisionContext",
    "DecisionError",
    "DecisionJournal",
    "LLMClient",
    "LLMError",
    "NullClient",
    "OpenAICompatibleClient",
    "ServiceState",
    "SymbolContext",
    "build_client",
    "build_symbol_context",
    "build_user_prompt",
    "describe_positions",
    "parse_decision",
    "summarise_order_book",
]
