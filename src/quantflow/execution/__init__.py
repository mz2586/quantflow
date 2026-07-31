"""Execution engine: the single path from a strategy signal to a venue."""

from __future__ import annotations

from quantflow.execution.engine import (
    ExecutionEngine,
    ExecutionResult,
    ExecutionStats,
    build_exit_request,
    should_trigger_stop,
)

__all__ = [
    "ExecutionEngine",
    "ExecutionResult",
    "ExecutionStats",
    "build_exit_request",
    "should_trigger_stop",
]
