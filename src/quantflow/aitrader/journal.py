"""The audit trail: prompt, response, order and resulting PnL.

An AI trading decision that cannot be reconstructed afterwards is not reviewable, and an
unreviewable bot is one you cannot debug and should not run. So every cycle is recorded
whole — including the exact prompt, because "why did it buy" is unanswerable without the
input the model actually saw, and prompts change as the context builder changes.

Records are append-only and PnL is attached *later*, when the position closes. The result
of a decision is not known at the time it is made, and a journal that pretended otherwise
would be recording a guess.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO

logger = get_logger(__name__)

#: Prompts are long. Truncating in memory keeps a long-running service from growing
#: without bound; the file sink keeps the full text.
MAX_INMEMORY_PROMPT_CHARS = 4000


@dataclass(slots=True)
class CycleRecord:
    """One decision cycle, start to finish."""

    started_at: datetime
    model: str
    mode: str
    system_prompt: str
    user_prompt: str
    response: str
    context: dict[str, Any]
    outcome: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Filled in when the resulting position closes. None means "not yet known", which is
    #: different from zero and must not be rendered as it.
    realized_pnl: Decimal | None = None
    order_id: str | None = None

    def to_dict(self, *, full_prompts: bool = True) -> dict[str, Any]:
        """Serialise. Set ``full_prompts=False`` for the in-memory tail."""
        system = self.system_prompt
        user = self.user_prompt
        if not full_prompts:
            system = system[:MAX_INMEMORY_PROMPT_CHARS]
            user = user[:MAX_INMEMORY_PROMPT_CHARS]
        return {
            "started_at": self.started_at.isoformat(),
            "model": self.model,
            "mode": self.mode,
            "system_prompt": system,
            "user_prompt": user,
            "response": self.response,
            "context": self.context,
            "outcome": self.outcome,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "order_id": self.order_id,
        }


@dataclass
class DecisionJournal:
    """Append-only record of every cycle.

    Keeps a bounded tail in memory for the API and, when a path is configured, writes
    every record to newline-delimited JSON. The file is the audit trail; memory is just a
    window onto it.
    """

    path: Path | None = None
    max_in_memory: int = 200
    _records: list[CycleRecord] = field(default_factory=list, init=False)

    def append(self, record: CycleRecord) -> None:
        """Record a cycle."""
        self._records.append(record)
        if len(self._records) > self.max_in_memory:
            del self._records[: len(self._records) - self.max_in_memory]

        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict()) + "\n")
        except OSError as exc:
            # Losing the audit trail must not stop trading, but it must be loud: a bot
            # running without a journal is one nobody can review afterwards.
            logger.exception("aitrader.journal_write_failed", path=str(self.path), error=str(exc))

    def attach_pnl(self, order_id: str, realized_pnl: Decimal) -> bool:
        """Attach a realised result to the cycle that produced an order.

        Returns True when a matching record was found. The result of a decision is not
        known when the decision is made, so it is written back on close rather than
        guessed at the time.
        """
        for record in reversed(self._records):
            if record.order_id == order_id:
                record.realized_pnl = realized_pnl
                return True
        return False

    def note_order(self, order_id: str) -> None:
        """Tag the most recent cycle with the order it produced."""
        if self._records:
            self._records[-1].order_id = order_id

    @property
    def records(self) -> tuple[CycleRecord, ...]:
        """The in-memory tail, oldest first."""
        return tuple(self._records)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent cycles, newest first, with prompts truncated."""
        return [record.to_dict(full_prompts=False) for record in reversed(self._records[-limit:])]

    def summary(self) -> dict[str, Any]:
        """Aggregate counters over the retained window."""
        traded = [r for r in self._records if r.outcome.get("traded")]
        settled = [r for r in self._records if r.realized_pnl is not None]
        total = sum((r.realized_pnl or ZERO for r in settled), ZERO)
        tokens_in = sum(r.input_tokens or 0 for r in self._records)
        tokens_out = sum(r.output_tokens or 0 for r in self._records)
        return {
            "cycles_retained": len(self._records),
            "cycles_that_traded": len(traded),
            "settled_trades": len(settled),
            "realized_pnl": str(total),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CycleRecord]:
        return iter(self._records)


__all__ = ["MAX_INMEMORY_PROMPT_CHARS", "CycleRecord", "DecisionJournal"]
