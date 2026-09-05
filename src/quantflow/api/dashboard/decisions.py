"""Why the engine is not trading, reconstructed from the orchestrator's own log.

When a trading bot is flat, the only question that matters is *why*. "No open positions"
is compatible with a healthy engine correctly declining a bad market and with an engine
that has silently stopped evaluating anything at all, and a dashboard that cannot tell
those apart is worse than no dashboard: it invites the operator to assume the first.

The orchestrator already records every bar it declines, with the reason attached. Those
records are structured log events, not database rows, so this module reads them back:

``orchestrator.selected``
    A candidate won. Carries strategy, direction, confidence, score, the component scores
    behind the score, and the runner-up.
``orchestrator.all_gated``
    Every candidate failed the economic gates. Carries the count and the first reason.
``orchestrator.all_deselected``
    Every candidate failed confluence, regime expectancy or correlation.
``orchestrator.no_trade``
    A candidate led the field but scored below the floor.
``risk.order_denied``
    Selection succeeded and the risk engine refused the resulting order.

Reading is incremental. The log is hundreds of megabytes and grows continuously, so the
reader remembers how far it has consumed and, on every refresh after the first, reads only
the bytes appended since. A dashboard panel must never cost a full-file scan.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from quantflow.core.logging import get_logger

logger = get_logger(__name__)

#: Events that describe a decision about whether to trade.
DECISION_EVENTS = frozenset(
    {
        "orchestrator.selected",
        "orchestrator.all_gated",
        "orchestrator.all_deselected",
        "orchestrator.no_trade",
        "risk.order_denied",
    }
)

#: Events describing what the engine is configured to do, rather than what it decided.
FACT_EVENTS = frozenset({"demo_bot.starting", "meme.universe_selected", "asset.class_selected"})

#: The venue's refusals when a product category needs an agreement nobody has signed.
#:
#: Three distinct agreements, verified live on 2026-08-14: metals answer 110123, crude oil
#: 110125, and equities and index products 110126. Signing one leaves the other two
#: refusing, so a single code would report a class as tradable that is not.
#:
#: Still matched textually as a fallback. The engine now emits a dedicated
#: ``paper.class_quarantined`` event carrying the asset class directly, which is preferred
#: because it names the class rather than requiring it to be inferred from a symbol.
PRODUCT_AGREEMENT_CODES: frozenset[str] = frozenset({"110123", "110125", "110126"})

#: The engine's own event for "this class is set aside until an agreement is signed".
QUARANTINE_EVENT = "paper.class_quarantined"

#: Where a venue code may legitimately appear: the engine's own ``venue_code=`` field, or
#: the ``retCode`` of a raw venue body echoed into the message.
_CODE_FIELD = re.compile(r"(?:venue_code=|\"retCode\"\s*:\s*)(\d+)")


def _agreement_codes_in(line: str) -> set[str]:
    """Agreement codes this line actually reports.

    Read from the two fields that carry a venue code, never from bare digits anywhere in
    the line. The codes are six-digit numbers and a trading log is full of six-digit
    numbers — prices, quantities, order ids, durations. Substring matching them marked the
    crypto book, which trades perfectly well, as blocked by an agreement.
    """
    return {code for code in _CODE_FIELD.findall(line) if code in PRODUCT_AGREEMENT_CODES}


def _is_agreement_block(line: str) -> bool:
    """Whether this line reports an unsigned-agreement refusal."""
    return QUARANTINE_EVENT in line or bool(_agreement_codes_in(line))


#: Cheap substring pre-filter. Applied before the regex so the overwhelming majority of
#: lines — tick and ticker noise — are discarded without any parsing at all.
_PREFILTER = (
    "orchestrator.",
    "risk.order_denied",
    "demo_bot.starting",
    "meme.universe_selected",
    "asset.class_selected",
    QUARANTINE_EVENT,
)

#: How far back to look on the very first read, in bytes. Bounded because the file has no
#: upper size: an unbounded first read would stall the API for as long as the disk took.
DEFAULT_LOOKBACK_BYTES = 32 * 1024 * 1024

#: How many parsed decisions to retain in memory.
DEFAULT_HISTORY = 500

#: Shortest string that can be a quoted value: the two quote characters themselves.
_QUOTED_MINIMUM = 2

_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+"
    r"\[(?P<level>\w+)\s*\]\s+"
    r"(?P<event>[\w.]+)\s+"
    r"\[(?P<logger>[\w.]+)\]\s*"
    r"(?P<rest>.*)$"
)

# A key, then either a quoted string, a bracketed group, or a bare token. Quoted values may
# contain spaces, which is why the pairs cannot simply be split on whitespace.
_PAIR = re.compile(
    r"(\w+)=("
    r"'(?:[^'\\]|\\.)*'"  # single-quoted
    r'|"(?:[^"\\]|\\.)*"'  # double-quoted
    r"|\{[^}]*\}"  # dict literal, e.g. component scores
    r"|\[[^\]]*\]"  # list literal, e.g. rule names
    r"|\S+"  # bare token
    r")"
)

#: Reason text to rejection category. Ordered most specific first: several reasons mention
#: more than one concept, and the first match should be the one the operator would name.
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("correlation", ("correlation",)),
    ("risk_reward", ("risk/reward", "risk-reward", "reward-to-risk", "r:r", "rr ")),
    ("cost", ("cost", "spread", "fee", "edge after", "uneconomic", "economic gate")),
    ("confluence", ("confluence", "corroborat", "agreement", "unconfirmed")),
    ("liquidity", ("liquidity", "illiquid", "turnover", "volume", "book depth")),
    ("sizing", ("size", "sizing", "notional", "quantity", "min_qty", "step")),
    (
        "risk",
        (
            "consecutive loss",
            "drawdown",
            "exposure",
            "kill switch",
            "daily loss",
            "cooldown",
            "halted",
            "limit of",
            "max position",
        ),
    ),
    ("regime", ("regime", "expectancy")),
    ("score_floor", ("below floor", "below the floor", "score")),
)


def since(decisions: list[Decision], started_at: datetime | None) -> list[Decision]:
    """Only the decisions this session could have made.

    The engine's log is one append-only file shared by every run — 576 MB across 57 engine
    starts on this deployment — and its lines carry ``session_id=***redacted***``, so the
    session cannot be recovered from the log itself. Without a boundary the Overview page
    simply reports the tail of the file, which is a previous run's work.

    Two minutes into a fresh session the panel showed *500 evaluated · 149 selected · 351
    declined*, on a session that had decided nothing at all.

    The session's own start time is the discriminator, and it is exact: an engine cannot
    have decided anything before it existed. ``None`` keeps everything, because a session
    with no recorded start cannot be filtered and showing nothing would be its own lie.

    Args:
        decisions: Parsed decisions, oldest first.
        started_at: When the current session started.

    Returns:
        The decisions at or after ``started_at``.

    """
    if started_at is None:
        return decisions
    return [item for item in decisions if item.timestamp >= started_at]


def categorise(reason: str) -> str:
    """Bucket a free-text rejection reason.

    The raw reason is always carried alongside the category. The category exists to make
    an aggregate possible, not to replace the engine's own words — a bucket that hides the
    reason would make a novel failure look like a familiar one.

    Args:
        reason: The engine's reason text.

    Returns:
        A category key, or ``"other"`` when nothing matches.

    """
    lowered = reason.lower()
    for category, needles in _CATEGORY_RULES:
        if any(needle in lowered for needle in needles):
            return category
    return "other"


@dataclass(frozen=True, slots=True)
class Decision:
    """One evaluated bar, and what the engine decided to do about it."""

    timestamp: datetime
    event: str
    symbol: str | None
    outcome: str
    strategy: str | None
    direction: str | None
    score: str | None
    confidence: str | None
    candidates: int | None
    regime: str | None
    reason: str | None
    category: str | None
    components: dict[str, str]
    runner_up: str | None

    def to_dict(self) -> dict[str, Any]:
        """Wire form.

        Fields the engine does not record are emitted as ``None`` and rendered by the
        client as ``NOT RECORDED`` — never as a zero, which would read as a measurement.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "event": self.event,
            "symbol": self.symbol,
            "outcome": self.outcome,
            "strategy": self.strategy,
            "direction": self.direction,
            "score": self.score,
            "confidence": self.confidence,
            "candidates": self.candidates,
            "regime": self.regime,
            "reason": self.reason,
            "rejection_category": self.category,
            # Component scores are unit-free 0-1 weights, not money. `cost` here is the
            # cost *score*, not an estimated cost in USDT: the engine does not log an
            # absolute cost, and presenting this as one would invent a number.
            "component_scores": self.components,
            "runner_up": self.runner_up,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Decision | None:
        """Rebuild a decision published by the engine, or ``None`` if it cannot be read.

        Never raises. One malformed entry in the feed must not take out the decision panel,
        and it is dropped rather than rendered half-parsed.
        """
        try:
            moment = payload["timestamp"]
            if not isinstance(moment, str):
                return None
            components = payload.get("component_scores") or {}
            return cls(
                timestamp=datetime.fromisoformat(moment),
                event=str(payload.get("event") or ""),
                symbol=_opt_str(payload.get("symbol")),
                outcome=str(payload.get("outcome") or ""),
                strategy=_opt_str(payload.get("strategy")),
                direction=_opt_str(payload.get("direction")),
                score=_opt_str(payload.get("score")),
                confidence=_opt_str(payload.get("confidence")),
                candidates=(
                    int(payload["candidates"]) if payload.get("candidates") is not None else None
                ),
                regime=_opt_str(payload.get("regime")),
                reason=_opt_str(payload.get("reason")),
                category=_opt_str(payload.get("rejection_category")),
                components=(
                    {str(k): str(v) for k, v in components.items()}
                    if isinstance(components, Mapping)
                    else {}
                ),
                runner_up=_opt_str(payload.get("runner_up")),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _opt_str(value: Any) -> str | None:
    """Coerce to ``str``, preserving ``None`` so "absent" stays distinct from empty."""
    return None if value is None else str(value)


#: Redis key prefix for decisions the engine publishes about itself.
#:
#: The parser below is correct; where it *runs* is what was wrong. The API reads the engine
#: log through a macOS bind mount, and on 2026-08-16 the container's view of a 576 MB
#: continuously-appended file sat 145 lines and 15 minutes behind the host's: the engine had
#: selected a candidate, sized it and submitted it to the venue at 12:45, while the dashboard
#: reported "no decisions found in the log tail" and showed STARTING.
#:
#: So the engine parses its own log — on the host, where its view is by definition current —
#: and publishes the result to Redis, which is a TCP service that either answers with the
#: current value or fails. It is the same reasoning that moved liveness to a heartbeat, and
#: the file read remains as a fallback for an engine too old to publish.
DECISION_FEED_PREFIX = "decisions:engine"

#: How long a published feed survives. Long enough to cover a multi-hour session, so the
#: panel still has history after a quiet spell rather than emptying between bars.
DECISION_FEED_TTL_SECONDS = 21_600.0


def decision_feed_key(session_id: str) -> str:
    """Redis key holding one session's published decisions."""
    return f"{DECISION_FEED_PREFIX}:{session_id}"


def build_feed(decisions: list[Decision], facts: EngineFacts) -> dict[str, Any]:
    """The payload the engine publishes: what it decided, and what it is configured to do."""
    return {"decisions": [item.to_dict() for item in decisions], "facts": facts.to_dict()}


def parse_feed(payload: Any) -> tuple[list[Decision], EngineFacts | None] | None:
    """Decode a published feed, or ``None`` when nothing usable was stored.

    ``None`` means "no feed" and is distinct from an empty decision list, which means "the
    engine published, and has decided nothing yet". The caller falls back to the file only
    for the former; treating an empty feed as absent would resurrect the stale reader.

    A bare list is accepted as decisions with no facts, so a feed from an engine that
    published before facts were carried still reads rather than being discarded whole.
    """
    if isinstance(payload, list):
        entries: Any = payload
        raw_facts: Any = None
    elif isinstance(payload, Mapping):
        entries = payload.get("decisions") or []
        raw_facts = payload.get("facts")
        if not isinstance(entries, list):
            return None
    else:
        return None

    parsed = (Decision.from_dict(e) for e in entries if isinstance(e, Mapping))
    decisions = [item for item in parsed if item is not None]
    facts = EngineFacts.from_dict(raw_facts) if isinstance(raw_facts, Mapping) else None
    return decisions, facts


_OUTCOMES = {
    "orchestrator.selected": "SELECTED",
    "orchestrator.all_gated": "GATED",
    "orchestrator.all_deselected": "DESELECTED",
    "orchestrator.no_trade": "BELOW_SCORE_FLOOR",
    "risk.order_denied": "RISK_BLOCKED",
}


def parse_pairs(rest: str) -> dict[str, str]:
    """Extract ``key=value`` pairs from a rendered log line's tail.

    Args:
        rest: Everything after the logger name.

    Returns:
        Mapping of key to value, with surrounding quotes removed.

    """
    pairs: dict[str, str] = {}
    for key, raw in _PAIR.findall(rest):
        value = raw
        if len(value) >= _QUOTED_MINIMUM and value[0] in "'\"" and value[-1] == value[0]:
            value = value[1:-1].replace("\\'", "'").replace('\\"', '"')
        pairs[key] = value
    return pairs


def _components(raw: str | None) -> dict[str, str]:
    """Parse the ``{'confidence': '1.00', ...}`` component-score literal."""
    if not raw:
        return {}
    return dict(re.findall(r"'(\w+)':\s*'([^']*)'", raw))


def _timestamp(line: str) -> datetime | None:
    """The timestamp at the head of a rendered log line, if it parses."""
    match = _LINE.match(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group("ts").replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_line(line: str) -> Decision | None:
    """Parse one rendered log line into a decision, or ``None`` if it is not one.

    Args:
        line: A single line from the engine log.

    Returns:
        The decision, or ``None`` for any line that is not a decision event.

    """
    match = _LINE.match(line)
    if match is None:
        return None
    event = match.group("event")
    if event not in DECISION_EVENTS:
        return None

    try:
        timestamp = datetime.fromisoformat(match.group("ts").replace("Z", "+00:00"))
    except ValueError:
        return None

    fields = parse_pairs(match.group("rest"))
    reason = fields.get("first_reason") or fields.get("reason") or None
    if event == "orchestrator.no_trade":
        best = fields.get("best")
        best_score = fields.get("best_score")
        floor = fields.get("floor")
        reason = f"best candidate {best} scored {best_score}, below the floor of {floor}"

    candidates: int | None
    try:
        candidates = int(fields["candidates"]) if "candidates" in fields else None
    except ValueError:
        candidates = None

    return Decision(
        timestamp=timestamp,
        event=event,
        symbol=fields.get("symbol"),
        outcome=_OUTCOMES.get(event, "UNKNOWN"),
        strategy=fields.get("strategy") or fields.get("best"),
        direction=fields.get("direction") or fields.get("side"),
        score=fields.get("score") or fields.get("best_score"),
        confidence=fields.get("confidence"),
        candidates=candidates,
        regime=fields.get("regime"),
        reason=reason,
        category=categorise(reason) if reason else None,
        components=_components(fields.get("components")),
        runner_up=fields.get("runner_up") or None,
    )


def _string_list(raw: str | None) -> list[str]:
    """Parse a rendered Python list literal such as ``['BTC/USDT', 'ETH/USDT']``."""
    if not raw:
        return []
    return re.findall(r"'([^']+)'", raw)


@dataclass(frozen=True, slots=True)
class EngineFacts:
    """What the engine said about itself when it last started.

    Read from the engine's own startup log line rather than from configuration files,
    because the file on disk is what the engine will use *next* time — after a restart
    changed a flag, the two disagree, and only the log describes the process that is
    actually running.
    """

    started_at: datetime | None = None
    mode: str | None = None
    env: str | None = None
    timeframe: str | None = None
    symbols: tuple[str, ...] = ()
    strategy: str | None = None
    pool: str | None = None
    starting_equity: str | None = None
    equity_source: str | None = None
    max_concurrent: str | None = None
    meme_symbols: tuple[str, ...] = ()
    meme_discovered: str | None = None
    #: Symbols grouped by the asset class the engine itself assigned them at startup.
    #:
    #: The venue's ``symbolType`` is the only authority on whether ``SNDK`` is an equity
    #: or a token, and it is not recorded anywhere the dashboard can reach — not in the
    #: trades table, and not derivable from the ticker. The engine logs its own answer per
    #: class, so that answer is read back rather than guessed at a second time.
    class_symbols: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Symbols the venue refused for want of a signed product agreement.
    agreement_blocked: tuple[str, ...] = ()
    agreement_blocked_at: datetime | None = None
    #: Asset classes set aside for the same reason, named by the engine rather than
    #: inferred. A class can be blocked with no symbol recorded, and a symbol's class is
    #: not always derivable from its ticker, so both are kept.
    agreement_blocked_classes: tuple[str, ...] = ()
    #: The venue codes actually seen, so the operator is told which agreements to sign.
    agreement_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Wire form."""
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "mode": self.mode,
            "env": self.env,
            "timeframe": self.timeframe,
            "symbols": list(self.symbols),
            "strategy": self.strategy,
            "strategy_pool": self.pool,
            "starting_equity": self.starting_equity,
            "equity_source": self.equity_source,
            "max_concurrent": self.max_concurrent,
            "meme_symbols": list(self.meme_symbols),
            "meme_discovered": self.meme_discovered,
            "class_symbols": {name: list(items) for name, items in self.class_symbols.items()},
            "agreement_blocked_symbols": list(self.agreement_blocked),
            "agreement_blocked_classes": list(self.agreement_blocked_classes),
            "agreement_codes": list(self.agreement_codes),
            "agreement_blocked_at": (
                self.agreement_blocked_at.isoformat() if self.agreement_blocked_at else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EngineFacts | None:
        """Rebuild facts published by the engine, or ``None`` if they cannot be read.

        Never raises, for the same reason :meth:`Decision.from_dict` does not: one bad field
        must degrade a panel, not remove it.
        """
        try:
            classes = payload.get("class_symbols") or {}
            return cls(
                started_at=_parse_moment(payload.get("started_at")),
                mode=_opt_str(payload.get("mode")),
                env=_opt_str(payload.get("env")),
                timeframe=_opt_str(payload.get("timeframe")),
                symbols=_str_tuple(payload.get("symbols")),
                strategy=_opt_str(payload.get("strategy")),
                pool=_opt_str(payload.get("strategy_pool")),
                starting_equity=_opt_str(payload.get("starting_equity")),
                equity_source=_opt_str(payload.get("equity_source")),
                max_concurrent=_opt_str(payload.get("max_concurrent")),
                meme_symbols=_str_tuple(payload.get("meme_symbols")),
                meme_discovered=_opt_str(payload.get("meme_discovered")),
                class_symbols=(
                    {str(k): _str_tuple(v) for k, v in classes.items()}
                    if isinstance(classes, Mapping)
                    else {}
                ),
                agreement_blocked=_str_tuple(payload.get("agreement_blocked_symbols")),
                agreement_blocked_at=_parse_moment(payload.get("agreement_blocked_at")),
                agreement_blocked_classes=_str_tuple(payload.get("agreement_blocked_classes")),
                agreement_codes=_str_tuple(payload.get("agreement_codes")),
            )
        except (TypeError, ValueError):
            return None


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a published sequence to a tuple of strings."""
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _parse_moment(value: Any) -> datetime | None:
    """Parse a published ISO timestamp, or ``None`` when absent or malformed."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class DecisionLog:
    """Incremental reader over the engine's log file.

    Holds the byte offset already consumed so each refresh reads only what the engine has
    appended since. Detects truncation and rotation by watching the file size, and starts
    over from a bounded lookback when it sees either.
    """

    __slots__ = ("_decisions", "_facts", "_lookback", "_offset", "_path")

    def __init__(
        self,
        path: Path,
        *,
        lookback_bytes: int = DEFAULT_LOOKBACK_BYTES,
        history: int = DEFAULT_HISTORY,
    ) -> None:
        """Create a reader for ``path``.

        Args:
            path: The engine log file.
            lookback_bytes: How far back to read on the first refresh.
            history: How many decisions to retain.

        """
        self._path = path
        self._lookback = lookback_bytes
        self._offset: int | None = None
        self._decisions: deque[Decision] = deque(maxlen=history)
        self._facts = EngineFacts()

    @property
    def path(self) -> Path:
        """The file being read."""
        return self._path

    def facts(self) -> EngineFacts:
        """What the engine reported about itself at its last start."""
        return self._facts

    def _absorb_fact(self, line: str, event: str, fields: dict[str, str]) -> None:
        """Fold one configuration line into the accumulated engine facts."""
        moment = _timestamp(line)
        if event == "demo_bot.starting":
            self._facts = replace(
                self._facts,
                started_at=moment,
                mode=fields.get("mode"),
                env=fields.get("env"),
                timeframe=fields.get("timeframe"),
                symbols=tuple(_string_list(fields.get("symbols"))),
                strategy=fields.get("strategy"),
                pool=fields.get("pool"),
                starting_equity=fields.get("starting_equity"),
                equity_source=fields.get("equity_source"),
                max_concurrent=fields.get("max_concurrent"),
                # A restart supersedes any earlier block: the refused symbols are not part
                # of the run that has just begun unless it refuses them again.
                agreement_blocked=(),
                agreement_blocked_at=None,
                agreement_blocked_classes=(),
                agreement_codes=(),
                # A restart rediscovers its universe; carrying the previous run's classes
                # forward would show markets this process never subscribed to.
                class_symbols={},
            )
        elif event == "meme.universe_selected":
            self._facts = replace(
                self._facts,
                meme_symbols=tuple(_string_list(fields.get("enabled"))),
                meme_discovered=fields.get("discovered"),
            )
        elif event == "asset.class_selected":
            name = fields.get("asset_class")
            if name:
                merged = dict(self._facts.class_symbols)
                merged[name] = tuple(_string_list(fields.get("enabled")))
                self._facts = replace(self._facts, class_symbols=merged)

    def refresh(self) -> list[Decision]:
        """Consume any new bytes and return the retained decisions, oldest first.

        Blocking: intended to be called through :func:`asyncio.to_thread`.

        Returns:
            The retained decisions in chronological order.

        """
        try:
            size = self._path.stat().st_size
        except OSError as exc:
            logger.info("dashboard.decision_log_unavailable", path=str(self._path), error=str(exc))
            return list(self._decisions)

        start = self._offset
        if start is None or start > size:
            # First read, or the file was rotated or truncated beneath us.
            start = max(0, size - self._lookback)
            self._decisions.clear()
        if start == size:
            return list(self._decisions)

        try:
            with self._path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(size - start)
        except OSError as exc:
            logger.info("dashboard.decision_log_read_failed", error=str(exc))
            return list(self._decisions)

        text = chunk.decode("utf-8", errors="replace")
        consumed = size
        if not text.endswith("\n"):
            # A partial trailing line is left for the next refresh rather than parsed
            # half-formed, which would drop or corrupt the newest decision.
            cut = text.rfind("\n")
            if cut == -1:
                return list(self._decisions)
            consumed = start + len(text[: cut + 1].encode("utf-8"))
            text = text[: cut + 1]

        lines = text.split("\n")
        if self._offset is None and start > 0 and lines:
            # The first line of a *mid-file* seek is almost certainly a fragment. When the
            # read began at byte zero — a small or freshly rotated log — that first line is
            # complete, and discarding it would silently lose the oldest decision.
            lines = lines[1:]

        for line in lines:
            if not any(needle in line for needle in _PREFILTER):
                continue

            if _is_agreement_block(line):
                self._note_agreement_block(line)
                continue

            match = _LINE.match(line)
            if match is not None and match.group("event") in FACT_EVENTS:
                self._absorb_fact(line, match.group("event"), parse_pairs(match.group("rest")))
                continue

            decision = parse_line(line)
            if decision is not None:
                self._decisions.append(decision)

        self._offset = consumed
        return list(self._decisions)

    def _note_agreement_block(self, line: str) -> None:
        """Record a venue refusal caused by an unsigned product agreement.

        The engine surfaces this as an ordinary exchange error, so the symbol has to be
        recovered from the surrounding context rather than a dedicated field. Only the
        symbols the venue actually refused are recorded — an asset class the operator
        merely disabled is a different state and must not be conflated with a block.
        """
        symbols = {
            symbol for symbol in re.findall(r"\b([A-Z0-9]{2,10}/[A-Z]{3,5})\b", line) if symbol
        }
        classes = set(re.findall(r"asset_class=(\w+)", line))
        codes = _agreement_codes_in(line)
        if not symbols and not classes:
            return
        self._facts = replace(
            self._facts,
            agreement_blocked=tuple(sorted(set(self._facts.agreement_blocked) | symbols)),
            agreement_blocked_at=_timestamp(line) or self._facts.agreement_blocked_at,
            agreement_blocked_classes=tuple(
                sorted(set(self._facts.agreement_blocked_classes) | classes)
            ),
            agreement_codes=tuple(sorted(set(self._facts.agreement_codes) | codes)),
        )


def summarise(decisions: list[Decision]) -> dict[str, Any]:
    """Aggregate decisions into the counts the dashboard shows above the detail table.

    Args:
        decisions: Parsed decisions, oldest first.

    Returns:
        A JSON-safe summary: totals by outcome, by rejection category, and by symbol.

    """
    outcomes = Counter(item.outcome for item in decisions)
    categories = Counter(item.category for item in decisions if item.category is not None)
    symbols = Counter(item.symbol for item in decisions if item.symbol is not None)

    total = len(decisions)
    selected = outcomes.get("SELECTED", 0)
    return {
        "evaluated": total,
        "selected": selected,
        "declined": total - selected,
        "by_outcome": dict(outcomes),
        "by_rejection_category": dict(categories.most_common()),
        "by_symbol": dict(symbols.most_common(20)),
        "first_at": decisions[0].timestamp.isoformat() if decisions else None,
        "last_at": decisions[-1].timestamp.isoformat() if decisions else None,
        # Named so the client can state the window honestly rather than implying the
        # figures cover the whole session.
        "window": (
            "decisions retained from the tail of the engine log, not the whole session"
            if decisions
            else "no decisions found in the log tail"
        ),
    }


__all__ = [
    "DECISION_EVENTS",
    "DEFAULT_HISTORY",
    "DEFAULT_LOOKBACK_BYTES",
    "FACT_EVENTS",
    "PRODUCT_AGREEMENT_CODES",
    "QUARANTINE_EVENT",
    "Decision",
    "DecisionLog",
    "EngineFacts",
    "categorise",
    "parse_line",
    "parse_pairs",
    "since",
    "summarise",
]
