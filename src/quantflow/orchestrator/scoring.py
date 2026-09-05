"""Candidate construction and the scoring model used to rank them.

Kept apart from the orchestrator itself so the ranking can be tested as a pure function:
given a set of candidates and a context, the winner must be reproducible without an engine,
a database or a market.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import MarketRegime, SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.signals import Signal

#: Closed trades a strategy needs before its live record is allowed to move its score.
#: Below this the sample is noise, and letting three lucky trades promote a strategy is
#: exactly how a decision engine ends up chasing whatever ran hottest last hour.
MIN_TRADES_FOR_EVIDENCE = 20

#: Weight of each component in the final score. They sum to 1.0 so a score stays in [0,1]
#: and stays readable: 0.6 means "six tenths of everything this model knows how to want".
WEIGHTS = {
    "confidence": Decimal("0.20"),
    "risk_reward": Decimal("0.25"),
    "regime": Decimal("0.20"),
    "evidence": Decimal("0.20"),
    "cost": Decimal("0.10"),
    "correlation": Decimal("0.05"),
}

#: Score below which a candidate is not worth trading at all.
#:
#: A candidate with every component at neutral scores 0.5, so a floor below that would
#: admit setups the model has nothing positive to say about. 0.60 requires a candidate to
#: be clearly better than neutral on the weighted blend rather than merely the best of a
#: weak field — chosen from that reasoning, not fitted to observed live scores.
MIN_SCORE_TO_TRADE = Decimal("0.60")

#: Risk/reward at or above which the R:R component saturates.
RR_SATURATION = Decimal("3.0")

# --------------------------------------------------------------------------- #
# Hard gates
#
# These are not weighted terms. A candidate failing any of them is rejected outright,
# whatever the rest of its score looks like. The diagnosis that motivated them: a median
# reward:risk of 0.959 on a 25% win rate cannot be profitable at any selection quality —
# no amount of good ranking rescues a payoff structure that needs a 70% hit rate.
# --------------------------------------------------------------------------- #

#: Minimum reward:risk a candidate must offer. Below this the arithmetic cannot work at
#: the win rates these strategies actually achieve.
MIN_RISK_REWARD = Decimal("1.5")

#: Minimum expected edge, as a fraction of notional, that must survive round-trip costs.
#: A target that clears the fee by a hair is not an opportunity.
#:
#: Lowered from 0.40% to 0.35% on 2026-08-17, by explicit decision, and the reason it is
#: defensible rather than arbitrary is that it still clears this module's own stated basis:
#: :data:`MIN_REWARD_TO_COST` requires 3x the round-trip cost, and at the 0.11% measured on
#: this account that is 0.33%. The old 0.40% was stricter than the rule it came from.
#:
#: Measured over 18 hours on a BTC/ETH-only universe: 16 of 58 blocked bars were candidates
#: at 0.3512-0.3639% — clearing 3x cost, refused by the floor alone. This admits those and
#: nothing weaker. The confluence requirement of two independent families is deliberately
#: untouched, so corroboration is still required before capital is committed.
#:
#: Override with QF_MIN_NET_EDGE_PCT. It is read here rather than hard-coded so the floor
#: can be put back without a code change if the admitted trades underperform.
MIN_NET_EDGE_PCT = Decimal(os.environ.get("QF_MIN_NET_EDGE_PCT", "0.0035"))

#: Multiple of round-trip cost the expected reward must exceed.
MIN_REWARD_TO_COST = Decimal("3")

#: Net edge a candidate must offer to be taken on ONE strategy family's opinion alone.
#:
#: Confluence normally requires two independent families, and the reason is sound: one
#: strategy agreeing with itself is not corroboration. But by the time a candidate reaches
#: the confluence check it has already cleared every economic gate above — reward:risk of at
#: least 1.5, a valid stop and target, liquidity, and a net edge past the floor — so
#: refusing it purely for want of a second opinion discards setups that are individually
#: sound. Measured 2026-08-17 on a BTC/ETH universe: 14 of 15 blocked bars were confluence
#: alone, with nothing else against them.
#:
#: So a lone family may carry a trade, but it has to pay for the missing corroboration:
#: twice the ordinary edge floor. That keeps this a filter on *quality* rather than a
#: relaxation into activity — the two live winners this session came in at 0.76% and 1.50%
#: net edge, so the bar is demonstrably reachable by the setups worth having.
#:
#: Override with QF_SOLO_FAMILY_MIN_NET_EDGE. Set it absurdly high to restore strict
#: two-family confluence without a code change.
SOLO_FAMILY_MIN_NET_EDGE = Decimal(
    os.environ.get("QF_SOLO_FAMILY_MIN_NET_EDGE", str(MIN_NET_EDGE_PCT * 2))
)

#: Concurrent entry legs allowed per symbol. 1 disables pyramiding entirely.
#:
#: A second leg is NOT a second position at the venue — Bybit nets in one-way mode, so it
#: enlarges the existing position and moves its average entry. That is why adding is
#: permitted only into a position that is not underwater: on a long, a leg added at a
#: higher price raises the average, which is pyramiding into strength; a leg added at a
#: lower price would be averaging down wearing a different name.
MAX_LEGS_PER_SYMBOL = int(os.environ.get("QF_BOT_MAX_LEGS_PER_SYMBOL", "1"))

#: How much better a second leg must score than the thesis already open, when it comes from
#: the same strategy family and the same regime.
#:
#: Same strategy + same direction + same regime + near-identical score is the same opinion
#: twice, and the whole session's evidence says that is what a naive pyramid would buy:
#: 132 selections, 100% long, four correlated trend families, scores inside a 0.008 band.
#: A different family or a changed regime counts as new on its own; an identical setup has
#: to prove itself with a materially better score.
PYRAMID_MIN_SCORE_IMPROVEMENT = Decimal(os.environ.get("QF_PYRAMID_MIN_SCORE_IMPROVEMENT", "0.02"))

#: Positions in the same strategy allowed before further candidates from it are refused,
#: so the book cannot fill with one idea expressed ten times.
MAX_POSITIONS_PER_STRATEGY = 3


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    """A strategy's realised record within the current session."""

    strategy_id: str
    trades: int = 0
    wins: int = 0
    net_pnl: Decimal = ZERO

    @property
    def is_meaningful(self) -> bool:
        """Whether the sample is large enough to move a score."""
        return self.trades >= MIN_TRADES_FOR_EVIDENCE

    @property
    def win_rate(self) -> Decimal:
        """Wins over trades, or zero when there are none."""
        return Decimal(self.wins) / Decimal(self.trades) if self.trades else ZERO


@dataclass(frozen=True, slots=True)
class Candidate:
    """One strategy's actionable proposal for one symbol on one bar."""

    symbol: Symbol
    strategy_id: str
    direction: SignalDirection
    confidence: Decimal
    entry: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    timestamp: datetime
    signal: Signal
    regime: MarketRegime = MarketRegime.UNKNOWN
    score: Decimal = ZERO
    components: dict[str, Decimal] = field(default_factory=dict)
    rejection: str | None = None

    @property
    def risk_reward(self) -> Decimal | None:
        """Reward over risk, or ``None`` when either leg is missing.

        Absent rather than assumed: a candidate with no stop has *unknown* risk, and
        scoring it as though the risk were zero would rank it above every protected
        candidate on the board.
        """
        if self.stop_loss is None or self.take_profit is None:
            return None
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        if risk <= ZERO:
            return None
        return reward / risk

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"{self.strategy_id}/{self.symbol.slashed} {self.direction.value} "
            f"conf={self.confidence:.2f} score={self.score:.3f}"
        )


def _risk_reward_component(candidate: Candidate) -> Decimal:
    """R:R mapped into [0,1], saturating at :data:`RR_SATURATION`.

    A candidate with no measurable R:R scores zero here rather than being discarded: the
    risk engine will attach its own default stop, so the trade is still possible — it is
    simply less attractive than one whose own author specified where it is wrong.
    """
    ratio = candidate.risk_reward
    if ratio is None:
        return ZERO
    return min(ratio / RR_SATURATION, ONE)


def _regime_component(candidate: Candidate, regime: MarketRegime) -> Decimal:
    """How well the strategy family suits the observed regime.

    Neutral (0.5) when the regime is unknown — an unknown regime is not evidence for or
    against anything, and scoring it as a penalty would quietly suppress every candidate
    whenever regime detection is unavailable.
    """
    if regime is MarketRegime.UNKNOWN:
        return Decimal("0.5")

    trending = regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)
    family = _family_of(candidate.strategy_id)
    if family in {"trend", "breakout"}:
        return Decimal("0.9") if trending else Decimal("0.3")
    if family == "reversion":
        return Decimal("0.3") if trending else Decimal("0.9")
    if family == "volatility":
        return Decimal("0.7")
    return Decimal("0.5")


def _family_of(strategy_id: str) -> str:
    """Coarse family label, derived from the strategy id."""
    if "reversion" in strategy_id:
        return "reversion"
    if "breakout" in strategy_id or "thrust" in strategy_id or "squeeze" in strategy_id:
        return "breakout"
    if "expansion" in strategy_id:
        return "volatility"
    if strategy_id in {"ema_cross", "macd_trend", "triple_ma", "keltner_trend", "momentum_roc"}:
        return "trend"
    return "other"


def _evidence_component(record: StrategyRecord | None) -> Decimal:
    """The strategy's own realised record, once there is enough of it.

    Returns a flat neutral until :data:`MIN_TRADES_FOR_EVIDENCE` trades exist. This is the
    guard against a strategy with three good trades dominating the engine.
    """
    if record is None or not record.is_meaningful:
        return Decimal("0.5")
    # Win rate is a bounded, directly comparable summary; PnL magnitude is not, since one
    # outsized winner would swamp the ranking.
    return min(max(record.win_rate, ZERO), ONE)


def net_edge_of(candidate: Candidate, *, cost_rate: Decimal) -> Decimal | None:
    """Expected edge after round-trip costs, or ``None`` when it cannot be measured.

    Extracted so the selection layer can judge a candidate on the same arithmetic the
    economic gate used, rather than recomputing it and risking the two disagreeing.
    """
    if candidate.entry <= ZERO or candidate.take_profit is None:
        return None
    reward_pct = abs(candidate.take_profit - candidate.entry) / candidate.entry
    return reward_pct - cost_rate


def gate_candidate(  # noqa: PLR0911 - one return per gate reads better than nesting
    candidate: Candidate,
    *,
    cost_rate: Decimal,
    strategy_position_counts: dict[str, int] | None = None,
) -> str | None:
    """Return a rejection reason, or ``None`` if the candidate may be scored.

    Applied before ranking, because these are questions of whether a trade is worth making
    at all rather than of how it compares with its rivals. Ranking a set of candidates that
    are all uneconomic just picks the least bad one and trades it.
    """
    ratio = candidate.risk_reward
    if ratio is None:
        return "no measurable reward:risk (missing stop or target)"
    if ratio < MIN_RISK_REWARD:
        return f"reward:risk {ratio:.2f} below the {MIN_RISK_REWARD} floor"

    if candidate.entry <= ZERO or candidate.take_profit is None:
        return "no usable entry or target"
    reward_pct = abs(candidate.take_profit - candidate.entry) / candidate.entry
    net_edge = reward_pct - cost_rate
    if net_edge < MIN_NET_EDGE_PCT:
        return (
            f"expected edge {net_edge:.4%} after {cost_rate:.4%} costs is below "
            f"the {MIN_NET_EDGE_PCT:.4%} floor"
        )
    if cost_rate > ZERO and reward_pct < cost_rate * MIN_REWARD_TO_COST:
        return f"reward {reward_pct:.4%} is under {MIN_REWARD_TO_COST}x round-trip cost"

    counts = strategy_position_counts or {}
    held = counts.get(candidate.strategy_id, 0)
    if held >= MAX_POSITIONS_PER_STRATEGY:
        return f"{candidate.strategy_id} already holds {held} positions"
    return None


def _cost_component(candidate: Candidate, cost_rate: Decimal) -> Decimal:
    """How much of the expected reward the round-trip fee consumes.

    A candidate whose target is barely beyond its costs is not an opportunity, however
    confident its author is.
    """
    if candidate.take_profit is None or candidate.entry <= ZERO:
        return Decimal("0.5")
    reward_pct = abs(candidate.take_profit - candidate.entry) / candidate.entry
    if reward_pct <= ZERO:
        return ZERO
    drag = cost_rate / reward_pct
    return max(ONE - min(drag, ONE), ZERO)


def _correlation_component(candidate: Candidate, open_symbols: frozenset[Symbol]) -> Decimal:
    """Penalise adding another position in a symbol already held.

    Cross-asset correlation is enforced by the risk engine's correlation rule; this is the
    narrower, certain case — a second position in the same symbol is the same bet twice.
    """
    return ZERO if candidate.symbol in open_symbols else ONE


def score_candidate(
    candidate: Candidate,
    *,
    regime: MarketRegime,
    records: dict[str, StrategyRecord],
    open_symbols: frozenset[Symbol],
    cost_rate: Decimal,
) -> Candidate:
    """Return ``candidate`` with its score and per-component breakdown attached."""
    components = {
        "confidence": min(max(candidate.confidence, ZERO), ONE),
        "risk_reward": _risk_reward_component(candidate),
        "regime": _regime_component(candidate, regime),
        "evidence": _evidence_component(records.get(candidate.strategy_id)),
        "cost": _cost_component(candidate, cost_rate),
        "correlation": _correlation_component(candidate, open_symbols),
    }
    total = sum((WEIGHTS[name] * value for name, value in components.items()), ZERO)
    from dataclasses import replace

    return replace(candidate, score=total, components=components, regime=regime)


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Best first. Ties break on strategy id so the order is deterministic."""
    return sorted(candidates, key=lambda item: (-item.score, item.strategy_id))


__all__ = [
    "MIN_SCORE_TO_TRADE",
    "MIN_TRADES_FOR_EVIDENCE",
    "WEIGHTS",
    "Candidate",
    "StrategyRecord",
    "rank",
    "score_candidate",
]
