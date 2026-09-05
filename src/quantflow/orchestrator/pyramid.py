"""Whether a second entry on a symbol is a new opportunity or the same one twice.

Pyramiding is authorised, stacking is not, and the difference has to be decided by
something stricter than "another candidate appeared". The evidence from this account is
unambiguous about what a naive pyramid would buy: 132 selections in one session, every one
long, from four correlated trend-following families, with scores inside a 0.008 band. Left
ungated, a second leg would almost always be the first leg wearing a different indicator.

So a leg is admitted only when the case for it has actually changed — a different strategy
family, a different market regime, or a materially better score than the thesis already
running. Everything else the engine already demands (edge floor, reward:risk, liquidity,
stop, cooldown, exposure, correlation) still applies on top; this adds a requirement, it
never removes one.

One further guard, and it is the one that keeps pyramiding honest: a leg may only be added
to a position that is not underwater. Bybit nets in one-way mode, so a second leg does not
open a second position — it enlarges the first and moves its average entry. Adding to a
loser would therefore be averaging down under another name, whatever the intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantflow.core.precision import ZERO
from quantflow.domain.enums import MarketRegime, SignalDirection
from quantflow.orchestrator.scoring import PYRAMID_MIN_SCORE_IMPROVEMENT
from quantflow.orchestrator.selection import strategy_family


@dataclass(frozen=True, slots=True)
class EntryThesis:
    """The case that opened a position, kept so a later candidate can be compared to it."""

    strategy_id: str
    direction: SignalDirection
    regime: MarketRegime
    score: Decimal


def pyramid_verdict(
    existing: EntryThesis,
    *,
    strategy_id: str,
    direction: SignalDirection,
    regime: MarketRegime,
    score: Decimal,
    unrealized_pnl: Decimal,
    min_improvement: Decimal = PYRAMID_MIN_SCORE_IMPROVEMENT,
) -> tuple[bool, str]:
    """Whether a second leg is a genuinely new opportunity.

    Args:
        existing: The thesis already running on this symbol.
        strategy_id: Candidate's strategy.
        direction: Candidate's direction.
        regime: Regime the candidate was produced in.
        score: Candidate's orchestrator score.
        unrealized_pnl: Mark-to-market PnL of the open position.
        min_improvement: Score gap required when nothing else has changed.

    Returns:
        ``(allowed, reason)``. The reason is recorded either way, so a refusal is as
        legible in the decision log as an admission.

    """
    if unrealized_pnl < ZERO:
        return False, (
            f"existing position is underwater ({unrealized_pnl}); adding here would be "
            "averaging down, since the venue nets legs into one position"
        )

    if direction is not existing.direction:
        return False, (
            f"candidate is {direction.value} against an open {existing.direction.value}; "
            "that is a reversal for the exit logic to decide, not a second leg"
        )

    # The SAME taxonomy confluence uses. Inventing a second one here was a real defect,
    # caught in test: a substring guess put `triple_ma` in a family of its own, so it and
    # `keltner_trend` — both trend followers, the two strategies that actually compete on
    # this account — read as "different families" and a duplicate leg would have been
    # admitted. One definition of independence, shared by both gates.
    new_family = strategy_family(strategy_id)
    old_family = strategy_family(existing.strategy_id)
    if new_family != old_family:
        return True, f"different strategy family ({old_family} -> {new_family})"

    if regime is not existing.regime:
        return True, f"regime changed ({existing.regime.value} -> {regime.value})"

    improvement = score - existing.score
    if improvement >= min_improvement:
        return True, (
            f"score improved {improvement} over the open thesis "
            f"({existing.score} -> {score}), clearing the {min_improvement} bar"
        )

    return False, (
        f"same family ({new_family}), same regime ({regime.value}), same direction, and "
        f"score {score} is not {min_improvement} better than the open {existing.score}: "
        "this is the same thesis twice"
    )


__all__ = ["EntryThesis", "pyramid_verdict"]
