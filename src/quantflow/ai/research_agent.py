"""The AI Research Agent: analyse, explain, recommend — never trade.

The agent has no path to the exchange. It cannot construct an order, cannot emit a signal
and is not a `Strategy`. Everything it produces is a *finding*: a description of something
that happened, with evidence attached and a recommendation a person can accept or ignore.
That separation is structural rather than a matter of discipline — there is no method here
that returns anything an execution path would accept.

The design constraint that shapes the rest: **every recommendation must name the evidence
that produced it**. An agent that says "reduce position size" without saying which trades,
which regime and which numbers led there is asking to be trusted rather than checked, and
an unfalsifiable recommendation is worse than none — it cannot be argued with and it
cannot be wrong.

Findings are deterministic given the same inputs. No sampling, no temperature, no model
call in the default path: a recommendation that changes between runs on identical data
cannot be reasoned about, and the analysis here is arithmetic that does not need a
language model to perform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.positions import ClosedTrade
from quantflow.intelligence.regime import RegimeProfile
from quantflow.lab.attribution import RegimeBreakdown
from quantflow.lab.diagnosis import Diagnosis

#: Consecutive losses that constitute a streak worth reporting.
STREAK_THRESHOLD = 4

#: Multiple of the streak threshold at which a streak becomes urgent rather than notable.
URGENT_STREAK_MULTIPLE = 2

#: Symbols below which "concentrated in one symbol" is not a meaningful claim. With a
#: single symbol traded, 100% of losses come from it by definition.
MIN_SYMBOLS_FOR_CONCENTRATION = 2

#: Share of total loss concentrated in one symbol before it is called out.
CONCENTRATION_THRESHOLD = Decimal("0.50")

#: Trades below which the agent declines to draw conclusions.
MIN_TRADES_FOR_ANALYSIS = 20


class Severity(StrEnum):
    """How much attention a finding deserves."""

    INFO = "info"
    NOTABLE = "notable"
    URGENT = "urgent"


class FindingKind(StrEnum):
    """What sort of observation this is."""

    LOSS_ANALYSIS = "loss_analysis"
    REGIME_CHANGE = "regime_change"
    REGIME_MISMATCH = "regime_mismatch"
    COST_DRAG = "cost_drag"
    CONCENTRATION = "concentration"
    STREAK = "streak"
    IMPROVEMENT = "improvement"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class Finding:
    """One observation, its evidence and what to do about it."""

    kind: FindingKind
    severity: Severity
    headline: str
    #: The numbers this conclusion rests on. Never empty for a non-informational finding:
    #: a recommendation without evidence cannot be checked, and cannot be wrong.
    evidence: Mapping[str, str]
    recommendation: str
    #: Which strategy this concerns, when it concerns one in particular.
    strategy_id: str | None = None

    def explain(self) -> str:
        """Headline, evidence and recommendation as one readable block."""
        lines = [self.headline]
        for key, value in self.evidence.items():
            lines.append(f"  - {key}: {value}")
        lines.append(f"  -> {self.recommendation}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "kind": str(self.kind),
            "severity": str(self.severity),
            "headline": self.headline,
            "evidence": dict(self.evidence),
            "recommendation": self.recommendation,
            "strategy_id": self.strategy_id,
        }


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """Everything the agent concluded from one body of evidence."""

    findings: tuple[Finding, ...]
    generated_at: datetime
    trades_analysed: int
    #: Statements the agent explicitly declines to make, and why.
    withheld: tuple[str, ...] = field(default_factory=tuple)

    @property
    def urgent(self) -> tuple[Finding, ...]:
        """Findings that need attention now."""
        return tuple(item for item in self.findings if item.severity is Severity.URGENT)

    def by_kind(self, kind: FindingKind) -> tuple[Finding, ...]:
        """Findings of one sort."""
        return tuple(item for item in self.findings if item.kind is kind)

    def summary(self) -> str:
        """A short readable digest."""
        if not self.findings:
            return f"No findings from {self.trades_analysed} trades."
        counts: dict[str, int] = {}
        for item in self.findings:
            counts[str(item.severity)] = counts.get(str(item.severity), 0) + 1
        parts = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
        return f"{len(self.findings)} finding(s) from {self.trades_analysed} trades: {parts}"

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "trades_analysed": self.trades_analysed,
            "summary": self.summary(),
            "findings": [item.to_dict() for item in self.findings],
            "withheld": list(self.withheld),
        }


class ResearchAgent:
    """Analyses losses, detects regime change, and explains every recommendation.

    Deliberately not a `Strategy` and deliberately without an exchange gateway. The only
    thing it returns is a report.
    """

    def analyse_losses(
        self, trades: Sequence[ClosedTrade], *, strategy_id: str | None = None
    ) -> list[Finding]:
        """Examine losing trades for a pattern worth acting on."""
        losers = [trade for trade in trades if trade.net_pnl <= ZERO]
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            return [
                Finding(
                    kind=FindingKind.INSUFFICIENT_DATA,
                    severity=Severity.INFO,
                    headline=f"{len(trades)} trades is too few to analyse",
                    evidence={"trades": str(len(trades)), "minimum": str(MIN_TRADES_FOR_ANALYSIS)},
                    recommendation="Collect more trades before drawing conclusions.",
                    strategy_id=strategy_id,
                )
            ]
        if not losers:
            return []

        findings: list[Finding] = []
        findings.extend(self._loss_concentration(losers, strategy_id))
        findings.extend(self._loss_streak(trades, strategy_id))
        findings.extend(self._fee_burden(losers, strategy_id))
        return findings

    def detect_regime_change(
        self, previous: RegimeProfile | None, current: RegimeProfile | None
    ) -> list[Finding]:
        """Report a change in market conditions.

        Reports the *axes that changed* rather than "the regime changed". Volatility
        moving from normal to high while direction holds is a sizing problem; direction
        flipping while volatility holds is a signal problem. Collapsing them into one
        message would leave a reader unable to tell which they have.
        """
        if previous is None or current is None:
            return []
        if previous.matches(current):
            return []

        changed: dict[str, str] = {}
        if previous.direction is not current.direction:
            changed["direction"] = f"{previous.direction} -> {current.direction}"
        if previous.structure is not current.structure:
            changed["structure"] = f"{previous.structure} -> {current.structure}"
        if previous.volatility is not current.volatility:
            changed["volatility"] = f"{previous.volatility} -> {current.volatility}"

        severity = (
            Severity.URGENT
            if "volatility" in changed and current.volatility == "high"
            else Severity.NOTABLE
        )
        return [
            Finding(
                kind=FindingKind.REGIME_CHANGE,
                severity=severity,
                headline=f"Regime changed: {previous.label} -> {current.label}",
                evidence={
                    **changed,
                    "trend_strength": f"{current.trend.strength:.2f}",
                    "volatility_vs_baseline": (f"{current.volatility_measure.relative_level:.2f}x"),
                    "observed_at": current.timestamp.isoformat(),
                },
                recommendation=(
                    "Volatility has risen; reduce size or widen stops before the next entry."
                    if "volatility" in changed and current.volatility == "high"
                    else "Check that active strategies are validated for the new regime."
                ),
            )
        ]

    def assess_regime_fit(self, breakdown: RegimeBreakdown, *, strategy_id: str) -> list[Finding]:
        """Report a strategy that works in some conditions and not others."""
        if not breakdown.is_regime_dependent:
            return []
        best = breakdown.best()
        worst = breakdown.worst()
        if best is None or worst is None:
            return []

        return [
            Finding(
                kind=FindingKind.REGIME_MISMATCH,
                severity=Severity.NOTABLE,
                headline=(
                    f"{strategy_id} is regime dependent: profitable in "
                    f"{', '.join(breakdown.profitable_regimes)}, losing in "
                    f"{', '.join(breakdown.losing_regimes)}"
                ),
                evidence={
                    "best_regime": f"{best.regime} ({best.net_pnl:+.2f} over {best.trade_count})",
                    "worst_regime": (
                        f"{worst.regime} ({worst.net_pnl:+.2f} over {worst.trade_count})"
                    ),
                    "blended_would_hide": (f"{best.net_pnl + worst.net_pnl:+.2f} net across both"),
                },
                recommendation=(
                    f"Gate {strategy_id} to {', '.join(breakdown.profitable_regimes)} rather "
                    "than discarding it on a blended average that describes neither regime."
                ),
                strategy_id=strategy_id,
            )
        ]

    def interpret_diagnosis(self, diagnosis: Diagnosis, *, strategy_id: str) -> list[Finding]:
        """Turn a laboratory diagnosis into an actionable finding."""
        if not diagnosis.is_fixable_by_execution:
            return []
        evidence: dict[str, str] = {"cause": str(diagnosis.cause)}
        if diagnosis.frictionless_return is not None:
            evidence["return_without_costs"] = f"{diagnosis.frictionless_return:.2%}"
        if diagnosis.cost_share is not None:
            evidence["costs_as_share_of_gross"] = f"{diagnosis.cost_share:.1%}"
        if diagnosis.edge_per_trade is not None:
            evidence["gross_edge_per_trade"] = f"{diagnosis.edge_per_trade:.4%}"

        return [
            Finding(
                kind=FindingKind.COST_DRAG,
                severity=Severity.NOTABLE,
                headline=f"{strategy_id} has a signal that costs are consuming",
                evidence=evidence,
                recommendation=diagnosis.recommendation,
                strategy_id=strategy_id,
            )
        ]

    def report(
        self,
        trades: Sequence[ClosedTrade],
        *,
        now: datetime,
        strategy_id: str | None = None,
        breakdown: RegimeBreakdown | None = None,
        diagnosis: Diagnosis | None = None,
        previous_regime: RegimeProfile | None = None,
        current_regime: RegimeProfile | None = None,
    ) -> ResearchReport:
        """Assemble every finding the available evidence supports."""
        findings: list[Finding] = []
        withheld: list[str] = []

        findings.extend(self.analyse_losses(trades, strategy_id=strategy_id))
        findings.extend(self.detect_regime_change(previous_regime, current_regime))
        if breakdown is not None and strategy_id is not None:
            findings.extend(self.assess_regime_fit(breakdown, strategy_id=strategy_id))
        elif strategy_id is not None:
            withheld.append("regime fit: no per-regime breakdown supplied")
        if diagnosis is not None and strategy_id is not None:
            findings.extend(self.interpret_diagnosis(diagnosis, strategy_id=strategy_id))
        elif strategy_id is not None:
            withheld.append("cost diagnosis: no laboratory diagnosis supplied")
        if previous_regime is None or current_regime is None:
            withheld.append("regime change: needs two observations to compare")

        order = {Severity.URGENT: 0, Severity.NOTABLE: 1, Severity.INFO: 2}
        findings.sort(key=lambda item: order[item.severity])
        return ResearchReport(
            findings=tuple(findings),
            generated_at=now,
            trades_analysed=len(trades),
            withheld=tuple(withheld),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _loss_concentration(
        self, losers: Sequence[ClosedTrade], strategy_id: str | None
    ) -> list[Finding]:
        """Flag losses concentrated in one symbol."""
        total = sum((abs(trade.net_pnl) for trade in losers), ZERO)
        if total <= ZERO:
            return []

        by_symbol: dict[str, Decimal] = {}
        for trade in losers:
            key = str(trade.symbol)
            by_symbol[key] = by_symbol.get(key, ZERO) + abs(trade.net_pnl)

        symbol, worst = max(by_symbol.items(), key=lambda pair: pair[1])
        share = safe_divide(worst, total)
        if share < CONCENTRATION_THRESHOLD or len(by_symbol) < MIN_SYMBOLS_FOR_CONCENTRATION:
            return []

        return [
            Finding(
                kind=FindingKind.CONCENTRATION,
                severity=Severity.NOTABLE,
                headline=f"{share:.0%} of losses came from {symbol}",
                evidence={
                    "symbol": symbol,
                    "loss_from_symbol": f"{worst:.2f}",
                    "total_loss": f"{total:.2f}",
                    "symbols_traded": str(len(by_symbol)),
                },
                recommendation=(
                    f"Check whether {symbol} suits this strategy at all before tuning "
                    "parameters that apply to every symbol equally."
                ),
                strategy_id=strategy_id,
            )
        ]

    def _loss_streak(self, trades: Sequence[ClosedTrade], strategy_id: str | None) -> list[Finding]:
        """Flag the longest run of consecutive losses."""
        ordered = sorted(trades, key=lambda trade: trade.exit_time)
        longest = current = 0
        for trade in ordered:
            current = current + 1 if trade.net_pnl <= ZERO else 0
            longest = max(longest, current)

        if longest < STREAK_THRESHOLD:
            return []
        return [
            Finding(
                kind=FindingKind.STREAK,
                severity=(
                    Severity.URGENT
                    if longest >= STREAK_THRESHOLD * URGENT_STREAK_MULTIPLE
                    else Severity.NOTABLE
                ),
                headline=f"Longest losing streak was {longest} trades",
                evidence={
                    "streak": str(longest),
                    "total_trades": str(len(trades)),
                    "cooldown_threshold": str(STREAK_THRESHOLD),
                },
                recommendation=(
                    "Confirm the loss-streak cooldown is enabled; a run this long usually "
                    "means the regime turned and the strategy kept firing into it."
                ),
                strategy_id=strategy_id,
            )
        ]

    def _fee_burden(self, losers: Sequence[ClosedTrade], strategy_id: str | None) -> list[Finding]:
        """Flag losses that were caused by fees rather than by direction."""
        fee_only = [trade for trade in losers if trade.gross_pnl > ZERO and trade.net_pnl <= ZERO]
        if not fee_only:
            return []
        share = safe_divide(Decimal(len(fee_only)), Decimal(len(losers)))
        return [
            Finding(
                kind=FindingKind.COST_DRAG,
                severity=Severity.URGENT if share >= Decimal("0.3") else Severity.NOTABLE,
                headline=(
                    f"{len(fee_only)} of {len(losers)} losing trades were directionally "
                    "correct and lost only to fees"
                ),
                evidence={
                    "fee_only_losses": str(len(fee_only)),
                    "total_losses": str(len(losers)),
                    "share": f"{share:.0%}",
                    "fees_on_those_trades": (
                        f"{sum((trade.fees for trade in fee_only), ZERO):.2f}"
                    ),
                },
                recommendation=(
                    "These trades called direction correctly. The fix is execution or "
                    "holding period, not the entry logic."
                ),
                strategy_id=strategy_id,
            )
        ]


__all__ = [
    "CONCENTRATION_THRESHOLD",
    "MIN_SYMBOLS_FOR_CONCENTRATION",
    "MIN_TRADES_FOR_ANALYSIS",
    "STREAK_THRESHOLD",
    "URGENT_STREAK_MULTIPLE",
    "Finding",
    "FindingKind",
    "ResearchAgent",
    "ResearchReport",
    "Severity",
]
