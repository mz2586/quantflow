"""A second leg must be a new opportunity, never the same one twice.

The live evidence these guard against: 132 selections in one session, every one long, from
four correlated trend-following families, scores inside a 0.008 band. Ungated, a pyramid
would have bought the same opinion repeatedly.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.enums import MarketRegime, SignalDirection
from quantflow.orchestrator.pyramid import EntryThesis, pyramid_verdict
from quantflow.orchestrator.selection import strategy_family


def thesis(**overrides: object) -> EntryThesis:
    base: dict[str, object] = {
        "strategy_id": "triple_ma",
        "direction": SignalDirection.LONG,
        "regime": MarketRegime.RANGE,
        "score": Decimal("0.664"),
    }
    base.update(overrides)
    return EntryThesis(**base)  # type: ignore[arg-type]


def verdict(**overrides: object) -> tuple[bool, str]:
    base: dict[str, object] = {
        "strategy_id": "triple_ma",
        "direction": SignalDirection.LONG,
        "regime": MarketRegime.RANGE,
        "score": Decimal("0.665"),
        "unrealized_pnl": Decimal("12"),
    }
    base.update(overrides)
    return pyramid_verdict(thesis(), **base)  # type: ignore[arg-type]


class TestDuplicateThesisIsRefused:
    def test_same_family_regime_and_score_is_refused(self) -> None:
        # The exact live pattern: keltner_trend long at 0.662 beside triple_ma long at
        # 0.664 — different name, same trend family, same regime, same direction.
        allowed, why = verdict(strategy_id="keltner_trend", score=Decimal("0.662"))
        assert allowed is False
        assert "same thesis twice" in why

    def test_an_identical_repeat_is_refused(self) -> None:
        allowed, why = verdict()
        assert allowed is False
        assert "not" in why


class TestAveragingDownIsRefused:
    def test_an_underwater_position_never_takes_a_leg(self) -> None:
        # The venue nets legs into one position, so adding to a loser moves the average
        # entry against us. That is averaging down whatever the intent.
        allowed, why = verdict(strategy_id="rsi_reversion", unrealized_pnl=Decimal("-0.01"))
        assert allowed is False
        assert "averaging down" in why

    def test_underwater_beats_even_a_materially_different_signal(self) -> None:
        allowed, _ = verdict(
            strategy_id="bollinger_reversion",
            regime=MarketRegime.BULL_TREND,
            score=Decimal("0.90"),
            unrealized_pnl=Decimal("-50"),
        )
        assert allowed is False


class TestOppositeDirectionIsNotALeg:
    def test_a_reversal_is_left_to_the_exit_logic(self) -> None:
        allowed, why = verdict(direction=SignalDirection.SHORT)
        assert allowed is False
        assert "reversal" in why


class TestGenuinelyNewOpportunitiesAreAllowed:
    def test_a_different_strategy_family_qualifies(self) -> None:
        allowed, why = verdict(strategy_id="rsi_reversion")
        assert allowed is True
        assert "different strategy family" in why

    def test_a_regime_change_qualifies(self) -> None:
        allowed, why = verdict(regime=MarketRegime.BULL_TREND)
        assert allowed is True
        assert "regime changed" in why

    def test_a_materially_better_score_qualifies(self) -> None:
        allowed, why = verdict(score=Decimal("0.70"))
        assert allowed is True
        assert "score improved" in why

    def test_a_marginally_better_score_does_not(self) -> None:
        allowed, _ = verdict(score=Decimal("0.6741"))
        assert allowed is False


class TestFamilyClassification:
    """Independence is decided by the same taxonomy confluence uses, not a second one."""

    def test_the_two_strategies_that_actually_compete_share_a_family(self) -> None:
        # triple_ma and keltner_trend win almost every selection on this account. If these
        # read as different families, a duplicate leg gets admitted — which an earlier
        # substring-based implementation did.
        assert strategy_family("triple_ma") == strategy_family("keltner_trend") == "trend"

    def test_reversion_strategies_share_a_family(self) -> None:
        assert strategy_family("rsi_reversion") == strategy_family("zscore_reversion")

    def test_a_reversion_differs_from_a_trend(self) -> None:
        assert strategy_family("rsi_reversion") != strategy_family("keltner_trend")
