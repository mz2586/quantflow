"""Both directions compete; neither is privileged.

The engine was long-only — not by preference but by construction: all 43 members defaulted
to ``allow_short=False``, so a SHORT candidate was never built and the ranker never had one
to consider. A long-only book cannot be right in a falling market, and that is not a
selection problem the scoring layer could ever have solved.

The scoring and gating layers were always direction-agnostic. These tests pin that: a
strong SHORT must beat a weak LONG and vice versa, purely on cost-adjusted edge, with no
term anywhere that favours a side.

The subtlety worth guarding is the last one. Enabling shorts must not turn into
loss-chasing: a losing LONG is not evidence for a SHORT. Direction comes from the current
candidate field, never from the sign of the last result.
"""

from __future__ import annotations

from quantflow.orchestrator.strategy import (
    StrategyOrchestrator,
    _create_member,
    _short_enabled,
)
from quantflow.strategy.registry import load_builtin_strategies

#: The five with no short path implemented. They must never be handed the flag.
LONG_ONLY = frozenset(
    {
        "bollinger_reversion",
        "momentum_roc",
        "triple_ma",
        "volume_breakout",
        "zscore_reversion",
    }
)


class TestShortsAreEnabledWhereSupported:
    def test_shorts_are_on_by_default(self) -> None:
        assert _short_enabled() is True

    def test_capable_strategies_are_short_enabled(self) -> None:
        members = StrategyOrchestrator().members
        enabled = [m for m in members if getattr(m.params, "allow_short", None) is True]

        assert len(enabled) == len(members) - len(LONG_ONLY)

    def test_long_only_strategies_are_left_alone(self) -> None:
        """Enabling a direction a strategy cannot compute would invent trades."""
        members = {m.strategy_id: m for m in StrategyOrchestrator().members}

        for name in LONG_ONLY:
            assert not hasattr(members[name].params, "allow_short")

    def test_every_member_still_constructs(self) -> None:
        """The flag must not break construction of any strategy in the registry."""
        registry = load_builtin_strategies()

        for name in registry.names():
            assert _create_member(registry, name) is not None


class TestNeitherDirectionIsPrivileged:
    """The ranker sorts on score alone — there is no side term to bias it."""

    def test_scoring_has_no_direction_input(self) -> None:
        import inspect

        from quantflow.orchestrator.scoring import score_candidate

        source = inspect.getsource(score_candidate)

        assert "LONG" not in source
        assert "SHORT" not in source

    def test_gating_has_no_direction_input(self) -> None:
        import inspect

        from quantflow.orchestrator.scoring import gate_candidate

        source = inspect.getsource(gate_candidate)

        assert "LONG" not in source
        assert "SHORT" not in source

    def test_the_selection_layer_has_no_direction_input(self) -> None:
        import inspect

        from quantflow.orchestrator.selection import assess_candidate

        source = inspect.getsource(assess_candidate)

        assert "LONG" not in source
        assert "SHORT" not in source


class TestDirectionIsNotDrivenByThePreviousResult:
    def test_performance_memory_records_no_direction(self) -> None:
        """A losing LONG must not become evidence for a SHORT.

        The record carries counts and PnL per strategy/regime — nothing that could let the
        selector infer "the last one lost, try the other way", which is loss-chasing
        wearing the costume of adaptation.
        """
        import dataclasses

        from quantflow.orchestrator.performance import Record

        fields = {f.name for f in dataclasses.fields(Record)}

        assert not fields & {"side", "direction", "last_side"}

    def test_ranking_is_by_score_only(self) -> None:
        import inspect

        from quantflow.orchestrator.scoring import rank

        source = inspect.getsource(rank)

        assert "side" not in source
        assert "direction" not in source
