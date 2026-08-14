"""Intrabar position management: react to live price, not to the next candle close.

Strategy exits only fire on completed 5m/15m bars. A position that runs in our favour
mid-bar has, until now, sat on the static stop it was opened with — so a favourable
excursion could round-trip back to the entry stop and the engine would not notice until
the bar closed. This layer watches every tick.

The invariant these tests exist to defend, above all others: **the stop ratchets**. It may
move only in the position's favour, never away from it, under any code path. A stop that
loosens is worse than no stop, because the position is reported as protected while it is
not. It is asserted here across a long adversarial price walk rather than on a handful of
fixtures, because a monotonicity bug hides easily in the cases nobody thought to write.

Two other properties are deliberately pinned:

* **Breakeven includes fees.** Moving the stop to the raw entry price books a small loss
  every time it is hit. "Breakeven" that loses money is a misnomer with a cost attached.
* **A stage fires once.** Re-firing would emit duplicate close orders on every subsequent
  tick — the failure mode that turns a profit-taking feature into an order storm.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.domain.enums import PositionSide
from quantflow.position.intrabar import (
    ActionKind,
    IntrabarConfig,
    ManagementAction,
    PositionState,
    is_stale,
    on_price,
    resolve_actions,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ENTRY = Decimal("100")
QTY = Decimal("10")
ATR = Decimal("0.5")


def config(**overrides: object) -> IntrabarConfig:
    base: dict[str, object] = {"enabled": True}
    base.update(overrides)
    return IntrabarConfig(**base)  # type: ignore[arg-type]


def long_state(stop: str = "99", target: str | None = "105") -> PositionState:
    return PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=ENTRY,
        quantity=QTY,
        stop=Decimal(stop),
        opened_at=NOW,
        target=Decimal(target) if target else None,
    )


def short_state(stop: str = "101", target: str | None = "95") -> PositionState:
    return PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.SHORT,
        entry_price=ENTRY,
        quantity=QTY,
        stop=Decimal(stop),
        opened_at=NOW,
        target=Decimal(target) if target else None,
    )


def tick(
    state: PositionState, price: str, *, cfg: IntrabarConfig | None = None, at: int = 1
) -> tuple[PositionState, ManagementAction]:
    return on_price(
        state,
        Decimal(price),
        atr=ATR,
        config=cfg or config(),
        now=NOW + timedelta(seconds=at),
    )


def walk(
    state: PositionState, prices: list[str], cfg: IntrabarConfig | None = None
) -> tuple[PositionState, list[ManagementAction]]:
    actions: list[ManagementAction] = []
    for i, price in enumerate(prices, start=1):
        state, action = tick(state, price, cfg=cfg, at=i)
        actions.append(action)
    return state, actions


class TestLongStages:
    def test_stage_one_moves_the_stop_to_breakeven_plus_fees(self) -> None:
        """+0.25% intrabar: the position can no longer become a loser."""
        state, action = tick(long_state(), "100.30")

        assert action.kind is ActionKind.MOVE_STOP
        assert state.current_stop >= ENTRY

    def test_breakeven_covers_fees_not_just_the_entry_price(self) -> None:
        """A stop at the raw entry books a loss once fees are paid."""
        state, _ = tick(long_state(), "100.30")

        assert state.current_stop > ENTRY

    def test_stage_two_locks_profit(self) -> None:
        """+0.50% intrabar: the stop sits above entry by roughly the locked amount."""
        state, _ = walk(long_state(), ["100.30", "100.60"])

        assert state.current_stop > ENTRY * Decimal("1.001")

    def test_stage_three_takes_partial_profit(self) -> None:
        _, actions = walk(long_state(), ["100.30", "100.60", "100.80"])

        assert any(a.kind is ActionKind.PARTIAL_CLOSE for a in actions)

    def test_partial_close_quantity_is_the_configured_fraction(self) -> None:
        _, actions = walk(long_state(), ["100.30", "100.60", "100.80"])
        partial = next(a for a in actions if a.kind is ActionKind.PARTIAL_CLOSE)

        assert partial.close_quantity == (QTY * Decimal("0.33")).quantize(partial.close_quantity)

    def test_a_reversal_after_stage_two_exits_without_a_candle(self) -> None:
        """The whole point: protection fires on the tick, not on the bar close."""
        state, _ = walk(long_state(), ["100.30", "100.60"])
        protected = state.current_stop

        _, action = tick(state, str(protected - Decimal("0.01")), at=9)

        assert action.kind is ActionKind.FULL_CLOSE

    def test_a_reversal_after_stage_two_still_exits_in_profit(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60"])

        assert state.current_stop > ENTRY


class TestShortStages:
    def test_stage_one_moves_the_stop_to_breakeven_plus_fees(self) -> None:
        state, action = tick(short_state(), "99.70")

        assert action.kind is ActionKind.MOVE_STOP
        assert state.current_stop < ENTRY

    def test_stage_two_locks_profit(self) -> None:
        state, _ = walk(short_state(), ["99.70", "99.40"])

        assert state.current_stop < ENTRY * Decimal("0.999")

    def test_stage_three_takes_partial_profit(self) -> None:
        _, actions = walk(short_state(), ["99.70", "99.40", "99.20"])

        assert any(a.kind is ActionKind.PARTIAL_CLOSE for a in actions)

    def test_a_reversal_after_stage_two_exits_without_a_candle(self) -> None:
        state, _ = walk(short_state(), ["99.70", "99.40"])

        _, action = tick(state, str(state.current_stop + Decimal("0.01")), at=9)

        assert action.kind is ActionKind.FULL_CLOSE


class TestTargetAndInvalidation:
    def test_crossing_the_target_between_closes_exits(self) -> None:
        _, action = tick(long_state(target="105"), "105.10")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_crossing_the_initial_stop_exits(self) -> None:
        _, action = tick(long_state(stop="99"), "98.90")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_a_short_crossing_its_target_exits(self) -> None:
        _, action = tick(short_state(target="95"), "94.90")

        assert action.kind is ActionKind.FULL_CLOSE


class TestIndependenceFromTheCandleLoop:
    def test_protection_never_consults_strategy_state(self) -> None:
        """A strategy HOLD cannot suppress protection: the module has no idea it exists.

        `on_price` takes only the position, the price, ATR and config — there is no
        parameter through which a strategy opinion could arrive, which is the structural
        guarantee behind requirement 7.
        """
        import inspect

        parameters = set(inspect.signature(on_price).parameters)

        assert parameters == {"state", "price", "atr", "config", "now"}

    def test_a_long_gap_between_ticks_does_not_refire_a_stage(self) -> None:
        """A websocket reconnect leaves a hole in the tick stream, not a reset."""
        state, _ = walk(long_state(), ["100.30", "100.60"])
        stages_before = state.stages_done

        state, _ = tick(state, "100.61", at=6000)

        assert state.stages_done == stages_before

    def test_water_marks_survive_a_reconnect_gap(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.90", "100.40"])

        assert state.high_water == Decimal("100.90")


class TestTheStopRatchets:
    def test_the_stop_never_moves_against_a_long(self) -> None:
        """The invariant, over an adversarial walk rather than a chosen fixture."""
        prices = [
            "100.30",
            "100.10",
            "100.60",
            "100.20",
            "100.80",
            "100.05",
            "101.00",
            "100.40",
            "101.50",
            "100.90",
        ]
        state = long_state()
        stops = [state.current_stop]
        for i, price in enumerate(prices, start=1):
            state, action = tick(state, price, at=i)
            if action.kind is ActionKind.FULL_CLOSE:
                break
            stops.append(state.current_stop)

        assert stops == sorted(stops)

    def test_the_stop_never_moves_against_a_short(self) -> None:
        prices = [
            "99.70",
            "99.90",
            "99.40",
            "99.80",
            "99.20",
            "99.95",
            "99.00",
            "99.60",
            "98.50",
            "99.10",
        ]
        state = short_state()
        stops = [state.current_stop]
        for i, price in enumerate(prices, start=1):
            state, action = tick(state, price, at=i)
            if action.kind is ActionKind.FULL_CLOSE:
                break
            stops.append(state.current_stop)

        assert stops == sorted(stops, reverse=True)


class TestNoDuplicateOrders:
    def test_a_stage_fires_at_most_once(self) -> None:
        """Re-firing would emit a close order on every subsequent tick."""
        _, actions = walk(long_state(), ["100.80", "100.81", "100.82", "100.83", "100.84"])
        partials = [a for a in actions if a.kind is ActionKind.PARTIAL_CLOSE]

        assert len(partials) == 1

    def test_no_action_is_emitted_when_nothing_changed(self) -> None:
        state, _ = tick(long_state(), "100.30")
        _, action = tick(state, "100.30", at=2)

        assert action.kind is ActionKind.NONE


class TestPartialCloseAccounting:
    def test_remaining_quantity_is_reduced(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60", "100.80"])

        assert state.quantity < QTY

    def test_the_original_quantity_is_preserved(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60", "100.80"])

        assert state.original_quantity == QTY

    def test_the_entry_price_is_unchanged_by_a_partial(self) -> None:
        """A partial exit realises PnL; it does not re-price the remainder."""
        state, _ = walk(long_state(), ["100.30", "100.60", "100.80"])

        assert state.entry_price == ENTRY

    def test_realized_pnl_is_recorded(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60", "100.80"])

        assert state.realized_pnl > 0


class TestRestartPreservesState:
    def test_state_round_trips(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60"])

        restored = PositionState.from_dict(state.to_dict())

        assert restored == state

    def test_stages_survive_the_round_trip(self) -> None:
        """Otherwise a restart re-fires every stage and duplicates the orders."""
        state, _ = walk(long_state(), ["100.30", "100.60", "100.80"])

        restored = PositionState.from_dict(state.to_dict())

        assert restored.stages_done == state.stages_done

    def test_the_ratcheted_stop_survives_the_round_trip(self) -> None:
        state, _ = walk(long_state(), ["100.30", "100.60"])

        assert PositionState.from_dict(state.to_dict()).current_stop == state.current_stop


class TestPriority:
    def test_a_higher_priority_action_wins(self) -> None:
        from quantflow.position.intrabar import PRIORITY_INTRABAR, PRIORITY_RISK_FLATTEN

        flatten = ManagementAction(
            kind=ActionKind.FULL_CLOSE, reason="kill switch", priority=PRIORITY_RISK_FLATTEN
        )
        trail = ManagementAction(
            kind=ActionKind.MOVE_STOP,
            new_stop=Decimal("100"),
            reason="trail",
            priority=PRIORITY_INTRABAR,
        )

        assert resolve_actions([trail, flatten]) is flatten

    def test_order_of_submission_does_not_matter(self) -> None:
        from quantflow.position.intrabar import PRIORITY_INTRABAR, PRIORITY_STRATEGY_EXIT

        intrabar = ManagementAction(
            kind=ActionKind.FULL_CLOSE, reason="trail hit", priority=PRIORITY_INTRABAR
        )
        strategy = ManagementAction(
            kind=ActionKind.FULL_CLOSE, reason="strategy exit", priority=PRIORITY_STRATEGY_EXIT
        )

        assert resolve_actions([intrabar, strategy]) is intrabar
        assert resolve_actions([strategy, intrabar]) is intrabar


class TestDisabledAndStale:
    def test_disabled_never_acts(self) -> None:
        """Default OFF: a change to how every position exits is opt-in."""
        _, action = tick(long_state(), "100.80", cfg=config(enabled=False))

        assert action.kind is ActionKind.NONE

    def test_the_default_config_is_disabled(self) -> None:
        assert IntrabarConfig().enabled is False

    @pytest.mark.parametrize(("age_seconds", "expected"), [(1, False), (30, True), (600, True)])
    def test_staleness_is_detected(self, age_seconds: int, expected: bool) -> None:
        assert is_stale(timedelta(seconds=age_seconds), timedelta(seconds=10)) is expected


class TestTrailing:
    def test_the_trail_respects_an_absolute_floor_in_a_calm_market(self) -> None:
        """Zero ATR must not collapse the trail onto the price itself."""
        state = long_state()
        state, _ = on_price(state, Decimal("101"), atr=Decimal("0"), config=config(), now=NOW)

        assert state.current_stop < Decimal("101")

    def test_types_stay_decimal(self) -> None:
        state, _ = tick(long_state(), "100.60")

        assert isinstance(state.current_stop, Decimal)
        assert isinstance(state.high_water, Decimal)
