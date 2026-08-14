"""The net-profit exit: bank the trade the moment it is genuinely worth banking.

A 15m position that goes +0.25% and comes back to flat has, until now, produced nothing
except two fee payments and an entry in the trade log. The strategy could not object — it
does not speak until the bar closes — and the ladder was still waiting for its next rung.

So this layer asks a different question on every tick: *if I closed right now, what would
I actually keep?* Gross move, minus the entry fee, the exit fee, the half-spread and
expected slippage. When that number clears a configured buffer the position is closed on
that tick: no candle, no target, no strategy opinion, no stage 3.

The rule that keeps it honest is the one in the middle of this file: a position that is
green on the screen but red once the book is priced in **stays open**. An exit rule that
harvests the spread is not a profit engine, it is a fee engine with good manners.

Every price here arrives as a ticker tick. The word "candle" appears nowhere in the
execution path under test, which is the property
:func:`test_the_decision_cannot_see_a_candle_or_a_strategy` pins structurally.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.live.intrabar_manager import IntrabarManager
from quantflow.position.intrabar import (
    PRIORITY_INTRABAR,
    PRIORITY_NET_PROFIT_EXIT,
    PRIORITY_RISK_FLATTEN,
    PRIORITY_STRATEGY_EXIT,
    ActionKind,
    IntrabarConfig,
    ManagementAction,
    PositionState,
    net_profit_pct,
    on_price,
    resolve_actions,
)

BTC = Symbol.parse("BTC/USDT")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
ENTRY = Decimal("100")
QTY = Decimal("10")
ATR = Decimal("0.5")

#: The default cost model, itemised: 0.06% in, 0.06% out, 0.02% spread, 0.02% slippage.
COSTS = Decimal("0.0016")
BUFFER = Decimal("0.0005")

#: Costs plus buffer, so the exit fires at +0.21% gross and not a tick before.
THRESHOLD_LONG = ENTRY * (Decimal("1") + COSTS + BUFFER)  # 100.21
THRESHOLD_SHORT = ENTRY * (Decimal("1") - COSTS - BUFFER)  # 99.79


def config(**overrides: object) -> IntrabarConfig:
    base: dict[str, object] = {
        "enabled": True,
        "net_profit_exit_enabled": True,
        "entry_fee_pct": Decimal("0.0006"),
        "exit_fee_pct": Decimal("0.0006"),
        "spread_pct": Decimal("0.0002"),
        "slippage_pct": Decimal("0.0002"),
        "min_net_profit_pct": BUFFER,
    }
    base.update(overrides)
    return IntrabarConfig(**base)  # type: ignore[arg-type]


def long_state(**overrides: object) -> PositionState:
    state = PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=ENTRY,
        quantity=QTY,
        stop=Decimal("99"),
        opened_at=NOW,
    )
    return state if not overrides else _replace(state, **overrides)


def short_state(**overrides: object) -> PositionState:
    state = PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.SHORT,
        entry_price=ENTRY,
        quantity=QTY,
        stop=Decimal("101"),
        opened_at=NOW,
    )
    return state if not overrides else _replace(state, **overrides)


def _replace(state: PositionState, **overrides: object) -> PositionState:
    return replace(state, **overrides)  # type: ignore[arg-type]


def tick(
    state: PositionState,
    price: str,
    *,
    cfg: IntrabarConfig | None = None,
    at: int = 1,
) -> tuple[PositionState, ManagementAction]:
    """Deliver one live ticker price. There is no candle in this call."""
    return on_price(
        state,
        Decimal(price),
        atr=ATR,
        config=cfg or config(),
        now=NOW + timedelta(seconds=at),
    )


class TestItClosesWhenTheProfitIsReal:
    """(1) and (2): a position that becomes net profitable between candles is closed."""

    def test_a_long_that_turns_net_profitable_is_closed_on_the_tick(self) -> None:
        _, action = tick(long_state(), "100.25")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_long_close_is_ranked_as_the_net_profit_exit(self) -> None:
        _, action = tick(long_state(), "100.25")

        assert action.priority == PRIORITY_NET_PROFIT_EXIT

    def test_the_long_close_takes_the_whole_remaining_position(self) -> None:
        state, action = tick(long_state(), "100.25")

        assert action.close_quantity == QTY
        assert state.quantity == Decimal("0")

    def test_a_short_that_turns_net_profitable_is_closed_on_the_tick(self) -> None:
        """(2) The mirror image. Same rule, opposite direction, no second code path."""
        _, action = tick(short_state(), "99.75")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_short_close_is_ranked_as_the_net_profit_exit(self) -> None:
        _, action = tick(short_state(), "99.75")

        assert action.priority == PRIORITY_NET_PROFIT_EXIT

    def test_the_close_happens_far_below_the_first_ladder_rung(self) -> None:
        """It does not wait for stage 1 at +0.25%, let alone stage 3 at +0.75%."""
        _, action = tick(long_state(), "100.22")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_close_happens_without_reaching_the_target(self) -> None:
        _, action = tick(long_state(target=Decimal("110")), "100.25")

        assert action.kind is ActionKind.FULL_CLOSE
        assert "net profit exit" in action.reason


class TestItRefusesToCloseAProfitThatIsNotOne:
    """(3) and (4): the buffer is the whole point of the rule."""

    def test_a_green_position_below_the_cost_line_stays_open(self) -> None:
        """+0.10% gross is -0.06% net. Closing it would be paying to lose money."""
        state, action = tick(long_state(), "100.10")

        assert action.kind is ActionKind.NONE
        assert state.quantity == QTY

    def test_that_position_really_is_net_negative(self) -> None:
        assert net_profit_pct(long_state(), Decimal("100.10"), config()) < Decimal("0")

    def test_a_short_green_below_the_cost_line_stays_open(self) -> None:
        state, action = tick(short_state(), "99.90")

        assert action.kind is ActionKind.NONE
        assert state.quantity == QTY

    def test_a_position_between_costs_and_buffer_stays_open(self) -> None:
        """Net +0.02% is real money and still not worth a round trip's risk."""
        _, action = tick(long_state(), "100.18")

        assert action.kind is ActionKind.NONE

    def test_crossing_the_threshold_closes_immediately(self) -> None:
        """(4) Three ticks: below, below, through. The third one exits."""
        state = long_state()
        kinds = []
        for index, price in enumerate(["100.05", "100.15", "100.22"], start=1):
            state, action = tick(state, price, at=index)
            kinds.append(action.kind)

        assert kinds == [ActionKind.NONE, ActionKind.NONE, ActionKind.FULL_CLOSE]

    def test_nothing_is_closed_while_the_position_is_red(self) -> None:
        _, action = tick(long_state(), "99.95")

        assert action.kind is not ActionKind.FULL_CLOSE


class TestTheExactBoundary:
    """A threshold that is off by a tick is a different rule with the same name."""

    def test_exactly_at_the_buffer_closes(self) -> None:
        _, action = tick(long_state(), str(THRESHOLD_LONG))

        assert action.kind is ActionKind.FULL_CLOSE

    def test_exactly_at_the_buffer_nets_exactly_the_buffer(self) -> None:
        assert net_profit_pct(long_state(), THRESHOLD_LONG, config()) == BUFFER

    def test_one_hundredth_of_a_percent_short_of_it_does_not(self) -> None:
        _, action = tick(long_state(), "100.20")

        assert action.kind is ActionKind.NONE

    def test_the_short_boundary_mirrors_it_exactly(self) -> None:
        _, action = tick(short_state(), str(THRESHOLD_SHORT))

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_short_side_one_tick_inside_the_boundary_stays_open(self) -> None:
        _, action = tick(short_state(), "99.80")

        assert action.kind is ActionKind.NONE

    @pytest.mark.parametrize("move", ["0.0010", "0.0021", "0.0030", "0.0100"])
    def test_both_sides_compute_the_same_net_profit(self, move: str) -> None:
        """Symmetry, proven arithmetically rather than by two hand-picked prices."""
        step = ENTRY * Decimal(move)
        long_net = net_profit_pct(long_state(), ENTRY + step, config())
        short_net = net_profit_pct(short_state(), ENTRY - step, config())

        assert long_net == short_net

    @pytest.mark.parametrize("move", ["0.0010", "0.0021", "0.0030", "0.0100"])
    def test_both_sides_reach_the_same_decision(self, move: str) -> None:
        step = ENTRY * Decimal(move)
        _, long_action = tick(long_state(), str(ENTRY + step))
        _, short_action = tick(short_state(), str(ENTRY - step))

        assert long_action.kind is short_action.kind


class TestItCanBeSwitchedOff:
    def test_disabling_it_leaves_the_ladder_in_charge(self) -> None:
        _, action = tick(long_state(), "100.30", cfg=config(net_profit_exit_enabled=False))

        assert action.kind is ActionKind.MOVE_STOP

    def test_disabling_it_lets_a_position_run_past_the_threshold(self) -> None:
        state, _ = tick(long_state(), "100.25", cfg=config(net_profit_exit_enabled=False))

        assert state.quantity == QTY

    def test_it_is_off_when_the_whole_layer_is_off(self) -> None:
        _, action = tick(long_state(), "100.50", cfg=config(enabled=False))

        assert action.kind is ActionKind.NONE

    def test_a_wider_cost_model_moves_the_threshold_out(self) -> None:
        """An illiquid symbol should need a bigger move, and it is one config value.

        +0.25% no longer clears 0.34% of costs, so the ladder takes the tick instead — the
        position is protected rather than harvested.
        """
        state, action = tick(long_state(), "100.25", cfg=config(slippage_pct=Decimal("0.0020")))

        assert action.kind is ActionKind.MOVE_STOP
        assert state.quantity == QTY


class TestItRanksBelowRiskAndAboveEverythingElse:
    def test_an_account_flatten_outranks_a_profitable_exit(self) -> None:
        flatten = ManagementAction(
            kind=ActionKind.FULL_CLOSE, reason="kill switch", priority=PRIORITY_RISK_FLATTEN
        )
        _, profit = tick(long_state(), "100.25")

        assert resolve_actions([profit, flatten]) is flatten

    def test_the_profit_exit_outranks_the_trail_and_the_strategy(self) -> None:
        _, profit = tick(long_state(), "100.25")
        trail = ManagementAction(
            kind=ActionKind.MOVE_STOP, new_stop=Decimal("100"), priority=PRIORITY_INTRABAR
        )
        strategy = ManagementAction(kind=ActionKind.FULL_CLOSE, priority=PRIORITY_STRATEGY_EXIT)

        assert resolve_actions([trail, strategy, profit]) is profit

    def test_it_ranks_strictly_below_a_risk_flatten(self) -> None:
        assert PRIORITY_RISK_FLATTEN < PRIORITY_NET_PROFIT_EXIT


class TestProtectedAndPartiallyClosedPositions:
    """(9) A position that has already banked a partial is still managed correctly."""

    def _after_a_partial(self) -> PositionState:
        """Stage 3 has fired: two thirds left, a ratcheted stop, profit already realised."""
        return long_state(
            quantity=Decimal("6.7"),
            original_quantity=QTY,
            current_stop=Decimal("100.32"),
            stages_done=frozenset({0, 1, 2}),
            realized_pnl=Decimal("2.64"),
            high_water=Decimal("100.80"),
        )

    def test_only_the_remaining_quantity_is_closed(self) -> None:
        _, action = tick(self._after_a_partial(), "100.60")

        assert action.close_quantity == Decimal("6.7")

    def test_the_earlier_realised_profit_is_not_lost(self) -> None:
        state, _ = tick(self._after_a_partial(), "100.60")

        assert state.realized_pnl > Decimal("2.64")

    def test_the_position_ends_flat(self) -> None:
        state, _ = tick(self._after_a_partial(), "100.60")

        assert state.is_closed

    def test_a_later_tick_on_a_flat_position_does_nothing(self) -> None:
        state, _ = tick(self._after_a_partial(), "100.60")

        _, action = tick(state, "100.70", at=2)

        assert action.kind is ActionKind.NONE

    def test_a_crossed_protective_stop_still_closes_the_remainder(self) -> None:
        """Below the profit threshold the locked stop is what acts, and it still acts."""
        state, action = tick(self._after_a_partial(), "100.10")

        assert action.kind is ActionKind.FULL_CLOSE
        assert state.is_closed

    def test_that_close_is_ranked_as_intrabar_protection_not_a_profit_exit(self) -> None:
        _, action = tick(self._after_a_partial(), "100.10")

        assert action.priority == PRIORITY_INTRABAR


class TestStateSurvivesARestart:
    """(10) Everything the rule depends on has to survive a process restart."""

    def test_state_round_trips_exactly(self) -> None:
        state, _ = tick(long_state(), "100.10")

        assert PositionState.from_dict(state.to_dict()) == state

    def test_the_last_price_survives_the_round_trip(self) -> None:
        """Without it a restart cannot tell how fast the last move was."""
        state, _ = tick(long_state(), "100.10")

        assert PositionState.from_dict(state.to_dict()).last_price == Decimal("100.10")

    def test_the_invalidation_level_survives_the_round_trip(self) -> None:
        state = long_state(invalidation_price=Decimal("99.5"))

        assert PositionState.from_dict(state.to_dict()).invalidation_price == Decimal("99.5")

    def test_an_old_payload_without_the_new_fields_still_loads(self) -> None:
        """A restart onto the new build must not fail on state the old build wrote."""
        payload = long_state().to_dict()
        del payload["invalidation_price"]
        del payload["last_price"]

        restored = PositionState.from_dict(payload)

        assert restored.invalidation_price is None
        assert restored.last_price is None

    def test_a_restored_state_reaches_the_same_decision(self) -> None:
        state, _ = tick(long_state(), "100.10")

        _, action = tick(PositionState.from_dict(state.to_dict()), "100.25", at=2)

        assert action.kind is ActionKind.FULL_CLOSE


class TestNoCandleAndNoStrategy:
    def test_the_decision_cannot_see_a_candle_or_a_strategy(self) -> None:
        """The structural guarantee: there is no parameter for either to arrive through."""
        assert set(inspect.signature(on_price).parameters) == {
            "state",
            "price",
            "atr",
            "config",
            "now",
        }


# --------------------------------------------------------------------------- #
# Through the live manager and a venue that answers consistently
# --------------------------------------------------------------------------- #


class FakeTicker:
    def __init__(self, price: str) -> None:
        self.last = Decimal(price)
        self.timestamp = NOW


class FakeStream:
    def __init__(self, prices: list[str]) -> None:
        self._prices = prices

    async def watch_ticker(self, symbol: Symbol) -> Any:
        for price in self._prices:
            yield FakeTicker(price)
            await asyncio.sleep(0)


class RecordingGateway:
    """Records what reached the venue, and then answers *as* that venue.

    The manager re-reads the book immediately before acting and reconciles against it on a
    cadence, so a fake whose ``fetch_positions`` contradicts the orders it just accepted is
    not a stricter test — it is an inconsistent exchange.
    """

    def __init__(
        self,
        *,
        entry: Decimal = ENTRY,
        stop: Decimal = Decimal("99"),
        reject_orders: bool = False,
    ) -> None:
        self.stops: list[Decimal] = []
        self.orders: list[Any] = []
        self.calls: list[str] = []
        self._entry = entry
        self._quantity = QTY
        self._stop = stop
        self._reject = reject_orders

    async def fetch_positions(self) -> list[dict[str, Any]]:
        self.calls.append("fetch_positions")
        if self._quantity <= 0:
            return []
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": str(self._quantity),
                "entryPrice": str(self._entry),
                "info": {"stopLoss": str(self._stop)},
            }
        ]

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        assert stop_loss is not None
        self.calls.append("set_trading_stop")
        self.stops.append(stop_loss)
        self._stop = stop_loss
        return stop_loss

    async def submit_order(self, request: Any) -> Any:
        self.calls.append("submit_order")
        if self._reject:
            raise RuntimeError("venue rejected the order")
        self.orders.append(request)
        self._quantity = max(Decimal("0"), self._quantity - request.quantity)
        return request

    @property
    def venue_stop(self) -> Decimal:
        return self._stop


async def run(gateway: RecordingGateway, prices: list[str]) -> IntrabarManager:
    manager = IntrabarManager(gateway, FakeStream(prices), config(), clock=lambda: NOW)
    manager.track(long_state())
    manager.set_atr(BTC, ATR)
    await manager.start([BTC])
    await asyncio.sleep(0.05)
    await manager.stop()
    return manager


class TestItReachesTheVenue:
    async def test_a_ticker_price_alone_produces_a_close_order(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25"])

        assert len(gateway.orders) == 1

    async def test_the_close_is_reduce_only(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25"])

        assert gateway.orders[0].reduce_only is True

    async def test_the_close_sells_the_whole_position(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25"])

        assert gateway.orders[0].quantity == QTY

    async def test_no_zero_quantity_order_is_ever_sent(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25", "100.26", "100.27"])

        assert all(order.quantity > 0 for order in gateway.orders)

    async def test_a_position_below_the_threshold_is_left_alone(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.10", "100.15", "100.18"])

        assert not gateway.orders


class TestNoDuplicateCloses:
    """(5) Ticks arrive milliseconds apart; the position may only be closed once."""

    async def test_a_burst_of_qualifying_ticks_produces_one_order(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25", "100.26", "100.27", "100.28", "100.29", "100.30"])

        assert len(gateway.orders) == 1

    async def test_the_symbol_is_no_longer_managed_afterwards(self) -> None:
        gateway = RecordingGateway()

        manager = await run(gateway, ["100.25", "100.26", "100.27"])

        assert manager.state_for(BTC) is None


class TestTheVenueConfirmsBeforeLocalStateFinalises:
    """(6) and (8): nothing is believed until the exchange has been asked."""

    async def test_the_venue_is_read_before_the_order_is_sent(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["100.25"])

        assert gateway.calls.index("fetch_positions") < gateway.calls.index("submit_order")

    async def test_a_rejected_close_leaves_the_position_managed(self) -> None:
        """If the order did not land, the position is still ours to protect."""
        gateway = RecordingGateway(reject_orders=True)

        manager = await run(gateway, ["100.25"])

        assert manager.state_for(BTC) is not None

    async def test_a_rejected_close_leaves_the_quantity_intact(self) -> None:
        gateway = RecordingGateway(reject_orders=True)

        manager = await run(gateway, ["100.25"])
        state = manager.state_for(BTC)

        assert state is not None
        assert state.quantity == QTY

    async def test_the_exchange_stop_is_untouched_by_a_profit_close(self) -> None:
        """(8) The stop that protects the position stays exactly where the venue has it."""
        gateway = RecordingGateway()

        await run(gateway, ["100.25"])

        assert not gateway.stops
        assert gateway.venue_stop == Decimal("99")

    async def test_the_exchange_stop_survives_a_rejected_close(self) -> None:
        gateway = RecordingGateway(reject_orders=True)

        await run(gateway, ["100.25", "100.26"])

        assert gateway.venue_stop == Decimal("99")


class TestReconciliationStaysAuthoritative:
    """(7) The venue's book wins every disagreement, before and after the exit."""

    async def test_a_different_entry_price_replaces_local_state(self) -> None:
        gateway = RecordingGateway(entry=Decimal("101"))

        manager = await run(gateway, ["100.10"])
        state = manager.state_for(BTC)

        assert state is not None
        assert state.entry_price == Decimal("101")

    async def test_the_replaced_position_is_judged_on_its_own_entry(self) -> None:
        """+0.25% against the *venue's* entry is still red; nothing may be closed."""
        gateway = RecordingGateway(entry=Decimal("101"))

        await run(gateway, ["100.25", "100.26"])

        assert not gateway.orders

    async def test_a_position_gone_from_the_venue_is_untracked(self) -> None:
        gateway = RecordingGateway()

        manager = await run(gateway, ["100.25", "100.26", "100.27"])

        assert manager.monitored == ()

    async def test_the_venue_stop_is_pulled_into_local_state(self) -> None:
        gateway = RecordingGateway(stop=Decimal("99.5"))

        manager = await run(gateway, ["100.10"])
        state = manager.state_for(BTC)

        assert state is not None
        assert state.current_stop == Decimal("99.5")
