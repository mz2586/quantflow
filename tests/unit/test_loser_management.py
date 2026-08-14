"""Losing positions are managed between candles too, not just winning ones.

The winner rules answer *when is this worth banking*. These answer the other half, which
is where the money actually goes: a position that is wrong stays wrong for fifteen minutes
at a time, and the only thing watching it is a static stop that was placed before anything
happened.

Four rules, and every threshold is derived from the position's **own** stop distance, its
**own** ATR and the cost model — never a flat percentage applied to every asset, because
"1% adverse" is a rounding error on one symbol and a blown thesis on another:

* **Hard max loss** — the definitive loss the risk engine sized the trade on, acted on the
  tick it is reached instead of at the next bar close.
* **Thesis invalidation** — only ever against a level someone supplied. `on_price` is pure
  and is never handed strategy state, so an unsupplied level means the rule is inactive,
  not that a thesis gets invented from price action.
* **Loss acceleration** — deep *and* abnormal, or fast: an exit earlier than the full stop
  when the move is not the sort of move this instrument makes.
* **Stale loser** — red for longer than the configured holding period, having never shown
  the edge it was opened for.

And the rule that stops all four from being a hair-trigger, tested on both sides: a
position that is merely negative by noise **stays open**. That is the mirror of the
winner's cost buffer, and without it this file would describe a machine for realising
small losses.

Nothing here amends or cancels the exchange-side stop. Every rule is an exit that arrives
*earlier* than the venue stop would; the venue stop keeps working if this process dies, and
the tests at the bottom assert that it is still sitting there untouched afterwards.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.live.intrabar_manager import IntrabarManager
from quantflow.position.intrabar import (
    PRIORITY_EXCHANGE_STOP,
    PRIORITY_LOSS_ACCELERATION,
    PRIORITY_NET_PROFIT_EXIT,
    PRIORITY_RISK_FLATTEN,
    PRIORITY_THESIS_INVALIDATION,
    PRIORITY_TIME_EXIT,
    ActionKind,
    IntrabarConfig,
    ManagementAction,
    PositionState,
    hard_max_loss_price,
    on_price,
    resolve_actions,
)

BTC = Symbol.parse("BTC/USDT")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
ENTRY = Decimal("100")
QTY = Decimal("10")

#: A two-point stop on a hundred-point instrument, so "60% of the stop distance" is a
#: price and not an abstraction: 98.80 for a long, 101.20 for a short.
STOP_LONG = Decimal("98")
STOP_SHORT = Decimal("102")

#: 0.4 makes 1.5x ATR = 0.60 and 2x ATR = 0.80, both comfortably inside the two-point stop
#: so the ATR clause and the stop-fraction clause can be tested apart from each other.
ATR = Decimal("0.4")


def config(**overrides: object) -> IntrabarConfig:
    base: dict[str, object] = {
        "enabled": True,
        "net_profit_exit_enabled": True,
        "min_net_profit_pct": Decimal("0.0005"),
        "loss_accel_stop_fraction": Decimal("0.6"),
        "loss_accel_atr_multiple": Decimal("1.5"),
        "loss_accel_burst_atr_multiple": Decimal("2"),
        "loss_accel_window": timedelta(seconds=60),
        "stale_loser_after": timedelta(hours=1),
    }
    base.update(overrides)
    return IntrabarConfig(**base)  # type: ignore[arg-type]


def long_state(**overrides: object) -> PositionState:
    state = PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=ENTRY,
        quantity=QTY,
        stop=STOP_LONG,
        opened_at=NOW,
    )
    return state if not overrides else replace(state, **overrides)  # type: ignore[arg-type]


def short_state(**overrides: object) -> PositionState:
    state = PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.SHORT,
        entry_price=ENTRY,
        quantity=QTY,
        stop=STOP_SHORT,
        opened_at=NOW,
    )
    return state if not overrides else replace(state, **overrides)  # type: ignore[arg-type]


def tick(
    state: PositionState,
    price: str,
    *,
    cfg: IntrabarConfig | None = None,
    at: int = 1,
    atr: Decimal | None = ATR,
) -> tuple[PositionState, ManagementAction]:
    """One live ticker price, ``at`` seconds after the position opened. No candle."""
    return on_price(
        state,
        Decimal(price),
        atr=atr,
        config=cfg or config(),
        now=NOW + timedelta(seconds=at),
    )


class TestHardMaxLoss:
    """The risk model's own limit, enforced on a tick instead of at the next bar close."""

    def test_the_level_is_the_stop_the_position_was_sized_on(self) -> None:
        assert hard_max_loss_price(long_state(), config()) == STOP_LONG

    def test_the_short_level_mirrors_it(self) -> None:
        assert hard_max_loss_price(short_state(), config()) == STOP_SHORT

    def test_a_long_reaching_it_exits_immediately(self) -> None:
        _, action = tick(long_state(), "98.00")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_long_exit_is_ranked_as_the_hard_stop(self) -> None:
        _, action = tick(long_state(), "97.90")

        assert action.priority == PRIORITY_EXCHANGE_STOP
        assert "hard max loss" in action.reason

    def test_a_short_reaching_it_exits_immediately(self) -> None:
        _, action = tick(short_state(), "102.00")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_short_exit_is_ranked_as_the_hard_stop(self) -> None:
        _, action = tick(short_state(), "102.10")

        assert action.priority == PRIORITY_EXCHANGE_STOP

    def test_the_whole_position_is_closed(self) -> None:
        state, _ = tick(long_state(), "98.00")

        assert state.is_closed

    def test_a_price_one_tick_inside_it_does_not_trip_the_hard_rule(self) -> None:
        """98.01 is not the max loss. (It is an accelerating one — tested separately.)"""
        _, action = tick(long_state(), "98.01", cfg=config(loss_accel_enabled=False))

        assert action.kind is ActionKind.NONE

    def test_a_configurable_multiple_moves_the_level(self) -> None:
        """Half the sized risk, for an operator who wants a tighter definitive loss."""
        cfg = config(hard_max_loss_r=Decimal("0.5"))

        assert hard_max_loss_price(long_state(), cfg) == Decimal("99.0")

    def test_it_can_be_switched_off(self) -> None:
        _, action = tick(long_state(), "98.00", cfg=config(hard_max_loss_enabled=False))

        assert "hard max loss" not in action.reason

    def test_a_position_with_no_stop_room_has_no_hard_level(self) -> None:
        """No risk distance means no risk model to enforce; inventing one is not an option."""
        flat = long_state(current_stop=ENTRY, initial_stop=ENTRY)

        assert hard_max_loss_price(flat, config()) is None

    def test_an_emergency_flatten_still_outranks_it(self) -> None:
        _, hard = tick(long_state(), "98.00")
        flatten = ManagementAction(
            kind=ActionKind.FULL_CLOSE, reason="kill switch", priority=PRIORITY_RISK_FLATTEN
        )

        assert resolve_actions([hard, flatten]) is flatten


class TestThesisInvalidation:
    """Opt-in by construction: a level arrives on the state or the rule does not exist."""

    def test_a_long_trading_through_a_supplied_level_exits(self) -> None:
        _, action = tick(long_state(invalidation_price=Decimal("99.50")), "99.40")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_exit_is_ranked_as_thesis_invalidation(self) -> None:
        _, action = tick(long_state(invalidation_price=Decimal("99.50")), "99.40")

        assert action.priority == PRIORITY_THESIS_INVALIDATION
        assert "thesis invalidated" in action.reason

    def test_it_fires_well_before_the_hard_stop(self) -> None:
        state, _ = tick(long_state(invalidation_price=Decimal("99.50")), "99.40")

        assert state.is_closed
        assert Decimal("99.40") > STOP_LONG

    def test_a_short_trading_through_its_level_exits(self) -> None:
        _, action = tick(short_state(invalidation_price=Decimal("100.50")), "100.60")

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_short_exit_is_ranked_the_same_way(self) -> None:
        _, action = tick(short_state(invalidation_price=Decimal("100.50")), "100.60")

        assert action.priority == PRIORITY_THESIS_INVALIDATION

    def test_the_same_price_does_nothing_when_no_level_was_supplied(self) -> None:
        """The rule is inactive rather than guessed. This is the whole design."""
        _, action = tick(long_state(), "99.40")

        assert action.kind is ActionKind.NONE

    def test_a_price_short_of_the_level_does_nothing(self) -> None:
        _, action = tick(long_state(invalidation_price=Decimal("99.50")), "99.60")

        assert action.kind is ActionKind.NONE

    def test_it_can_be_switched_off(self) -> None:
        _, action = tick(
            long_state(invalidation_price=Decimal("99.50")),
            "99.40",
            cfg=config(invalidation_exit_enabled=False),
        )

        assert action.kind is ActionKind.NONE

    def test_the_level_survives_a_restart(self) -> None:
        state = long_state(invalidation_price=Decimal("99.50"))

        assert PositionState.from_dict(state.to_dict()).invalidation_price == Decimal("99.50")


class TestLossAcceleration:
    """An exit earlier than the full stop, when the move is not an ordinary move."""

    def test_a_deep_and_abnormal_excursion_exits_before_the_stop(self) -> None:
        """1.20 against a 2.00 stop, and 3x ATR. The stop is 98; this closes at 98.80."""
        _, action = tick(long_state(), "98.80")

        assert action.kind is ActionKind.FULL_CLOSE
        assert action.priority == PRIORITY_LOSS_ACCELERATION

    def test_it_really_is_earlier_than_the_full_stop(self) -> None:
        state, _ = tick(long_state(), "98.80")

        assert state.is_closed
        assert Decimal("98.80") > STOP_LONG

    def test_the_short_side_mirrors_it(self) -> None:
        _, action = tick(short_state(), "101.20")

        assert action.kind is ActionKind.FULL_CLOSE
        assert action.priority == PRIORITY_LOSS_ACCELERATION

    def test_a_shallower_excursion_is_left_alone(self) -> None:
        """1.00 of a 2.00 stop is half the risk the trade was sized to take."""
        _, action = tick(long_state(), "99.00")

        assert action.kind is ActionKind.NONE

    def test_a_deep_excursion_that_is_normal_for_the_instrument_is_left_alone(self) -> None:
        """Same 1.20 move, but on a symbol whose ATR is 2.00. That is a Tuesday."""
        _, action = tick(long_state(), "98.80", atr=Decimal("2"))

        assert action.kind is ActionKind.NONE

    def test_a_fast_adverse_move_exits_even_when_it_is_shallow(self) -> None:
        """0.90 against the position in ten seconds is 2.25x ATR: a gap, not a drift."""
        state, _ = tick(long_state(), "99.90", at=1)

        _, action = tick(state, "99.00", at=11)

        assert action.kind is ActionKind.FULL_CLOSE
        assert action.priority == PRIORITY_LOSS_ACCELERATION

    def test_the_same_move_spread_over_an_hour_is_not_acceleration(self) -> None:
        """Identical prices, identical depth, arriving slowly. Nothing fires."""
        state, _ = tick(long_state(), "99.90", at=1)

        _, action = tick(state, "99.00", at=600)

        assert action.kind is ActionKind.NONE

    def test_the_fast_rule_needs_a_previous_tick(self) -> None:
        """On the first tick there is no "since when", so speed is unknowable."""
        _, action = tick(long_state(), "99.00", at=1)

        assert action.kind is ActionKind.NONE

    def test_the_fast_rule_mirrors_on_the_short_side(self) -> None:
        state, _ = tick(short_state(), "100.10", at=1)

        _, action = tick(state, "101.00", at=11)

        assert action.kind is ActionKind.FULL_CLOSE

    def test_a_fast_move_in_our_favour_is_not_a_loss(self) -> None:
        state, _ = tick(long_state(), "99.90", at=1, cfg=config(net_profit_exit_enabled=False))

        _, action = tick(state, "100.80", at=11, cfg=config(net_profit_exit_enabled=False))

        assert action.kind is not ActionKind.FULL_CLOSE

    def test_it_can_be_switched_off(self) -> None:
        _, action = tick(long_state(), "98.80", cfg=config(loss_accel_enabled=False))

        assert action.kind is ActionKind.NONE

    def test_the_fraction_is_configurable(self) -> None:
        """40% of the stop distance, for a shorter leash — one config value, not a rewrite."""
        _, action = tick(long_state(), "99.20", cfg=config(loss_accel_stop_fraction=Decimal("0.4")))

        assert action.kind is ActionKind.FULL_CLOSE


class TestOrdinaryNoiseIsNotAnExit:
    """The mirror of the winner's cost buffer. Without this, the rules above are a shredder."""

    def test_a_slightly_negative_long_stays_open(self) -> None:
        state, action = tick(long_state(), "99.95")

        assert action.kind is ActionKind.NONE
        assert state.quantity == QTY

    def test_a_slightly_negative_short_stays_open(self) -> None:
        state, action = tick(short_state(), "100.05")

        assert action.kind is ActionKind.NONE
        assert state.quantity == QTY

    def test_a_walk_of_small_negative_ticks_never_closes(self) -> None:
        state = long_state()
        kinds = []
        for index, price in enumerate(["99.97", "99.92", "99.98", "99.90", "99.95"], start=1):
            state, action = tick(state, price, at=index * 5)
            kinds.append(action.kind)

        assert kinds == [ActionKind.NONE] * 5
        assert state.quantity == QTY

    def test_the_short_walk_mirrors_it(self) -> None:
        state = short_state()
        kinds = []
        for index, price in enumerate(["100.03", "100.08", "100.02", "100.10", "100.05"], start=1):
            state, action = tick(state, price, at=index * 5)
            kinds.append(action.kind)

        assert kinds == [ActionKind.NONE] * 5
        assert state.quantity == QTY

    def test_a_position_that_recovers_is_still_open_to_recover_into(self) -> None:
        state, _ = tick(long_state(), "99.90", at=1)
        state, _ = tick(state, "99.95", at=20)

        assert not state.is_closed


class TestStaleLoser:
    """Red for too long, without ever having shown the edge it was opened for."""

    def test_a_long_still_red_after_the_timeout_exits(self) -> None:
        _, action = tick(long_state(), "99.90", at=7200)

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_exit_is_ranked_as_a_time_exit(self) -> None:
        _, action = tick(long_state(), "99.90", at=7200)

        assert action.priority == PRIORITY_TIME_EXIT
        assert "stale loser" in action.reason

    def test_a_short_still_red_after_the_timeout_exits(self) -> None:
        _, action = tick(short_state(), "100.10", at=7200)

        assert action.kind is ActionKind.FULL_CLOSE
        assert action.priority == PRIORITY_TIME_EXIT

    def test_it_does_not_fire_before_the_timeout(self) -> None:
        _, action = tick(long_state(), "99.90", at=600)

        assert action.kind is ActionKind.NONE

    def test_it_does_not_fire_on_a_position_that_is_green(self) -> None:
        """Old is not the same as failing. A working trade is left to work."""
        _, action = tick(long_state(), "100.10", at=7200, cfg=config(net_profit_exit_enabled=False))

        assert action.kind is ActionKind.NONE

    def test_a_position_that_did_show_its_edge_is_not_stale(self) -> None:
        """It earned the buffer once; that is the profit rules' business, not this rule's."""
        ran_up = long_state(high_water=Decimal("100.80"))

        _, action = tick(ran_up, "99.90", at=7200)

        assert action.kind is ActionKind.NONE

    def test_a_position_whose_best_never_cleared_costs_is_stale(self) -> None:
        """+0.10% at its best never covered 0.16% of costs. It never had the edge."""
        never_worked = long_state(high_water=Decimal("100.10"))

        _, action = tick(never_worked, "99.90", at=7200)

        assert action.kind is ActionKind.FULL_CLOSE

    def test_it_can_be_switched_off(self) -> None:
        _, action = tick(long_state(), "99.90", at=7200, cfg=config(stale_loser_enabled=False))

        assert action.kind is ActionKind.NONE

    def test_the_timeout_is_configurable(self) -> None:
        cfg = config(stale_loser_after=timedelta(minutes=5))

        _, action = tick(long_state(), "99.90", at=600, cfg=cfg)

        assert action.kind is ActionKind.FULL_CLOSE

    def test_the_clock_is_never_read_inside_the_decision(self) -> None:
        """Elapsed time comes from ``now``: the same inputs must give the same answer."""
        first = tick(long_state(), "99.90", at=7200)[1]
        second = tick(long_state(), "99.90", at=7200)[1]

        assert first.kind is second.kind is ActionKind.FULL_CLOSE


class TestWinnersAndLosersRankAgainstEachOther:
    def test_a_hard_stop_outranks_a_profitable_exit(self) -> None:
        assert PRIORITY_EXCHANGE_STOP < PRIORITY_NET_PROFIT_EXIT

    def test_invalidation_outranks_acceleration_which_outranks_profit(self) -> None:
        assert PRIORITY_THESIS_INVALIDATION < PRIORITY_LOSS_ACCELERATION < PRIORITY_NET_PROFIT_EXIT

    def test_the_stale_loser_is_the_weakest_reason_to_close(self) -> None:
        assert PRIORITY_TIME_EXIT > PRIORITY_NET_PROFIT_EXIT

    def test_a_position_that_is_green_enough_is_banked_not_held(self) -> None:
        _, action = tick(long_state(), "100.25")

        assert action.kind is ActionKind.FULL_CLOSE
        assert action.priority == PRIORITY_NET_PROFIT_EXIT

    def test_state_round_trips_after_a_loser_tick(self) -> None:
        state, _ = tick(long_state(invalidation_price=Decimal("99.5")), "99.90", at=30)

        assert PositionState.from_dict(state.to_dict()) == state


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
    """Records what reached the venue, and answers ``fetch_positions`` consistently with it."""

    def __init__(self, *, entry: Decimal = ENTRY, reject_orders: bool = False) -> None:
        self.stops: list[Decimal] = []
        self.orders: list[Any] = []
        self.calls: list[str] = []
        self._entry = entry
        self._quantity = QTY
        self._stop = STOP_LONG
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


async def run(
    gateway: RecordingGateway,
    prices: list[str],
    *,
    state: PositionState | None = None,
) -> IntrabarManager:
    manager = IntrabarManager(gateway, FakeStream(prices), config(), clock=lambda: NOW)
    manager.track(state or long_state())
    manager.set_atr(BTC, ATR)
    await manager.start([BTC])
    await asyncio.sleep(0.05)
    await manager.stop()
    return manager


class TestALoserExitReachesTheVenue:
    async def test_a_hard_stop_price_on_the_ticker_closes_the_position(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["98.00"])

        assert len(gateway.orders) == 1

    async def test_the_close_is_reduce_only_and_the_full_size(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["98.00"])

        assert gateway.orders[0].reduce_only is True
        assert gateway.orders[0].quantity == QTY

    async def test_an_accelerating_loss_closes_above_the_stop(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["98.80"])

        assert len(gateway.orders) == 1

    async def test_a_burst_of_losing_ticks_produces_one_close(self) -> None:
        """Ticks arrive milliseconds apart; one loser is not six close orders."""
        gateway = RecordingGateway()

        await run(gateway, ["98.00", "97.90", "97.80", "97.70", "97.60"])

        assert len(gateway.orders) == 1

    async def test_noise_never_reaches_the_venue_at_all(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["99.97", "99.92", "99.98", "99.95"])

        assert not gateway.orders
        assert not gateway.stops

    async def test_the_thesis_rule_stays_inactive_without_a_supplied_level(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["99.40", "99.45"])

        assert not gateway.orders

    async def test_a_supplied_invalidation_level_closes_the_position(self) -> None:
        gateway = RecordingGateway()

        manager = IntrabarManager(gateway, FakeStream(["99.40"]), config(), clock=lambda: NOW)
        manager.track(long_state())
        manager.set_invalidation(BTC, Decimal("99.50"))
        manager.set_atr(BTC, ATR)
        await manager.start([BTC])
        await asyncio.sleep(0.05)
        await manager.stop()

        assert len(gateway.orders) == 1


class TestTheExchangeStopStaysInForce:
    async def test_a_loser_exit_never_amends_the_venue_stop(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["98.80"])

        assert not gateway.stops
        assert gateway.venue_stop == STOP_LONG

    async def test_a_rejected_close_leaves_the_stop_exactly_where_it_was(self) -> None:
        """The one case that matters: our exit failed, so the venue's must still be there."""
        gateway = RecordingGateway(reject_orders=True)

        await run(gateway, ["98.80", "98.70"])

        assert gateway.venue_stop == STOP_LONG

    async def test_a_rejected_close_leaves_the_position_managed(self) -> None:
        gateway = RecordingGateway(reject_orders=True)

        manager = await run(gateway, ["98.80"])
        state = manager.state_for(BTC)

        assert state is not None
        assert state.quantity == QTY

    async def test_the_venue_is_read_before_any_close_is_sent(self) -> None:
        gateway = RecordingGateway()

        await run(gateway, ["98.00"])

        assert gateway.calls.index("fetch_positions") < gateway.calls.index("submit_order")


class TestReconciliationStaysAuthoritative:
    async def test_the_venue_entry_price_replaces_a_stale_local_one(self) -> None:
        gateway = RecordingGateway(entry=Decimal("99"))

        manager = await run(gateway, ["99.05"])
        state = manager.state_for(BTC)

        assert state is not None
        assert state.entry_price == Decimal("99")

    async def test_a_loss_rule_is_judged_against_the_venue_position(self) -> None:
        """98.80 is a deep loss against an entry of 100 — and a *winner* against 98.50.

        The venue says the position was opened at 98.50, so that is the position that
        exists. Judging it against the entry price this process happened to remember is
        how a profitable trade gets closed as an emergency.
        """
        gateway = RecordingGateway(entry=Decimal("98.50"))

        await run(gateway, ["98.80"])

        assert len(gateway.orders) == 1
        assert "net profit exit" in gateway.orders[0].metadata["reason"]

    async def test_a_closed_position_is_untracked_rather_than_re_managed(self) -> None:
        gateway = RecordingGateway()

        manager = await run(gateway, ["98.00", "97.90", "97.80"])

        assert manager.monitored == ()

    async def test_state_survives_a_reconnect_gap(self) -> None:
        """The stream ends and resubscribes; management resumes from the venue's book."""
        gateway = RecordingGateway()

        manager = await run(gateway, ["99.90"])
        state = manager.state_for(BTC)

        assert state is not None
        assert PositionState.from_dict(state.to_dict()) == state
