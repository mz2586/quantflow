"""The third loop: keeping the manager's book equal to the venue's.

`test_intrabar_management.py` proves the decision is right and
`test_intrabar_runner_integration.py` proves a tick reaches the venue. Neither of them can
catch the failure these tests exist for, because both assume the position the manager is
managing is the position the venue is holding.

Live, it was not. The manager adopted the venue's positions once at startup and never
looked again, so:

* a position that closed at the venue stayed tracked, and every subsequent tick tried to
  amend a stop on a flat symbol — ``retCode 10001 "can not set tp/sl/ts for zero position"``
  nineteen thousand times over;
* a **new** position on the same symbol, at a different entry and a different size, was
  measured against the *old* entry, so its profit stages were computed against a trade that
  no longer existed.

The regression test below is that exact XRP sequence, with the real numbers off the venue:
an old position at 0.9993 closes, a new one appears at 1.0035, and the price is 1.0072 —
+0.37%, past the +0.25% first rung. The old code moved no stop at all. The new code must
adopt the new position, recognise the rung on the first tick, and amend the stop of the
position that actually exists.

The fakes are deliberately mutable: `FakeGateway.book` is reassigned mid-test to represent
the venue changing underneath a manager that is not looking, which is the whole subject.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.live import intrabar_manager
from quantflow.live.intrabar_manager import (
    DEFAULT_RECONCILE_SECONDS,
    STOP_RETRY_COOLDOWN_SECONDS,
    IntrabarManager,
    improves_materially,
    parse_venue_positions,
    reconcile_seconds_from_env,
)
from quantflow.position.intrabar import IntrabarConfig, PositionState

XRP = Symbol.parse("XRP/USDT")
BTC = Symbol.parse("BTC/USDT")
ETH = Symbol.parse("ETH/USDT")
NOW = datetime(2026, 8, 13, tzinfo=UTC)

#: The live numbers. Kept as named constants so a test that accidentally asserts against
#: the *old* position is obvious on sight rather than a matter of counting decimal places.
OLD_ENTRY = Decimal("0.9993")
OLD_QUANTITY = Decimal("400")
OLD_STOP = Decimal("0.9950")
NEW_ENTRY = Decimal("1.0035")
NEW_QUANTITY = Decimal("498.2")
NEW_STOP = Decimal("0.9993")
NEW_TARGET = Decimal("1.011")
MARK = Decimal("1.0072")

#: Fee rate the default config uses, so breakeven+fees can be asserted exactly.
FEE = Decimal("0.0012")


def venue_row(
    symbol: str,
    *,
    side: str = "long",
    contracts: str,
    entry: str,
    stop: str | None,
    target: str | None = None,
) -> dict[str, Any]:
    """One row shaped like CCXT's unified position payload, suffix and all."""
    return {
        "symbol": f"{symbol}:USDT",
        "side": side,
        "contracts": contracts,
        "entryPrice": entry,
        "info": {"stopLoss": stop or "0", "takeProfit": target or "0"},
    }


def old_xrp() -> dict[str, Any]:
    return venue_row(
        "XRP/USDT", contracts=str(OLD_QUANTITY), entry=str(OLD_ENTRY), stop=str(OLD_STOP)
    )


def new_xrp() -> dict[str, Any]:
    return venue_row(
        "XRP/USDT",
        contracts=str(NEW_QUANTITY),
        entry=str(NEW_ENTRY),
        stop=str(NEW_STOP),
        target=str(NEW_TARGET),
    )


class FakeTicker:
    def __init__(self, price: Decimal) -> None:
        self.last = price
        self.timestamp = NOW


class ScriptedStream:
    """Yields the prices queued for a symbol, then blocks rather than reconnecting."""

    def __init__(self, prices: dict[Symbol, list[Decimal]] | None = None) -> None:
        self.prices = prices or {}

    async def watch_ticker(self, symbol: Symbol) -> Any:
        for price in self.prices.get(symbol, []):
            yield FakeTicker(price)
            await asyncio.sleep(0)
        # Park instead of returning: a stream that returns is a reconnect, and a reconnect
        # would keep re-arming reconciliation and make call counts unassertable.
        await asyncio.sleep(3600)
        yield FakeTicker(Decimal("1"))  # pragma: no cover - unreachable, keeps this a generator


class FakeGateway:
    """A venue whose book can be swapped out from under the manager."""

    def __init__(
        self, book: list[dict[str, Any]] | None = None, *, accept_stop: bool = True
    ) -> None:
        self.book: list[dict[str, Any]] = list(book or [])
        self.stops: list[tuple[Symbol, Decimal]] = []
        self.orders: list[Any] = []
        self.fetches = 0
        self._accept = accept_stop

    async def fetch_positions(self) -> list[dict[str, Any]]:
        self.fetches += 1
        return [dict(row) for row in self.book]

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        if not self._accept:
            return None
        assert stop_loss is not None
        self.stops.append((symbol, stop_loss))
        # Mirror the real gateway: the venue now holds the new stop, so the next
        # reconciliation reads it back.
        for row in self.book:
            if row["symbol"].startswith(str(symbol).replace("/", "/")):
                row["info"] = {**row["info"], "stopLoss": str(stop_loss)}
        return stop_loss

    async def submit_order(self, request: Any) -> Any:
        self.orders.append(request)
        return request


def manager(
    gateway: FakeGateway,
    *,
    prices: dict[Symbol, list[Decimal]] | None = None,
    enabled: bool = True,
    reconcile_seconds: float = DEFAULT_RECONCILE_SECONDS,
) -> IntrabarManager:
    return IntrabarManager(
        gateway,
        ScriptedStream(prices),
        IntrabarConfig(enabled=enabled),
        reconcile_seconds=reconcile_seconds,
        clock=lambda: NOW,
    )


async def tick(mgr: IntrabarManager, symbol: Symbol, price: Decimal) -> None:
    """Deliver one tick directly, with no stream in the way."""
    await mgr._on_tick(symbol, price, NOW)


def tracked_state(mgr: IntrabarManager, symbol: Symbol) -> PositionState:
    state = mgr.state_for(symbol)
    assert state is not None, f"{symbol} is not tracked"
    return state


# --------------------------------------------------------------------------- #
# The live failure, reproduced
# --------------------------------------------------------------------------- #
class TestTheXrpRegression:
    """Old position closes, new one opens at a different price, price is past stage 1."""

    @staticmethod
    async def replay(gateway: FakeGateway) -> IntrabarManager:
        mgr = manager(gateway)
        await mgr.reconcile()  # startup: the old position is what exists
        assert tracked_state(mgr, XRP).entry_price == OLD_ENTRY

        gateway.book = [new_xrp()]  # the venue moved on while nobody asked
        await mgr.reconcile()
        await tick(mgr, XRP, MARK)
        return mgr

    async def test_the_old_position_is_gone_from_local_state(self) -> None:
        mgr = await self.replay(FakeGateway([old_xrp()]))

        assert tracked_state(mgr, XRP).entry_price != OLD_ENTRY

    async def test_the_new_position_is_adopted_at_the_venue_entry(self) -> None:
        mgr = await self.replay(FakeGateway([old_xrp()]))

        assert tracked_state(mgr, XRP).entry_price == NEW_ENTRY

    async def test_the_new_position_is_adopted_at_the_venue_quantity(self) -> None:
        mgr = await self.replay(FakeGateway([old_xrp()]))

        assert tracked_state(mgr, XRP).quantity == NEW_QUANTITY

    async def test_stage_one_is_recognised_on_the_first_tick_after_adoption(self) -> None:
        """+0.37% is past +0.25%: the rung fires now, not after another crossing."""
        mgr = await self.replay(FakeGateway([old_xrp()]))

        assert 0 in tracked_state(mgr, XRP).stages_done

    async def test_a_stop_amendment_reaches_the_venue(self) -> None:
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert [symbol for symbol, _ in gateway.stops] == [XRP]

    async def test_the_amended_stop_is_computed_from_the_new_entry(self) -> None:
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert gateway.stops[-1][1] == NEW_ENTRY * (Decimal("1") + FEE)

    async def test_the_amended_stop_is_above_the_new_entry(self) -> None:
        """The live symptom: the venue stop sat at 0.9993, below entry, and never moved."""
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert gateway.stops[-1][1] > NEW_ENTRY

    async def test_the_old_stop_is_never_re_sent(self) -> None:
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert OLD_STOP not in [stop for _, stop in gateway.stops]

    async def test_no_stop_is_derived_from_the_old_entry(self) -> None:
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert OLD_ENTRY * (Decimal("1") + FEE) not in [stop for _, stop in gateway.stops]

    async def test_the_old_quantity_is_never_used(self) -> None:
        gateway = FakeGateway([old_xrp()])

        await self.replay(gateway)

        assert OLD_QUANTITY not in [getattr(order, "quantity", None) for order in gateway.orders]


# --------------------------------------------------------------------------- #
# (A) a position disappears
# --------------------------------------------------------------------------- #
class TestPositionDisappears:
    async def test_a_closed_position_is_untracked(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = []
        await mgr.reconcile()

        assert mgr.state_for(XRP) is None

    async def test_no_amendment_is_sent_after_it_closed(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = []
        await mgr.reconcile()

        await tick(mgr, XRP, MARK)

        assert not gateway.stops

    async def test_a_zero_contract_row_counts_as_closed(self) -> None:
        """The venue reports a flat symbol as a row with zero contracts, not by omission."""
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [
            venue_row("XRP/USDT", contracts="0", entry=str(NEW_ENTRY), stop=str(NEW_STOP))
        ]
        await mgr.reconcile()

        assert mgr.monitored == ()

    async def test_a_position_that_vanishes_between_decision_and_action_is_untracked(
        self,
    ) -> None:
        """The pre-action re-read is what turns 20k rejections into one untrack."""
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = []  # filled between the tick arriving and the amendment going out

        await tick(mgr, XRP, MARK)

        assert mgr.state_for(XRP) is None
        assert not gateway.stops


# --------------------------------------------------------------------------- #
# (B) a position appears after startup
# --------------------------------------------------------------------------- #
class TestPositionAppears:
    async def test_a_position_opened_after_startup_is_adopted(self) -> None:
        gateway = FakeGateway([])
        mgr = manager(gateway)
        await mgr.reconcile()
        assert mgr.monitored == ()

        gateway.book = [new_xrp()]
        await mgr.reconcile()

        assert list(mgr.monitored) == [XRP]

    async def test_it_is_adopted_with_the_venue_stop_and_target(self) -> None:
        gateway = FakeGateway([])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = [new_xrp()]

        await mgr.reconcile()

        state = tracked_state(mgr, XRP)
        assert (state.current_stop, state.target) == (NEW_STOP, NEW_TARGET)

    async def test_a_position_without_a_venue_stop_is_not_adopted(self) -> None:
        """Nothing to ratchet, and inventing a stop is a risk decision this layer never made."""
        gateway = FakeGateway([venue_row("XRP/USDT", contracts="100", entry="1.0", stop=None)])
        mgr = manager(gateway)

        await mgr.reconcile()

        assert mgr.monitored == ()


# --------------------------------------------------------------------------- #
# (C) quantity changes / (H) partial fills
# --------------------------------------------------------------------------- #
class TestQuantityChanges:
    async def test_a_reduced_quantity_is_taken_from_the_venue(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [
            venue_row("XRP/USDT", contracts="300", entry=str(NEW_ENTRY), stop=str(NEW_STOP))
        ]
        await mgr.reconcile()

        assert tracked_state(mgr, XRP).quantity == Decimal("300")

    async def test_a_reduced_quantity_keeps_the_original_size_for_partial_sizing(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = [
            venue_row("XRP/USDT", contracts="300", entry=str(NEW_ENTRY), stop=str(NEW_STOP))
        ]

        await mgr.reconcile()

        assert tracked_state(mgr, XRP).original_quantity == NEW_QUANTITY

    async def test_an_increased_quantity_at_the_same_entry_is_taken_from_the_venue(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = [
            venue_row("XRP/USDT", contracts="900", entry=str(NEW_ENTRY), stop=str(NEW_STOP))
        ]

        await mgr.reconcile()

        state = tracked_state(mgr, XRP)
        assert (state.quantity, state.original_quantity) == (Decimal("900"), Decimal("900"))

    async def test_a_close_is_sized_to_the_quantity_the_venue_actually_holds(self) -> None:
        """A partial fill since the decision must not produce an order for size that is gone."""
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()
        # Stage 3 wants 33% of 10 = 3.3, but only 2 contracts are left by the time it fires.
        gateway.book = [venue_row("BTC/USDT", contracts="2", entry="100", stop="99")]

        await tick(mgr, BTC, Decimal("100.80"))

        assert [order.quantity for order in gateway.orders] == [Decimal("2")]

    async def test_a_zero_quantity_order_is_never_sent(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()
        mgr.track(
            PositionState.from_entry(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("10"),
                stop=Decimal("99"),
                opened_at=NOW,
            )
        )
        gateway.book = [venue_row("BTC/USDT", contracts="0", entry="100", stop="99")]

        await tick(mgr, BTC, Decimal("100.80"))

        assert not gateway.orders


# --------------------------------------------------------------------------- #
# (D) entry changes / (M) water marks must not leak
# --------------------------------------------------------------------------- #
class TestEntryChanges:
    @staticmethod
    async def run_up_then_replace(gateway: FakeGateway) -> IntrabarManager:
        """Let a position run favourably, then have the venue replace it at a new entry."""
        mgr = manager(gateway)
        await mgr.reconcile()
        await tick(mgr, BTC, Decimal("100.90"))  # marks a high water and fires rungs
        gateway.book = [venue_row("BTC/USDT", contracts="10", entry="200", stop="198")]
        await mgr.reconcile()
        return mgr

    async def test_the_entry_is_replaced(self) -> None:
        mgr = await self.run_up_then_replace(
            FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        )

        assert tracked_state(mgr, BTC).entry_price == Decimal("200")

    async def test_the_old_high_water_does_not_survive(self) -> None:
        mgr = await self.run_up_then_replace(
            FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        )

        assert tracked_state(mgr, BTC).high_water == Decimal("200")

    async def test_the_old_fired_stages_do_not_survive(self) -> None:
        """A replacement must not inherit profit the new position never earned."""
        mgr = await self.run_up_then_replace(
            FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        )

        assert tracked_state(mgr, BTC).stages_done == frozenset()

    async def test_a_side_flip_replaces_the_position(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [
            venue_row("BTC/USDT", side="short", contracts="10", entry="100", stop="101")
        ]
        await mgr.reconcile()

        assert tracked_state(mgr, BTC).side is PositionSide.SHORT

    async def test_a_replacement_without_a_venue_stop_is_untracked_not_kept(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [venue_row("BTC/USDT", contracts="10", entry="200", stop=None)]
        await mgr.reconcile()

        assert mgr.state_for(BTC) is None


# --------------------------------------------------------------------------- #
# (E) the venue stop changes
# --------------------------------------------------------------------------- #
class TestVenueStopChanges:
    async def test_a_tightened_venue_stop_is_adopted(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [venue_row("BTC/USDT", contracts="10", entry="100", stop="99.5")]
        await mgr.reconcile()

        assert tracked_state(mgr, BTC).current_stop == Decimal("99.5")

    async def test_a_loosened_venue_stop_is_adopted_too(self) -> None:
        """Believing in a stop the exchange is not holding is the failure, not the fix."""
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [venue_row("BTC/USDT", contracts="10", entry="100", stop="98")]
        await mgr.reconcile()

        assert tracked_state(mgr, BTC).current_stop == Decimal("98")

    async def test_a_changed_venue_target_is_adopted(self) -> None:
        gateway = FakeGateway(
            [venue_row("BTC/USDT", contracts="10", entry="100", stop="99", target="105")]
        )
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [venue_row("BTC/USDT", contracts="10", entry="100", stop="99", target="110")]
        await mgr.reconcile()

        assert tracked_state(mgr, BTC).target == Decimal("110")


# --------------------------------------------------------------------------- #
# (F) stop / target fills
# --------------------------------------------------------------------------- #
class TestProtectiveFills:
    async def test_a_stop_fill_untracks_the_symbol(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = []  # the venue stop filled
        await mgr.reconcile()

        assert mgr.monitored == ()

    async def test_a_target_fill_untracks_the_symbol(self) -> None:
        gateway = FakeGateway(
            [venue_row("BTC/USDT", contracts="10", entry="100", stop="99", target="105")]
        )
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = []  # the venue target filled
        await mgr.reconcile()

        assert mgr.monitored == ()

    async def test_after_a_fill_ticks_produce_nothing_at_all(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book = []
        await mgr.reconcile()

        for price in ("100.30", "100.60", "100.90", "98.00"):
            await tick(mgr, BTC, Decimal(price))

        assert not gateway.stops
        assert not gateway.orders


# --------------------------------------------------------------------------- #
# (G) reconnect
# --------------------------------------------------------------------------- #
class TestReconnect:
    async def test_a_reconnect_restores_the_exact_venue_state(self) -> None:
        """Whatever happened during the gap, the state after it is the venue's."""
        gateway = FakeGateway([old_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [new_xrp()]  # everything below happened while the stream was down
        await mgr.reconcile()

        state = tracked_state(mgr, XRP)
        assert (state.entry_price, state.quantity, state.current_stop, state.target) == (
            NEW_ENTRY,
            NEW_QUANTITY,
            NEW_STOP,
            NEW_TARGET,
        )

    async def test_a_stream_error_asks_for_a_reconciliation(self) -> None:
        class BrokenStream:
            async def watch_ticker(self, symbol: Symbol) -> Any:
                raise ConnectionResetError("stream dropped")
                yield  # pragma: no cover - keeps this a generator

        gateway = FakeGateway([new_xrp()])
        mgr = IntrabarManager(
            gateway,
            BrokenStream(),
            IntrabarConfig(enabled=True),
            reconcile_seconds=0.1,
            clock=lambda: NOW,
        )
        await mgr.start([XRP])
        gateway.book = []
        await asyncio.sleep(0.4)
        await mgr.stop()

        assert mgr.monitored == ()


# --------------------------------------------------------------------------- #
# (I) a stage already reached when the position is adopted
# --------------------------------------------------------------------------- #
class TestStageAlreadyReachedOnAdoption:
    async def test_the_rung_fires_on_the_very_first_tick(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        await tick(mgr, XRP, MARK)

        assert gateway.stops

    async def test_adoption_alone_invents_no_favourable_excursion(self) -> None:
        """Water marks start at entry: the run-up before adoption cannot be reconstructed."""
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)

        await mgr.reconcile()

        state = tracked_state(mgr, XRP)
        assert (state.high_water, state.low_water) == (NEW_ENTRY, NEW_ENTRY)

    async def test_adoption_alone_fires_no_stage(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway)

        await mgr.reconcile()

        assert tracked_state(mgr, XRP).stages_done == frozenset()

    async def test_three_rungs_already_cleared_all_fire_at_once(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        await tick(mgr, BTC, Decimal("100.80"))  # +0.80%: past all three rungs

        assert tracked_state(mgr, BTC).stages_done == frozenset({0, 1, 2})


# --------------------------------------------------------------------------- #
# (J) several symbols
# --------------------------------------------------------------------------- #
class TestMultipleSymbols:
    async def test_each_symbol_reconciles_independently(self) -> None:
        gateway = FakeGateway(
            [
                new_xrp(),
                venue_row("BTC/USDT", contracts="10", entry="100", stop="99"),
                venue_row("ETH/USDT", contracts="5", entry="2000", stop="1980"),
            ]
        )
        mgr = manager(gateway)
        await mgr.reconcile()

        gateway.book = [
            new_xrp(),  # unchanged
            venue_row("BTC/USDT", contracts="10", entry="105", stop="104"),  # replaced
        ]  # ETH closed
        await mgr.reconcile()

        assert (
            tracked_state(mgr, XRP).entry_price,
            tracked_state(mgr, BTC).entry_price,
            mgr.state_for(ETH),
        ) == (NEW_ENTRY, Decimal("105"), None)

    async def test_the_whole_book_costs_one_request(self) -> None:
        """Per-symbol polling would multiply the rate-limit cost for no extra information."""
        gateway = FakeGateway(
            [
                new_xrp(),
                venue_row("BTC/USDT", contracts="10", entry="100", stop="99"),
                venue_row("ETH/USDT", contracts="5", entry="2000", stop="1980"),
            ]
        )
        mgr = manager(gateway)

        await mgr.reconcile()

        assert gateway.fetches == 1


# --------------------------------------------------------------------------- #
# (K) / (L) no duplicates
# --------------------------------------------------------------------------- #
class TestNoDuplicates:
    async def test_a_burst_of_ticks_moves_the_stop_once(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        for _ in range(5):
            await tick(mgr, BTC, Decimal("100.30"))

        assert len(gateway.stops) == 1

    async def test_reconciling_between_ticks_does_not_re_fire_a_rung(self) -> None:
        """The venue reading its own amended stop back must not look like a fresh reason."""
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        for _ in range(5):
            await tick(mgr, BTC, Decimal("100.30"))
            await mgr.reconcile()

        assert len(gateway.stops) == 1

    async def test_a_burst_of_ticks_produces_one_close(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()

        for price in ("100.30", "100.60", "100.05", "100.04", "100.03"):
            await tick(mgr, BTC, Decimal(price))

        assert len(gateway.orders) == 1

    async def test_reconciling_during_an_in_flight_close_does_not_re_adopt(self) -> None:
        """The venue still shows the position; re-adopting it would close it twice."""
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()
        await tick(mgr, BTC, Decimal("100.30"))
        await tick(mgr, BTC, Decimal("100.05"))  # full close submitted, venue not updated yet

        await mgr.reconcile()
        await tick(mgr, BTC, Decimal("100.04"))

        assert len(gateway.orders) == 1

    async def test_the_symbol_is_managed_again_once_the_close_lands(self) -> None:
        gateway = FakeGateway([venue_row("BTC/USDT", contracts="10", entry="100", stop="99")])
        mgr = manager(gateway)
        await mgr.reconcile()
        await tick(mgr, BTC, Decimal("100.30"))
        await tick(mgr, BTC, Decimal("100.05"))

        gateway.book = [venue_row("BTC/USDT", contracts="4", entry="120", stop="118")]
        await mgr.reconcile()

        assert tracked_state(mgr, BTC).entry_price == Decimal("120")


# --------------------------------------------------------------------------- #
# Cadence and configuration
# --------------------------------------------------------------------------- #
class TestCadence:
    async def test_the_loop_reconciles_without_being_asked(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway, reconcile_seconds=0.1)
        await mgr.start([XRP])

        gateway.book = []
        await asyncio.sleep(0.4)
        monitored = mgr.monitored
        await mgr.stop()

        assert monitored == ()

    async def test_startup_reconciles_before_the_first_tick(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway, prices={XRP: [MARK]}, reconcile_seconds=30)

        await mgr.start([XRP])
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert gateway.stops

    async def test_a_disabled_manager_never_reads_the_venue(self) -> None:
        gateway = FakeGateway([new_xrp()])
        mgr = manager(gateway, enabled=False)

        await mgr.start([XRP])
        await mgr.stop()

        assert gateway.fetches == 0

    async def test_a_fetch_failure_leaves_the_last_good_state_alone(self) -> None:
        class FlakyGateway(FakeGateway):
            async def fetch_positions(self) -> list[dict[str, Any]]:
                if self.fetches:
                    self.fetches += 1
                    raise ConnectionError("venue unreachable")
                return await super().fetch_positions()

        gateway = FlakyGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        await mgr.reconcile()

        assert tracked_state(mgr, XRP).entry_price == NEW_ENTRY

    def test_the_cadence_defaults_to_two_seconds(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("QF_INTRABAR_RECONCILE_SECONDS", raising=False)

        assert reconcile_seconds_from_env() == DEFAULT_RECONCILE_SECONDS

    def test_the_cadence_is_configurable(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("QF_INTRABAR_RECONCILE_SECONDS", "0.5")

        assert reconcile_seconds_from_env() == pytest.approx(0.5)

    def test_a_zero_cadence_is_clamped_rather_than_becoming_a_request_storm(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setenv("QF_INTRABAR_RECONCILE_SECONDS", "0")

        assert reconcile_seconds_from_env() > 0

    def test_an_unparseable_cadence_falls_back_to_the_default(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("QF_INTRABAR_RECONCILE_SECONDS", "soon")

        assert reconcile_seconds_from_env() == DEFAULT_RECONCILE_SECONDS


# --------------------------------------------------------------------------- #
# The venue snaps the stop to its own tick
# --------------------------------------------------------------------------- #
class SnappingGateway(FakeGateway):
    """Bybit's actual behaviour, which reconciliation turned into a request storm live.

    Three things happen together on a real amendment and all three matter:

    * the venue **snaps** the requested price to its tick, so the stop it ends up holding
      is never quite the stop that was asked for;
    * the read-back immediately afterwards can still show the *old* level;
    * asking again for a level it already holds is refused with ``retCode 34040``.

    With reconciliation pulling the snapped value back into local state, an unsnapped
    request one ten-thousandth above it looks like a fresh improvement on every single
    tick. That is how one profit rung produced 504 rejected requests in five minutes.
    """

    TICK = Decimal("0.0001")

    def __init__(self, book: list[dict[str, Any]] | None = None) -> None:
        super().__init__(book)
        self.attempts = 0
        self.read_back_lagging = True

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        assert stop_loss is not None
        self.attempts += 1
        snapped = stop_loss.quantize(self.TICK, rounding=ROUND_DOWN)
        for row in self.book:
            if not row["symbol"].startswith(f"{symbol}"):
                continue
            if Decimal(row["info"]["stopLoss"]) == snapped:
                raise RuntimeError('bybit {"retCode":34040,"retMsg":"not modified","result":{}}')
            row["info"] = {**row["info"], "stopLoss": str(snapped)}
        self.stops.append((symbol, snapped))
        if self.read_back_lagging:
            # The venue took it, but its position endpoint has not caught up.
            return None
        return snapped


class TestTheVenueSnapsTheStop:
    @staticmethod
    async def drive(gateway: SnappingGateway, ticks: int = 6) -> IntrabarManager:
        mgr = manager(gateway)
        await mgr.reconcile()
        for _ in range(ticks):
            await tick(mgr, XRP, MARK)
            await mgr.reconcile()
        return mgr

    async def test_the_rung_produces_exactly_one_amendment(self) -> None:
        gateway = SnappingGateway([new_xrp()])

        await self.drive(gateway)

        assert gateway.attempts == 1

    async def test_the_venue_snapped_stop_is_what_gets_stored(self) -> None:
        """Storing the unsnapped request is what makes the next tick see work to do."""
        gateway = SnappingGateway([new_xrp()])

        mgr = await self.drive(gateway)

        assert tracked_state(mgr, XRP).current_stop == Decimal("1.0047")

    async def test_the_rung_is_recorded_despite_the_lagging_read_back(self) -> None:
        gateway = SnappingGateway([new_xrp()])

        mgr = await self.drive(gateway)

        assert 0 in tracked_state(mgr, XRP).stages_done

    async def test_the_stop_still_ends_up_above_entry(self) -> None:
        gateway = SnappingGateway([new_xrp()])

        await self.drive(gateway)

        assert gateway.stops[-1][1] > NEW_ENTRY

    @staticmethod
    async def venue_ahead_of_local(gateway: SnappingGateway) -> IntrabarManager:
        """Local state still on the old stop while the venue already holds the new one.

        The real race: the amendment landed, the process has not reconciled yet, and the
        next tick asks for a level the venue is already holding. The sub-tick guard cannot
        catch this one — local and venue disagree by a lot — so the ``34040`` reply is the
        only thing standing between one rung and an unbounded retry loop.
        """
        mgr = manager(gateway)
        await mgr.reconcile()
        gateway.book[0]["info"] = {**gateway.book[0]["info"], "stopLoss": "1.0047"}
        for _ in range(4):
            await tick(mgr, XRP, MARK)
        return mgr

    async def test_not_modified_is_a_confirmation_not_a_failure(self) -> None:
        """The venue already holds the level: there is nothing left to do, and no error."""
        gateway = SnappingGateway([new_xrp()])

        mgr = await self.venue_ahead_of_local(gateway)

        assert tracked_state(mgr, XRP).stages_done == frozenset({0})

    async def test_not_modified_costs_exactly_one_attempt(self) -> None:
        gateway = SnappingGateway([new_xrp()])

        await self.venue_ahead_of_local(gateway)

        assert gateway.attempts == 1

    async def test_the_venue_level_is_taken_as_the_local_stop(self) -> None:
        gateway = SnappingGateway([new_xrp()])

        mgr = await self.venue_ahead_of_local(gateway)

        assert tracked_state(mgr, XRP).current_stop == Decimal("1.0047")


class TestSubTickTrailChurn:
    """The trail recomputes every tick; the venue rounds every result to the same price.

    Live, this produced two amendment requests a second against a stop that had not moved:
    reconciliation stored the venue's snapped 0.13637, the next tick's trail asked for
    0.136371810, the ratchet called that an improvement, and the venue called it "not
    modified". Neither component was wrong on its own.
    """

    async def test_a_sub_tick_improvement_sends_no_request(self) -> None:
        gateway = SnappingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        state = tracked_state(mgr, XRP)
        # The venue already holds a stop a whisker below what the rung will ask for.
        mgr.track(replace(state, current_stop=Decimal("1.0047")))

        for _ in range(30):
            await tick(mgr, XRP, MARK)

        assert gateway.attempts == 0

    async def test_the_rung_is_still_recorded_when_the_request_is_skipped(self) -> None:
        """Skipping the request must not leave the rung armed to try again forever."""
        gateway = SnappingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        mgr.track(replace(tracked_state(mgr, XRP), current_stop=Decimal("1.0047")))

        await tick(mgr, XRP, MARK)

        assert 0 in tracked_state(mgr, XRP).stages_done

    async def test_the_stop_stays_at_the_venue_value(self) -> None:
        gateway = SnappingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        mgr.track(replace(tracked_state(mgr, XRP), current_stop=Decimal("1.0047")))

        await tick(mgr, XRP, MARK)

        assert tracked_state(mgr, XRP).current_stop == Decimal("1.0047")

    async def test_a_real_improvement_is_still_sent(self) -> None:
        """The floor is on the request, not on the protection."""
        gateway = SnappingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        await tick(mgr, XRP, MARK)

        assert gateway.attempts == 1

    def test_a_one_basis_point_improvement_clears_the_floor(self) -> None:
        assert improves_materially(PositionSide.LONG, Decimal("100"), Decimal("100.01"))

    def test_a_sub_basis_point_improvement_does_not(self) -> None:
        assert not improves_materially(PositionSide.LONG, Decimal("100"), Decimal("100.0001"))

    def test_a_loosening_never_counts_as_an_improvement(self) -> None:
        assert not improves_materially(PositionSide.LONG, Decimal("100"), Decimal("90"))

    def test_a_short_improves_by_moving_down(self) -> None:
        assert improves_materially(PositionSide.SHORT, Decimal("100"), Decimal("99.9"))

    def test_a_short_loosening_upward_is_rejected(self) -> None:
        assert not improves_materially(PositionSide.SHORT, Decimal("100"), Decimal("100.5"))


class TestFailedAmendmentsAreThrottled:
    class RefusingGateway(FakeGateway):
        def __init__(self, book: list[dict[str, Any]] | None = None) -> None:
            super().__init__(book)
            self.attempts = 0

        async def set_trading_stop(
            self, symbol: Symbol, *, stop_loss: Decimal | None = None
        ) -> Decimal | None:
            self.attempts += 1
            raise RuntimeError('bybit {"retCode":10001,"retMsg":"something is wrong"}')

    async def test_a_rejected_amendment_is_not_retried_on_every_tick(self) -> None:
        """One rejection must not become one rejected request per tick."""
        gateway = self.RefusingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        for _ in range(20):
            await tick(mgr, XRP, MARK)

        assert gateway.attempts == 1

    async def test_the_rung_is_not_recorded_when_the_venue_refused(self) -> None:
        """Nothing may advance on a stop the exchange did not accept."""
        gateway = self.RefusingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()

        await tick(mgr, XRP, MARK)

        assert tracked_state(mgr, XRP).stages_done == frozenset()

    async def test_it_is_retried_once_the_cooldown_expires(self, monkeypatch: Any) -> None:
        gateway = self.RefusingGateway([new_xrp()])
        mgr = manager(gateway)
        await mgr.reconcile()
        await tick(mgr, XRP, MARK)

        clock = time.monotonic() + STOP_RETRY_COOLDOWN_SECONDS + 1
        monkeypatch.setattr(intrabar_manager, "time", SimpleNamespace(monotonic=lambda: clock))
        await tick(mgr, XRP, MARK)

        assert gateway.attempts == 2


# --------------------------------------------------------------------------- #
# Parsing the venue payload
# --------------------------------------------------------------------------- #
class TestParsingTheVenueBook:
    def test_the_ccxt_settlement_suffix_is_stripped(self) -> None:
        book = parse_venue_positions([new_xrp()])

        assert list(book) == [XRP]

    def test_a_zero_stop_reads_as_no_stop(self) -> None:
        """Bybit spells "no stop" as the string "0", which is not a price."""
        book = parse_venue_positions(
            [venue_row("XRP/USDT", contracts="100", entry="1.0", stop="0")]
        )

        assert book[XRP].stop is None

    def test_a_short_position_is_recognised(self) -> None:
        book = parse_venue_positions(
            [venue_row("XRP/USDT", side="short", contracts="100", entry="1.0", stop="1.1")]
        )

        assert book[XRP].side is PositionSide.SHORT

    def test_a_short_position_quantity_is_absolute(self) -> None:
        book = parse_venue_positions(
            [venue_row("XRP/USDT", side="short", contracts="-100", entry="1.0", stop="1.1")]
        )

        assert book[XRP].quantity == Decimal("100")

    def test_an_unparseable_row_is_skipped_not_fatal(self) -> None:
        book = parse_venue_positions(
            [
                {"symbol": "???", "side": "long", "contracts": "1", "entryPrice": "1", "info": {}},
                new_xrp(),
            ]
        )

        assert list(book) == [XRP]


class TestProtectionIsClassifiedByOrderType:
    """A ratcheted stop must never be mistaken for a take-profit."""

    @staticmethod
    def _order(symbol: str, trigger: str, price: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            symbol=symbol,
            reduce_only=True,
            trigger_price=Decimal(trigger),
            price=Decimal(price) if price is not None else None,
        )

    def test_a_stop_ratcheted_above_entry_is_still_a_stop(self) -> None:
        # The live regression, 2026-08-17: a BTC long entered at 64,301.40 had its trail
        # ratchet the stop to 64,378.50 — above entry, because it was locking in profit.
        # Classifying by position relative to entry read that as the take-profit, and the
        # manager closed the winner two seconds later at a fifth of its real target.
        from quantflow.live.intrabar_manager import resolve_protection

        stop, target = resolve_protection(
            Symbol.parse("BTC/USDT"),
            [
                self._order("BTC/USDT", "64378.5", None),  # ratcheted stop, market
                self._order("BTC/USDT", "65274.2", "65274.2"),  # real target, limit
            ],
        )
        assert stop == Decimal("64378.5")
        assert target == Decimal("65274.2")

    def test_a_normal_stop_below_entry_is_a_stop(self) -> None:
        from quantflow.live.intrabar_manager import resolve_protection

        stop, target = resolve_protection(
            Symbol.parse("ETH/USDT"),
            [
                self._order("ETH/USDT", "1892.40", None),
                self._order("ETH/USDT", "1938.21", "1938.21"),
            ],
        )
        assert stop == Decimal("1892.40")
        assert target == Decimal("1938.21")

    def test_orders_for_other_symbols_are_ignored(self) -> None:
        from quantflow.live.intrabar_manager import resolve_protection

        stop, target = resolve_protection(
            Symbol.parse("BTC/USDT"), [self._order("ETH/USDT", "1892.40", None)]
        )
        assert stop is None
        assert target is None
