"""One position, three thresholds, ticks only — the exact sequence the operator asked for.

15m is the *entry* timeframe. Once a position exists, nothing about its exit may wait for
a bar to close. This test drives a single position from a known entry price through all
three profit stages using nothing but ticker prices, and asserts that each crossing
produces its action at the venue immediately.

There is deliberately no candle in this file. Not a mocked one, not a stub — the word does
not appear in the execution path under test. That is the property being proven: the exit
path has no dependency on the bar loop at all.

The last assertion is the one that protects against over-eager exits: a position that is
merely green by a hair must NOT be closed. Stage 1 is a threshold, not a hair-trigger.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.live.intrabar_manager import IntrabarManager
from quantflow.position.intrabar import (
    ActionKind,
    IntrabarConfig,
    PositionState,
    ProfitStage,
    StageAction,
    on_price,
)

BTC = Symbol.parse("BTC/USDT")
NOW = datetime(2026, 1, 1, tzinfo=UTC)

#: A round entry so every threshold lands on an obvious price.
ENTRY = Decimal("100")
QTY = Decimal("10")

#: The exact prices that cross each configured stage.
STAGE_1_PRICE = Decimal("100.25")  # +0.25%
STAGE_2_PRICE = Decimal("100.50")  # +0.50%
STAGE_3_PRICE = Decimal("100.75")  # +0.75%
NOISE_PRICE = Decimal("100.01")  # +0.01% — must do nothing


def config() -> IntrabarConfig:
    return IntrabarConfig(
        stages=(
            ProfitStage(trigger_pct=Decimal("0.0025"), action=StageAction.BREAKEVEN),
            ProfitStage(
                trigger_pct=Decimal("0.0050"),
                action=StageAction.LOCK_PROFIT,
                lock_pct=Decimal("0.0020"),
            ),
            ProfitStage(
                trigger_pct=Decimal("0.0075"),
                action=StageAction.PARTIAL_EXIT,
                partial_fraction=Decimal("0.33"),
            ),
        ),
        enabled=True,
    )


def opened_position() -> PositionState:
    """A position opened at a known price, as if by a completed 15m candle."""
    return PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=ENTRY,
        quantity=QTY,
        stop=Decimal("99"),
        opened_at=NOW,
        target=Decimal("110"),
    )


def send(state: PositionState, price: Decimal) -> tuple[PositionState, Any]:
    """Deliver one live tick. No candle is involved."""
    return on_price(state, price, atr=Decimal("0.4"), config=config(), now=NOW)


class TestTheFullStageSequence:
    def test_noise_profit_does_nothing(self) -> None:
        """+0.01% is not a reason to touch a position."""
        _, action = send(opened_position(), NOISE_PRICE)

        assert action.kind is ActionKind.NONE

    def test_stage_one_amends_the_stop_immediately(self) -> None:
        state, action = send(opened_position(), STAGE_1_PRICE)

        assert action.kind is ActionKind.MOVE_STOP
        assert state.current_stop >= ENTRY

    def test_stage_two_locks_profit(self) -> None:
        state, _ = send(opened_position(), STAGE_1_PRICE)
        state, action = send(state, STAGE_2_PRICE)

        assert action.kind is ActionKind.MOVE_STOP
        assert state.current_stop > ENTRY

    def test_stage_three_takes_partial_profit(self) -> None:
        state, _ = send(opened_position(), STAGE_1_PRICE)
        state, _ = send(state, STAGE_2_PRICE)
        state, action = send(state, STAGE_3_PRICE)

        assert action.kind is ActionKind.PARTIAL_CLOSE
        assert action.close_quantity is not None
        assert action.close_quantity > 0

    def test_the_stop_only_ever_improves_across_the_sequence(self) -> None:
        state = opened_position()
        stops = [state.current_stop]
        for price in (NOISE_PRICE, STAGE_1_PRICE, STAGE_2_PRICE, STAGE_3_PRICE):
            state, _ = send(state, price)
            stops.append(state.current_stop)

        assert stops == sorted(stops)

    def test_the_remainder_keeps_running_after_the_partial(self) -> None:
        """Stage 3 takes some off; it does not flatten a winner."""
        state = opened_position()
        for price in (STAGE_1_PRICE, STAGE_2_PRICE, STAGE_3_PRICE):
            state, _ = send(state, price)

        assert 0 < state.quantity < QTY


class FakeTicker:
    def __init__(self, price: Decimal) -> None:
        self.last = price
        self.timestamp = NOW


class FakeStream:
    def __init__(self, prices: list[Decimal]) -> None:
        self._prices = prices

    async def watch_ticker(self, symbol: Symbol) -> Any:
        for price in self._prices:
            yield FakeTicker(price)
            await asyncio.sleep(0)


class RecordingGateway:
    """Records what reached the venue — and then answers *as* that venue.

    The position endpoint has to move with the amendments and the fills. The manager
    re-reads the book immediately before it acts and reconciles against it continuously, so
    a fake whose ``fetch_positions`` contradicts its own ``set_trading_stop`` is not a
    stricter test, it is an inconsistent exchange.
    """

    def __init__(self) -> None:
        self.stops: list[Decimal] = []
        self.orders: list[Any] = []
        self._quantity = QTY
        self._stop = Decimal("99")

    async def fetch_positions(self) -> list[dict[str, Any]]:
        if self._quantity <= 0:
            return []
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": str(self._quantity),
                "entryPrice": str(ENTRY),
                "info": {"stopLoss": str(self._stop), "takeProfit": "110"},
            }
        ]

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        assert stop_loss is not None
        self.stops.append(stop_loss)
        self._stop = stop_loss
        return stop_loss

    async def submit_order(self, request: Any) -> Any:
        self.orders.append(request)
        self._quantity = max(Decimal("0"), self._quantity - request.quantity)
        return request


class TestTheSequenceReachesTheVenue:
    """The same three crossings, this time through the live manager and the gateway."""

    async def _run(self) -> RecordingGateway:
        gateway = RecordingGateway()
        manager = IntrabarManager(
            gateway,
            FakeStream([NOISE_PRICE, STAGE_1_PRICE, STAGE_2_PRICE, STAGE_3_PRICE]),
            config(),
        )
        manager.track(opened_position())
        manager.set_atr(BTC, Decimal("0.4"))
        await manager.start([BTC])
        await asyncio.sleep(0.05)
        await manager.stop()
        return gateway

    async def test_two_stop_amendments_reach_the_venue(self) -> None:
        """Stage 1 and stage 2 each send a real amendment."""
        gateway = await self._run()

        assert len(gateway.stops) >= 2

    async def test_the_stop_amendments_improve_in_order(self) -> None:
        gateway = await self._run()

        assert gateway.stops == sorted(gateway.stops)

    async def test_a_reduce_only_partial_close_reaches_the_venue(self) -> None:
        gateway = await self._run()

        assert any(getattr(o, "reduce_only", False) for o in gateway.orders)

    async def test_the_partial_is_not_a_full_flatten(self) -> None:
        gateway = await self._run()
        partial = next(o for o in gateway.orders if getattr(o, "reduce_only", False))

        assert partial.quantity < QTY
