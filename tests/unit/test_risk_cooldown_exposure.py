"""Per-symbol loss cooldown, and gross exposure reaching persisted state."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from quantflow.core.clock import FrozenClock
from quantflow.core.config import RiskSettings
from quantflow.domain.enums import OrderSide, SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from tests.conftest import REFERENCE_TIME
from tests.unit.test_risk import instrument as make_instrument

BTC = Symbol.parse("BTC/USDT")
XRP = Symbol.parse("XRP/USDT")


def engine(clock: FrozenClock) -> RiskEngine:
    return RiskEngine(RiskSettings(), clock=clock)


def entry_for(symbol: Symbol, price: str) -> Signal:
    value = Decimal(price)
    return Signal(
        symbol=symbol,
        direction=SignalDirection.LONG,
        timestamp=REFERENCE_TIME,
        strategy_id="test",
        reference_price=value,
        stop_loss_price=value * Decimal("0.98"),
        take_profit_price=value * Decimal("1.04"),
    )


def snapshot(cash: str = "10000") -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=REFERENCE_TIME,
        base_currency="USDT",
        cash=Decimal(cash),
        positions=(),
        mark_prices={BTC: Decimal("50000"), XRP: Decimal("1")},
    )


async def approve(risk: RiskEngine, symbol: Symbol, price: str):  # type: ignore[no-untyped-def]
    return await risk.evaluate_signal(
        entry_for(symbol, price),
        portfolio=snapshot(),
        instrument=make_instrument(symbol),
        reference_price=Decimal(price),
        volatility=Decimal(price) * Decimal("0.01"),
    )


class TestPerSymbolCooldown:
    async def test_losing_streak_blocks_only_its_own_symbol(self) -> None:
        """A run of losses on XRP must not stop BTC being traded."""
        clock = FrozenClock(REFERENCE_TIME)
        risk = engine(clock)
        await risk.start()

        limit = risk.settings.consecutive_loss_limit
        for _ in range(limit + 1):
            risk.record_trade_result(Decimal("-5"), closed_at=clock.now(), symbol=XRP)

        blocked = await approve(risk, XRP, "1")
        assert not blocked.approved
        assert "consecutive_loss_cooldown" in {v.rule for v in blocked.denials}

        allowed = await approve(risk, BTC, "50000")
        assert allowed.approved, "an unrelated symbol must stay tradable"

    async def test_a_win_clears_only_that_symbols_streak(self) -> None:
        clock = FrozenClock(REFERENCE_TIME)
        risk = engine(clock)
        await risk.start()

        limit = risk.settings.consecutive_loss_limit
        for symbol in (XRP, BTC):
            for _ in range(limit + 1):
                risk.record_trade_result(Decimal("-5"), closed_at=clock.now(), symbol=symbol)

        risk.record_trade_result(Decimal("10"), closed_at=clock.now(), symbol=BTC)

        assert (await approve(risk, BTC, "50000")).approved
        assert not (await approve(risk, XRP, "1")).approved

    async def test_cooldown_expires_on_its_own(self) -> None:
        clock = FrozenClock(REFERENCE_TIME)
        risk = engine(clock)
        await risk.start()

        limit = risk.settings.consecutive_loss_limit
        for _ in range(limit + 1):
            risk.record_trade_result(Decimal("-5"), closed_at=clock.now(), symbol=XRP)
        assert not (await approve(risk, XRP, "1")).approved

        clock.set(REFERENCE_TIME + timedelta(minutes=risk.settings.loss_cooldown_minutes + 1))
        assert (await approve(risk, XRP, "1")).approved

    async def test_losses_on_different_symbols_do_not_accumulate(self) -> None:
        """Five losses across five symbols is not a five-loss streak anywhere."""
        clock = FrozenClock(REFERENCE_TIME)
        risk = engine(clock)
        await risk.start()

        for name in ("ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"):
            risk.record_trade_result(
                Decimal("-5"), closed_at=clock.now(), symbol=Symbol.parse(name)
            )

        assert (await approve(risk, BTC, "50000")).approved
        assert (await approve(risk, XRP, "1")).approved


class TestGrossExposure:
    def test_equity_point_carries_exposure(self) -> None:
        manager = PortfolioManager(starting_equity=Decimal("10000"))
        manager.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                fee=Decimal("5"),
                fee_currency="USDT",
                timestamp=REFERENCE_TIME,
            )
        )
        manager.update_mark_price(BTC, Decimal("51000"))
        point = manager.record_equity(REFERENCE_TIME)

        assert point.gross_exposure == Decimal("5100.0")
        assert point.position_count == 1

    def test_exposure_is_zero_only_when_flat(self) -> None:
        manager = PortfolioManager(starting_equity=Decimal("10000"))
        point = manager.record_equity(REFERENCE_TIME)
        assert point.gross_exposure == Decimal("0")

    def test_unmarked_position_does_not_stop_the_sample(self) -> None:
        """A missing mark must not raise; the curve has to keep advancing."""
        manager = PortfolioManager(starting_equity=Decimal("10000"))
        manager.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                fee=Decimal("5"),
                fee_currency="USDT",
                timestamp=REFERENCE_TIME,
            )
        )
        point = manager.record_equity(REFERENCE_TIME)
        assert point.position_count == 1
        assert point.gross_exposure >= Decimal("0")
