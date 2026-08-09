"""Risk engine: sizing, every rule's allow and deny path, and the kill switch.

The critical invariant this file exists to protect: **no entry can reach a venue without a
stop loss, and no order can bypass the engine.**
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.core.config import RiskSettings, Severity
from quantflow.core.errors import KillSwitchEngagedError, RiskViolationError, ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide, OrderType, SignalDirection
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Fill, OrderRequest
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import Position
from quantflow.domain.signals import Signal
from quantflow.risk.engine import (
    RiskEngine,
    assert_protected,
    daily_loss_headroom,
    drawdown_headroom,
    exposure_headroom,
    summarise_headroom,
)
from quantflow.risk.killswitch import KillSwitch
from quantflow.risk.rules import (
    InstrumentRule,
    KillSwitchRule,
    MaxConcurrentPositionsRule,
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
    MaxPositionSizeRule,
    MaxTotalExposureRule,
    OrderNotionalRule,
    OrderRateRule,
    RiskContext,
    StopLossRequiredRule,
    SufficientCashRule,
    TradingHaltedRule,
    build_default_rules,
)
from quantflow.risk.sizing import (
    FixedFractionalSizer,
    FixedNotionalSizer,
    SizingRequest,
    VolatilityTargetSizer,
    build_sizer,
)
from tests.conftest import REFERENCE_TIME


def settings(**overrides: object) -> RiskSettings:
    return RiskSettings(**overrides)  # type: ignore[arg-type]


def permissive_settings(**overrides: object) -> RiskSettings:
    """Limits wide enough that a sizing test measures the sizer, not the caps.

    `max_total_exposure_pct` is raised alongside `max_position_pct` because the settings
    model (correctly) refuses a per-position cap above the aggregate one.
    """
    kwargs: dict[str, object] = {
        "max_position_pct": Decimal("1"),
        "max_total_exposure_pct": Decimal("1"),
        "max_order_notional": Decimal("1000000"),
        "consecutive_loss_limit": 100,
        "max_correlated_positions": 50,
    }
    kwargs.update(overrides)
    return settings(**kwargs)


def instrument(symbol: Symbol, **overrides: object) -> Instrument:
    kwargs: dict[str, object] = {
        "symbol": symbol,
        "price_tick": Decimal("0.01"),
        "quantity_step": Decimal("0.00001"),
        "min_quantity": Decimal("0.00001"),
        "min_notional": Decimal("5"),
    }
    kwargs.update(overrides)
    return Instrument(**kwargs)  # type: ignore[arg-type]


def portfolio(
    *,
    cash: Decimal = Decimal("10000"),
    positions: tuple[Position, ...] = (),
    prices: dict[Symbol, Decimal] | None = None,
    peak_equity: Decimal = Decimal("10000"),
    day_start_equity: Decimal = Decimal("10000"),
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=REFERENCE_TIME,
        base_currency="USDT",
        cash=cash,
        positions=positions,
        mark_prices=prices or {},
        peak_equity=peak_equity,
        day_start_equity=day_start_equity,
    )


def long_position(symbol: Symbol, quantity: str, price: str) -> Position:
    position, _ = Position(symbol=symbol).apply_fill(
        Fill(
            fill_id="f",
            order_id="o",
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal(quantity),
            price=Decimal(price),
            fee=ZERO,
            fee_currency="USDT",
            timestamp=REFERENCE_TIME,
        )
    )
    return position


def request(symbol: Symbol, **overrides: object) -> OrderRequest:
    kwargs: dict[str, object] = {
        "symbol": symbol,
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("0.1"),
        "stop_loss_price": Decimal("49000"),
    }
    kwargs.update(overrides)
    return OrderRequest(**kwargs)  # type: ignore[arg-type]


def context(symbol: Symbol, **overrides: object) -> RiskContext:
    kwargs: dict[str, object] = {
        "request": request(symbol),
        "portfolio": portfolio(),
        "instrument": instrument(symbol),
        "reference_price": Decimal("50000"),
        "now": REFERENCE_TIME,
        "settings": settings(),
    }
    kwargs.update(overrides)
    return RiskContext(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
class TestFixedFractionalSizer:
    def test_risk_at_the_stop_matches_the_budget(self, btc: Symbol) -> None:
        config = permissive_settings()
        sizer = FixedFractionalSizer(config, risk_per_trade=Decimal("0.01"))
        result = sizer.size(
            SizingRequest(
                equity=Decimal("10000"),
                price=Decimal("50000"),
                instrument=instrument(btc),
                stop_loss_price=Decimal("49000"),
            )
        )
        # Budget 100, stop distance 1000 -> 0.1 units, risking exactly 100 at the stop.
        assert result.quantity == Decimal("0.1")
        assert result.risk_amount == Decimal("100")

    def test_a_wider_stop_produces_a_smaller_position(self, btc: Symbol) -> None:
        config = permissive_settings()
        sizer = FixedFractionalSizer(config, risk_per_trade=Decimal("0.01"))
        common = {
            "equity": Decimal("10000"),
            "price": Decimal("50000"),
            "instrument": instrument(btc),
        }
        tight = sizer.size(SizingRequest(**common, stop_loss_price=Decimal("49500")))  # type: ignore[arg-type]
        wide = sizer.size(SizingRequest(**common, stop_loss_price=Decimal("45000")))  # type: ignore[arg-type]
        assert tight.quantity > wide.quantity
        # Loss at the stop is the same either way — that is the entire point.
        assert tight.risk_amount == wide.risk_amount == Decimal("100")

    def test_refuses_without_a_stop(self, btc: Symbol) -> None:
        # "Risk 1% of equity" has no meaning without a defined loss point.
        sizer = FixedFractionalSizer(settings(), risk_per_trade=Decimal("0.01"))
        with pytest.raises(ValidationError, match="requires a stop loss"):
            sizer.size(
                SizingRequest(
                    equity=Decimal("10000"),
                    price=Decimal("50000"),
                    instrument=instrument(btc),
                )
            )

    def test_conviction_scales_the_size(self, btc: Symbol) -> None:
        config = permissive_settings()
        sizer = FixedFractionalSizer(config, risk_per_trade=Decimal("0.01"))
        common = {
            "equity": Decimal("10000"),
            "price": Decimal("50000"),
            "instrument": instrument(btc),
            "stop_loss_price": Decimal("49000"),
        }
        full = sizer.size(SizingRequest(**common))  # type: ignore[arg-type]
        half = sizer.size(SizingRequest(**common, conviction=Decimal("0.5")))  # type: ignore[arg-type]
        assert half.quantity == full.quantity / 2

    def test_position_cap_binds(self, btc: Symbol) -> None:
        # Risk budget alone would allow far more than the position limit permits.
        sizer = FixedFractionalSizer(
            settings(max_position_pct=Decimal("0.1")), risk_per_trade=Decimal("0.5")
        )
        result = sizer.size(
            SizingRequest(
                equity=Decimal("10000"),
                price=Decimal("50000"),
                instrument=instrument(btc),
                stop_loss_price=Decimal("49900"),
            )
        )
        assert result.capped_by == "max_position_pct"
        assert result.notional <= Decimal("1000")

    def test_cash_binds_on_a_spot_account(self, btc: Symbol) -> None:
        sizer = FixedFractionalSizer(permissive_settings(), risk_per_trade=Decimal("0.5"))
        result = sizer.size(
            SizingRequest(
                equity=Decimal("10000"),
                price=Decimal("50000"),
                instrument=instrument(btc),
                stop_loss_price=Decimal("49900"),
                available_cash=Decimal("500"),
            )
        )
        assert result.capped_by == "available_cash"
        assert result.notional <= Decimal("500")

    def test_dust_is_rejected_rather_than_submitted(self, btc: Symbol) -> None:
        sizer = FixedFractionalSizer(settings(), risk_per_trade=Decimal("0.01"))
        result = sizer.size(
            SizingRequest(
                equity=Decimal("1"),
                price=Decimal("50000"),
                instrument=instrument(btc, min_notional=Decimal("10")),
                stop_loss_price=Decimal("49000"),
            )
        )
        assert not result.is_tradable
        assert result.capped_by in {"below_min_notional", "below_venue_min_quantity"}

    def test_size_is_snapped_to_the_venue_lot_grid(self, btc: Symbol) -> None:
        sizer = FixedFractionalSizer(permissive_settings(), risk_per_trade=Decimal("0.01"))
        result = sizer.size(
            SizingRequest(
                equity=Decimal("10000"),
                price=Decimal("50000"),
                instrument=instrument(btc, quantity_step=Decimal("0.001")),
                stop_loss_price=Decimal("49123"),
            )
        )
        assert result.quantity % Decimal("0.001") == ZERO

    @pytest.mark.parametrize("risk", [Decimal("0"), Decimal("-0.1"), Decimal("1.5")])
    def test_invalid_risk_per_trade(self, risk: Decimal) -> None:
        with pytest.raises(ValidationError, match="risk_per_trade"):
            FixedFractionalSizer(settings(), risk_per_trade=risk)


class TestOtherSizers:
    def test_volatility_target_requires_an_atr(self, btc: Symbol) -> None:
        sizer = VolatilityTargetSizer(settings())
        with pytest.raises(ValidationError, match="ATR"):
            sizer.size(
                SizingRequest(
                    equity=Decimal("10000"),
                    price=Decimal("50000"),
                    instrument=instrument(btc),
                )
            )

    def test_volatility_target_sizes_inversely_to_volatility(self, btc: Symbol) -> None:
        config = permissive_settings()
        sizer = VolatilityTargetSizer(config, target_volatility_pct=Decimal("0.01"))
        common = {
            "equity": Decimal("10000"),
            "price": Decimal("50000"),
            "instrument": instrument(btc),
        }
        calm = sizer.size(SizingRequest(**common, volatility=Decimal("100")))  # type: ignore[arg-type]
        wild = sizer.size(SizingRequest(**common, volatility=Decimal("1000")))  # type: ignore[arg-type]
        assert calm.quantity > wild.quantity

    def test_fixed_notional_ignores_the_stop(self, btc: Symbol) -> None:
        sizer = FixedNotionalSizer(
            settings(max_position_pct=Decimal("0.1"), max_order_notional=Decimal("100000")),
            allocation_pct=Decimal("0.1"),
        )
        result = sizer.size(
            SizingRequest(
                equity=Decimal("10000"),
                price=Decimal("50000"),
                instrument=instrument(btc),
            )
        )
        assert result.notional == pytest.approx(Decimal("1000"), abs=Decimal("1"))

    def test_build_sizer_by_name(self) -> None:
        assert build_sizer(settings(), "fixed_fractional").name == "fixed_fractional"
        assert build_sizer(settings(), "volatility_target").name == "volatility_target"
        with pytest.raises(ValidationError, match="unknown sizing method"):
            build_sizer(settings(), "magic")


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #
class TestStopLossRule:
    def test_entry_without_a_stop_is_always_rejected(self, btc: Symbol) -> None:
        """The single most important assertion in the system."""
        verdict = StopLossRequiredRule().evaluate(
            context(btc, request=request(btc, stop_loss_price=None))
        )
        assert not verdict.allowed
        assert verdict.severity is Severity.CRITICAL

    def test_entry_with_a_valid_stop_is_allowed(self, btc: Symbol) -> None:
        assert StopLossRequiredRule().evaluate(context(btc)).allowed

    def test_long_stop_above_entry_is_rejected(self, btc: Symbol) -> None:
        verdict = StopLossRequiredRule().evaluate(
            context(btc, request=request(btc, stop_loss_price=Decimal("51000")))
        )
        assert not verdict.allowed
        assert "not below entry" in verdict.message

    def test_short_stop_below_entry_is_rejected(self, btc: Symbol) -> None:
        verdict = StopLossRequiredRule().evaluate(
            context(
                btc,
                request=request(btc, side=OrderSide.SELL, stop_loss_price=Decimal("49000")),
            )
        )
        assert not verdict.allowed
        assert "not above entry" in verdict.message

    def test_excessively_wide_stop_is_rejected(self, btc: Symbol) -> None:
        verdict = StopLossRequiredRule().evaluate(
            context(
                btc,
                request=request(btc, stop_loss_price=Decimal("10000")),
                settings=settings(max_stop_loss_pct=Decimal("0.1")),
            )
        )
        assert not verdict.allowed
        assert "exceeds the maximum" in verdict.message

    def test_can_be_disabled_by_configuration(self, btc: Symbol) -> None:
        verdict = StopLossRequiredRule().evaluate(
            context(
                btc,
                request=request(btc, stop_loss_price=None),
                settings=settings(require_stop_loss=False),
            )
        )
        assert verdict.allowed

    def test_exits_are_exempt(self, btc: Symbol) -> None:
        # Refusing an exit for lacking a stop would trap the position permanently.
        verdict = StopLossRequiredRule().evaluate(
            context(
                btc,
                request=request(btc, side=OrderSide.SELL, stop_loss_price=None, reduce_only=True),
            )
        )
        assert verdict.allowed


class TestHardStops:
    def test_kill_switch_blocks_entries(self, btc: Symbol) -> None:
        verdict = KillSwitchRule().evaluate(context(btc, kill_switch_engaged=True))
        assert not verdict.allowed
        assert verdict.halts_trading

    def test_kill_switch_still_permits_exits(self, btc: Symbol) -> None:
        verdict = KillSwitchRule().evaluate(
            context(
                btc,
                kill_switch_engaged=True,
                request=request(btc, side=OrderSide.SELL, reduce_only=True, stop_loss_price=None),
            )
        )
        assert verdict.allowed

    def test_trading_halted_blocks_entries(self, btc: Symbol) -> None:
        assert not TradingHaltedRule().evaluate(context(btc, trading_halted=True)).allowed

    def test_clear_state_allows(self, btc: Symbol) -> None:
        assert KillSwitchRule().evaluate(context(btc)).allowed
        assert TradingHaltedRule().evaluate(context(btc)).allowed


class TestExposureRules:
    def test_position_size_limit(self, btc: Symbol) -> None:
        # 0.1 BTC at 50k = 5000 notional against 10k equity = 50%, above the 10% cap.
        verdict = MaxPositionSizeRule().evaluate(context(btc))
        assert not verdict.allowed
        assert verdict.limit == Decimal("0.1")

    def test_position_size_limit_counts_existing_exposure(self, btc: Symbol) -> None:
        existing = long_position(btc, "0.015", "50000")  # 750 notional
        verdict = MaxPositionSizeRule().evaluate(
            context(
                btc,
                request=request(btc, quantity=Decimal("0.006")),  # +300 -> 1050 > 1000
                portfolio=portfolio(
                    cash=Decimal("9250"),
                    positions=(existing,),
                    prices={btc: Decimal("50000")},
                ),
            )
        )
        assert not verdict.allowed

    def test_position_size_within_the_limit_is_allowed(self, btc: Symbol) -> None:
        assert (
            MaxPositionSizeRule()
            .evaluate(context(btc, request=request(btc, quantity=Decimal("0.01"))))
            .allowed
        )

    def test_total_exposure_limit(self, btc: Symbol, eth: Symbol) -> None:
        existing = long_position(eth, "2", "2500")  # 5000 notional
        verdict = MaxTotalExposureRule().evaluate(
            context(
                btc,
                request=request(btc, quantity=Decimal("0.05")),  # +2500 -> 7500 / 10000
                portfolio=portfolio(
                    cash=Decimal("5000"),
                    positions=(existing,),
                    prices={eth: Decimal("2500")},
                    peak_equity=Decimal("10000"),
                ),
                settings=settings(max_total_exposure_pct=Decimal("0.6")),
            )
        )
        assert not verdict.allowed

    def test_concurrent_position_limit(self, btc: Symbol, eth: Symbol) -> None:
        verdict = MaxConcurrentPositionsRule().evaluate(
            context(
                btc,
                portfolio=portfolio(
                    positions=(long_position(eth, "1", "2500"),),
                    prices={eth: Decimal("2500")},
                ),
                settings=settings(max_concurrent_positions=1),
            )
        )
        assert not verdict.allowed

    def test_adding_to_an_existing_position_is_not_a_new_slot(self, btc: Symbol) -> None:
        verdict = MaxConcurrentPositionsRule().evaluate(
            context(
                btc,
                portfolio=portfolio(
                    positions=(long_position(btc, "0.001", "50000"),),
                    prices={btc: Decimal("50000")},
                ),
                settings=settings(max_concurrent_positions=1),
            )
        )
        assert verdict.allowed

    def test_leverage_limit(self, btc: Symbol) -> None:
        verdict = MaxLeverageRule().evaluate(
            context(btc, settings=settings(max_leverage=Decimal("1")))
        )
        # 5000 notional on 10000 equity is 0.5x, inside a 1x limit.
        assert verdict.allowed

    def test_zero_equity_is_a_hard_denial(self, btc: Symbol) -> None:
        empty = PortfolioSnapshot(timestamp=REFERENCE_TIME, base_currency="USDT", cash=ZERO)
        verdict = MaxPositionSizeRule().evaluate(context(btc, portfolio=empty))
        assert not verdict.allowed
        assert verdict.severity is Severity.CRITICAL


class TestLossRules:
    def test_daily_loss_halts_but_does_not_latch(self, btc: Symbol) -> None:
        verdict = MaxDailyLossRule().evaluate(
            context(
                btc,
                portfolio=portfolio(cash=Decimal("9600"), day_start_equity=Decimal("10000")),
                settings=settings(max_daily_loss_pct=Decimal("0.03")),
            )
        )
        assert not verdict.allowed
        assert verdict.halts_trading
        # A bad day is normal; it should not require an operator to clear a switch.
        assert not verdict.engages_kill_switch

    def test_daily_loss_within_the_limit(self, btc: Symbol) -> None:
        assert (
            MaxDailyLossRule()
            .evaluate(context(btc, portfolio=portfolio(cash=Decimal("9900"))))
            .allowed
        )

    def test_profit_never_trips_the_daily_rule(self, btc: Symbol) -> None:
        assert (
            MaxDailyLossRule()
            .evaluate(context(btc, portfolio=portfolio(cash=Decimal("11000"))))
            .allowed
        )

    def test_drawdown_latches_the_kill_switch(self, btc: Symbol) -> None:
        verdict = MaxDrawdownRule().evaluate(
            context(
                btc,
                portfolio=portfolio(cash=Decimal("8000"), peak_equity=Decimal("10000")),
                settings=settings(max_drawdown_pct=Decimal("0.15")),
            )
        )
        assert not verdict.allowed
        assert verdict.engages_kill_switch

    def test_drawdown_within_the_limit(self, btc: Symbol) -> None:
        assert (
            MaxDrawdownRule()
            .evaluate(context(btc, portfolio=portfolio(cash=Decimal("9500"))))
            .allowed
        )

    def test_order_rate_limit(self, btc: Symbol) -> None:
        verdict = OrderRateRule().evaluate(
            context(btc, orders_last_minute=10, settings=settings(max_orders_per_minute=10))
        )
        assert not verdict.allowed
        assert verdict.severity is Severity.CRITICAL

    def test_order_rate_applies_to_exits_too(self, btc: Symbol) -> None:
        # A runaway loop must be stopped regardless of direction.
        verdict = OrderRateRule().evaluate(
            context(
                btc,
                orders_last_minute=10,
                settings=settings(max_orders_per_minute=10),
                request=request(btc, reduce_only=True, side=OrderSide.SELL, stop_loss_price=None),
            )
        )
        assert not verdict.allowed


class TestVenueAndCashRules:
    def test_instrument_rules_are_enforced(self, btc: Symbol) -> None:
        verdict = InstrumentRule().evaluate(
            context(
                btc,
                request=request(btc, quantity=Decimal("0.000001")),
                instrument=instrument(btc, min_quantity=Decimal("0.001")),
            )
        )
        assert not verdict.allowed

    def test_order_notional_upper_bound(self, btc: Symbol) -> None:
        verdict = OrderNotionalRule().evaluate(
            context(btc, settings=settings(max_order_notional=Decimal("1000")))
        )
        assert not verdict.allowed

    def test_order_notional_lower_bound_for_entries(self, btc: Symbol) -> None:
        verdict = OrderNotionalRule().evaluate(
            context(
                btc,
                request=request(btc, quantity=Decimal("0.0001")),  # 5 notional
                settings=settings(min_order_notional=Decimal("10")),
            )
        )
        assert not verdict.allowed

    def test_dust_exits_are_permitted(self, btc: Symbol) -> None:
        # Closing a tiny leftover position must always be allowed.
        verdict = OrderNotionalRule().evaluate(
            context(
                btc,
                request=request(
                    btc,
                    quantity=Decimal("0.0001"),
                    reduce_only=True,
                    side=OrderSide.SELL,
                    stop_loss_price=None,
                ),
                settings=settings(min_order_notional=Decimal("10")),
            )
        )
        assert verdict.allowed

    def test_insufficient_cash_is_rejected(self, btc: Symbol) -> None:
        verdict = SufficientCashRule().evaluate(
            context(btc, portfolio=portfolio(cash=Decimal("100")))
        )
        assert not verdict.allowed

    def test_sells_do_not_consume_cash(self, btc: Symbol) -> None:
        verdict = SufficientCashRule().evaluate(
            context(
                btc,
                request=request(btc, side=OrderSide.SELL, stop_loss_price=Decimal("51000")),
                portfolio=portfolio(cash=Decimal("1")),
            )
        )
        assert verdict.allowed

    def test_fee_headroom_is_reserved(self, btc: Symbol) -> None:
        # An order consuming the last cent is rejected by the venue for being unable to
        # pay its own commission, so it is refused here first.
        verdict = SufficientCashRule().evaluate(
            context(
                btc,
                request=request(btc, quantity=Decimal("0.02")),  # exactly 1000 notional
                portfolio=portfolio(cash=Decimal("1000")),
            )
        )
        assert not verdict.allowed


# --------------------------------------------------------------------------- #
# Kill switch
# --------------------------------------------------------------------------- #
class TestKillSwitch:
    async def test_starts_clear(self) -> None:
        switch = KillSwitch()
        assert not switch.engaged
        switch.require_clear()

    async def test_engage_and_clear(self, clock) -> None:
        switch = KillSwitch(clock=clock)
        await switch.engage("drawdown breach")
        assert switch.state.engaged
        assert switch.state.reason == "drawdown breach"
        assert switch.state.engaged_at == REFERENCE_TIME
        with pytest.raises(KillSwitchEngagedError, match="drawdown breach"):
            switch.require_clear()

        # Bound to a local: mypy narrows `switch.state.engaged` from the assertion
        # above and would otherwise treat the post-clear branch as unreachable.
        cleared = await switch.clear(actor="operator")
        assert not cleared.engaged
        switch.require_clear()

    async def test_engage_is_idempotent(self) -> None:
        switch = KillSwitch()
        await switch.engage("first reason")
        await switch.engage("second reason")
        assert switch.state.reason == "first reason"

    async def test_load_without_a_database_is_a_noop(self) -> None:
        switch = KillSwitch()
        assert not (await switch.load()).engaged


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class TestRiskEngine:
    def _engine(self, clock, **overrides: object) -> RiskEngine:
        return RiskEngine(
            settings(
                max_position_pct=Decimal("0.5"),
                max_total_exposure_pct=Decimal("1"),
                max_order_notional=Decimal("100000"),
                **overrides,
            ),
            clock=clock,
        )

    async def test_approves_a_well_formed_order(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).approve(
            request(btc, quantity=Decimal("0.02")),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert decision.approved
        assert decision.request is not None
        assert decision.denials == ()

    async def test_denies_an_entry_without_a_stop(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).approve(
            request(btc, quantity=Decimal("0.02"), stop_loss_price=None),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert decision.blocking_rule == "stop_loss_required"
        with pytest.raises(RiskViolationError, match="stop loss"):
            decision.raise_if_denied()

    async def test_all_rules_are_evaluated_not_short_circuited(self, btc: Symbol, clock) -> None:
        # Reporting every violation at once beats making an operator fix them one per run.
        decision = await self._engine(clock, max_concurrent_positions=1).approve(
            request(btc, quantity=Decimal("100"), stop_loss_price=None),
            portfolio=portfolio(cash=Decimal("100")),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert len(decision.denials) > 1

    async def test_drawdown_breach_latches_the_switch(self, btc: Symbol, clock) -> None:
        engine = self._engine(clock, max_drawdown_pct=Decimal("0.1"))
        decision = await engine.approve(
            request(btc, quantity=Decimal("0.02")),
            portfolio=portfolio(cash=Decimal("8000"), peak_equity=Decimal("10000")),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert decision.engaged_kill_switch
        assert engine.kill_switch.engaged
        assert engine.is_halted

    async def test_halt_lifts_at_the_next_utc_day(self, btc: Symbol, clock) -> None:
        engine = self._engine(clock)
        engine.halt_for_the_day("test")
        assert engine.is_halted
        clock.advance(delta=timedelta(days=1, seconds=1))
        assert not engine.is_halted

    async def test_order_rate_tracking_prunes_old_entries(self, clock) -> None:
        engine = self._engine(clock)
        for _ in range(5):
            engine.record_order()
        assert engine.orders_in_last_minute() == 5
        clock.advance(seconds=61)
        assert engine.orders_in_last_minute() == 0

    async def test_signal_to_order_attaches_a_default_stop(self, btc: Symbol, clock) -> None:
        # A strategy with no opinion on stops must not produce a naked position.
        engine = self._engine(clock)
        decision = await engine.evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.LONG,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
                reference_price=Decimal("50000"),
            ),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert decision.approved
        assert decision.request is not None
        assert decision.request.stop_loss_price is not None
        assert decision.request.stop_loss_price < Decimal("50000")

    async def test_signal_stop_is_preserved(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.LONG,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
                reference_price=Decimal("50000"),
                stop_loss_price=Decimal("48000"),
            ),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert decision.approved
        assert decision.request is not None
        assert decision.request.stop_loss_price == Decimal("48000")

    async def test_hold_signals_produce_no_order(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).evaluate_signal(
            Signal.hold(btc, REFERENCE_TIME, "test"),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert "not actionable" in decision.reason

    async def test_close_signal_produces_a_reduce_only_order(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.CLOSE,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
            ),
            portfolio=portfolio(
                cash=Decimal("5000"),
                positions=(long_position(btc, "0.1", "50000"),),
                prices={btc: Decimal("50000")},
            ),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert decision.approved
        assert decision.request is not None
        assert decision.request.reduce_only
        assert decision.request.side is OrderSide.SELL
        assert decision.request.quantity == Decimal("0.1")

    async def test_close_without_a_position_is_refused(self, btc: Symbol, clock) -> None:
        decision = await self._engine(clock).evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.CLOSE,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
            ),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert "no open position" in decision.reason

    async def test_exits_are_allowed_while_halted(self, btc: Symbol, clock) -> None:
        # Halting entries must never trap an open position.
        engine = self._engine(clock)
        engine.halt_for_the_day("daily loss")
        decision = await engine.evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.CLOSE,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
            ),
            portfolio=portfolio(
                cash=Decimal("5000"),
                positions=(long_position(btc, "0.1", "50000"),),
                prices={btc: Decimal("50000")},
            ),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert decision.approved

    async def test_entries_are_blocked_while_halted(self, btc: Symbol, clock) -> None:
        engine = self._engine(clock)
        engine.halt_for_the_day("daily loss")
        decision = await engine.evaluate_signal(
            Signal(
                symbol=btc,
                direction=SignalDirection.LONG,
                timestamp=REFERENCE_TIME,
                strategy_id="test",
                reference_price=Decimal("50000"),
                stop_loss_price=Decimal("49000"),
            ),
            portfolio=portfolio(),
            instrument=instrument(btc),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved
        assert decision.blocking_rule == "trading_halted"

    async def test_conviction_scales_the_resulting_size(self, btc: Symbol, clock) -> None:
        engine = self._engine(clock)
        sizes = []
        for conviction in (Decimal("1"), Decimal("0.5")):
            decision = await engine.evaluate_signal(
                Signal(
                    symbol=btc,
                    direction=SignalDirection.LONG,
                    timestamp=REFERENCE_TIME,
                    strategy_id="test",
                    conviction=conviction,
                    reference_price=Decimal("50000"),
                    stop_loss_price=Decimal("49000"),
                ),
                portfolio=portfolio(),
                instrument=instrument(btc),
                reference_price=Decimal("50000"),
            )
            assert decision.request is not None
            sizes.append(decision.request.quantity)
        assert sizes[1] < sizes[0]

    def test_describe_exposes_the_configuration(self, clock) -> None:
        described = self._engine(clock).describe()
        assert described["sizer"] == "fixed_fractional"
        assert "stop_loss_required" in described["rules"]
        assert described["kill_switch"]["engaged"] is False

    def test_default_rule_set_covers_every_mandated_control(self) -> None:
        names = {rule.name for rule in build_default_rules()}
        # These map one-to-one onto the platform's stated risk requirements.
        assert {
            "stop_loss_required",
            "max_position_pct",
            "max_daily_loss",
            "max_drawdown",
            "max_concurrent_positions",
            "kill_switch",
        } <= names


class TestAssertProtected:
    def test_rejects_an_unprotected_entry(self, btc: Symbol) -> None:
        # Deliberately redundant with the rule: this catches a future refactor that
        # introduces a path around the engine.
        with pytest.raises(RiskViolationError, match="unprotected entry"):
            assert_protected(request(btc, stop_loss_price=None), settings())

    def test_allows_a_protected_entry(self, btc: Symbol) -> None:
        assert_protected(request(btc), settings())

    def test_allows_a_reduce_only_order(self, btc: Symbol) -> None:
        assert_protected(
            request(btc, reduce_only=True, side=OrderSide.SELL, stop_loss_price=None),
            settings(),
        )


class TestHeadroom:
    def test_daily_loss_headroom(self) -> None:
        assert daily_loss_headroom(
            portfolio(cash=Decimal("9800")), settings(max_daily_loss_pct=Decimal("0.03"))
        ) == Decimal("100")

    def test_drawdown_headroom(self) -> None:
        assert drawdown_headroom(
            portfolio(cash=Decimal("9500")), settings(max_drawdown_pct=Decimal("0.15"))
        ) == Decimal("0.10")

    def test_exposure_headroom(self, btc: Symbol) -> None:
        snapshot = portfolio(
            cash=Decimal("5000"),
            positions=(long_position(btc, "0.1", "50000"),),
            prices={btc: Decimal("50000")},
        )
        headroom = exposure_headroom(snapshot, settings(max_total_exposure_pct=Decimal("0.6")))
        assert headroom == Decimal("6000") - Decimal("5000")

    def test_summarise(self) -> None:
        summary = summarise_headroom(portfolio(), settings())
        assert set(summary) >= {"daily_loss", "drawdown_pct", "exposure", "positions"}
