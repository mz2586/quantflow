"""Tests for the capital-preservation rules: weekly loss, correlation, loss cooldown.

These three exist because the limits that came before them all have the same blind spot:
they each look at one order, one day, or one symbol in isolation. A slow bleed, a book
that is one bet wearing five costumes, and a strategy firing into a regime that has
already turned are all invisible to a per-order check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.core.config import RiskSettings
from quantflow.domain.enums import OrderSide, OrderType
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.risk.correlation import (
    MIN_OBSERVATIONS,
    CorrelationMatrix,
    pearson,
    returns_from_prices,
)
from quantflow.risk.rules import (
    ConsecutiveLossCooldownRule,
    CorrelationLimitRule,
    MaxWeeklyLossRule,
    RiskContext,
)

NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)
BTC = Symbol(base="BTC", quote="USDT")
ETH = Symbol(base="ETH", quote="USDT")
SOL = Symbol(base="SOL", quote="USDT")


def context(
    *,
    equity: str = "10000",
    week_start_equity: str | None = None,
    consecutive_losses: int = 0,
    last_loss_at: datetime | None = None,
    correlated: tuple[str, ...] = (),
    settings: RiskSettings | None = None,
) -> RiskContext:
    """A risk context for a modest long entry."""
    return RiskContext(
        request=OrderRequest(
            symbol=BTC,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            stop_loss_price=Decimal("59000"),
        ),
        # Equity is derived from cash plus marks; with no open positions the two are
        # the same, which keeps these cases about the rule rather than about valuation.
        portfolio=PortfolioSnapshot(
            base_currency="USDT",
            cash=Decimal(equity),
            timestamp=NOW,
        ),
        instrument=Instrument(symbol=BTC),
        reference_price=Decimal("60000"),
        now=NOW,
        settings=settings or RiskSettings(),
        week_start_equity=Decimal(week_start_equity) if week_start_equity else None,
        consecutive_losses=consecutive_losses,
        last_loss_at=last_loss_at,
        correlated_open_symbols=correlated,
    )


class TestMaxWeeklyLoss:
    """A daily limit alone permits a slow bleed that never trips it."""

    def test_a_profitable_week_is_allowed(self) -> None:
        verdict = MaxWeeklyLossRule().check(context(equity="11000", week_start_equity="10000"))
        assert verdict.allowed

    def test_a_small_weekly_loss_is_allowed(self) -> None:
        # 5% down against an 8% ceiling.
        verdict = MaxWeeklyLossRule().check(context(equity="9500", week_start_equity="10000"))
        assert verdict.allowed

    def test_the_weekly_ceiling_halts_trading(self) -> None:
        # The case the daily rule cannot see: five days at 2.9% each pass the daily check
        # every time and still leave a 14% hole.
        verdict = MaxWeeklyLossRule().check(context(equity="8600", week_start_equity="10000"))
        assert not verdict.allowed
        assert verdict.halts_trading

    def test_no_baseline_means_no_opinion(self) -> None:
        # Refusing to trade because the week has not started yet would be absurd.
        assert MaxWeeklyLossRule().check(context(week_start_equity=None)).allowed

    def test_a_zero_baseline_is_not_divided_by(self) -> None:
        assert MaxWeeklyLossRule().check(context(week_start_equity="0")).allowed


class TestCorrelationLimit:
    """Position count assumes independence. Crypto positions are rarely independent."""

    def test_an_uncorrelated_book_is_allowed(self) -> None:
        assert CorrelationLimitRule().check(context(correlated=())).allowed

    def test_below_the_cap_is_allowed(self) -> None:
        assert CorrelationLimitRule().check(context(correlated=("ETH/USDT",))).allowed

    def test_at_the_cap_is_denied(self) -> None:
        verdict = CorrelationLimitRule().check(context(correlated=("ETH/USDT", "SOL/USDT")))
        assert not verdict.allowed
        assert "ETH/USDT" in verdict.message

    def test_the_cap_is_configurable(self) -> None:
        settings = RiskSettings(max_correlated_positions=3)
        verdict = CorrelationLimitRule().check(
            context(correlated=("ETH/USDT", "SOL/USDT"), settings=settings)
        )
        assert verdict.allowed


class TestConsecutiveLossCooldown:
    """A losing streak usually means the regime turned. Firing into it compounds it."""

    def test_no_streak_is_allowed(self) -> None:
        assert ConsecutiveLossCooldownRule().check(context()).allowed

    def test_below_the_limit_is_allowed(self) -> None:
        verdict = ConsecutiveLossCooldownRule().check(
            context(consecutive_losses=3, last_loss_at=NOW - timedelta(minutes=1))
        )
        assert verdict.allowed

    def test_the_streak_pauses_new_entries(self) -> None:
        verdict = ConsecutiveLossCooldownRule().check(
            context(consecutive_losses=4, last_loss_at=NOW - timedelta(minutes=1))
        )
        assert not verdict.allowed
        # A brake, not a latch: it must not need an operator to clear it.
        assert not verdict.halts_trading

    def test_the_cooldown_expires_on_its_own(self) -> None:
        verdict = ConsecutiveLossCooldownRule().check(
            context(consecutive_losses=9, last_loss_at=NOW - timedelta(minutes=241))
        )
        assert verdict.allowed

    def test_a_streak_with_no_timestamp_does_not_block_forever(self) -> None:
        # Without a clock to expire against, blocking would be permanent.
        verdict = ConsecutiveLossCooldownRule().check(
            context(consecutive_losses=99, last_loss_at=None)
        )
        assert verdict.allowed


class TestCorrelationMath:
    """The estimate must refuse to guess when it cannot measure."""

    def test_identical_series_correlate_perfectly(self) -> None:
        series = [Decimal(str(value)) for value in range(1, MIN_OBSERVATIONS + 20)]
        returns = returns_from_prices(series)
        result = pearson(returns, returns)
        assert result is not None
        assert result > Decimal("0.99")

    def test_inverted_series_correlate_negatively(self) -> None:
        rising = returns_from_prices([Decimal(str(v)) for v in range(100, 200)])
        falling = tuple(-value for value in rising)
        result = pearson(rising, falling)
        assert result is not None
        assert result < Decimal("-0.99")

    def test_too_few_observations_is_unknown_not_zero(self) -> None:
        # Reporting zero would claim independence that was never measured, which is the
        # precise failure this module exists to prevent.
        short = tuple(Decimal("0.01") for _ in range(MIN_OBSERVATIONS - 1))
        assert pearson(short, short) is None

    def test_a_flat_series_has_no_correlation(self) -> None:
        flat = tuple(Decimal("0") for _ in range(MIN_OBSERVATIONS + 5))
        assert pearson(flat, flat) is None

    def test_a_zero_price_does_not_produce_an_infinite_return(self) -> None:
        prices = [Decimal("10"), Decimal("0"), Decimal("5"), Decimal("6")]
        assert all(value.is_finite() for value in returns_from_prices(prices))

    def test_the_matrix_is_order_independent(self) -> None:
        rising = returns_from_prices([Decimal(str(v)) for v in range(100, 200)])
        matrix = CorrelationMatrix.from_returns({BTC: rising, ETH: rising})
        assert matrix.between(BTC, ETH) == matrix.between(ETH, BTC)

    def test_a_symbol_correlates_with_itself(self) -> None:
        assert CorrelationMatrix(values={}).between(BTC, BTC) == Decimal("1")

    def test_negative_correlation_still_counts_as_concentration(self) -> None:
        # An inverted bet is still a bet on the same thing.
        rising = returns_from_prices([Decimal(str(v)) for v in range(100, 200)])
        falling = tuple(-value for value in rising)
        matrix = CorrelationMatrix.from_returns({BTC: rising, ETH: falling})
        hits = matrix.correlated_with(BTC, [ETH], threshold=Decimal("0.8"))
        assert hits == (ETH,)

    def test_unmeasurable_pairs_are_absent_not_zero(self) -> None:
        short = tuple(Decimal("0.01") * Decimal(i) for i in range(5))
        matrix = CorrelationMatrix.from_returns({BTC: short, ETH: short})
        assert matrix.between(BTC, ETH) is None


class TestSettingsCoherence:
    """Incoherent limits must fail at construction, not at the first breach."""

    def test_daily_cannot_exceed_weekly(self) -> None:
        with pytest.raises(ValueError, match="max_daily_loss_pct"):
            RiskSettings(max_daily_loss_pct=Decimal("0.10"), max_weekly_loss_pct=Decimal("0.05"))

    def test_weekly_cannot_exceed_drawdown(self) -> None:
        with pytest.raises(ValueError, match="max_weekly_loss_pct"):
            RiskSettings(max_weekly_loss_pct=Decimal("0.30"), max_drawdown_pct=Decimal("0.15"))

    def test_the_defaults_are_coherent(self) -> None:
        settings = RiskSettings()
        assert settings.max_daily_loss_pct <= settings.max_weekly_loss_pct
        assert settings.max_weekly_loss_pct <= settings.max_drawdown_pct
