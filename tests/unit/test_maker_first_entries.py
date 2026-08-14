"""Entries go passive; exits stay aggressive.

Converting entries at the single point where a Signal becomes an OrderRequest, rather than
in twenty-two strategies, means every strategy gets maker pricing without any of them
knowing about it — and there is exactly one place where the behaviour can be wrong.

The asymmetry is deliberate and is the whole design:

* **Entries are patient.** Nothing is lost by not entering. A missed setup costs zero; a
  taker entry costs 0.06% every single time. So entries rest passively, post-only, and are
  abandoned if unfilled.
* **Exits are not.** A protective stop that waits for a passive fill is not protection. A
  reduce-only exit crosses the spread and pays taker, because the cost of being slow there
  is unbounded and the fee is not.

Default OFF. A change that alters how every order in the system is priced must be switched
on deliberately, not inherited by a config that predates it.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.config import RiskSettings
from quantflow.domain.enums import OrderType


class TestSettings:
    def test_maker_first_is_off_by_default(self) -> None:
        """No config written before this feature silently changes execution."""
        assert RiskSettings().maker_first_entries is False

    def test_entry_lifetime_is_bounded_by_default(self) -> None:
        """An unbounded resting entry can fill on a signal that has expired."""
        assert RiskSettings().entry_limit_max_bars >= 1


class TestConversion:
    def test_an_entry_becomes_a_post_only_limit(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        order_type, price, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert (order_type, price, post_only) == (OrderType.LIMIT, Decimal("100"), True)

    def test_disabled_leaves_the_order_untouched(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        assert as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=False,
        ) == (OrderType.MARKET, None, False)

    def test_a_strategy_that_already_chose_a_limit_is_respected(self) -> None:
        """A strategy with its own price knows something this layer does not."""
        from quantflow.risk.engine import as_maker_entry

        _, price, post_only = as_maker_entry(
            OrderType.LIMIT,
            limit_price=Decimal("99"),
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert price == Decimal("99")
        assert post_only is True

    def test_a_stop_entry_is_not_converted(self) -> None:
        """A stop entry is triggered by price moving away; passive pricing contradicts it."""
        from quantflow.risk.engine import as_maker_entry

        order_type, _, post_only = as_maker_entry(
            OrderType.STOP_MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert order_type is OrderType.STOP_MARKET
        assert post_only is False

    def test_price_is_decimal(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        _, price, _ = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert isinstance(price, Decimal)


class TestExitsStayAggressive:
    def test_a_reduce_only_exit_is_never_post_only(self) -> None:
        """Protection that waits for a passive fill is not protection."""
        from quantflow.risk.engine import as_maker_entry

        order_type, _, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
            is_entry=False,
        )

        assert order_type is OrderType.MARKET
        assert post_only is False
