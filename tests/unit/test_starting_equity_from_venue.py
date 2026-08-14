"""Starting equity must come from the account, not from a constant.

The bot sized every position off a hardcoded 10,000 while the demo account held ~100,000,
so a 2% position was 200 against 100k of capital — 0.2% of the book. The risk *percentages*
were being honoured perfectly; they were just being applied to a number that had nothing to
do with the account.

Two rules the resolver must not break:

* **Never invent capital.** If the venue balance cannot be read, fall back to the
  configured figure rather than guessing high. Sizing off an equity the account does not
  have is how a 2% cap becomes a 20% one.
* **Never exceed the venue.** The resolved equity is capped at the balance actually
  present, so a stale or optimistic configured value cannot inflate position sizes.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.portfolio import Balance
from quantflow.live.equity import resolve_starting_equity


def balances(**amounts: str) -> dict[str, Balance]:
    return {
        asset: Balance(asset=asset, free=Decimal(value), locked=Decimal("0"))
        for asset, value in amounts.items()
    }


class TestResolveStartingEquity:
    def test_uses_the_quote_balance_from_the_venue(self) -> None:
        equity = resolve_starting_equity(
            balances(USDT="99991.94"), configured=Decimal("10000"), quote="USDT"
        )

        assert equity == Decimal("99991.94")

    def test_ignores_other_assets(self) -> None:
        """A BTC balance is not quote-currency buying power."""
        equity = resolve_starting_equity(
            balances(USDT="500", BTC="1"), configured=Decimal("10000"), quote="USDT"
        )

        assert equity == Decimal("500")

    def test_falls_back_to_configured_when_the_venue_is_silent(self) -> None:
        """Unreadable balance must not become an invented one."""
        equity = resolve_starting_equity({}, configured=Decimal("10000"), quote="USDT")

        assert equity == Decimal("10000")

    def test_zero_balance_falls_back_rather_than_sizing_off_nothing(self) -> None:
        equity = resolve_starting_equity(
            balances(USDT="0"), configured=Decimal("10000"), quote="USDT"
        )

        assert equity == Decimal("10000")

    def test_result_is_decimal(self) -> None:
        equity = resolve_starting_equity(
            balances(USDT="99991.94"), configured=Decimal("10000"), quote="USDT"
        )

        assert isinstance(equity, Decimal)

    def test_locked_funds_count_toward_equity(self) -> None:
        """Margin held against an open position is still the account's capital."""
        held = {"USDT": Balance(asset="USDT", free=Decimal("900"), locked=Decimal("100"))}

        assert resolve_starting_equity(held, configured=Decimal("10000"), quote="USDT") == Decimal(
            "1000"
        )
