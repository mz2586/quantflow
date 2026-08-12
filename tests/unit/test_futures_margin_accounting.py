"""Phase 6: futures accounting is margin + mark-to-market, not spot cash math.

The defect: `PortfolioManager` subtracted the full notional on a BUY and credited it on a
SELL. On a linear perp you post margin and settle PnL — you do not buy an asset, and a short
brings in no cash. Equity, and therefore every equity-derived limit (position size, exposure,
drawdown, daily/weekly loss, leverage), was fictional on a futures account.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.config import MarketType
from quantflow.domain.enums import OrderSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.portfolio.manager import PortfolioManager
from tests.conftest import REFERENCE_TIME

BTC = Symbol.parse("BTC/USDT")
START = Decimal("10000")


def fill(side: OrderSide, *, quantity: str, price: str, fee: str = "0", seq: int = 1) -> Fill:
    return Fill(
        fill_id=f"f{seq}",
        order_id=f"o{seq}",
        symbol=BTC,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency="USDT",
        timestamp=REFERENCE_TIME,
    )


def futures(leverage: str = "1") -> PortfolioManager:
    return PortfolioManager(
        starting_equity=START, market_type=MarketType.FUTURE, leverage=Decimal(leverage)
    )


def spot() -> PortfolioManager:
    return PortfolioManager(starting_equity=START, market_type=MarketType.SPOT)


class TestAShortDoesNotInflateCash:
    def test_opening_a_short_credits_no_cash(self) -> None:
        """The headline defect: on spot math a short read as income."""
        book = futures()
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000"))
        assert book.cash == START, "a perp short must not create cash"

    def test_spot_still_credits_a_sale(self) -> None:
        """The old behaviour is correct for spot and must be untouched."""
        book = spot()
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000"))
        assert book.cash == START + Decimal("5000")

    def test_opening_a_long_does_not_spend_the_notional(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        assert book.cash == START, "margin is reserved, not spent"

    def test_only_fees_move_cash_while_a_position_is_open(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000", fee="3"))
        assert book.cash == START - Decimal("3")


class TestMarginIsReserved:
    def test_margin_is_posted_against_the_position(self) -> None:
        book = futures(leverage="1")
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        assert book.margin_posted == Decimal("5000")

    def test_leverage_reduces_the_margin_required(self) -> None:
        book = futures(leverage="10")
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        assert book.margin_posted == Decimal("500")

    def test_margin_is_released_when_the_position_closes(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000", seq=1))
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000", seq=2))
        assert book.margin_posted == Decimal("0")

    def test_a_partial_close_releases_proportional_margin(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000", seq=1))
        book.apply_fill(fill(OrderSide.SELL, quantity="0.05", price="50000", seq=2))
        assert book.margin_posted == Decimal("2500")


class TestEquityIsMarginBased:
    def test_equity_is_unchanged_by_merely_opening_a_position(self) -> None:
        """Opening a perp moves no value; only PnL and fees do."""
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("50000"))
        assert book.equity() == START

    def test_equity_tracks_unrealised_pnl_not_notional(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("51000"))
        # 0.1 BTC up 1000 = +100. Spot math would have reported ~15,100.
        assert book.equity() == START + Decimal("100")

    def test_a_short_gains_when_price_falls(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("49000"))
        assert book.equity() == START + Decimal("100")

    def test_realised_pnl_lands_in_cash_on_close(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000", seq=1))
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="51000", seq=2))
        assert book.cash == START + Decimal("100")
        assert book.equity() == START + Decimal("100")

    def test_spot_equity_still_counts_the_asset(self) -> None:
        book = spot()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("50000"))
        assert book.equity() == START


class TestEveryLimitReadsMarginEquity:
    def test_the_snapshot_carries_the_market_type(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("50000"))
        snapshot = book.snapshot(REFERENCE_TIME)
        assert snapshot.market_type is MarketType.FUTURE
        assert snapshot.margin_posted == Decimal("5000")

    def test_snapshot_equity_matches_the_manager(self) -> None:
        """Risk rules read the snapshot; it must not disagree with the book."""
        book = futures()
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("49000"))
        assert book.snapshot(REFERENCE_TIME).equity == book.equity()

    def test_drawdown_is_measured_against_margin_equity(self) -> None:
        """With spot math a losing short showed a profit, so drawdown read as zero."""
        book = futures()
        book.apply_fill(fill(OrderSide.SELL, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("55000"))  # short is 500 down
        snapshot = book.snapshot(REFERENCE_TIME)
        assert snapshot.equity == START - Decimal("500")
        assert snapshot.drawdown_pct > Decimal("0")

    def test_free_margin_excludes_what_is_reserved(self) -> None:
        book = futures()
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("50000"))
        snapshot = book.snapshot(REFERENCE_TIME)
        assert snapshot.free_margin == START - Decimal("5000")

    def test_leverage_is_computed_from_margin_equity(self) -> None:
        book = futures(leverage="10")
        book.apply_fill(fill(OrderSide.BUY, quantity="0.1", price="50000"))
        book.update_mark_price(BTC, Decimal("50000"))
        snapshot = book.snapshot(REFERENCE_TIME)
        # 5,000 notional against 10,000 equity.
        assert snapshot.leverage == Decimal("0.5")


class TestVenueAccountIsTheLiveSource:
    """Live equity comes from the venue, not from a reconstruction of fills."""

    def test_unified_account_totals_are_preferred(self) -> None:
        from quantflow.live.reconcile import parse_venue_account

        account = parse_venue_account(
            {
                "info": {
                    "result": {
                        "list": [
                            {
                                "totalEquity": "165451.40496807",
                                "totalAvailableBalance": "99934.38509407",
                                "totalInitialMargin": "1200.5",
                            }
                        ]
                    }
                }
            },
            [],
        )
        assert account.equity == Decimal("165451.40496807")
        assert account.available == Decimal("99934.38509407")
        assert account.margin_posted == Decimal("1200.5")

    def test_it_falls_back_to_the_quote_balance(self) -> None:
        from quantflow.live.reconcile import parse_venue_account

        account = parse_venue_account({"info": {}, "USDT": {"total": "5000"}}, [])
        assert account.equity == Decimal("5000")

    def test_a_drifting_local_equity_is_detected(self) -> None:
        """The point of holding both: disagreement must be visible, not averaged away."""
        from quantflow.live.reconcile import VenueAccount

        account = VenueAccount(
            equity=Decimal("10000"),
            available=Decimal("9000"),
            margin_posted=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
        )
        assert account.matches(Decimal("10000.4"), tolerance=Decimal("1"))
        assert not account.matches(Decimal("10500"), tolerance=Decimal("1"))

    def test_money_stays_decimal(self) -> None:
        from quantflow.live.reconcile import parse_venue_account

        account = parse_venue_account({"info": {"totalEquity": "1234.5678"}}, [])
        assert isinstance(account.equity, Decimal)
        assert isinstance(account.available, Decimal)
        assert isinstance(account.margin_posted, Decimal)
