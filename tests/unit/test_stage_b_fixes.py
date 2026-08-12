"""Stage B permanent fixes: order-ack parsing, min-notional sizing, venue reconciliation.

Each covers a defect found by running against the live demo venue, not by reading code.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.core.config import RiskSettings
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.exchange.bybit.mapping import parse_order
from quantflow.live.reconcile import (
    VenuePosition,
    parse_venue_positions,
    reconcile,
)
from quantflow.risk.sizing import FixedFractionalSizer, SizingRequest

BTC = Symbol.parse("BTC/USDT")


def instrument(
    *, min_quantity: str = "0.01", step: str = "0.01", min_notional: str = "5"
) -> Instrument:
    return Instrument(
        symbol=BTC,
        price_tick=Decimal("0.1"),
        quantity_step=Decimal(step),
        min_quantity=Decimal(min_quantity),
        min_notional=Decimal(min_notional),
    )


class TestOrderAckParsing:
    """Fix 1 — a V5 ack carries only orderId/orderLinkId, never an amount."""

    def test_an_ack_without_an_amount_does_not_raise(self) -> None:
        """This raised before, reporting failure for an order the venue had accepted."""
        ack = {
            "id": "venue-abc",
            "clientOrderId": "qf-1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "info": {"orderId": "venue-abc", "orderLinkId": "qf-1"},
        }
        order = parse_order(ack, local_order_id="local-1", fallback_quantity=Decimal("0.01"))
        assert order.quantity == Decimal("0.01")
        assert order.venue_order_id == "venue-abc"

    def test_a_venue_reported_amount_wins_over_the_fallback(self) -> None:
        """The fallback is a stand-in, never an override."""
        ack = {"id": "v", "symbol": "BTC/USDT", "side": "buy", "type": "market", "amount": 0.05}
        order = parse_order(ack, local_order_id="l", fallback_quantity=Decimal("0.01"))
        assert order.quantity == Decimal("0.05")

    def test_without_a_fallback_a_zero_amount_still_raises(self) -> None:
        """The invariant is intact where there is genuinely nothing to fall back to."""
        from quantflow.core.errors import ValidationError

        ack = {"id": "v", "symbol": "BTC/USDT", "side": "buy", "type": "market"}
        with pytest.raises(ValidationError, match="quantity must be positive"):
            parse_order(ack, local_order_id="l")

    def test_ack_id_detection_accepts_the_v5_result_shape(self) -> None:
        from quantflow.exchange.bybit.rest import BybitGateway

        assert BybitGateway._ack_has_order_id({"id": "x"})
        assert BybitGateway._ack_has_order_id({"info": {"result": {"orderId": "x"}}})
        assert BybitGateway._ack_has_order_id({"info": {"orderLinkId": "qf-1"}})
        assert not BybitGateway._ack_has_order_id({"info": {}})
        assert not BybitGateway._ack_has_order_id(None)


class TestMinNotionalSizing:
    """Fix 2 — a sub-minimum size must round up or skip, never submit zero."""

    def _request(self, *, equity: str, price: str, inst: Instrument) -> SizingRequest:
        return SizingRequest(
            equity=Decimal(equity),
            price=Decimal(price),
            instrument=inst,
            stop_loss_price=Decimal(price) * Decimal("0.98"),
            available_cash=Decimal(equity),
        )

    def test_a_sub_minimum_size_rounds_up_when_the_caps_allow_it(self) -> None:
        # Equity 1000 at 1% risk over a 1200 stop distance sizes to ~0.008 BTC - under the
        # 0.01 lot floor. The 0.01 minimum is 600 notional, which these caps permit.
        settings = RiskSettings(
            max_position_pct=Decimal("0.8"),
            max_total_exposure_pct=Decimal("0.9"),
            max_order_notional=Decimal("5000"),
        )
        sizer = FixedFractionalSizer(settings)
        result = sizer.size(self._request(equity="1000", price="60000", inst=instrument()))
        assert result.quantity == Decimal("0.01"), "rounded up to the venue lot minimum"
        assert result.quantity > ZERO_DECIMAL

    def test_it_skips_when_the_minimum_would_breach_a_cap(self) -> None:
        """Honouring the venue floor must never breach our own ceiling."""
        # The same 0.01 minimum is 600 notional, far beyond a 1-per-position cap.
        settings = RiskSettings(
            max_position_pct=Decimal("0.001"),
            max_total_exposure_pct=Decimal("0.01"),
            max_order_notional=Decimal("100"),
            min_order_notional=Decimal("5"),
        )
        sizer = FixedFractionalSizer(settings)
        result = sizer.size(self._request(equity="1000", price="60000", inst=instrument()))
        assert result.quantity == ZERO_DECIMAL
        assert result.capped_by == "below_venue_min_quantity"

    def test_a_skip_is_never_a_zero_quantity_order(self) -> None:
        settings = RiskSettings(
            max_position_pct=Decimal("0.001"), max_total_exposure_pct=Decimal("0.01")
        )
        sizer = FixedFractionalSizer(settings)
        result = sizer.size(self._request(equity="1000", price="60000", inst=instrument()))
        assert result.quantity == ZERO_DECIMAL
        assert result.capped_by is not None, "a skip must carry a reason"


ZERO_DECIMAL = Decimal("0")


class TestVenueReconciliation:
    """Check (d) — a restart must adopt what the venue holds."""

    def test_a_live_position_is_parsed_from_the_venue_payload(self) -> None:
        positions = parse_venue_positions(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "info": {
                        "size": "0.01",
                        "side": "Buy",
                        "avgPrice": "63000",
                        "stopLoss": "60000",
                    },
                }
            ]
        )
        assert len(positions) == 1
        assert positions[0].symbol == BTC
        assert positions[0].quantity == Decimal("0.01")
        assert positions[0].is_protected

    def test_flat_entries_are_ignored(self) -> None:
        """Bybit lists flat symbols; treating them as positions invents drift."""
        assert parse_venue_positions([{"symbol": "BTC/USDT:USDT", "info": {"size": "0"}}]) == []

    def test_an_unknown_venue_position_is_reported(self) -> None:
        """The exact acceptance case: the bot must notice what it does not know about."""
        venue = [VenuePosition(BTC, "buy", Decimal("0.01"), Decimal("63000"), Decimal("60000"))]
        report = reconcile(venue, known_symbols=set())
        assert report.unknown_locally
        assert not report.is_clean

    def test_a_known_protected_position_reconciles_cleanly(self) -> None:
        venue = [VenuePosition(BTC, "buy", Decimal("0.01"), Decimal("63000"), Decimal("60000"))]
        report = reconcile(venue, known_symbols={BTC})
        assert report.is_clean
        assert report.is_safe_to_trade

    def test_an_unprotected_position_blocks_trading(self) -> None:
        """A live position with no stop is the disqualifying case."""
        venue = [VenuePosition(BTC, "buy", Decimal("0.01"), Decimal("63000"), None)]
        report = reconcile(venue, known_symbols={BTC})
        assert report.unprotected
        assert not report.is_safe_to_trade

    def test_the_summary_states_the_drift(self) -> None:
        venue = [VenuePosition(BTC, "buy", Decimal("0.01"), Decimal("63000"), None)]
        assert "unprotected=1" in reconcile(venue, known_symbols=set()).summary()
