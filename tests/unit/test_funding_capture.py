"""Market-neutral funding capture: the economics, before any of it is believed.

The trade is a delta-hedged pair — short perpetual, long spot, equal notional — held to
collect funding. Price direction cancels between the legs, so the only revenue is funding
and the only question is whether it clears the cost of putting the hedge on and taking it
off again.

That cost is unavoidable and paid twice. Two legs in and two legs out at Bybit's 0.06%
taker fee is ~0.24% of notional per round trip, against funding of roughly 0.01% per 8h
settlement. So a position must survive on the order of eight days of continuously
favourable funding merely to break even. These tests pin the arithmetic that decides it.

Sign convention, which is the thing most easily got backwards: a **positive** funding rate
means longs pay shorts. This strategy is short the perp, so positive rates are revenue and
negative rates are a bill.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.neutral.funding_capture import (
    FundingCaptureParams,
    funding_payment,
    round_trip_cost,
    simulate_funding_capture,
)

NOTIONAL = Decimal("10000")
TAKER = Decimal("0.0006")


class TestFundingPayment:
    def test_short_receives_when_the_rate_is_positive(self) -> None:
        """Positive rate: longs pay shorts, and this strategy is short the perp."""
        assert funding_payment(NOTIONAL, Decimal("0.0001")) == Decimal("1.0000")

    def test_short_pays_when_the_rate_is_negative(self) -> None:
        """A negative rate is a bill, not a smaller revenue."""
        assert funding_payment(NOTIONAL, Decimal("-0.0001")) == Decimal("-1.0000")

    def test_zero_rate_pays_nothing(self) -> None:
        assert funding_payment(NOTIONAL, Decimal("0")) == Decimal("0")

    def test_payment_is_decimal(self) -> None:
        assert isinstance(funding_payment(NOTIONAL, Decimal("0.0001")), Decimal)


class TestRoundTripCost:
    def test_charges_both_legs_in_and_out(self) -> None:
        """Four taker fills per round trip: perp+spot to open, perp+spot to close."""
        assert round_trip_cost(NOTIONAL, taker_fee=TAKER) == Decimal("24.0000")

    def test_cost_scales_with_notional(self) -> None:
        assert round_trip_cost(Decimal("20000"), taker_fee=TAKER) == Decimal("48.0000")

    def test_slippage_is_added_on_every_leg(self) -> None:
        """Slippage is a real cost of crossing, not an optional extra."""
        with_slip = round_trip_cost(NOTIONAL, taker_fee=TAKER, slippage_bps=Decimal("1"))

        assert with_slip > round_trip_cost(NOTIONAL, taker_fee=TAKER)


class TestSimulation:
    def test_a_single_settlement_cannot_pay_for_its_own_round_trip(self) -> None:
        """One 8h collection at 0.01% earns 1.00 against ~24.00 of cost."""
        result = simulate_funding_capture(
            [(Decimal("0.0001"), True)],
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert result.net_pnl < 0

    def test_a_long_favourable_run_can_clear_costs(self) -> None:
        """Enough consecutive positive settlements and the trade is genuinely profitable."""
        result = simulate_funding_capture(
            [(Decimal("0.0001"), True)] * 60,
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert result.net_pnl > 0

    def test_costs_are_charged_once_per_position_not_per_settlement(self) -> None:
        """Holding through a settlement is free; opening and closing is not."""
        held = simulate_funding_capture(
            [(Decimal("0.0001"), True)] * 10,
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert held.round_trips == 1

    def test_churn_is_punished(self) -> None:
        """Alternating in and out pays the round trip every time."""
        churned = simulate_funding_capture(
            [(Decimal("0.0001"), True), (Decimal("-0.0001"), False)] * 5,
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert churned.round_trips == 5

    def test_flat_periods_collect_nothing(self) -> None:
        result = simulate_funding_capture(
            [(Decimal("0.0001"), False)] * 10,
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert result.funding_collected == Decimal("0")

    def test_gross_funding_and_costs_reconcile_to_net(self) -> None:
        """No hidden term: net is exactly what was collected minus what was paid."""
        result = simulate_funding_capture(
            [(Decimal("0.0001"), True)] * 30,
            params=FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER),
        )

        assert result.net_pnl == result.funding_collected - result.costs_paid
