"""Targets must be wide enough that the trade is worth making.

The measured failure this exists to fix: over 16 live trades the average winning move was
0.197% of notional while the round trip cost 0.11%. Winners were 1.8x their own execution
cost, and at a 37.5% win rate that is a losing system by construction — gross edge +3.70
against fees of 37.43.

The fix is not a bigger position or a looser filter. It is refusing to take a trade whose
target sits so close that costs eat the win, and giving trades that clear that bar enough
room to actually pay for themselves.

Distance is set from ATR rather than a flat percentage, because "far enough to be worth
it" is a different price on gold than on a meme coin, and a single fixed number would be
too tight on one and absurd on the other.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.enums import OrderSide
from quantflow.risk.targets import (
    MAX_TARGET_ATR_MULTIPLE,
    MIN_TARGET_COST_MULTIPLE,
    cost_aware_target,
)

ENTRY = Decimal("100")
#: 0.11% round trip, the taker cost actually observed on Bybit.
COST = Decimal("0.0011")


class TestTargetsClearTheirOwnCost:
    def test_a_target_inside_the_cost_floor_is_widened(self) -> None:
        # 0.2% away: the exact shape of the losing trades. Cost is 55% of the win.
        near = ENTRY * Decimal("1.002")

        widened = cost_aware_target(
            side=OrderSide.BUY, entry=ENTRY, target=near, atr=None, cost_rate=COST
        )

        assert widened is not None
        distance = widened - ENTRY
        assert distance >= ENTRY * COST * MIN_TARGET_COST_MULTIPLE

    def test_a_target_already_beyond_the_floor_is_untouched(self) -> None:
        # The strategy knows something this layer does not; it is never overridden downward.
        far = ENTRY * Decimal("1.05")

        assert (
            cost_aware_target(side=OrderSide.BUY, entry=ENTRY, target=far, atr=None, cost_rate=COST)
            == far
        )

    def test_a_short_target_is_widened_downward(self) -> None:
        near = ENTRY * Decimal("0.998")

        widened = cost_aware_target(
            side=OrderSide.SELL, entry=ENTRY, target=near, atr=None, cost_rate=COST
        )

        assert widened is not None
        assert widened < near, "a short's target must move further below entry, not above"
        assert ENTRY - widened >= ENTRY * COST * MIN_TARGET_COST_MULTIPLE

    def test_volatility_widens_the_target_beyond_the_cost_floor(self) -> None:
        """A volatile market needs more room than the cost floor alone provides.

        The floor answers "is this worth the fee". ATR answers "is this reachable". A
        target inside a single bar's typical range is noise, and one flat percentage across
        every asset would be far too tight on a meme coin and absurd on gold.
        """
        quiet = cost_aware_target(
            side=OrderSide.BUY,
            entry=ENTRY,
            target=ENTRY * Decimal("1.001"),
            atr=Decimal("0.05"),
            cost_rate=COST,
        )
        volatile = cost_aware_target(
            side=OrderSide.BUY,
            entry=ENTRY,
            target=ENTRY * Decimal("1.001"),
            atr=Decimal("2.00"),
            cost_rate=COST,
        )

        assert quiet is not None
        assert volatile is not None
        assert volatile > quiet, "a wider ATR must produce a wider target"

    def test_no_target_stays_absent(self) -> None:
        # A strategy that set no target is not given one here; that is an exit decision.
        assert (
            cost_aware_target(
                side=OrderSide.BUY, entry=ENTRY, target=None, atr=None, cost_rate=COST
            )
            is None
        )

    def test_a_zero_or_negative_entry_is_left_alone(self) -> None:
        # Never compute a distance from a price that cannot be real.
        target = ENTRY * Decimal("1.002")
        assert (
            cost_aware_target(
                side=OrderSide.BUY, entry=Decimal("0"), target=target, atr=None, cost_rate=COST
            )
            == target
        )

    def test_the_result_is_decimal(self) -> None:
        widened = cost_aware_target(
            side=OrderSide.BUY,
            entry=ENTRY,
            target=ENTRY * Decimal("1.001"),
            atr=Decimal("1"),
            cost_rate=COST,
        )
        assert isinstance(widened, Decimal)


class TestVolatilityCeiling:
    """A target must be somewhere the market can actually go."""

    def test_an_unreachable_strategy_target_is_pulled_back(self) -> None:
        # The audited case: a 1.2% target proposed in a market whose ATR was 0.14% — 8.6
        # ATR away. Across 14 audited trades, targets sat at a median 5.69 ATR and NOT ONE
        # was reached.
        entry = Decimal("64300")
        atr = Decimal("90.02")  # 0.14% of entry
        target = cost_aware_target(
            side=OrderSide.BUY,
            entry=entry,
            target=entry * Decimal("1.012"),
            atr=atr,
            cost_rate=Decimal("0.0011"),
        )
        assert target is not None
        assert (target - entry) / atr == MAX_TARGET_ATR_MULTIPLE

    def test_the_ceiling_applies_to_shorts_symmetrically(self) -> None:
        entry = Decimal("64300")
        atr = Decimal("90.02")
        target = cost_aware_target(
            side=OrderSide.SELL,
            entry=entry,
            target=entry * Decimal("0.988"),
            atr=atr,
            cost_rate=Decimal("0.0011"),
        )
        assert target is not None
        assert (entry - target) / atr == MAX_TARGET_ATR_MULTIPLE

    def test_a_target_inside_the_ceiling_is_untouched_by_it(self) -> None:
        # The cost floor still widens; the ceiling only binds when it is exceeded.
        entry = Decimal("64300")
        atr = Decimal("300")
        target = cost_aware_target(
            side=OrderSide.BUY,
            entry=entry,
            target=entry * Decimal("1.003"),
            atr=atr,
            cost_rate=Decimal("0.0011"),
        )
        assert target is not None
        assert (target - entry) / atr < MAX_TARGET_ATR_MULTIPLE

    def test_without_atr_the_ceiling_cannot_apply(self) -> None:
        # No ATR means no measure of what this market travels. Guessing a ceiling would be
        # worse than the unbounded behaviour it replaces.
        entry = Decimal("64300")
        wanted = entry * Decimal("1.012")
        target = cost_aware_target(
            side=OrderSide.BUY, entry=entry, target=wanted, atr=None, cost_rate=Decimal("0.0011")
        )
        assert target == wanted

    def test_no_target_still_means_no_target(self) -> None:
        assert (
            cost_aware_target(
                side=OrderSide.BUY,
                entry=Decimal("64300"),
                target=None,
                atr=Decimal("90"),
                cost_rate=Decimal("0.0011"),
            )
            is None
        )
