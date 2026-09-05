"""Take-profit distance, judged against what the trade costs to make.

A target is only worth reaching if getting there pays for the round trip several times
over. Measured on this account across sixteen live trades: the average winning move was
0.197% of notional against a round-trip cost of 0.11%. Winners were **1.8x** their own
execution cost, and at a 37.5% win rate that is a losing system by arithmetic rather than
by luck — gross edge +3.70, fees 37.43, net −33.73.

Nothing about the signals was wrong. The system was harvesting moves barely larger than the
fee charged to harvest them.

Two distances are computed and the wider wins:

* **The cost floor** — a multiple of the round-trip cost. Answers *is this worth the fee*.
* **The volatility floor** — a multiple of ATR. Answers *is this reachable*. A target
  inside a single bar's typical range is noise, and it is deliberately not a flat
  percentage: "far enough" is a different number on gold than on a meme coin, and one
  constant would be far too tight on one and absurd on the other.

A target already beyond both is never moved. If a strategy chose a wider level it knows
something this layer does not, and this only ever widens — it cannot pull a target closer,
and it never touches the stop.
"""

from __future__ import annotations

import os
from decimal import Decimal

from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide

#: How many times the round-trip cost a target must sit beyond entry.
#:
#: At 3x, cost is a third of the gross win rather than the 56% actually observed. Combined
#: with the measured 37.5% win rate that is the difference between a negative and a
#: marginally positive expectancy; below 3 the arithmetic does not clear.
MIN_TARGET_COST_MULTIPLE = Decimal("3")

#: How many ATRs a target should sit beyond entry when volatility is the binding constraint.
#:
#: One ATR is inside the noise of a single bar; two is a move the market has to actually
#: make, and is reachable within the holding periods this timeframe produces.
TARGET_ATR_MULTIPLE = Decimal("2")

#: The furthest a target may sit from entry, in ATRs. A CEILING, not a floor.
#:
#: This module could previously only widen. A strategy that proposed a 1.2% target in a
#: market whose ATR was 0.14% got that target unchallenged — 8.6 ATR away — and the expected
#: edge was then computed to it, reporting 1.09% on a move the market was not making.
#:
#: Measured on 2026-08-18 over 18 live trades, the maximum favourable excursion actually
#: reached, in ATRs: p25 0.47, median 1.16, p75 3.07, p90 4.39, max 7.50. Targets in the
#: audited sample sat at a median of 5.69 ATR — above the 90th percentile — and **0 of 14
#: were ever reached**. Nothing above roughly 3 ATR is a target; it is a wish.
#:
#: Set at the measured p75, so a target is placed where a quarter of trades historically
#: got to. That is deliberately not the median: a target the average trade reaches is a
#: target too close to pay for its own costs.
#:
#: The consequence is intended and is the point of the change. When volatility is too low
#: for a 3-ATR move to clear the cost gate, the expected edge falls below the floor and the
#: trade is refused — instead of a target being manufactured to make the arithmetic pass.
#: Override with QF_MAX_TARGET_ATR_MULTIPLE.
MAX_TARGET_ATR_MULTIPLE = Decimal(os.environ.get("QF_MAX_TARGET_ATR_MULTIPLE", "3"))


def cost_aware_target(
    *,
    side: OrderSide,
    entry: Decimal,
    target: Decimal | None,
    atr: Decimal | None,
    cost_rate: Decimal,
) -> Decimal | None:
    """Widen a take-profit until it is worth the cost of reaching it.

    Args:
        side: Which way the position faces. A long's target sits above entry, a short's
            below, and the floor is applied in the direction that means *further away*.
        entry: Reference entry price.
        target: The strategy's chosen target, or ``None`` if it set none.
        atr: Recent average true range, when known.
        cost_rate: Round-trip execution cost as a fraction of notional.

    Returns:
        The target to use — the strategy's own if it already clears both floors, otherwise
        the floor. ``None`` when the strategy set no target, because inventing one here
        would be making an exit decision that belongs to the strategy.

    """
    if target is None or entry <= ZERO:
        return target

    floor_distance = entry * cost_rate * MIN_TARGET_COST_MULTIPLE
    if atr is not None and atr > ZERO:
        floor_distance = max(floor_distance, atr * TARGET_ATR_MULTIPLE)

    distance = max(abs(target - entry), floor_distance)

    # The ceiling. Only applicable when ATR is known — without it there is no measure of
    # what this market can actually travel, and guessing one would be worse than the
    # unbounded behaviour it replaces.
    if atr is not None and atr > ZERO:
        distance = min(distance, atr * MAX_TARGET_ATR_MULTIPLE)

    if side is OrderSide.BUY:
        return entry + distance
    return entry - distance


__all__ = ["MIN_TARGET_COST_MULTIPLE", "TARGET_ATR_MULTIPLE", "cost_aware_target"]
