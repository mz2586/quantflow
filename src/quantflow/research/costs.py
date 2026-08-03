"""Trading cost assumptions for research runs.

A backtest is only evidence about the future if its costs resemble the ones that will
actually be paid. The two ways a research framework lies to itself are (a) omitting fees
entirely and (b) assuming a fill at the price that triggered the decision. Both are
corrected here, and both are made explicit rather than left to a default buried in an
engine.

The presets are deliberately **pessimistic**. A strategy that survives costs worse than
reality is a candidate; a strategy that only works at optimistic costs is a mirage, and
the difference between the two is the entire value of this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from quantflow.core.errors import ValidationError
from quantflow.exchange.simulator import FeeModel, FixedSlippage, SlippageModel, VolumeShareSlippage

#: Binance spot taker fee at the base VIP tier, paid on both legs of a round trip.
#: No BNB discount and no fee-tier improvement is assumed — both are privileges that can
#: be withdrawn, and a strategy that needs them is not robust.
BINANCE_SPOT_TAKER: Final = Decimal("0.001")

#: Binance spot maker fee at the base tier. Present for completeness; every strategy in
#: the library trades market orders, so the taker rate is what actually applies.
BINANCE_SPOT_MAKER: Final = Decimal("0.001")


@dataclass(frozen=True, slots=True)
class CostModel:
    """A named, fully explicit set of trading cost assumptions."""

    name: str
    description: str
    fees: FeeModel
    slippage: SlippageModel
    #: Human-readable summary, carried into the report so a reader never has to guess
    #: which assumptions produced the numbers they are looking at.
    summary: str

    def round_trip_cost_pct(self) -> Decimal:
        """Approximate fee cost of one round trip, as a fraction of notional.

        Fees only — slippage depends on order size relative to bar volume and cannot be
        reduced to a constant. Useful as a sanity anchor: a strategy whose average trade
        return is below this number cannot be profitable, whatever the backtest says.
        """
        taker = self.fees.taker_rate if self.fees.taker_rate is not None else BINANCE_SPOT_TAKER
        return taker * 2


def realistic() -> CostModel:
    """The default research cost model: Binance base-tier taker fees plus volume slippage.

    Slippage scales with the share of a bar's volume the order consumes, so a strategy is
    penalised for wanting size the market could not actually supply. A flat basis-point
    assumption would let a strategy trade an unlimited quantity at a fixed cost, which is
    the single most common way a backtest flatters an illiquid idea.
    """
    return CostModel(
        name="realistic",
        description="Binance spot base tier, market orders, volume-scaled slippage",
        fees=FeeModel(maker_rate=BINANCE_SPOT_MAKER, taker_rate=BINANCE_SPOT_TAKER),
        slippage=VolumeShareSlippage(),
        summary=(
            f"taker {BINANCE_SPOT_TAKER:.3%} / maker {BINANCE_SPOT_MAKER:.3%} per fill "
            f"({BINANCE_SPOT_TAKER * 2:.2%} per round trip), volume-scaled slippage, "
            "market orders filled at the next bar's open"
        ),
    )


def pessimistic() -> CostModel:
    """Double fees and a fixed 10 bp slippage floor on top.

    Used as a robustness check rather than a headline. A strategy whose ranking survives
    this is not living on the edge of its cost assumptions; one that collapses was never
    really profitable, it was profitable *at one specific cost estimate*.
    """
    doubled = BINANCE_SPOT_TAKER * 2
    return CostModel(
        name="pessimistic",
        description="Double fees plus a fixed 10 bp slippage floor",
        fees=FeeModel(maker_rate=doubled, taker_rate=doubled),
        slippage=FixedSlippage(rate=Decimal("0.001")),
        summary=(
            f"taker {doubled:.3%} per fill ({doubled * 2:.2%} per round trip), "
            "fixed 10 bp slippage — a stress test, not a forecast"
        ),
    )


def zero_cost() -> CostModel:
    """No fees, no slippage.

    Included **only** so a report can quantify how much of a strategy's edge is consumed
    by trading costs. It is never a basis for a decision: every number it produces is
    unachievable by construction.
    """
    return CostModel(
        name="zero_cost",
        description="No fees, no slippage — diagnostic only, never achievable",
        fees=FeeModel(maker_rate=Decimal("0"), taker_rate=Decimal("0")),
        slippage=FixedSlippage(rate=Decimal("0")),
        summary="no fees, no slippage — diagnostic baseline, not achievable in practice",
    )


#: Every preset, by name.
COST_MODELS: Final[dict[str, Callable[[], CostModel]]] = {
    "realistic": realistic,
    "pessimistic": pessimistic,
    "zero_cost": zero_cost,
}


def build_cost_model(name: str) -> CostModel:
    """Look up a cost preset by name.

    Raises:
        ValidationError: if the name is not a known preset.

    """
    factory = COST_MODELS.get(name)
    if factory is None:
        raise ValidationError(
            f"unknown cost model {name!r}; expected one of {sorted(COST_MODELS)}",
            field="cost_model",
        )
    return factory()
