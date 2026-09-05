"""Position sizing.

Sizing lives in the risk layer, never in a strategy. A strategy expresses *what* it wants
to do and how strongly; how much capital that translates into is a risk decision, and
keeping the boundary sharp is what makes it impossible for a strategy to size its way
around a position limit.

Every sizer answers the same question: given equity, a price and a stop, how many units can
we buy such that the loss at the stop stays inside the configured risk budget?
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from quantflow.core.config import RiskSettings
from quantflow.core.errors import ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO, quantize_up, safe_divide
from quantflow.domain.instruments import Instrument

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Inputs to a sizing decision."""

    equity: Decimal
    price: Decimal
    instrument: Instrument
    stop_loss_price: Decimal | None = None
    conviction: Decimal = ONE
    #: Notional already committed on this symbol — open position plus resting entries.
    #:
    #: Subtracted from the position cap so a second leg is sized into the room that is
    #: LEFT, not given the whole allowance again. Without this, pyramiding is impossible
    #: rather than merely limited: leg one takes ~17.6% of a 20% cap, leg two is sized to
    #: another ~17.6%, and the aggregate check refuses it every time. The ceiling does not
    #: move; the legs share it.
    committed_notional: Decimal = ZERO
    available_cash: Decimal | None = None
    volatility: Decimal | None = None
    """ATR in price units, for volatility-targeting sizers."""

    def __post_init__(self) -> None:
        """Validate the request."""
        if self.equity <= ZERO:
            raise ValidationError(f"equity must be positive for sizing, got {self.equity}")
        if self.price <= ZERO:
            raise ValidationError(f"price must be positive for sizing, got {self.price}")
        if not (ZERO <= self.conviction <= ONE):
            raise ValidationError(f"conviction must be in [0, 1], got {self.conviction}")

    @property
    def stop_distance(self) -> Decimal | None:
        """Absolute distance from entry to stop, if a stop was supplied."""
        if self.stop_loss_price is None:
            return None
        return abs(self.price - self.stop_loss_price)

    @property
    def cash(self) -> Decimal:
        """Cash available, defaulting to full equity."""
        return self.available_cash if self.available_cash is not None else self.equity


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of a sizing decision."""

    quantity: Decimal
    notional: Decimal
    risk_amount: Decimal
    """Quote-currency loss if the stop is hit."""
    method: str
    capped_by: str | None = None
    """Which constraint bound the size, if any. Surfaced so an operator can see *why*
    a position came out smaller than expected."""
    detail: str | None = None
    """The numbers behind a refusal, in words. ``capped_by`` names the rule; this says
    which values tripped it, so a rejection can be diagnosed from the log line alone
    instead of being re-derived from the venue afterwards."""

    @property
    def is_tradable(self) -> bool:
        """Whether the result is a size worth submitting."""
        return self.quantity > ZERO


class PositionSizer(ABC):
    """Base class for sizing strategies."""

    name: str = "base"

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    @abstractmethod
    def _raw_quantity(self, request: SizingRequest) -> tuple[Decimal, str]:
        """Compute the unconstrained quantity and a description of the method."""

    def size(self, request: SizingRequest) -> SizingResult:
        """Compute a final, venue-legal, risk-capped quantity.

        The pipeline is: raw size → hard caps → conviction scaling → venue lot rounding →
        minimum-viability check. Caps are applied *before* rounding so rounding can only
        ever reduce the position, never push it back over a limit.

        Conviction attenuates *after* the caps, and the order matters. Scaling first made
        conviction inert: the raw risk-based size is routinely two to three times
        ``max_position_pct`` allows, so the clamp that followed discarded the scaling and
        every tier landed on the identical capped quantity. Measured on the live account —
        equity 49,608, cap 9,921 — WEAK and VERY_STRONG both produced 16.35 BNB. Because
        conviction is a fraction in ``[0, 1]`` it can only ever reduce, so applying it last
        preserves every cap invariant while letting the full range of the gradient reach the
        order.
        """
        raw, method = self._raw_quantity(request)
        if raw <= ZERO:
            return SizingResult(ZERO, ZERO, ZERO, method, capped_by="non_positive_raw_size")

        allowed, cap_reason = self._apply_caps(raw, request)
        capped = allowed * request.conviction
        # Conviction is the binding constraint only when it cut below what the caps allowed.
        reason = "conviction" if capped < allowed else cap_reason
        instrument = request.instrument
        quantity = instrument.normalize_quantity(capped)

        floor = self._venue_floor_quantity(instrument, request.price)
        if quantity < floor:
            # Rounding down has taken the order under the venue's smallest legal order.
            # Rounding UP to it is allowed only when the larger position still respects
            # every hard cap - otherwise honouring the venue's floor would breach our own
            # ceiling, which is not a trade worth making. Skipping is the correct outcome,
            # and it is logged rather than silently dropped.
            bumped = floor
            bumped_notional = instrument.notional(bumped, request.price)
            # Net of what the symbol already holds, for the same reason _apply_caps is:
            # otherwise the bump-to-venue-minimum path can round a leg UP past a cap the
            # sizer had just correctly reduced it below.
            max_position_value = (
                request.equity * self.settings.max_position_pct - request.committed_notional
            )
            if (
                bumped_notional <= max_position_value
                and bumped_notional <= self.settings.max_order_notional
            ):
                logger.info(
                    "sizing.rounded_up_to_venue_minimum",
                    symbol=str(instrument.symbol),
                    requested=str(quantity),
                    minimum=str(bumped),
                    notional=str(bumped_notional),
                )
                quantity = bumped
            else:
                logger.info(
                    "sizing.skipped_below_venue_minimum",
                    symbol=str(instrument.symbol),
                    requested=str(quantity),
                    minimum=str(bumped),
                    minimum_notional=str(bumped_notional),
                    max_position_value=str(max_position_value),
                    max_order_notional=str(self.settings.max_order_notional),
                    reason="the venue minimum would breach a position or order cap",
                )
                return SizingResult(
                    ZERO,
                    ZERO,
                    ZERO,
                    method,
                    capped_by="below_venue_min_quantity",
                    detail=(
                        f"the venue's smallest {instrument.symbol} lot is {bumped} "
                        f"(worth {bumped_notional} at {request.price}), which exceeds the "
                        f"{max_position_value} allowed by max_position_pct on equity "
                        f"{request.equity} or the {self.settings.max_order_notional} "
                        f"max_order_notional"
                    ),
                )

        notional = instrument.notional(quantity, request.price)
        if notional < instrument.min_notional or notional < self.settings.min_order_notional:
            logger.info(
                "sizing.skipped_below_min_notional",
                symbol=str(instrument.symbol),
                quantity=str(quantity),
                notional=str(notional),
                venue_minimum=str(instrument.min_notional),
                configured_minimum=str(self.settings.min_order_notional),
            )
            return SizingResult(
                ZERO,
                ZERO,
                ZERO,
                method,
                capped_by="below_min_notional",
                detail=(
                    f"{quantity} {instrument.symbol} is worth {notional} at {request.price}, "
                    f"under the venue minimum {instrument.min_notional} / configured minimum "
                    f"{self.settings.min_order_notional}"
                ),
            )

        # Last gate before the size leaves the risk layer. Every rule above works on one
        # constraint at a time; this re-checks the finished number against the venue's whole
        # rule set, so a quantity that is legal by each step but illegal overall is refused
        # here rather than by the exchange.
        try:
            instrument.validate_order(quantity, request.price, check_price_tick=False)
        except ValidationError as exc:
            logger.warning(
                "sizing.rejected_by_venue_rules",
                symbol=str(instrument.symbol),
                quantity=str(quantity),
                error=exc.message,
            )
            return SizingResult(
                ZERO, ZERO, ZERO, method, capped_by="violates_venue_rules", detail=exc.message
            )

        distance = request.stop_distance
        risk_amount = quantity * distance if distance is not None else ZERO

        return SizingResult(
            quantity=quantity,
            notional=notional,
            risk_amount=risk_amount,
            method=method,
            capped_by=reason,
        )

    @staticmethod
    def _venue_floor_quantity(instrument: Instrument, price: Decimal) -> Decimal:
        """The smallest step-aligned quantity the venue will actually accept at ``price``.

        A venue states two independent floors and satisfying one does not satisfy the
        other. Bybit's ``XAUUSDT`` has a minimum quantity of 0.001 and a minimum notional
        of 5 USDT; with gold near 3,400 that smallest lot is worth 3.40, so the lot minimum
        is *below* the notional minimum and an order sized to it is rejected. Bumping to
        ``min_quantity`` alone therefore produced a quantity that passed the lot check and
        then failed the notional check immediately afterwards, and the sizer returned zero
        for a market it could have traded perfectly well at 0.002.

        Taking the maximum of the two floors - with the notional-derived one rounded *up*
        onto the lot grid, since rounding it down would land back under the minimum it was
        computed to clear - yields the true smallest legal order. On instruments where the
        lot minimum already dominates, which is most of them, this returns exactly
        ``min_quantity`` and nothing changes.

        The result is not asserted to be tradable: it is still checked against every
        position and order cap by the caller, which may well refuse it.
        """
        step = instrument.quantity_step
        floor = instrument.min_quantity
        unit_value = price * instrument.contract_size
        if unit_value > ZERO and instrument.min_notional > ZERO:
            needed = quantize_up(instrument.min_notional / unit_value, step)
            floor = max(floor, needed)
        return floor

    def _apply_caps(self, quantity: Decimal, request: SizingRequest) -> tuple[Decimal, str | None]:
        """Clamp a quantity to every configured hard cap."""
        price = request.price
        candidates: list[tuple[Decimal, str]] = [(quantity, "")]

        max_position_value = (
            request.equity * self.settings.max_position_pct - request.committed_notional
        )
        if max_position_value <= ZERO:
            return ZERO, "max_position_pct"
        candidates.append((max_position_value / price, "max_position_pct"))

        candidates.append((self.settings.max_order_notional / price, "max_order_notional"))

        # Leverage is capped by whichever of us and the venue permits less. The venue's
        # ceiling genuinely varies by instrument - Bybit allows 100x on metals and crypto,
        # 50x on large-cap single names and 25x on the quieter index ETFs - so a configured
        # limit that happens to exceed it would size an order the venue rejects outright.
        # In practice the configured limit is the tighter of the two and this changes
        # nothing; it exists so that stops being true loudly rather than at the exchange.
        leverage = min(self.settings.max_leverage, request.instrument.max_leverage)

        # Cash is the binding constraint on spot: you cannot buy what you cannot pay for.
        if leverage <= ONE:
            candidates.append((request.cash / price, "available_cash"))
        else:
            candidates.append(((request.cash * leverage) / price, "max_leverage"))

        if request.instrument.max_quantity is not None:
            candidates.append((request.instrument.max_quantity, "venue_max_quantity"))

        smallest, reason = min(candidates, key=lambda item: item[0])
        return smallest, (reason or None)


class FixedFractionalSizer(PositionSizer):
    """Risk a fixed fraction of equity per trade, measured to the stop.

    The industry default, and the only method that keeps loss-per-trade constant across
    instruments of wildly different volatility. Requires a stop: without one, "risk 1% of
    equity" has no defined meaning, so the sizer refuses rather than silently guessing.
    """

    name = "fixed_fractional"

    def __init__(self, settings: RiskSettings, *, risk_per_trade: Decimal | None = None) -> None:
        super().__init__(settings)
        self.risk_per_trade = (
            risk_per_trade if risk_per_trade is not None else settings.max_daily_loss_pct / 3
        )
        if not (ZERO < self.risk_per_trade <= ONE):
            raise ValidationError(f"risk_per_trade must be in (0, 1], got {self.risk_per_trade}")

    def _raw_quantity(self, request: SizingRequest) -> tuple[Decimal, str]:
        distance = request.stop_distance
        if distance is None or distance <= ZERO:
            raise ValidationError(
                "fixed fractional sizing requires a stop loss; "
                "risk per trade is undefined without one"
            )
        budget = request.equity * self.risk_per_trade
        return safe_divide(budget, distance), self.name


class VolatilityTargetSizer(PositionSizer):
    """Size so each position contributes a similar volatility to the portfolio.

    Uses ATR rather than the stop distance, so two positions with the same stop but very
    different underlying volatility do not contribute equally different risk.
    """

    name = "volatility_target"

    def __init__(
        self,
        settings: RiskSettings,
        *,
        target_volatility_pct: Decimal = Decimal("0.01"),
    ) -> None:
        super().__init__(settings)
        if not (ZERO < target_volatility_pct <= ONE):
            raise ValidationError(
                f"target_volatility_pct must be in (0, 1], got {target_volatility_pct}"
            )
        self.target_volatility_pct = target_volatility_pct

    def _raw_quantity(self, request: SizingRequest) -> tuple[Decimal, str]:
        volatility = request.volatility
        if volatility is None or volatility <= ZERO:
            raise ValidationError("volatility targeting requires an ATR estimate", method=self.name)
        budget = request.equity * self.target_volatility_pct
        return safe_divide(budget, volatility), self.name


class FixedNotionalSizer(PositionSizer):
    """Allocate a fixed fraction of equity as notional, ignoring the stop.

    Simple and predictable, but loss per trade varies with the stop distance. Useful as a
    baseline when comparing sizing methods, and as a fallback when no stop is available.
    """

    name = "fixed_notional"

    def __init__(self, settings: RiskSettings, *, allocation_pct: Decimal | None = None) -> None:
        super().__init__(settings)
        self.allocation_pct = (
            allocation_pct if allocation_pct is not None else settings.max_position_pct
        )
        if not (ZERO < self.allocation_pct <= ONE):
            raise ValidationError(f"allocation_pct must be in (0, 1], got {self.allocation_pct}")

    def _raw_quantity(self, request: SizingRequest) -> tuple[Decimal, str]:
        return safe_divide(request.equity * self.allocation_pct, request.price), self.name


def build_sizer(settings: RiskSettings, method: str = "fixed_fractional") -> PositionSizer:
    """Construct a sizer by name.

    Raises:
        ValidationError: for an unknown method, listing the valid options.

    """
    sizers: dict[str, type[PositionSizer]] = {
        FixedFractionalSizer.name: FixedFractionalSizer,
        VolatilityTargetSizer.name: VolatilityTargetSizer,
        FixedNotionalSizer.name: FixedNotionalSizer,
    }
    sizer_class = sizers.get(method)
    if sizer_class is None:
        raise ValidationError(
            f"unknown sizing method {method!r}; available: {', '.join(sorted(sizers))}"
        )
    return sizer_class(settings)
