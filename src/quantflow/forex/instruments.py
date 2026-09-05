"""FX instrument metadata.

An FX instrument is not a crypto pair with a different name. Size is quoted in *lots* of a
venue-defined ``contract_size``, not in units of the base asset; the price grid is a
``point`` derived from ``digits``, and a "pip" is ten points on the 3- and 5-digit quotes
every modern venue uses; the money value of a price move comes from ``tick_value``, which
already folds in the contract size *and* the account-currency conversion. Reusing the
crypto quantity formula here produces a position roughly 100,000x the intended size.

Nothing in this module imports a venue. Adapters build :class:`ForexInstrument` from
whatever their API returns; the domain layer only ever sees this shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Final

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, quantize_down
from quantflow.domain.enums import OrderSide
from quantflow.forex.sessions import DAYS_IN_WEEK, SessionWindow, require_weekday

#: The seven conventional major pairs, in the order we want them prioritised.
#:
#: This tuple is a **ranking**, never a source of instruments. Symbols are always
#: discovered from the venue — a name appearing here does not mean it is tradable, and a
#: venue symbol missing from here is still perfectly tradable, just ranked after the
#: majors. See :func:`prioritise_symbols`.
MAJORS: Final[tuple[str, ...]] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
)

#: Wednesday, in Python's ``weekday()`` convention, is the usual triple-swap day.
DEFAULT_TRIPLE_SWAP_WEEKDAY: Final = 2

#: Quote precisions on which one pip is ten points (the "fractional pip" quotes).
_FRACTIONAL_PIP_DIGITS: Final = frozenset({3, 5})

_MAJOR_RANK: Final[dict[str, int]] = {symbol: rank for rank, symbol in enumerate(MAJORS)}


class TradeMode(StrEnum):
    """What the venue currently permits on a symbol."""

    DISABLED = "disabled"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    CLOSE_ONLY = "close_only"
    FULL = "full"

    @classmethod
    def from_mt5(cls, code: int) -> TradeMode:
        """Map an MT5 ``SYMBOL_TRADE_MODE_*`` code, defaulting unknown codes to disabled.

        Defaulting *closed* rather than open is deliberate: an unrecognised mode must never
        be read as permission to trade.
        """
        return _MT5_TRADE_MODES.get(code, TradeMode.DISABLED)

    def allows(self, side: OrderSide) -> bool:
        """Whether a new position may be *opened* on ``side``."""
        if self is TradeMode.FULL:
            return True
        if self is TradeMode.LONG_ONLY:
            return side is OrderSide.BUY
        if self is TradeMode.SHORT_ONLY:
            return side is OrderSide.SELL
        return False


_MT5_TRADE_MODES: Final[dict[int, TradeMode]] = {
    0: TradeMode.DISABLED,
    1: TradeMode.LONG_ONLY,
    2: TradeMode.SHORT_ONLY,
    3: TradeMode.CLOSE_ONLY,
    4: TradeMode.FULL,
}


def mt5_weekday_to_python(day: int) -> int:
    """Convert an MT5 ``ENUM_DAY_OF_WEEK`` (0 = Sunday) to Python's (0 = Monday).

    The off-by-one between these two conventions silently moves the triple-swap charge to
    the wrong day, which is why it is converted once, here, and asserted in tests.
    """
    if not 0 <= day < DAYS_IN_WEEK:
        raise ValidationError(f"MT5 weekday must be 0-6, got {day}", weekday=day)
    return (day - 1) % DAYS_IN_WEEK


def python_weekday_to_mt5(day: int) -> int:
    """Convert Python's ``weekday()`` (0 = Monday) to MT5's ``ENUM_DAY_OF_WEEK``."""
    require_weekday(day)
    return (day + 1) % DAYS_IN_WEEK


def normalise_symbol(symbol: str) -> str:
    """Strip venue decoration so ``EURUSD+``, ``eurusd.raw`` and ``EUR/USD`` all match.

    Everything from the first ``.`` is treated as a venue suffix and dropped; any remaining
    punctuation is stripped. Used only for ranking and lookup — the venue's own spelling is
    what gets sent back to the venue, so normalisation never rewrites the symbol we trade.
    """
    stem = symbol.split(".", 1)[0]
    return "".join(character for character in stem if character.isalnum()).upper()


def prioritise_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    """Order discovered symbols majors-first, then alphabetically.

    Purely a reordering of what was passed in. Duplicates collapse; nothing is invented,
    so an empty input yields an empty result rather than the majors.
    """
    unique = list(dict.fromkeys(symbols))
    majors = [s for s in unique if normalise_symbol(s) in _MAJOR_RANK]
    others = [s for s in unique if normalise_symbol(s) not in _MAJOR_RANK]
    majors.sort(key=lambda s: _MAJOR_RANK[normalise_symbol(s)])
    others.sort()
    return tuple(majors + others)


@dataclass(frozen=True, slots=True)
class ForexInstrument:
    """Everything needed to size, cost and route an order on one FX symbol.

    All numeric fields are :class:`~decimal.Decimal`. ``margin_rate`` of zero means the
    venue did not tell us, in which case :attr:`leverage` is ``None`` rather than a guess.
    """

    symbol: str
    base: str
    quote: str
    contract_size: Decimal
    min_lot: Decimal
    max_lot: Decimal
    lot_step: Decimal
    digits: int
    point: Decimal
    tick_size: Decimal
    tick_value: Decimal
    margin_rate: Decimal = ZERO
    trade_mode: TradeMode = TradeMode.FULL
    sessions: tuple[SessionWindow, ...] = ()
    spread_points: Decimal = ZERO
    commission_per_lot: Decimal = ZERO
    swap_long: Decimal = ZERO
    swap_short: Decimal = ZERO
    triple_swap_weekday: int = DEFAULT_TRIPLE_SWAP_WEEKDAY
    tradable: bool = True
    venue: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Reject metadata that would make sizing or costing meaningless."""
        if not self.symbol.strip():
            raise ValidationError("symbol must not be blank")
        for name in ("contract_size", "min_lot", "max_lot", "lot_step", "point", "tick_size"):
            value: Decimal = getattr(self, name)
            if value <= ZERO:
                raise ValidationError(f"{name} must be positive", symbol=self.symbol, field=name)
        if self.min_lot > self.max_lot:
            raise ValidationError(
                "min_lot must not exceed max_lot", symbol=self.symbol, field="min_lot"
            )
        if self.digits < 0:
            raise ValidationError("digits must not be negative", symbol=self.symbol)
        if self.tick_value < ZERO:
            raise ValidationError("tick_value must not be negative", symbol=self.symbol)
        if not ZERO <= self.margin_rate <= Decimal("1"):
            raise ValidationError(
                "margin_rate must be a fraction between 0 and 1",
                symbol=self.symbol,
                margin_rate=str(self.margin_rate),
            )
        require_weekday(self.triple_swap_weekday, field="triple_swap_weekday")

    # ------------------------------------------------------------------ pip maths
    @property
    def is_jpy_quoted(self) -> bool:
        """Whether the quote currency is JPY, which prices to 2/3 digits, not 4/5."""
        return self.quote.upper() == "JPY"

    @property
    def points_per_pip(self) -> Decimal:
        """How many points make one pip on this quote precision."""
        return Decimal("10") if self.digits in _FRACTIONAL_PIP_DIGITS else Decimal("1")

    @property
    def pip_size(self) -> Decimal:
        """The price increment of one pip."""
        return self.point * self.points_per_pip

    @property
    def value_per_point_per_lot(self) -> Decimal:
        """Account-currency value of a one-point move on one lot.

        ``tick_value`` is the venue's own answer and already carries the account-currency
        conversion, so it is preferred. When the venue reports zero — some feeds omit it
        for unselected symbols — this falls back to ``contract_size * point``, which is
        correct only when the account currency *is* the quote currency; adapters should
        prefer supplying a real tick value.
        """
        if self.tick_value > ZERO:
            return self.tick_value * self.point / self.tick_size
        return self.contract_size * self.point

    @property
    def pip_value_per_lot(self) -> Decimal:
        """Account-currency value of a one-pip move on one lot."""
        return self.value_per_point_per_lot * self.points_per_pip

    @property
    def leverage(self) -> Decimal | None:
        """Leverage implied by ``margin_rate``, or ``None`` when the venue did not say."""
        if self.margin_rate <= ZERO:
            return None
        return Decimal("1") / self.margin_rate

    # ------------------------------------------------------------ price conversions
    def price_to_points(self, price_delta: Decimal) -> Decimal:
        """Convert an absolute price distance into points."""
        return abs(price_delta) / self.point

    def points_to_price(self, points: Decimal) -> Decimal:
        """Convert a point distance into a price distance."""
        return points * self.point

    def round_price(self, price: Decimal) -> Decimal:
        """Snap a price onto the venue's tick grid."""
        return quantize_down(price, self.tick_size)

    def notional(self, lots: Decimal, price: Decimal) -> Decimal:
        """Base-currency notional of ``lots`` at ``price``."""
        return lots * self.contract_size * price

    # ------------------------------------------------------------------- lot grid
    def quantise_lots(self, lots: Decimal) -> Decimal:
        """Snap a lot size onto the venue's volume grid, downward.

        The grid is anchored on ``min_lot``, not on zero: venues that quote a minimum that
        is not itself a multiple of the step (0.03 min on a 0.02 step) will reject an order
        placed on the zero-anchored grid. Anything below ``min_lot`` returns zero — callers
        must treat that as a rejection, not as a size.
        """
        if lots < self.min_lot:
            return ZERO
        capped = min(lots, self.max_lot)
        steps = quantize_down(capped - self.min_lot, self.lot_step)
        return self.min_lot + steps

    def can_trade(self, side: OrderSide) -> bool:
        """Whether a new position may be opened on ``side`` right now."""
        return self.tradable and self.trade_mode.allows(side)
