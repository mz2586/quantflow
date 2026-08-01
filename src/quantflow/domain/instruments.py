"""Symbols and instrument metadata."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Self

from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, round_price, round_quantity


@dataclass(frozen=True, slots=True, order=True)
class Symbol:
    """A trading pair, normalised to ``BASE/QUOTE``.

    Binance's REST API uses ``BTCUSDT`` while CCXT uses ``BTC/USDT``; we standardise on the
    slashed form internally and convert at the exchange boundary.
    """

    base: str
    quote: str

    def __post_init__(self) -> None:
        """Validate and normalise the components."""
        base, quote = self.base.strip().upper(), self.quote.strip().upper()
        if not base or not quote:
            raise ValidationError(
                f"symbol requires base and quote, got {self.base!r}/{self.quote!r}"
            )
        if base == quote:
            raise ValidationError(f"base and quote must differ, got {base}")
        if not base.isalnum() or not quote.isalnum():
            raise ValidationError(f"symbol components must be alphanumeric: {base}/{quote}")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "quote", quote)

    @classmethod
    def parse(cls, value: str | Symbol) -> Self | Symbol:
        """Parse ``"BTC/USDT"``, ``"BTC-USDT"``, ``"btc_usdt"`` or a known concatenation."""
        if isinstance(value, Symbol):
            return value
        raw = value.strip().upper()
        for separator in ("/", "-", "_", ":"):
            if separator in raw:
                base, _, quote = raw.partition(separator)
                return cls(base=base, quote=quote)
        # Concatenated form (Binance native): split on a known quote suffix.
        for suffix in KNOWN_QUOTE_ASSETS:
            if raw.endswith(suffix) and len(raw) > len(suffix):
                return cls(base=raw[: -len(suffix)], quote=suffix)
        raise ValidationError(
            f"cannot parse symbol {value!r}; use BASE/QUOTE form",
            value=value,
        )

    @property
    def slashed(self) -> str:
        """Canonical internal representation, e.g. ``BTC/USDT``."""
        return f"{self.base}/{self.quote}"

    @property
    def concatenated(self) -> str:
        """Binance native representation, e.g. ``BTCUSDT``."""
        return f"{self.base}{self.quote}"

    def __str__(self) -> str:
        return self.slashed


#: Ordered longest-first so ``BTCUSDT`` does not mis-split on a shorter suffix.
KNOWN_QUOTE_ASSETS: tuple[str, ...] = (
    "USDT",
    "USDC",
    "TUSD",
    "FDUSD",
    "BUSD",
    "USD",
    "EUR",
    "TRY",
    "BTC",
    "ETH",
    "BNB",
)


@dataclass(frozen=True, slots=True)
class Instrument:
    """Tradability rules for a symbol on a specific venue.

    Sourced from the exchange's own metadata (``GET /api/v3/exchangeInfo`` via CCXT
    ``load_markets``). Never hard-code these — Binance changes them.
    """

    symbol: Symbol
    market_type: MarketType = MarketType.SPOT
    price_tick: Decimal = Decimal("0.01")
    quantity_step: Decimal = Decimal("0.00001")
    min_quantity: Decimal = Decimal("0.00001")
    max_quantity: Decimal | None = None
    min_notional: Decimal = Decimal("10")
    max_notional: Decimal | None = None
    maker_fee: Decimal = Decimal("0.001")
    taker_fee: Decimal = Decimal("0.001")
    max_leverage: Decimal = Decimal("1")
    active: bool = True
    contract_size: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        """Validate the rule set for internal consistency."""
        if self.price_tick <= ZERO:
            raise ValidationError(f"price_tick must be positive for {self.symbol}")
        if self.quantity_step <= ZERO:
            raise ValidationError(f"quantity_step must be positive for {self.symbol}")
        if self.min_quantity < ZERO:
            raise ValidationError(f"min_quantity cannot be negative for {self.symbol}")
        if self.max_quantity is not None and self.max_quantity < self.min_quantity:
            raise ValidationError(f"max_quantity below min_quantity for {self.symbol}")
        if self.maker_fee < ZERO or self.taker_fee < ZERO:
            raise ValidationError(f"fees cannot be negative for {self.symbol}")
        if self.max_leverage < Decimal("1"):
            raise ValidationError(f"max_leverage must be >= 1 for {self.symbol}")

    def normalize_price(self, price: Decimal, *, side_is_buy: bool = True) -> Decimal:
        """Snap a price to the venue's tick grid, conservatively for the given side."""
        return round_price(price, self.price_tick, side_is_buy=side_is_buy)

    def normalize_quantity(self, quantity: Decimal) -> Decimal:
        """Snap a quantity down to the venue's lot grid."""
        return round_quantity(quantity, self.quantity_step)

    def notional(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Quote-currency value of ``quantity`` at ``price``."""
        return abs(quantity) * price * self.contract_size

    def validate_order(
        self, quantity: Decimal, price: Decimal, *, check_price_tick: bool = True
    ) -> None:
        """Assert an order satisfies every venue rule.

        Args:
            quantity: Order size.
            price: The order's price, or a reference price for notional checks.
            check_price_tick: Whether ``price`` must sit on the venue's tick grid. Pass
                ``False`` for a **market** order: it has no price of its own, so the
                reference is a mark, mid or last price that legitimately need not be
                tick-aligned. Validating it would reject perfectly valid market orders.

        Raises:
            ValidationError: with the specific rule that failed.

        """
        absolute = abs(quantity)
        if not self.active:
            raise ValidationError(f"{self.symbol} is not tradable", symbol=str(self.symbol))
        if absolute < self.min_quantity:
            raise ValidationError(
                f"quantity {absolute} below minimum {self.min_quantity} for {self.symbol}",
                symbol=str(self.symbol),
                rule="min_quantity",
            )
        if self.max_quantity is not None and absolute > self.max_quantity:
            raise ValidationError(
                f"quantity {absolute} above maximum {self.max_quantity} for {self.symbol}",
                symbol=str(self.symbol),
                rule="max_quantity",
            )
        if absolute % self.quantity_step != ZERO:
            raise ValidationError(
                f"quantity {absolute} is not a multiple of step {self.quantity_step}",
                symbol=str(self.symbol),
                rule="quantity_step",
            )
        if check_price_tick and price % self.price_tick != ZERO:
            raise ValidationError(
                f"price {price} is not a multiple of tick {self.price_tick}",
                symbol=str(self.symbol),
                rule="price_tick",
            )
        value = self.notional(quantity, price)
        if value < self.min_notional:
            raise ValidationError(
                f"notional {value} below minimum {self.min_notional} for {self.symbol}",
                symbol=str(self.symbol),
                rule="min_notional",
            )
        if self.max_notional is not None and value > self.max_notional:
            raise ValidationError(
                f"notional {value} above maximum {self.max_notional} for {self.symbol}",
                symbol=str(self.symbol),
                rule="max_notional",
            )

    def fee_rate(self, *, is_maker: bool) -> Decimal:
        """Fee rate for the given liquidity role."""
        return self.maker_fee if is_maker else self.taker_fee
