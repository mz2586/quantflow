"""The multi-asset universe: what the venue lists beyond crypto, and what is worth touching.

Bybit's linear-perpetual category is not only crypto. The same USDT-settled contract
machinery carries gold, silver, WTI, Brent, single-name equities and index ETFs, and
``/v5/market/instruments-info`` labels each one in a ``symbolType`` field that CCXT passes
through verbatim in ``market["info"]``. On the demo host that field partitions the category
into ``""`` (crypto), ``innovation`` (crypto, newer listings), ``stock`` and ``commodity``.

This module answers the same two questions :mod:`quantflow.universe.meme` answers, and
draws the line between them in the same place.

**Which class a market belongs to** is *mostly* discovered. ``symbolType`` is the venue's
own statement and is used as the primary key, which is the important difference from
:mod:`~quantflow.universe.meme`: that module has to hand-maintain its membership because
Bybit tags nothing as a meme coin. Two refinements are needed on top of the venue's label,
and both are honestly curated rather than derived:

* ``commodity`` covers metals and energy together, and they are not the same instrument.
  Gold and crude share a tag, not a driver — one is a monetary asset, the other an
  industrial one with a physical delivery curve behind it. :data:`METAL_ROOTS` splits them.
* ``stock`` covers operating companies and index ETFs together. ``SPY`` is tagged exactly
  as ``AAPL`` is, and an index is a portfolio: it cannot gap on an earnings miss, and its
  realised volatility is the diversified residue of its constituents'.
  :data:`INDEX_ETF_ROOTS` splits them, and is a hand-maintained list of tickers.

**Whether a listed market is tradable right now** is entirely measured, exactly as it is
for memes, and for the same reason: a listing is not liquidity. What changes across classes
is not the *shape* of the test but the numbers it is run against, because the same
threshold means opposite things in two markets:

* :mod:`~quantflow.universe.meme` requires a bar to move at least 0.4% before it will
  trade. Gold's typical 15m bar moves 0.18% and SPY's moves 0.04%. Applied unchanged, the
  meme floor does not filter these markets, it deletes them — and it would also delete BTC.
* Conversely a 10% single-bar move is a routine afternoon for a meme coin and a
  once-a-decade event in Brent. A breaker calibrated to the first cannot protect the second.

So the limits are per class, the reasoning for each is recorded on the field, and the four
rejection groups — liquidity, freshness, regime, order — are the same four in the same
order as :func:`quantflow.universe.meme.assess_eligibility`, returning that module's
:class:`~quantflow.universe.meme.EligibilityVerdict` rather than a second verdict type.

Nothing here fetches anything. Every measurement arrives as a plain value the caller took
at a known instant, because a module that can reach for data is a module that can reach
*forward* for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum

from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.universe.meme import EligibilityVerdict, is_meme, strip_multiplier

#: The quote asset the multi-asset universe is built against.
#:
#: Every non-crypto perpetual Bybit lists is USDT-settled, so this is a statement of fact
#: about the venue rather than a preference. It is declared anyway, because the eligibility
#: numbers below are all denominated in it.
ASSET_QUOTE_ASSET = "USDT"


class AssetClass(StrEnum):
    """What kind of thing a market is, for the purposes of risk and strategy selection.

    A :class:`~enum.StrEnum` so a class survives a round trip through a log line, a JSON
    payload or an env var without a codec. The values are the strings that appear in logs.

    The classes are distinguished by what *drives* them, not by what venue lists them.
    Every member here trades as an ordinary USDT linear perpetual through the same gateway,
    the same sizing and the same reconciliation — the distinction exists because the price
    process differs, and so must the thresholds applied to it.
    """

    #: Major and mid-cap digital assets. The venue's untagged and ``innovation`` markets.
    CRYPTO = "crypto"
    #: Meme coins, as classified by :mod:`quantflow.universe.meme`'s curated roots.
    MEME = "meme"
    #: Precious metals: gold, silver, and anything else on an ISO 4217 ``X``-prefixed root.
    METAL = "metal"
    #: Energy: crude oil benchmarks and their relatives.
    ENERGY = "energy"
    #: Single-name equities — operating companies with idiosyncratic, event-driven risk.
    EQUITY = "equity"
    #: Index and sector ETFs — diversified baskets, including leveraged ones.
    INDEX = "index"


#: The venue's ``symbolType`` for markets it considers commodities.
VENUE_TYPE_COMMODITY = "commodity"
#: The venue's ``symbolType`` for markets it considers equities, ETFs included.
VENUE_TYPE_STOCK = "stock"

#: Precious-metal roots, hand-maintained.
#:
#: The venue tags gold and crude alike as ``commodity``; nothing in the payload separates a
#: monetary metal from a barrel of oil. These are the ISO 4217 codes for precious metals,
#: which is why the fallback in :func:`classify_asset_class` treats *any* unknown
#: ``X``-prefixed commodity root as a metal: that prefix is reserved by the standard for
#: exactly this, so a future ``XPT`` listing classifies correctly without an edit here.
#:
#: Listing the four explicitly anyway keeps the common case readable and independent of
#: that inference.
METAL_ROOTS: frozenset[str] = frozenset({"XAU", "XAG", "XPT", "XPD"})

#: Energy roots, hand-maintained.
#:
#: ``CL`` is WTI and ``BZ`` is Brent, following the futures tickers Bybit borrowed. The
#: others are listed in advance of any listing, on the same principle as
#: :data:`~quantflow.universe.meme.MEME_BASE_ASSETS`: an entry the venue does not list
#: costs nothing, while a missing entry silently routes an energy contract through the
#: wrong volatility band.
ENERGY_ROOTS: frozenset[str] = frozenset({"CL", "BZ", "NG", "RB", "HO", "WTI", "BRENT"})

#: Index and sector ETF roots, hand-maintained — and genuinely a curated opinion.
#:
#: This is the one place in the module where the venue is no help at all: Bybit tags
#: ``SPY`` and ``AAPL`` identically as ``stock``, and no field distinguishes a fund from a
#: company. The list is therefore reviewed by a human and described as such, exactly as
#: :data:`~quantflow.universe.meme.MEME_BASE_ASSETS` is.
#:
#: Leveraged and inverse ETFs (``TQQQ``, ``SQQQ``, ``SOXL``, ``SOXS``, ``KORU``) are
#: included deliberately. They are still baskets — they carry no single-name earnings
#: risk — and their daily-reset leverage shows up as volatility, which the eligibility
#: band already measures and bounds. Excluding them would leave the index class populated
#: only by instruments too quiet to cover a round trip on this timeframe.
#:
#: Misclassifying an ETF as an equity is the safe direction of error: it lands in the
#: stricter volatility band and simply competes as a single name. The reverse — a company
#: treated as a diversified basket — is the one to avoid, so entries are added only when
#: the ticker is known to be a fund.
INDEX_ETF_ROOTS: frozenset[str] = frozenset(
    {
        # Broad market
        "SPY",
        "VOO",
        "VTI",
        "QQQ",
        "DIA",
        "IWM",
        # International and country
        "EEM",
        "EFA",
        "EWY",
        "EWJ",
        "EWZ",
        "FXI",
        "KWEB",
        # Sector
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "SMH",
        "ARKK",
        # Leveraged and inverse
        "TQQQ",
        "SQQQ",
        "SOXL",
        "SOXS",
        "SPXL",
        "SPXS",
        "UPRO",
        "UDOW",
        "SDOW",
        "TNA",
        "TZA",
        "LABU",
        "LABD",
        "YINN",
        "YANG",
        "KORU",
    }
)


def classify_asset_class(symbol: Symbol, venue_symbol_type: str) -> AssetClass:
    """Decide which :class:`AssetClass` a market belongs to.

    The venue's ``symbolType`` is the primary key and is trusted where it is decisive.
    Refinement happens only inside the two tags that conflate genuinely different
    instruments, using the curated root sets above:

    * ``commodity`` -> :attr:`~AssetClass.METAL` or :attr:`~AssetClass.ENERGY`
    * ``stock``     -> :attr:`~AssetClass.INDEX` or :attr:`~AssetClass.EQUITY`
    * anything else -> :attr:`~AssetClass.MEME` or :attr:`~AssetClass.CRYPTO`, deferring to
      :func:`quantflow.universe.meme.is_meme` so there is exactly one meme definition in
      the codebase rather than a second one that can drift from it.

    The crypto branch is the default rather than an error case on purpose. An unrecognised
    or newly-invented ``symbolType`` lands in :attr:`~AssetClass.CRYPTO`, which is the
    conservative outcome: the crypto band is the one calibrated for the widest range of
    behaviour, and a market that does not belong there will fail the measured eligibility
    checks rather than trade under thresholds nobody chose for it.

    The base asset is de-prefixed first, so a hypothetical ``1000``-multiplier listing of
    any of these classifies on its root rather than on the venue's basket convention.
    """
    root, _ = strip_multiplier(symbol.base)
    tag = venue_symbol_type.strip().lower()

    if tag == VENUE_TYPE_COMMODITY:
        if root in ENERGY_ROOTS:
            return AssetClass.ENERGY
        # ISO 4217 reserves the X prefix for precious metals; see METAL_ROOTS.
        if root in METAL_ROOTS or root.startswith("X"):
            return AssetClass.METAL
        return AssetClass.ENERGY

    if tag == VENUE_TYPE_STOCK:
        return AssetClass.INDEX if root in INDEX_ETF_ROOTS else AssetClass.EQUITY

    return AssetClass.MEME if is_meme(symbol) else AssetClass.CRYPTO


#: Classes that are not crypto, i.e. everything this module exists to add.
NON_CRYPTO_CLASSES: frozenset[AssetClass] = frozenset(
    {AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX}
)


@dataclass(frozen=True, slots=True)
class AssetMarket:
    """One listed, classified market plus the venue rules that constrain an order.

    A flattened projection of :class:`~quantflow.domain.instruments.Instrument`, carrying
    only what an eligibility, sizing or cost decision reads, with the resolved
    :class:`AssetClass` alongside. Flattened for the same reason
    :class:`~quantflow.universe.meme.MemeMarket` is: the class is the property most easily
    forgotten, and it sits at the same level as the tick and the step rather than one
    dereference away.

    Fees and leverage come from the venue rather than from a per-class constant. Bybit
    charges 1bp maker / 6bp taker on these perpetuals against 10bp/10bp on spot, and caps
    leverage at 100x on metals, 50x on large-cap single names and 25x on the quieter index
    ETFs. Assuming a crypto fee schedule here would misprice every gate downstream.
    """

    symbol: Symbol
    asset_class: AssetClass
    base_root: str
    multiplier: Decimal
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    max_quantity: Decimal | None
    min_notional: Decimal
    contract_size: Decimal
    maker_fee: Decimal
    taker_fee: Decimal
    max_leverage: Decimal
    #: Minutes between funding settlements, as the venue reports them. 240 on metals, 480
    #: on energy and equities — a 2x difference in the funding cost of the same holding
    #: period, which is why it is carried per market rather than assumed to be 8 hours.
    funding_interval_minutes: int | None
    active: bool
    market_type: MarketType


def build_asset_market(instrument: Instrument) -> AssetMarket:
    """Project one venue instrument into a classified :class:`AssetMarket`."""
    root, multiplier = strip_multiplier(instrument.symbol.base)
    return AssetMarket(
        symbol=instrument.symbol,
        asset_class=classify_asset_class(instrument.symbol, instrument.venue_symbol_type),
        base_root=root,
        multiplier=multiplier,
        price_tick=instrument.price_tick,
        quantity_step=instrument.quantity_step,
        min_quantity=instrument.min_quantity,
        max_quantity=instrument.max_quantity,
        min_notional=instrument.min_notional,
        contract_size=instrument.contract_size,
        maker_fee=instrument.maker_fee,
        taker_fee=instrument.taker_fee,
        max_leverage=instrument.max_leverage,
        funding_interval_minutes=instrument.funding_interval_minutes,
        active=instrument.active,
        market_type=instrument.market_type,
    )


def discover_asset_universe(
    instruments: Iterable[Instrument],
    *,
    classes: Iterable[AssetClass] | None = None,
) -> list[AssetMarket]:
    """Filter a venue's instrument list down to classified, listed, active markets.

    Three filters, all cheap and all load-bearing: the quote must be
    :data:`ASSET_QUOTE_ASSET`, the venue must report the market active, and the resolved
    class must be one of ``classes`` (every class, when ``None``).

    Inactive markets are dropped here rather than at order time. A delisted instrument
    still returns candles and a stale ticker, so a caller that only checks for the presence
    of data will happily size a position in a market that will reject it.

    Sorted by symbol, so two runs over the same venue snapshot produce the same universe in
    the same order. Map iteration order is not a promise any exchange client makes, and a
    universe that reshuffles is a universe whose logs cannot be diffed.
    """
    wanted = frozenset(classes) if classes is not None else None
    markets: list[AssetMarket] = []
    for instrument in instruments:
        if instrument.symbol.quote != ASSET_QUOTE_ASSET or not instrument.active:
            continue
        market = build_asset_market(instrument)
        if wanted is not None and market.asset_class not in wanted:
            continue
        markets.append(market)
    return sorted(markets, key=lambda market: market.symbol)


# --------------------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------------------

#: Basis points in one whole unit, for spread arithmetic.
_BPS = Decimal("10000")

#: Venue taker fee on these perpetuals, used to *derive* the volatility floors below.
#:
#: 6bp a side, so 12bp round trip. This is not a threshold anything is rejected on — it is
#: the number the ``min_volatility`` defaults are computed from, recorded here so the
#: derivation is visible rather than implied.
PERP_TAKER_FEE_BPS = Decimal("6")


@dataclass(frozen=True, slots=True)
class AssetEligibilityLimits:
    """The thresholds a market must clear before a position is allowed in it.

    Field-for-field the same shape as
    :class:`~quantflow.universe.meme.EligibilityLimits`, because the *questions* do not
    change across asset classes — only the answers do. See :data:`LIMITS_BY_CLASS` for the
    per-class values and the reasoning behind each departure from the meme defaults.

    The test any default has to pass is unchanged: "can this number be defended without
    naming an instrument". A limit derived from round-trip cost or from the order size the
    risk engine actually produces generalises to the next listing; one reverse-engineered
    from last month's gold chart does not.
    """

    #: Minimum 24h quote volume, in USDT.
    min_quote_volume_24h: Decimal
    #: Maximum bid/ask spread, in basis points of the mid.
    max_spread_bps: Decimal
    #: Maximum age of the quote used for the decision.
    max_ticker_age: timedelta
    #: Maximum age of the most recent closed bar.
    max_candle_age: timedelta
    #: Minimum per-bar volatility, as a fraction of price.
    min_volatility: Decimal
    #: Maximum per-bar volatility, as a fraction of price.
    max_volatility: Decimal
    #: Flash-move breaker: maximum last-bar range as a multiple of the typical bar range.
    max_bar_range_multiple: Decimal
    #: Flash-move breaker: maximum absolute single-bar return, as a fraction.
    max_abs_bar_return: Decimal
    #: Maximum share of the recent bar's quote volume one order may take.
    max_bar_volume_share: Decimal
    #: Minimum stop distance, in venue price ticks.
    min_stop_ticks: Decimal


#: The 24h quote-volume floor applied to every non-crypto class, in USDT.
#:
#: Derived from the order the risk engine actually produces, not chosen for roundness.
#: Working backwards: at a ~50k account and a 5% per-position cap the largest order is
#: ~2,500 USDT; :attr:`~AssetEligibilityLimits.max_bar_volume_share` allows an order to be
#: at most 2% of a bar's traded value, so a bar must carry ~125,000 USDT; a 24h session is
#: 96 bars of 15 minutes. 125,000 x 96 is ~12M, and 10M is that figure with the rounding
#: taken off rather than added, since the bar-volume ceiling re-checks the same constraint
#: against the *actual* order at decision time and is the binding test either way.
#:
#: This is what excludes most of the venue's 193 listed single names: liquidity in a
#: synthetic equity perpetual is a small fraction of the liquidity in the share itself, and
#: a floor that ignored that would admit markets whose every order moves the price.
NON_CRYPTO_MIN_QUOTE_VOLUME_24H = Decimal("10000000")

#: Per-class thresholds. Every departure from the meme defaults is justified on the field.
LIMITS_BY_CLASS: Mapping[AssetClass, AssetEligibilityLimits] = {
    # ---------------------------------------------------------------- metals
    AssetClass.METAL: AssetEligibilityLimits(
        min_quote_volume_24h=NON_CRYPTO_MIN_QUOTE_VOLUME_24H,
        # Same 10bp ceiling as memes, and derived the same way: a taker round trip costs
        # 12bp, so a 10bp spread crossed twice would already triple the cost of the trade.
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        # 0.12% = exactly the 12bp round-trip taker fee. The average bar must at minimum
        # cover the cost of the round trip, or the strategy is buying lottery tickets with
        # a guaranteed fee. Gold's typical 15m bar runs ~0.18% and silver's ~0.30%, so this
        # admits both while still rejecting a market that has genuinely stopped moving.
        # The meme floor of 0.4% would reject gold, silver, crude, Brent and BTC alike.
        min_volatility=PERP_TAKER_FEE_BPS * 2 / _BPS,
        # 3%. A 3% move in gold inside 15 minutes is a monetary-policy shock, not a regime
        # a 15m strategy was fitted to. Far below the meme ceiling of 6% because the tail
        # being guarded against is different in kind: a meme can triple, gold cannot.
        max_volatility=Decimal("0.03"),
        max_bar_range_multiple=Decimal("4"),
        # 4% in a single bar. The absolute complement to the relative multiple, set an
        # order of magnitude tighter than the meme breaker's 10% because a bar that large
        # in a monetary metal means the macro assumption underneath every open position
        # has just changed.
        max_abs_bar_return=Decimal("0.04"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
    # ---------------------------------------------------------------- energy
    AssetClass.ENERGY: AssetEligibilityLimits(
        min_quote_volume_24h=NON_CRYPTO_MIN_QUOTE_VOLUME_24H,
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        min_volatility=PERP_TAKER_FEE_BPS * 2 / _BPS,
        # 4%, one point wider than metals. Crude carries a supply shock that gold does not:
        # an OPEC headline or a strait closure repricing the front month several percent is
        # a recurring feature of the instrument rather than a break in it.
        max_volatility=Decimal("0.04"),
        max_bar_range_multiple=Decimal("4"),
        max_abs_bar_return=Decimal("0.05"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
    # -------------------------------------------------------------- equities
    AssetClass.EQUITY: AssetEligibilityLimits(
        min_quote_volume_24h=NON_CRYPTO_MIN_QUOTE_VOLUME_24H,
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        min_volatility=PERP_TAKER_FEE_BPS * 2 / _BPS,
        # 5%. Single names carry idiosyncratic event risk that an index diversifies away —
        # an earnings miss, a guidance cut, a halt. Wider than a metal's ceiling because
        # that is the instrument behaving normally, and still bounded because a stop
        # outside a 5% bar cannot be funded at a usable size.
        max_volatility=Decimal("0.05"),
        max_bar_range_multiple=Decimal("4"),
        # 6% in one bar. An equity perpetual tracks an underlying that closes overnight and
        # at weekends, so the perpetual absorbs the gap in a single bar when the cash
        # market reopens. That is a price the book never printed, and a stop cannot protect
        # against it — the correct response is to stand aside until the range normalises.
        max_abs_bar_return=Decimal("0.06"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
    # --------------------------------------------------------------- indices
    AssetClass.INDEX: AssetEligibilityLimits(
        min_quote_volume_24h=NON_CRYPTO_MIN_QUOTE_VOLUME_24H,
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        min_volatility=PERP_TAKER_FEE_BPS * 2 / _BPS,
        # 5%, matching equities rather than metals, because the class deliberately includes
        # 3x leveraged ETFs. A 5% bar in SOXL is a 1.7% bar in the semiconductor index
        # underneath it — unusual, not broken. An unleveraged index would never approach
        # this ceiling, so it costs nothing to admit the leveraged ones.
        max_volatility=Decimal("0.05"),
        max_bar_range_multiple=Decimal("4"),
        max_abs_bar_return=Decimal("0.06"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
    # ---------------------------------------------------------------- crypto
    AssetClass.CRYPTO: AssetEligibilityLimits(
        # 5M, the meme floor. Crypto majors clear it by three orders of magnitude; the
        # floor exists for the long tail of the 504 untagged listings, not for BTC.
        min_quote_volume_24h=Decimal("5000000"),
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        # Same round-trip derivation as every other class. Note this is a third of the meme
        # floor: BTC's typical 15m bar is ~0.20% and would fail a 0.4% test, which is the
        # clearest evidence that the meme numbers are meme numbers and not general ones.
        min_volatility=PERP_TAKER_FEE_BPS * 2 / _BPS,
        max_volatility=Decimal("0.06"),
        max_bar_range_multiple=Decimal("4"),
        max_abs_bar_return=Decimal("0.10"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
    # ------------------------------------------------------------------ meme
    # The meme band, restated here so every class resolves through one table. The values
    # are deliberately identical to quantflow.universe.meme.EligibilityLimits: memes are
    # still filtered by that module in the live path, and two tables that disagree about
    # the same market would be worse than either of them alone.
    AssetClass.MEME: AssetEligibilityLimits(
        min_quote_volume_24h=Decimal("5000000"),
        max_spread_bps=Decimal("10"),
        max_ticker_age=timedelta(seconds=30),
        max_candle_age=timedelta(minutes=30),
        min_volatility=Decimal("0.004"),
        max_volatility=Decimal("0.06"),
        max_bar_range_multiple=Decimal("4"),
        max_abs_bar_return=Decimal("0.10"),
        max_bar_volume_share=Decimal("0.02"),
        min_stop_ticks=Decimal("10"),
    ),
}


def limits_for(asset_class: AssetClass) -> AssetEligibilityLimits:
    """The thresholds for one asset class.

    Raises rather than falling back to a default set. A class with no declared limits is a
    class nobody has thought about, and silently trading it under someone else's numbers is
    exactly the failure this table exists to prevent.
    """
    limits = LIMITS_BY_CLASS.get(asset_class)
    if limits is None:  # pragma: no cover - unreachable while the table is complete
        raise ValidationError(f"no eligibility limits declared for asset class {asset_class}")
    return limits


@dataclass(frozen=True, slots=True)
class AssetEligibilityInputs:
    """Everything the eligibility check needs, already measured by the caller.

    Plain values, deliberately: no client, no repository, no ``fetch`` of any kind. If this
    object could obtain its own data it could obtain data from after the decision instant,
    and the resulting look-ahead would be invisible because the code would look correct.
    ``ticker_age`` and ``candle_age`` are what let the check reason about *when* the
    measurement was taken.

    ``intended_quantity`` is in contracts and ``intended_price`` is the contract price, so
    their product is the real quote-currency notional. :attr:`AssetMarket.multiplier` is
    decoded at discovery and never re-applied here — applying it twice is the same bug as
    never applying it.
    """

    market: AssetMarket
    #: Rolling 24h traded value in the quote asset, as the venue reports it.
    quote_volume_24h: Decimal
    #: Best bid and ask at the decision instant.
    bid: Decimal
    ask: Decimal
    #: How stale the quote and the last closed bar were when the decision was made.
    ticker_age: timedelta
    candle_age: timedelta
    #: Per-bar volatility as a fraction of price (ATR/close, or a realised bar-return sigma).
    volatility: Decimal
    #: Last closed bar's high-minus-low, in price units.
    last_bar_range: Decimal
    #: The recent normal for that range — an ATR or a median — in price units.
    typical_bar_range: Decimal
    #: Last closed bar's signed open-to-close return, as a fraction.
    last_bar_return: Decimal
    #: Last closed bar's traded value in the quote asset.
    bar_quote_volume: Decimal
    #: The order being contemplated, and the protective stop that would accompany it.
    intended_quantity: Decimal
    intended_price: Decimal
    stop_distance: Decimal

    @property
    def mid(self) -> Decimal:
        """Mid price, or zero when the quote is not two-sided."""
        if self.bid <= ZERO or self.ask <= ZERO:
            return ZERO
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal:
        """Bid/ask spread in basis points of the mid, or zero when there is no mid."""
        return safe_divide((self.ask - self.bid) * _BPS, self.mid)

    @property
    def intended_notional(self) -> Decimal:
        """Quote-currency value of the contemplated order."""
        return abs(self.intended_quantity) * self.intended_price * self.market.contract_size


def _liquidity_reasons(inputs: AssetEligibilityInputs, limits: AssetEligibilityLimits) -> list[str]:
    """Rejections that concern whether the market can absorb an order at a sane price."""
    reasons: list[str] = []
    market = inputs.market

    if not market.active:
        reasons.append(f"{market.symbol} is not active on the venue")

    if inputs.quote_volume_24h < limits.min_quote_volume_24h:
        reasons.append(
            f"24h quote volume {inputs.quote_volume_24h:,.0f} below the "
            f"{limits.min_quote_volume_24h:,.0f} {market.asset_class} liquidity floor"
        )

    if inputs.mid <= ZERO or inputs.ask < inputs.bid:
        reasons.append(f"no valid two-sided quote: bid {inputs.bid}, ask {inputs.ask}")
    elif inputs.spread_bps > limits.max_spread_bps:
        reasons.append(
            f"spread {inputs.spread_bps:.1f}bps wider than the {limits.max_spread_bps}bps maximum"
        )

    return reasons


def _freshness_reasons(inputs: AssetEligibilityInputs, limits: AssetEligibilityLimits) -> list[str]:
    """Rejections that concern whether the decision is being made on current information.

    This is the group that matters most for equities and indices. Their underlying closes
    overnight, at weekends and on exchange holidays, and while the perpetual keeps quoting
    through those windows the price it quotes stops being informed by anything. A stale bar
    is the honest signal that the market has gone quiet, and standing aside is the correct
    response — not extrapolating from the last print before the close.
    """
    reasons: list[str] = []

    if inputs.ticker_age > limits.max_ticker_age:
        reasons.append(
            f"quote is {inputs.ticker_age.total_seconds():.0f}s old, past the "
            f"{limits.max_ticker_age.total_seconds():.0f}s staleness limit"
        )

    if inputs.candle_age > limits.max_candle_age:
        reasons.append(
            f"last bar is {inputs.candle_age.total_seconds() / 60:.0f}min old, past the "
            f"{limits.max_candle_age.total_seconds() / 60:.0f}min staleness limit"
        )

    return reasons


def _volatility_reasons(
    inputs: AssetEligibilityInputs, limits: AssetEligibilityLimits
) -> list[str]:
    """Rejections that concern the regime: too quiet to pay, too wild to control, or broken."""
    reasons: list[str] = []

    if inputs.volatility < limits.min_volatility:
        reasons.append(
            f"volatility {inputs.volatility:.4f} below the {limits.min_volatility} "
            f"{inputs.market.asset_class} floor; the average bar cannot cover the "
            "round-trip cost"
        )
    elif inputs.volatility > limits.max_volatility:
        reasons.append(
            f"volatility {inputs.volatility:.4f} above the {limits.max_volatility} "
            f"{inputs.market.asset_class} ceiling; a stop outside the noise would exceed "
            "the risk budget"
        )

    # Relative flash test. Skipped when the typical range is zero, which means there is no
    # "normal" to be a multiple of — the absolute test below still applies, so a flat feed
    # cannot become a free pass.
    if inputs.typical_bar_range > ZERO:
        multiple = inputs.last_bar_range / inputs.typical_bar_range
        if multiple > limits.max_bar_range_multiple:
            reasons.append(
                f"flash move: last bar range is {multiple:.1f}x the typical range, "
                f"above the {limits.max_bar_range_multiple}x breaker"
            )

    if abs(inputs.last_bar_return) > limits.max_abs_bar_return:
        reasons.append(
            f"flash move: last bar returned {inputs.last_bar_return:.2%}, beyond the "
            f"{limits.max_abs_bar_return:.2%} single-bar breaker"
        )

    return reasons


def _order_reasons(inputs: AssetEligibilityInputs, limits: AssetEligibilityLimits) -> list[str]:
    """Rejections that concern the specific order, against venue rules and against the tape."""
    reasons: list[str] = []
    market = inputs.market
    notional = inputs.intended_notional
    quantity = abs(inputs.intended_quantity)

    if quantity < market.min_quantity:
        reasons.append(
            f"quantity {quantity} below the venue minimum {market.min_quantity} for {market.symbol}"
        )

    if market.max_quantity is not None and quantity > market.max_quantity:
        reasons.append(
            f"quantity {quantity} above the venue maximum {market.max_quantity} for {market.symbol}"
        )

    if notional < market.min_notional:
        reasons.append(
            f"notional {notional} below the venue minimum {market.min_notional} for {market.symbol}"
        )

    # Liquidity-aware ceiling: the order is measured against what actually traded in the
    # last bar, not against displayed depth. Depth on these books is quoted by market
    # makers who withdraw it the moment size arrives; traded volume is the only number that
    # was real. This is the check that keeps the thin end of the 193 listed equities out.
    volume_cap = inputs.bar_quote_volume * limits.max_bar_volume_share
    if notional > volume_cap:
        share = safe_divide(notional, inputs.bar_quote_volume)
        reasons.append(
            f"order is {share:.2%} of the last bar's volume, above the "
            f"{limits.max_bar_volume_share:.2%} liquidity ceiling"
        )

    # A stop must clear both the tick grid and the live spread; whichever is wider governs.
    tick_floor = market.price_tick * limits.min_stop_ticks
    spread = inputs.ask - inputs.bid if inputs.ask >= inputs.bid else ZERO
    required_stop = max(tick_floor, spread)
    if inputs.stop_distance < required_stop:
        reasons.append(
            f"stop distance {inputs.stop_distance} below the required {required_stop} "
            f"({limits.min_stop_ticks} ticks, and never inside the {spread} spread)"
        )

    return reasons


def assess_eligibility(
    inputs: AssetEligibilityInputs,
    limits: AssetEligibilityLimits | None = None,
) -> EligibilityVerdict:
    """Decide whether one market may be traded right now, on measurements alone.

    ``limits`` defaults to the band for the market's own class, which is the case the
    caller wants almost always; passing them explicitly is for tests and for an operator
    tightening one class without touching the table.

    The four groups are evaluated in full and their reasons concatenated — liquidity,
    freshness, regime, then the order itself — so the verdict describes the whole state of
    the market rather than the first thing that happened to be checked. Ordering is fixed
    for readable logs and carries no priority: any single reason is disqualifying.

    This is a veto, never a recommendation. An empty reason list means "nothing here
    forbids a trade", which is a much weaker statement than "trade this", and the strategy,
    cost and risk layers still have to agree before anything is sent.
    """
    resolved = limits if limits is not None else limits_for(inputs.market.asset_class)
    reasons = [
        *_liquidity_reasons(inputs, resolved),
        *_freshness_reasons(inputs, resolved),
        *_volatility_reasons(inputs, resolved),
        *_order_reasons(inputs, resolved),
    ]
    return EligibilityVerdict(eligible=not reasons, reasons=tuple(reasons))


# --------------------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------------------

#: Per-class slippage allowance, in basis points per leg.
#:
#: Slippage is what the venue does *not* report: fees are stated, the spread is observable,
#: and the difference between the quoted mid and the achieved fill is neither. These are
#: allowances rather than measurements and are labelled as such.
#:
#: They are ordered by book depth, which is the thing slippage is actually a function of.
#: Metals sit tightest (gold quotes 0.02bp wide on this venue and the book behind it is the
#: deepest of the non-crypto set); energy and indices next; single-name equities widest,
#: because a synthetic perpetual on one company is the thinnest book here and the one most
#: likely to move on a single order.
#:
#: A wrong allowance is not symmetric in consequence: too high refuses trades that were
#: marginally profitable, too low takes trades that were not. They are therefore set at the
#: pessimistic end of what the measured spreads support.
SLIPPAGE_BPS_BY_CLASS: Mapping[AssetClass, Decimal] = {
    AssetClass.METAL: Decimal("1"),
    AssetClass.ENERGY: Decimal("2"),
    AssetClass.INDEX: Decimal("2"),
    AssetClass.EQUITY: Decimal("3"),
    AssetClass.CRYPTO: Decimal("2"),
    AssetClass.MEME: Decimal("5"),
}

#: Net edge a candidate must retain *after* every cost, as a fraction of notional.
#:
#: 0.2%. Clearing costs exactly is not a reason to trade: it is a reason to be indifferent,
#: and an indifferent trade still consumes a position slot, a margin allocation and a share
#: of the drawdown budget. The buffer is what makes the expected outcome positive rather
#: than merely non-negative.
DEFAULT_NET_PROFIT_BUFFER = Decimal("0.002")

#: Assumed holding period when a caller does not state one, for the funding term.
#:
#: Eight hours — long enough that funding is not silently ignored on a position held
#: across a settlement, short enough to describe an intraday 15m strategy. Callers that
#: know their expected holding period should pass it; this is the honest default, not a
#: measurement.
DEFAULT_HOLDING_PERIOD = timedelta(hours=8)


@dataclass(frozen=True, slots=True)
class CostInputs:
    """What a round trip in one market will cost, decomposed into its four real parts.

    Fees come from the venue via :class:`AssetMarket`, the spread from the live quote, the
    slippage from a per-class allowance, and funding from a measured rate and the venue's
    own settlement interval. Nothing is a blanket constant across classes — which is the
    whole point, since Bybit charges 1bp/6bp on these perpetuals against 10bp/10bp on spot
    and settles metals every 4 hours against 8 for everything else.
    """

    market: AssetMarket
    bid: Decimal
    ask: Decimal
    #: Whether the entry and exit are expected to rest as maker orders. Taker is assumed,
    #: because an entry that must be filled now is the case the gate has to survive.
    is_maker: bool = False
    #: Funding rate per settlement interval, signed as the venue quotes it: positive means
    #: longs pay shorts. ``None`` when unknown — see :meth:`funding_cost_rate`.
    funding_rate: Decimal | None = None
    #: Which way the contemplated position runs, for the sign of the funding term.
    side_is_long: bool = True
    #: How long the position is expected to be held.
    holding_period: timedelta = DEFAULT_HOLDING_PERIOD
    #: Per-leg slippage override, in bps. Defaults to the market's class allowance.
    slippage_bps: Decimal | None = None

    @property
    def mid(self) -> Decimal:
        """Mid price, or zero when the quote is not two-sided."""
        if self.bid <= ZERO or self.ask <= ZERO:
            return ZERO
        return (self.bid + self.ask) / Decimal(2)

    @property
    def fee_rate(self) -> Decimal:
        """Round-trip commission, from the venue's own schedule for this instrument."""
        rate = self.market.maker_fee if self.is_maker else self.market.taker_fee
        return rate * Decimal(2)

    @property
    def spread_rate(self) -> Decimal:
        """Cost of crossing the spread, as a fraction of notional.

        Charged once, not twice. A taker round trip crosses half the spread on the way in
        and half on the way out, which sums to one full spread — double-counting it is the
        most common way a cost model talks itself out of every profitable trade.

        A maker round trip is credited nothing here: resting on the bid may earn the spread
        or may simply not fill, and a cost model is the wrong place to book an outcome that
        depends on adverse selection.
        """
        if self.mid <= ZERO or self.ask < self.bid:
            return ZERO
        return safe_divide(self.ask - self.bid, self.mid)

    @property
    def slippage_rate(self) -> Decimal:
        """Slippage over both legs, as a fraction of notional."""
        per_leg = (
            self.slippage_bps
            if self.slippage_bps is not None
            else SLIPPAGE_BPS_BY_CLASS.get(self.market.asset_class, Decimal("5"))
        )
        return per_leg * Decimal(2) / _BPS

    @property
    def funding_periods(self) -> Decimal:
        """Settlements the position is expected to be held across.

        Fractional on purpose. A position held two hours against a four-hour interval has a
        50% chance of paying one settlement, and charging it half of one is a better
        estimate than charging it zero — which is what rounding down would do, and would
        make every short hold look free.
        """
        interval = self.market.funding_interval_minutes
        if not interval or interval <= 0:
            return ZERO
        minutes = Decimal(str(self.holding_period.total_seconds())) / Decimal(60)
        return safe_divide(minutes, Decimal(interval))

    @property
    def funding_cost_rate(self) -> Decimal:
        """Funding over the expected hold, signed: positive is paid, negative is received.

        ``None`` for the rate means the caller could not measure it, and the honest
        response is zero rather than a guess. That is a *known* understatement of cost, and
        it is recorded as such in :meth:`AllInCost.assumptions` so a gate decision that
        turned on it can be identified afterwards rather than trusted silently.

        Only perpetuals fund. A market the venue reports no funding interval for
        contributes nothing here regardless of what rate is supplied.
        """
        if self.funding_rate is None:
            return ZERO
        signed = self.funding_rate if self.side_is_long else -self.funding_rate
        return signed * self.funding_periods


@dataclass(frozen=True, slots=True)
class AllInCost:
    """A decomposed round-trip cost, in fractions of notional.

    Decomposed rather than a single scalar because the components are actionable in
    different ways: a fee is a fact, a spread is a reason to wait, slippage is a reason to
    size down, and funding is a reason to hold for less time. A gate that reports only the
    total tells an operator that the trade was uneconomic without telling them what to
    change.
    """

    fees: Decimal
    spread: Decimal
    slippage: Decimal
    funding: Decimal
    #: Costs that were assumed rather than measured, named so a decision can be audited.
    assumptions: tuple[str, ...] = ()

    @property
    def total(self) -> Decimal:
        """All-in round-trip cost as a fraction of notional."""
        return self.fees + self.spread + self.slippage + self.funding


def all_in_cost(inputs: CostInputs) -> AllInCost:
    """Decompose the round-trip cost of one contemplated position."""
    assumptions: list[str] = []
    if inputs.funding_rate is None and inputs.market.funding_interval_minutes:
        assumptions.append(
            f"funding excluded: no rate measured for {inputs.market.symbol}, which settles "
            f"every {inputs.market.funding_interval_minutes}min — the total understates cost"
        )
    if inputs.mid <= ZERO:
        assumptions.append(
            f"spread excluded: no two-sided quote for {inputs.market.symbol} "
            f"(bid {inputs.bid}, ask {inputs.ask}) — the total understates cost"
        )
    assumptions.append(
        f"slippage is a {inputs.market.asset_class} class allowance, not a measurement"
    )
    return AllInCost(
        fees=inputs.fee_rate,
        spread=inputs.spread_rate,
        slippage=inputs.slippage_rate,
        funding=inputs.funding_cost_rate,
        assumptions=tuple(assumptions),
    )


@dataclass(frozen=True, slots=True)
class CostVerdict:
    """Whether an expected move survives its own costs, and by how much."""

    clears: bool
    expected_move: Decimal
    cost: AllInCost
    net_edge: Decimal
    buffer: Decimal
    reason: str | None = None


def clears_costs(
    expected_move: Decimal,
    inputs: CostInputs,
    *,
    buffer: Decimal = DEFAULT_NET_PROFIT_BUFFER,
) -> CostVerdict:
    """Whether a candidate's expected move exceeds all-in cost plus the net-profit buffer.

    ``expected_move`` is the distance to the target as a fraction of the entry price — the
    gross move the strategy expects, before anything is taken out of it.

    The test is ``expected_move - total_cost >= buffer``. Deliberately a floor on the
    *net* number rather than a multiple of the gross: a multiple would let a market with
    high costs qualify on a large expected move that leaves nothing behind, which is the
    arithmetic by which a strategy trades all day and ends flat.
    """
    if expected_move < ZERO:
        raise ValidationError(f"expected move cannot be negative, got {expected_move}")
    if buffer < ZERO:
        raise ValidationError(f"net-profit buffer cannot be negative, got {buffer}")

    cost = all_in_cost(inputs)
    net = expected_move - cost.total
    if net < buffer:
        return CostVerdict(
            clears=False,
            expected_move=expected_move,
            cost=cost,
            net_edge=net,
            buffer=buffer,
            reason=(
                f"expected move {expected_move:.4%} leaves {net:.4%} after "
                f"{cost.total:.4%} all-in costs (fees {cost.fees:.4%}, spread "
                f"{cost.spread:.4%}, slippage {cost.slippage:.4%}, funding "
                f"{cost.funding:.4%}), below the {buffer:.4%} net-profit buffer"
            ),
        )
    return CostVerdict(
        clears=True, expected_move=expected_move, cost=cost, net_edge=net, buffer=buffer
    )


# --------------------------------------------------------------------------------------
# Strategy gating
# --------------------------------------------------------------------------------------

#: Strategy families each asset class admits, keyed by the family
#: :func:`quantflow.orchestrator.selection.strategy_family` reports.
#:
#: Driven by the existing family map rather than by a second per-strategy list, so a
#: strategy added to ``_FAMILIES`` is gated correctly without a matching edit here.
#:
#: Crypto and memes admit everything; the taxonomy was built on them and the whole library
#: was validated against them. The non-crypto classes exclude two families, both for
#: reasons about the instrument rather than about the strategies:
#:
#: * **volume** — ``obv_trend``, ``volume_breakout``, ``money_flow_index``,
#:   ``accumulation_distribution``, ``volume_price_divergence``. A synthetic perpetual on
#:   AAPL turns over a few million a day against many billions in the share itself. Its
#:   volume series measures participation in the *derivative*, not in the asset, so a
#:   volume strategy reading it is reading a shadow and mistaking it for the tape. This is
#:   the one exclusion that would be wrong to relax.
#: * **structure** — ``swing_structure``. Swing pivots need a continuous tape to be
#:   defined. Equity and index perpetuals inherit the gap when their underlying reopens,
#:   and commodities carry session breaks, so a "pivot" is regularly an artefact of a
#:   market that was closed rather than a level anyone traded at.
#:
#: An unmapped family is refused outside crypto. A strategy nobody has classified reads an
#: information source nobody has checked against these instruments, and admitting it by
#: default would make the gate decorative.
ALLOWED_FAMILIES_BY_CLASS: Mapping[AssetClass, frozenset[str] | None] = {
    # None means "no family restriction".
    AssetClass.CRYPTO: None,
    AssetClass.MEME: None,
    AssetClass.METAL: frozenset({"trend", "momentum", "breakout", "reversion", "volatility"}),
    AssetClass.ENERGY: frozenset({"trend", "momentum", "breakout", "reversion", "volatility"}),
    AssetClass.EQUITY: frozenset({"momentum", "breakout", "trend", "reversion", "volatility"}),
    AssetClass.INDEX: frozenset({"momentum", "breakout", "trend", "reversion", "volatility"}),
}

#: The default a strategy declares when it says nothing: every class.
#:
#: Permissive on purpose. The 44 strategies in the library predate this taxonomy and none
#: of them was written with an asset class in mind, so declaring them all crypto-only would
#: be inventing an opinion nobody holds. The real restriction is
#: :data:`ALLOWED_FAMILIES_BY_CLASS`, which is derived from what a strategy *reads*;
#: ``supported_asset_classes`` exists so a strategy with a genuine instrument dependency
#: can narrow itself, and the two are ANDed.
ALL_ASSET_CLASSES: frozenset[AssetClass] = frozenset(AssetClass)


def family_supports_class(family: str, asset_class: AssetClass) -> bool:
    """Whether an information source is meaningful for this asset class.

    Takes the family as a value rather than looking it up from a strategy id, so this
    module depends on nothing above it. ``quantflow.orchestrator`` owns the id-to-family
    map and imports the universe layer; having the universe layer import back would make
    the two packages mutually dependent, and importing either would then depend on which
    one Python happened to reach first.
    """
    allowed = ALLOWED_FAMILIES_BY_CLASS.get(asset_class)
    if allowed is None:
        return True
    return family in allowed


def strategy_supports_class(
    family: str,
    asset_class: AssetClass,
    *,
    declared: frozenset[str] | None = None,
) -> bool:
    """Whether a strategy may run on a market of this class.

    Two independent vetoes, ANDed: what the strategy declares about itself via
    ``supported_asset_classes`` (``declared``; ``None`` or empty means every class), and
    what its family implies about whether its inputs mean anything here. Either can refuse
    and neither can override the other — a strategy that declares itself universal is still
    refused on an equity if it reads volume, and a strategy in an admitted family is still
    refused if it declared itself crypto-only.

    ``declared`` holds plain strings rather than :class:`AssetClass` members so that
    :class:`~quantflow.strategy.base.Strategy` can carry the attribute without importing
    this module. :class:`AssetClass` is a :class:`~enum.StrEnum`, so a member and its value
    hash identically and the membership test works either way round.
    """
    if declared and asset_class not in declared:
        return False
    return family_supports_class(family, asset_class)


def gating_reason(strategy_id: str, family: str, asset_class: AssetClass) -> str:
    """Why a strategy was refused on this class, in words, for the decision log."""
    return f"{strategy_id} reads {family}, which is not admitted for {asset_class} markets"


#: Key under which the engine records a symbol's asset class in
#: :attr:`~quantflow.strategy.base.StrategyContext.metadata`.
#:
#: The context carries a :class:`~quantflow.domain.instruments.Symbol`, and a symbol alone
#: cannot be classified: ``XAU`` is gold only because the venue said so, and a
#: ticker-shaped guess would read it as a token. The engine holds the instrument and
#: therefore the venue's label, so it resolves the class once per decision and records it
#: here rather than making every consumer re-derive it from data it does not have.
ASSET_CLASS_METADATA_KEY = "asset_class"


def asset_class_from_metadata(
    metadata: Mapping[str, object], *, default: AssetClass = AssetClass.CRYPTO
) -> AssetClass:
    """Read the asset class a strategy context was built with.

    Falls back to ``default`` when the key is absent or unrecognised, which is what happens
    for any engine or test that predates the metadata being populated. Defaulting to
    :attr:`~AssetClass.CRYPTO` keeps that path behaving exactly as it did before this
    module existed — every family is admitted for crypto — so an unpopulated context loses
    no capability and gains no gate it was never designed for.
    """
    raw = metadata.get(ASSET_CLASS_METADATA_KEY)
    if isinstance(raw, AssetClass):
        return raw
    if isinstance(raw, str):
        try:
            return AssetClass(raw)
        except ValueError:
            return default
    return default


__all__ = [
    "ALLOWED_FAMILIES_BY_CLASS",
    "ALL_ASSET_CLASSES",
    "ASSET_CLASS_METADATA_KEY",
    "ASSET_QUOTE_ASSET",
    "DEFAULT_HOLDING_PERIOD",
    "DEFAULT_NET_PROFIT_BUFFER",
    "ENERGY_ROOTS",
    "INDEX_ETF_ROOTS",
    "LIMITS_BY_CLASS",
    "METAL_ROOTS",
    "NON_CRYPTO_CLASSES",
    "NON_CRYPTO_MIN_QUOTE_VOLUME_24H",
    "PERP_TAKER_FEE_BPS",
    "SLIPPAGE_BPS_BY_CLASS",
    "VENUE_TYPE_COMMODITY",
    "VENUE_TYPE_STOCK",
    "AllInCost",
    "AssetClass",
    "AssetEligibilityInputs",
    "AssetEligibilityLimits",
    "AssetMarket",
    "CostInputs",
    "CostVerdict",
    "all_in_cost",
    "assess_eligibility",
    "asset_class_from_metadata",
    "build_asset_market",
    "classify_asset_class",
    "clears_costs",
    "discover_asset_universe",
    "family_supports_class",
    "gating_reason",
    "limits_for",
    "strategy_supports_class",
]
