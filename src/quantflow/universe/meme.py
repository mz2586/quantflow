"""The meme-coin universe: what the venue actually lists, and what is worth touching.

Two questions live here and they are not the same question.

**Which markets are meme markets** is, honestly, *not* discoverable. Bybit's instrument
metadata carries tick size, lot step, notional bounds, leverage and a status flag — and no
category, sector or tag of any kind. Nothing in the payload distinguishes ``DOGE/USDT``
from ``LINK/USDT``. So :data:`MEME_BASE_ASSETS` is a **curated list, hand-maintained**, and
this module does not pretend otherwise. What *is* genuinely discovered from the venue is
everything that matters operationally: which of those assets are listed at all, whether
they are active, what their tick and step and minimum notional are, and under which of
Bybit's ``1000``/``10000``/``1000000`` contract conventions each one trades.

**Whether a listed meme market is tradable right now** is entirely measured. A meme coin is
not a small-cap altcoin with a funnier name; the failure modes are different in kind:

* Liquidity is real for an hour and gone for a week. A 24h volume floor, not a listing date.
* Spreads on the same symbol range from 2bps to 200bps within a session, and a spread wide
  enough eats the entire distribution of moves the strategy is trying to capture.
* Volatility is bimodal. Too low and there is nothing to pay for the round trip; too high
  and the stop that would survive the noise is wider than the risk budget allows.
* The characteristic disaster is the vertical bar — a listing pump, a liquidation cascade,
  an influencer post. The system must refuse those, not chase them. Being late to a 40%
  candle is not an edge, it is being the exit liquidity for whoever was early.

Every threshold below is declared once, with the reasoning for it, and none was chosen by
trying values until a particular coin passed. A threshold tuned to admit ``FARTCOIN`` is
not a threshold, it is a hard-coded opinion about ``FARTCOIN`` wearing a limit's clothing.

Nothing in this module fetches anything. Every measurement arrives as a plain value that
the caller took at a known instant, because a module that can reach for data is a module
that can reach *forward* for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from quantflow.core.config import MarketType
from quantflow.core.errors import ValidationError
from quantflow.core.precision import ONE, ZERO, safe_divide
from quantflow.domain.instruments import Instrument, Symbol

#: The quote asset the universe is built against.
#:
#: USDT-quoted linear contracts are where meme liquidity actually is. The same coin listed
#: against USDC or as an inverse contract is a *different, thinner book* with its own
#: spread — treating them as interchangeable would import a liquidity assumption that the
#: second book does not honour.
MEME_QUOTE_ASSET = "USDT"

#: Curated meme-coin base-asset roots, stored **without** any venue size prefix.
#:
#: Hand-maintained, and deliberately so — see the module docstring. The honest description
#: is "an opinion about which tickers are meme coins", reviewed by a human, not a fact
#: retrieved from the exchange. It is a *root* list: ``PEPE`` covers Bybit's ``1000PEPE``
#: contract, so adding a coin does not require knowing which multiplier the venue chose.
#:
#: Being a superset of what any one venue lists is intentional. An entry that is not listed
#: simply never appears in the discovered universe, which costs nothing; a *missing* entry
#: silently routes a meme coin through the major-asset sizing path, which costs money.
MEME_BASE_ASSETS: frozenset[str] = frozenset(
    {
        "ACT",
        "BABYDOGE",
        "BOME",
        "BONK",
        "BRETT",
        "BTT",
        "CAT",
        "DOGE",
        "DOGS",
        "FARTCOIN",
        "FLOKI",
        "GOAT",
        "LADYS",
        "MEME",
        "MEW",
        "MOG",
        "MOODENG",
        "NEIRO",
        "ORDI",
        "PENGU",
        "PEPE",
        "PNUT",
        "POPCAT",
        "RATS",
        "SATS",
        "SHIB",
        "SPX",
        "TRUMP",
        "TURBO",
        "WIF",
    }
)

#: Venue size prefixes, **longest first**.
#:
#: Bybit quotes low-priced tokens as a basket so the price sits on a sane tick grid:
#: ``1000PEPE`` is one contract on a thousand PEPE. Order matters and is load-bearing —
#: ``"1000000BABYDOGE"`` also starts with ``"1000"``, so a shortest-first scan would strip
#: the wrong prefix and leave a root of ``000BABYDOGE`` that matches nothing.
_MULTIPLIER_PREFIXES: tuple[tuple[str, Decimal], ...] = (
    ("1000000", Decimal("1000000")),
    ("10000", Decimal("10000")),
    ("1000", Decimal("1000")),
)


def strip_multiplier(base: str) -> tuple[str, Decimal]:
    """Split a venue base asset into its root and its contract multiplier.

    ``"1000PEPE" -> ("PEPE", Decimal("1000"))``, ``"DOGE" -> ("DOGE", Decimal("1"))``.

    This is not cosmetic. A contract on ``1000PEPE`` is priced, ticked and sized a thousand
    times the token, so a multiplier read as 1 turns every notional, stop distance and
    price-per-tick downstream into a number that is wrong by three orders of magnitude, in
    the direction that looks plausible.

    A prefix is only stripped when a real root remains behind it and that root begins with
    a letter. A hypothetical listing of a token literally named ``1000`` — or one whose
    name starts with digits — would otherwise be shredded into a root that matches nothing,
    and silently dropping a market is worse than not recognising a multiplier.
    """
    raw = base.strip().upper()
    for prefix, multiplier in _MULTIPLIER_PREFIXES:
        if raw.startswith(prefix):
            root = raw[len(prefix) :]
            if root and root[0].isalpha():
                return root, multiplier
    return raw, ONE


def is_meme(symbol: Symbol) -> bool:
    """Whether a symbol's base asset is a curated meme coin, prefix conventions included."""
    root, _ = strip_multiplier(symbol.base)
    return root in MEME_BASE_ASSETS


@dataclass(frozen=True, slots=True)
class MemeMarket:
    """One listed, meme-classified market plus the venue rules that constrain an order.

    A flattened projection of :class:`~quantflow.domain.instruments.Instrument`, carrying
    only the fields an eligibility or sizing decision actually reads, plus the decoded
    multiplier. Flattened rather than wrapping the instrument so that the multiplier — the
    single most misreadable property of a meme contract — is impossible to forget to apply:
    it sits at the same level as the tick and the step, not one dereference away.
    """

    symbol: Symbol
    base_root: str
    multiplier: Decimal
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    min_notional: Decimal
    active: bool
    market_type: MarketType


def discover_meme_universe(instruments: Iterable[Instrument]) -> list[MemeMarket]:
    """Filter a venue's instrument list down to the tradable meme markets.

    Three filters, all of them cheap and all of them load-bearing: the base asset must be a
    curated meme root, the quote must be :data:`MEME_QUOTE_ASSET`, and the venue must
    report the market as active. Delisted and suspended markets are dropped here rather
    than at order time — an inactive instrument still returns candles and a stale ticker,
    so a strategy that only checks for data will happily size a position in a market that
    will reject it.

    The result is sorted by symbol so that two runs over the same venue snapshot produce
    the same universe in the same order. Iteration order of a market map is not a promise
    any exchange client makes, and a universe that reshuffles is a universe whose logs
    cannot be diffed.
    """
    markets: list[MemeMarket] = []
    for instrument in instruments:
        symbol = instrument.symbol
        if symbol.quote != MEME_QUOTE_ASSET or not instrument.active:
            continue
        root, multiplier = strip_multiplier(symbol.base)
        if root not in MEME_BASE_ASSETS:
            continue
        markets.append(
            MemeMarket(
                symbol=symbol,
                base_root=root,
                multiplier=multiplier,
                price_tick=instrument.price_tick,
                quantity_step=instrument.quantity_step,
                min_quantity=instrument.min_quantity,
                min_notional=instrument.min_notional,
                active=instrument.active,
                market_type=instrument.market_type,
            )
        )
    return sorted(markets, key=lambda market: market.symbol)


@dataclass(frozen=True, slots=True)
class EligibilityLimits:
    """The thresholds a meme market must clear before a position is allowed in it.

    Each default is justified on its own field. The test that any of them has to pass is
    not "does this improve a backtest" — it is "can this number be defended without naming
    a coin". A limit derived from round-trip cost or from the fill model generalises to the
    next listing; a limit reverse-engineered from last month's ``WIF`` chart does not.
    """

    #: Minimum 24h quote volume, in USDT.
    #:
    #: The floor exists for the **exit**, not the entry. Getting into anything is easy; a
    #: position taken in a market doing under a few million a day has to be unwound into
    #: whoever happens to be quoting, and on a meme that is frequently nobody. Five million
    #: is a round order-of-magnitude choice: it makes a four-figure position a negligible
    #: share of daily flow with a wide margin, and no meme coin sits just below it in a way
    #: that would make the exact figure matter.
    min_quote_volume_24h: Decimal = Decimal("5000000")

    #: Maximum bid/ask spread, in basis points of the mid.
    #:
    #: Derived, not picked. A taker round trip on this venue costs roughly 5.5bps a side in
    #: fees; crossing a 10bp spread adds it again on entry and exit. That puts the floor on
    #: a profitable move at ~30bps before any edge exists. Allowing 20bp spreads would put
    #: it past 50bps, which is most of the distribution of intraday meme bar moves — the
    #: strategy would be paying the market maker for the privilege of being right.
    max_spread_bps: Decimal = Decimal("10")

    #: Maximum age of the quote used for the decision.
    #:
    #: Thirty seconds. A meme coin can travel several percent inside a minute, so a quote
    #: older than this is not a quote, it is a memory. Stale-but-plausible data is the
    #: dangerous case: it never looks wrong, it just prices the order against a book that
    #: has already moved.
    max_ticker_age: timedelta = timedelta(seconds=30)

    #: Maximum age of the most recent closed bar.
    #:
    #: Thirty minutes, i.e. two bars on the shortest timeframe this system trades. One
    #: missing bar is a feed hiccup; two consecutive missing bars means the data path is
    #: broken or the market has stopped trading, and both of those are reasons to stand
    #: aside rather than to extrapolate. Callers on a longer timeframe should scale this to
    #: two of their own bars.
    max_candle_age: timedelta = timedelta(minutes=30)

    #: Minimum per-bar volatility, as a fraction of price.
    #:
    #: 0.4%, i.e. roughly twice the ~21bp round-trip cost implied by the spread and fee
    #: limits above. Below it the average bar cannot pay for the trade even when the
    #: direction is right, and a strategy that trades anyway is buying lottery tickets with
    #: a guaranteed fee. This is the "nothing to trade" rejection, and it fires more often
    #: than people expect: a meme coin that has stopped moving is a dead one.
    min_volatility: Decimal = Decimal("0.004")

    #: Maximum per-bar volatility, as a fraction of price.
    #:
    #: 6%. Above this a stop placed outside the noise is wider than any sane per-trade risk
    #: budget can fund at a usable size, and the alternative — a tight stop inside a 6% bar
    #: — is not risk control, it is a coin flip with extra fees. The gap between bars also
    #: stops being a rounding error, and a stop cannot protect against a price the book
    #: never printed.
    max_volatility: Decimal = Decimal("0.06")

    #: Flash-move breaker: maximum last-bar range as a multiple of the typical bar range.
    #:
    #: 4x. A bar four times the recent normal is not a trend accelerating, it is a
    #: liquidation cascade, a listing shock, or a single post. The regime the strategy was
    #: fitted to has just stopped applying, so the correct action is to refuse, wait for the
    #: range to normalise, and forgo the move. Refusing costs the occasional real breakout;
    #: chasing costs a fill at the top of a wick.
    max_bar_range_multiple: Decimal = Decimal("4")

    #: Flash-move breaker: maximum absolute single-bar return, as a fraction.
    #:
    #: 10%. The absolute complement to the relative multiple above, and it is needed
    #: because the relative test has a blind spot: a coin whose *normal* bar range is
    #: already enormous never trips a multiple of its own normality. Two tests, one
    #: relative and one absolute, so that neither a calm coin going vertical nor a
    #: permanently violent one slips through.
    max_abs_bar_return: Decimal = Decimal("0.10")

    #: Maximum share of the recent bar's quote volume one order may take.
    #:
    #: 2%. The paper fill model rejects anything over 10% of bar volume, so this sits five
    #: times inside that: the goal is not to be *accepted*, it is to not be the reason the
    #: price moved. On a meme book, where displayed depth is a fraction of traded volume,
    #: an order that is a visible share of the tape gets front-run by the same bots that
    #: provide the liquidity.
    max_bar_volume_share: Decimal = Decimal("0.02")

    #: Minimum stop distance, in venue price ticks.
    #:
    #: Ten ticks, and separately never inside the prevailing spread. A stop closer than the
    #: spread is triggered by the quote oscillating between its own two sides — it does not
    #: measure an adverse move, it measures the market existing. That converts a risk
    #: control into a fee generator, and on a meme's tick grid the distinction is only a few
    #: ticks wide.
    min_stop_ticks: Decimal = Decimal("10")


#: The limits used when a caller does not supply their own.
DEFAULT_ELIGIBILITY_LIMITS = EligibilityLimits()

#: Per-bar volatility treated as the "major asset" baseline for size comparison.
#:
#: 1%, roughly what BTC or ETH does on a short intraday bar in an unremarkable regime. It
#: is the anchor :func:`meme_size_factor` measures a meme against, not a threshold anything
#: is rejected on, so its precise value moves sizing smoothly rather than switching
#: behaviour at a cliff.
MAJOR_REFERENCE_VOLATILITY = Decimal("0.01")

#: Hard ceiling on a meme position as a fraction of the equivalent major-asset position.
#:
#: 0.5. Even a meme behaving perfectly — tight spread, deep book, calm tape — gets at most
#: half the notional the same signal would earn on a major. What is being sized around is
#: not the volatility that has been observed but the volatility that is available: the
#: overnight delisting, the bridge exploit, the founder's rug. None of that is in the
#: return series until it is the only thing in it.
MEME_SIZE_CEILING = Decimal("0.5")

#: Basis points in one whole unit, for spread arithmetic.
_BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class EligibilityInputs:
    """Everything the eligibility check needs, already measured by the caller.

    Plain values, deliberately: no client, no repository, no ``fetch`` of any kind. If this
    object could obtain its own data it could obtain data from after the decision instant,
    and the resulting look-ahead would be invisible because the code would look correct.
    Each field is a number someone recorded at a stated moment, and ``ticker_age`` and
    ``candle_age`` are what let the check reason about *when* that moment was.

    ``intended_quantity`` is in contracts, and ``intended_price`` is the contract price —
    so on a ``1000PEPE`` market both already carry the multiplier and their product is the
    real USDT notional. This is why :attr:`MemeMarket.multiplier` is decoded at discovery
    and never re-applied here: applying it twice is the same bug as never applying it.
    """

    market: MemeMarket
    #: Rolling 24h traded value in the quote asset.
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
        return abs(self.intended_quantity) * self.intended_price


@dataclass(frozen=True, slots=True)
class EligibilityVerdict:
    """Whether a meme market may be traded right now, and every reason it may not.

    ``reasons`` is populated on rejection and empty on acceptance. All failures are
    reported, never just the first: an operator who fixes one objection and rediscovers the
    next on the following bar learns the constraint one round trip at a time, and a market
    failing on volume *and* spread *and* staleness is a qualitatively different situation
    from one failing on a single borderline check.
    """

    eligible: bool
    reasons: tuple[str, ...] = ()


def _liquidity_reasons(inputs: EligibilityInputs, limits: EligibilityLimits) -> list[str]:
    """Rejections that concern whether the market can absorb an order at a sane price."""
    reasons: list[str] = []

    if not inputs.market.active:
        reasons.append(f"{inputs.market.symbol} is not active on the venue")

    if inputs.quote_volume_24h < limits.min_quote_volume_24h:
        reasons.append(
            f"24h quote volume {inputs.quote_volume_24h:,.0f} below the "
            f"{limits.min_quote_volume_24h:,.0f} liquidity floor"
        )

    if inputs.mid <= ZERO or inputs.ask < inputs.bid:
        reasons.append(f"no valid two-sided quote: bid {inputs.bid}, ask {inputs.ask}")
    elif inputs.spread_bps > limits.max_spread_bps:
        reasons.append(
            f"spread {inputs.spread_bps:.1f}bps wider than the {limits.max_spread_bps}bps maximum"
        )

    return reasons


def _freshness_reasons(inputs: EligibilityInputs, limits: EligibilityLimits) -> list[str]:
    """Rejections that concern whether the decision is being made on current information."""
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


def _volatility_reasons(inputs: EligibilityInputs, limits: EligibilityLimits) -> list[str]:
    """Rejections that concern the regime: too quiet to pay, too wild to control, or broken."""
    reasons: list[str] = []

    if inputs.volatility < limits.min_volatility:
        reasons.append(
            f"volatility {inputs.volatility:.4f} below the {limits.min_volatility} floor; "
            "the average bar cannot cover the round-trip cost"
        )
    elif inputs.volatility > limits.max_volatility:
        reasons.append(
            f"volatility {inputs.volatility:.4f} above the {limits.max_volatility} ceiling; "
            "a stop outside the noise would exceed the risk budget"
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


def _order_reasons(inputs: EligibilityInputs, limits: EligibilityLimits) -> list[str]:
    """Rejections that concern the specific order, against venue rules and against the tape."""
    reasons: list[str] = []
    market = inputs.market
    notional = inputs.intended_notional
    quantity = abs(inputs.intended_quantity)

    if quantity < market.min_quantity:
        reasons.append(
            f"quantity {quantity} below the venue minimum {market.min_quantity} for {market.symbol}"
        )

    if notional < market.min_notional:
        reasons.append(
            f"notional {notional} below the venue minimum {market.min_notional} for {market.symbol}"
        )

    # Liquidity-aware ceiling: the order is measured against what actually traded in the
    # last bar, not against displayed depth. Depth on a meme book is quoted by bots that
    # withdraw it the moment size arrives; traded volume is the only number that was real.
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
    inputs: EligibilityInputs,
    limits: EligibilityLimits = DEFAULT_ELIGIBILITY_LIMITS,
) -> EligibilityVerdict:
    """Decide whether one meme market may be traded right now, on measurements alone.

    The four groups are evaluated in full and their reasons concatenated — liquidity,
    freshness, regime, then the order itself — so the verdict describes the whole state of
    the market rather than the first thing that happened to be checked. Ordering is fixed
    for readable logs, and carries no priority: any single reason is disqualifying.

    This is a veto, never a recommendation. An empty reason list means "nothing here
    forbids a trade", which is a much weaker statement than "trade this", and the strategy
    and risk layers still have to agree before anything is sent.
    """
    reasons = [
        *_liquidity_reasons(inputs, limits),
        *_freshness_reasons(inputs, limits),
        *_volatility_reasons(inputs, limits),
        *_order_reasons(inputs, limits),
    ]
    return EligibilityVerdict(eligible=not reasons, reasons=tuple(reasons))


def meme_size_factor(
    volatility: Decimal,
    *,
    reference_volatility: Decimal = MAJOR_REFERENCE_VOLATILITY,
    ceiling: Decimal = MEME_SIZE_CEILING,
) -> Decimal:
    """The fraction of a major-asset position a meme is allowed, given its volatility.

    ``ceiling * reference / (reference + volatility)``. Two properties are the whole point,
    and both hold for every non-negative volatility:

    * It is **strictly below one**, capped at ``ceiling``. Even a perfectly-behaved meme at
      zero volatility gets half a major's size, because the risk being sized around is not
      the volatility that was measured — it is the delisting, the exploit and the rug, none
      of which appear in a return series until they are the only thing in it.
    * It is **strictly decreasing** in volatility, smoothly. A hyperbola rather than a
      lookup table of bands: a coin should not double its position because its ATR crossed
      a threshold by a basis point, and a smooth curve has no such cliff to sit on.

    At the reference volatility the factor is half the ceiling; at ten times it, a
    twentieth. The tail shrinks fast enough that an extremely volatile market prices itself
    out on its own, usually below the venue minimum notional — at which point
    :func:`assess_eligibility` refuses the order outright, which is the intended outcome.
    """
    if volatility < ZERO:
        raise ValidationError(f"volatility cannot be negative, got {volatility}")
    if reference_volatility <= ZERO:
        raise ValidationError(f"reference volatility must be positive, got {reference_volatility}")
    if not (ZERO < ceiling < ONE):
        raise ValidationError(f"size ceiling must lie in (0, 1), got {ceiling}")
    return ceiling * reference_volatility / (reference_volatility + volatility)


def size_for_meme(
    baseline_notional: Decimal,
    volatility: Decimal,
    *,
    reference_volatility: Decimal = MAJOR_REFERENCE_VOLATILITY,
    ceiling: Decimal = MEME_SIZE_CEILING,
) -> Decimal:
    """Reduce a position size that the risk engine already approved, because it is a meme.

    ``baseline_notional`` is what the equivalent signal would have been given on a major
    asset, after the global risk engine has had its say. This function returns a strictly
    smaller number, scaled down further as volatility rises.

    **It only ever reduces.** The global risk engine remains authoritative: per-trade risk,
    daily and weekly loss limits, correlation caps and exposure ceilings are all decided
    there, and nothing here can loosen any of them. This is a second, narrower veto applied
    on top — the output is bounded above by ``ceiling * baseline_notional``, so a bug in
    this module can only ever make a position too small. That asymmetry is deliberate; the
    opposite arrangement is how a sizing helper ends up quietly overriding a risk limit.

    A negative baseline (a short expressed as a signed notional) keeps its sign: the
    reduction is applied to the magnitude, so the direction survives untouched.
    """
    factor = meme_size_factor(
        volatility, reference_volatility=reference_volatility, ceiling=ceiling
    )
    return baseline_notional * factor


__all__ = [
    "DEFAULT_ELIGIBILITY_LIMITS",
    "MAJOR_REFERENCE_VOLATILITY",
    "MEME_BASE_ASSETS",
    "MEME_QUOTE_ASSET",
    "MEME_SIZE_CEILING",
    "EligibilityInputs",
    "EligibilityLimits",
    "EligibilityVerdict",
    "MemeMarket",
    "assess_eligibility",
    "discover_meme_universe",
    "is_meme",
    "meme_size_factor",
    "size_for_meme",
    "strip_multiplier",
]
