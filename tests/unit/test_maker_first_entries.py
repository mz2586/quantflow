"""Entries go passive; exits stay aggressive.

Converting entries at the single point where a Signal becomes an OrderRequest, rather than
in twenty-two strategies, means every strategy gets maker pricing without any of them
knowing about it — and there is exactly one place where the behaviour can be wrong.

The asymmetry is deliberate and is the whole design:

* **Entries are patient.** Nothing is lost by not entering. A missed setup costs zero; a
  taker entry costs 0.06% every single time. So entries rest passively, post-only, and are
  abandoned if unfilled.
* **Exits are not.** A protective stop that waits for a passive fill is not protection. A
  reduce-only exit crosses the spread and pays taker, because the cost of being slow there
  is unbounded and the fee is not.

Default OFF. A change that alters how every order in the system is priced must be switched
on deliberately, not inherited by a config that predates it.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.config import RiskSettings
from quantflow.domain.enums import OrderSide, OrderStatus, OrderType
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Order, OrderRequest
from tests.conftest import REFERENCE_TIME


class TestSettings:
    def test_maker_first_is_off_by_default(self) -> None:
        """No config written before this feature silently changes execution."""
        assert RiskSettings().maker_first_entries is False

    def test_entry_lifetime_is_bounded_by_default(self) -> None:
        """An unbounded resting entry can fill on a signal that has expired."""
        assert RiskSettings().entry_limit_max_bars >= 1


class TestConversion:
    def test_an_entry_becomes_a_post_only_limit(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        order_type, price, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert (order_type, price, post_only) == (OrderType.LIMIT, Decimal("100"), True)

    def test_disabled_leaves_the_order_untouched(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        assert as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=False,
        ) == (OrderType.MARKET, None, False)

    def test_a_strategy_that_already_chose_a_limit_is_respected(self) -> None:
        """A strategy with its own price knows something this layer does not."""
        from quantflow.risk.engine import as_maker_entry

        _, price, post_only = as_maker_entry(
            OrderType.LIMIT,
            limit_price=Decimal("99"),
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert price == Decimal("99")
        assert post_only is True

    def test_a_stop_entry_is_not_converted(self) -> None:
        """A stop entry is triggered by price moving away; passive pricing contradicts it."""
        from quantflow.risk.engine import as_maker_entry

        order_type, _, post_only = as_maker_entry(
            OrderType.STOP_MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert order_type is OrderType.STOP_MARKET
        assert post_only is False

    def test_price_is_decimal(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        _, price, _ = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
        )

        assert isinstance(price, Decimal)


class TestExitsStayAggressive:
    def test_a_reduce_only_exit_is_never_post_only(self) -> None:
        """Protection that waits for a passive fill is not protection."""
        from quantflow.risk.engine import as_maker_entry

        order_type, _, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("100"),
            enabled=True,
            is_entry=False,
        )

        assert order_type is OrderType.MARKET
        assert post_only is False


class TestRestingEntriesExpire:
    """A passive entry that never fills must be abandoned, not left resting.

    ``entry_limit_max_bars`` has existed in the configuration since maker-first was
    written, and until now nothing enforced it. That is only harmless while maker-first is
    off. Switched on, an unfilled post-only entry rests at the venue indefinitely and can
    fill hours later on a setup that expired long ago — the strategy's stop and target were
    computed for a market that no longer exists, and the position is sized against a
    signal nobody would take now.

    Missing a trade costs nothing. Filling an expired one costs whatever the market did in
    between.
    """

    def test_an_entry_older_than_the_limit_is_expired(self) -> None:
        from quantflow.risk.engine import entry_has_expired

        assert entry_has_expired(bars_resting=4, max_bars=3) is True

    def test_an_entry_within_the_limit_keeps_resting(self) -> None:
        from quantflow.risk.engine import entry_has_expired

        assert entry_has_expired(bars_resting=3, max_bars=3) is False
        assert entry_has_expired(bars_resting=0, max_bars=3) is False

    def test_a_zero_limit_expires_immediately(self) -> None:
        # A configuration of zero means "do not rest at all", and must not be read as
        # "rest forever" through a falsy check.
        from quantflow.risk.engine import entry_has_expired

        assert entry_has_expired(bars_resting=1, max_bars=0) is True

    def test_a_negative_limit_is_treated_as_no_resting(self) -> None:
        from quantflow.risk.engine import entry_has_expired

        assert entry_has_expired(bars_resting=1, max_bars=-5) is True


class TestPostOnlyReachesTheVenue:
    """The post-only flag must survive the trip to Bybit, or maker-first does nothing.

    ``as_maker_entry`` converts an entry to a post-only limit and every layer above the
    gateway carries the flag. The gateway then built its parameters from
    ``time_in_force`` alone and never looked at ``post_only``, so the order left as a plain
    GTC limit priced at the touch — which crosses immediately and is charged taker.

    Measured live on 2026-08-15 minutes after enabling maker-first: the entry submitted as
    ``order_type=limit`` filled at **0.0550%**, byte-identical to every taker fill before
    it. The whole point of the feature is the 0.01% maker rate; without this the flag is
    decoration and the fee bill is unchanged.
    """

    @staticmethod
    def _request(*, post_only: bool) -> OrderRequest:
        return OrderRequest(
            symbol=Symbol.parse("BTC/USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            post_only=post_only,
            strategy_id="test",
        )

    def test_a_post_only_request_is_sent_as_post_only(self) -> None:
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(post_only=True))

        assert params["timeInForce"] == "PostOnly"

    def test_an_ordinary_limit_keeps_its_time_in_force(self) -> None:
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(post_only=False))

        assert params["timeInForce"] == "GTC"


class TestRestingEntryStopConfirmation:
    """A resting entry has no position yet, so there is nothing to confirm a stop against.

    ``_require_stop_on_venue`` proves protection by finding the stop attached to an **open
    position**. A market entry fills instantly so the position is there to inspect. A
    post-only limit does not: it rests until the market comes to it, and until then the
    account holds no exposure at all.

    Applying the check anyway finds no position, concludes the stop failed to attach, and
    raises — which fails the whole session. Live on 2026-08-15, twelve minutes after
    enabling maker-first: *"stop failed to attach on the venue for SOL/USDT; the entry was
    closed reduce-only"*, with no SOL position in existence to be unprotected.

    The protection itself is never in doubt: ``stopLoss`` travels in the order parameters,
    so the venue attaches it at the moment of fill. What must be deferred is the *read-back*
    — and only while the order is genuinely unfilled.
    """

    @staticmethod
    def _order(*, filled: Decimal, status: OrderStatus) -> Order:
        return Order(
            order_id="o-1",
            client_order_id="c-1",
            symbol=Symbol.parse("BTC/USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            filled_quantity=filled,
            status=status,
            created_at=REFERENCE_TIME,
            updated_at=REFERENCE_TIME,
            post_only=True,
        )

    def test_an_unfilled_resting_entry_defers_confirmation(self) -> None:
        from quantflow.exchange.bybit.rest import stop_confirmation_is_due

        order = self._order(filled=Decimal("0"), status=OrderStatus.NEW)

        assert stop_confirmation_is_due(order) is False

    def test_a_filled_entry_must_be_confirmed_immediately(self) -> None:
        # The case the invariant exists for: real exposure, so protection is not optional.
        from quantflow.exchange.bybit.rest import stop_confirmation_is_due

        order = self._order(filled=Decimal("1"), status=OrderStatus.FILLED)

        assert stop_confirmation_is_due(order) is True

    def test_a_partial_fill_must_be_confirmed(self) -> None:
        # Partially filled is still a live position, and a live position needs its stop.
        from quantflow.exchange.bybit.rest import stop_confirmation_is_due

        order = self._order(filled=Decimal("0.4"), status=OrderStatus.PARTIALLY_FILLED)

        assert stop_confirmation_is_due(order) is True

    def test_a_post_only_rejection_needs_no_stop(self) -> None:
        """A rejected post-only order created no exposure, so there is nothing to protect.

        Bybit refuses a post-only order that would cross rather than filling it as taker —
        which is the whole point of post-only, and an entirely normal outcome. The order
        comes back terminal with zero fills.

        Treating "terminal" as "has exposure" made that normal outcome fatal: the code
        demanded a stop for a position that was never opened, failed to find one, and then
        tried to flatten it — the venue answering *"current position is zero, cannot fix
        reduce-only order qty"*. That killed the session twice on 2026-08-15, at 11:45 and
        12:15.

        Exposure is fills. A cancelled, rejected or expired order with no fill is simply an
        order that did not happen.
        """
        from quantflow.exchange.bybit.rest import stop_confirmation_is_due

        rejected = self._order(filled=Decimal("0"), status=OrderStatus.REJECTED)
        cancelled = self._order(filled=Decimal("0"), status=OrderStatus.CANCELLED)

        assert stop_confirmation_is_due(rejected) is False
        assert stop_confirmation_is_due(cancelled) is False

    def test_a_terminal_order_that_did_fill_still_needs_its_stop(self) -> None:
        # The invariant must survive the fix: a filled order is exposure, terminal or not.
        from quantflow.exchange.bybit.rest import stop_confirmation_is_due

        filled = self._order(filled=Decimal("1"), status=OrderStatus.FILLED)
        part = self._order(filled=Decimal("0.5"), status=OrderStatus.CANCELLED)

        assert stop_confirmation_is_due(filled) is True
        assert stop_confirmation_is_due(part) is True, "a partial fill is a live position"


class TestTakeProfitExecutesAsMaker:
    """The target should rest as a limit; the stop must never stop being a market order.

    Exits were the untouched half of the fee bill. Maker entries cut the entry side to
    0.0200%, but every exit still crossed at 0.0550% — and roughly half of all fills are
    exits, so half the saving was being left behind.

    A take-profit is the one exit that can afford to be passive: it fires into a favourable
    move, and if it does not fill the position is still open, still protected, and still
    managed. A stop cannot: it exists precisely for the case where price is running away,
    and a passive stop that waits for a better price is not protection. So ``tpOrderType``
    becomes Limit and ``slOrderType`` stays Market — the asymmetry is the whole point.
    """

    @staticmethod
    def _request(*, target: Decimal | None) -> OrderRequest:
        return OrderRequest(
            symbol=Symbol.parse("BTC/USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            stop_loss_price=Decimal("49000"),
            take_profit_price=target,
            strategy_id="test",
        )

    def test_the_target_rests_as_a_limit_in_partial_mode(self) -> None:
        """The target earns the maker rate; the stop keeps market execution.

        The route matters and was found the hard way. ``tpLimitPrice`` with no mode gives
        *"tpLimitPrice can not have a value when tpSlMode is empty"*; adding
        ``tpslMode=Full`` gives *"tpOrderType only support Market when tpSlMode is Full"*.
        Both cost a live session. ``Partial`` is the mode that accepts a limit target, and
        it was verified against the demo venue before shipping.

        Exits are roughly half of all fills, so leaving them aggressive discarded half the
        saving maker-first exists for.
        """
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(target=Decimal("51000")))

        assert params["takeProfit"] == "51000"
        assert params["tpslMode"] == "Partial"
        assert params["tpOrderType"] == "Limit"
        assert params["tpLimitPrice"] == "51000"

    def test_both_legs_are_sized_to_the_whole_position(self) -> None:
        """Partial mode sizes each leg, so both must cover the entire position.

        The danger unique to this mode: a stop sized to less than the position leaves the
        remainder unprotected while every local record insists it is covered. Both sizes
        are therefore stated explicitly rather than left to a venue default.
        """
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(target=Decimal("51000")))

        assert params["tpSize"] == "0.01"
        assert params["slSize"] == "0.01", "the stop must cover the whole position"
        assert params["slOrderType"] == "Market", "a stop must never wait for a price"

    def test_the_stop_stays_a_market_order(self) -> None:
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(target=Decimal("51000")))

        assert params["stopLoss"] == "49000"
        assert params["slOrderType"] == "Market"
        assert "slLimitPrice" not in params, "a stop must never wait for a price"

    def test_no_target_adds_no_take_profit_parameters(self) -> None:
        from quantflow.exchange.bybit.rest import bybit_order_params

        params = bybit_order_params(self._request(target=None))

        assert "takeProfit" not in params
        assert "tpOrderType" not in params
        assert "tpslMode" not in params


class TestPassiveEntryOffset:
    """A post-only entry must rest inside the touch, not at a stale close price."""

    def test_the_passive_price_lands_on_the_tick_grid(self) -> None:
        # An offset computed as a fraction of price almost never lands on the grid, and the
        # venue rejects the whole order: "price 64296.63810 is not a multiple of tick 0.10"
        # blocked eleven consecutive candidates on 2026-08-18.
        from quantflow.risk.engine import as_maker_entry

        for side, tick, ref in (
            (OrderSide.BUY, Decimal("0.10"), Decimal("64308.5")),
            (OrderSide.SELL, Decimal("0.10"), Decimal("64308.5")),
            (OrderSide.BUY, Decimal("0.01"), Decimal("1907.67")),
            (OrderSide.SELL, Decimal("0.01"), Decimal("1907.67")),
        ):
            _, price, _ = as_maker_entry(
                OrderType.MARKET,
                limit_price=None,
                reference_price=ref,
                enabled=True,
                side=side,
                price_tick=tick,
            )
            assert price is not None
            assert price % tick == 0, f"{price} is not a multiple of {tick}"
            # Snapping must never push the order to the aggressive side.
            if side is OrderSide.BUY:
                assert price < ref
            else:
                assert price > ref

    def test_a_buy_is_posted_below_the_reference(self) -> None:
        # Resting exactly AT the bar close was the defect: the close is already stale when
        # the order lands, so an adverse tick makes a passive order aggressive and the
        # venue cancels it. 7 of 15 entry attempts were lost that way on 2026-08-17/18.
        from quantflow.risk.engine import PASSIVE_ENTRY_OFFSET_PCT, as_maker_entry

        _, price, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("64000"),
            enabled=True,
            side=OrderSide.BUY,
        )
        assert post_only is True
        assert price == Decimal("64000") - Decimal("64000") * PASSIVE_ENTRY_OFFSET_PCT
        assert price < Decimal("64000")

    def test_a_sell_is_posted_above_the_reference(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        _, price, _ = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("64000"),
            enabled=True,
            side=OrderSide.SELL,
        )
        assert price is not None
        assert price > Decimal("64000")

    def test_without_a_side_the_behaviour_is_unchanged(self) -> None:
        # The offset can only be applied when the direction is known; absent it, the old
        # at-the-touch behaviour stands rather than guessing a side.
        from quantflow.risk.engine import as_maker_entry

        _, price, _ = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("64000"),
            enabled=True,
        )
        assert price == Decimal("64000")

    def test_a_strategys_own_limit_price_is_still_never_moved(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        _, price, post_only = as_maker_entry(
            OrderType.LIMIT,
            limit_price=Decimal("63500"),
            reference_price=Decimal("64000"),
            enabled=True,
            side=OrderSide.BUY,
        )
        assert price == Decimal("63500")
        assert post_only is True

    def test_exits_are_never_offset(self) -> None:
        from quantflow.risk.engine import as_maker_entry

        order_type, price, post_only = as_maker_entry(
            OrderType.MARKET,
            limit_price=None,
            reference_price=Decimal("64000"),
            enabled=True,
            is_entry=False,
            side=OrderSide.SELL,
        )
        assert order_type is OrderType.MARKET
        assert price is None
        assert post_only is False
