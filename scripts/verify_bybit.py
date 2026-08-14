#!/usr/bin/env python
"""Verify the Bybit adapter against a real account.

Read-only by default. The order placement and cancellation steps are gated behind an
explicit flag *and* an interactive confirmation, because they move real money and no
amount of "it's only a test order" makes an unintended fill reversible.

Credentials are read from the environment or `.env` and are never printed. The report
shows only whether a key is present, never its value or any prefix of it — a partial key
in a log is still a partial key in a log.

    python scripts/verify_bybit.py                 # read-only checks
    python scripts/verify_bybit.py --place-order   # adds steps 6-8, asks first
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantflow.core.config import ExchangeSettings, get_settings
from quantflow.core.errors import QuantFlowError
from quantflow.domain.enums import OrderSide, OrderType
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.bybit import BybitGateway


@dataclass
class Step:
    """One verification step and its outcome."""

    name: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """One line for the console."""
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class Report:
    """The full integration result."""

    steps: list[Step] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add(self, name: str, ok: bool, detail: str, **data: Any) -> Step:
        """Record a step."""
        step = Step(name, ok, detail, data)
        self.steps.append(step)
        print(step.render(), flush=True)
        return step

    @property
    def passed(self) -> bool:
        """Whether every step succeeded."""
        return all(step.ok for step in self.steps)


async def verify(*, place_order: bool, symbol: Symbol) -> Report:
    """Run the verification sequence."""
    report = Report()
    settings = get_settings()
    exchange: ExchangeSettings = settings.exchange

    # --- 0. Credentials present? ------------------------------------------------
    if not exchange.has_credentials:
        report.add(
            "credentials",
            False,
            "no API key/secret configured. Bybit signs every private request with an "
            "HMAC of key AND secret; a key alone cannot authenticate. Set "
            "QF_EXCHANGE__API_KEY and QF_EXCHANGE__API_SECRET in .env (gitignored).",
        )
        return report
    report.add(
        "credentials",
        True,
        f"key and secret present, testnet={exchange.testnet}, "
        f"market_type={exchange.market_type.value}",
    )

    gateway = BybitGateway(exchange)
    try:
        # --- 1. Authentication --------------------------------------------------
        try:
            await gateway.connect()
            report.add(
                "authentication",
                True,
                f"signed request accepted by {'testnet' if exchange.testnet else 'MAINNET'}",
            )
        except QuantFlowError as exc:
            report.add("authentication", False, f"rejected: {exc}")
            return report

        # --- 2. Account type ----------------------------------------------------
        # V5 returns different balance structures for UNIFIED and CLASSIC accounts, so
        # this has to be established before any balance figure can be trusted.
        try:
            raw = await gateway.raw_account_info()
            report.add(
                "account_type",
                True,
                f"{raw.get('unifiedMarginStatus', 'unknown')} "
                f"(marginMode={raw.get('marginMode', 'unknown')})",
                **raw,
            )
        except (QuantFlowError, AttributeError) as exc:
            report.add(
                "account_type",
                False,
                f"could not determine: {exc}. Balance parsing may be wrong for this account type.",
            )

        # --- 3. Balances --------------------------------------------------------
        try:
            balances = await gateway.fetch_balances()
            funded = {k: v for k, v in balances.items() if v.total > 0}
            report.add(
                "balances",
                True,
                f"{len(balances)} currencies, {len(funded)} funded: "
                + (", ".join(f"{k}={v.total}" for k, v in sorted(funded.items())) or "none"),
            )
        except QuantFlowError as exc:
            report.add("balances", False, str(exc))

        # --- 4. Positions -------------------------------------------------------
        try:
            positions = await gateway.fetch_positions()
            report.add(
                "positions",
                True,
                f"{len(positions)} open"
                + (
                    ": "
                    + ", ".join(f"{p['symbol']} {p['side']} {p['contracts']}" for p in positions)
                    if positions
                    else ""
                ),
            )
        except (QuantFlowError, AttributeError) as exc:
            report.add(
                "positions",
                False,
                f"{exc} (spot accounts have no positions endpoint; this is expected on spot)",
            )

        # --- 5. Open orders -----------------------------------------------------
        try:
            orders = await gateway.fetch_open_orders()
            report.add("open_orders", True, f"{len(orders)} working")
        except QuantFlowError as exc:
            report.add("open_orders", False, str(exc))

        if not place_order:
            report.add(
                "order_lifecycle",
                True,
                "skipped: read-only run. Pass --place-order to test placement, "
                "cancellation and fills.",
            )
            return report

        # --- 6-8. Order lifecycle -----------------------------------------------
        await _order_lifecycle(gateway, symbol, report, exchange)
    finally:
        await gateway.aclose()

    return report


async def _order_lifecycle(
    gateway: BybitGateway, symbol: Symbol, report: Report, exchange: ExchangeSettings
) -> None:
    """Place, cancel and inspect a deliberately unfillable limit order.

    A limit order priced far from the market is used rather than a market order: it
    exercises the full placement and cancellation path without any realistic chance of
    filling, so the test cannot accidentally take a position.
    """
    ticker = await gateway.fetch_ticker(symbol)
    instrument = await gateway.get_instrument(symbol)

    # 50% below the bid, rounded to the venue's tick, and the minimum permitted size.
    price = instrument.round_price(ticker.bid * Decimal("0.5"), side=OrderSide.BUY)
    quantity = instrument.min_quantity
    notional = price * quantity
    if notional < instrument.min_notional:
        quantity = instrument.round_quantity(instrument.min_notional / price * Decimal("1.1"))
        notional = price * quantity

    print(
        f"\nAbout to place a REAL order on "
        f"{'TESTNET' if exchange.testnet else 'MAINNET'}:\n"
        f"  {symbol} BUY LIMIT {quantity} @ {price} (~{notional} {symbol.quote})\n"
        f"  This is 50% below the market and should not fill.\n",
        flush=True,
    )
    # Read on a worker thread: input() blocks the event loop, and a confirmation prompt
    # that stalls the loop would also stall the gateway's keepalives.
    answer = (await asyncio.to_thread(input, "Type 'yes' to proceed: ")).strip().lower()
    if answer != "yes":
        report.add("order_placement", True, "declined by operator; nothing was sent")
        return

    request = OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
    )
    try:
        order = await gateway.submit_order(request)
        report.add(
            "order_placement",
            True,
            f"accepted: venue_id={order.venue_order_id} status={order.status.value}",
        )
    except QuantFlowError as exc:
        report.add("order_placement", False, str(exc))
        return

    try:
        fetched = await gateway.fetch_order(order.order_id, symbol)
        report.add("order_fetch", True, f"status={fetched.status.value}")
    except QuantFlowError as exc:
        report.add("order_fetch", False, str(exc))

    try:
        cancelled = await gateway.cancel_order(order.order_id, symbol)
        report.add("order_cancel", True, f"status={cancelled.status.value}")
    except QuantFlowError as exc:
        report.add("order_cancel", False, f"{exc} — CHECK THE VENUE, an order may be live")

    try:
        fills = await gateway.fetch_my_trades(symbol, limit=5)
        report.add(
            "fills",
            True,
            f"{len(fills)} recent fills on {symbol} (0 expected for an unfilled order)",
        )
    except QuantFlowError as exc:
        report.add("fills", False, str(exc))


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Verify the Bybit adapter against a real account.")
    parser.add_argument(
        "--place-order",
        action="store_true",
        help="Also place and cancel a test order. Asks for confirmation first.",
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()

    parsed = Symbol.parse(args.symbol)
    assert isinstance(parsed, Symbol)

    report = asyncio.run(verify(place_order=args.place_order, symbol=parsed))
    print()
    print(f"{sum(1 for s in report.steps if s.ok)}/{len(report.steps)} steps passed")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
