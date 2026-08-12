#!/usr/bin/env python
"""Stage B: verify the safety fixes against the LIVE Bybit demo venue.

Every check asserts against what the *exchange* reports, never what the local record
believes. That distinction is the whole point: the review found a system that reported
trading it was not doing, so a check that trusts local state proves nothing.

Refuses to run unless the resolved endpoint is api-demo. Never touches mainnet.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from quantflow.core.config import ExchangeEnv, get_settings
from quantflow.core.errors import ExchangeError
from quantflow.domain.enums import OrderSide, OrderType, TimeInForce
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.bybit.rest import BybitGateway

SYMBOL = Symbol.parse("BTC/USDT")

#: Target notional per harness order. The venue's own minimum lot wins if it is larger -
#: BTC perps step at 0.001, so a smaller target would round to zero and submit nothing.
TARGET_NOTIONAL = Decimal("100")


@dataclass
class Check:
    """One verification result."""

    name: str
    passed: bool = False
    detail: str = ""
    venue_ids: list[str] = field(default_factory=list)


def line(text: str) -> None:
    print(text, flush=True)


async def _position_for(gateway: BybitGateway, symbol: Symbol) -> dict[str, Any] | None:
    """The venue's view of a position, or None."""
    wanted = symbol.concatenated
    for position in await gateway.fetch_positions():
        raw = str(position.get("symbol", "")).replace("/", "").replace(":USDT", "")
        if raw == wanted:
            info = position.get("info", {})
            size = info.get("size") if isinstance(info, dict) else None
            if size not in (None, "", "0", 0):
                return position
    return None


async def _entry_request(gateway: BybitGateway, *, stop_pct: Decimal) -> OrderRequest:
    """A small protected long, sized to TARGET_NOTIONAL or the venue minimum."""
    instrument = await gateway.get_instrument(SYMBOL)
    ticker = await gateway.fetch_ticker(SYMBOL)
    price = ticker.price_for(OrderSide.BUY)
    quantity = instrument.normalize_quantity(TARGET_NOTIONAL / price)
    quantity = max(quantity, instrument.min_quantity)
    return OrderRequest(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        time_in_force=TimeInForce.GTC,
        stop_loss_price=instrument.normalize_price(price * (Decimal("1") - stop_pct)),
        take_profit_price=instrument.normalize_price(price * (Decimal("1") + stop_pct * 3)),
        strategy_id="stage_b_harness",
    )


async def check_a_real_order(gateway: BybitGateway) -> Check:
    """(a) A submitted order must exist ON THE VENUE, not merely locally."""
    result = Check("a. REAL ORDER REACHES VENUE")
    try:
        request = await _entry_request(gateway, stop_pct=Decimal("0.05"))
        order = await gateway.submit_order(request)
        result.venue_ids.append(order.venue_order_id or order.order_id)

        await asyncio.sleep(2)
        position = await _position_for(gateway, SYMBOL)
        trades = await gateway.fetch_my_trades(SYMBOL, limit=5)

        if position is None and not trades:
            result.detail = (
                "local record shows submitted but the venue reports no position and no fills"
            )
            return result
        size = position["info"]["size"] if position else "0"
        result.passed = True
        result.detail = f"venue position size={size}, recent venue fills={len(trades)}"
    except Exception as exc:
        result.detail = f"{type(exc).__name__}: {exc}"[:200]
    return result


async def check_b_stop_on_venue(gateway: BybitGateway) -> Check:
    """(b) The stop must be held by the exchange, not only in memory."""
    result = Check("b. STOP ATTACHED ON VENUE")
    try:
        position = await _position_for(gateway, SYMBOL)
        if position is None:
            result.detail = "no open position to inspect"
            return result
        info = position.get("info", {})
        stop = info.get("stopLoss")
        if stop in (None, "", "0", 0):
            result.detail = "venue reports NO server-side stopLoss - position is naked"
            return result
        result.passed = True
        result.detail = f"venue stopLoss={stop}"
    except Exception as exc:
        result.detail = f"{type(exc).__name__}: {exc}"[:200]
    return result


async def check_c_drawdown_flatten(gateway: BybitGateway) -> Check:
    """(c) A drawdown breach must latch the switch AND close on the venue."""
    from quantflow.core.clock import SystemClock
    from quantflow.core.config import RiskSettings
    from quantflow.risk.engine import RiskEngine
    from quantflow.risk.monitor import LossMonitor

    result = Check("c. DRAWDOWN FLATTEN REACHES VENUE")
    try:
        before = await _position_for(gateway, SYMBOL)
        if before is None:
            result.detail = "no open position to flatten"
            return result

        closed_ids: list[str] = []

        async def flatten(reason: str) -> list[Any]:  # noqa: ARG001 - monitor callback shape
            """Close reduce-only through the real gateway."""
            position = await _position_for(gateway, SYMBOL)
            if position is None:
                return []
            size = Decimal(str(position["info"]["size"]))
            instrument = await gateway.get_instrument(SYMBOL)
            closing = OrderRequest(
                symbol=SYMBOL,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=instrument.normalize_quantity(size),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
                strategy_id="stage_b_flatten",
            )
            order = await gateway.submit_order(closing)
            closed_ids.append(order.venue_order_id or order.order_id)
            return [order]

        # Tightened so the monitor engages on the next sample. Restored implicitly - this is
        # a local object, the deployed config is never written to.
        tight = RiskSettings(
            max_drawdown_pct=Decimal("0.0001"),
            max_weekly_loss_pct=Decimal("0.00005"),
            max_daily_loss_pct=Decimal("0.00001"),
        )
        risk = RiskEngine(tight, clock=SystemClock())
        await risk.start()
        monitor = LossMonitor(risk, tight, flatten=flatten)

        from quantflow.domain.portfolio import PortfolioSnapshot

        breach = await monitor.check(
            PortfolioSnapshot(
                timestamp=SystemClock().now(),
                base_currency="USDT",
                cash=Decimal("9000"),
                positions=(),
                mark_prices={},
                peak_equity=Decimal("10000"),
                day_start_equity=Decimal("10000"),
            )
        )
        await asyncio.sleep(2)
        after = await _position_for(gateway, SYMBOL)

        result.venue_ids.extend(closed_ids)
        if breach is None:
            result.detail = "monitor did not detect the breach"
        elif not risk.kill_switch.engaged:
            result.detail = "breach detected but the kill switch did not latch"
        elif after is not None:
            result.detail = f"switch latched but the venue still shows size={after['info']['size']}"
        else:
            result.passed = True
            result.detail = (
                f"breach={breach.rule}, switch latched, venue position closed reduce-only"
            )
    except Exception as exc:
        result.detail = f"{type(exc).__name__}: {exc}"[:200]
    return result


async def check_d_restart_reconciliation(gateway: BybitGateway) -> Check:
    """(d) A restart must adopt the venue position and confirm its stop."""
    from quantflow.live.reconcile import parse_venue_positions, reconcile

    result = Check("d. RESTART RECONCILIATION")
    try:
        request = await _entry_request(gateway, stop_pct=Decimal("0.05"))
        order = await gateway.submit_order(request)
        result.venue_ids.append(order.venue_order_id or order.order_id)
        await asyncio.sleep(3)

        # A restarted process knows nothing. Reconciliation runs against the venue before
        # any signal is acted on, which is the whole point: a position the system does not
        # know it holds cannot be sized against or stopped out by logic that never sees it.
        raw = await gateway.fetch_positions()
        venue_positions = parse_venue_positions(raw)
        report = reconcile(venue_positions, known_symbols=set())

        if not venue_positions:
            result.detail = "no venue position to reconcile against"
            return result

        adopted = [p for p in report.unknown_locally if p.symbol == SYMBOL]
        if not adopted:
            result.detail = "reconciliation did not surface the venue position"
            return result

        position = adopted[0]
        if not position.is_protected:
            result.detail = (
                f"adopted {position.symbol} qty={position.quantity} but the venue holds NO "
                "stop - reconciliation correctly refuses to trade around it"
            )
            return result

        result.passed = True
        result.detail = (
            f"adopted {position.symbol} qty={position.quantity} "
            f"stop={position.stop_loss_price} confirmed on venue; "
            f"{report.summary()}; safe_to_trade={report.is_safe_to_trade}"
        )
    except Exception as exc:
        result.detail = f"{type(exc).__name__}: {exc}"[:200]
    return result


async def check_e_cleanup(gateway: BybitGateway) -> Check:
    """(e) Leave the demo account flat."""
    result = Check("e. CLEANUP")
    try:
        for order in await gateway.fetch_open_orders(SYMBOL):
            with_id = order.venue_order_id or order.order_id
            try:
                await gateway.cancel_order(order.order_id, SYMBOL, quantity=order.quantity)
                result.venue_ids.append(f"cancelled:{with_id}")
            except (ExchangeError, ValueError) as exc:
                # Cleanup is best-effort; a cancel that fails must not mask the results.
                result.venue_ids.append(f"cancel-failed:{with_id}:{type(exc).__name__}")

        position = await _position_for(gateway, SYMBOL)
        if position is not None:
            size = Decimal(str(position["info"]["size"]))
            instrument = await gateway.get_instrument(SYMBOL)
            closing = OrderRequest(
                symbol=SYMBOL,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=instrument.normalize_quantity(size),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
                strategy_id="stage_b_cleanup",
            )
            order = await gateway.submit_order(closing)
            result.venue_ids.append(f"closed:{order.venue_order_id or order.order_id}")
            await asyncio.sleep(2)

        remaining = await _position_for(gateway, SYMBOL)
        result.passed = remaining is None
        result.detail = "demo account flat" if result.passed else "a position remains open"
    except Exception as exc:
        result.detail = f"{type(exc).__name__}: {exc}"[:200]
    return result


async def main() -> int:
    settings = get_settings()
    exchange = settings.exchange

    line("=" * 78)
    line("STAGE B - LIVE DEMO VENUE VERIFICATION")
    line("=" * 78)
    line(f"env      : {exchange.resolved_env.value}")
    line(f"endpoint : {exchange.endpoint}")
    line(f"market   : {exchange.market_type.value}")

    # Hard abort: this harness sends real orders and must never reach production.
    if exchange.resolved_env is not ExchangeEnv.DEMO:
        line(f"\nABORT: env is {exchange.resolved_env.value}, expected demo")
        return 1
    try:
        exchange.assert_not_mainnet()
    except ValueError as exc:
        line(f"\nABORT: {exc}")
        return 1
    if "api-demo" not in exchange.endpoint:
        line(f"\nABORT: endpoint {exchange.endpoint} is not api-demo")
        return 1
    line("safety   : endpoint confirmed api-demo, mainnet refused\n")

    gateway = BybitGateway(exchange)
    checks: list[Check] = []
    try:
        await gateway.connect()
        checks.append(await check_a_real_order(gateway))
        checks.append(await check_b_stop_on_venue(gateway))
        checks.append(await check_c_drawdown_flatten(gateway))
        checks.append(await check_d_restart_reconciliation(gateway))
        checks.append(await check_e_cleanup(gateway))
    finally:
        await gateway.aclose()

    line("")
    line("=" * 78)
    line(f"{'CHECK':<36}{'RESULT':<8}DETAIL")
    line("-" * 78)
    for check in checks:
        line(f"{check.name:<36}{'PASS' if check.passed else 'FAIL':<8}{check.detail}")
        for venue_id in check.venue_ids:
            line(f"{'':<44}venue id: {venue_id}")
    line("=" * 78)
    overall = all(check.passed for check in checks)
    line(f"STAGE B: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
