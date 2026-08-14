#!/usr/bin/env python
"""Flatten every position and cancel every working order on the Bybit DEMO account.

Written for one situation: the venue holds positions a restarted session knows nothing
about, so venue state and bot state disagree. Closing them makes the two agree at zero.

DEMO ONLY, checked before anything is cancelled or closed. This script closes positions -
run against mainnet it would liquidate a real book, so the guard is not a formality.

Cancel first, then close. A reduce-only exit sent while a stop conditional is still working
can leave the stop orphaned on a position that no longer exists.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

from quantflow.core.config import ExchangeEnv, get_settings
from quantflow.exchange.bybit.rest import BybitGateway


async def main() -> int:
    settings = get_settings()
    env = settings.exchange.resolved_env
    if env is not ExchangeEnv.DEMO:
        sys.stderr.write(f"REFUSING: environment is '{env.value}', not demo. Nothing touched.\n")
        return 2

    gateway = BybitGateway(settings.exchange)
    try:
        client = gateway._client
        await client.load_markets()

        positions = [
            p for p in await gateway.fetch_positions() if Decimal(str(p.get("contracts") or 0)) != 0
        ]
        if not positions:
            print("no open positions — nothing to flatten")

        venue_symbols = sorted({str(p.get("symbol")) for p in positions})

        # 1. Cancel working orders (the StopLoss / TakeProfit conditionals).
        cancelled = 0
        for symbol in venue_symbols:
            for order in await client.fetch_open_orders(symbol):
                info = order.get("info", {})
                try:
                    await client.cancel_order(order["id"], symbol)
                    cancelled += 1
                    print(
                        f"cancelled {symbol:<16} {info.get('stopOrderType') or 'order':<12} "
                        f"trigger={info.get('triggerPrice') or '-'}"
                    )
                except Exception as exc:
                    print(f"  ! cancel failed {symbol} {order['id']}: {str(exc)[:120]}")

        # 2. Close each position with a reduce-only market order the opposite way.
        closed = 0
        for position in positions:
            symbol = str(position.get("symbol"))
            side = str(position.get("side") or "").lower()
            quantity = abs(Decimal(str(position.get("contracts") or 0)))
            if quantity == 0:
                continue
            exit_side = "sell" if side == "long" else "buy"
            try:
                await client.create_order(
                    symbol,
                    "market",
                    exit_side,
                    float(quantity),
                    None,
                    {"reduceOnly": True, "category": "linear"},
                )
                closed += 1
                print(f"closed    {symbol:<16} {side} {quantity} via {exit_side} market reduceOnly")
            except Exception as exc:
                print(f"  ! close failed {symbol}: {str(exc)[:160]}")

        print(f"\ncancelled {cancelled} order(s), closed {closed} position(s)")

        # 3. Read back. The venue is the authority, not our own count of what we sent.
        await asyncio.sleep(2)
        remaining = [
            p for p in await gateway.fetch_positions() if Decimal(str(p.get("contracts") or 0)) != 0
        ]
        still_open = 0
        for symbol in venue_symbols:
            still_open += len(await client.fetch_open_orders(symbol))
        print(f"VERIFY: {len(remaining)} position(s), {still_open} working order(s) remain")
        return 0 if not remaining and not still_open else 1
    finally:
        await gateway.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
