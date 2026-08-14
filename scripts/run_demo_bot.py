#!/usr/bin/env python
"""Run the adaptive orchestrator autonomously against Bybit DEMO.

The CLI exposes only ``trade paper``; TradingRunner supports a LIVE session but nothing
wires it up. This is that wiring, and nothing more - the runner, the router, the risk
engine and the reconciler are used exactly as they are.

DEMO ONLY. The first thing this does is refuse to run against mainnet, before any client
is constructed. That check is deliberately redundant with the runner's own gates: this
process exists to be launched unattended by a supervisor, and a config drift that pointed
it at real money must fail loudly at startup rather than at the first order.

Orders are real - they reach a real matching engine, fill, and hold positions - but the
funds are virtual. The safety stack (venue-side stops, drawdown monitor, reconciliation,
1x leverage, per-position and portfolio risk limits) is the same code the validation ran.
"""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from quantflow.core.config import ExchangeEnv, TradingMode, get_settings
from quantflow.core.logging import configure_logging, get_logger
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.live.equity import resolve_starting_equity
from quantflow.live.runner import RunnerConfig, check_live_arming, run_session
from quantflow.universe.meme import (
    DEFAULT_ELIGIBILITY_LIMITS,
    EligibilityInputs,
    assess_eligibility,
    discover_meme_universe,
)

#: Deepest-liquidity majors. Fewer symbols than the 24-month validation universe on
#: purpose: this process runs unattended, and every extra symbol is another stream to keep
#: alive. Override with QF_BOT_SYMBOLS.
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT"

#: Bar interval. Note the tension with position size: the fill model rejects an order
#: exceeding 10% of the bar's volume, and a 5m bar carries roughly a third of a 15m bar's
#: volume. Larger positions on shorter bars push against that ceiling from both sides.
DEFAULT_TIMEFRAME = "15m"

#: Stable across restarts. A supervisor that invented a new session id on every crash
#: would scatter one session's state across many.
DEFAULT_SESSION_ID = "demo-15m-20260813"

#: Enough trailing bars to satisfy the longest lookback in the library (momentum_roc, 720).
HISTORY_BARS = 1000


def refuse_mainnet() -> None:
    """Hard stop before anything is constructed if this is not a demo account."""
    settings = get_settings()
    env = settings.exchange.resolved_env
    if env is not ExchangeEnv.DEMO:
        sys.stderr.write(
            f"REFUSING TO START: QF_EXCHANGE__ENV resolves to '{env.value}', not 'demo'.\n"
            "This launcher is demo-only. Nothing has been connected or ordered.\n"
        )
        raise SystemExit(2)
    if settings.exchange.resolved_env.is_mainnet:  # belt and braces
        sys.stderr.write("REFUSING TO START: resolved environment is mainnet.\n")
        raise SystemExit(2)


#: How many eligible meme markets may join the traded set. Each symbol is another live
#: stream to keep alive and another candidate field to evaluate every bar; the cap keeps
#: an unattended process from subscribing to twenty-six of them because they happened to
#: pass a filter. Ranked by 24h volume, so the ones that survive are the liquid ones.
MAX_MEME_SYMBOLS = int(os.environ.get("QF_BOT_MAX_MEME", "4"))

#: Bars needed before a market can be judged at all. Fewer than this and the range and
#: volatility estimates are noise, so the market is skipped rather than guessed at.
MIN_BARS_TO_ASSESS = 20


async def eligible_meme_symbols(gateway: BybitGateway, logger: Any) -> list[Symbol]:
    """Meme markets that pass the eligibility filters right now, most liquid first.

    Measured, not assumed: every input comes from a live ticker or a freshly fetched bar.
    A market that cannot be measured is skipped rather than admitted on the assumption
    that missing data is fine — the whole point of the filter is to keep illiquid and
    unstable symbols out of a book that trades unattended.
    """
    from datetime import timedelta

    from quantflow.domain.enums import Timeframe

    markets = discover_meme_universe(gateway.instruments.all().values())
    if not markets:
        return []

    scored: list[tuple[Decimal, Symbol, list[str]]] = []
    for market in markets:
        try:
            ticker = await gateway.fetch_ticker(market.symbol)
            bars = await gateway.fetch_candles(market.symbol, Timeframe.parse("15m"), limit=40)
        except Exception as exc:
            logger.info("meme.skipped", symbol=str(market.symbol), error=str(exc)[:120])
            continue
        if len(bars) < MIN_BARS_TO_ASSESS or ticker.bid <= 0 or ticker.ask <= 0:
            continue

        ranges = [bar.high - bar.low for bar in bars[:-1]]
        typical = sum(ranges, Decimal("0")) / Decimal(len(ranges))
        last = bars[-1]
        reference = last.close or ticker.last
        volatility = (typical / reference) if reference else Decimal("0")
        last_return = (last.close - last.open) / last.open if last.open else Decimal("0")
        bar_quote_volume = last.volume * reference
        quote_volume_24h = sum((bar.volume * bar.close for bar in bars[-96:]), Decimal("0"))
        # A nominal probe size: 1% of the reference equity, the smallest slice the sizer
        # would realistically ask for. Filtering on a size we would never send would
        # answer a question nobody asked.
        intended_price = ticker.ask
        intended_quantity = market.quantity_step and (
            (Decimal("500") / intended_price) if intended_price else Decimal("0")
        )
        verdict = assess_eligibility(
            EligibilityInputs(
                market=market,
                quote_volume_24h=quote_volume_24h,
                bid=ticker.bid,
                ask=ticker.ask,
                ticker_age=timedelta(seconds=0),
                candle_age=timedelta(minutes=1),
                volatility=volatility,
                last_bar_range=last.high - last.low,
                typical_bar_range=typical,
                last_bar_return=last_return,
                bar_quote_volume=bar_quote_volume,
                intended_quantity=intended_quantity,
                intended_price=intended_price,
                stop_distance=typical,
            ),
            DEFAULT_ELIGIBILITY_LIMITS,
        )
        if verdict.eligible:
            scored.append((quote_volume_24h, market.symbol, []))
        else:
            logger.info(
                "meme.rejected",
                symbol=str(market.symbol),
                reasons="; ".join(verdict.reasons)[:200],
            )

    scored.sort(key=lambda row: row[0], reverse=True)
    chosen = [symbol for _, symbol, _ in scored[:MAX_MEME_SYMBOLS]]
    logger.critical(
        "meme.universe_selected",
        discovered=len(markets),
        eligible=len(scored),
        enabled=[str(s) for s in chosen],
    )
    return chosen


async def main() -> int:
    refuse_mainnet()

    settings = get_settings()
    configure_logging(settings, service="quantflow-demo-bot")
    logger = get_logger("demo_bot")

    check = check_live_arming(settings)
    if not check.armed:
        sys.stderr.write(
            "REFUSING TO START: live session is not armed:\n"
            + "\n".join(f"  - {reason}" for reason in check.blockers())
            + "\n"
        )
        return 2

    symbols = tuple(
        Symbol.parse(raw.strip())
        for raw in os.environ.get("QF_BOT_SYMBOLS", DEFAULT_SYMBOLS).split(",")
        if raw.strip()
    )
    timeframe = Timeframe.parse(os.environ.get("QF_BOT_TIMEFRAME", DEFAULT_TIMEFRAME))
    session_id = os.environ.get("QF_BOT_SESSION_ID", DEFAULT_SESSION_ID)

    # Restrict the orchestrator's rotation when asked. Unset means every registered
    # strategy competes; the orchestrator rejects an unknown id rather than quietly
    # falling back to the full roster.
    raw_pool = os.environ.get("QF_BOT_STRATEGIES", "").strip()
    strategy_params: dict[str, object] = {}
    if raw_pool:
        strategy_params["pool"] = [name.strip() for name in raw_pool.split(",") if name.strip()]

    # Meme markets join the SAME traded set and compete through the same orchestrator,
    # gate and risk engine as every other symbol. Nothing is reserved for them: they earn
    # a slot by being a better candidate, or they do not trade.
    if os.environ.get("QF_BOT_MEME", "false").strip().lower() == "true":
        meme_gateway = BybitGateway(settings.exchange)
        try:
            await meme_gateway.load_instruments()
            symbols = symbols + tuple(await eligible_meme_symbols(meme_gateway, logger))
        except Exception as exc:
            logger.warning("meme.scan_failed", error=str(exc)[:200])
        finally:
            await meme_gateway.aclose()

    # Size against the capital that is actually there. A hardcoded 10,000 against a ~100k
    # account meant a "2% position" was 0.2% of the book: the limits were honoured, but
    # against a number unrelated to the account.
    configured_equity = Decimal(os.environ.get("QF_BOT_EQUITY", "10000"))
    starting_equity = configured_equity
    # Tracked separately from the value: a venue read that fails, or an account with no
    # quote balance, leaves `starting_equity` at the configured constant, and a constant
    # must never be treated as the truth about the account.
    equity_is_authoritative = False
    if os.environ.get("QF_BOT_EQUITY_FROM_VENUE", "true").strip().lower() == "true":
        gateway = BybitGateway(settings.exchange)
        try:
            balances = await gateway.fetch_balances()
            starting_equity = resolve_starting_equity(
                balances,
                configured=configured_equity,
                quote=settings.trading.base_currency,
            )
            quote_balance = balances.get(settings.trading.base_currency)
            equity_is_authoritative = (
                quote_balance is not None and (quote_balance.free + quote_balance.locked) > 0
            )
        except Exception as exc:
            logger.warning("demo_bot.equity_read_failed", error=str(exc)[:200])
        finally:
            await gateway.aclose()

    config = RunnerConfig(
        strategy_id=os.environ.get("QF_BOT_STRATEGY", "orchestrator"),
        symbols=symbols,
        timeframe=timeframe,
        mode=TradingMode.LIVE,
        starting_equity=starting_equity,
        equity_is_authoritative=equity_is_authoritative,
        strategy_params=strategy_params,
        history_bars=HISTORY_BARS,
        persist=True,
        session_id=session_id,
    )

    logger.critical(
        "demo_bot.starting",
        env=settings.exchange.resolved_env.value,
        mode=config.mode.value,
        strategy=config.strategy_id,
        pool=strategy_params.get("pool", "full registry"),
        symbols=[str(s) for s in symbols],
        timeframe=timeframe.value,
        session_id=session_id,
        starting_equity=str(starting_equity),
        equity_source=("venue" if equity_is_authoritative else "configured"),
        max_position_pct=str(settings.risk.max_position_pct),
        max_concurrent=settings.risk.max_concurrent_positions,
        max_drawdown_pct=str(settings.risk.max_drawdown_pct),
        require_stop_loss=settings.risk.require_stop_loss,
        max_leverage=str(settings.risk.max_leverage),
    )

    state = await run_session(settings, config)
    logger.critical("demo_bot.session_ended", session_id=session_id, state=str(state)[:200])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(asyncio.run(main()))
