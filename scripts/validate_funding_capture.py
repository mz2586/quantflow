#!/usr/bin/env python
"""Backtest market-neutral funding capture on two years of real Bybit funding.

Same discipline as the directional validation: one fixed chronological split, the holdout
never consulted to choose anything, every cost charged, and no parameter fitted.

**The decision rule is parameter-free by construction.** Hold the hedge while the last
published funding rate was positive; be flat otherwise. The threshold is zero — not a
level chosen because it produced a better number. Any "enter above 0.008%" rule would be a
fitted parameter wearing a plausible face, and with 2,224 settlements per symbol there is
more than enough room to fit one that means nothing.

**Where this is deliberately optimistic**, and it matters for how the verdict should be
read:

* The hedge is assumed perfect: the perp and spot legs offset exactly, so basis moves
  contribute nothing. In reality the basis wanders and the hedge is rebalanced, which
  costs more.
* The rate is known at the settlement it is applied to. A live book decides *before* the
  rate prints.
* Only one round trip is charged per position, with no rebalancing crossings.

Every one of those flatters the result. So a negative verdict here is conclusive: the
trade cannot be rescued by better execution of the same idea.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quantflow.neutral.funding_capture import (
    SETTLEMENTS_PER_DAY,
    FundingCaptureParams,
    simulate_funding_capture,
)

#: Repo root, derived from this file's location so the script works from any checkout.
REPO = Path(__file__).resolve().parent.parent
FUNDING_CACHE = REPO / "scratchpad" / "funding-cache.json"
OUT = REPO / "reports" / "funding-capture-validation.json"

#: The same holdout fraction the directional validation used, so the two are comparable.
OOS_FRACTION = 0.30

#: Bybit's published taker fee. Both legs, both ways.
TAKER_FEE = Decimal("0.0006")

#: One basis point of slippage per crossing. Conservative for majors, and stated rather
#: than assumed away.
SLIPPAGE_BPS = Decimal("1")

#: Per-leg notional. The figure is arbitrary and cancels out of every ratio; results scale
#: linearly, so the sign of the verdict does not depend on it.
NOTIONAL = Decimal("10000")


def load_funding() -> dict[str, list[tuple[datetime, Decimal]]]:
    raw = json.loads(FUNDING_CACHE.read_text())
    out: dict[str, list[tuple[datetime, Decimal]]] = {}
    for symbol, stamps in raw.items():
        rows = sorted((datetime.fromisoformat(iso), Decimal(rate)) for iso, rate in stamps)
        out[symbol] = rows
    return out


def decide(rates: list[Decimal]) -> list[bool]:
    """Hold while the previous published rate was positive. No threshold, no lookahead.

    The first settlement is never held: there is no prior rate to have seen, and assuming
    one would be a free lookahead at the start of every series.
    """
    decisions = [False]
    decisions.extend(previous > 0 for previous in rates[:-1])
    return decisions


def decide_always(rates: list[Decimal]) -> list[bool]:
    """Hold the hedge continuously. No timing, no threshold, nothing to fit.

    Included because the sign-following rule turns out to *churn*: rates oscillate around
    zero even in a persistently positive regime, and every flip pays a full round trip. A
    rule that trades less is not a refinement here — it is the simpler hypothesis, and the
    honest one to test alongside. It also eats the negative settlements, which is the cost
    of not timing.
    """
    return [True] * len(rates)


def summarise(symbol: str, rows: list[tuple[datetime, Decimal]]) -> dict[str, object]:
    rates = [rate for _, rate in rows]
    holds = decide(rates)
    split = int(len(rows) * (1 - OOS_FRACTION))

    params = FundingCaptureParams(notional=NOTIONAL, taker_fee=TAKER_FEE, slippage_bps=SLIPPAGE_BPS)
    in_sample = simulate_funding_capture(
        list(zip(rates[:split], holds[:split], strict=True)), params=params
    )
    out_sample = simulate_funding_capture(
        list(zip(rates[split:], holds[split:], strict=True)), params=params
    )

    # The same holdout, the simpler rule: hold throughout, pay one round trip.
    always = decide_always(rates)
    always_is = simulate_funding_capture(
        list(zip(rates[:split], always[:split], strict=True)), params=params
    )
    always_oos = simulate_funding_capture(
        list(zip(rates[split:], always[split:], strict=True)), params=params
    )

    positive = sum(1 for rate in rates if rate > 0)
    return {
        "symbol": symbol,
        "settlements": len(rows),
        "positive_rate_pct": round(100 * positive / len(rows), 1),
        "mean_rate_bps": str(round(sum(rates) / len(rates) * Decimal("10000"), 4)),
        "split_at": rows[split][0].isoformat(),
        "in_sample": in_sample.to_dict(),
        "out_of_sample": out_sample.to_dict(),
        "always_hold_in_sample": always_is.to_dict(),
        "always_hold_out_of_sample": always_oos.to_dict(),
    }


def main() -> int:
    funding = load_funding()
    if not funding:
        print("no funding data")
        return 1

    results = [summarise(symbol, rows) for symbol, rows in sorted(funding.items())]

    total_oos_net = sum(Decimal(str(r["out_of_sample"]["net_pnl"])) for r in results)
    total_oos_funding = sum(Decimal(str(r["out_of_sample"]["funding_collected"])) for r in results)
    total_oos_costs = sum(Decimal(str(r["out_of_sample"]["costs_paid"])) for r in results)
    winners = [r for r in results if Decimal(str(r["out_of_sample"]["net_pnl"])) > 0]

    print(f"\nMARKET-NEUTRAL FUNDING CAPTURE — {len(results)} symbols")
    print(f"notional {NOTIONAL}/leg · taker {TAKER_FEE} · slippage {SLIPPAGE_BPS}bp/crossing")
    print("rule: hold while the previous rate was positive (threshold 0, not fitted)\n")
    header = (
        f"{'symbol':<12}{'pos%':>6}{'mean bp':>9}"
        f"{'IS net':>12}{'OOS net':>12}{'OOS fund':>12}{'OOS cost':>11}{'trips':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        oos = r["out_of_sample"]
        ins = r["in_sample"]
        print(
            f"{r['symbol']:<12}{r['positive_rate_pct']:>6}{r['mean_rate_bps']:>9}"
            f"{float(ins['net_pnl']):>12.2f}{float(oos['net_pnl']):>12.2f}"
            f"{float(oos['funding_collected']):>12.2f}{float(oos['costs_paid']):>11.2f}"
            f"{oos['round_trips']:>7}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL OOS':<12}{'':>6}{'':>9}{'':>12}{float(total_oos_net):>12.2f}"
        f"{float(total_oos_funding):>12.2f}{float(total_oos_costs):>11.2f}"
    )

    coverage = total_oos_funding / total_oos_costs if total_oos_costs else Decimal("0")
    print(f"\nOut-of-sample funding collected covers {coverage:.2f}x of costs paid")
    print(f"Symbols net-positive out-of-sample: {len(winners)} of {len(results)}")
    verdict = "NET-POSITIVE" if total_oos_net > 0 else "NET-NEGATIVE"
    print(f"VERDICT (sign-following rule): {verdict} out-of-sample after all costs.")

    # The simpler rule, same holdout, same costs.
    always_net = sum(Decimal(str(r["always_hold_out_of_sample"]["net_pnl"])) for r in results)
    always_funding = sum(
        Decimal(str(r["always_hold_out_of_sample"]["funding_collected"])) for r in results
    )
    always_costs = sum(Decimal(str(r["always_hold_out_of_sample"]["costs_paid"])) for r in results)
    always_winners = [
        r for r in results if Decimal(str(r["always_hold_out_of_sample"]["net_pnl"])) > 0
    ]
    always_verdict = "NET-POSITIVE" if always_net > 0 else "NET-NEGATIVE"

    print(f"\n{'ALWAYS-HOLD (no timing, one round trip)':<44}")
    header2 = f"{'symbol':<12}{'OOS net':>12}{'OOS fund':>12}{'OOS cost':>11}{'trips':>7}"
    print(header2)
    print("-" * len(header2))
    for r in results:
        a = r["always_hold_out_of_sample"]
        print(
            f"{r['symbol']:<12}{float(a['net_pnl']):>12.2f}"
            f"{float(a['funding_collected']):>12.2f}{float(a['costs_paid']):>11.2f}"
            f"{a['round_trips']:>7}"
        )
    print("-" * len(header2))
    print(
        f"{'TOTAL OOS':<12}{float(always_net):>12.2f}"
        f"{float(always_funding):>12.2f}{float(always_costs):>11.2f}"
    )
    print(f"Symbols net-positive out-of-sample: {len(always_winners)} of {len(results)}")
    print(f"VERDICT (always-hold rule): {always_verdict} out-of-sample after all costs.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "rule": "hold while previous published rate > 0; threshold not fitted",
                "costs": {
                    "taker_fee": str(TAKER_FEE),
                    "slippage_bps_per_crossing": str(SLIPPAGE_BPS),
                    "legs_per_round_trip": 4,
                    "notional_per_leg": str(NOTIONAL),
                },
                "assumptions_that_flatter_the_result": [
                    "perfect delta hedge: basis moves contribute nothing",
                    "rate known at the settlement it is applied to",
                    "no rebalancing crossings beyond the single round trip",
                ],
                "oos_fraction": OOS_FRACTION,
                "settlements_per_day": SETTLEMENTS_PER_DAY,
                "total_oos_net_pnl": str(total_oos_net),
                "total_oos_funding": str(total_oos_funding),
                "total_oos_costs": str(total_oos_costs),
                "symbols_net_positive_oos": len(winners),
                "verdict": verdict,
                "always_hold": {
                    "total_oos_net_pnl": str(always_net),
                    "total_oos_funding": str(always_funding),
                    "total_oos_costs": str(always_costs),
                    "symbols_net_positive_oos": len(always_winners),
                    "verdict": always_verdict,
                },
                "results": results,
            },
            indent=1,
        )
    )
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
