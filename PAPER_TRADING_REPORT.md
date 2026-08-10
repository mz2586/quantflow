# Paper Trading Report

**Generated:** 2026-08-10 14:09 UTC · **Session:** `paper-live`
**Strategy:** volume_breakout · **Timeframe:** 1d
**Status:** running
**Data:** live Bybit V5 · **Execution:** simulated · **Live trading:** disabled

| Metric | Value |
|---|---:|
| Starting equity | 10,000.00 USDT |
| Current equity | **14,311.68 USDT** |
| Net return | **+43.12%** |
| Closed trades | 123 |
| Win rate | **47.2%** (58 wins / 65 losses) |
| Profit factor | **1.56** |
| Max drawdown | 11.40% |
| Average trade duration | 219.9 h (9.2 days) |
| Largest winning trade | +292.62 |
| Largest losing trade | -158.67 |
| Total fees | 263.03 |
| **Profitable** | **YES** |

## Assessment

The strategy is **profitable** over 123 closed trades.

Profit factor 1.56 means 1.56 USDT earned for every 1.00 lost. Win rate is
47.2%, so most trades still lose —
the account grows because the average win (207.69) exceeds the average loss
(118.99), not because losses are rare.

Fees consumed 2.2% of gross profit.

## Caveats

- Fills are simulated. Real execution adds queue position and partial fills that a
  simulator cannot reproduce, and both work against the trader.
- Most trades here come from historical daily bars replayed through the live engine. The
  realtime session adds to them at roughly one trade per fortnight per symbol.
- No strategy tested has beaten simply holding these assets. This configuration wins on
  drawdown (11.4% against 85.9% for buy-and-hold), not on return.
