/**
 * Resilience tests for the dashboard shell.
 *
 * These exist because of a real outage: a stale API returned a `/risk/status` payload with
 * no `kill_switch` object, `risk?.kill_switch.engaged` threw during render, React unmounted
 * the entire tree, and the operator got a blank white page — no equity, no positions, no
 * kill switch, and no clue why. A dashboard for a live trading system must degrade, not
 * disappear, so "the API sent something incomplete" is asserted here rather than trusted.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

/** A complete, well-formed payload for every endpoint the dashboard calls. */
function completePayloads(): Record<string, unknown> {
  return {
    "/readyz": { ready: true, components: [], trading_mode: "live", kill_switch_engaged: false },
    "/api/v1/portfolio/sessions": [
      {
        session_id: "live-15m-test",
        mode: "live",
        status: "running",
        strategy_id: "orchestrator",
        symbols: ["ETH/USDT"],
        timeframe: "15m",
        starting_equity: "10000",
        final_equity: null,
        started_at: "2026-08-10T00:00:00Z",
        finished_at: null,
        error: null,
      },
    ],
    "/api/v1/portfolio/trades": [],
    "/api/v1/portfolio": {
      base_currency: "USDT",
      equity: "10000",
      cash: "9000",
      starting_equity: "10000",
      total_return_pct: "0",
      realized_pnl: "0",
      unrealized_pnl: "0",
      fees_paid: "0",
      gross_exposure: "0",
      leverage: "0",
      drawdown_pct: "0",
      daily_pnl: "0",
      position_count: 0,
      positions: [],
    },
    "/api/v1/risk/status": {
      trading_halted: false,
      kill_switch: { engaged: false, reason: null, engaged_at: null, engaged_by: null },
      limits: {
        max_position_pct: "0.1",
        max_total_exposure_pct: "0.5",
        max_concurrent_positions: 5,
        max_daily_loss_pct: "0.05",
        max_drawdown_pct: "0.2",
        max_leverage: "3",
        require_stop_loss: true,
        max_order_notional: "5000",
        max_orders_per_minute: 10,
      },
      headroom: {},
      sizer: "fixed_fractional",
      rules: ["daily_loss"],
    },
    "/api/v1/risk/events": [],
    "/api/v1/strategies": [],
    "/api/v1/market/series": [],
    "/api/v1/analytics/review": {
      trade_count: 0,
      by_strategy: [],
      by_symbol: [],
      by_side: [],
      streaks: { longest_win: 0, longest_loss: 0, current: 0 },
      concentration: {
        top_trade_share: "0",
        profit_without_best: "0",
        is_concentrated: false,
        rests_on_one_trade: false,
      },
      warnings: [],
    },
    "/api/v1/account/fills": {
      symbol: "BTC/USDT",
      count: 0,
      realized_pnl: "0",
      total_fees: "0",
      fills: [],
    },
    "/api/v1/account": {
      venue: "bybit",
      network: "demo",
      authenticated: true,
      total_balance: "10000",
      available_balance: "9000",
      balances: [],
      positions: [],
      position_count: 0,
      unrealized_pnl: "0",
      open_orders: [],
      open_order_count: 0,
    },
  };
}

/**
 * Route a request to its payload. Longest match wins, so `/api/v1/account/fills` is not
 * served by the `/api/v1/account` entry.
 */
function install(payloads: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: unknown) => {
      const url = String(input).split("?")[0] ?? "";
      const key = Object.keys(payloads)
        .filter((candidate) => url.endsWith(candidate))
        .sort((a, b) => b.length - a.length)[0];
      return Promise.resolve(
        new Response(JSON.stringify(key === undefined ? {} : payloads[key]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }),
  );
}

beforeEach(() => {
  // recharts measures its container; jsdom has neither observer.
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe(): void {
        // no-op: nothing here depends on layout measurement
      }
      unobserve(): void {
        // no-op
      }
      disconnect(): void {
        // no-op
      }
    },
  );
  // The dashboard opens a websocket for live updates. It is a latency improvement only,
  // and none of these assertions depend on it.
  vi.stubGlobal(
    "WebSocket",
    class {
      close(): void {
        // no-op: the socket is never opened in these tests
      }
    },
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App resilience to incomplete API payloads", () => {
  it("renders the dashboard when every payload is well-formed", async () => {
    install(completePayloads());
    render(<App />);

    expect(await screen.findByText("QuantFlow")).toBeTruthy();
    await waitFor(() => {
      expect(screen.getByText("Halt trading")).toBeTruthy();
    });
  });

  it("still renders when /risk/status omits kill_switch — the blank-page regression", async () => {
    const payloads = completePayloads();
    // Exactly what a stale API returned: a risk status with no kill switch object at all.
    const stale = { ...(payloads["/api/v1/risk/status"] as Record<string, unknown>) };
    delete stale.kill_switch;
    payloads["/api/v1/risk/status"] = stale;
    install(payloads);

    render(<App />);

    // The page must still be there. Before the fix this threw and unmounted everything.
    expect(await screen.findByText("QuantFlow")).toBeTruthy();
    await waitFor(() => {
      expect(screen.getByText("Registered strategies (0)")).toBeTruthy();
    });
    // A missing kill switch reads as "not engaged" rather than taking the page down.
    expect(screen.getByText("Halt trading")).toBeTruthy();
  });

  it("still renders when payloads omit every nested object and list", async () => {
    // The general case: an API a version behind, or one returning half-built responses.
    install({
      "/readyz": { ready: false, trading_mode: "live", kill_switch_engaged: false },
      "/api/v1/portfolio/sessions": [
        { session_id: "s1", mode: "live", status: "running", strategy_id: "x", timeframe: "15m" },
      ],
      "/api/v1/portfolio/trades": [],
      "/api/v1/portfolio": { base_currency: "USDT", equity: "1", cash: "1", daily_pnl: "0" },
      "/api/v1/risk/status": { trading_halted: false },
      "/api/v1/risk/events": [],
      "/api/v1/strategies": [],
      "/api/v1/market/series": [],
      "/api/v1/analytics/review": { trade_count: 0 },
      "/api/v1/account/fills": { symbol: "BTC/USDT", count: 0 },
      "/api/v1/account": { authenticated: true, total_balance: "1", available_balance: "1" },
    });

    render(<App />);

    expect(await screen.findByText("QuantFlow")).toBeTruthy();
    await waitFor(() => {
      expect(screen.getByText("Registered strategies (0)")).toBeTruthy();
    });
  });

  it("still renders when list endpoints return objects instead of arrays", async () => {
    const payloads = completePayloads();
    payloads["/api/v1/strategies"] = { unexpected: "shape" };
    payloads["/api/v1/risk/events"] = { unexpected: "shape" };
    payloads["/api/v1/portfolio/trades"] = { unexpected: "shape" };
    install(payloads);

    render(<App />);

    expect(await screen.findByText("QuantFlow")).toBeTruthy();
    await waitFor(() => {
      expect(screen.getByText("Registered strategies (0)")).toBeTruthy();
    });
  });
});
