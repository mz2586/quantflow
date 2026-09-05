/**
 * Whole-app resilience tests.
 *
 * The failure these exist to prevent has happened: `risk?.kill_switch.engaged` guarded
 * `risk` and then dereferenced `kill_switch` unguarded. `undefined.engaged` threw, React
 * unmounted the entire tree, and the operator got a blank page — which during an incident
 * is indistinguishable from a dead machine.
 *
 * Every case below feeds the app a payload shaped like something a stale, half-rebuilt or
 * failing API would send, and asserts the page still renders.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function respond(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: new Headers(),
    json: () => Promise.resolve(body),
  } as Response;
}

/** Route each endpoint to a fixture, defaulting to an empty object. */
function stubApi(routes: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      const key = Object.keys(routes).find((route) => url.includes(route));
      if (key === undefined) return Promise.resolve(respond({}));
      const value = routes[key];
      if (value instanceof Error) return Promise.reject(value);
      return Promise.resolve(respond(value));
    }),
  );
  // The websocket is a latency improvement, never the source of truth; stubbed so the
  // tests exercise the polling path.
  vi.stubGlobal(
    "WebSocket",
    class {
      close(): void {
        // The socket is a latency improvement; these tests exercise the polling path.
      }
    },
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("blank-page resistance", () => {
  it("renders when every endpoint returns an empty object", async () => {
    stubApi({});
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("QuantFlow")).toBeTruthy();
    });
  });

  it("renders when every endpoint returns null fields", async () => {
    stubApi({
      "/dashboard/summary": {
        session: null,
        status: null,
        venue: null,
        risk: null,
        trading_performance: null,
        session_equity: null,
        fees: null,
        book_reconciliation: null,
        decisions: null,
        engine: null,
      },
      "/dashboard/equity": { points: null },
      "/dashboard/trades": { trades: null },
      "/dashboard/analytics": { by_strategy: null, by_symbol: null, by_side: null },
      "/dashboard/decisions": { decisions: null, summary: null },
      "/dashboard/freshness": {},
      "/dashboard/asset-classes": { asset_classes: null },
    });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("QuantFlow")).toBeTruthy();
    });
  });

  it("renders when lists arrive as objects instead of arrays", async () => {
    stubApi({
      "/dashboard/summary": {
        venue: { positions: { nope: true }, open_orders: "not-an-array" },
      },
      "/dashboard/trades": { trades: { nope: true } },
      "/dashboard/analytics": { by_strategy: 42 },
      "/dashboard/equity": { points: "oops" },
    });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("QuantFlow")).toBeTruthy();
    });
  });

  /** The exact original crash: `risk` present, `kill_switch` absent. */
  it("renders when risk is present but its nested object is missing", async () => {
    stubApi({ "/dashboard/summary": { risk: {} } });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("QuantFlow")).toBeTruthy();
    });
  });

  it("renders when every request fails", async () => {
    stubApi({
      "/dashboard/summary": new TypeError("Failed to fetch"),
      "/dashboard/equity": new TypeError("Failed to fetch"),
      "/dashboard/trades": new TypeError("Failed to fetch"),
      "/dashboard/analytics": new TypeError("Failed to fetch"),
      "/dashboard/decisions": new TypeError("Failed to fetch"),
      "/dashboard/freshness": new TypeError("Failed to fetch"),
      "/dashboard/asset-classes": new TypeError("Failed to fetch"),
      "/readyz": new TypeError("Failed to fetch"),
    });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("QuantFlow")).toBeTruthy();
    });
    await waitFor(() => {
      expect(screen.getAllByText(/temporarily unavailable/i).length).toBeGreaterThan(0);
    });
  });
});

describe("live state", () => {
  const summary = {
    generated_at: "2026-08-14T13:48:00+00:00",
    session: { session_id: "demo-15m-20260813", status: "running", timeframe: "15m" },
    status: { state: "TRADING", detail: "5 position(s) open on the venue" },
    venue: {
      venue: "bybit",
      network: "demo",
      account: {
        trading_equity_usdt: "49899.34635401",
        available_usdt: "37470.99081325",
        total_portfolio_value_usdt: null,
        unpriced_assets: [],
        other_assets: [],
      },
      position_count: 5,
      open_order_count: 10,
      freshness: { available: true, stale: false, fetched_at: "2026-08-14T13:48:00+00:00" },
    },
    trading_performance: {
      closed_trades: 70,
      net_realized_pnl: "-64.805979060000",
      gross_realized_pnl: "6.289550000000",
      total_fees: "71.095529060000",
      win_rate: "0.4",
      profit_factor: "0.3743",
    },
    session_equity: { peak_equity: "49940.479558200000", current_drawdown_pct: "0.0009014" },
    book_reconciliation: { venue_open_positions: 5, venue_open_orders: 10 },
    risk: { kill_switch_engaged: false, trading_halted: false },
    engine: { timeframe: "15m", pid: null, symbols: ["BTC/USDT"] },
  };

  it("shows the derived trading status prominently", async () => {
    stubApi({ "/dashboard/summary": summary });
    render(<App />);
    await waitFor(() => {
      // Twice by design: the headline tile above the fold, and the full state in the
      // trading-status panel that explains it.
      expect(screen.getAllByText("TRADING").length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.getByText(/5 position\(s\) open on the venue/)).toBeTruthy();
  });

  it("labels a demo venue as virtual funds, never as real money", async () => {
    stubApi({ "/dashboard/summary": summary });
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/demo · virtual funds/i)).toBeTruthy();
    });
    expect(screen.queryByText(/REAL MONEY/)).toBeNull();
  });

  it("never renders the naive cross-asset total anywhere on the page", async () => {
    stubApi({ "/dashboard/summary": summary });
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("TRADING").length).toBeGreaterThan(0);
    });
    const body = document.body.textContent;
    expect(body).not.toContain("99,904.01");
    expect(body).not.toContain("99,901.35");
  });

  it("reports the engine PID as NOT RECORDED rather than inventing one", async () => {
    stubApi({ "/dashboard/summary": summary });
    render(<App />);
    // Engine internals moved off the first screen: the operator opens Diagnostics for
    // them, and the claim under test is that the panel still refuses to invent a pid.
    await waitFor(() => {
      expect(screen.getAllByText("TRADING").length).toBeGreaterThan(0);
    });
    fireEvent.click(screen.getByRole("button", { name: "Diagnostics" }));
    await waitFor(() => {
      expect(screen.getByText(/engine pid/i)).toBeTruthy();
    });
    expect(screen.getByText(/no pid file/i)).toBeTruthy();
  });

  it("surfaces a stale venue reading instead of presenting it as live", async () => {
    stubApi({
      "/dashboard/summary": {
        ...summary,
        venue: {
          ...summary.venue,
          freshness: {
            available: true,
            stale: true,
            error: "timed out after 8s",
            fetched_at: "2026-08-14T13:20:00+00:00",
          },
        },
      },
    });
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText(/STALE/i).length).toBeGreaterThan(0);
    });
  });

  it("shows DISCONNECTED when the venue cannot be reached", async () => {
    stubApi({
      "/dashboard/summary": {
        ...summary,
        status: { state: "DISCONNECTED", detail: "the venue could not be reached" },
        venue: { freshness: { available: false, stale: false, error: "connection refused" } },
      },
    });
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("DISCONNECTED").length).toBeGreaterThan(0);
    });
  });

  it("polls without any user interaction", async () => {
    stubApi({ "/dashboard/summary": summary });
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("TRADING").length).toBeGreaterThan(0);
    });
    const calls = (globalThis.fetch as unknown as { mock: { calls: unknown[] } }).mock.calls.length;
    expect(calls).toBeGreaterThan(1);
  });
});
