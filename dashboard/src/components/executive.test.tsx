/**
 * The executive summary's job is to be unambiguous about money.
 *
 * These tests pin the three things that have actually gone wrong on this dashboard
 * before: a cross-asset sum presented as a balance, an "in trades" figure with no stated
 * basis, and a rejection reason that contradicts the position count without saying so.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExecutiveSummary, PnlSection, TradingStatusPanel, UniverseStrip } from "./executive";
import type { AssetClassesResponse, PnlResponse, Summary } from "../lib/api";

const summary: Summary = {
  generated_at: "2026-08-14T19:00:00Z",
  status: { state: "WAITING FOR QUALIFIED SIGNAL", detail: "no candidate qualified" },
  venue: {
    network: "demo",
    authenticated: true,
    account: {
      trading_equity_usdt: "49901.61959601",
      available_usdt: "49901.61959601",
      total_portfolio_value_usdt: "164480.76",
    },
    deployed: { notional_usdt: "0", margin_usdt: "0", position_count: 0 },
    positions: [],
    position_count: 0,
    unrealized_pnl: "0",
  },
  trading_performance: {
    closed_trades: 77,
    gross_realized_pnl: "25.018330",
    total_fees: "90.217362",
    net_realized_pnl: "-65.199032",
  },
  session_equity: { peak_equity: "49940.48", current_drawdown_pct: "0.00078" },
  decisions: {
    evaluated: 26,
    selected: 8,
    declined: 18,
    by_rejection_category: { correlation: 18 },
  },
};

describe("executive summary", () => {
  it("labels in-trades with its basis rather than leaving it ambiguous", () => {
    render(<ExecutiveSummary summary={summary} />);
    expect(screen.getByText(/in trades \(notional\)/i)).toBeTruthy();
    // The margin figure rides alongside, because the two differ by the leverage multiple.
    expect(screen.getByText(/margin/i)).toBeTruthy();
  });

  it("never shows the naive cross-asset total as a balance", () => {
    render(<ExecutiveSummary summary={summary} />);
    const body = document.body.textContent;
    expect(body).toContain("49,901.62");
    expect(body).not.toContain("164,480.76");
  });

  it("shows the capital base drawn from the venue wallet", () => {
    // The wallet holds ~50k and the session is scoped to 10k. Showing only one of the
    // two would either overstate the capital at risk or look like missing funds.
    render(
      <ExecutiveSummary
        summary={{ ...summary, session_equity: { ...summary.session_equity, capital_base: "49774.71" } }}
      />,
    );
    expect(screen.getByText(/capital base/i)).toBeTruthy();
    const body = document.body.textContent;
    expect(body).toContain("49,774.71");
    expect(body).toContain("49,901.62");
  });

  it("shows a loss as a loss", () => {
    render(<ExecutiveSummary summary={summary} />);
    expect(screen.getAllByText("−65.20").length).toBeGreaterThan(0);
  });
});

describe("trading status", () => {
  it("calls out a correlation rejection that contradicts an empty book", () => {
    render(<TradingStatusPanel summary={summary} decisions={null} />);
    expect(screen.getByText("correlation")).toBeTruthy();
    expect(screen.getByText(/do not exist/i)).toBeTruthy();
  });

  it("stays quiet when the rejection reason is consistent with the book", () => {
    const consistent: Summary = {
      ...summary,
      venue: { ...summary.venue, position_count: 3 },
    };
    render(<TradingStatusPanel summary={consistent} decisions={null} />);
    expect(screen.queryByText(/do not exist/i)).toBeNull();
  });
});

describe("profit and loss", () => {
  const pnl: PnlResponse = {
    order: ["TODAY", "SESSION"],
    periods: {
      TODAY: { closed_trades: 35, gross_profit: "10", gross_loss: "44.58", fees: "40", net_pnl: "-34.58", win_rate: "0.5", scope: "this session, since 00:00 UTC" },
      SESSION: { closed_trades: 77, gross_profit: "25.02", gross_loss: "90.22", fees: "90.22", net_pnl: "-65.20", scope: "this session, all time" },
    },
  };

  it("reports gross profit, gross loss and fees rather than only a net figure", () => {
    render(
      <PnlSection pnl={pnl} period="SESSION" onPeriod={() => undefined} chart={<div />} />,
    );
    expect(screen.getByText(/gross profit/i)).toBeTruthy();
    expect(screen.getByText(/gross loss/i)).toBeTruthy();
    expect(screen.getByText(/^fees$/i)).toBeTruthy();
    expect(screen.getByText(/net pnl/i)).toBeTruthy();
  });

  it("marks an undefined profit factor as undefined, never as zero", () => {
    render(
      <PnlSection pnl={pnl} period="SESSION" onPeriod={() => undefined} chart={<div />} />,
    );
    expect(screen.getByText(/undefined — no losing trade/i)).toBeTruthy();
  });
});

describe("market universe", () => {
  const classes: AssetClassesResponse = {
    asset_classes: [
      { asset_class: "crypto", state: "ACTIVE", symbols: ["BTC/USDT"], symbol_count: 1, open_positions: 0 },
      {
        asset_class: "metal",
        state: "ACTIVE, ORDERS BLOCKED",
        symbols: ["XAU/USDT", "XAG/USDT"],
        symbol_count: 2,
        open_positions: 0,
        reason: "venue refuses orders pending a signed product agreement",
      },
    ],
  };

  it("distinguishes a subscribed-but-unorderable class from a disabled one", () => {
    render(
      <UniverseStrip
        classes={classes}
        summary={{ engine: { agreement_codes: ["110123"] } }}
      />,
    );
    expect(screen.getByText("ACTIVE, ORDERS BLOCKED")).toBeTruthy();
    // The operator is told which agreement to sign, not merely that something is wrong.
    expect(screen.getByText(/110123/)).toBeTruthy();
  });
});
