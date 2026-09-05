/**
 * Panel rendering tests.
 *
 * The fixtures are the **real live account** at the time this dashboard was rebuilt, not
 * invented numbers: USDT 49,899.34635401, USDC 50,000, 1 BTC, 1 ETH. That matters because
 * the defect being guarded against produced a plausible-looking figure from exactly this
 * account, and a test with rounder numbers would not have caught it.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  AnalyticsResponse,
  AssetClassesResponse,
  DecisionsResponse,
  LedgerResponse,
  Summary,
} from "../lib/api";
import {
  AnalyticsPanels,
  AssetClassesPanel,
  DecisionsPanel,
  FeesPanel,
  OrdersPanel,
  PositionsPanel,
  TradeLedgerPanel,
  VenueAccountPanel,
} from "./panels";

/** The live venue account, verbatim. */
const LIVE_ACCOUNT: Summary = {
  venue: {
    venue: "bybit",
    network: "demo",
    authenticated: true,
    account: {
      trading_equity_usdt: "49899.34635401",
      available_usdt: "37470.99081325",
      locked_usdt: "12428.35554076",
      quote_asset: "USDT",
      other_assets: [
        {
          asset: "BTC",
          quantity: "1.0",
          free: "1.0",
          locked: "0.0",
          price_usdt: "62678.25",
          value_usdt: "62678.25",
          valuation_source: "bybit BTC/USDT book mid @ 2026-08-14T13:48:00+00:00",
          unpriced_reason: null,
        },
        {
          asset: "ETH",
          quantity: "1.0",
          free: "1.0",
          locked: "0.0",
          price_usdt: "1869.065",
          value_usdt: "1869.065",
          valuation_source: "bybit ETH/USDT book mid @ 2026-08-14T13:48:00+00:00",
          unpriced_reason: null,
        },
        {
          asset: "USDC",
          quantity: "50000.0",
          free: "50000.0",
          locked: "0.0",
          price_usdt: "1.00025",
          value_usdt: "50012.5",
          valuation_source: "bybit USDC/USDT book mid @ 2026-08-14T13:48:00+00:00",
          unpriced_reason: null,
        },
      ],
      unpriced_assets: [],
      total_portfolio_value_usdt: "164459.16135401",
      valuation_method:
        "USDT balance held at par, plus each other asset valued at its current bybit order-book mid against USDT",
      valued_at: "2026-08-14T13:48:00+00:00",
    },
    positions: [],
    position_count: 0,
    unrealized_pnl: "0",
    open_orders: [],
    open_order_count: 0,
    freshness: {
      source: "bybit venue account read",
      fetched_at: "2026-08-14T13:48:00+00:00",
      age_seconds: 2,
      stale: false,
      error: null,
      available: true,
    },
  },
};

describe("venue account — the 99,904.01 regression", () => {
  /**
   * The defect: `total = sum(free + locked for every asset)` produced
   * `49,902.01 + 50,000 + 1 + 1 = 99,904.01` and labelled it "Total balance". It has no
   * unit, and it is roughly double the capital the engine can deploy.
   */
  it("never renders the naive cross-asset sum as a balance", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    const body = document.body.textContent;

    // The exact reported figure, and the arithmetic sum of the live fixture. Neither may
    // appear anywhere on the page in any formatting.
    expect(body).not.toContain("99,904.01");
    expect(body).not.toContain("99904.01");
    expect(body).not.toContain("99,901.35");
    expect(body).not.toContain("99901.35");
  });

  it("never labels anything simply 'Total balance'", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    expect(screen.queryByText(/^total balance$/i)).toBeNull();
  });

  it("reports trading equity as the USDT balance alone", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    expect(screen.getByText(/trading equity \(usdt\)/i)).toBeTruthy();
    expect(screen.getByText("49,899.35 USDT")).toBeTruthy();
  });

  it("reports available USDT as the venue's free USDT", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    expect(screen.getByText("37,470.99 USDT")).toBeTruthy();
  });

  it("lists the other assets separately with their own quantities", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    for (const asset of ["BTC", "ETH", "USDC"]) {
      expect(screen.getByText(asset)).toBeTruthy();
    }
    expect(screen.getByText("50000")).toBeTruthy();
  });

  it("labels any multi-asset total in USDT, with method and timestamp", () => {
    render(<VenueAccountPanel summary={LIVE_ACCOUNT} />);
    expect(screen.getByText(/total portfolio value/i)).toBeTruthy();
    expect(screen.getByText("164,459.16 USDT")).toBeTruthy();
    expect(screen.getByText(/order-book mid against USDT/i)).toBeTruthy();
  });

  it("withholds the total entirely when any asset could not be priced", () => {
    const unpriceable: Summary = {
      venue: {
        ...LIVE_ACCOUNT.venue,
        account: {
          ...LIVE_ACCOUNT.venue?.account,
          other_assets: [
            {
              asset: "WEIRD",
              quantity: "5",
              price_usdt: null,
              value_usdt: null,
              valuation_source: null,
              unpriced_reason: "no current price for WEIRD/USDT",
            },
          ],
          unpriced_assets: ["WEIRD"],
          total_portfolio_value_usdt: null,
          valuation_method: null,
          valued_at: null,
        },
      },
    };
    render(<VenueAccountPanel summary={unpriceable} />);
    expect(screen.getByText(/withheld — WEIRD could not be priced/i)).toBeTruthy();
    expect(screen.getAllByText(/NOT RECORDED/i).length).toBeGreaterThan(0);
  });
});

describe("venue account — degradation", () => {
  it("says when the last successful update was rather than going blank", () => {
    render(
      <VenueAccountPanel
        summary={{
          venue: {
            freshness: {
              available: false,
              stale: true,
              error: "timed out after 8s",
              fetched_at: "2026-08-14T13:40:00+00:00",
            },
          },
        }}
      />,
    );
    expect(screen.getByText(/temporarily unavailable/i)).toBeTruthy();
    expect(screen.getByText(/last successful update/i)).toBeTruthy();
    expect(screen.getByText(/timed out after 8s/i)).toBeTruthy();
  });

  it("renders with a completely empty payload without throwing", () => {
    expect(() => render(<VenueAccountPanel summary={{}} />)).not.toThrow();
  });

  it("renders with a null summary without throwing", () => {
    expect(() => render(<VenueAccountPanel summary={null} />)).not.toThrow();
  });
});

describe("positions and orders", () => {
  const withBook: Summary = {
    venue: {
      ...LIVE_ACCOUNT.venue,
      positions: [
        {
          symbol: "SOL/USDT:USDT",
          side: "long",
          quantity: "33.1",
          entry_price: "75.38",
          mark_price: "75.39",
          notional_usdt: "2495.4",
          unrealized_pnl: "0.331",
          leverage: "1",
          liquidation_price: null,
          venue_stop_loss: "74.1",
          venue_take_profit: "77.2",
          opened_at: "2026-08-14T13:45:00+00:00",
        },
      ],
      position_count: 1,
      open_orders: [
        {
          order_id: "abc",
          venue_order_id: "8c44539a-1756",
          symbol: "SOL/USDT",
          side: "sell",
          type: "market",
          status: "new",
          quantity: "33.1",
          filled_quantity: "0",
          price: null,
          trigger_price: "77.2",
          reduce_only: true,
          purpose: "take_profit",
          created_at: "2026-08-14T13:45:00+00:00",
        },
      ],
      open_order_count: 1,
    },
    book_reconciliation: {
      venue_open_positions: 5,
      database_open_positions: 1,
      venue_open_orders: 10,
      database_open_orders: 0,
      positions_match: false,
      orders_match: false,
    },
  };

  it("shows the venue's order status verbatim, never a stale NEW", () => {
    const [order] = withBook.venue?.open_orders ?? [];
    const filled: Summary = {
      venue: { ...withBook.venue, open_orders: order ? [{ ...order, status: "filled" }] : [] },
    };
    render(<OrdersPanel summary={filled} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("filled")).toBeTruthy();
    expect(within(table).queryByText("new")).toBeNull();
  });

  it("flags a position-count divergence between venue and database", () => {
    render(<PositionsPanel summary={withBook} />);
    expect(screen.getByText(/venue reports/i).textContent).toMatch(/5[\s\S]*1/);
  });

  it("flags an order-count divergence", () => {
    render(<OrdersPanel summary={withBook} />);
    expect(screen.getByText(/venue has/i)).toBeTruthy();
  });

  it("marks a position with no stop as UNPROTECTED", () => {
    const [position] = withBook.venue?.positions ?? [];
    const unprotected: Summary = {
      venue: {
        ...withBook.venue,
        positions: position ? [{ ...position, venue_stop_loss: null }] : [],
      },
    };
    render(<PositionsPanel summary={unprotected} />);
    expect(screen.getByText("UNPROTECTED")).toBeTruthy();
  });

  it("declares profit stage and loser-exit state NOT RECORDED", () => {
    render(<PositionsPanel summary={withBook} />);
    expect(screen.getByText(/profit stage, net-profit-exit eligibility/i)).toBeTruthy();
  });

  it("renders an empty book without throwing", () => {
    expect(() =>
      render(<PositionsPanel summary={{ venue: { positions: [], position_count: 0 } }} />),
    ).not.toThrow();
    expect(screen.getByText(/no open positions on the venue/i)).toBeTruthy();
  });
});

describe("fees", () => {
  it("makes fees exceeding gross profit impossible to miss", () => {
    render(
      <FeesPanel
        fees={{
          total_fees: "71.095529060000",
          gross_realized_pnl: "6.289550000000",
          net_realized_pnl: "-64.805979060000",
          closed_trades: 70,
          average_fee_per_trade: "1.01565",
          total_entry_notional: "69204.76387",
          average_fee_pct_of_notional: "0.001027321315531858",
          fee_to_gross_ratio: "11.30375449117981",
          fees_exceed_gross_profit: true,
        }}
      />,
    );
    expect(screen.getByText(/net-losing entirely on cost/i)).toBeTruthy();
    expect(screen.getByText(/11\.3×/)).toBeTruthy();
  });

  it("shows the observed round-trip cost measured from real fills", () => {
    render(
      <FeesPanel
        fees={{ average_fee_pct_of_notional: "0.001027321315531858", total_fees: "71.09" }}
      />,
    );
    expect(screen.getAllByText(/observed round-trip cost/i).length).toBeGreaterThan(0);
    expect(screen.getByText("0.1027%")).toBeTruthy();
  });

  it("declares the entry/exit fee split NOT RECORDED", () => {
    render(<FeesPanel fees={{ entry_fees: null, exit_fees: null }} />);
    expect(screen.getAllByText(/NOT RECORDED/i).length).toBeGreaterThanOrEqual(2);
  });
});

describe("decision engine", () => {
  const decisions: DecisionsResponse = {
    source: "scratchpad/bot.log",
    summary: {
      evaluated: 25,
      selected: 0,
      declined: 25,
      by_outcome: { DESELECTED: 25 },
      by_rejection_category: { correlation: 25 },
      window: "decisions retained from the tail of the engine log",
    },
    decisions: [
      {
        timestamp: "2026-08-14T13:00:06+00:00",
        event: "orchestrator.all_deselected",
        symbol: "ETH/USDT",
        outcome: "DESELECTED",
        candidates: 5,
        regime: "range",
        reason: "correlation 1.00 with an open position exceeds 0.85",
        rejection_category: "correlation",
        strategy: null,
        direction: null,
        score: null,
        confidence: null,
        component_scores: {},
        runner_up: null,
      },
    ],
  };

  it("answers why the bot is flat", () => {
    render(<DecisionsPanel decisions={decisions} openPositions={3} />);
    expect(screen.getByText(/correlation 1\.00 with an open position exceeds 0\.85/)).toBeTruthy();
    expect(screen.getByText(/bars evaluated/i)).toBeTruthy();
  });

  /**
   * The specific contradiction to surface: every rejection blames correlation "with an
   * open position" while the venue reports none open. Both cannot be true. The dashboard
   * states it; diagnosing the orchestrator is out of scope.
   */
  it("surfaces the correlation-versus-zero-positions contradiction", () => {
    render(<DecisionsPanel decisions={decisions} openPositions={0} />);
    expect(screen.getByText(/cannot both be true/i)).toBeTruthy();
    expect(screen.getByText(/stale ownership state/i)).toBeTruthy();
  });

  it("does not claim a contradiction when positions really are open", () => {
    render(<DecisionsPanel decisions={decisions} openPositions={5} />);
    expect(screen.queryByText(/cannot both be true/i)).toBeNull();
  });

  it("refuses to present the cost score as a cost in USDT", () => {
    render(<DecisionsPanel decisions={decisions} openPositions={1} />);
    expect(screen.getByText(/unit-free component scores/i)).toBeTruthy();
  });

  it("renders with no decisions at all without throwing", () => {
    expect(() => render(<DecisionsPanel decisions={null} openPositions={null} />)).not.toThrow();
  });
});

describe("analytics", () => {
  const analytics: AnalyticsResponse = {
    min_sample: 10,
    by_strategy: [
      { key: "momentum_roc", trades: 22, net_pnl: "-12.4", win_rate: "0.4", reliable: true },
      { key: "adx_trend", trades: 2, net_pnl: "3.1", win_rate: "1", reliable: false },
      { key: null, trades: 5, net_pnl: "-8.0", win_rate: "0.2", reliable: false },
    ],
    by_side: [
      { key: "long", trades: 37, net_pnl: "-41.99", fees: "30.40", win_rate: "0.432", reliable: true },
      { key: "short", trades: 33, net_pnl: "-22.82", fees: "40.69", win_rate: "0.364", reliable: true },
    ],
    by_symbol: [
      { key: "SOL/USDT", asset_class: "crypto", trades: 18, net_pnl: "-20.24", reliable: true },
      { key: "FARTCOIN/USDT", asset_class: "meme", trades: 11, net_pnl: "0.71", reliable: true },
    ],
    exit_reason_available: false,
    exit_reason_note: "the engine records no exit reason on a closed trade",
  };

  it("never calls a two-trade strategy reliable", () => {
    render(<AnalyticsPanels analytics={analytics} />);
    expect(screen.getAllByText(/insufficient sample/i).length).toBeGreaterThanOrEqual(2);
  });

  it("shows unattributed trades as unattributed rather than inventing a strategy", () => {
    render(<AnalyticsPanels analytics={analytics} />);
    expect(screen.getByText("unattributed")).toBeTruthy();
  });

  it("splits long and short", () => {
    render(<AnalyticsPanels analytics={analytics} />);
    expect(screen.getByText("long")).toBeTruthy();
    expect(screen.getByText("short")).toBeTruthy();
  });

  it("declares exit analytics NOT RECORDED instead of fabricating buckets", () => {
    render(<AnalyticsPanels analytics={analytics} />);
    expect(screen.getByText(/records no exit reason/i)).toBeTruthy();
  });

  it("renders a zero-trade session without throwing", () => {
    expect(() =>
      render(<AnalyticsPanels analytics={{ by_strategy: [], by_symbol: [], by_side: [] }} />),
    ).not.toThrow();
  });
});

describe("trade ledger", () => {
  const ledger: LedgerResponse = {
    total: 70,
    trades: [
      {
        trade_number: 70,
        trade_id: "03e266cc",
        symbol: "BNB/USDT",
        asset_class: "crypto",
        side: "long",
        quantity: "4.12",
        entry_time: "2026-08-14T11:30:22+00:00",
        exit_time: "2026-08-14T12:38:30+00:00",
        entry_price: "604.9",
        exit_price: "604.9",
        holding_seconds: 4087,
        gross_pnl: "0",
        total_fees: "2.7414068",
        net_pnl: "-2.7414068",
        return_pct: "-0.0011",
        strategy_id: null,
        exit_reason: null,
        mfe: null,
        mae: null,
      },
    ],
    not_recorded: {
      exit_reason: "no exit-reason column exists; notes is null for every trade",
      mfe: "maximum favourable excursion is not measured by the engine",
      mae: "maximum adverse excursion is not measured by the engine",
    },
  };

  it("renders NOT RECORDED for exit reason, never a blank or a zero", () => {
    render(<TradeLedgerPanel ledger={ledger} />);
    expect(screen.getAllByText(/NOT RECORDED/i).length).toBeGreaterThan(0);
  });

  it("shows fees and net PnL for each trade", () => {
    render(<TradeLedgerPanel ledger={ledger} />);
    expect(screen.getByText("2.74")).toBeTruthy();
    expect(screen.getByText("−2.74")).toBeTruthy();
  });

  it("renders an empty ledger without throwing", () => {
    expect(() => render(<TradeLedgerPanel ledger={{ total: 0, trades: [] }} />)).not.toThrow();
    expect(screen.getByText(/no closed trades in this session yet/i)).toBeTruthy();
  });

  it("renders a ledger with missing fields without throwing", () => {
    expect(() =>
      render(<TradeLedgerPanel ledger={{ trades: [{ trade_id: "bare" }] }} />),
    ).not.toThrow();
  });
});

describe("asset classes", () => {
  const classes: AssetClassesResponse = {
    asset_classes: [
      { asset_class: "crypto", state: "ACTIVE", symbol_count: 5, open_positions: 3, data_live: true, symbols: ["BTC/USDT"] },
      { asset_class: "meme", state: "ACTIVE", symbol_count: 1, open_positions: 1, data_live: true, symbols: ["FARTCOIN/USDT"] },
      {
        asset_class: "metal",
        state: "IMPLEMENTED, BLOCKED",
        symbol_count: 0,
        open_positions: 0,
        data_live: false,
        symbols: [],
        reason: "the venue refused an order in this class pending a product agreement",
      },
      {
        asset_class: "forex",
        state: "IMPLEMENTED, NOT WIRED",
        symbol_count: 0,
        open_positions: 0,
        data_live: false,
        symbols: [],
        reason: "not imported by the live trading loop and no broker credentials",
      },
    ],
  };

  it("marks only genuinely subscribed classes ACTIVE", () => {
    render(<AssetClassesPanel classes={classes} />);
    expect(screen.getAllByText("ACTIVE")).toHaveLength(2);
  });

  it("distinguishes blocked from merely disabled and from not wired", () => {
    render(<AssetClassesPanel classes={classes} />);
    expect(screen.getByText("IMPLEMENTED, BLOCKED")).toBeTruthy();
    expect(screen.getByText("IMPLEMENTED, NOT WIRED")).toBeTruthy();
    expect(screen.getByText(/pending a product agreement/i)).toBeTruthy();
  });

  it("renders with no classes without throwing", () => {
    expect(() => render(<AssetClassesPanel classes={null} />)).not.toThrow();
  });
});
