/**
 * Typed API client.
 *
 * Two rules run through every type below.
 *
 * **Money is a string, end to end.** Every monetary and high-precision value arrives as a
 * JSON string and stays one until the moment it is formatted for display. `0.1 + 0.2 !==
 * 0.3` in JavaScript, and a dashboard that silently renders a position size or a PnL
 * wrong is worse than one that renders nothing.
 *
 * **Anything the server may omit is declared optional.** These types describe JSON from a
 * process deployed separately from this bundle: a stale or half-rebuilt API omits fields
 * the frontend was written against. That is how the page went blank once — `risk?.
 * kill_switch.engaged` guarded `risk` and then dereferenced `kill_switch` unguarded,
 * `undefined.engaged` threw, and a throw during render unmounts React's entire tree.
 * Marking these optional makes the compiler enforce the handling rather than leaving it
 * to discipline.
 */

/** A monetary or high-precision value. Kept as a string. */
export type Money = string;

/** A value the engine does not record. Rendered as `NOT RECORDED`, never as zero. */
export type NotRecorded = null;

export interface HealthResponse {
  status: "ok";
  version: string;
  environment: string;
}

export interface ComponentHealth {
  name: string;
  healthy: boolean;
  detail?: string | null;
  latency_ms?: number | null;
}

export interface ReadinessResponse {
  ready: boolean;
  components?: ComponentHealth[];
  trading_mode?: string;
  kill_switch_engaged?: boolean;
}

/** Provenance attached to every block that can go stale. */
export interface Freshness {
  source?: string;
  fetched_at?: string | null;
  age_seconds?: number | null;
  stale?: boolean;
  error?: string | null;
  available?: boolean;
}

export interface SessionRef {
  session_id: string;
  mode?: string;
  status?: string;
  strategy_id?: string;
  timeframe?: string;
  base_currency?: string;
  symbols?: string[];
  starting_equity?: Money;
  started_at?: string | null;
  created_at?: string | null;
  is_running?: boolean;
  /** Why this session was chosen, shown so the operator never has to guess. */
  selection_basis?: string;
}

export interface EngineStatus {
  state?: string;
  detail?: string;
  evidence?: Record<string, unknown>;
}

/** One non-USDT holding, valued only where an authoritative price existed. */
export interface OtherAsset {
  asset: string;
  free?: Money;
  locked?: Money;
  quantity?: Money;
  price_usdt?: Money | null;
  value_usdt?: Money | null;
  valuation_source?: string | null;
  unpriced_reason?: string | null;
}

/**
 * The venue's account, with units kept apart.
 *
 * `trading_equity_usdt` and `available_usdt` are the only figures the engine sizes
 * against. `total_portfolio_value_usdt` is present **only** when every holding could be
 * priced, and is always accompanied by its method and timestamp. There is deliberately no
 * field that sums across assets without conversion.
 */
export interface VenueAccount {
  trading_equity_usdt?: Money;
  available_usdt?: Money;
  locked_usdt?: Money;
  quote_asset?: string;
  other_assets?: OtherAsset[];
  other_assets_value_usdt?: Money | null;
  unpriced_assets?: string[];
  total_portfolio_value_usdt?: Money | null;
  valuation_method?: string | null;
  valued_at?: string | null;
}

export interface VenuePosition {
  symbol: string;
  side?: string;
  quantity?: Money;
  entry_price?: Money;
  mark_price?: Money;
  notional_usdt?: Money;
  unrealized_pnl?: Money;
  leverage?: Money;
  liquidation_price?: Money | null;
  margin_mode?: string | null;
  venue_stop_loss?: Money | null;
  venue_take_profit?: Money | null;
  opened_at?: string | null;
}

export interface VenueOrder {
  order_id: string;
  venue_order_id?: string | null;
  client_order_id?: string | null;
  symbol: string;
  side?: string;
  type?: string;
  /** Read from the venue every refresh, so it never shows NEW for a filled order. */
  status?: string;
  time_in_force?: string;
  quantity?: Money;
  filled_quantity?: Money;
  remaining_quantity?: Money;
  price?: Money | null;
  trigger_price?: Money | null;
  average_fill_price?: Money;
  reduce_only?: boolean;
  purpose?: "stop_loss" | "take_profit" | null;
  created_at?: string;
}

/**
 * Capital committed to open positions.
 *
 * Two figures rather than one: they differ by the leverage multiple, so a panel that
 * showed only "in trades" without saying which it meant would misstate committed capital
 * by that factor.
 */
export interface Deployed {
  notional_usdt?: Money;
  margin_usdt?: Money;
  position_count?: number;
  basis?: string;
}

export interface VenueBlock {
  venue?: string;
  network?: string;
  authenticated?: boolean;
  account?: VenueAccount;
  deployed?: Deployed | null;
  positions?: VenuePosition[];
  position_count?: number;
  position_error?: string | null;
  unrealized_pnl?: Money | null;
  open_orders?: VenueOrder[];
  open_order_count?: number;
  freshness?: Freshness;
}

export interface TradingPerformance {
  closed_trades?: number;
  gross_realized_pnl?: Money;
  total_fees?: Money;
  net_realized_pnl?: Money;
  today_net_pnl?: Money;
  today_closed_trades?: number;
  today_fees?: Money;
  win_count?: number;
  loss_count?: number;
  win_rate?: Money | null;
  profit_factor?: Money | null;
  gross_profit?: Money;
  gross_loss?: Money;
  average_net_pnl?: Money | null;
  best_trade?: Money | null;
  worst_trade?: Money | null;
  average_holding_seconds?: Money;
  first_exit_at?: string | null;
  last_exit_at?: string | null;
  sample_is_thin?: boolean;
}

export interface SessionEquity {
  capital_base?: Money;
  capital_base_source?: string;
  starting_equity?: Money;
  latest_equity?: Money | null;
  latest_cash?: Money | null;
  latest_unrealized_pnl?: Money | null;
  latest_realized_pnl?: Money | null;
  latest_gross_exposure?: Money | null;
  latest_at?: string | null;
  peak_equity?: Money | null;
  current_drawdown_pct?: Money | null;
  max_drawdown_pct?: Money | null;
  return_pct?: Money | null;
  return_basis?: string;
  snapshot_count?: number;
  history_from?: string | null;
  history_to?: string | null;
}

export interface FeeAnalysis {
  total_fees?: Money;
  gross_realized_pnl?: Money;
  net_realized_pnl?: Money;
  closed_trades?: number;
  average_fee_per_trade?: Money | null;
  total_entry_notional?: Money;
  average_fee_pct_of_notional?: Money | null;
  fee_to_gross_ratio?: Money | null;
  fees_exceed_gross_profit?: boolean;
  entry_fees?: Money | null;
  exit_fees?: Money | null;
  not_recorded?: Record<string, string>;
}

/** Where QuantFlow's record and the venue disagree about what is open. */
export interface BookReconciliation {
  venue_open_positions?: number | null;
  database_open_positions?: number;
  venue_open_orders?: number | null;
  database_open_orders?: number;
  positions_match?: boolean;
  orders_match?: boolean;
  authority?: string;
}

export interface DecisionSummary {
  evaluated?: number;
  selected?: number;
  declined?: number;
  by_outcome?: Record<string, number>;
  by_rejection_category?: Record<string, number>;
  by_symbol?: Record<string, number>;
  first_at?: string | null;
  last_at?: string | null;
  window?: string;
  freshness?: Freshness;
}

export interface SupervisorHistory {
  available?: boolean;
  path?: string;
  error?: string;
  events?: string[];
  restart_count?: number;
  exit_count?: number;
  /** Exits with rc=137 — SIGKILL, which on this host has meant the OS reclaiming memory. */
  killed_count?: number;
}

/** What the running engine reported about itself when it started. */
export interface EngineFacts {
  started_at?: string | null;
  mode?: string | null;
  env?: string | null;
  timeframe?: string | null;
  symbols?: string[];
  strategy?: string | null;
  strategy_pool?: string | null;
  starting_equity?: string | null;
  equity_source?: string | null;
  max_concurrent?: string | null;
  meme_symbols?: string[];
  meme_discovered?: string | null;
  agreement_blocked_symbols?: string[];
  agreement_blocked_at?: string | null;
  /** Asset classes the venue set aside, named by the engine rather than inferred. */
  agreement_blocked_classes?: string[];
  /** The venue codes actually seen — one per agreement still to be signed. */
  agreement_codes?: string[];
  class_symbols?: Record<string, string[]>;
  /** Always null: no pid file exists and the API cannot observe host processes. */
  pid?: number | null;
  pid_note?: string;
  supervisor?: SupervisorHistory;
}

export interface Coverage {
  earliest_fill_at?: string | null;
  orders_without_fills?: number;
  gap_from?: string | null;
  gap_to?: string | null;
  has_gap?: boolean;
  note?: string;
}

/** The engine's own liveness, from its Redis heartbeat — never inferred from decisions. */
export interface EngineHealth {
  state?: "RUNNING" | "DEGRADED" | "STALE" | "STOPPED" | "UNKNOWN";
  detail?: string;
  evidence?: {
    pid?: number | null;
    heartbeat_age_seconds?: number;
    last_candle_at?: string | null;
    last_decision_at?: string | null;
    last_reconcile_at?: string | null;
    open_positions?: number | null;
  };
}

export interface Summary {
  engine_health?: EngineHealth;
  generated_at?: string;
  session?: SessionRef;
  status?: EngineStatus;
  venue?: VenueBlock;
  trading_performance?: TradingPerformance;
  session_equity?: SessionEquity;
  session_book?: { open_positions?: number; open_orders?: number; source?: string };
  fees?: FeeAnalysis;
  book_reconciliation?: BookReconciliation;
  risk?: { kill_switch_engaged?: boolean; trading_halted?: boolean; available?: boolean };
  decisions?: DecisionSummary;
  engine?: EngineFacts;
}

/** One period's profit and loss, with the fee bill kept visible beside the net. */
export interface PnlPeriod {
  closed_trades?: number;
  gross_profit?: Money;
  gross_loss?: Money;
  fees?: Money;
  net_pnl?: Money;
  win_count?: number;
  loss_count?: number;
  win_rate?: Money | null;
  profit_factor?: Money | null;
  sample_is_thin?: boolean;
  scope?: string;
}

export interface CumulativePoint {
  at?: string;
  cumulative_net?: Money;
  cumulative_gross?: Money;
  cumulative_fees?: Money;
}

export interface PnlResponse {
  session_id?: string;
  ranges?: string[];
  order?: string[];
  periods?: Record<string, PnlPeriod>;
  generated_at?: string;
  cumulative?: {
    window?: string;
    points?: CumulativePoint[];
    point_count?: number;
    truncated?: boolean;
    source?: string;
  };
}

export interface PositionsResponse {
  available?: boolean;
  positions?: VenuePosition[];
  position_count?: number;
  unrealized_pnl?: Money | null;
  deployed?: Deployed;
  error?: string | null;
  freshness?: Freshness;
  not_recorded?: string[];
}

export interface OrdersResponse {
  available?: boolean;
  orders?: VenueOrder[];
  order_count?: number;
  error?: string | null;
  freshness?: Freshness;
}

export interface EquityPoint {
  timestamp?: string;
  equity?: Money;
  cash?: Money;
  realized_pnl?: Money;
  unrealized_pnl?: Money;
  running_peak?: Money;
  drawdown_pct?: Money;
  recorded_drawdown_pct?: Money;
  position_count?: number;
}

export interface Discontinuity {
  at?: string;
  from_equity?: Money;
  to_equity?: Money;
  change_pct?: Money;
  likely_cause?: string;
}

export interface EquityResponse {
  session_id?: string;
  window?: string;
  ranges?: string[];
  points?: EquityPoint[];
  point_count?: number;
  stride?: number;
  available_from?: string | null;
  available_to?: string | null;
  total_snapshots?: number;
  truncated?: boolean;
  discontinuities?: Discontinuity[];
  continuous?: boolean;
  history_note?: string;
}

export interface LedgerTrade {
  trade_number?: number;
  trade_id: string;
  symbol?: string;
  asset_class?: string;
  side?: string;
  quantity?: Money;
  entry_time?: string;
  exit_time?: string;
  entry_price?: Money;
  exit_price?: Money;
  entry_notional?: Money;
  holding_seconds?: number;
  gross_pnl?: Money;
  total_fees?: Money;
  net_pnl?: Money;
  return_pct?: Money;
  fee_share_of_gross?: Money | null;
  strategy_id?: string | null;
  regime?: string | null;
  notes?: string | null;
  /** Every field below is absent from the schema; rendered as NOT RECORDED. */
  entry_fee?: NotRecorded;
  exit_fee?: NotRecorded;
  exit_reason?: NotRecorded;
  mfe?: NotRecorded;
  mae?: NotRecorded;
  order_ids?: NotRecorded;
  position_id?: NotRecorded;
  venue_fill_ids?: NotRecorded;
}

export interface LedgerResponse {
  session_id?: string;
  coverage?: Coverage;
  trades?: LedgerTrade[];
  total?: number;
  limit?: number;
  offset?: number;
  not_recorded?: Record<string, string>;
}

export interface AttributionGroup {
  key?: string | null;
  asset_class?: string;
  trades?: number;
  gross_pnl?: Money;
  fees?: Money;
  net_pnl?: Money;
  wins?: number;
  win_rate?: Money;
  average_net_pnl?: Money;
  profit_factor?: Money | null;
  fee_share_of_gross?: Money | null;
  best?: Money | null;
  worst?: Money | null;
  /** False when the sample is too small for the numbers to mean anything. */
  reliable?: boolean;
}

export interface AnalyticsResponse {
  session_id?: string;
  by_strategy?: AttributionGroup[];
  by_symbol?: AttributionGroup[];
  by_side?: AttributionGroup[];
  by_exit_reason?: AttributionGroup[];
  exit_reason_available?: boolean;
  exit_reason_note?: string;
  strategy_attribution_note?: string | null;
  min_sample?: number;
}

export interface DecisionRow {
  timestamp?: string;
  event?: string;
  symbol?: string | null;
  outcome?: string;
  strategy?: string | null;
  direction?: string | null;
  score?: string | null;
  confidence?: string | null;
  candidates?: number | null;
  regime?: string | null;
  reason?: string | null;
  rejection_category?: string | null;
  component_scores?: Record<string, string>;
  runner_up?: string | null;
}

export interface DecisionsResponse {
  decisions?: DecisionRow[];
  summary?: DecisionSummary;
  source?: string;
  freshness?: Freshness;
  not_recorded?: Record<string, string>;
}

export interface CandleFreshness {
  symbol?: string;
  last_open_time?: string | null;
  age_seconds?: number | null;
}

export interface FreshnessResponse {
  generated_at?: string;
  state?: string;
  session_id?: string;
  timeframe?: string;
  venue_sync?: Freshness;
  engine_log?: Freshness;
  last_decision_at?: string | null;
  last_equity_snapshot_at?: string | null;
  last_equity_snapshot_age_seconds?: number | null;
  last_order_at?: string | null;
  last_candle_at?: string | null;
  candles?: CandleFreshness[];
  /** Why a stale stored candle does not mean the engine has lost its market data. */
  candle_note?: string;
  reconciliation?: { last_venue_read_at?: string | null; note?: string };
}

export interface AssetClassRow {
  asset_class?: string;
  description?: string;
  /**
   * ACTIVE | ACTIVE, ORDERS BLOCKED | IMPLEMENTED, BLOCKED | IMPLEMENTED, NOT ENABLED |
   * IMPLEMENTED, NOT WIRED
   */
  state?: string;
  reason?: string | null;
  symbols?: string[];
  symbol_count?: number;
  open_positions?: number;
  data_live?: boolean;
}

export interface AssetClassesResponse {
  session_id?: string;
  timeframe?: string;
  asset_classes?: AssetClassRow[];
  symbol_count?: number;
  venue_available?: boolean;
  source?: string;
}

export interface KillSwitch {
  engaged: boolean;
  reason?: string | null;
  engaged_at?: string | null;
  engaged_by?: string | null;
}

export interface ApiErrorBody {
  error: { code: string; message: string; request_id?: string | null };
}

/** An error carrying the server's own code and request id, so a report is traceable. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly requestId: string | null;

  constructor(status: number, code: string, message: string, requestId: string | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

const BASE = "/api/v1";

/**
 * How long a single call may hang before it is abandoned.
 *
 * A request with no timeout never fails when the API is merely unreachable — it stays
 * pending. The dashboard re-polls, so an API that is down leaves the tab accumulating
 * never-resolving requests until it locks up. This actually happened: one endpoint ran an
 * unbounded aggregate over a million rows and every poll piled another stuck request on
 * top of the last.
 */
const REQUEST_TIMEOUT_MS = 12_000;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Built through `Headers` rather than object spread: `HeadersInit` is legitimately a
  // `Headers` instance or an entry array, and spreading either into an object literal
  // yields `{}` — the caller's headers would vanish without a word.
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers,
      signal: init?.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === "TimeoutError";
    throw new ApiError(
      0,
      timedOut ? "timeout" : "network_error",
      timedOut ? `no response from the API within ${REQUEST_TIMEOUT_MS / 1000}s` : "API unreachable",
      null,
    );
  }

  if (!response.ok) {
    let code = "http_error";
    let message = `${response.status} ${response.statusText}`;
    let requestId: string | null = response.headers.get("X-Request-ID");
    try {
      const body = (await response.json()) as Partial<ApiErrorBody>;
      if (body.error) {
        code = body.error.code;
        message = body.error.message;
        requestId = body.error.request_id ?? requestId;
      }
    } catch {
      // A non-JSON error body is not itself worth reporting; the status is the useful part.
    }
    throw new ApiError(response.status, code, message, requestId);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/healthz"),
  readiness: () => request<ReadinessResponse>("/readyz"),

  /** The polled endpoint: header, status, venue account and session performance. */
  summary: () => request<Summary>(`${BASE}/dashboard/summary`),
  equity: (window: string) =>
    request<EquityResponse>(`${BASE}/dashboard/equity?window=${encodeURIComponent(window)}`),
  trades: (limit = 200, offset = 0) =>
    request<LedgerResponse>(`${BASE}/dashboard/trades?limit=${limit}&offset=${offset}`),
  analytics: () => request<AnalyticsResponse>(`${BASE}/dashboard/analytics`),
  decisions: (limit = 60) =>
    request<DecisionsResponse>(`${BASE}/dashboard/decisions?limit=${limit}`),
  freshness: () => request<FreshnessResponse>(`${BASE}/dashboard/freshness`),
  pnl: (window: string) =>
    request<PnlResponse>(`${BASE}/dashboard/pnl?window=${encodeURIComponent(window)}`),
  positions: () => request<PositionsResponse>(`${BASE}/dashboard/positions`),
  orders: () => request<OrdersResponse>(`${BASE}/dashboard/orders`),
  assetClasses: () => request<AssetClassesResponse>(`${BASE}/dashboard/asset-classes`),

  /** Engage or clear the kill switch. A reason is mandatory when engaging. */
  setKillSwitch: (engaged: boolean, reason: string, actor = "dashboard") =>
    request<KillSwitch>(`${BASE}/risk/kill-switch`, {
      method: "POST",
      body: JSON.stringify({ engaged, reason: reason || null, actor }),
    }),
};

/**
 * Subscribe to the live event stream.
 *
 * Reconnects with backoff. Returns a function that closes the socket and stops
 * reconnecting — without it, a component unmount would leave a socket reconnecting
 * forever in the background.
 */
export function subscribe(
  onEvent: (channel: string, data: unknown) => void,
  onStatus?: (connected: boolean) => void,
): () => void {
  let socket: WebSocket | null = null;
  let attempt = 0;
  let closed = false;
  let timer: number | undefined;

  const connect = () => {
    if (closed) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${window.location.host}${BASE}/ws`);

    socket.onopen = () => {
      attempt = 0;
      onStatus?.(true);
    };
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as { channel?: string; data?: unknown };
        if (payload.channel) onEvent(payload.channel, payload.data);
      } catch {
        // A malformed frame is dropped rather than tearing down a working stream.
      }
    };
    socket.onclose = () => {
      onStatus?.(false);
      if (closed) return;
      const delay = Math.min(30_000, 1_000 * 2 ** attempt);
      attempt += 1;
      timer = window.setTimeout(connect, delay);
    };
    socket.onerror = () => socket?.close();
  };

  connect();

  return () => {
    closed = true;
    if (timer) window.clearTimeout(timer);
    socket?.close();
  };
}

/**
 * Coerce an API collection to an array before rendering.
 *
 * The difference between one empty panel and a blank page: `.map` and `.length` on
 * `undefined` throw, and a throw during render unmounts the entire tree.
 */
export function list<T>(value: readonly T[] | null | undefined): readonly T[] {
  return Array.isArray(value) ? (value as readonly T[]) : [];
}
