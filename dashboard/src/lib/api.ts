/**
 * Typed API client.
 *
 * Every monetary field arrives as a **string**, never a JSON number, and is kept that way
 * until the moment it is formatted for display. `0.1 + 0.2 !== 0.3` in JavaScript, and a
 * dashboard that silently renders a position size or a PnL wrong is worse than one that
 * renders nothing at all.
 */

/** A monetary or high-precision value. Kept as a string end to end. */
export type Money = string;

export interface HealthResponse {
  status: "ok";
  version: string;
  environment: string;
}

export interface ComponentHealth {
  name: string;
  healthy: boolean;
  detail: string | null;
  latency_ms: number | null;
}

export interface ReadinessResponse {
  ready: boolean;
  components: ComponentHealth[];
  trading_mode: string;
  kill_switch_engaged: boolean;
}

export interface KillSwitch {
  engaged: boolean;
  reason: string | null;
  engaged_at: string | null;
  engaged_by: string | null;
}

export interface RiskLimits {
  max_position_pct: Money;
  max_total_exposure_pct: Money;
  max_concurrent_positions: number;
  max_daily_loss_pct: Money;
  max_drawdown_pct: Money;
  max_leverage: Money;
  require_stop_loss: boolean;
  max_order_notional: Money;
  max_orders_per_minute: number;
}

export interface RiskStatus {
  trading_halted: boolean;
  kill_switch: KillSwitch;
  limits: RiskLimits;
  headroom: Record<string, string>;
  sizer: string;
  rules: string[];
}

export interface RiskEvent {
  rule: string;
  severity: string;
  message: string;
  symbol: string | null;
  observed_value: Money | null;
  limit_value: Money | null;
  blocked_order: boolean;
  halted_trading: boolean;
  created_at: string;
}

export interface Position {
  symbol: string;
  side: string;
  quantity: Money;
  average_entry_price: Money;
  mark_price: Money | null;
  unrealized_pnl: Money;
  unrealized_pnl_pct: Money;
  realized_pnl: Money;
  stop_loss_price: Money | null;
  take_profit_price: Money | null;
  opened_at: string | null;
  strategy_id: string | null;
}

export interface Portfolio {
  base_currency: string;
  equity: Money;
  cash: Money;
  starting_equity: Money;
  total_return_pct: Money;
  realized_pnl: Money;
  unrealized_pnl: Money;
  fees_paid: Money;
  gross_exposure: Money;
  leverage: Money;
  drawdown_pct: Money;
  daily_pnl: Money;
  position_count: number;
  positions: Position[];
}

export interface Candle {
  open_time: string;
  open: Money;
  high: Money;
  low: Money;
  close: Money;
  volume: Money;
  quote_volume: Money;
  trades: number;
}

export interface CandlesResponse {
  symbol: string;
  timeframe: string;
  count: number;
  candles: Candle[];
  /** Missing bars detected. Non-zero means the chart has holes. */
  gaps: number;
}

export interface SeriesSummary {
  symbol: string;
  timeframe: string;
  bars: number;
  start: string | null;
  end: string | null;
}

export interface StrategyDescription {
  strategy_id: string;
  description: string;
  warmup_bars: number;
  defaults: Record<string, unknown>;
  schema: Record<string, unknown>;
}

export interface Trade {
  symbol: string;
  side: string;
  quantity: Money;
  entry_price: Money;
  exit_price: Money;
  entry_time: string;
  exit_time: string;
  gross_pnl: Money;
  fees: Money;
  net_pnl: Money;
  return_pct: Money;
  holding_hours: Money;
  strategy_id: string | null;
}

export interface Attribution {
  key: string;
  trade_count: number;
  net_pnl: Money;
  gross_pnl: Money;
  fees: Money;
  win_count: number;
  win_rate: Money;
  average_pnl: Money;
  best: Money;
  worst: Money;
  fee_drag_pct: Money;
  /** False when the sample is too small for the numbers to mean anything. */
  reliable: boolean;
}

export interface PerformanceReview {
  trade_count: number;
  by_strategy: Attribution[];
  by_symbol: Attribution[];
  by_side: Attribution[];
  streaks: { longest_win: number; longest_loss: number; current: number };
  concentration: {
    top_trade_share: Money;
    profit_without_best: Money;
    is_concentrated: boolean;
    rests_on_one_trade: boolean;
  };
  /** Plain-language caveats. Rendered prominently rather than buried. */
  warnings: string[];
}

export interface Session {
  session_id: string;
  mode: string;
  status: string;
  strategy_id: string;
  symbols: string[];
  timeframe: string;
  starting_equity: Money;
  final_equity: Money | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
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
 * Render a symbol for use in a URL path.
 *
 * `BTC/USDT` cannot go in a path segment: percent-encoding it to `BTC%2FUSDT` does not
 * help, because the server decodes it before routing and then sees an extra segment. The
 * hyphenated form round-trips cleanly and the API parses it back to the canonical symbol.
 */
export function pathSymbol(symbol: string): string {
  return encodeURIComponent(symbol.replace("/", "-"));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Built through `Headers` rather than object spread: `HeadersInit` is legitimately a
  // `Headers` instance or an entry array, and spreading either into an object literal
  // yields `{}` or index keys — the caller's headers would vanish without a word.
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  const response = await fetch(path, { ...init, headers });

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
      // A non-JSON error body is not itself an error worth reporting; the status is
      // already the useful part.
    }
    throw new ApiError(response.status, code, message, requestId);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/healthz"),
  readiness: () => request<ReadinessResponse>("/readyz"),

  portfolio: () => request<Portfolio>(`${BASE}/portfolio`),
  /**
   * Closed trades. Pass `sessionId` to scope them to one run.
   *
   * Without a session the API falls back to a trailing time window, which silently drops
   * anything older — a run over backfilled history can then show a single trade out of ten
   * and read as a working panel. The dashboard always scopes explicitly.
   */
  trades: (limit = 100, sessionId?: string | null) =>
    request<Trade[]>(
      `${BASE}/portfolio/trades?limit=${limit}` +
        (sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ""),
    ),
  sessions: () => request<Session[]>(`${BASE}/portfolio/sessions`),

  riskStatus: () => request<RiskStatus>(`${BASE}/risk/status`),
  riskEvents: (limit = 50) => request<RiskEvent[]>(`${BASE}/risk/events?limit=${limit}`),

  /** Engage or clear the kill switch. A reason is mandatory when engaging. */
  setKillSwitch: (engaged: boolean, reason: string, actor = "dashboard") =>
    request<KillSwitch>(`${BASE}/risk/kill-switch`, {
      method: "POST",
      body: JSON.stringify({ engaged, reason: reason || null, actor }),
    }),

  strategies: () => request<StrategyDescription[]>(`${BASE}/strategies`),
  series: () => request<SeriesSummary[]>(`${BASE}/market/series`),
  candles: (symbol: string, timeframe: string, limit = 300) =>
    request<CandlesResponse>(
      `${BASE}/market/candles/${pathSymbol(symbol)}?timeframe=${timeframe}&limit=${limit}`,
    ),

  /** Performance review. Scoped to one session when given, for the reason above. */
  review: (days = 90, sessionId?: string | null) =>
    request<PerformanceReview>(
      `${BASE}/analytics/review?days=${days}` +
        (sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ""),
    ),
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
        const payload = JSON.parse(event.data as string) as {
          channel?: string;
          data?: unknown;
        };
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
