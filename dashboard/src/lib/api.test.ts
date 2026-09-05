import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, list } from "./api";

/** The shape of a `fetch` call, for assertions against the mock's recorded calls. */
type FetchArgs = [input: RequestInfo | URL, init?: RequestInit];

/** `vi.fn()` erases the call signature; this restores it for indexing assertions. */
function calls(mock: { mock: { calls: unknown[] } }): FetchArgs[] {
  return mock.mock.calls as FetchArgs[];
}

function jsonResponse(body: unknown, init: Partial<Response> = {}): Response {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    headers: new Headers(),
    json: () => Promise.resolve(body),
  } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("list", () => {
  // The guard that keeps one malformed payload from blanking the page: `.map` on undefined
  // throws, and a throw during render unmounts React's entire tree.
  it("coerces anything that is not an array to an empty array", () => {
    expect(list([1, 2])).toEqual([1, 2]);
    expect(list(undefined)).toEqual([]);
    expect(list(null)).toEqual([]);
    expect(list({ nope: true } as unknown as never[])).toEqual([]);
  });
});

describe("request errors", () => {
  it("surfaces the server's own error code and message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            { error: { code: "not_found", message: "no trading session has ever run" } },
            { ok: false, status: 404 },
          ),
        ),
      ),
    );

    await expect(api.summary()).rejects.toMatchObject({
      code: "not_found",
      status: 404,
      message: "no trading session has ever run",
    });
  });

  it("reports a timeout as an ApiError rather than hanging forever", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new DOMException("timed out", "TimeoutError"))),
    );

    const error = await api.summary().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("timeout");
  });

  it("reports an unreachable API distinctly from a timeout", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
    );

    const error = await api.summary().catch((caught: unknown) => caught);
    expect((error as ApiError).code).toBe("network_error");
  });

  it("falls back to the HTTP status when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 502,
          statusText: "Bad Gateway",
          headers: new Headers(),
          json: () => Promise.reject(new Error("not json")),
        } as unknown as Response),
      ),
    );

    await expect(api.summary()).rejects.toMatchObject({ status: 502, code: "http_error" });
  });

  it("sends a finite timeout signal on every request", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse({})));
    vi.stubGlobal("fetch", fetchMock);

    await api.summary();

    const [, init] = calls(fetchMock)[0] ?? [];
    expect(init?.signal).toBeDefined();
  });
});

describe("endpoints", () => {
  it("scopes the equity request to the requested window", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse({ points: [] })));
    vi.stubGlobal("fetch", fetchMock);

    await api.equity("24H");

    expect(calls(fetchMock)[0]?.[0]).toBe("/api/v1/dashboard/equity?window=24H");
  });

  it("pages the trade ledger", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse({ trades: [] })));
    vi.stubGlobal("fetch", fetchMock);

    await api.trades(50, 100);

    expect(calls(fetchMock)[0]?.[0]).toBe("/api/v1/dashboard/trades?limit=50&offset=100");
  });

  it("sends the reason when engaging the kill switch", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse({ engaged: true })));
    vi.stubGlobal("fetch", fetchMock);

    await api.setKillSwitch(true, "manual halt");

    const [, init] = calls(fetchMock)[0] ?? [];
    const body = typeof init?.body === "string" ? init.body : "{}";
    expect(JSON.parse(body)).toMatchObject({ engaged: true, reason: "manual halt" });
  });
});
