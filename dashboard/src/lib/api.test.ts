/** Tests for the API client's wire handling. */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, pathSymbol } from "./api";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("pathSymbol", () => {
  it("hyphenates a slashed symbol", () => {
    // A slash cannot survive a URL path segment: the server decodes %2F before routing
    // and then sees an extra segment, which is a 404 rather than a symbol.
    expect(pathSymbol("BTC/USDT")).toBe("BTC-USDT");
  });

  it("leaves an already-hyphenated symbol alone", () => {
    expect(pathSymbol("BTC-USDT")).toBe("BTC-USDT");
  });

  it("escapes anything else", () => {
    expect(pathSymbol("A B")).toBe("A%20B");
  });
});

describe("request headers", () => {
  it("keeps caller headers supplied as a Headers instance", async () => {
    // The regression: spreading a `Headers` into an object literal yields `{}`, so the
    // caller's headers silently vanished and the server saw none of them.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    await api.trades(10);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  it("scopes trades to a session when one is given", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await api.trades(25, "dashboard-demo");
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("limit=25");
    expect(url).toContain("session_id=dashboard-demo");
  });

  it("omits the session parameter when there is no session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await api.trades(25, null);
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).not.toContain("session_id");
  });
});

describe("error handling", () => {
  it("surfaces the server's code and request id", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { error: { code: "not_found", message: "no session", request_id: "abc123" } },
          { status: 404, statusText: "Not Found" },
        ),
      ),
    );

    await expect(api.portfolio()).rejects.toMatchObject({
      name: "ApiError",
      code: "not_found",
      message: "no session",
      requestId: "abc123",
      status: 404,
    });
  });

  it("falls back to the status when the body is not the error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("<html>502</html>", { status: 502, statusText: "Bad Gateway" }),
      ),
    );

    const error = await api.portfolio().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("http_error");
    expect((error as ApiError).status).toBe(502);
  });
});
