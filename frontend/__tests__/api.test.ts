import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, buildQuery } from "@/lib/api";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function mockFetch(response: Response | Error) {
  const fn = vi.fn<typeof fetch>(async () => {
    if (response instanceof Error) throw response;
    return response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("error envelope", () => {
  it("turns {error:{code,message,details}} into a typed ApiError", async () => {
    mockFetch(
      jsonResponse(404, {
        error: { code: "not_found", message: "Document not found", details: { id: "doc-1" } },
      }),
    );

    const error = await api.getDocument("doc-1").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 404,
      code: "not_found",
      message: "Document not found",
      details: { id: "doc-1" },
    });
  });

  it("keeps validation details for 422 responses", async () => {
    const details = [{ loc: ["body", "fields", "total"], msg: "Not a valid amount" }];
    mockFetch(jsonResponse(422, { error: { code: "validation_error", message: "Invalid values", details } }));

    const error = await api
      .patchInvoice("inv-1", { fields: { total: "abc" }, line_items: null, comment: null })
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(422);
    expect((error as ApiError).code).toBe("validation_error");
    expect((error as ApiError).details).toEqual(details);
  });

  it("passes through the proxy's backend_unavailable envelope", async () => {
    mockFetch(
      jsonResponse(502, {
        error: { code: "backend_unavailable", message: "The processing service is not reachable." },
      }),
    );

    await expect(api.getStats()).rejects.toMatchObject({ status: 502, code: "backend_unavailable" });
  });

  it("still produces an ApiError when the error body is not JSON", async () => {
    mockFetch(new Response("<html>Bad gateway</html>", { status: 502, statusText: "Bad Gateway" }));

    const error = await api.getStats().catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 502, code: "http_502" });
    expect((error as ApiError).message).toMatch(/502/);
  });

  it("reports a network failure as status 0 / network_error", async () => {
    mockFetch(new TypeError("Failed to fetch"));

    await expect(api.getStats()).rejects.toMatchObject({ status: 0, code: "network_error" });
  });
});

describe("requests", () => {
  it("calls the same-origin /api/v1 path and drops empty query values", async () => {
    const fetchMock = mockFetch(jsonResponse(200, { items: [], total: 0, limit: 20, offset: 0 }));

    await api.listDocuments({ status: "failed", file_kind: undefined, q: "", limit: 20, offset: 0 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/documents?status=failed&limit=20&offset=0");
    expect(init?.method ?? "GET").toBe("GET");
  });

  it("sends JSON bodies with a JSON content type", async () => {
    const fetchMock = mockFetch(jsonResponse(200, { id: "inv-1" }));

    await api.approveInvoice("inv-1", "Checked against PO");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/invoices/inv-1/approve");
    expect(init?.method).toBe("POST");
    expect(new Headers(init?.headers).get("content-type")).toBe("application/json");
    expect(JSON.parse(String(init?.body))).toEqual({ comment: "Checked against PO" });
  });

  it("encodes path segments", async () => {
    const fetchMock = mockFetch(jsonResponse(200, {}));

    await api.getInvoice("a/b c");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/invoices/a%2Fb%20c");
  });

  it("resolves to undefined for 204 No Content", async () => {
    mockFetch(new Response(null, { status: 204 }));

    await expect(api.deleteDocument("doc-1")).resolves.toBeUndefined();
  });

  it("builds query strings from booleans, numbers and repeated values", () => {
    expect(buildQuery({ a: 1, b: true, c: false, d: null, e: undefined, f: "", g: ["x", "y"] })).toBe(
      "?a=1&b=true&c=false&g=x&g=y",
    );
    expect(buildQuery({})).toBe("");
  });
});
