// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import { actorHeaderValue, proxyToBackend, type ProxyOptions } from "@/lib/proxy";

type FetchArgs = [input: string | URL | Request, init?: RequestInit & { duplex?: "half" }];

function setup(response: Response | Error = new Response("{}", { status: 200 }), options: Partial<ProxyOptions> = {}) {
  const fetchImpl = vi.fn<(...args: FetchArgs) => Promise<Response>>(async () => {
    if (response instanceof Error) throw response;
    return response;
  });
  const call = (request: Request, path: string[]) =>
    proxyToBackend(request, path, { backendUrl: "http://api.internal:8000", fetchImpl, ...options });
  return { fetchImpl, call };
}

describe("proxyToBackend", () => {
  it("forwards method, path and query to the backend /api prefix", async () => {
    const { fetchImpl, call } = setup();

    await call(new Request("http://ui.local/api/v1/documents?status=failed&limit=20"), ["v1", "documents"]);

    const [url, init] = fetchImpl.mock.calls[0];
    expect(String(url)).toBe("http://api.internal:8000/api/v1/documents?status=failed&limit=20");
    expect(init?.method).toBe("GET");
    expect(init?.body).toBeUndefined();
  });

  it("defaults to http://localhost:8000 and tolerates a trailing slash", async () => {
    const first = setup(undefined, { backendUrl: undefined });
    await first.call(new Request("http://ui.local/api/v1/stats"), ["v1", "stats"]);
    expect(String(first.fetchImpl.mock.calls[0][0])).toBe("http://localhost:8000/api/v1/stats");

    const second = setup(undefined, { backendUrl: "http://api:8000/" });
    await second.call(new Request("http://ui.local/api/v1/stats"), ["v1", "stats"]);
    expect(String(second.fetchImpl.mock.calls[0][0])).toBe("http://api:8000/api/v1/stats");
  });

  it("adds the API key and forwards only allow-listed request headers", async () => {
    const { fetchImpl, call } = setup(undefined, { apiKey: "server-side-key" });

    await call(
      new Request("http://ui.local/api/v1/documents", {
        headers: {
          accept: "application/json",
          cookie: "session=abc",
          authorization: "Basic b3BzOnMzY3JldA==",
          "x-api-key": "browser-supplied",
          "x-forwarded-for": "10.0.0.1",
          "user-agent": "test",
        },
      }),
      ["v1", "documents"],
    );

    const headers = new Headers(fetchImpl.mock.calls[0][1]?.headers);
    expect(headers.get("x-api-key")).toBe("server-side-key");
    expect(headers.get("accept")).toBe("application/json");
    expect(headers.get("cookie")).toBeNull();
    expect(headers.get("authorization")).toBeNull();
    expect(headers.get("host")).toBeNull();
    expect(headers.get("x-forwarded-for")).toBeNull();
    expect(headers.get("user-agent")).toBeNull();
  });

  it("identifies the actor as web-ui and never forwards a client-supplied X-Actor", async () => {
    const { fetchImpl, call } = setup();

    await call(new Request("http://ui.local/api/v1/stats", { headers: { "x-actor": "ceo@example.com" } }), ["v1", "stats"]);

    expect(new Headers(fetchImpl.mock.calls[0][1]?.headers).get("x-actor")).toBe("web-ui");
  });

  it("uses the basic-auth username as the actor when UI auth is on", async () => {
    const { fetchImpl, call } = setup(new Response("{}", { status: 200 }), { basicAuth: "ops:s3cret" });

    await call(
      new Request("http://ui.local/api/v1/invoices/i1", {
        method: "PATCH",
        headers: { authorization: "Basic " + btoa("ops:s3cret"), "x-actor": "someone-else", "content-type": "application/json" },
        body: "{}",
      }),
      ["v1", "invoices", "i1"],
    );

    expect(new Headers(fetchImpl.mock.calls[0][1]?.headers).get("x-actor")).toBe("ops");
  });

  it.each([
    ["Zoë", "Zo%C3%AB"],
    ["用户", "%E7%94%A8%E6%88%B7"],
  ])("keeps the request working when the basic-auth username is not Latin-1 (%s)", async (username, expected) => {
    const { fetchImpl, call } = setup(new Response("{}", { status: 200 }), { basicAuth: `${username}:s3cret` });
    const token = btoa(String.fromCharCode(...new TextEncoder().encode(`${username}:s3cret`)));

    const response = await call(
      new Request("http://ui.local/api/v1/stats", { headers: { authorization: `Basic ${token}` } }),
      ["v1", "stats"],
    );

    expect(response.status).toBe(200);
    expect(new Headers(fetchImpl.mock.calls[0][1]?.headers).get("x-actor")).toBe(expected);
  });

  it("caps the actor at 100 characters on an escape boundary and falls back to web-ui", () => {
    const long = actorHeaderValue("用".repeat(50));
    expect(long.length).toBeLessThanOrEqual(100);
    expect(long).toMatch(/^(%E7%94%A8)+$/);
    expect(actorHeaderValue(String.fromCharCode(0xd800))).toBe("web-ui");
    expect(actorHeaderValue(undefined)).toBe("web-ui");
  });

  it("does not invent an API key when none is configured", async () => {
    const { fetchImpl, call } = setup();

    await call(new Request("http://ui.local/api/v1/stats", { headers: { "x-api-key": "from-browser" } }), ["v1", "stats"]);

    expect(new Headers(fetchImpl.mock.calls[0][1]?.headers).get("x-api-key")).toBeNull();
  });

  it("streams request bodies through with their content type", async () => {
    const { fetchImpl, call } = setup(new Response('{"items":[]}', { status: 202 }));
    const form = new FormData();
    form.append("files", new File(["%PDF-1.7"], "a.pdf", { type: "application/pdf" }));
    const request = new Request("http://ui.local/api/v1/documents", { method: "POST", body: form });

    const response = await call(request, ["v1", "documents"]);

    const init = fetchImpl.mock.calls[0][1];
    expect(init?.method).toBe("POST");
    expect(init?.duplex).toBe("half");
    expect(init?.body).toBeInstanceOf(ReadableStream);
    expect(new Headers(init?.headers).get("content-type")).toMatch(/^multipart\/form-data; boundary=/);
    expect(response.status).toBe(202);
  });

  it("returns the backend status and body with safe headers only", async () => {
    const upstream = new Response('{"error":{"code":"not_found","message":"Document not found"}}', {
      status: 404,
      headers: {
        "content-type": "application/json",
        "content-disposition": 'attachment; filename="a.pdf"',
        "retry-after": "5",
        "set-cookie": "tracking=1",
        server: "uvicorn",
        "content-length": "999",
      },
    });
    const { call } = setup(upstream);

    const response = await call(new Request("http://ui.local/api/v1/documents/x"), ["v1", "documents", "x"]);

    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: { code: "not_found", message: "Document not found" } });
    expect(response.headers.get("content-type")).toBe("application/json");
    expect(response.headers.get("content-disposition")).toBe('attachment; filename="a.pdf"');
    expect(response.headers.get("retry-after")).toBe("5");
    expect(response.headers.get("set-cookie")).toBeNull();
    expect(response.headers.get("server")).toBeNull();
    expect(response.headers.get("content-length")).toBeNull();
  });

  it("passes 204 No Content through without a body", async () => {
    const { call } = setup(new Response(null, { status: 204 }));

    const response = await call(new Request("http://ui.local/api/v1/documents/x", { method: "DELETE" }), [
      "v1",
      "documents",
      "x",
    ]);

    expect(response.status).toBe(204);
    expect(response.body).toBeNull();
  });

  it("answers 502 with the standard envelope when the backend is unreachable", async () => {
    const { call } = setup(new TypeError("fetch failed", { cause: { code: "ECONNREFUSED" } }));

    const response = await call(new Request("http://ui.local/api/v1/stats"), ["v1", "stats"]);

    expect(response.status).toBe(502);
    const body = await response.json();
    expect(body.error.code).toBe("backend_unavailable");
    expect(typeof body.error.message).toBe("string");
  });

  it("refuses path segments that could escape the API prefix", async () => {
    const { fetchImpl, call } = setup();

    for (const path of [["v1", ".."], ["..", "admin"], ["v1", "a/b"], ["v1", "a\\b"], ["."]]) {
      const response = await call(new Request("http://ui.local/api/x"), path);
      expect(response.status).toBe(400);
    }
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("blocks cross-site state-changing requests but allows same-origin ones", async () => {
    const { fetchImpl, call } = setup(new Response(null, { status: 202 }));

    const crossSite = await call(
      new Request("http://ui.local/api/v1/jobs/reprocess-failed", { method: "POST", headers: { "sec-fetch-site": "cross-site" } }),
      ["v1", "jobs", "reprocess-failed"],
    );
    expect(crossSite.status).toBe(403);
    expect((await crossSite.json()).error.code).toBe("forbidden");
    expect(fetchImpl).not.toHaveBeenCalled();

    const sameOrigin = await call(
      new Request("http://ui.local/api/v1/jobs/reprocess-failed", { method: "POST", headers: { "sec-fetch-site": "same-origin" } }),
      ["v1", "jobs", "reprocess-failed"],
    );
    expect(sameOrigin.status).toBe(202);

    const crossSiteRead = await call(
      new Request("http://ui.local/api/v1/stats", { headers: { "sec-fetch-site": "cross-site" } }),
      ["v1", "stats"],
    );
    expect(crossSiteRead.status).not.toBe(403);
  });

  it("requires the UI basic-auth credentials when they are configured", async () => {
    const { fetchImpl, call } = setup(new Response("{}", { status: 200 }), { basicAuth: "ops:s3cret" });

    const missing = await call(new Request("http://ui.local/api/v1/stats"), ["v1", "stats"]);
    expect(missing.status).toBe(401);
    expect(missing.headers.get("www-authenticate")).toMatch(/^Basic realm=/);
    expect((await missing.json()).error.code).toBe("unauthorized");

    const wrong = await call(
      new Request("http://ui.local/api/v1/stats", { headers: { authorization: "Basic " + btoa("ops:nope") } }),
      ["v1", "stats"],
    );
    expect(wrong.status).toBe(401);
    expect(fetchImpl).not.toHaveBeenCalled();

    const ok = await call(
      new Request("http://ui.local/api/v1/stats", { headers: { authorization: "Basic " + btoa("ops:s3cret") } }),
      ["v1", "stats"],
    );
    expect(ok.status).toBe(200);
    expect(new Headers(fetchImpl.mock.calls[0][1]?.headers).get("authorization")).toBeNull();
  });
});
