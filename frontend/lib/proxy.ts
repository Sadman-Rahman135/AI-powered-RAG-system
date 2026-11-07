/**
 * Server-side forwarding of same-origin `/api/...` calls to the backend (`app/api/[...path]/route.ts`).
 *
 * - Forwards method, query string, the streamed body and an allow-list of request headers; adds `X-API-Key` from
 *   the server's environment so the key never reaches the browser, and `X-Actor` (the basic-auth username, or
 *   "web-ui") for the audit trail. Cookies, `Host`, `Authorization`, a client-supplied `X-Actor` and every other
 *   request header stay behind.
 * - Returns the backend's status and body with an allow-list of response headers (no `Set-Cookie`, no
 *   `Content-Length`/`Content-Encoding`, which no longer describe the re-streamed body).
 * - Refuses cross-site state-changing requests: the injected key would otherwise let any page the user visits
 *   trigger uploads or reprocessing (CSRF).
 * - Enforces the optional UI basic auth itself, because the page-level proxy does not run for `/api` (running
 *   there would buffer, and truncate, large upload bodies).
 */
import { BASIC_AUTH_CHALLENGE, isAuthorized, parseBasicAuthConfig } from "@/lib/basic-auth";

export interface ProxyOptions {
  /** Backend origin; defaults to http://localhost:8000. */
  backendUrl?: string;
  /** Sent as `X-API-Key` when set. */
  apiKey?: string;
  /** `user:pass`; when set, requests must carry these basic-auth credentials. */
  basicAuth?: string;
  fetchImpl?: typeof fetch;
}

export const DEFAULT_BACKEND_URL = "http://localhost:8000";
/** `X-Actor` sent when UI basic auth is off (otherwise the signed-in username is sent). */
export const DEFAULT_ACTOR = "web-ui";

const FORWARDED_REQUEST_HEADERS = ["accept", "content-type", "content-disposition"] as const;
const FORWARDED_RESPONSE_HEADERS = [
  "content-type",
  "content-disposition",
  "cache-control",
  "etag",
  "last-modified",
  "retry-after",
  "x-request-id",
  "x-ratelimit-limit",
  "x-ratelimit-remaining",
  "x-ratelimit-reset",
] as const;
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const BODYLESS_STATUSES = new Set([101, 204, 205, 304]);

function errorResponse(status: number, code: string, message: string, headers: Record<string, string> = {}): Response {
  return Response.json(
    { error: { code, message, details: null } },
    { status, headers: { "cache-control": "no-store", ...headers } },
  );
}

/** Cross-site per Fetch Metadata; falls back to comparing `Origin` with the request host for older browsers. */
function isCrossSite(request: Request): boolean {
  const site = request.headers.get("sec-fetch-site");
  if (site) return site !== "same-origin" && site !== "none";
  const origin = request.headers.get("origin");
  if (!origin) return false; // not sent by a browser, so not a CSRF vector
  try {
    return new URL(origin).host !== new URL(request.url).host;
  } catch {
    return true;
  }
}

const MAX_ACTOR_LENGTH = 100;

/**
 * A header-safe actor: the username percent-encoded (so "Zoë" is sent as "Zo%C3%AB"), cut at 100 characters on an
 * escape boundary. Header values must be Latin-1, so a raw non-Latin-1 name would make every proxied call throw.
 * Never throws; anything unusable becomes "web-ui".
 */
export function actorHeaderValue(username: string | null | undefined): string {
  if (!username) return DEFAULT_ACTOR;
  let encoded = "";
  try {
    for (const char of username) {
      const piece = encodeURIComponent(char); // throws on a lone surrogate
      if (encoded.length + piece.length > MAX_ACTOR_LENGTH) break;
      encoded += piece;
    }
  } catch {
    return DEFAULT_ACTOR;
  }
  return encoded || DEFAULT_ACTOR;
}

function isSafeSegment(segment: string): boolean {
  return segment !== "" && segment !== "." && segment !== ".." && !segment.includes("/") && !segment.includes("\\");
}

export async function proxyToBackend(
  request: Request,
  path: readonly string[],
  options: ProxyOptions = {},
): Promise<Response> {
  const credentials = parseBasicAuthConfig(options.basicAuth);
  if (credentials && !isAuthorized(request.headers.get("authorization"), credentials)) {
    return errorResponse(401, "unauthorized", "Sign in to use this application.", {
      "www-authenticate": BASIC_AUTH_CHALLENGE,
    });
  }

  const method = request.method.toUpperCase();
  if (!SAFE_METHODS.has(method) && isCrossSite(request)) {
    return errorResponse(403, "forbidden", "Requests from other sites are not allowed.");
  }
  if (path.length === 0 || !path.every(isSafeSegment)) {
    return errorResponse(400, "invalid_path", "This API path is not valid.");
  }

  const base = (options.backendUrl || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
  const target = `${base}/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;

  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (options.apiKey) headers.set("x-api-key", options.apiKey);
  // Who made the change, for the backend's audit trail. Set here only: a client-supplied X-Actor is never forwarded
  // (it is not in the allow-list above).
  headers.set("x-actor", actorHeaderValue(credentials?.username));

  const init: RequestInit & { duplex?: "half" } = { method, headers, redirect: "manual", cache: "no-store" };
  if (!SAFE_METHODS.has(method) && request.body) {
    init.body = request.body;
    init.duplex = "half"; // stream the upload instead of buffering it
  }

  let upstream: Response;
  try {
    upstream = await (options.fetchImpl ?? fetch)(target, init);
  } catch {
    return errorResponse(502, "backend_unavailable", "The processing service is not reachable. Try again in a moment.");
  }

  const responseHeaders = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  const body = BODYLESS_STATUSES.has(upstream.status) || method === "HEAD" ? null : upstream.body;
  return new Response(body, { status: upstream.status, statusText: upstream.statusText, headers: responseHeaders });
}
