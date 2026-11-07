/**
 * Liveness probe for the container healthcheck: answers without basic auth (excluded in `proxy.ts`) and without
 * calling the backend, so it reports on this server only.
 */
export function GET(): Response {
  return Response.json({ status: "ok" }, { headers: { "cache-control": "no-store" } });
}
