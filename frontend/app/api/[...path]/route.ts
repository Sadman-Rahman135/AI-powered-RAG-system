/**
 * Same-origin API proxy: `/api/<path>` -> `${BACKEND_URL}/api/<path>`, adding `X-API-Key: BACKEND_API_KEY`
 * server-side. Runs on the Node.js runtime (the default; Cache Components forbids a `runtime` export).
 * See `lib/proxy.ts` for what is forwarded, what is stripped, and why.
 */
import { proxyToBackend } from "@/lib/proxy";

async function handler(request: Request, context: { params: Promise<{ path: string[] }> }): Promise<Response> {
  const { path } = await context.params;
  return proxyToBackend(request, path, {
    backendUrl: process.env.BACKEND_URL,
    apiKey: process.env.BACKEND_API_KEY,
    basicAuth: process.env.UI_BASIC_AUTH,
  });
}

export { handler as DELETE, handler as GET, handler as HEAD, handler as PATCH, handler as POST, handler as PUT };
