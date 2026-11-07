/**
 * Optional HTTP basic auth for the UI pages (`UI_BASIC_AUTH=user:pass`). This is Next.js 16's `proxy` file
 * convention, which replaces the deprecated `middleware.ts`.
 *
 * `/api/*` is excluded on purpose: requests that pass through this layer have their bodies buffered (and cut at
 * 10 MB), which would break large uploads. The API route handler enforces the same credentials itself.
 */
import { NextResponse, type NextRequest } from "next/server";

import { BASIC_AUTH_CHALLENGE, isAuthorized, parseBasicAuthConfig } from "@/lib/basic-auth";

export function proxy(request: NextRequest) {
  const credentials = parseBasicAuthConfig(process.env.UI_BASIC_AUTH);
  if (!credentials || isAuthorized(request.headers.get("authorization"), credentials)) {
    return NextResponse.next();
  }
  return new NextResponse("Sign in to use DocIntel.", {
    status: 401,
    headers: {
      "www-authenticate": BASIC_AUTH_CHALLENGE,
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

export const config = {
  // `/healthz` stays open for the container healthcheck.
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico|icon.svg|healthz).*)"],
};
