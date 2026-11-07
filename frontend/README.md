# DocIntel web UI

Next.js 16 (App Router, Cache Components) + TypeScript + Tailwind CSS 4 + shadcn/ui (Base UI) + TanStack Query.
Finance-operations staff upload documents, follow processing, inspect and correct extracted invoice fields,
approve records, search and compare.

## Run

```bash
npm install
BACKEND_URL=http://localhost:8000 npm run dev      # http://localhost:3000
```

Container: `docker build -t docintel-web .` (multi-stage, `node:24-alpine`, non-root, port 3000, healthcheck on
`/healthz`); pass the variables below at runtime. Without Docker: `npm run build`, then run the standalone server:

```bash
cp -r .next/static .next/standalone/.next/static && cp -r public .next/standalone/public
PORT=3000 BACKEND_URL=http://api:8000 BACKEND_API_KEY=... node .next/standalone/server.js
```

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Where `/api/*` is forwarded (server side only). |
| `BACKEND_API_KEY` | unset | Sent as `X-API-Key` to the backend; never reaches the browser. |
| `UI_BASIC_AUTH` | unset | `user:pass` turns on HTTP basic auth for pages and the API proxy. |

## Scripts

`npm run lint` · `npm run typecheck` (`tsc --noEmit`) · `npm test` (Vitest, once) · `npm run test:watch` · `npm run build`

## How it fits together

- The browser only calls same-origin `/api/v1/...`. `app/api/[...path]/route.ts` → `lib/proxy.ts` forwards method,
  query, the streamed body and `accept` / `content-type` / `content-disposition`, adds the API key, drops cookies,
  `Host`, `Authorization` and any client `X-Actor`, sets `X-Actor` (the basic-auth username, else `web-ui`) for the
  backend's audit trail, and returns the backend status, body and a safe subset of headers. Backend down → 502
  `{"error":{"code":"backend_unavailable",...}}`. Cross-site state-changing requests are refused (CSRF).
- `proxy.ts` (Next 16's replacement for `middleware.ts`) enforces the optional basic auth on pages. It excludes
  `/api`, because requests passing through it get their bodies buffered and cut at 10 MB; the API proxy checks the
  same credentials itself.
- `lib/types.ts` mirrors the REST contract (amounts are strings, dates ISO strings); `lib/api.ts` turns the error
  envelope into `ApiError {status, code, message, details}`; `lib/queries.ts` holds the TanStack Query hooks, which
  poll every 2 s while anything shown is queued or processing.
- Uploads (`components/upload-dropzone.tsx`, `lib/upload.ts`) use XMLHttpRequest for progress, multipart field
  `files`, at most 50 files per request (larger selections go out in sequential batches).

| Route | View |
|---|---|
| `/` | Stats, multi-file upload, documents table (filters, live progress, reprocess, delete) |
| `/documents/[id]` | Header, stage progress, tabs: extracted data (editable), validation, text, history, cross-verification. URL params `tab`, `invoice`, `page` |
| `/review` | Records that need review, with issue summaries |
| `/search` | Hybrid / keyword / semantic search with filters; state lives in the URL |
| `/compare` | Field-by-field comparison of two records or documents (`type`, `left`, `right` params) |

## Tests

Vitest + Testing Library + jsdom in `__tests__/`; `fetch` and `XMLHttpRequest` are mocked, no backend needed.
Stable `data-testid`s for end-to-end tests: `upload-input`, `upload-row`, `upload-status`, `documents-row`,
`status-badge`, `field-input-<field>`, `save-invoice`, `approve-invoice`, `search-input`, `search-mode`,
`search-result`, `compare-left` / `compare-right`, `compare-option`, `compare-submit`, `compare-result`,
`compare-field-row`, `delete-document`, `confirm-delete`, `reprocess-document`, `reprocess-failed`.
