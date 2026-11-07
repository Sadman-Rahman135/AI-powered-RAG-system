# Security

This document lists what the system protects, from whom, and every control that is actually implemented, with the
file that implements it. It ends with what is deliberately out of scope and a checklist for production.

## Threat model (summary)

DocIntel processes confidential financial documents uploaded by staff or integrations and stores their extracted
content. The assets are the documents, the extracted records and the API keys / LLM credentials.

| Threat | Examples | Main controls |
|---|---|---|
| Unauthenticated access to data or processing | Reading invoices, uploading files, using LLM credit | API keys on every data route, rate limits, production startup checks, optional UI basic auth |
| Hostile file uploads | Decompression bombs, zip bombs, malformed PDFs/images, path traversal in names, spoofed extensions, XML entity expansion | Size/count limits, magic-byte detection, sanitised names, random storage keys, bomb and page/row/pixel guards, `defusedxml`, typed parse errors |
| Injection | SQL injection through filters/search, header injection through file names | ORM/bound parameters only, escaped `LIKE`, validated sort values, RFC 6266 header encoding |
| Prompt injection | Text inside an invoice telling the model to change amounts or reveal data | Untrusted-data framing, schema-constrained output, grounding, arithmetic validation, human review queue |
| Browser-side attacks | CSRF against the web proxy, clickjacking, XSS, key theft from the browser | Cross-site guard in the proxy, server-side key injection, CSP and framing headers |
| Information leakage | Document text or SQL parameters in logs and error messages, stack traces to clients | Log hygiene, generic 500 envelope, `hide_parameters`, `no-store` caching |
| Abuse / denial of service | Request floods, giant bodies, huge documents | Rate limits, body caps, per-file/page/row limits, bounded worker concurrency, attempt deadline (`JOB_MAX_RUNTIME_S`), per-page OCR timeout (`OCR_PAGE_TIMEOUT_S`) |
| Misconfiguration | Default keys, SQLite or wildcard CORS in production | `APP_ENV=production` refuses to start |

Trust boundaries: browser -> web server (Next.js) -> API -> database / Redis / storage / LLM provider. Uploaded
documents and LLM output are untrusted; everything inside the compose network is trusted.

## Implemented controls

### Authentication and key handling

- **API keys** (`backend/app/security/auth.py`). When `API_KEYS` is set, every `/api/v1` route except `GET /health`
  requires `X-API-Key` or `Authorization: Bearer`. Keys are compared with `hmac.compare_digest` against every configured
  key without early exit, so timing does not reveal which key matched or how much of it was right.
- Keys are never logged or stored. The caller is identified by `sha256(key)`; the audit identity is the `X-Actor`
  header the web proxy sends (sanitised: control characters removed, 100 characters), else `api-key:<8 hex of the
  hash>`, else `anonymous`.
- Failed authentication attempts are rate limited per client IP (the limiter runs before the key check:
  `backend/app/api/deps.py::guard`).
- **The key never reaches the browser.** The Next.js server holds `BACKEND_API_KEY` and adds `X-API-Key` when it
  forwards `/api/*` calls (`frontend/lib/proxy.ts`, `frontend/app/api/[...path]/route.ts`). The browser only talks to
  its own origin; the variable is read at runtime, never baked into the image or sent to the client.
- **Optional UI basic auth** (`UI_BASIC_AUTH=user:pass`): enforced for pages by `frontend/proxy.ts` and for `/api/*`
  by the route handler itself, with a constant-time comparison (`frontend/lib/basic-auth.ts`). The username becomes the
  audit identity.
- **Secrets** come from the environment (`.env` is git-ignored; only `.env.example` with placeholders is committed).
  Settings hold provider keys as `SecretStr`, unwrapped only when a client is built (`backend/app/config.py`,
  `backend/app/extraction/llm/factory.py`). Images contain no secrets; web variables are read at container start.

### Upload hardening

- **Size and count limits** (`backend/app/security/uploads.py`): a declared `Content-Length` over the cap is refused with
  413 before any byte is read; bodies without one (chunked) are counted while they arrive and cut off. Cap for the upload
  route: `min(MAX_FILES_PER_REQUEST, 5) x (MAX_UPLOAD_MB + 16 KB)` (about 1000 MiB with the defaults; the UI reads it as
  `max_request_mb` from `GET /api/v1/config` and batches by bytes); every other route: 1 MiB. Multipart parsing is bounded
  in files, fields and part size. The per-file limit is enforced while the body streams
  (`_BoundedMultiPartParser`): at most `MAX_UPLOAD_MB` of any part is spooled to temporary disk, the rest is counted and
  discarded, and the file is rejected as a per-item `file_too_large` while the other files of the request are processed.
  The request cap bounds what a whole request can spool. Storage writes are also size-checked and delete partial files
  (`backend/app/storage.py`).
- **Content-based type detection** (`backend/app/ingestion/sniff.py`): the kind (PDF, PNG/JPEG/TIFF, XLSX, XLS, CSV) is
  decided from magic bytes and container structure; the extension is only a hint (a mismatch becomes a document
  warning). Unsupported files (including `.xlsb` and non-Excel zips) and empty files are rejected per item.
- **Sanitised names, random storage keys.** `sanitize_filename` (`backend/app/pipeline/intake.py`) strips paths, drive
  letters, `..`, control and bidi-format characters and limits the length; the name is display-only. Files are stored as
  `YYYY/MM/<uuid4 hex><suffix>` where the suffix comes from the detected type, validated against `^(\.[a-z0-9]{1,8})?$`;
  storage lookups reject keys with unexpected characters (`backend/app/storage.py`). Downloads use `Content-Disposition:
  attachment` with an escaped RFC 5987 filename and `X-Content-Type-Options: nosniff` (`content_disposition`).
- **Bomb and size guards in the parsers** (`backend/app/ingestion/`): PDF page limit `MAX_PDF_PAGES`; image/TIFF pixel
  limit `MAX_IMAGE_PIXELS` with Pillow's decompression-bomb error *and* warning treated as errors (`images.py`); XLSX total
  uncompressed size `MAX_XLSX_UNCOMPRESSED_MB` and per-member compression ratio `MAX_XLSX_COMPRESSION_RATIO` checked before
  openpyxl opens the file; sheet rows/columns capped by `MAX_SHEET_ROWS` / `MAX_SHEET_COLS`; a cap of 20,000 layout spans per
  page (`layout.py`).
- **XML safety:** openpyxl is used read-only with `defusedxml` installed as its XML parser
  (`backend/app/ingestion/spreadsheets.py`), blocking entity-expansion and external-entity attacks.
- **Typed failures** (`backend/app/ingestion/guard.py`): any unexpected exception from a parsing library on hostile input
  becomes a typed `corrupt_file` / `file_too_large` error recorded on the document; password-protected PDFs fail as
  `encrypted_file` (an empty password is tried first). A bad file never takes a worker down.
- With `OCR_WORKERS > 0` (the Docker default for workers is 4) OCR runs in a separate spawned process pool, so a crash in
  native OCR code does not run inside the worker's own process. A pooled OCR page that takes longer than
  `OCR_PAGE_TIMEOUT_S` (default 120 s) is treated as an OCR outage (retryable `ocr_unavailable`), not as a bad file.
- **Time limits.** Every processing attempt has a deadline, `JOB_MAX_RUNTIME_S` (default 4 h): once it passes, the attempt
  stops writing its heartbeat, the stale-job sweeper reclaims the job and it is re-queued, or failed as `job_timeout` when
  no attempt is left; a superseded attempt stops at its next stage boundary and cannot overwrite the takeover. Celery's own
  task time limits are not used (they are inert with the thread pool). A thread stuck in native code cannot be killed and
  keeps its pool slot until the worker restarts. In-process OCR (`OCR_WORKERS=0`) has no per-page timeout and relies on
  the attempt deadline only.

### SQL safety

All queries go through SQLAlchemy ORM expressions with bound parameters. Sort fields are validated `Literal`s; substring
filters use `ILIKE ... ESCAPE` with `%`, `_` and `\` escaped (`backend/app/api/queries.py`). The PostgreSQL search path
uses `text()` with bound parameters, and the only interpolated SQL fragments are internal constants (column lists,
filter clauses generated from typed filter objects), never request strings (`backend/app/search/service.py`).
`websearch_to_tsquery` parses the user's query safely. The engine is created with `hide_parameters=True`
(`backend/app/db/session.py`) so a SQL error never carries bound values (invoice data, file names) into logs or responses.

### Rate limiting

Fixed-window limits per caller (a valid API key fingerprint combined with the sanitised `X-Actor` the web proxy sends, so
each UI user has a separate budget; a valid key without an actor uses the key alone; no valid key, or auth off, uses the
client IP and ignores `X-Actor`) for uploads, search and everything else
(`backend/app/security/ratelimit.py`; defaults `60/minute`, `300/minute`, `1200/minute`). Counters are shared through
Redis when `REDIS_URL` is set. A limiter storage failure lets traffic through and logs a warning (availability over
strictness). Behind a reverse proxy, uvicorn runs with `--proxy-headers` (the image `CMD`) and the proxy's address must be
trusted via `--forwarded-allow-ips` for the real client IP to be used.

### CORS, headers and caching

- CORS (`backend/app/main.py`): origins from `CORS_ORIGINS` only (default `http://localhost:3000`), no credentials
  (`allow_credentials=False`: keys travel in headers, never cookies), methods `GET POST PATCH DELETE OPTIONS`, a fixed header
  allow-list. The wildcard is refused in production.
- API security headers on every response, including streamed files (`backend/app/security/headers.py`):
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Content-Security-Policy:
  default-src 'none'; frame-ancestors 'none'` (relaxed only on `/docs` and `/redoc`), `Cache-Control: no-store` on `/api/`.
- Web UI headers (`frontend/next.config.ts`): a CSP limiting scripts, styles, images, fonts and connections to the own
  origin (plus the inline bootstrap scripts Next.js emits), `object-src 'none'`, `frame-ancestors 'none'`, `X-Frame-Options:
  DENY`, `nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, a restrictive `Permissions-Policy`; the
  `X-Powered-By` header is off.

### CSRF and the web proxy

Because the proxy adds the API key to every request, any page the signed-in user visits could otherwise trigger uploads or
reprocessing. `frontend/lib/proxy.ts` refuses state-changing requests (anything but GET/HEAD/OPTIONS) that are cross-site
by Fetch Metadata (`Sec-Fetch-Site`), falling back to an `Origin` versus host comparison for older browsers, with
`403 forbidden`. Caveat: the `Origin` fallback compares with the host of the request URL the web server sees, so behind
a reverse proxy that rewrites `Host`, a browser that does not send `Sec-Fetch-Site` can have its legitimate writes
refused with 403; pass the original `Host` through the proxy. A request with neither header is treated as not
browser-originated. It also forwards only an allow-list of request headers (never cookies, `Authorization`, `Host` or a
client-supplied `X-Actor`), only an allow-list of response headers (no `Set-Cookie`), rejects path segments such as `..`,
`/` or `\`, does not follow redirects, and returns a generic `502 backend_unavailable` when the API is down.

### LLM prompt-injection controls

Invoices are untrusted text that is fed to a model, so the design assumes the model can be manipulated and limits what
that can achieve (`backend/app/extraction/`):

1. **Untrusted-data framing.** The system prompt states that the document is data, not instructions, and the text is
   placed inside `<document>` tags; a literal `<document>` or `</document>` inside the text is defanged so a document
   cannot close its own data block (`llm/prompt.py::wrap_document`).
2. **Schema-constrained output.** The reply must match a closed JSON Schema of nullable strings (structured outputs on
   Claude and Gemini, grammar-constrained decoding on llama.cpp). The model cannot emit tool calls, free text or extra
   fields, and no model output is executed or interpreted as a command.
3. **Verbatim copy and grounding.** The model must copy values as printed; each value must occur in the source text and
   be supported by it after parsing (`extractor.py`, `grounding.py`). A value the document does not support becomes `null`
   and an `UNGROUNDED_VALUE_REJECTED` note: an injected "set the total to 1" or "supplier = Evil Corp" is dropped unless
   the document really contains it.
4. **Arithmetic validation and merge.** Totals, taxes and line items must reconcile (`validation/`, `merge.py`); the
   merge only chooses among values the rules extractor and the grounded LLM produced.
5. **Human in the loop.** Any error or warning puts the record in the review queue; corrections are audited.
6. **No ambient authority.** The model has no tools, network or database access; the LLM call is the only thing a
   document's text reaches. Provider errors, refusals and timeouts fall back to rules (`LLM_UNAVAILABLE`).
7. **Bounded input and output:** the input budget (derived from the model's context, capped by `LLM_MAX_INPUT_CHARS`),
   `LLM_MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT_S`, `LLM_CONCURRENCY`.

With Claude or Gemini, document text is sent to that provider; use the local llama.cpp provider (or `LLM_PROVIDER=none`)
when documents must not leave the machine.

### Log hygiene and error handling

- Logs are structured (structlog; JSON in Docker) and carry IDs, codes, counts and timings, never document text or
  extracted values. The access log records the method, the *route template* (`/api/v1/documents/{document_id}`), status and
  duration, never raw paths, query strings, bodies or file names (`backend/app/api/middleware.py`,
  `backend/app/logging_setup.py`). Uvicorn's own access log is disabled.
- Unexpected exceptions return a generic `500 internal_error` with only the request id; the exception text goes to the log
  because it may contain document data. Validation errors never echo input values (`backend/app/api/errors.py`).
- Rejected uploads and parse failures store typed codes and generic messages, not file contents.

### Container and deployment hardening

- Both images run as a non-root user (`docintel`, uid 10001; `nextjs`, uid 1001) with health checks (`backend/Dockerfile`,
  `frontend/Dockerfile`). The web image is a multi-stage Next.js standalone build.
- All published ports are bound to `127.0.0.1` in `docker-compose.yml`; put a TLS-terminating reverse proxy in front for
  remote access (see `docs/DEPLOYMENT.md`). The model files are mounted read-only into the `llm` container.
- Models are baked into the backend image, so workers need no outbound network except for the chosen LLM provider.
- The Redis broker carries only job ids; documents and results are never in the queue.
- Dependencies are pinned in `backend/requirements.lock` and `frontend/package-lock.json`.

### Production startup checks

With `APP_ENV=production` the API refuses to start (`Settings.production_problems`, `backend/app/main.py`) when
`API_KEYS` is empty, contains a known default (`change-me`, `dev-local-key-change-me`, `changeme`, `test`,
`replace-with-a-long-random-key-0000`) or a key shorter than 24 characters, `DATABASE_URL` is SQLite, or `CORS_ORIGINS`
contains `*`. Note the `.env.example` key is a placeholder that production mode rejects on purpose.

### Data integrity as a security property

Corrections are append-only with who/when/old/new/why; re-processing never deletes them. Money is stored as exact decimals.
Duplicate files (same sha256) and duplicate invoices are flagged. Deleting a document removes its file and all derived rows.

## Pre-release checks (2026-11-07)

Run before the repository was published:

- **Secrets.** No real credential is in the repository. `.env`, `backend/.env` and `frontend/.env*` are git-ignored, and
  every key, password and token value from the local `.env` files was searched for in every published file: none
  found. A pattern scan (Google, Anthropic, OpenAI, GitHub, AWS, Hugging Face and Slack tokens, private keys, JWTs,
  passwords in URLs or code) found only placeholders, the documented development default and test fixtures.
  `.env.example` holds placeholders only.
- **Dependencies.** `pip-audit` on `backend/requirements.lock` and `requirements-dev.lock`: no known vulnerabilities.
  `npm audit --omit=dev` (what the web image ships): 0 vulnerabilities. Including development tools, `npm audit`
  reports 13 low/moderate advisories, all in build-time tooling (`minimatch` under the ESLint config, `deepmerge`
  under the shadcn CLI). None runs in the shipped application, and the suggested fixes are major-version downgrades
  incompatible with Next.js 16, so they are left for an upstream update.
- **Live stack.** Without a key or with a wrong key the API answers 401; API and web responses carry the security
  headers above; a cross-site POST through the web proxy is refused (403); compose publishes every port on 127.0.0.1
  only.

## Known limitations and out of scope

- **Single trust domain.** There are API keys but no users, roles or per-tenant isolation (no SSO, OAuth/OIDC, RBAC or
  multi-tenancy). Any holder of a key can read and change every record.
- **No encryption at rest** at the application level: PostgreSQL data, Redis AOF and the storage volume rely on disk /
  volume encryption provided by the host. TLS is not terminated by the application.
- **No malware scanning.** Files are parsed defensively but not scanned by an antivirus. Downloaded originals are served as
  attachments with `nosniff`; do not open them on a sensitive machine without scanning.
- **Key rotation is manual:** change `API_KEYS` (comma-separated, so old and new can overlap) and restart; keys are not
  stored in the database. There are no per-key scopes or quotas other than the shared rate limits.
- **Rate limiting is per API process group:** with no Redis, counters are per process (`uvicorn --workers 2` doubles the
  budget); with Redis they are shared. Fixed windows allow bursts at window edges.
- **Timeouts are coarse.** Pooled OCR has a per-page timeout (`OCR_PAGE_TIMEOUT_S`); in-process OCR (`OCR_WORKERS=0`)
  and the other parsers have none and rely on size limits and the attempt deadline (`JOB_MAX_RUNTIME_S`, default 4 h),
  which re-queues the job but cannot kill a thread stuck in native code.
- **Basic auth has no brute-force limiting.** `UI_BASIC_AUTH` uses a constant-time comparison but nothing slows repeated
  guesses; for internet exposure put the UI behind a reverse proxy or SSO that rate limits and authenticates.
- **LLM providers see document text** (Claude, Gemini). Their retention policies are outside this system.
- **Web UI CSP allows inline scripts/styles** because Next.js emits inline bootstrap code and the build uses no nonces.
- **Default development credentials** (`POSTGRES_PASSWORD=docintel-dev-only`) are for local use only.
- Original files and rows are not purged automatically; there is no retention policy.

## Production hardening checklist

- [ ] `APP_ENV=production`; `API_KEYS` with at least 24 random characters per key (`python -c "import secrets;
      print(secrets.token_urlsafe(32))"`), `BACKEND_API_KEY` set to one of them.
- [ ] A strong `POSTGRES_PASSWORD`; do not publish the database or Redis ports (remove the `ports` entries or keep them on
      127.0.0.1); Redis on a private network, ideally with a password in `REDIS_URL`.
- [ ] TLS in front of the web UI and API (reverse proxy), HSTS enabled there; `CORS_ORIGINS` set to the real UI origin;
      uvicorn started with `--forwarded-allow-ips` for the proxy so rate limits see client IPs.
- [ ] `UI_BASIC_AUTH` or an authenticating proxy / SSO in front of the web UI (the UI itself has no user accounts);
      basic auth has no brute-force limiting, so use the proxy/SSO for anything internet-facing.
- [ ] Encrypted volumes for `pgdata`, `storage` and `redisdata`; backups encrypted and tested (`docs/DEPLOYMENT.md`).
- [ ] Decide the LLM data policy: local model or `LLM_PROVIDER=none` for confidential data; otherwise review the provider's
      terms and retention.
- [ ] Antivirus / content-disarm step in front of uploads if untrusted parties can upload.
- [ ] Ship JSON logs to a central store with access control; confirm no sensitive data in your proxy's access logs.
- [ ] Pin and update base images and dependencies regularly; rebuild images for security fixes.
- [ ] Keep `RATE_LIMIT_ENABLED=true` and tune `RATE_LIMIT_*`; set resource limits on containers.
- [ ] Rotate keys on staff changes; monitor `GET /api/v1/ready` and 401/429 rates.
