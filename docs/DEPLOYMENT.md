# Deployment

How to run DocIntel "live" with Docker Compose, scale it, back it up, upgrade it and harden it. For a first local run
see the quick start in the [README](../README.md).

## Services

`docker-compose.yml` (project name `docintel`):

| Service | Image / build | Role | Host port (127.0.0.1) |
|---|---|---|---|
| `db` | `pgvector/pgvector:pg17` | PostgreSQL with `vector`; volume `pgdata` | 5433 -> 5432 |
| `redis` | `redis:7-alpine` | Celery broker and shared rate-limit counters; AOF on, volume `redisdata` | 6380 -> 6379 |
| `migrate` | `backend/` image | One-shot `alembic upgrade head`; `api` and `worker` wait for it to finish | - |
| `api` | `backend/` image | FastAPI under uvicorn (`--workers 2 --proxy-headers`); health check on `/api/v1/health` | 8000 |
| `worker` | `backend/` image | Celery worker (`--pool=threads --concurrency=2`, queue `documents`); OCR pool of `OCR_WORKERS` processes | - |
| `web` | `frontend/` image | Next.js standalone server; proxies `/api/*` to `http://api:8000` and adds the API key; `/healthz` health check | 3000 |
| `llm` (profile `llm`) | `ghcr.io/ggml-org/llama.cpp:server-b11580` (GPU: `server-cuda-b11580`) | Local OpenAI-compatible model server, models from `./models` (read-only) | 8080 |

Volumes: `pgdata` (database), `redisdata` (broker), `storage` (uploaded originals, mounted at `/data/storage` in `api`,
`worker` and `migrate`). Start order is enforced with health checks: `db` and `redis` healthy -> `migrate` completed ->
`api` -> `web`.

```bash
cp .env.example .env            # then set API_KEYS and BACKEND_API_KEY (same value) and POSTGRES_PASSWORD
docker compose up -d --build
docker compose ps
```

`docker-compose.yml` loads `.env` into the backend containers (`api`, `worker`, `migrate`) through `env_file`, so every
backend setting in `.env` (for example `MAX_UPLOAD_MB`, `RATE_LIMIT_UPLOAD`, `JOB_MAX_ATTEMPTS`, `JOB_MAX_RUNTIME_S`,
`OCR_PAGE_TIMEOUT_S`, `ANTHROPIC_EFFORT`, `LLM_CONTEXT_TOKENS`, `LLM_EXTRA_BODY`) reaches them. The container-specific
values in the `environment` block take precedence over the file: the database and Redis URLs, `QUEUE_BACKEND=celery`,
`STORAGE_DIR`, `LLM_BASE_URL` (taken from `LLM_BASE_URL_IN_DOCKER`) and `LOG_JSON`; the worker also sets `OCR_WORKERS`
(default 4). Edit `.env`, then recreate the containers (`docker compose up -d api worker`). The full list of settings is
in the README configuration table.

## LLM options

- **None:** leave all LLM variables empty; extraction runs on rules alone.
- **Claude / Gemini:** set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in `.env`, then `docker compose up -d api worker`.
- **Local GPU model:** put the GGUF in `./models`, then
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm up -d llm` and
  `docker compose up -d api worker` (with `LLM_BASE_URL_IN_DOCKER=http://llm:8080/v1`). Without a GPU omit the second `-f`.
  Keep `LLM_GGUF` and `LLM_MODEL` in step with the file you downloaded.

**Context variables for bigger models** (all optional, in `.env`; read by `docker-compose.yml` and
`docker-compose.gpu.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_CTX` | `16384` | llama.cpp total context (`-c`) |
| `LLM_PARALLEL` | `2` | llama.cpp slots (`--parallel`); each request gets `LLM_CTX / LLM_PARALLEL` tokens (8192 by default) |
| `LLM_NGL` | `auto` | GPU layers (`-ngl`, GPU file only): `auto` fits layers to free VRAM, `99` forces all and fails to load if the model does not fit |
| `LLM_CONTEXT_TOKENS` | unset | Backend side: tokens one request may use, when the server does not report it |
| `LLM_MAX_INPUT_CHARS` | `60000` | Backend side: upper bound of the document text per call |

The backend sizes its input budget to the model: it asks llama.cpp `/props` (vLLM: `/v1/models`) once for the per-slot
context and uses `(context - LLM_MAX_OUTPUT_TOKENS - prompt overhead) x 3` characters, capped by `LLM_MAX_INPUT_CHARS`
(the budget is logged once at start of use). With the default 16384 / 2 slots and `LLM_MAX_OUTPUT_TOKENS=4096` this is
about 9,200 characters per call; a larger `LLM_CTX`, `LLM_PARALLEL=1` or a smaller `LLM_MAX_OUTPUT_TOKENS` gives more.
llama.cpp started with a unified KV cache (`--kv-unified`) reports the whole shared context per slot, so set
`LLM_CONTEXT_TOKENS` to `context / slots` in that case. Text beyond the budget is truncated (head and tail kept). A model
with 14B or more parameters needs correspondingly more VRAM/RAM; the context itself also costs memory.

`GET /api/v1/ready` (llm field) and `GET /api/v1/stats` show what the API resolved; `llm.fallback_documents` and
`llm.fallback_records` in `/stats` count documents and records that fell back to the rules result because the LLM was
unavailable, failed or refused.

## Scaling

- **More workers:** `docker compose up -d --scale worker=3`. Each worker container runs 2 job threads
  (`--concurrency=2`) and, with `OCR_WORKERS=4`, up to 4 OCR processes, each holding an OCR engine and one rendered page.
  Size `--scale` x `OCR_WORKERS` to the CPU cores and memory available; lower `OCR_WORKERS` if workers are OOM-killed.
  Recommendation: `OCR_WORKERS=4` (the compose default), or `3` when a local LLM shares the machine so OCR leaves CPU to
  the model. Each pool process gets an even share of the CPUs (`max(1, cpus // workers)` threads for ONNX Runtime, OpenMP
  and OpenCV), because letting every process use all cores made pages slower. Measured on a 6-core / 12-thread machine
  with 12 scanned pages at 200 DPI (warm pool, OCR only): 4 workers 1.52 s/page (about 39.5 pages/min), 3 workers 1.60
  s/page, 2 workers 1.83 s/page, in-process (`OCR_WORKERS=0`) 1.75 s/page (about 34 pages/min). Before the thread split, 4
  workers took 2.77 s/page. Your figures depend on the CPU and on what else is running.
- **Per-page OCR timeout:** with `OCR_WORKERS > 0` a page or TIFF frame whose OCR takes longer than `OCR_PAGE_TIMEOUT_S`
  (default 120) counts as an OCR outage and the job is retried. In-process OCR has no per-page timeout.
  Workers are stateless: jobs, progress and results live in PostgreSQL, so workers can be added or removed at any time. A
  worker that dies mid-job is detected by its missing heartbeat (`JOB_STALE_AFTER_S`, default 300) and the job is
  re-queued by any worker's sweeper (runs at start and every minute); Celery also redelivers unacknowledged messages.
- **LLM throughput:** the local `llm` service handles `LLM_PARALLEL` (default 2) parallel requests; each worker limits concurrent LLM calls
  with `LLM_CONCURRENCY`. For heavy volume use a cloud provider or run more llama.cpp instances.
- **API:** the image starts 2 uvicorn workers; run more API containers behind a load balancer if needed (the `api` service
  publishes a fixed host port, so remove the `ports` entry and use a proxy for several replicas). Rate-limit counters are
  shared through Redis.
- **Database:** `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` are per process; keep (API processes + worker threads) x pool within
  PostgreSQL's `max_connections`.
- **Large PDFs:** pages are parsed and OCR'd one at a time and progress updates per page (indexing reports progress per
  batch too); one job keeps one worker thread busy for the whole document. An attempt may run for at most
  `JOB_MAX_RUNTIME_S` (default 14400 s = 4 h): after that it stops heartbeating, the sweeper re-queues it, or fails it as
  `job_timeout` when no attempts are left. The broker's visibility timeout (6 h) deliberately outlasts this deadline.
  Celery task time limits are not used.
- **Memory for very large documents** (measured on a 1000-page digital PDF): the parsed document (28,587 lines, 190,695
  positioned spans) holds about 45 MiB and the process RSS grew by about 63 MiB during the parse; the 1,506 chunk
  vectors are kept as Python float lists until the document is saved, about 18 MiB (computed). Both grow linearly with
  pages and are bounded by `MAX_PDF_PAGES` (2000), so budget roughly 2 x 63 MiB per running job at that limit. A worker
  holds `--concurrency` (2) jobs at once, each process loads the ONNX embedding model once, and each OCR pool process
  is a separate process with its own engine. OCR page images are rendered inside the OCR job and dropped after it.
  Indexing a 200-page digital PDF took about 29 s of the job after the embedding speed-up (measured; was about 56 s).
- **Docker Desktop memory:** on Windows and macOS the containers share one VM (WSL2 or the Docker Desktop VM) that gets a
  fixed share of the host RAM. The `llm` container (model weights plus context), the whole stack (PostgreSQL, Redis, 2
  API processes, the worker with its OCR pool) and heavy jobs together can exhaust it, and the kernel then OOM-kills a
  process. Either run llama.cpp natively on the host (point `LLM_BASE_URL_IN_DOCKER` at
  `http://host.docker.internal:8080/v1`, see README "Use your own model"), raise the VM memory (`memory=` in
  `%UserProfile%\.wslconfig` for WSL2, then `wsl --shutdown`; or the Docker Desktop resources setting), or lower
  `OCR_WORKERS` and the worker count.

## Health, readiness and logs

| Probe | Where | Meaning |
|---|---|---|
| `GET /api/v1/health` | API, no key needed | process is up (used by the container health check) |
| `GET /api/v1/ready` | API, needs the key when `API_KEYS` is set | database answers and the queue backend responds (503 otherwise); reports `llm` configuration |
| `GET /healthz` | web | server is up, no basic auth, does not call the backend |
| `celery inspect ping` | worker | the worker health check |

```bash
curl -s http://localhost:8000/api/v1/health
curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/ready
docker compose logs -f api worker           # JSON lines; every request has a request_id
docker compose logs --since 10m worker | grep -i error
```

Logs contain identifiers, codes and timings but no document text. `processing_events` (via
`GET /api/v1/documents/{id}/events`) holds the per-document processing history and `GET /api/v1/jobs?status=failed`
lists failed jobs; `POST /api/v1/jobs/reprocess-failed` re-queues them.

## Backups and restore

State is in three places: PostgreSQL (`pgdata`), the stored originals (`storage` volume) and Redis (broker only; jobs
are recoverable from the database, so it need not be backed up).

```bash
# database (custom format, compressed)
docker compose exec -T db pg_dump -U docintel -Fc docintel > docintel-$(date +%F).dump

# original files
docker run --rm -v docintel_storage:/data -v "$PWD":/backup alpine tar czf /backup/storage-$(date +%F).tgz -C /data .
```

Restore into an empty database and volume (stop `api` and `worker` first):

```bash
docker compose stop api worker web
docker compose exec -T db dropdb -U docintel docintel && docker compose exec -T db createdb -U docintel docintel
docker compose exec -T db pg_restore -U docintel -d docintel --no-owner < docintel-2026-11-05.dump
docker run --rm -v docintel_storage:/data -v "$PWD":/backup alpine sh -c "cd /data && tar xzf /backup/storage-2026-11-05.tgz"
docker compose up -d
```

`pg_restore` needs the `vector` and `pg_trgm` extensions, which the `pgvector/pgvector` image provides. Back up the
database and the storage volume together: records reference files by storage key. The volume name is
`<project>_storage`, i.e. `docintel_storage`, because of `name: docintel` in the compose file.

## Upgrades

```bash
git pull
docker compose build
docker compose run --rm migrate          # alembic upgrade head (also runs automatically on `up`)
docker compose up -d                     # recreates changed services; workers finish or redeliver in-flight jobs
```

`docker compose up -d` also reruns the `migrate` service before starting `api` and `worker`. Take a database backup
first. Roll back an image by checking out the previous commit and rebuilding; downgrade the schema only with a backup, e.g.
`docker compose run --rm migrate alembic downgrade -1`. Pending queued jobs survive restarts (state is in PostgreSQL;
Redis has AOF persistence).

## Production environment checklist

Set in `.env` before `docker compose up -d`:

- `APP_ENV=production` - the API then refuses to start with empty/default/short (< 24 chars) `API_KEYS`, SQLite, or `CORS_ORIGINS`
  containing `*` (the compose stack already uses PostgreSQL).
- `API_KEYS=<key1>[,<key2>]` and `BACKEND_API_KEY=<key1>`: random, >= 24 characters
  (`python -c "import secrets; print(secrets.token_urlsafe(32))"`). Never reuse the `.env.example` placeholder.
- `POSTGRES_PASSWORD` changed from `docintel-dev-only`.
- `CORS_ORIGINS=https://docintel.example.com` (the UI origin; the browser normally calls the web server, not the API).
- `UI_BASIC_AUTH=user:strong-password`, or protect the UI at the proxy.
- A decided LLM policy (`LLM_PROVIDER`, keys) and `EXTRACTION_MODE`.
- `OCR_WORKERS` and `--scale worker=N` sized for the host; enough disk for `storage` and `pgdata`.
- Backups scheduled and a restore tested; log shipping configured.
- Ports: leave `5433`, `6380`, `8080` on 127.0.0.1 or remove them; expose only the proxy.

See [SECURITY.md](SECURITY.md) for the full hardening checklist and limitations.

## Reverse proxy and TLS

Compose binds everything to `127.0.0.1`, so remote access goes through a proxy on the host (or in the compose network)
that terminates TLS. Point it at the web server (`:3000`); the web server forwards `/api/*` to the API, so the API need
not be public. Uploads can be large (one request up to about `min(MAX_FILES_PER_REQUEST, 5)` x `MAX_UPLOAD_MB`, 1000 MiB by default), so lift the proxy's body-size and
timeout limits and do not buffer request bodies.

Caddy example:

```
docintel.example.com {
    reverse_proxy 127.0.0.1:3000 {
        flush_interval -1
    }
}
```

nginx example (inside a TLS `server` block):

```
client_max_body_size 0;
proxy_request_buffering off;
proxy_read_timeout 600s;
location / { proxy_pass http://127.0.0.1:3000; proxy_set_header Host $host; proxy_set_header X-Forwarded-For $remote_addr; }
```

If clients also call the API directly through the proxy, uvicorn (started with `--proxy-headers` in the image) only trusts
forwarded headers from `127.0.0.1` by default; set the environment variable `FORWARDED_ALLOW_IPS` (add it to the `x-backend`
block) to the proxy's address, otherwise per-IP rate limits see the proxy's IP. Set HSTS at the proxy.
