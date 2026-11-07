# DocIntel - AI-powered document intelligence

DocIntel ingests business documents (PDF, JPG/PNG/TIFF, XLSX/XLS, CSV - single files or batches, PDFs of about 1000
pages), extracts text from digital PDFs, OCRs scanned pages and images, parses spreadsheets, and extracts invoice
data (number, dates, supplier, customer, subtotal, tax, total, currency, line items) as structured JSON **without
inventing values**: every value is either found in the document or `null`. It validates the arithmetic, flags
duplicates and cross-checks PDF against PDF and PDF against Excel, queues whatever needs a human, lets the reviewer
correct values with an audit trail, and makes everything searchable by keyword and by meaning - through a REST API
and a Next.js web interface. Heavy work (OCR, LLM calls) runs in background workers.

The project answers the assignment `Ai Engineer Task 1.3.pdf` (7 tasks, 9 deliverables); see the mapping below.

## Assignment tasks -> features

| # | Task | Where it lives |
|---|---|---|
| 1 | Document processing (multi-format, OCR, spreadsheets, bad/corrupt files) | `backend/app/ingestion/` - magic-byte sniffing, PDF text + per-page OCR fallback, images (EXIF, deskew, contrast), TIFF frames, XLSX/XLS/CSV parsers, typed errors (`corrupt_file`, `encrypted_file`, `file_too_large`, `empty_file`, `unsupported_file_type`, `ocr_unavailable`) |
| 2 | AI information extraction | `backend/app/extraction/` - layout-aware rules extractor + LLM extractor (Claude, Gemini or a local llama.cpp model) with schema-constrained output, grounding (values not in the text become `null`), merge and arithmetic reconciliation |
| 3 | Data validation | `backend/app/validation/` - missing fields, invalid amounts/dates, subtotal/tax/line-items/total consistency, duplicates, PDF-PDF and PDF-Excel cross-verification, review flags (21 issue codes, see `docs/API.md`) |
| 4 | Backend API | `backend/app/api/` + `backend/app/pipeline/` + `backend/app/queue/` - uploads, status, results, search, audited corrections, approval, failed-job reprocessing; Celery + Redis background processing |
| 5 | Database and search | `backend/app/db/`, `backend/alembic/`, `backend/app/search/` - PostgreSQL 17 + pgvector + full-text + trigram; keyword, semantic and hybrid search with structured filters; schema in `docs/DATABASE.md` |
| 6 | Frontend | `frontend/` - Next.js: multi-file upload, live progress, record inspection with confidence/evidence, issue review, inline correction and approval, search, side-by-side compare |
| 7 | Testing and evaluation | `backend/tests/`, `frontend/__tests__/`, `backend/eval/` (synthetic labeled dataset generator, runner, report); results in `docs/TEST_RESULTS.md` and `docs/EVALUATION.md` |

## Architecture

```mermaid
flowchart LR
    user([Finance user]) --> web["Next.js web UI<br/>(:3000)"]
    client([API client]) -->|X-API-Key| api
    web -->|"/api/* proxy<br/>adds X-API-Key server-side"| api["FastAPI REST API<br/>(:8000)"]
    api -->|"validate, store file,<br/>create job"| db[("PostgreSQL 17<br/>pgvector + tsvector + pg_trgm")]
    api --> storage[("File storage<br/>(volume, UUID keys)")]
    api -->|enqueue job id| redis[("Redis<br/>Celery broker")]
    redis --> worker["Celery worker(s)<br/>scale horizontally"]
    worker --> storage
    worker --> db
    worker -->|OCR process pool| ocr["RapidOCR PP-OCRv6<br/>(Tesseract optional)"]
    worker -->|"segment text<br/>(JSON-schema output)"| llm{{"LLM provider<br/>Claude · Gemini ·<br/>llama.cpp (Qwen3.5-9B / Qwen3-4B)"}}
    worker --> emb["fastembed<br/>bge-small-en-v1.5"]
```

The API only validates, stores, hashes and enqueues; workers do the heavy lifting. Job state lives in PostgreSQL (the
broker is disposable). Details, the processing sequence diagram and the design trade-offs are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the schema is in [docs/DATABASE.md](docs/DATABASE.md).

## Quick start (Docker)

Requirements: Docker Desktop / Docker Engine with Compose v2, about 8 GB of free RAM. The backend image bakes in the
OCR and embedding models, so no model is downloaded at runtime (the first build needs internet).

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # run twice or reuse one value
# edit .env: put the generated value in BOTH API_KEYS and BACKEND_API_KEY
docker compose up -d --build
```

(PowerShell: `Copy-Item .env.example .env`.) Then open **http://localhost:3000**, drop some invoices on the
dashboard and follow the processing. The API is on http://localhost:8000 (Swagger UI at `/docs`).

`docker compose up` starts `db` (pgvector/pgvector:pg17), `redis`, `migrate` (one-shot `alembic upgrade head`), `api`,
`worker` and `web`. Without any LLM configured, extraction runs on the rules extractor alone (still grounded and
validated); add one of the providers below for higher accuracy on unusual layouts.

> Compose loads `.env` into the backend containers (`env_file`), so every setting in the
> [Configuration](#configuration) table reaches them. The container-specific values in `docker-compose.yml` (database
> and Redis URLs, `QUEUE_BACKEND=celery`, `STORAGE_DIR`, `LLM_BASE_URL` taken from `LLM_BASE_URL_IN_DOCKER`) take
> precedence over the file. Recreate the containers after editing `.env`: `docker compose up -d api worker`.

### Choose an LLM (optional)

`LLM_PROVIDER=auto` (the default) uses the first configured provider: Claude, then Gemini, then a local server.

**Claude.** Set `ANTHROPIC_API_KEY=...` in `.env` (model `claude-opus-5-5` by default, `ANTHROPIC_MODEL` to change,
`ANTHROPIC_EFFORT` defaults to `medium`), then `docker compose up -d`.

**Gemini.** Set `GEMINI_API_KEY=...` in `.env` (model `gemini-3-flash-preview` by default, `GEMINI_MODEL` to change).

**Local model (llama.cpp, no data leaves the machine).** Put a GGUF file in `./models` (git-ignored). Qwen3.5-9B
Q4_K_M is the default:

```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models
```

A smaller, faster alternative is Qwen3-4B-Instruct-2507 Q4_K_M:

```bash
hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models
# .env: LLM_GGUF=Qwen3-4B-Instruct-2507-Q4_K_M.gguf  LLM_MODEL=qwen3-4b
```

Start the `llm` service (profile `llm`) and point the containers at it - `.env.example` already sets
`LLM_BASE_URL_IN_DOCKER=http://llm:8080/v1`:

```bash
# NVIDIA GPU (needs the NVIDIA Container Toolkit / Docker Desktop with WSL2 GPU support)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm up -d llm
# CPU only
docker compose --profile llm up -d llm
docker compose up -d api worker        # pick up the LLM settings
```

With the GPU file llama.cpp fits as many layers as free VRAM allows (`LLM_NGL=auto`), so a 6 GB GPU runs Qwen3-4B fully
and Qwen3.5-9B partly. By default the server runs with a 16384-token context shared by 2 slots (8k each; `LLM_CTX`,
`LLM_PARALLEL`). The backend sizes the text it sends to the model's per-request context automatically (see below).
Check what is active: `GET /api/v1/config` and `GET /api/v1/stats` report the provider and model; `GET /api/v1/ready`
reports `llm: none | configured | misconfigured`.

### Use your own model

The LLM is pluggable; no code changes are needed. Pick one of the setups below, edit `.env`, then run
`docker compose up -d api worker`.

**(a) Any local GGUF through llama.cpp (the `llm` compose service).** Put the file in `./models` and set in `.env`:

```bash
LLM_PROVIDER=local
LLM_GGUF=<file name in ./models>
LLM_MODEL=<alias the backend sends>
LLM_BASE_URL_IN_DOCKER=http://llm:8080/v1     # containers; LLM_BASE_URL=http://localhost:8080/v1 for a backend on the host
# optional: LLM_CTX=16384  LLM_PARALLEL=2  LLM_NGL=auto   (total context, parallel slots, GPU layers; 99 = all layers)
```

```bash
# NVIDIA GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm up -d llm
# CPU only: the same without the gpu file
docker compose --profile llm up -d llm
```

A bigger model from Hugging Face, for example Qwen3.5-9B Q4_K_M (the default) or any other GGUF:

```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models
# .env: LLM_GGUF=Qwen3.5-9B-Q4_K_M.gguf  LLM_MODEL=qwen3.5-9b
```

Models in the 14B or 32B class need considerably more VRAM or RAM (weights plus the context's KV cache); llama.cpp keeps
the layers that do not fit on the CPU, which works but is slow. Each request gets `LLM_CTX / LLM_PARALLEL` tokens, so
raise `LLM_CTX` (or lower `LLM_PARALLEL`) to give a larger model's requests more room.

**(b) llama.cpp running natively.** Download a llama.cpp release for your platform (on Windows the CUDA build with
`llama-server.exe`) and start it yourself, for example:

```powershell
.\llama-server.exe -m models\Qwen3.5-9B-Q4_K_M.gguf --alias qwen3.5-9b --host 0.0.0.0 --port 8080 -c 16384 --parallel 2 --jinja -ngl auto
```

Use the same `.env` settings: `LLM_PROVIDER=local`, `LLM_MODEL` equal to the alias, `LLM_BASE_URL=http://localhost:8080/v1`
for a backend on the host, and `LLM_BASE_URL_IN_DOCKER=http://host.docker.internal:8080/v1` for the Docker containers.
Running the model natively keeps its memory out of the Docker Desktop VM (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

**(c) Other OpenAI-compatible servers (Ollama, LM Studio, vLLM).** Point the backend at the server's `/v1` endpoint:
`LLM_PROVIDER=local`, `LLM_BASE_URL=...`, `LLM_MODEL=<model name the server uses>`, and `LLM_API_KEY` if it wants one.
Two optional settings:
- `LLM_EXTRA_BODY`: a JSON object merged into every request, for example to switch off thinking on a Qwen model served
  by vLLM: `LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}`. Unset, that switch is sent only to
  llama.cpp servers and other servers get the plain OpenAI body; `{}` sends nothing extra.
- `LLM_CONTEXT_TOKENS`: tokens one request may use, when the server does not report its context (or runs llama.cpp
  with a unified KV cache, where the reported context is the whole shared pool: set it to context / slots).

**(d) Claude or Gemini.** `ANTHROPIC_API_KEY` (default model `claude-opus-5-5`, `ANTHROPIC_MODEL` to change) or
`GEMINI_API_KEY` (`GEMINI_MODEL`, default `gemini-3-flash-preview`). Document text is sent to that provider.
A free-tier Google AI Studio key has small quotas: on this project's key `gemini-3-flash-preview` allowed 20 requests
per day (about one invoice each), so a free key suits trying the provider, not processing a batch. When a 429 asks to
retry within a minute the backend waits and retries; when the quota is spent it falls back to the rules result at once
(the record gets the info issue `LLM_UNAVAILABLE`, and the worker log says "quota exhausted"). With `LLM_PROVIDER=auto`, setting `GEMINI_API_KEY` makes Gemini win over a
local server, so set `LLM_PROVIDER=local` to keep using the local model. On the free tier Google may use the content
to improve its products; enable billing on the key's Google Cloud project before sending real documents.

**How it is chosen and sized.** `LLM_PROVIDER=auto` picks the first configured provider in the order anthropic, gemini,
local (a local server counts when `LLM_BASE_URL` is set); set `LLM_PROVIDER` explicitly to force one, or `none` for rules
only. For local and other OpenAI-compatible models the input budget follows the model's context automatically: the
backend reads the per-request context from llama.cpp `/props` (vLLM: `/v1/models`) once, and sends
`(context - LLM_MAX_OUTPUT_TOKENS - prompt) x 3` characters at most, never more than `LLM_MAX_INPUT_CHARS`. Claude and
Gemini use `LLM_MAX_INPUT_CHARS`. Longer documents are truncated (head and tail kept).

**What to expect.** A stronger model generally improves extraction on hard layouts and unusual wording. It cannot fix
OCR misreads on very poor scans (the model only sees the OCR text), and it plays no part in spreadsheet parsing, which is
rule-based. Bigger local models are slower, especially on small GPUs where some layers stay on the CPU. Measured
results for the configurations that were run are in [docs/EVALUATION.md](docs/EVALUATION.md).

### Ports

| Port (127.0.0.1 only) | Service |
|---|---|
| 3000 | web UI (Next.js) |
| 8000 | REST API, Swagger UI at `/docs`, ReDoc at `/redoc` |
| 5433 | PostgreSQL (container port 5432) |
| 6380 | Redis (container port 6379) |
| 8080 | local LLM server (`llm` profile only) |

All published ports are bound to 127.0.0.1. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for scaling workers, backups,
upgrades and a production checklist.

## Local development without Docker

Python 3.12 and Node 24 are expected. The backend defaults to SQLite (`backend/data/app.db`), the in-process `local`
queue and local file storage, so no Postgres or Redis is needed.

```bash
cd backend
uv venv --python 3.12                       # or: python -m venv .venv
uv pip install -r requirements-dev.lock     # or: pip install -r requirements-dev.lock
```

Settings are read from the repo-root `.env`, then `backend/.env`; real environment variables win. **If you copied
`.env.example` to `.env` for Docker, it points a host-side backend at the compose Postgres and Redis**
(`DATABASE_URL=...localhost:5433`, `QUEUE_BACKEND=celery`). For a zero-dependency run override them:

```bash
# bash
export DATABASE_URL=sqlite:///data/app.db QUEUE_BACKEND=local API_KEYS=
uvicorn app.main:app --reload --port 8000
```

```powershell
# PowerShell
$env:DATABASE_URL = "sqlite:///data/app.db"; $env:QUEUE_BACKEND = "local"; $env:API_KEYS = ""
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

(`fastapi dev app/main.py` works too.) On SQLite the tables are created at startup and jobs run in a thread pool in
the API process (`LOCAL_QUEUE_WORKERS`, default 2); jobs left `queued` by a previous run are re-queued at startup.
With PostgreSQL run `alembic upgrade head` first (`docker compose up -d db redis`, then `cd backend && alembic upgrade
head`).

Frontend:

```bash
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev      # http://localhost:3000
```

PowerShell: `$env:BACKEND_URL = "http://localhost:8000"; npm run dev`. Add `BACKEND_API_KEY=<key>` when the backend
has `API_KEYS` set. See [frontend/README.md](frontend/README.md).

## Configuration

Backend settings are defined in `backend/app/config.py` and read from environment variables (case-insensitive).
In Docker the same names in `.env` apply to the backend containers (compose loads the file through `env_file`); where
`docker-compose.yml` sets a container-specific value, it is noted in the row and takes precedence.

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test` or `production`. `production` refuses to start with no/default/short API keys, SQLite or wildcard CORS |
| `APP_NAME` | `DocIntel` | Title of the API |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_JSON` | `false` | JSON log lines (compose forces `true`) |
| `DATABASE_URL` | `sqlite:///<backend>/data/app.db` | SQLAlchemy URL; use `postgresql+psycopg://...` for PostgreSQL. In Docker: built from `POSTGRES_PASSWORD`. |
| `DB_POOL_SIZE` | `10` | Connection pool size (PostgreSQL) |
| `DB_MAX_OVERFLOW` | `20` | Extra pooled connections |
| `REDIS_URL` | unset | Celery broker and shared rate-limit counters. In Docker: `redis://redis:6379/0`. |
| `QUEUE_BACKEND` | `local` | `inline` (synchronous, tests), `local` (thread pool in the API process), `celery`. In Docker: `celery`. |
| `LOCAL_QUEUE_WORKERS` | `2` | Threads of the `local` queue |
| `STORAGE_DIR` | `<backend>/data/storage` | Where uploaded originals are stored. In Docker: `/data/storage` (volume). |
| `MAX_UPLOAD_MB` | `200` | Largest accepted file (enforced while the upload streams; an oversized file is rejected as `file_too_large`) |
| `MAX_FILES_PER_REQUEST` | `50` | Files per upload request. The request body is capped at `min(MAX_FILES_PER_REQUEST, 5) x MAX_UPLOAD_MB` (1000 MiB by default, `max_request_mb` in `GET /api/v1/config`) |
| `MAX_PDF_PAGES` | `2000` | PDF page / TIFF frame limit |
| `MAX_IMAGE_PIXELS` | `120000000` | Decompression-bomb guard for images |
| `MAX_SHEET_ROWS` | `200000` | Rows read per sheet |
| `MAX_SHEET_COLS` | `500` | Columns read per sheet |
| `MAX_XLSX_UNCOMPRESSED_MB` | `500` | Zip-bomb guard: total uncompressed workbook size |
| `MAX_XLSX_COMPRESSION_RATIO` | `200` | Zip-bomb guard: per-member compression ratio |
| `OCR_ENGINE` | `rapidocr` | `rapidocr` or `tesseract` |
| `OCR_DPI` | `200` | Render resolution for OCR of PDF pages |
| `OCR_WORKERS` | `0` | `0` = OCR inline in the calling process, `N` = process pool of N. In Docker: worker: `OCR_WORKERS` (default `4`). |
| `OCR_PAGE_TIMEOUT_S` | `120.0` | Pooled OCR (`OCR_WORKERS` > 0): a page or TIFF frame taking longer counts as an OCR outage and the job is retried. In-process OCR has no per-page timeout |
| `OCR_MIN_TEXT_CHARS` | `25` | A PDF page with fewer alphanumeric characters is OCR'd |
| `OCR_LOW_CONFIDENCE` | `0.75` | Mean OCR confidence below this raises `OCR_LOW_QUALITY` |
| `EXTRACTION_MODE` | `hybrid` | `rules`, `llm` or `hybrid` |
| `DEFAULT_DOLLAR_CURRENCY` | `USD` | What a bare `$` means when nothing else says |
| `DATE_ORDER` | `auto` | `auto`, `DMY` or `MDY` for ambiguous dates |
| `LLM_PROVIDER` | `auto` | `auto`, `anthropic`, `gemini`, `local` or `none` |
| `ANTHROPIC_API_KEY` | unset | Claude API key |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model |
| `ANTHROPIC_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh` or `max` |
| `GEMINI_API_KEY` | unset | Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint of a local server, e.g. `http://localhost:8080/v1`. In Docker: from `LLM_BASE_URL_IN_DOCKER`. |
| `LLM_MODEL` | `qwen3.5-9b` | Model alias sent to the local server |
| `LLM_API_KEY` | unset | Bearer token for the local server, if it needs one |
| `LLM_TIMEOUT_S` | `180.0` | Per-request LLM timeout |
| `LLM_MAX_INPUT_CHARS` | `60000` | Upper bound of the document text sent to the LLM per call; for local models the effective budget is derived from the model's context and never exceeds this |
| `LLM_CONTEXT_TOKENS` | unset | Tokens one request may use on the local / OpenAI-compatible model; unset = ask the server (llama.cpp `/props`, vLLM `/v1/models`). Set it when the server does not report it or uses a unified KV cache |
| `LLM_EXTRA_BODY` | unset | JSON object merged into every OpenAI-compatible request, e.g. `{"chat_template_kwargs":{"enable_thinking":false}}`; unset = that switch only for llama.cpp servers, `{}` = nothing extra |
| `LLM_CONCURRENCY` | `2` | Concurrent LLM calls |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | Output token budget |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` or `hashing` (deterministic, for tests) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `EMBEDDING_DIM` | `384` | Embedding dimension (matches the `chunks.embedding` column) |
| `EMBEDDING_CACHE_DIR` | unset | Model cache directory (image sets `/opt/models/fastembed`) |
| `CHUNK_MAX_CHARS` | `1200` | Search chunk size |
| `CHUNK_OVERLAP` | `150` | Search chunk overlap |
| `AMOUNT_TOLERANCE` | `0.02` | Absolute tolerance for money equality |
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.6` | Field confidence below this raises `LOW_CONFIDENCE` |
| `MAX_REASONABLE_AMOUNT` | `1000000000` | Larger amounts are `INVALID_AMOUNT` |
| `TAX_RATE_MAX_PERCENT` | `30` | Larger tax rates raise `TAX_RATE_UNUSUAL` |
| `JOB_MAX_ATTEMPTS` | `3` | Attempts per job (transient errors retry with backoff) |
| `JOB_STALE_AFTER_S` | `300` | A running job without a heartbeat for this long is considered dead |
| `JOB_MAX_RUNTIME_S` | `14400` | Deadline of one processing attempt (4 h): after it the attempt stops heartbeating, so the sweeper re-queues it or fails it as `job_timeout` |
| `API_KEYS` | empty | Comma-separated accepted API keys; empty disables auth (refused in production) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `RATE_LIMIT_ENABLED` | `true` | Switch rate limiting off with `false` |
| `RATE_LIMIT_UPLOAD` | `60/minute` | Upload budget per caller |
| `RATE_LIMIT_SEARCH` | `300/minute` | Search budget per caller |
| `RATE_LIMIT_DEFAULT` | `1200/minute` | Budget of every other route |
| `DOCINTEL_IGNORE_DOTENV` | unset | `1` makes the backend ignore `.env` files (tests, evaluation) |

Compose-level variables (in `.env`, used by `docker-compose.yml`): `POSTGRES_PASSWORD` (default `docintel-dev-only`),
`BACKEND_API_KEY` (key the web server sends to the API), `UI_BASIC_AUTH` (`user:pass` protecting the web UI),
`LLM_GGUF` (file in `./models` served by the `llm` service, default `Qwen3.5-9B-Q4_K_M.gguf`), `LLM_BASE_URL_IN_DOCKER`, and for the `llm` service `LLM_CTX` (total context, default
`16384`), `LLM_PARALLEL` (slots, default `2`) and, with the GPU file, `LLM_NGL` (GPU layers, default `auto`).
`LLM_BASE_URL` in `.env` is the host-side URL (for backend processes run outside Docker).

The web server reads `BACKEND_URL`, `BACKEND_API_KEY` and `UI_BASIC_AUTH` at runtime
([frontend/README.md](frontend/README.md)).

## API overview

REST under `/api/v1`, JSON, errors as `{"error": {"code", "message", "details"}}`. When `API_KEYS` is set, every route
except `GET /api/v1/health` needs `X-API-Key: <key>` (or `Authorization: Bearer <key>`).

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/documents` | Upload one or many files (multipart field `files`) -> 202 with one result per file |
| `GET /api/v1/documents`, `GET /api/v1/documents/{id}` | List/filter, detail with records and issues |
| `GET /api/v1/documents/{id}/status` / `/results` / `/pages` / `/events` / `/file` | Poll status, records, extracted text, history, original file |
| `POST /api/v1/documents/{id}/reprocess`, `POST /api/v1/jobs/reprocess-failed` | Re-run one document / every failed document |
| `GET /api/v1/invoices`, `GET/PATCH /api/v1/invoices/{id}` | Filter records, correct fields (audited) |
| `POST /api/v1/invoices/{id}/approve`, `GET /api/v1/invoices/{id}/corrections` | Approve after review, correction history |
| `GET /api/v1/search` | Keyword, semantic or hybrid search with structured filters |
| `POST /api/v1/cross-verify`, `GET /api/v1/cross-verifications` | Compare two records / list comparisons |
| `GET /api/v1/stats`, `/health`, `/ready`, `/config` | Dashboard numbers, liveness, readiness, limits |

```bash
curl -H "X-API-Key: $API_KEY" -F "files=@invoice.pdf" -F "files=@ledger.xlsx" http://localhost:8000/api/v1/documents
```

Full reference with request/response shapes and examples: [docs/API.md](docs/API.md). Machine-readable schema:
[docs/openapi.json](docs/openapi.json) (also served at http://localhost:8000/openapi.json; interactive docs at
http://localhost:8000/docs).

## Testing

Backend (from `backend/`, with the virtualenv active; every test runs against a fresh SQLite database, the inline
queue, the hashing embedder and no LLM, and ignores `.env`):

```bash
python -m pytest tests/unit -q          # ingestion, OCR, spreadsheets, extraction, validation, search, pipeline, eval
python -m pytest tests/api -q           # REST API incl. security, hardening and the full upload-to-search flow
python -m ruff check . && python -m mypy app     # lint and types
```

Integration tests (marker `integration`) run the same code against PostgreSQL + pgvector (migrations, full-text and
vector search) and are skipped unless `TEST_DATABASE_URL` is set. **They create and drop tables, so point them at a
throwaway database, never at your data**:

```bash
docker run -d --name docintel-pg-test -p 5434:5432 -e POSTGRES_USER=docintel -e POSTGRES_PASSWORD=docintel \
  -e POSTGRES_DB=docintel_test pgvector/pgvector:pg17
TEST_DATABASE_URL=postgresql+psycopg://docintel:docintel@localhost:5434/docintel_test python -m pytest tests/integration -q
```

PowerShell: `$env:TEST_DATABASE_URL = "postgresql+psycopg://docintel:docintel@localhost:5434/docintel_test"; python -m pytest tests/integration -q`.
`TEST_REDIS_URL` additionally points integration tests at a Redis. Tests marked `live_llm` call a real model and are
only meaningful with `LLM_BASE_URL` or an API key set; `slow` marks long-running tests.

Frontend (from `frontend/`):

```bash
npm run lint && npm run typecheck && npm test     # Vitest + Testing Library
```

End-to-end (Playwright, specs in `frontend/e2e/`) run against the running compose stack with
`npm run test:e2e` from `frontend/` (a bare `npx playwright test` runs the same e2e project). The walkthrough video
overwrites `video/*`, so it is recorded only on request with `npm run test:walkthrough` (or `WALKTHROUGH=1`).
Commands, counts and coverage of the last run are in [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

## Evaluation

The evaluation drives the real pipeline over a labeled dataset that is generated from code with a fixed seed (6 invoice
templates; T1-T3 are the development split, T4-T6 the held-out split). From `backend/`:

```bash
# 1. generate the dataset (git-ignored, deterministic for a seed); --scale adds the ~1000-page and 100-page PDFs
python -m eval.synth.build --out eval/dataset --seed 42 --scale

# 2. run one or more configurations: rules | hybrid-local-4b | hybrid-local-9b | hybrid-gemini | hybrid-claude
python -m eval.run_eval --dataset eval/dataset --config rules --out eval/results/rules.json
python -m eval.run_eval --dataset eval/dataset --config hybrid-local-9b --out eval/results/hybrid-local-9b.json
python -m eval.run_eval --dataset eval/dataset --config rules --scale-only --out eval/results/rules-scale.json

# 3. write the Markdown reports: the overview of every run, and one report per model (each next to the rules baseline)
python -m eval.report "eval/results/*.json" --notes eval/notes/overview.md --out ../docs/EVALUATION.md
python -m eval.report eval/results/rules.json eval/results/rules-postfix.json \
  eval/results/hybrid-local-4b.json eval/results/hybrid-local-4b-postfix.json \
  --title "Evaluation: Qwen3-4B (local, llama.cpp)" --notes eval/notes/qwen3-4b.md --out ../docs/evaluation/qwen3-4b.md
```

Each run writes its own log next to its results, `eval/results/logs/<results name>.log` (only when the run succeeds,
so a refused run never replaces a good log), and the reports name the log of every run they include. `--notes` adds a
hand-written Markdown file (run notes kept in `eval/notes/`) after the summary, so regenerating a report keeps it.

The local configurations expect a llama.cpp server on port 8080 (Qwen3.5-9B, alias `qwen3.5-9b`) or 8081 (Qwen3-4B,
alias `qwen3-4b`); the cloud configurations need `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` (environment or `.env`).
`--limit N`, `--include-scale`, `--embedding hashing`, `--ocr-workers N` and `--skip-llm-check` are available; see
`python -m eval.run_eval --help`. Methodology, measured metrics and limitations are in
[docs/EVALUATION.md](docs/EVALUATION.md), with a separate report per model in
[docs/evaluation/qwen3-4b.md](docs/evaluation/qwen3-4b.md) and [docs/evaluation/gemini.md](docs/evaluation/gemini.md)
(Gemini: not measured, by the owner's decision, after the free-tier key's daily quota proved too small for a full
run; the page says why and how to measure it).

## Security

API-key authentication with constant-time comparison, upload hardening (limits, magic-byte type detection, sanitised
names, random storage keys, decompression-bomb guards), ORM-only SQL, rate limits, CORS allow-list, security headers,
a CSRF guard and server-side key injection in the web proxy, prompt-injection controls around the LLM, logs without
document text, non-root containers and production startup checks. Threat model, every control with its code location,
limitations and a hardening checklist: [docs/SECURITY.md](docs/SECURITY.md).

## Project structure

```
.
├── README.md
├── docker-compose.yml           # db, redis, migrate, api, worker, web, llm (profile)
├── docker-compose.gpu.yml       # NVIDIA override for the llm service
├── .env.example
├── models/                      # GGUF files for the local LLM (git-ignored)
├── docs/
│   ├── ARCHITECTURE.md  DATABASE.md  API.md  SECURITY.md  DEPLOYMENT.md  AI_USAGE.md
│   ├── TEST_RESULTS.md  EVALUATION.md  openapi.json
│   └── evaluation/              # one evaluation report per model (qwen3-4b.md, gemini.md)
├── video/                       # recorded walkthrough (walkthrough.mp4 + .webm, screenshots/, README.md timeline)
├── backend/
│   ├── Dockerfile  pyproject.toml  requirements*.lock  alembic/
│   ├── app/
│   │   ├── main.py  config.py  storage.py  logging_setup.py
│   │   ├── api/                 # routes, schemas, errors, middleware, corrections
│   │   ├── security/            # auth, rate limits, headers, upload hardening
│   │   ├── ingestion/           # sniffing, PDF, images, OCR, spreadsheets, layout
│   │   ├── extraction/          # rules, LLM providers + prompt, grounding, merge, segmentation
│   │   ├── validation/          # rules, duplicates, cross-verification, review status
│   │   ├── pipeline/            # intake, processor (stages, progress, retries), review
│   │   ├── queue/               # inline, local and Celery backends
│   │   ├── search/              # chunking, embeddings, keyword/semantic/hybrid search
│   │   ├── db/                  # models, session, PostgreSQL extras
│   │   └── domain/              # shared schemas (InvoiceData, IssueCode, ...)
│   ├── eval/                    # dataset generator, runner, metrics, report
│   └── tests/                   # unit/, api/, integration/
└── frontend/                    # Next.js app, components, same-origin API proxy, Vitest tests
```

## Deliverables checklist

| # | Deliverable | Where |
|---|---|---|
| 1 | Git repository with source code | this repository (branch `main`) |
| 2 | README and setup instructions | this file; [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| 3 | Architecture diagram and database schema | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DATABASE.md](docs/DATABASE.md) |
| 4 | Working backend API and frontend | `backend/app/`, `frontend/`; [docs/API.md](docs/API.md), [docs/openapi.json](docs/openapi.json) |
| 5 | PDF, image, Excel and CSV processing pipelines | `backend/app/ingestion/`, `backend/app/extraction/`, `backend/app/pipeline/` |
| 6 | Automated tests and test results | `backend/tests/`, `frontend/__tests__/`, `frontend/e2e/`; [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) |
| 7 | Evaluation report with measured metrics | [docs/EVALUATION.md](docs/EVALUATION.md); generator in `backend/eval/` |
| 8 | Recorded walkthrough | [video/walkthrough.mp4](video/walkthrough.mp4) (2 min 49 s; WebM original, screenshots and timeline in [video/](video/)) |
| 9 | AI prompts and AI-assistance disclosure | [docs/AI_USAGE.md](docs/AI_USAGE.md) |

## Troubleshooting

- **Port already in use** (3000, 8000, 5433, 6380, 8080): another service holds the port. Stop it, or change the host
  side of the mapping in `docker-compose.yml` (for example `"127.0.0.1:8001:8000"`). PostgreSQL and Redis use 5433 and
  6380 on the host precisely to avoid clashing with local installs on 5432/6379.
- **`api` stays unhealthy / exits at start**: `docker compose logs api`. With `APP_ENV=production` the API refuses default
  or short (< 24 characters) keys, SQLite and `CORS_ORIGINS=*` and prints which one.
- **401 from the web UI after changing keys**: `API_KEYS` (API) and `BACKEND_API_KEY` (web) must contain the same value;
  recreate both containers: `docker compose up -d api worker web`.
- **Documents stay `queued`**: the worker is not running or cannot reach Redis (`docker compose ps`, `docker compose logs
  worker`). `GET /api/v1/ready` shows database/queue state. Workers re-send lost queued jobs once a minute.
- **Local LLM not used**: `docker compose --profile llm ps`; the GGUF file name must match `LLM_GGUF` and exist in
  `./models`; `LLM_BASE_URL_IN_DOCKER` must be set for the containers (`http://llm:8080/v1`), and `api worker` must be
  recreated after editing `.env`. `curl http://localhost:8080/v1/models` should answer.
- **GPU not visible in Docker**: use Docker Desktop with the WSL2 backend and a current NVIDIA driver (Windows), or the
  NVIDIA Container Toolkit (Linux); test with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
  Without a GPU use the CPU command (no `-f docker-compose.gpu.yml`); expect slow extraction.
- **Model download**: `hf download` needs `huggingface_hub[cli]`; downloads resume if interrupted. Files must end up in
  `./models` (the folder is mounted read-only into the `llm` container).
- **Windows paths and shells**: use forward slashes or quoted paths; the venv interpreter is
  `backend\.venv\Scripts\python.exe`; set variables with `$env:NAME = "value"` in PowerShell. Avoid placing the repo
  in a synced/OneDrive folder (slow file watching, locked files). `glob` patterns such as `eval/results/*.json` are
  expanded by `eval.report` itself, so they work in PowerShell too.
- **Out of memory while OCR-ing big PDFs**: lower `OCR_WORKERS` (each OCR process holds an engine and one rendered page).
- **Frontend cannot reach the API in dev**: set `BACKEND_URL` (default `http://localhost:8000`) and, when the API has keys,
  `BACKEND_API_KEY`.
