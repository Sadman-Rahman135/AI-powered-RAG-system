# Test results

Last full run: **2026-11-05**, on one laptop: AMD Ryzen 7 7445HS (6 cores / 12 threads), 15.3 GB RAM, NVIDIA RTX
4050 Laptop GPU (6 GB), Windows 11. The backend tests, lint and types ran on the latest
code, commit `4b225a6`. Coverage, the frontend checks and the end-to-end run ran at `119ca45`; the commits after it
change only the Gemini provider, the evaluation tooling and documents, none of which the frontend or the e2e stack
(local model) uses. On 2026-11-07, just before publishing, the backend suite (PostgreSQL and live-LLM tests included) and the Vitest suite were run once more on the published code: 1468 passed, 1 skipped, and 127 passed. Every count below was measured; nothing is copied from an earlier run. Commit IDs quoted in the documents (such as `7907c58`) name the point in the development history at which a measurement was taken; the published repository holds the finished work as a single commit, so they identify runs rather than browsable commits. Accuracy of the extraction itself is measured separately, over a labeled
dataset, in [EVALUATION.md](EVALUATION.md).

## Summary

| Check | Command | Result | Time |
| :--- | :--- | :--- | ---: |
| Backend tests (unit, API, PostgreSQL integration, live local LLM) | `pytest` in `backend/` | **1468 passed, 1 skipped** | 2 min 50 s |
| Backend lint | `ruff check .` | no findings | |
| Backend types | `mypy app` | no issues in 92 source files | |
| Backend coverage | `pytest --cov=app` | **96 %** of statements (9,333; 379 never run) | 3 min 34 s |
| Frontend unit and component tests (Vitest) | `npm test -- --maxWorkers=2` | **127 passed** (18 files) | 20 s |
| Frontend lint | `npm run lint` | no findings | |
| Frontend types | `npm run typecheck` | no errors | |
| End-to-end in a real browser (Playwright) against the Docker stack | `npm run test:e2e` | **21 passed** | 2.0 min |

The one skipped backend test needs the Tesseract binary, which is not installed on this machine. The default OCR
engine, RapidOCR, is covered, and Tesseract was checked separately inside the Docker image, where it is installed.

## Backend tests by area

| Area | Tests | What they cover |
| :--- | ---: | :--- |
| `tests/unit/ingestion` | 105 | File sniffing; PDF text and scanned pages; images, including multi-frame TIFF; XLSX/XLS/CSV; the OCR process pool and its timeouts; corrupted and truncated files; decompression-bomb guards |
| `tests/unit/extraction` | 662 | Label vocabulary, value parsers, the rules extractor, segmentation of multi-invoice files, the spreadsheet mapper, the LLM providers, prompt and grounding, the input budget, merging rules and LLM answers |
| `tests/unit/validation` | 206 | Missing fields, arithmetic checks (total, line items, tax), duplicates, inconsistencies, review status, PDF-to-PDF and PDF-to-Excel cross-verification |
| `tests/unit/pipeline` | 109 | Background job stages, retries, the heartbeat and attempt deadline, stale-job reclaim, reprocessing and deletion, storage; 4 of them run on PostgreSQL |
| `tests/unit/search` | 68 | Keyword, semantic and hybrid (RRF) search, filters, embeddings |
| `tests/unit/eval` | 72 | The synthetic dataset generator, the evaluation runner, the metrics and the report |
| `tests/unit/test_contracts.py` | 9 | The shared schemas and enums every module relies on |
| `tests/api` | 198 | Every REST endpoint: upload limits, authentication, rate limits, error envelope, security headers, pagination, corrections and approvals |
| `tests/integration` | 40 | Alembic migrations up and down, PostgreSQL full-text, trigram and pgvector search, including a needle among thousands of chunks |

## Coverage

Statement coverage of `backend/app` over the full run is 96 % (`pytest --cov=app --cov-report=term-missing` lists
the lines not run). The least-covered modules:

| Module | Coverage |
| :--- | ---: |
| `app/ingestion/images.py` | 79 % |
| `app/ingestion/spreadsheets.py` | 85 % |
| `app/ingestion/ocr.py` | 86 % |
| `app/queue/local.py` | 86 % |
| `app/ingestion/pdf.py` | 87 % |
| `app/api/routes/health.py` | 89 % |

Every other module is at 91 % or more. Frontend coverage is not measured (no coverage provider is installed for
Vitest); the frontend is covered by the 127 Vitest tests and the 21 end-to-end tests.

## What runs where

By default every test uses an isolated SQLite database, a temporary storage directory, the inline queue, a
deterministic hashing embedder and no LLM, so the suite needs no services and no network. Three groups need more and
**skip themselves** (with the reason printed) when it is missing:

| Group | Tests | Enabled by |
| :--- | ---: | :--- |
| PostgreSQL integration (marker `integration`) | 44 | `TEST_DATABASE_URL`, a throwaway database from the `pgvector/pgvector` image |
| Live local LLM | 2 | `LIVE_LLM_BASE_URL`, any running OpenAI-compatible server (llama.cpp here) |
| Tesseract engine | 1 | the `tesseract` binary on `PATH` |

So on a machine with no Docker and no LLM, `pytest` reports 1422 passed and 47 skipped. The run above had PostgreSQL
and the LLM enabled.

## How to run

Backend, from `backend/` with the virtual environment active:

    pytest                                   # everything that needs no services
    docker run -d --name docintel-pg-test -e POSTGRES_USER=docintel -e POSTGRES_PASSWORD=docintel \
      -e POSTGRES_DB=docintel_test -p 127.0.0.1:5434:5432 pgvector/pgvector:pg17
    TEST_DATABASE_URL=postgresql+psycopg://docintel:docintel@localhost:5434/docintel_test \
    LIVE_LLM_BASE_URL=http://127.0.0.1:8081/v1 pytest      # what the run above used
    pytest --cov=app --cov-report=term       # coverage
    ruff check .
    mypy app

Frontend, from `frontend/`:

    npm test -- --maxWorkers=2               # Vitest; two workers keep memory use low on a 16 GB laptop
    npm run lint
    npm run typecheck
    npm run test:e2e                         # needs the stack: docker compose up -d --build (from the repo root)

The end-to-end specs talk to the real stack at `E2E_BASE_URL` (default `http://127.0.0.1:3000`): the web UI and its
server-side API proxy, the API, the Celery worker with real OCR, PostgreSQL and Redis. Nothing is mocked. They upload
files from the evaluation dataset, so build it first (`python -m eval.synth.build --out eval/dataset --seed 42`, see
[EVALUATION.md](EVALUATION.md)). `npm run test:walkthrough` re-records the walkthrough video in `video/` and
is not part of the test run.

## End-to-end run

The stack was rebuilt from the code above (`docker compose up -d --build`) and ran with hybrid extraction on the local
Qwen3-4B model (llama.cpp b11580 on the host GPU, `LLM_PROVIDER=local`), RapidOCR and fastembed embeddings. The 21
tests run one at a time in a real Chromium:

| Spec | Tests | What it does |
| :--- | ---: | :--- |
| `upload.spec.ts` | 3 | Uploads a mixed batch (a PDF, a truncated PDF, random bytes named `.pdf`, an empty file): the empty file is refused in the browser, the fake PDF by the API, and the others are followed to their final status; uploads the same bytes again and checks the duplicate warning and the link to the original; deletes a document after confirming |
| `review.spec.ts` | 2 | Opens a record from the review queue, fills in a missing field and saves (audited, re-validated, issues updated); approves it, so it is reviewed and leaves the queue |
| `search.spec.ts` | 4 | Keyword search with highlighted matches, semantic search without the invoice's words, filter-only search |
| `compare.spec.ts` | 4 | A PDF invoice and its ledger row are cross-checked during processing, then compared field by field and as whole documents on the Compare page |
| `smoke.spec.ts` | 8 | Every page and every document tab loads without console errors; client-side navigation; not-found states |

**The first e2e run on this code failed 2 of 21 tests (18 passed, 1 did not run), and both failures were real bugs,**
now fixed, each with a regression test:

- **Approving without a comment returned 500** (`43cb623`). The UI sends `{"comment": null}` for an empty comment box,
  and the 2000-character limit on comments was applied to the `null` as well, which raised a `TypeError` inside
  validation. Saving a correction with `"comment": null`, or with a blank comment, failed the same way. The API tests
  had only sent `{}`, a real comment or no body. New test: `test_a_null_or_blank_comment_is_no_comment`.
- **A file uploaded while the documents table was still loading never appeared in it** (`119ca45`). The browser trace
  showed the page's first list request starting 25 ms before the upload, and no list request after it. TanStack Query
  reuses an in-flight request for a query that has no data yet, and that request had been answered before the new
  document existed; with nothing left processing, the table stopped polling. The refresh after an upload now cancels
  in-flight requests first. New test: `__tests__/queries.test.ts`.

The re-run after the fixes passed 21 of 21, and so did a last run after the API and worker were rebuilt at `c3e0dfc` (1.8 min). An earlier run of the same specs (at `7907c58`, before the final fix wave)
passed 21 of 21 as well; the API container it ran against was most likely built before the comment limit was added,
and the upload race depends on timing.

## Fixes after the evaluation was frozen

The primary evaluation numbers come from commit `7907c58`, frozen before the final review so that no fix could be
tuned on the measured runs. These commits came after it. Their combined effect on accuracy is shown separately in
EVALUATION.md (the `-postfix` columns, re-run at `e922bae`; the commits after `e922bae` do not change extraction behaviour),
and the tests above cover all of them.

| Commit | Area | Changes |
| :--- | :--- | :--- |
| `3a9ce26` | Settings | New settings: `OCR_PAGE_TIMEOUT_S`, `JOB_MAX_RUNTIME_S`, `LLM_CONTEXT_TOKENS`, `LLM_EXTRA_BODY` |
| `896a60c` | OCR | Each OCR worker process gets a fair share of the CPU threads (scanned pages ≈1.8× faster with 4 workers, 2.77 → 1.52 s/page); a per-page OCR timeout discards a hung pool and retries the page |
| `ab5dde2` | Extraction | LLM input sized to the model's context window (read from llama.cpp `/props` or `LLM_CONTEXT_TOKENS`); llama.cpp-only request fields are now opt-in (`LLM_EXTRA_BODY`), so other OpenAI-compatible servers work; fewer duplicate review flags; stricter grounding of invoice numbers; safer choice between two line-item lists; a truncated LLM answer is retried without line items instead of falling back entirely; label and parser fixes, including "Date of Supply" no longer read as the invoice date; the compose `llm` service takes `LLM_CTX`, `LLM_PARALLEL` and `LLM_NGL` |
| `29118a7` | Pipeline and API | Deleting a document re-checks the invoices it was cross-verified against; an attempt that outlives `JOB_MAX_RUNTIME_S` stops heartbeating and is reclaimed (Celery's own time limits do nothing with the thread pool); the per-file upload limit is enforced while the upload streams, not after it is spooled; stale-job reclaim is compare-and-set; progress writes check the attempt; one cached LLM client per configuration; rate limits per UI user; LLM fallback counts in `/stats`; indexing ≈1.9× faster (55.7 → 29.5 s per 200 pages) |
| `b116692`, `25ef15c` | Web UI | Any basic-auth user name is safe in the `X-Actor` header; Reprocess asks for confirmation; the walkthrough recording only runs when asked for; shadcn CLI moved to dev dependencies |
| `e922bae` | Web UI | Uploads are batched by size as well as count, files over the limit are rejected one by one with the reason, and the stats strip shows LLM fallbacks |
| `c4f8d0e` | Docs | Use-your-own-model guide; time and upload limits stated accurately; deployment memory and OCR-worker guidance |
| `0d0a185` | Evaluation | Post-fix re-runs; the report tells two runs of one configuration apart |
| `5b17bda` | Types | `mypy app` clean; variable renames and explicit narrowing only, no behaviour change |
| `43cb623` | API | An empty or blank review comment no longer returns 500 (see End-to-end run); `docs/openapi.json` re-exported |
| `119ca45` | Web UI | A file uploaded while the documents table is still loading now appears in it (see End-to-end run) |
| `5afbd28`, `944891e`, `31d2252` | Gemini | The Gemini provider made one attempt only, so a single 429 sent the invoice to the rules result. Found while starting a Gemini evaluation: timeouts and 5xx are now retried by the SDK, and a 429 waits the delay the server asks for when that is a minute or less, or falls back at once when the quota is spent ("quota exhausted, retry after N h"). Checked live against a spent free-tier key: fallback in 0.5 s |
| `4b225a6` | Evaluation | A separate report and log per model (`docs/evaluation/`); the runner writes each run's log; report `--title` and `--notes` |

## Known issues

Found in review, judged not to affect the default setup (hybrid extraction with a local or cloud LLM), and left for
a follow-up:

- **A tax rate only the LLM found is dropped** when the subtotal is tax-inclusive.
- **Space-grouped numbers.** A line item printed as `Widget 3 125.50` can have its quantity and price read as one
  space-grouped amount (`3125.50`).
- **Invoices that print only a tax-point date** get no invoice date, since "Date of Supply"/tax-point dates are no
  longer taken as the invoice date.
- **`EXTRACTION_MODE=llm` (no rules) loses line items** when a truncated answer is retried without them; hybrid mode
  keeps the rules' items.
- **Retry provenance.** A record produced by the retry without line items is not marked as such.
- **The context window from llama.cpp `/props` is cached** for the life of the process. After restarting the LLM server
  with a different `-c`, restart the API and worker too, or set `LLM_CONTEXT_TOKENS`.
- **vLLM reasoning models.** There is no built-in switch to turn their thinking off; pass the server's own field through
  `LLM_EXTRA_BODY`.
- **`JOB_MAX_RUNTIME_S=0`** labels every reclaimed job `job_timeout`, even when its worker died (the label only; the job
  is retried or failed either way).
- **`/stats` reads every record's notes** to count flags and fallbacks, so it slows down as the database grows; it needs
  a SQL aggregate for large deployments.
- **Search candidate pool.** Candidates are counted in chunks, not documents, so one very large spreadsheet can crowd
  other documents out of a page of results (bounded: 1000 chunks per ranking).
- **Basic auth has no brute-force limit**, and the UI's content security policy allows inline scripts (Next.js); put
  an internet-facing deployment behind a reverse proxy or SSO, as [SECURITY.md](SECURITY.md) says.
- **Formatting.** `ruff format --check` would reformat 30 hand-formatted backend files (compact vocabulary lists, for
  example). The formatter is not enforced; `ruff check` is.
