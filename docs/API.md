# REST API

Base URL `http://localhost:8000/api/v1`. Interactive documentation: `/docs` (Swagger UI) and `/redoc`; the schema is
`/openapi.json` and is exported to [openapi.json](openapi.json) (21 paths). All bodies are JSON except uploads
(multipart) and the original-file download. The examples assume:

```bash
export API=http://localhost:8000/api/v1
export API_KEY=<one of the keys in API_KEYS>
```

## Conventions

**Authentication.** When `API_KEYS` is set, every route except `GET /health` requires `X-API-Key: <key>` or
`Authorization: Bearer <key>` (`X-API-Key` wins if both are sent). A missing or wrong key returns `401 unauthorized`
with `WWW-Authenticate: Bearer`. When `API_KEYS` is empty (development only) authentication is off;
`GET /config` reports it as `auth_required`.

**Audit identity.** Corrections and approvals record who acted: the `X-Actor` header (trimmed, control characters
removed, at most 100 characters; the web proxy sets it), else `api-key:<first 8 hex of sha256(key)>`, else `anonymous`.

**Types.** Identifiers are UUIDs. Money is a plain decimal string (`"1234.50"`), never a float. Dates are
`YYYY-MM-DD`, timestamps ISO 8601 UTC. Absent values are `null`.

**Request id.** Every response carries `X-Request-ID` (an incoming value of up to 128 safe characters is kept, else a
UUID is generated). Quote it when reporting a problem; it appears in the server log.

**Responses are never cached**: `/api/` responses carry `Cache-Control: no-store`.

### Error envelope

Every error has the same shape:

```json
{"error": {"code": "validation_error", "message": "The request is invalid", "details": {"fields": {"limit": "Input should be less than or equal to 100"}}}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | Malformed request |
| 401 | `unauthorized` | Missing/invalid API key |
| 403 | `forbidden` | Not allowed |
| 404 | `not_found` | No such resource (or unknown route) |
| 405 | `method_not_allowed` | Wrong method for the route |
| 409 | `already_processing` | Reprocess/delete while the document is queued or being processed |
| 413 | `file_too_large` | Whole request body over the request cap (see Upload limits); a single oversized file is rejected per item instead |
| 415 | `unsupported_file_type` | Upload is not `multipart/form-data` |
| 422 | `validation_error` | Invalid parameters or body; `details.fields` maps field name to message (input values are never echoed) |
| 429 | `rate_limited` | Over the rate limit; `details` has `limit` and `retry_after_s` |
| 500 | `internal_error` | Unexpected failure; the message is generic, `details.request_id` identifies the log entry |
| 503 | `service_unavailable` | Dependency unavailable |

Any other status maps to `http_error`.

### Pagination

List endpoints take `limit` (1-100, default 20) and `offset` (default 0) and return `{"items": [...], "total": N,
"limit": L, "offset": O}` (pages, jobs and search omit some of these keys, see below). `GET .../events`,
`GET .../corrections` and `GET /cross-verifications` allow `limit` up to 1000 (default 500). Sorting uses
`sort=<field>` or `sort=-<field>` (descending).

### Rate limits

Fixed windows per caller (the API key when valid, combined with the `X-Actor` header when the web proxy sends one, so
each UI user gets their own budget; else the client IP, so failed logins are limited per IP):
uploads `RATE_LIMIT_UPLOAD` (`60/minute`), search `RATE_LIMIT_SEARCH` (`300/minute`), every other route except
`/health` `RATE_LIMIT_DEFAULT` (`1200/minute`). Over the limit: `429` with `Retry-After`, `X-RateLimit-Limit`,
`X-RateLimit-Remaining: 0`, `X-RateLimit-Reset`. Counters live in Redis when `REDIS_URL` is set.

### Upload limits

Per file `MAX_UPLOAD_MB` (200); files per request `MAX_FILES_PER_REQUEST` (50). The limit per file is enforced while the
body streams: at most `MAX_UPLOAD_MB` of a part is spooled, and an oversized file comes back as a per-item
`rejected` / `file_too_large` in the `202` response while the other files of the request are still processed. The whole
upload request body is capped at `min(MAX_FILES_PER_REQUEST, 5) x (MAX_UPLOAD_MB + 16 KB)` (about 1000 MiB with the
defaults), reported by `GET /config` as `max_request_mb`; a body over it (declared or streamed) gets `413`
`file_too_large` for the whole request, so send larger batches as several requests. Every other request body is capped
at 1 MiB. Accepted kinds, decided from the file's bytes (not its extension): PDF, PNG/JPEG/TIFF, XLSX, XLS, CSV.
`GET /config` lists the accepted extensions.

---

## Health and configuration

### `GET /health` - liveness (no auth, no rate limit)

```bash
curl $API/health          # {"status":"ok"}
```

### `GET /ready` - readiness

`200 {"status":"ok","database":"ok","queue":"ok","llm":"configured"}`; `503` with the same body shape and
`"status":"unavailable"` when the database or the queue is unreachable. `llm` is `none` (rules only), `configured` or
`misconfigured` (a provider was named without its key/URL); the model server is not called.

### `GET /config`

```json
{"max_upload_mb":200,"max_files_per_request":50,"max_request_mb":1000,"allowed_extensions":[".csv",".jfif",".jpe",".jpeg",".jpg",".pdf",".png",".tif",".tiff",".tsv",".txt",".xls",".xlsm",".xlsx"],"llm_provider":"anthropic","llm_model":"claude-opus-5-5","auth_required":true}
```

`max_request_mb` is the cap of one upload request body (all files together); clients can use it to batch uploads by
bytes. Extensions are hints only: the file kind is decided from the bytes (`.tsv` and `.txt` are accepted for delimited text).

---

## Documents

A document is one uploaded file. `status` is `queued`, `processing`, `completed`, `needs_review` or `failed`.

### `POST /documents` - upload (202)

Multipart field `files`, one or many parts.

```bash
curl -H "X-API-Key: $API_KEY" -F "files=@invoice.pdf" -F "files=@scan.jpg" -F "files=@ledger.xlsx" $API/documents
```

Response `202`, one item per file in request order. Rejected files create nothing:

```json
{"items": [
  {"filename": "invoice.pdf", "status": "queued", "document_id": "6f0c...", "job_id": "9a1d...", "duplicate_of_id": null, "error": null},
  {"filename": "scan.jpg", "status": "queued", "document_id": "...", "job_id": "...", "duplicate_of_id": "6f0c...", "error": null},
  {"filename": "notes.txt", "status": "rejected", "document_id": null, "job_id": null, "duplicate_of_id": null,
   "error": {"code": "unsupported_file_type", "message": "..."}}
]}
```

`duplicate_of_id` is set when the same bytes (sha256) were uploaded before; the file is still processed and gets a
`DUPLICATE_FILE` warning. Per-file error codes: `unsupported_file_type`, `empty_file`, `file_too_large`,
`internal_error` (could not store/save; retry); `file_too_large` is reported per item for a file over `MAX_UPLOAD_MB`.
Whole-request errors: `413` (body over the request cap `max_request_mb`), `415` (not
multipart), `422` (no `files` part, more than `MAX_FILES_PER_REQUEST` files, malformed multipart).

### `GET /documents` - list

Query: `status` (repeatable: `status=failed&status=needs_review`), `file_kind` (`pdf|image|xlsx|xls|csv`), `needs_review`
(bool), `q` (substring of filename, invoice number or supplier; max 200), `limit`, `offset`, `sort` (`created_at`,
`updated_at`, `processed_at`, `filename`, `status`, `size_bytes`, `processing_ms`, each optionally prefixed `-`;
default `-created_at`).

```bash
curl -H "X-API-Key: $API_KEY" "$API/documents?status=needs_review&limit=10&sort=-created_at"
```

```json
{"items": [{"id": "...", "filename": "inv-001.pdf", "file_kind": "pdf", "content_type": "application/pdf", "size_bytes": 48211,
  "sha256": "...", "status": "needs_review", "page_count": 2, "ocr_page_count": 0, "error_code": null, "error_message": null,
  "processing_ms": 1840, "duplicate_of_id": null, "created_at": "...", "updated_at": "...", "processed_at": "...",
  "invoice_count": 1, "issue_counts": {"error": 1, "warning": 0, "info": 0}, "job": {"id": "...", "status": "succeeded", "stage": "done", "...": "..."}}],
 "total": 1, "limit": 10, "offset": 0}
```

### `GET /documents/{id}` - detail

The list item plus `invoices` (full records, below), `document_issues` (document-level issues) and `extraction`
(`method`, `llm_provider`, `llm_model`, `prompt_version`). `404 not_found` for an unknown id.

### `GET /documents/{id}/status` - polling

```json
{"id": "...", "status": "processing", "updated_at": "...",
 "job": {"id": "...", "document_id": "...", "status": "running", "stage": "extracting", "progress_current": 120, "progress_total": 1000,
         "attempt": 1, "max_attempts": 3, "error": null, "created_at": "...", "started_at": "...", "finished_at": null}}
```

Job `status`: `queued`, `running`, `succeeded`, `failed`. `stage`: `parsing`, `ocr`, `extracting`, `validating`,
`indexing`, `done`. A failed document has `error_code` / `error_message` on the document (for example `corrupt_file`,
`encrypted_file`, `file_too_large`, `empty_file`, `unsupported_file_type`, `ocr_unavailable`).

### `GET /documents/{id}/results` - extracted records

`{"document_id": "...", "status": "...", "invoices": [InvoiceOut, ...]}`. Record shape:

```json
{"id": "...", "document_id": "...", "filename": "inv-001.pdf", "record_index": 0, "source_kind": "invoice_document",
 "page_start": 1, "page_end": 2, "sheet_name": null, "row_number": null,
 "invoice_number": "INV-2024-001", "invoice_date": "2024-03-05", "due_date": "2024-04-04",
 "supplier_name": "Acme GmbH", "customer_name": "Beta Ltd",
 "subtotal": "100.00", "discount": null, "shipping": null, "tax": "19.00", "tax_rate": "19", "total": "119.00", "currency": "EUR",
 "document_type": "invoice",
 "line_items": [{"description": "Widget", "quantity": "2", "unit_price": "50.00", "amount": "100.00"}],
 "field_meta": {"total": {"confidence": 0.97, "source": "rules+llm", "evidence": "119,00 €", "page": 2, "notes": []}},
 "extraction_method": "hybrid", "overall_confidence": 0.93, "review_status": "auto_approved",
 "issues": [{"id": 1, "invoice_id": "...", "document_id": "...", "code": "TOTAL_MISMATCH", "severity": "error", "field": "total", "message": "...", "details": {}}],
 "created_at": "...", "updated_at": "...", "reviewed_at": null, "reviewed_by": null}
```

Values the document does not contain are `null`, never guessed. `field_meta[field].source` is `rules`, `llm`,
`rules+llm`, `spreadsheet` or `manual`; `evidence` is the text the value was read from. `review_status` is
`auto_approved`, `needs_review` (any error or warning) or `reviewed` (a human approved).

Issue codes (severity): errors `MISSING_FIELD` (required fields), `INVALID_AMOUNT`, `INVALID_DATE`, `TOTAL_MISMATCH`,
`LINE_ITEMS_SUM_MISMATCH`, `LINE_ITEM_MATH`, `DUPLICATE_INVOICE`, `CROSS_VERIFICATION_MISMATCH`; warnings
`MISSING_FIELD` (recommended fields), `NEGATIVE_AMOUNT`, `FUTURE_DATE`, `TAX_RATE_MISMATCH`, `TAX_RATE_UNUSUAL`,
`LOW_CONFIDENCE`, `EXTRACTION_DISAGREEMENT`, `DUPLICATE_FILE`, `OCR_LOW_QUALITY`, `POSSIBLE_DUPLICATE`; info
`AMBIGUOUS_DATE`, `CURRENCY_INFERRED`, `UNGROUNDED_VALUE_REJECTED`, `LLM_UNAVAILABLE`.

### `GET /documents/{id}/pages` - extracted text per page or sheet

Query `limit`, `offset`. `{"items": [{"page_number": 1, "label": null, "method": "text", "text": "...", "ocr_confidence": null}], "total": 2}`.
`method` is `text`, `ocr`, `sheet` or `none`; `label` is the sheet name for spreadsheets.

### `GET /documents/{id}/events` - processing history

Oldest first. `{"items": [{"id": 1, "job_id": "...", "stage": "parsing", "level": "info", "message": "...", "data": {}, "created_at": "..."}]}`.

### `GET /documents/{id}/file` - original file

Returns the stored bytes with `Content-Type` as detected and `Content-Disposition: attachment` (sanitised name).
`404` if the file is missing.

```bash
curl -H "X-API-Key: $API_KEY" -OJ $API/documents/$DOC_ID/file
```

### `POST /documents/{id}/reprocess` - process again (202)

Creates a new job; results are replaced atomically, manual corrections stay in the audit trail.
`{"document_id": "...", "job_id": "...", "status": "queued"}`. `409 already_processing` when a job is queued or running.

### `DELETE /documents/{id}` - delete (204)

Removes the document, pages, records, issues, jobs, events, chunks, comparisons and the stored file. `409
already_processing` while a worker is actively running a job for it (a job whose worker stopped sending heartbeats does
not block deletion).

---

## Jobs

### `GET /jobs` - list processing jobs, newest first

Query `status` (`queued|running|succeeded|failed`), `limit`, `offset`. `{"items": [JobOut...], "total": N}`.

### `POST /jobs/reprocess-failed` - re-queue every failed document (202)

```bash
curl -X POST -H "X-API-Key: $API_KEY" $API/jobs/reprocess-failed     # {"requeued": 3, "job_ids": ["...", "...", "..."]}
```

Documents that already have a queued or running job are skipped.

---

## Invoices (extracted records)

### `GET /invoices` - list and filter

All filters optional, combined with AND; text filters are case-insensitive substrings. Query: `q` (number, supplier,
customer or filename), `supplier`, `customer`, `invoice_number` (also matches the normalised number, so `inv2024/1`
finds `INV-2024-1`), `currency` (3 letters), `date_from`, `date_to` (invoice date), `total_min`, `total_max`,
`review_status` (`auto_approved|needs_review|reviewed`), `has_errors` (bool: has an error-severity issue),
`document_id`, `limit`, `offset`, `sort` (`created_at`, `updated_at`, `invoice_date`, `due_date`, `total`,
`invoice_number`, `supplier_name`, `customer_name`, `overall_confidence`, optionally `-`; default `-created_at`).

```bash
curl -H "X-API-Key: $API_KEY" "$API/invoices?review_status=needs_review&supplier=acme&total_min=100&sort=-total"
```

### `GET /invoices/{id}` - one record (shape above). `404 not_found`.

### `PATCH /invoices/{id}` - correct fields (audited)

```bash
curl -X PATCH -H "X-API-Key: $API_KEY" -H "X-Actor: alice" -H "Content-Type: application/json" $API/invoices/$INV_ID -d '{
  "fields": {"total": "119.00", "invoice_date": "2024-03-05", "currency": "EUR"},
  "line_items": [{"description": "Widget", "quantity": "2", "unit_price": "50.00", "amount": "100.00"}],
  "comment": "total read wrongly by OCR"
}'
```

`fields` may contain `invoice_number`, `invoice_date`, `due_date`, `supplier_name`, `customer_name`, `subtotal`,
`discount`, `shipping`, `tax`, `tax_rate`, `total`, `currency`, `document_type`; a field present is set (`null`
clears it), absent fields are kept. `line_items` (optional, at most 1000) replaces the whole list. `comment` (at most
2000 characters) is stored with each change. The body is strict: amounts are decimal strings with at most 4 decimals,
dates `YYYY-MM-DD`, currency an upper-case ISO code, text without control characters (invoice number 100,
names 255, line description 1000 characters), unknown keys are refused. Each changed field becomes `source: manual` with
confidence 1.0, the record is validated again and the response is the updated record with fresh `issues` and
`review_status`. A 422 names the offending field (`details.fields`).

### `POST /invoices/{id}/approve` - mark as reviewed

Optional body `{"comment": "..."}`. Sets `review_status: reviewed`, `reviewed_by`, `reviewed_at` and updates the
document status; approving a reviewed record again changes nothing.

### `GET /invoices/{id}/corrections` - history

`{"items": [{"id": 1, "field": "total", "old_value": "118.00", "new_value": "119.00", "comment": "...", "corrected_by": "alice", "created_at": "..."}]}`,
oldest first; `limit` (up to 1000, default 500) and `offset`.

---

## Search

### `GET /search`

Query: `q` (up to 1000 characters; words, `"phrases"`, `-exclusion`, `or`), `mode` (`keyword`, `semantic`, `hybrid`;
default `hybrid`), the structured filters `supplier`, `customer`, `invoice_number`, `currency`, `date_from`, `date_to`,
`total_min`, `total_max`, `review_status`, `file_kind`, `document_id`, and `limit`, `offset`.

```bash
curl -H "X-API-Key: $API_KEY" "$API/search?q=consulting+services&mode=hybrid&currency=EUR&limit=5"
```

```json
{"items": [{"document_id": "...", "invoice_id": "...", "page_number": 1, "score": 0.0328, "snippet": "... «consulting» «services» ...",
            "match_type": "hybrid", "filename": "inv-001.pdf", "invoice_number": "INV-2024-001", "supplier_name": "Acme GmbH",
            "customer_name": "Beta Ltd", "invoice_date": "2024-03-05", "total": "119.00", "currency": "EUR", "review_status": "needs_review"}],
 "total": 1, "took_ms": 12, "mode": "hybrid"}
```

One hit per document or record, best first. `match_type` is `keyword`, `semantic`, `hybrid` (found by both) or
`filter` (a blank `q` with filters lists matching records, newest first, score 0). Scores are comparable only within one
result list and are not a relevance percentage. Matches in `snippet` are marked with « and ».

---

## Cross-verification

### `POST /cross-verify` - compare two records

Either two invoice ids or two document ids (documents are paired by normalised invoice number; a single record on each
side is compared directly):

```bash
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" $API/cross-verify \
  -d '{"left_document_id": "...", "right_document_id": "..."}'
# or {"left_invoice_id": "...", "right_invoice_id": "..."}
```

```json
{"id": "...", "left_invoice_id": "...", "right_invoice_id": "...",
 "left": {"document_id": "...", "filename": "inv.pdf", "invoice_number": "INV-1"},
 "right": {"document_id": "...", "filename": "ledger.xlsx", "invoice_number": "INV-1"},
 "kind": "pdf_excel", "status": "mismatch",
 "fields": [{"field": "total", "left": "119.00", "right": "191.00", "status": "mismatch", "note": null}],
 "line_items": {"left_count": 3, "right_count": 0, "matched": 0, "status": "missing"},
 "trigger": "manual", "created_at": "..."}
```

`kind`: `pdf_pdf`, `pdf_excel`, `other`. `status`: `match`, `mismatch`, `partial`. A pair has one stored comparison;
asking again updates it. `404` for an unknown id; `422` for the same id twice, not exactly one pair of ids, or
documents that share no invoice number and hold several records.

### `GET /cross-verifications` - stored comparisons (automatic and manual)

Query `invoice_id`, `document_id` (either side), `status` (`match|mismatch|partial`), `limit` (up to 1000), `offset`.
`{"items": [CrossVerificationOut...]}`, newest first. The pipeline also stores automatic comparisons
(`"trigger": "auto"`) for records sharing an invoice number across documents.

---

## Statistics

### `GET /stats`

```json
{"documents": {"total": 12, "by_status": {"queued": 0, "processing": 0, "completed": 7, "needs_review": 4, "failed": 1}},
 "invoices": {"total": 15, "by_review_status": {"auto_approved": 9, "needs_review": 5, "reviewed": 1}},
 "issues": {"by_code": {"TOTAL_MISMATCH": 2}, "by_severity": {"error": 2, "warning": 5, "info": 3}},
 "processing": {"avg_ms": 1840.5, "p95_ms": 5200, "pages_total": 87},
 "llm": {"provider": "anthropic", "model": "claude-opus-5-5", "fallback_documents": 0, "fallback_records": 0}}
```

`llm.fallback_documents` counts documents for which the LLM was unavailable, failed or refused (they carry an
`LLM_UNAVAILABLE` note, including hybrid mode with no provider configured) and `llm.fallback_records` the records of
those documents that used the rules result. The `LLM_UNAVAILABLE` note's `details` carry `reasons` and `records`.

The numbers above only illustrate the shape. `processing.avg_ms` and `p95_ms` are `null` until a document has been
processed.
