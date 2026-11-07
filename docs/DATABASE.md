# Database schema

PostgreSQL 17 with the `vector` (pgvector) and `pg_trgm` extensions. One database holds document metadata, extracted
text, structured results, validation issues, processing history, the correction audit trail and the search index
(full-text + vectors), so a document's state changes in one transaction. Models: `backend/app/db/models.py`;
migrations: `backend/alembic/`; PostgreSQL-only DDL (extensions, generated tsvector, HNSW/GIN/trigram indexes):
`backend/app/db/pg_extras.py`. Tests and local development can run the same models on SQLite (search then uses a
Python fallback).

## Entity–relationship diagram

```mermaid
erDiagram
    documents ||--o{ document_pages : "has pages / sheets"
    documents ||--o{ invoices : "yields records"
    documents ||--o{ processing_jobs : "is processed by"
    documents ||--o{ processing_events : "history"
    documents ||--o{ validation_issues : "document-level issues"
    documents ||--o{ chunks : "search index"
    documents ||--o{ corrections : "audit trail"
    documents |o--o{ documents : "duplicate_of"
    invoices ||--o{ validation_issues : "record issues"
    invoices ||--o{ chunks : "summary chunk"
    invoices |o--o{ corrections : "corrected field"
    invoices ||--o{ cross_verifications : "left"
    invoices ||--o{ cross_verifications : "right"
    processing_jobs |o--o{ processing_events : "emitted"

    documents {
        uuid id PK
        varchar filename "display name, sanitised"
        varchar content_type "sniffed from bytes"
        varchar file_kind "pdf | image | xlsx | xls | csv"
        bigint size_bytes
        char64 sha256 "duplicate-file detection"
        varchar storage_key "UUID path, never user input"
        varchar status "queued | processing | completed | needs_review | failed"
        int page_count
        int ocr_page_count
        varchar error_code "corrupt_file, encrypted_file, ..."
        text error_message
        int processing_ms
        uuid duplicate_of_id FK
        varchar extraction_method "rules | llm | hybrid | spreadsheet"
        varchar llm_provider
        varchar llm_model
        varchar prompt_version
        jsonb warnings
        timestamptz created_at
        timestamptz updated_at
        timestamptz processed_at
    }
    document_pages {
        bigint id PK
        uuid document_id FK
        int page_number "1-based page / frame / sheet"
        varchar label "sheet name"
        varchar method "text | ocr | sheet | none"
        text text
        float ocr_confidence
        int char_count
    }
    invoices {
        uuid id PK
        uuid document_id FK
        int record_index
        varchar source_kind "invoice_document | ledger_row"
        int page_start
        int page_end
        varchar sheet_name
        int row_number
        varchar invoice_number
        varchar invoice_number_norm "upper alnum, duplicate/cross-check key"
        date invoice_date
        date due_date
        varchar supplier_name
        varchar supplier_name_norm
        varchar customer_name
        numeric subtotal "18,4"
        numeric discount
        numeric shipping
        numeric tax
        numeric tax_rate
        numeric total
        char3 currency "ISO 4217"
        varchar document_type
        jsonb line_items
        jsonb field_meta "per field: confidence, source, evidence, page, notes"
        varchar extraction_method
        float overall_confidence
        varchar review_status "auto_approved | needs_review | reviewed"
        timestamptz reviewed_at
        varchar reviewed_by
    }
    validation_issues {
        bigint id PK
        uuid document_id FK
        uuid invoice_id FK "null = document-level"
        varchar code "TOTAL_MISMATCH, MISSING_FIELD, ..."
        varchar severity "error | warning | info"
        varchar field
        text message
        jsonb details
    }
    corrections {
        bigint id PK
        uuid document_id FK
        uuid invoice_id FK "SET NULL on re-processing"
        varchar field
        jsonb old_value
        jsonb new_value
        text comment
        varchar corrected_by
        timestamptz created_at
    }
    processing_jobs {
        uuid id PK
        uuid document_id FK
        varchar status "queued | running | succeeded | failed"
        int attempt
        int max_attempts
        varchar stage "parsing | ocr | extracting | validating | indexing | done"
        int progress_current
        int progress_total
        text error
        varchar worker_id
        timestamptz heartbeat_at
    }
    processing_events {
        bigint id PK
        uuid document_id FK
        uuid job_id FK
        varchar stage
        varchar level "info | warning | error"
        text message
        jsonb data
        timestamptz created_at
    }
    chunks {
        bigint id PK
        uuid document_id FK
        uuid invoice_id FK
        int page_number
        int chunk_index
        varchar kind "page_text | invoice_summary | sheet"
        text text
        vector384 embedding "bge-small-en-v1.5"
        tsvector search_tsv "GENERATED from text (PostgreSQL)"
    }
    cross_verifications {
        uuid id PK
        uuid left_invoice_id FK
        uuid right_invoice_id FK
        varchar kind "pdf_pdf | pdf_excel | other"
        varchar status "match | mismatch | partial"
        jsonb fields "per-field left/right/status"
        jsonb line_items "counts and matches"
        varchar trigger "auto | manual"
    }
```

## Tables

| Table | Purpose |
|---|---|
| `documents` | One row per uploaded file: what it is (sniffed type, hash), where it is stored, its processing outcome and timings. |
| `document_pages` | Extracted text per PDF page, image frame or spreadsheet sheet, with the method used (digital text vs OCR) and OCR confidence — the raw material for review, search and re-extraction. |
| `invoices` | One structured record per invoice found (a long PDF can hold hundreds; a ledger sheet one per row). Typed columns serve filtering and validation; `field_meta` keeps per-field confidence, source (rules / LLM / both / spreadsheet / manual), verbatim evidence and notes. |
| `validation_issues` | Every rule violation with code, severity, field and numeric details. `invoice_id IS NULL` marks document-level issues (duplicate file, low OCR quality, LLM unavailable). |
| `corrections` | Append-only audit trail of manual corrections (who, when, old → new, why). Survives re-processing. |
| `processing_jobs` | Background job state: attempts, stage, page progress, heartbeat (stale-job recovery), error. |
| `processing_events` | Processing history: one row per stage transition or notable event. |
| `chunks` | Search index: page-text windows and invoice summaries with a 384-d embedding and a generated `tsvector`. |
| `cross_verifications` | Field-by-field comparison of two records sharing an invoice number (PDF↔PDF, PDF↔Excel ledger), automatic or user-requested. |

## Indexes and why

| Index | Serves |
|---|---|
| `documents (status, created_at)` | dashboard lists and status filters |
| `documents (sha256)` | duplicate-file detection at upload |
| `documents (file_kind)`, `(duplicate_of_id)`, `(created_at)` | list filters and duplicate lookups |
| `invoices (invoice_number_norm, supplier_name_norm)` | duplicate detection and cross-verification lookups |
| `invoices (invoice_date)`, `(total)`, `(currency)`, `(review_status)` | keyword/structured filters and the review queue |
| GIN trigram on `invoices.supplier_name`, `invoices.customer_name` | fast case-insensitive substring filters (`ILIKE '%acme%'`) |
| GIN on `chunks.search_tsv` | full-text keyword search (`websearch_to_tsquery`, `ts_rank_cd`) |
| HNSW (`vector_cosine_ops`) on `chunks.embedding` | approximate nearest-neighbour semantic search |
| FKs on every `document_id` / `invoice_id` | cascades and per-document reads |

## Integrity rules

- Deleting a document cascades to its pages, records, issues, jobs, events, chunks and comparisons.
- Re-processing replaces a document's pages, records, issues, chunks and comparisons in one transaction; corrections
  stay (their `invoice_id` becomes NULL) so the audit trail is never lost.
- Money is `NUMERIC(18,4)` (never floating point); currency is a 3-letter ISO code.
- `(document_id, page_number)` and `(left_invoice_id, right_invoice_id)` are unique.
