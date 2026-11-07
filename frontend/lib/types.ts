/**
 * Types mirroring the REST API contract (docs/API.md).
 *
 * Conventions from the backend: money and other decimals are strings ("1234.50") so they never pass through
 * floating point; dates are ISO 8601 strings ("2024-03-05"); timestamps are ISO 8601 date-times; a value the
 * document does not contain is `null`, never a guess.
 */

/** ISO 8601 date ("2024-03-05"). */
export type IsoDate = string;
/** ISO 8601 date-time ("2026-11-04T21:07:00Z"). */
export type IsoDateTime = string;
/** Decimal serialized as a plain-notation string ("110.00"). */
export type Decimal = string;
export type Uuid = string;

export type DocumentStatus = "queued" | "processing" | "completed" | "needs_review" | "failed";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type JobStage = "parsing" | "ocr" | "extracting" | "validating" | "indexing";
export type FileKind = "pdf" | "image" | "xlsx" | "xls" | "csv";
export type Severity = "error" | "warning" | "info";
export type ReviewStatus = "auto_approved" | "needs_review" | "reviewed";
export type ExtractionSource = "rules" | "llm" | "rules+llm" | "spreadsheet" | "manual";
export type InvoiceDocumentType = "invoice" | "receipt" | "credit_note" | "unknown";
export type SourceKind = "invoice_document" | "ledger_row";
export type SearchMode = "hybrid" | "keyword" | "semantic";
/** How a hit was found; "filter" = structured filtering with a blank query (no relevance ranking). */
export type SearchMatchType = "keyword" | "semantic" | "hybrid" | "filter";
export type PageMethod = "text" | "ocr" | "sheet" | "none";
export type EventLevel = "info" | "warning" | "error";
export type CrossVerificationKind = "pdf_pdf" | "pdf_excel" | "other";
export type CrossVerificationStatus = "match" | "mismatch" | "partial";
/** Per-field comparison outcome. The contract leaves the exact values open; the UI also tolerates others. */
export type FieldComparisonStatus = "match" | "mismatch" | "missing";

/** Spec §5.3 — validation issue codes (exact API values). */
export type IssueCode =
  | "MISSING_FIELD"
  | "INVALID_AMOUNT"
  | "INVALID_DATE"
  | "TOTAL_MISMATCH"
  | "LINE_ITEMS_SUM_MISMATCH"
  | "LINE_ITEM_MATH"
  | "DUPLICATE_INVOICE"
  | "CROSS_VERIFICATION_MISMATCH"
  | "NEGATIVE_AMOUNT"
  | "FUTURE_DATE"
  | "TAX_RATE_MISMATCH"
  | "TAX_RATE_UNUSUAL"
  | "LOW_CONFIDENCE"
  | "EXTRACTION_DISAGREEMENT"
  | "DUPLICATE_FILE"
  | "OCR_LOW_QUALITY"
  | "POSSIBLE_DUPLICATE"
  | "AMBIGUOUS_DATE"
  | "CURRENCY_INFERRED"
  | "UNGROUNDED_VALUE_REJECTED"
  | "LLM_UNAVAILABLE";

/** Standard error envelope: `{"error": {"code", "message", "details"}}`. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export interface Paged<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Job {
  id: Uuid;
  document_id: Uuid;
  status: JobStatus;
  stage: JobStage | null;
  progress_current: number;
  progress_total: number;
  attempt: number;
  max_attempts: number;
  error: string | null;
  created_at: IsoDateTime;
  started_at: IsoDateTime | null;
  finished_at: IsoDateTime | null;
}

export interface IssueCounts {
  error: number;
  warning: number;
  info: number;
}

export interface Document {
  id: Uuid;
  filename: string;
  file_kind: FileKind;
  content_type: string;
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  page_count: number | null;
  ocr_page_count: number | null;
  error_code: string | null;
  error_message: string | null;
  processing_ms: number | null;
  duplicate_of_id: Uuid | null;
  created_at: IsoDateTime;
  updated_at: IsoDateTime;
  processed_at: IsoDateTime | null;
  invoice_count: number;
  issue_counts: IssueCounts;
  job: Job | null;
}

export interface Issue {
  id: number;
  invoice_id: Uuid | null;
  document_id: Uuid;
  code: IssueCode;
  severity: Severity;
  field: string | null;
  message: string;
  details: Record<string, unknown> | null;
}

export interface LineItem {
  description: string | null;
  quantity: Decimal | null;
  unit_price: Decimal | null;
  amount: Decimal | null;
}

export interface FieldMeta {
  confidence: number;
  source: ExtractionSource;
  evidence: string | null;
  page: number | null;
  notes: string[];
}

export interface Invoice {
  id: Uuid;
  document_id: Uuid;
  filename: string;
  record_index: number;
  source_kind: SourceKind;
  page_start: number | null;
  page_end: number | null;
  sheet_name: string | null;
  row_number: number | null;
  invoice_number: string | null;
  invoice_date: IsoDate | null;
  due_date: IsoDate | null;
  supplier_name: string | null;
  customer_name: string | null;
  subtotal: Decimal | null;
  discount: Decimal | null;
  shipping: Decimal | null;
  tax: Decimal | null;
  tax_rate: Decimal | null;
  total: Decimal | null;
  currency: string | null;
  document_type: InvoiceDocumentType;
  line_items: LineItem[];
  field_meta: Record<string, FieldMeta>;
  extraction_method: string;
  overall_confidence: number;
  review_status: ReviewStatus;
  issues: Issue[];
  created_at: IsoDateTime;
  updated_at: IsoDateTime;
  reviewed_at: IsoDateTime | null;
  reviewed_by: string | null;
}

/** Scalar invoice fields a reviewer can correct (everything except line items). */
export type EditableField =
  | "invoice_number"
  | "invoice_date"
  | "due_date"
  | "supplier_name"
  | "customer_name"
  | "subtotal"
  | "discount"
  | "shipping"
  | "tax"
  | "tax_rate"
  | "total"
  | "currency"
  | "document_type";

export interface ExtractionInfo {
  method: string | null;
  llm_provider: string | null;
  llm_model: string | null;
  prompt_version: string | null;
}

export interface DocumentDetail extends Document {
  invoices: Invoice[];
  document_issues: Issue[];
  extraction: ExtractionInfo;
}

export interface DocumentStatusResponse {
  id: Uuid;
  status: DocumentStatus;
  job: Job | null;
  updated_at: IsoDateTime;
}

export interface DocumentResults {
  document_id: Uuid;
  status: DocumentStatus;
  invoices: Invoice[];
}

export interface DocumentPage {
  page_number: number;
  label: string | null;
  method: PageMethod;
  text: string;
  ocr_confidence: number | null;
}

export interface PagesResponse {
  items: DocumentPage[];
  total: number;
}

export interface ProcessingEvent {
  id: number;
  job_id: Uuid | null;
  stage: string;
  level: EventLevel;
  message: string;
  data: Record<string, unknown> | null;
  created_at: IsoDateTime;
}

export interface ReprocessResponse {
  document_id: Uuid;
  job_id: Uuid;
  status: "queued";
}

export interface UploadItem {
  filename: string;
  status: "queued" | "rejected";
  document_id: Uuid | null;
  job_id: Uuid | null;
  duplicate_of_id: Uuid | null;
  error: { code: string; message: string } | null;
}

export interface UploadResponse {
  items: UploadItem[];
}

export interface InvoicePatch {
  fields: Partial<Record<EditableField, string | null>>;
  line_items: LineItem[] | null;
  comment: string | null;
}

export interface Correction {
  id: number;
  field: string;
  old_value: unknown;
  new_value: unknown;
  comment: string | null;
  corrected_by: string;
  created_at: IsoDateTime;
}

export interface SearchHit {
  document_id: Uuid;
  invoice_id: Uuid | null;
  page_number: number | null;
  /** Ranking score; not meaningful for `match_type: "filter"`. */
  score: number;
  /** Highlighted excerpt; matches are wrapped in « and ». */
  snippet: string;
  match_type: SearchMatchType;
  filename: string;
  invoice_number: string | null;
  supplier_name: string | null;
  customer_name: string | null;
  invoice_date: IsoDate | null;
  total: Decimal | null;
  currency: string | null;
  review_status: ReviewStatus | null;
}

export interface SearchResponse {
  items: SearchHit[];
  /** For semantic and hybrid searches this is the number of candidates ranked, not of relevant hits. */
  total: number;
  took_ms: number;
  mode: SearchMode;
}

export interface CrossVerificationSide {
  document_id: Uuid;
  filename: string;
  invoice_number: string | null;
}

export interface FieldComparison {
  field: string;
  left: unknown;
  right: unknown;
  status: FieldComparisonStatus | string;
  note: string | null;
}

export interface LineItemComparison {
  left_count: number;
  right_count: number;
  matched: number;
  status: CrossVerificationStatus | string;
}

export interface CrossVerification {
  id: Uuid;
  left_invoice_id: Uuid;
  right_invoice_id: Uuid;
  left: CrossVerificationSide;
  right: CrossVerificationSide;
  kind: CrossVerificationKind | string;
  status: CrossVerificationStatus | string;
  fields: FieldComparison[];
  line_items: LineItemComparison;
  trigger: "auto" | "manual" | string;
  created_at: IsoDateTime;
}

export type CrossVerifyRequest =
  | { left_invoice_id: Uuid; right_invoice_id: Uuid }
  | { left_document_id: Uuid; right_document_id: Uuid };

export interface JobsResponse {
  items: Job[];
  total: number;
}

export interface ReprocessFailedResponse {
  requeued: number;
  job_ids: Uuid[];
}

export interface Stats {
  documents: { total: number; by_status: Partial<Record<DocumentStatus, number>> };
  invoices: { total: number; by_review_status: Partial<Record<ReviewStatus, number>> };
  issues: {
    by_code: Partial<Record<IssueCode, number>>;
    by_severity: Partial<Record<Severity, number>>;
  };
  processing: { avg_ms: number | null; p95_ms: number | null; pages_total: number };
  llm: {
    provider: string | null;
    model: string | null;
    /** Documents / records for which the LLM failed or was unavailable, so the rules result was used. */
    fallback_documents: number;
    fallback_records: number;
  };
}

export interface AppConfig {
  max_upload_mb: number;
  max_files_per_request: number;
  /** Cap of one upload request body (all its files together), in MiB. */
  max_request_mb: number;
  allowed_extensions: string[];
  llm_provider: string | null;
  llm_model: string | null;
  auth_required: boolean;
}
