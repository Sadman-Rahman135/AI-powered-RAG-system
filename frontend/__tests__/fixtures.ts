/** Builders for contract-shaped test data. Every value mirrors the API's JSON exactly (amounts as strings). */
import type { Document, Invoice, Issue, Job, UploadItem } from "@/lib/types";

let issueId = 1;

export function makeIssue(overrides: Partial<Issue> = {}): Issue {
  return {
    id: issueId++,
    invoice_id: "inv-1",
    document_id: "doc-1",
    code: "TOTAL_MISMATCH",
    severity: "error",
    field: "total",
    message: "Subtotal + tax does not equal the total.",
    details: {},
    ...overrides,
  };
}

export function makeInvoice(overrides: Partial<Invoice> = {}): Invoice {
  return {
    id: "inv-1",
    document_id: "doc-1",
    filename: "acme-0042.pdf",
    record_index: 0,
    source_kind: "invoice_document",
    page_start: 1,
    page_end: 2,
    sheet_name: null,
    row_number: null,
    invoice_number: "INV-0042",
    invoice_date: "2024-03-05",
    due_date: "2024-04-04",
    supplier_name: "Acme Freight GmbH",
    customer_name: "Northwind Traders",
    subtotal: "100.00",
    discount: null,
    shipping: null,
    tax: "10.00",
    tax_rate: "10",
    total: "110.00",
    currency: "EUR",
    document_type: "invoice",
    line_items: [
      { description: "Pallet transport", quantity: "2", unit_price: "40.00", amount: "80.00" },
      { description: "Handling", quantity: "1", unit_price: "20.00", amount: "20.00" },
    ],
    field_meta: {
      invoice_number: { confidence: 0.97, source: "rules+llm", evidence: "Invoice No: INV-0042", page: 1, notes: [] },
      total: { confidence: 0.934, source: "rules", evidence: "Total due  110.00", page: 2, notes: [] },
      supplier_name: { confidence: 0.55, source: "llm", evidence: "Acme Freight GmbH", page: 1, notes: [] },
    },
    extraction_method: "hybrid",
    overall_confidence: 0.88,
    review_status: "needs_review",
    issues: [],
    created_at: "2026-11-04T10:00:00Z",
    updated_at: "2026-11-04T10:00:00Z",
    reviewed_at: null,
    reviewed_by: null,
    ...overrides,
  };
}

export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    document_id: "doc-1",
    status: "running",
    stage: "ocr",
    progress_current: 3,
    progress_total: 12,
    attempt: 1,
    max_attempts: 3,
    error: null,
    created_at: "2026-11-04T10:00:00Z",
    started_at: "2026-11-04T10:00:01Z",
    finished_at: null,
    ...overrides,
  };
}

export function makeDocument(overrides: Partial<Document> = {}): Document {
  return {
    id: "doc-1",
    filename: "acme-0042.pdf",
    file_kind: "pdf",
    content_type: "application/pdf",
    size_bytes: 245_760,
    sha256: "a".repeat(64),
    status: "completed",
    page_count: 2,
    ocr_page_count: 0,
    error_code: null,
    error_message: null,
    processing_ms: 1800,
    duplicate_of_id: null,
    created_at: "2026-11-04T10:00:00Z",
    updated_at: "2026-11-04T10:00:02Z",
    processed_at: "2026-11-04T10:00:02Z",
    invoice_count: 1,
    issue_counts: { error: 0, warning: 0, info: 0 },
    job: null,
    ...overrides,
  };
}

export function makeUploadItem(overrides: Partial<UploadItem> = {}): UploadItem {
  return {
    filename: "file.pdf",
    status: "queued",
    document_id: "doc-new",
    job_id: "job-new",
    duplicate_of_id: null,
    error: null,
    ...overrides,
  };
}
