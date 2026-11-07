/**
 * Presentation helpers: money, numbers, dates, sizes, durations, confidence buckets and the plain-language labels
 * used for API enum values.
 *
 * Amounts arrive as decimal strings and are formatted from the string itself (Intl.NumberFormat accepts exact
 * decimal strings), so a value is never rounded through floating point. A missing value renders as an em dash,
 * never as zero.
 */
import type {
  CrossVerificationKind,
  DocumentStatus,
  ExtractionSource,
  FileKind,
  Invoice,
  InvoiceDocumentType,
  JobStage,
  PageMethod,
  ReviewStatus,
  SearchMatchType,
  SearchMode,
  Severity,
} from "@/lib/types";

export const EMPTY = "—";

const LOCALE = "en-US";
const DECIMAL_RE = /^[+-]?\d+(\.\d+)?$/;
const CURRENCY_RE = /^[A-Z]{3}$/;
const MAX_FRACTION_DIGITS = 20;

const numberFormatCache = new Map<string, Intl.NumberFormat>();

function numberFormat(key: string, options: Intl.NumberFormatOptions): Intl.NumberFormat {
  let format = numberFormatCache.get(key);
  if (!format) {
    format = new Intl.NumberFormat(LOCALE, options);
    numberFormatCache.set(key, format);
  }
  return format;
}

/** Number of fraction digits after dropping trailing zeros ("110.0000" -> 0, "0.1230" -> 3). */
function significantScale(value: string): number {
  const dot = value.indexOf(".");
  if (dot === -1) return 0;
  return value.slice(dot + 1).replace(/0+$/, "").length;
}

function formatDecimalString(value: string, minFraction: number, maxFraction: number): string {
  const max = Math.min(Math.max(maxFraction, minFraction), MAX_FRACTION_DIGITS);
  const format = numberFormat(`plain|${minFraction}|${max}`, {
    minimumFractionDigits: minFraction,
    maximumFractionDigits: max,
  });
  return format.format(value as Intl.StringNumericLiteral);
}

function currencyDigits(currency: string): number {
  return numberFormat(`digits|${currency}`, { style: "currency", currency }).resolvedOptions().minimumFractionDigits ?? 2;
}

/** `"1234.5", "EUR"` -> `"€1,234.50"`; `null` -> `"—"`; unparseable input is shown verbatim. */
export function formatMoney(amount: string | null | undefined, currency?: string | null): string {
  if (amount == null) return EMPTY;
  const value = amount.trim();
  if (value === "") return EMPTY;
  if (!DECIMAL_RE.test(value)) return value;
  const scale = significantScale(value);

  if (currency && CURRENCY_RE.test(currency)) {
    try {
      const digits = currencyDigits(currency);
      const max = Math.min(Math.max(digits, scale), MAX_FRACTION_DIGITS);
      const format = numberFormat(`money|${currency}|${digits}|${max}`, {
        style: "currency",
        currency,
        minimumFractionDigits: digits,
        maximumFractionDigits: max,
      });
      return format.format(value as Intl.StringNumericLiteral);
    } catch {
      // An unknown code that Intl refuses: fall back to "<number> <code>" below.
    }
  }
  const plain = formatDecimalString(value, 2, scale);
  return currency ? `${plain} ${currency}` : plain;
}

/** Quantities and other decimals without forced fraction digits: `"1000"` -> `"1,000"`, `"1.5"` -> `"1.5"`. */
export function formatDecimal(value: string | null | undefined): string {
  if (value == null || value.trim() === "") return EMPTY;
  const trimmed = value.trim();
  if (!DECIMAL_RE.test(trimmed)) return trimmed;
  return formatDecimalString(trimmed, 0, significantScale(trimmed));
}

/** A tax rate printed as a percentage: `"20"` -> `"20%"`. */
export function formatPercent(rate: string | null | undefined): string {
  const formatted = formatDecimal(rate);
  return formatted === EMPTY ? EMPTY : `${formatted}%`;
}

export function formatCount(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return EMPTY;
  return numberFormat("count", { maximumFractionDigits: 0 }).format(value);
}

/** `plural(1, "error")` -> `"1 error"`, `plural(3, "error")` -> `"3 errors"`. */
export function plural(count: number, one: string, many = `${one}s`): string {
  return `${formatCount(count)} ${count === 1 ? one : many}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DATE_ONLY_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

/** `"2024-03-05"` -> `"5 Mar 2024"`. Month names avoid any day/month ambiguity; no time-zone shift. */
export function formatDate(value: string | null | undefined): string {
  if (value == null || value === "") return EMPTY;
  const match = DATE_ONLY_RE.exec(value);
  if (match) {
    const [, y, m, d] = match;
    const year = Number(y);
    const month = Number(m);
    const day = Number(d);
    const probe = new Date(Date.UTC(year, month - 1, day));
    if (probe.getUTCFullYear() === year && probe.getUTCMonth() === month - 1 && probe.getUTCDate() === day) {
      return `${day} ${MONTHS[month - 1]} ${year}`;
    }
    return value;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return `${parsed.getDate()} ${MONTHS[parsed.getMonth()]} ${parsed.getFullYear()}`;
}

/** A timestamp in the viewer's time zone: `"5 Mar 2024, 14:07"`. */
export function formatDateTime(value: string | null | undefined): string {
  if (value == null || value === "") return EMPTY;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const hh = String(parsed.getHours()).padStart(2, "0");
  const mm = String(parsed.getMinutes()).padStart(2, "0");
  return `${parsed.getDate()} ${MONTHS[parsed.getMonth()]} ${parsed.getFullYear()}, ${hh}:${mm}`;
}

/** Short relative time for recent timestamps ("3 min ago"); older ones fall back to a date. */
export function formatRelativeTime(value: string | null | undefined, now: number = Date.now()): string {
  if (value == null || value === "") return EMPTY;
  const time = new Date(value).getTime();
  if (Number.isNaN(time)) return value;
  const seconds = Math.round((now - time) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days} d ago`;
  return formatDate(value);
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes) || bytes < 0) return EMPTY;
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const format = numberFormat("bytes", { maximumFractionDigits: 1 });
  return `${format.format(value)} ${BYTE_UNITS[unit]}`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms) || ms < 0) return EMPTY;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${numberFormat("seconds", { maximumFractionDigits: 1 }).format(ms / 1000)} s`;
  const totalSeconds = Math.round(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return minutes > 0 ? `${hours} h ${minutes} min` : `${hours} h`;
  return seconds > 0 ? `${minutes} min ${seconds} s` : `${minutes} min`;
}

export type ConfidenceBucket = "high" | "medium" | "low" | "unknown";

/** At or above this a value is trusted at a glance. */
export const HIGH_CONFIDENCE = 0.9;
/** Below this the backend raises LOW_CONFIDENCE (setting `REVIEW_CONFIDENCE_THRESHOLD`). */
export const LOW_CONFIDENCE = 0.6;

export function confidenceBucket(confidence: number | null | undefined): ConfidenceBucket {
  if (confidence == null || Number.isNaN(confidence)) return "unknown";
  if (confidence >= HIGH_CONFIDENCE) return "high";
  if (confidence >= LOW_CONFIDENCE) return "medium";
  return "low";
}

export function formatConfidence(confidence: number | null | undefined): string {
  if (confidence == null || Number.isNaN(confidence)) return EMPTY;
  return `${Math.round(confidence * 100)}%`;
}

const ACRONYMS = new Set(["ocr", "llm", "pdf", "csv", "xls", "xlsx", "iso", "id", "po", "vat", "gst", "api"]);

/** `"TOTAL_MISMATCH"` -> `"Total mismatch"`, `"OCR_LOW_QUALITY"` -> `"OCR low quality"`. */
export function humanize(value: string): string {
  const words = value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => word.toLowerCase());
  return words
    .map((word, index) => {
      if (ACRONYMS.has(word)) return word.toUpperCase();
      return index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word;
    })
    .join(" ");
}

export const humanizeCode = humanize;

const FIELD_LABELS: Record<string, string> = {
  invoice_number: "Invoice number",
  invoice_date: "Invoice date",
  due_date: "Due date",
  supplier_name: "Supplier",
  customer_name: "Customer",
  subtotal: "Subtotal",
  discount: "Discount",
  shipping: "Shipping",
  tax: "Tax",
  tax_rate: "Tax rate",
  total: "Total",
  currency: "Currency",
  document_type: "Document type",
  line_items: "Line items",
  description: "Description",
  quantity: "Quantity",
  unit_price: "Unit price",
  amount: "Amount",
};

const LINE_ITEM_FIELD_RE = /^line_items\.(\d+)\.(\w+)$/;

/** Plain-language name of an invoice field (also `line_items.1.amount` -> `"Line 2 amount"`). */
export function fieldLabel(name: string): string {
  const lineItem = LINE_ITEM_FIELD_RE.exec(name);
  if (lineItem) {
    const part = FIELD_LABELS[lineItem[2]] ?? humanize(lineItem[2]);
    return `Line ${Number(lineItem[1]) + 1} ${part.toLowerCase()}`;
  }
  return FIELD_LABELS[name] ?? humanize(name);
}

export const DOCUMENT_STATUS_LABELS: Record<DocumentStatus, string> = {
  queued: "Queued",
  processing: "Processing",
  completed: "Completed",
  needs_review: "Needs review",
  failed: "Failed",
};

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  auto_approved: "Auto-approved",
  needs_review: "Needs review",
  reviewed: "Reviewed",
};

export const SEVERITY_LABELS: Record<Severity, { one: string; many: string }> = {
  error: { one: "Error", many: "Errors" },
  warning: { one: "Warning", many: "Warnings" },
  info: { one: "Info", many: "Info" },
};

export const SOURCE_LABELS: Record<ExtractionSource, string> = {
  rules: "Rules",
  llm: "LLM",
  "rules+llm": "Rules + LLM",
  spreadsheet: "Spreadsheet",
  manual: "Manual",
};

export const FILE_KIND_LABELS: Record<FileKind, string> = {
  pdf: "PDF",
  image: "Image",
  xlsx: "XLSX",
  xls: "XLS",
  csv: "CSV",
};

export const STAGE_LABELS: Record<JobStage, string> = {
  parsing: "Reading the file",
  ocr: "Recognising text (OCR)",
  extracting: "Extracting fields",
  validating: "Checking the numbers",
  indexing: "Indexing for search",
};

export const STAGE_ORDER: JobStage[] = ["parsing", "ocr", "extracting", "validating", "indexing"];

export const DOCUMENT_TYPE_LABELS: Record<InvoiceDocumentType, string> = {
  invoice: "Invoice",
  receipt: "Receipt",
  credit_note: "Credit note",
  unknown: "Unknown",
};

export const SEARCH_MODE_LABELS: Record<SearchMode, string> = {
  hybrid: "Hybrid",
  keyword: "Keyword",
  semantic: "Semantic",
};

export const MATCH_TYPE_LABELS: Record<SearchMatchType, string> = {
  keyword: "Keyword",
  semantic: "Semantic",
  hybrid: "Hybrid",
  filter: "Filter match",
};

export const PAGE_METHOD_LABELS: Record<PageMethod, string> = {
  text: "Text layer",
  ocr: "OCR",
  sheet: "Sheet",
  none: "No text",
};

export const CROSS_VERIFICATION_KIND_LABELS: Record<CrossVerificationKind, string> = {
  pdf_pdf: "PDF and PDF",
  pdf_excel: "PDF and spreadsheet",
  other: "Two records",
};

/** Look a value up in a label table, falling back to a humanized form of the raw value. */
export function labelFor<K extends string>(table: Record<K, string>, value: string | null | undefined): string {
  if (value == null || value === "") return EMPTY;
  return (table as Record<string, string>)[value] ?? humanize(value);
}

/** Where a record came from: `"Pages 3–4"`, `"Page 2"`, `"Sheet Ledger, row 12"`. */
export function formatRecordLocation(
  invoice: Pick<Invoice, "source_kind" | "page_start" | "page_end" | "sheet_name" | "row_number">,
): string {
  if (invoice.row_number != null) {
    return invoice.sheet_name ? `Sheet ${invoice.sheet_name}, row ${invoice.row_number}` : `Row ${invoice.row_number}`;
  }
  if (invoice.sheet_name) return `Sheet ${invoice.sheet_name}`;
  const { page_start: start, page_end: end } = invoice;
  if (start == null) return "";
  if (end == null || end === start) return `Page ${start}`;
  return `Pages ${start}–${end}`;
}

/** A short identity for a record, e.g. `"INV-0042"` or `"Record 3"` when the number is missing. */
export function recordTitle(invoice: Pick<Invoice, "invoice_number" | "record_index">): string {
  return invoice.invoice_number ?? `Record ${invoice.record_index + 1}`;
}
