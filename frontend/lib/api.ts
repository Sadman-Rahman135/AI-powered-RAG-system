/**
 * Typed client for the backend REST API.
 *
 * The browser only calls same-origin `/api/v1/...`; the Next.js route handler in `app/api/[...path]` forwards those
 * requests to the backend and adds the API key server-side. Every non-2xx response becomes an `ApiError` carrying
 * the standard envelope's `code`, `message` and `details`.
 */
import type {
  AppConfig,
  Correction,
  CrossVerification,
  CrossVerifyRequest,
  Document,
  DocumentDetail,
  DocumentResults,
  DocumentStatus,
  DocumentStatusResponse,
  FileKind,
  Invoice,
  InvoicePatch,
  JobsResponse,
  JobStatus,
  PagesResponse,
  Paged,
  ProcessingEvent,
  ReprocessFailedResponse,
  ReprocessResponse,
  ReviewStatus,
  SearchMode,
  SearchResponse,
  Stats,
} from "@/lib/types";

export const API_BASE = "/api/v1";

const NETWORK_MESSAGE = "Could not reach the server. Check your connection and try again.";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/** A readable message for any thrown value, for error states and toasts. */
export function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return "Something went wrong.";
}

export function networkError(): ApiError {
  return new ApiError(0, "network_error", NETWORK_MESSAGE);
}

interface ErrorEnvelope {
  error: { code: string; message?: string; details?: unknown };
}

function isEnvelope(value: unknown): value is ErrorEnvelope {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const error = (value as { error: unknown }).error;
  return typeof error === "object" && error !== null && typeof (error as { code?: unknown }).code === "string";
}

/** Build an ApiError from a failed response's status and raw body (shared by fetch and the XHR upload). */
export function apiErrorFrom(status: number, statusText: string, bodyText: string): ApiError {
  let parsed: unknown = null;
  if (bodyText) {
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      parsed = null;
    }
  }
  const fallback = `Request failed (${status}${statusText ? ` ${statusText}` : ""}).`;
  if (isEnvelope(parsed)) {
    return new ApiError(status, parsed.error.code, parsed.error.message || fallback, parsed.error.details ?? null);
  }
  if (typeof parsed === "object" && parsed !== null && "detail" in parsed) {
    // A framework-level error that bypassed the envelope (e.g. FastAPI's {"detail": ...}).
    const detail = (parsed as { detail: unknown }).detail;
    return new ApiError(status, `http_${status}`, typeof detail === "string" ? detail : fallback, detail);
  }
  return new ApiError(status, `http_${status}`, fallback);
}

type QueryScalar = string | number | boolean;
export type QueryValue = QueryScalar | null | undefined | readonly QueryScalar[];
export type QueryParams = Record<string, QueryValue>;

/** `{a: 1, b: "", c: null, g: ["x", "y"]}` -> `"?a=1&g=x&g=y"`; empty values are dropped. */
export function buildQuery(params: object): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params as QueryParams)) {
    if (value == null || value === "") continue;
    if (Array.isArray(value)) {
      for (const item of value) search.append(key, String(item));
    } else {
      search.append(key, String(value));
    }
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

interface RequestOptions {
  query?: object;
  json?: unknown;
  signal?: AbortSignal;
}

async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers({ accept: "application/json" });
  let body: string | undefined;
  if (options.json !== undefined) {
    headers.set("content-type", "application/json");
    body = JSON.stringify(options.json);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}${buildQuery(options.query ?? {})}`, {
      method,
      headers,
      body,
      signal: options.signal,
      credentials: "same-origin",
      cache: "no-store",
    });
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw networkError();
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw apiErrorFrom(response.status, response.statusText, text);
  }
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(response.status, "invalid_response", "The server sent a response the app could not read.");
  }
}

const segment = (value: string) => encodeURIComponent(value);

export interface DocumentListParams {
  status?: DocumentStatus | "";
  file_kind?: FileKind | "";
  q?: string;
  needs_review?: boolean;
  limit?: number;
  offset?: number;
  sort?: string;
}

export interface InvoiceFilterParams {
  q?: string;
  supplier?: string;
  customer?: string;
  invoice_number?: string;
  currency?: string;
  date_from?: string;
  date_to?: string;
  total_min?: string;
  total_max?: string;
  review_status?: ReviewStatus | "";
}

export interface InvoiceListParams extends InvoiceFilterParams {
  has_errors?: boolean;
  limit?: number;
  offset?: number;
  sort?: string;
}

export interface SearchApiParams extends InvoiceFilterParams {
  mode: SearchMode;
  file_kind?: FileKind | "";
  limit?: number;
  offset?: number;
}

export interface CrossVerificationListParams {
  invoice_id?: string;
  document_id?: string;
  status?: string;
}

export const api = {
  listDocuments: (params: DocumentListParams = {}, signal?: AbortSignal) =>
    request<Paged<Document>>("GET", "/documents", { query: params, signal }),
  getDocument: (id: string, signal?: AbortSignal) =>
    request<DocumentDetail>("GET", `/documents/${segment(id)}`, { signal }),
  getDocumentStatus: (id: string, signal?: AbortSignal) =>
    request<DocumentStatusResponse>("GET", `/documents/${segment(id)}/status`, { signal }),
  getDocumentResults: (id: string, signal?: AbortSignal) =>
    request<DocumentResults>("GET", `/documents/${segment(id)}/results`, { signal }),
  getDocumentPages: (id: string, params: { offset: number; limit: number }, signal?: AbortSignal) =>
    request<PagesResponse>("GET", `/documents/${segment(id)}/pages`, { query: params, signal }),
  getDocumentEvents: (id: string, signal?: AbortSignal) =>
    request<{ items: ProcessingEvent[] }>("GET", `/documents/${segment(id)}/events`, { signal }),
  documentFileUrl: (id: string) => `${API_BASE}/documents/${segment(id)}/file`,
  reprocessDocument: (id: string) => request<ReprocessResponse>("POST", `/documents/${segment(id)}/reprocess`),
  deleteDocument: (id: string) => request<void>("DELETE", `/documents/${segment(id)}`),

  listInvoices: (params: InvoiceListParams = {}, signal?: AbortSignal) =>
    request<Paged<Invoice>>("GET", "/invoices", { query: params, signal }),
  getInvoice: (id: string, signal?: AbortSignal) => request<Invoice>("GET", `/invoices/${segment(id)}`, { signal }),
  patchInvoice: (id: string, patch: InvoicePatch) =>
    request<Invoice>("PATCH", `/invoices/${segment(id)}`, { json: patch }),
  approveInvoice: (id: string, comment: string | null) =>
    request<Invoice>("POST", `/invoices/${segment(id)}/approve`, { json: { comment } }),
  getCorrections: (id: string, signal?: AbortSignal) =>
    request<{ items: Correction[] }>("GET", `/invoices/${segment(id)}/corrections`, { signal }),

  search: (params: SearchApiParams, signal?: AbortSignal) =>
    request<SearchResponse>("GET", "/search", { query: params, signal }),

  crossVerify: (body: CrossVerifyRequest) => request<CrossVerification>("POST", "/cross-verify", { json: body }),
  listCrossVerifications: (params: CrossVerificationListParams, signal?: AbortSignal) =>
    request<{ items: CrossVerification[] }>("GET", "/cross-verifications", { query: params, signal }),

  listJobs: (params: { status?: JobStatus; limit?: number; offset?: number } = {}, signal?: AbortSignal) =>
    request<JobsResponse>("GET", "/jobs", { query: params, signal }),
  reprocessFailed: () => request<ReprocessFailedResponse>("POST", "/jobs/reprocess-failed"),

  getStats: (signal?: AbortSignal) => request<Stats>("GET", "/stats", { signal }),
  getConfig: (signal?: AbortSignal) => request<AppConfig>("GET", "/config", { signal }),
};
