/**
 * Search page state <-> URL query string. The URL is the source of truth so results can be bookmarked, shared and
 * navigated with back/forward. Invalid values in a URL are dropped rather than sent to the API.
 */
import type { SearchApiParams } from "@/lib/api";
import type { FileKind, ReviewStatus, SearchMode } from "@/lib/types";

export interface SearchState {
  q: string;
  mode: SearchMode;
  supplier: string;
  customer: string;
  invoice_number: string;
  currency: string;
  date_from: string;
  date_to: string;
  total_min: string;
  total_max: string;
  review_status: ReviewStatus | "";
  file_kind: FileKind | "";
  /** 1-based results page. */
  page: number;
}

export const DEFAULT_SEARCH_STATE: SearchState = {
  q: "",
  mode: "hybrid",
  supplier: "",
  customer: "",
  invoice_number: "",
  currency: "",
  date_from: "",
  date_to: "",
  total_min: "",
  total_max: "",
  review_status: "",
  file_kind: "",
  page: 1,
};

export const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "semantic"];
export const REVIEW_STATUSES: readonly ReviewStatus[] = ["needs_review", "auto_approved", "reviewed"];
export const FILE_KINDS: readonly FileKind[] = ["pdf", "image", "xlsx", "xls", "csv"];

export const FILTER_KEYS = [
  "supplier",
  "customer",
  "invoice_number",
  "currency",
  "date_from",
  "date_to",
  "total_min",
  "total_max",
  "review_status",
  "file_kind",
] as const;

export type FilterKey = (typeof FILTER_KEYS)[number];

const TEXT_KEYS = ["supplier", "customer", "invoice_number"] as const;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const AMOUNT_RE = /^-?\d+(\.\d+)?$/;
const CURRENCY_RE = /^[A-Z]{3}$/;

function oneOf<T extends string>(allowed: readonly T[], value: string | null): T | "" {
  return value !== null && (allowed as readonly string[]).includes(value) ? (value as T) : "";
}

/** Trimmed, canonical values (currency upper-cased). */
function normalized(state: SearchState): SearchState {
  return {
    ...state,
    q: state.q.trim(),
    supplier: state.supplier.trim(),
    customer: state.customer.trim(),
    invoice_number: state.invoice_number.trim(),
    currency: state.currency.trim().toUpperCase(),
    date_from: state.date_from.trim(),
    date_to: state.date_to.trim(),
    total_min: state.total_min.trim(),
    total_max: state.total_max.trim(),
  };
}

export function searchStateToQuery(state: SearchState): string {
  const s = normalized(state);
  const params = new URLSearchParams();
  if (s.q) params.set("q", s.q);
  if (s.mode !== DEFAULT_SEARCH_STATE.mode) params.set("mode", s.mode);
  for (const key of FILTER_KEYS) {
    if (s[key]) params.set(key, s[key]);
  }
  if (s.page > 1) params.set("page", String(s.page));
  return params.toString();
}

export function searchStateFromQuery(input: URLSearchParams | string): SearchState {
  const params = typeof input === "string" ? new URLSearchParams(input) : input;
  const text = (key: string) => (params.get(key) ?? "").trim();
  const matching = (key: string, re: RegExp) => {
    const value = text(key);
    return re.test(value) ? value : "";
  };
  const page = Number.parseInt(params.get("page") ?? "", 10);
  const currency = text("currency").toUpperCase();

  const state: SearchState = {
    ...DEFAULT_SEARCH_STATE,
    q: text("q"),
    mode: oneOf(SEARCH_MODES, params.get("mode")) || DEFAULT_SEARCH_STATE.mode,
    currency: CURRENCY_RE.test(currency) ? currency : "",
    date_from: matching("date_from", DATE_RE),
    date_to: matching("date_to", DATE_RE),
    total_min: matching("total_min", AMOUNT_RE),
    total_max: matching("total_max", AMOUNT_RE),
    review_status: oneOf(REVIEW_STATUSES, params.get("review_status")),
    file_kind: oneOf(FILE_KINDS, params.get("file_kind")),
    page: Number.isFinite(page) && page > 1 ? page : 1,
  };
  for (const key of TEXT_KEYS) state[key] = text(key);
  return state;
}

/** True when there is a query or at least one filter (an empty query with filters is a structured search). */
export function hasSearchCriteria(state: SearchState): boolean {
  const s = normalized(state);
  return s.q !== "" || FILTER_KEYS.some((key) => s[key] !== "");
}

export function activeFilterCount(state: SearchState): number {
  const s = normalized(state);
  return FILTER_KEYS.filter((key) => s[key] !== "").length;
}

export function clearFilters(state: SearchState): SearchState {
  const cleared = { ...state, page: 1 };
  for (const key of FILTER_KEYS) cleared[key] = "";
  return cleared;
}

export function searchApiParams(state: SearchState, limit: number): SearchApiParams {
  const s = normalized(state);
  const filters = Object.fromEntries(FILTER_KEYS.filter((key) => s[key] !== "").map((key) => [key, s[key]])) as Pick<
    SearchApiParams,
    FilterKey
  >;
  return {
    ...(s.q ? { q: s.q } : {}),
    mode: s.mode,
    ...filters,
    limit,
    offset: (Math.max(s.page, 1) - 1) * limit,
  };
}
