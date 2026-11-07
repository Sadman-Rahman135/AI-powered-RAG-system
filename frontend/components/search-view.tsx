"use client";

/**
 * Search across every document. The URL holds the query, mode, filters and page, so a search can be bookmarked
 * and back/forward works; results show where each match came from with its highlighted snippet.
 */
import { SearchIcon, SearchX } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo } from "react";

import { PageHeader } from "@/components/page-header";
import { PaginationBar } from "@/components/pagination-bar";
import { EmptyState, QueryError } from "@/components/query-state";
import { SearchFilters } from "@/components/search-filters";
import { Snippet } from "@/components/snippet";
import { ReviewStatusBadge } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EMPTY, MATCH_TYPE_LABELS, SEARCH_MODE_LABELS, formatCount, formatDate, formatMoney, labelFor } from "@/lib/format";
import { useSearch } from "@/lib/queries";
import {
  hasSearchCriteria,
  searchApiParams,
  searchStateFromQuery,
  searchStateToQuery,
  type SearchState,
} from "@/lib/search-params";
import type { SearchHit, SearchResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 20;

/** Open the record a hit belongs to, or the page its text came from. */
function hitHref(hit: SearchHit): string {
  const base = `/documents/${encodeURIComponent(hit.document_id)}`;
  if (hit.invoice_id) return `${base}?invoice=${encodeURIComponent(hit.invoice_id)}`;
  if (hit.page_number != null) return `${base}?tab=text&page=${hit.page_number}`;
  return base;
}

function navigate(state: SearchState) {
  const query = searchStateToQuery(state);
  // Native history updates are synced into useSearchParams by the Next.js router, without a server round trip.
  window.history.pushState(null, "", query ? `/search?${query}` : "/search");
}

export function SearchView() {
  const searchParams = useSearchParams();
  const queryString = searchParams.toString();
  const state = useMemo(() => searchStateFromQuery(queryString), [queryString]);
  const enabled = hasSearchCriteria(state);
  const { data, isPending, isError, error, refetch, isFetching, isPlaceholderData } = useSearch(
    searchApiParams(state, PAGE_SIZE),
    enabled,
  );

  return (
    <>
      <PageHeader
        title="Search"
        description="Find invoices by words, numbers or meaning across every document, and narrow them down with filters."
      />
      <div className="rounded-lg border bg-card p-4">
        <SearchFilters key={queryString} value={state} onSubmit={navigate} pending={isFetching && !isPlaceholderData} />
      </div>

      {!enabled ? (
        <EmptyState
          icon={SearchIcon}
          title="Search every document"
          description="Try an invoice number such as INV-0042, a supplier name, an amount, or a phrase like “freight charges in March”. Filters alone also work: pick a supplier or a date range and search."
        />
      ) : isError && !data ? (
        <QueryError error={error} title="The search failed" onRetry={() => void refetch()} />
      ) : isPending ? (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Searching">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : (
        <SearchResults
          data={data}
          page={state.page}
          stale={isPlaceholderData}
          filterOnly={state.q === ""}
          onPageChange={(page) => navigate({ ...state, page })}
        />
      )}
    </>
  );
}

interface SearchResultsProps {
  data: SearchResponse;
  page: number;
  stale: boolean;
  /** A blank query with filters: structured filtering, no relevance ranking. */
  filterOnly: boolean;
  onPageChange: (page: number) => void;
}

/**
 * The results line. For semantic and hybrid searches `total` counts the candidates that were ranked, not hits that
 * are known to be relevant, so the wording only ever says "results" and how they were ordered.
 */
function summaryText(data: SearchResponse, filterOnly: boolean): string {
  const count = `${formatCount(data.total)} ${data.total === 1 ? "result" : "results"}`;
  if (filterOnly) return `${count} matching the filters`;
  const took = `${formatCount(Math.round(data.took_ms))} ms`;
  if (data.mode === "semantic") return `${count} in ${took}, ranked by similarity`;
  if (data.mode === "hybrid") return `${count} in ${took}, ranked by words and similarity`;
  return `${count} in ${took}, ${labelFor(SEARCH_MODE_LABELS, data.mode).toLowerCase()} search`;
}

export function SearchResults({ data, page, stale, filterOnly, onPageChange }: SearchResultsProps) {
  if (data.items.length === 0) {
    return (
      <EmptyState
        icon={SearchX}
        title={filterOnly ? "No records fit these filters" : "No results"}
        description={
          filterOnly
            ? "Widen the date or amount range, or remove a filter."
            : "Try semantic mode to search by meaning, use fewer words, or remove a filter."
        }
      />
    );
  }

  return (
    <section aria-label="Search results" className={cn("flex flex-col gap-3 transition-opacity", stale && "opacity-60")}>
      <p className="text-sm text-muted-foreground" aria-live="polite" data-testid="search-summary">
        {summaryText(data, filterOnly)}
      </p>
      <ol data-testid="search-results" className="flex flex-col gap-3">
        {data.items.map((hit, index) => (
          <li key={`${hit.document_id}:${hit.invoice_id ?? ""}:${hit.page_number ?? ""}:${index}`} data-testid="search-result">
            <article className="flex flex-col gap-2 rounded-lg border bg-card p-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="flex min-w-0 flex-col">
                  <Link href={hitHref(hit)} className="truncate font-medium underline-offset-2 hover:underline">
                    {hit.invoice_number ? <span className="figures">{hit.invoice_number}</span> : hit.filename}
                  </Link>
                  <span className="truncate text-xs text-muted-foreground">
                    {hit.filename}
                    {hit.page_number != null ? `, page ${hit.page_number}` : ""}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {hit.review_status ? <ReviewStatusBadge status={hit.review_status} /> : null}
                  <Badge variant="neutral" data-testid="match-type">
                    {labelFor(MATCH_TYPE_LABELS, hit.match_type)}
                  </Badge>
                  {/* Filter matches are not ranked, so their score carries no meaning. */}
                  {hit.match_type !== "filter" ? (
                    <span
                      className="figures text-xs text-muted-foreground"
                      title="Ranking score; it orders results and does not mean a result is relevant"
                      data-testid="score"
                    >
                      {hit.score.toFixed(3)}
                    </span>
                  ) : null}
                </div>
              </div>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
                <div className="min-w-0">
                  <dt className="text-xs text-muted-foreground">Supplier</dt>
                  <dd className="truncate">{hit.supplier_name ?? EMPTY}</dd>
                </div>
                <div className="min-w-0">
                  <dt className="text-xs text-muted-foreground">Customer</dt>
                  <dd className="truncate">{hit.customer_name ?? EMPTY}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Invoice date</dt>
                  <dd>{formatDate(hit.invoice_date)}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">Total</dt>
                  <dd className="figures">{formatMoney(hit.total, hit.currency)}</dd>
                </div>
              </dl>
              {hit.snippet ? (
                <p className="text-sm leading-relaxed text-muted-foreground">
                  <Snippet text={hit.snippet} />
                </p>
              ) : null}
            </article>
          </li>
        ))}
      </ol>
      <PaginationBar page={page} pageSize={PAGE_SIZE} total={data.total} onPageChange={onPageChange} noun="results" />
    </section>
  );
}
