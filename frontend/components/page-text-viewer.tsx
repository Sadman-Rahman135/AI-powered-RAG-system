"use client";

/** The text read from each page or sheet, a few pages at a time, with how it was read (text layer or OCR). */
import { FileText } from "lucide-react";
import { useState } from "react";

import { ConfidenceBadge } from "@/components/confidence-badge";
import { PaginationBar } from "@/components/pagination-bar";
import { EmptyState, QueryError } from "@/components/query-state";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { PAGE_METHOD_LABELS, labelFor } from "@/lib/format";
import { useDocumentPages } from "@/lib/queries";
import { isActiveDocument } from "@/lib/status";
import type { DocumentDetail } from "@/lib/types";

const PAGES_PER_VIEW = 5;

interface PageTextViewerProps {
  document: DocumentDetail;
  /** 1-based page number to open first (e.g. from a search hit). */
  initialPage?: number;
}

export function PageTextViewer({ document, initialPage = 1 }: PageTextViewerProps) {
  const [view, setView] = useState(() => Math.max(1, Math.ceil(initialPage / PAGES_PER_VIEW)));
  const offset = (view - 1) * PAGES_PER_VIEW;
  const { data, isPending, isError, error, refetch } = useDocumentPages(
    document.id,
    offset,
    PAGES_PER_VIEW,
    document.updated_at,
  );

  if (isError && !data) {
    return <QueryError error={error} title="The page text could not be loaded" onRetry={() => void refetch()} />;
  }
  if (isPending) {
    return (
      <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading page text">
        {Array.from({ length: 2 }, (_, i) => (
          <Skeleton key={i} className="h-40 w-full" />
        ))}
      </div>
    );
  }
  if (data.total === 0) {
    return (
      <EmptyState
        icon={FileText}
        title="No text yet"
        description={
          isActiveDocument(document)
            ? "The text appears here once the file has been read."
            : "No readable text was found in this file."
        }
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {data.items.map((page) => (
        <article key={page.page_number} data-testid="page-text" className="flex flex-col gap-2 rounded-lg border bg-card p-4">
          <header className="flex flex-wrap items-center gap-2">
            <h3 className="font-semibold">{page.label ? `Sheet ${page.label}` : `Page ${page.page_number}`}</h3>
            <Badge variant={page.method === "ocr" ? "info" : "neutral"}>{labelFor(PAGE_METHOD_LABELS, page.method)}</Badge>
            {page.ocr_confidence != null ? (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                OCR confidence <ConfidenceBadge value={page.ocr_confidence} />
              </span>
            ) : null}
          </header>
          {page.text ? (
            <pre className="figures max-h-[28rem] overflow-auto rounded-md bg-muted/50 p-3 text-xs leading-relaxed whitespace-pre-wrap">
              {page.text}
            </pre>
          ) : (
            <p className="text-sm text-muted-foreground">No text on this page.</p>
          )}
        </article>
      ))}
      <PaginationBar page={view} pageSize={PAGES_PER_VIEW} total={data.total} onPageChange={setView} noun="pages" />
    </div>
  );
}
