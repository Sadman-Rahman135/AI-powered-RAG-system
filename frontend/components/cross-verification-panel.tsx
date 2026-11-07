"use client";

/**
 * Automatic cross-verification results for a document (records sharing an invoice number with another document,
 * e.g. a PDF against its row in an Excel ledger), plus a way to compare it with any other document.
 */
import { ChevronDown, GitCompareArrows, Scale } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { ComparisonResult } from "@/components/comparison-result";
import { EmptyState, QueryError } from "@/components/query-state";
import { MatchBadge } from "@/components/status-badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { CROSS_VERIFICATION_KIND_LABELS, formatDateTime, labelFor } from "@/lib/format";
import { useCrossVerifications } from "@/lib/queries";
import type { CrossVerification, DocumentDetail } from "@/lib/types";
import { cn } from "@/lib/utils";

function counterpart(result: CrossVerification, documentId: string) {
  return result.left.document_id === documentId ? result.right : result.left;
}

export function CrossVerificationPanel({ document }: { document: DocumentDetail }) {
  const { data, isPending, isError, error, refetch } = useCrossVerifications({ document_id: document.id }, document.updated_at);
  // Mismatches start open, matches closed; a click flips the default for that result.
  const [flipped, setFlipped] = useState<ReadonlySet<string>>(new Set());

  function toggle(id: string) {
    setFlipped((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="max-w-[72ch] text-muted-foreground">
          Records that share an invoice number with another document are compared field by field, for example a PDF
          invoice against its row in a spreadsheet ledger.
        </p>
        <Link
          href={`/compare?type=documents&left=${encodeURIComponent(document.id)}`}
          className={buttonVariants({ variant: "outline" })}
          data-testid="compare-with"
        >
          <GitCompareArrows aria-hidden="true" data-icon="inline-start" />
          Compare with another document
        </Link>
      </div>

      {isError && !data ? (
        <QueryError error={error} title="Cross-verification results could not be loaded" onRetry={() => void refetch()} />
      ) : isPending ? (
        <Skeleton className="h-32 w-full" />
      ) : data.items.length === 0 ? (
        <EmptyState
          icon={Scale}
          title="No other document shares an invoice number with this one"
          description="Use Compare to check it against a specific document or ledger instead."
        />
      ) : (
        <ul className="flex flex-col gap-3">
          {data.items.map((result) => {
            const other = counterpart(result, document.id);
            const open = (result.status !== "match") !== flipped.has(result.id);
            const panelId = `cv-${result.id}`;
            return (
              <li key={result.id} data-testid="cross-verification" className="rounded-lg border bg-card">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
                  <MatchBadge status={result.status} />
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate">
                      Against{" "}
                      <Link href={`/documents/${other.document_id}`} className="font-medium underline-offset-2 hover:underline">
                        {other.filename}
                      </Link>
                      {other.invoice_number ? <span className="figures text-muted-foreground"> {other.invoice_number}</span> : null}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {labelFor(CROSS_VERIFICATION_KIND_LABELS, result.kind)}, {result.trigger === "auto" ? "checked automatically" : "compared on request"} on{" "}
                      {formatDateTime(result.created_at)}
                    </span>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    aria-expanded={open}
                    aria-controls={panelId}
                    onClick={() => toggle(result.id)}
                  >
                    {open ? "Hide fields" : "Show fields"}
                    <ChevronDown aria-hidden="true" data-icon="inline-end" className={cn("transition-transform", open && "rotate-180")} />
                  </Button>
                </div>
                {open ? (
                  <div id={panelId} className="border-t p-4">
                    <ComparisonResult result={result} currentDocumentId={document.id} showSummary={false} />
                  </div>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
