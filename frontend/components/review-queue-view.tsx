"use client";

/** Records that need a human: failed a check or read with low confidence. Each links to its record. */
import { ClipboardCheck, OctagonX, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useId, useState } from "react";

import { ConfidenceBadge } from "@/components/confidence-badge";
import { PageHeader } from "@/components/page-header";
import { PaginationBar } from "@/components/pagination-bar";
import { EmptyState, QueryError } from "@/components/query-state";
import { buttonVariants } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldLabel } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { InvoiceListParams } from "@/lib/api";
import { EMPTY, formatCount, formatDate, formatMoney, formatRecordLocation, humanizeCode, recordTitle } from "@/lib/format";
import { useInvoices } from "@/lib/queries";
import type { Invoice } from "@/lib/types";

const PAGE_SIZE = 20;

function recordHref(invoice: Invoice): string {
  return `/documents/${encodeURIComponent(invoice.document_id)}?invoice=${encodeURIComponent(invoice.id)}`;
}

function IssueSummary({ invoice }: { invoice: Invoice }) {
  const errors = invoice.issues.filter((issue) => issue.severity === "error");
  const warnings = invoice.issues.filter((issue) => issue.severity === "warning");
  const ordered = [...errors, ...warnings];
  const codes = [...new Set(ordered.map((issue) => humanizeCode(issue.code)))];
  const shown = codes.slice(0, 2);

  if (ordered.length === 0) return <span className="text-xs text-muted-foreground">Low confidence or information only</span>;
  return (
    <div className="flex flex-col gap-0.5">
      <span className="flex items-center gap-3 text-xs">
        {errors.length > 0 ? (
          <span className="figures inline-flex items-center gap-1 text-tone-danger-foreground">
            <OctagonX aria-hidden="true" className="size-3.5" />
            {formatCount(errors.length)}
            <span className="sr-only"> {errors.length === 1 ? "error" : "errors"}</span>
          </span>
        ) : null}
        {warnings.length > 0 ? (
          <span className="figures inline-flex items-center gap-1 text-tone-warning-foreground">
            <TriangleAlert aria-hidden="true" className="size-3.5" />
            {formatCount(warnings.length)}
            <span className="sr-only"> {warnings.length === 1 ? "warning" : "warnings"}</span>
          </span>
        ) : null}
      </span>
      <span className="truncate text-sm" title={codes.join(", ")}>
        {shown.join(", ")}
        {codes.length > shown.length ? <span className="text-muted-foreground"> and {codes.length - shown.length} more</span> : null}
      </span>
    </div>
  );
}

export function ReviewQueueView() {
  const [onlyErrors, setOnlyErrors] = useState(false);
  const [page, setPage] = useState(1);
  const id = useId();
  const params: InvoiceListParams = {
    review_status: "needs_review",
    has_errors: onlyErrors ? true : undefined,
    limit: PAGE_SIZE,
    offset: (page - 1) * PAGE_SIZE,
  };
  const { data, isPending, isError, error, refetch } = useInvoices(params);

  return (
    <>
      <PageHeader
        title="Review queue"
        description="Records that failed a check or were read with low confidence. Open one to correct it, then approve it."
      />
      <Field orientation="horizontal" className="w-fit">
        <Checkbox
          id={`${id}-errors`}
          checked={onlyErrors}
          onCheckedChange={(checked) => {
            setOnlyErrors(checked === true);
            setPage(1);
          }}
          data-testid="review-only-errors"
        />
        <FieldLabel htmlFor={`${id}-errors`}>Only records with errors</FieldLabel>
      </Field>

      {isError && !data ? (
        <QueryError error={error} title="The review queue could not be loaded" onRetry={() => void refetch()} />
      ) : isPending ? (
        <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading the review queue">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : data.items.length === 0 ? (
        <EmptyState
          icon={ClipboardCheck}
          title={onlyErrors ? "No records with errors" : "Nothing to review"}
          description="Records that fail a check, or whose fields were read with low confidence, appear here."
        />
      ) : (
        <>
          <div className="overflow-hidden rounded-lg border bg-card">
            <Table data-testid="review-table">
              <TableHeader>
                <TableRow className="bg-muted/50 hover:bg-muted/50">
                  <TableHead>Invoice</TableHead>
                  <TableHead>Supplier</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="w-[26%]">Checks</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead className="hidden xl:table-cell">Document</TableHead>
                  <TableHead>
                    <span className="sr-only">Action</span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.items.map((invoice) => (
                  <TableRow key={invoice.id} data-testid="review-row">
                    <TableCell>
                      <Link href={recordHref(invoice)} className="figures font-medium underline-offset-2 hover:underline">
                        {recordTitle(invoice)}
                      </Link>
                    </TableCell>
                    <TableCell className="max-w-48 truncate">{invoice.supplier_name ?? EMPTY}</TableCell>
                    <TableCell className="text-muted-foreground">{formatDate(invoice.invoice_date)}</TableCell>
                    <TableCell className="figures text-right">{formatMoney(invoice.total, invoice.currency)}</TableCell>
                    <TableCell className="max-w-0 whitespace-normal">
                      <IssueSummary invoice={invoice} />
                    </TableCell>
                    <TableCell>
                      <ConfidenceBadge value={invoice.overall_confidence} />
                    </TableCell>
                    <TableCell className="hidden max-w-56 xl:table-cell">
                      <span className="block truncate text-muted-foreground" title={invoice.filename}>
                        {invoice.filename}
                      </span>
                      <span className="text-xs text-muted-foreground">{formatRecordLocation(invoice)}</span>
                    </TableCell>
                    <TableCell className="text-right">
                      <Link href={recordHref(invoice)} className={buttonVariants({ variant: "outline", size: "sm" })}>
                        Review
                      </Link>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <PaginationBar page={page} pageSize={PAGE_SIZE} total={data.total} onPageChange={setPage} noun="records" />
        </>
      )}
    </>
  );
}
