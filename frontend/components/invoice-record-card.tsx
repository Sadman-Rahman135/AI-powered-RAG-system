"use client";

import { GitCompareArrows } from "lucide-react";
import Link from "next/link";

import { ConfidenceBadge } from "@/components/confidence-badge";
import { InvoiceEditor } from "@/components/invoice-editor";
import { ReviewStatusBadge } from "@/components/status-badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/components/ui/toast";
import { formatRecordLocation, humanize, recordTitle } from "@/lib/format";
import { useApproveInvoice, usePatchInvoice } from "@/lib/queries";
import type { Invoice, InvoicePatch } from "@/lib/types";
import { cn } from "@/lib/utils";

const SOURCE_KIND_LABELS: Record<string, string> = {
  invoice_document: "Invoice document",
  ledger_row: "Spreadsheet ledger row",
};

function remainingIssues(invoice: Invoice): string {
  const open = invoice.issues.filter((issue) => issue.severity !== "info").length;
  if (open === 0) return "Checked again: every check passes.";
  return `Checked again: ${open} ${open === 1 ? "issue remains" : "issues remain"}.`;
}

interface InvoiceRecordCardProps {
  invoice: Invoice;
  position: number;
  total: number;
  highlighted?: boolean;
}

export function InvoiceRecordCard({ invoice, position, total, highlighted = false }: InvoiceRecordCardProps) {
  const patch = usePatchInvoice();
  const approve = useApproveInvoice();
  const location = formatRecordLocation(invoice);

  async function save(changes: InvoicePatch) {
    const updated = await patch.mutateAsync({ id: invoice.id, patch: changes });
    toast.add({ title: "Changes saved", description: remainingIssues(updated), type: "success" });
    return updated;
  }

  async function approveRecord(comment: string | null) {
    const updated = await approve.mutateAsync({ id: invoice.id, comment });
    toast.add({ title: "Record approved", description: recordTitle(updated), type: "success" });
    return updated;
  }

  return (
    <Card
      id={`record-${invoice.id}`}
      data-testid="invoice-card"
      data-review-status={invoice.review_status}
      className={cn("scroll-mt-20", highlighted && "ring-2 ring-primary")}
    >
      <CardHeader className="border-b">
        <CardTitle className="figures text-base">{recordTitle(invoice)}</CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {total > 1 ? (
            <span>
              Record {position} of {total}
            </span>
          ) : null}
          {location ? <span>{location}</span> : null}
          <span>{SOURCE_KIND_LABELS[invoice.source_kind] ?? humanize(invoice.source_kind)}</span>
          <span>Extracted with {humanize(invoice.extraction_method)}</span>
          <span className="inline-flex items-center gap-1.5">
            Overall confidence <ConfidenceBadge value={invoice.overall_confidence} />
          </span>
        </CardDescription>
        <CardAction className="flex items-center gap-2">
          <ReviewStatusBadge status={invoice.review_status} />
          <Link
            href={`/compare?type=invoices&left=${encodeURIComponent(invoice.id)}`}
            className={buttonVariants({ variant: "ghost", size: "sm" })}
          >
            <GitCompareArrows aria-hidden="true" data-icon="inline-start" />
            Compare
          </Link>
        </CardAction>
      </CardHeader>
      <CardContent>
        <InvoiceEditor key={`${invoice.id}:${invoice.updated_at}`} invoice={invoice} onSave={save} onApprove={approveRecord} />
      </CardContent>
    </Card>
  );
}
