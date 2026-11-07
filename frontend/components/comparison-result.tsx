/**
 * Field-by-field comparison of two records (cross-verification). The centre gutter carries the accountant's tick
 * marks: a tick where both sides agree, a cross where they disagree, a dash where one side has no value.
 */
import { Check, CircleDashed, EqualApproximately, Minus, X } from "lucide-react";
import Link from "next/link";

import { MatchBadge } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import {
  CROSS_VERIFICATION_KIND_LABELS,
  EMPTY,
  fieldLabel,
  formatCount,
  formatDate,
  formatDateTime,
  formatMoney,
  formatPercent,
  humanize,
  labelFor,
} from "@/lib/format";
import type { CrossVerification, CrossVerificationSide, FieldComparison } from "@/lib/types";
import { cn } from "@/lib/utils";

const MONEY_FIELDS = new Set(["subtotal", "discount", "shipping", "tax", "total", "amount", "unit_price"]);
const DATE_FIELDS = new Set(["invoice_date", "due_date"]);

function currencyOf(fields: readonly FieldComparison[], side: "left" | "right"): string | null {
  const value = fields.find((f) => f.field === "currency")?.[side];
  return typeof value === "string" ? value : null;
}

/** Display a compared value in the same conventions as the rest of the app. */
export function formatComparedValue(field: string, value: unknown, currency: string | null): string {
  if (value === null || value === undefined || value === "") return EMPTY;
  if (typeof value === "string" || typeof value === "number") {
    const text = String(value);
    if (MONEY_FIELDS.has(field)) return formatMoney(text, currency);
    if (DATE_FIELDS.has(field)) return formatDate(text);
    if (field === "tax_rate") return formatPercent(text);
    return text;
  }
  if (Array.isArray(value)) return `${formatCount(value.length)} item${value.length === 1 ? "" : "s"}`;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return JSON.stringify(value);
}

const MARKS = {
  match: { icon: Check, label: "Match", className: "border-tone-success-foreground text-tone-success-foreground" },
  mismatch: { icon: X, label: "Mismatch", className: "border-tone-danger-foreground text-tone-danger-foreground" },
  partial: {
    icon: EqualApproximately,
    label: "Partial match",
    className: "border-tone-warning-foreground text-tone-warning-foreground",
  },
  missing: { icon: Minus, label: "Missing on one side", className: "border-input text-muted-foreground" },
} as const;

function TickMark({ status }: { status: string }) {
  const mark = MARKS[status as keyof typeof MARKS] ?? {
    icon: CircleDashed,
    label: humanize(status),
    className: "border-input text-muted-foreground",
  };
  const Icon = mark.icon;
  return (
    <span
      className={cn("mx-auto flex size-6 items-center justify-center rounded-full border-2 bg-card", mark.className)}
      title={mark.label}
    >
      <Icon aria-hidden="true" className="size-3.5" strokeWidth={3} />
      <span className="sr-only">{mark.label}</span>
    </span>
  );
}

function SideHeading({ side, isCurrent }: { side: CrossVerificationSide; isCurrent: boolean }) {
  return (
    <span className="flex min-w-0 flex-col gap-0.5">
      <span className="flex items-center gap-1.5">
        <Link
          href={`/documents/${side.document_id}`}
          className="truncate font-medium text-foreground underline-offset-2 hover:underline"
          title={side.filename}
        >
          {side.filename}
        </Link>
        {isCurrent ? <Badge variant="neutral">This document</Badge> : null}
      </span>
      <span className="figures text-xs font-normal text-muted-foreground">{side.invoice_number ?? "No invoice number"}</span>
    </span>
  );
}

interface ComparisonResultProps {
  result: CrossVerification;
  /** Marks which side is the document being viewed. */
  currentDocumentId?: string;
  /** Show the overall status line above the table (off when the caller already shows it). */
  showSummary?: boolean;
  className?: string;
}

export function ComparisonResult({ result, currentDocumentId, showSummary = true, className }: ComparisonResultProps) {
  const leftCurrency = currencyOf(result.fields, "left");
  const rightCurrency = currencyOf(result.fields, "right");
  const lines = result.line_items;

  return (
    <div className={cn("flex flex-col gap-3", className)} data-testid="compare-result" data-status={result.status}>
      {showSummary ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
          <MatchBadge status={result.status} />
          <span className="font-medium">{labelFor(CROSS_VERIFICATION_KIND_LABELS, result.kind)}</span>
          <span className="text-muted-foreground">
            {result.trigger === "auto" ? "Checked automatically" : "Compared on request"} on {formatDateTime(result.created_at)}
          </span>
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-lg border bg-card">
        <table className="w-full min-w-[640px] text-sm">
          <caption className="sr-only">
            Field comparison of {result.left.filename} and {result.right.filename}
          </caption>
          <thead className="border-b bg-muted/50 text-left">
            <tr>
              <th scope="col" className="w-40 px-3 py-2 font-medium">
                Field
              </th>
              <th scope="col" className="px-3 py-2 text-right">
                <SideHeading side={result.left} isCurrent={result.left.document_id === currentDocumentId} />
              </th>
              <th scope="col" className="w-14 px-2 py-2 text-center">
                <span className="sr-only">Result</span>
              </th>
              <th scope="col" className="px-3 py-2">
                <SideHeading side={result.right} isCurrent={result.right.document_id === currentDocumentId} />
              </th>
              <th scope="col" className="px-3 py-2 font-medium">
                Note
              </th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {result.fields.map((field) => (
              <tr
                key={field.field}
                data-testid="compare-field-row"
                data-field={field.field}
                data-status={field.status}
                className={cn(field.status === "mismatch" && "bg-tone-danger/50", field.status === "missing" && "bg-muted/40")}
              >
                <th scope="row" className="px-3 py-2 text-left font-medium">
                  {fieldLabel(field.field)}
                </th>
                <td className="figures px-3 py-2 text-right">{formatComparedValue(field.field, field.left, leftCurrency)}</td>
                <td className="px-2 py-1.5">
                  <TickMark status={field.status} />
                </td>
                <td className="figures px-3 py-2">{formatComparedValue(field.field, field.right, rightCurrency)}</td>
                <td className="px-3 py-2 text-muted-foreground">{field.note ?? ""}</td>
              </tr>
            ))}
            <tr data-testid="compare-line-items" data-status={lines.status} className="bg-muted/30">
              <th scope="row" className="px-3 py-2 text-left font-medium">
                Line items
              </th>
              <td className="figures px-3 py-2 text-right">{formatCount(lines.left_count)} lines</td>
              <td className="px-2 py-1.5">
                <TickMark status={lines.status} />
              </td>
              <td className="figures px-3 py-2">{formatCount(lines.right_count)} lines</td>
              <td className="px-3 py-2 text-muted-foreground">
                {formatCount(lines.matched)} matched by amount <MatchBadge status={lines.status} className="ml-1" />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
