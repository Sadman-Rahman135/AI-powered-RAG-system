"use client";

/** Processing history (every stage transition) and the audit trail of manual corrections. */
import { History, PencilLine } from "lucide-react";
import { useId, useState } from "react";

import { EmptyState, QueryError } from "@/components/query-state";
import { Badge } from "@/components/ui/badge";
import { Field, FieldLabel } from "@/components/ui/field";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  EMPTY,
  fieldLabel,
  formatCount,
  formatDateTime,
  formatDuration,
  formatRecordLocation,
  humanize,
  recordTitle,
} from "@/lib/format";
import { useCorrections, useDocumentEvents } from "@/lib/queries";
import { isActiveDocument } from "@/lib/status";
import type { DocumentDetail, ProcessingEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

const LEVEL_TONE = { info: "neutral", warning: "warning", error: "danger" } as const;

function formatAuditValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "empty";
  if (Array.isArray(value)) return `${formatCount(value.length)} line item${value.length === 1 ? "" : "s"}`;
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function EventData({ data }: { data: ProcessingEvent["data"] }) {
  const entries = Object.entries(data ?? {}).filter(
    ([, value]) => typeof value === "string" || typeof value === "number" || typeof value === "boolean",
  );
  if (entries.length === 0) return null;
  return (
    <dl className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
      {entries.map(([key, value]) => {
        // Timings arrive as milliseconds (`ms`, `parse_ms`, ...).
        const isDuration = (key === "ms" || key.endsWith("_ms")) && typeof value === "number";
        const label = key === "ms" ? "Time" : humanize(isDuration ? key.slice(0, -3) : key);
        return (
          <div key={key} className="flex gap-1">
            <dt>{label}:</dt>
            <dd className="figures">{isDuration ? formatDuration(value) : String(value)}</dd>
          </div>
        );
      })}
    </dl>
  );
}

function EventsTimeline({ document }: { document: DocumentDetail }) {
  const { data, isPending, isError, error, refetch } = useDocumentEvents(document.id, isActiveDocument(document));

  if (isError && !data) {
    return <QueryError error={error} title="The processing history could not be loaded" onRetry={() => void refetch()} />;
  }
  if (isPending) return <Skeleton className="h-48 w-full" />;
  if (data.items.length === 0) {
    return <EmptyState icon={History} title="No processing history yet" description="Each stage is recorded here as it runs." />;
  }

  return (
    <ol className="flex flex-col" data-testid="events-timeline">
      {data.items.map((event, index) => (
        <li key={event.id} className="relative flex gap-3 pb-4 last:pb-0">
          {index < data.items.length - 1 ? (
            <span aria-hidden="true" className="absolute top-4 left-[5px] h-full w-px bg-border" />
          ) : null}
          <span
            aria-hidden="true"
            className={cn(
              "relative mt-1.5 size-[11px] shrink-0 rounded-full border-2 bg-card",
              event.level === "error" ? "border-tone-danger-foreground" : event.level === "warning" ? "border-tone-warning-foreground" : "border-primary",
            )}
          />
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <time dateTime={event.created_at} className="figures text-xs text-muted-foreground">
                {formatDateTime(event.created_at)}
              </time>
              <Badge variant={LEVEL_TONE[event.level] ?? "neutral"}>{humanize(event.stage)}</Badge>
            </div>
            <p>{event.message}</p>
            <EventData data={event.data} />
          </div>
        </li>
      ))}
    </ol>
  );
}

function CorrectionsAudit({ document }: { document: DocumentDetail }) {
  const [selected, setSelected] = useState<string | null>(document.invoices[0]?.id ?? null);
  const invoiceId = document.invoices.some((invoice) => invoice.id === selected) ? selected : (document.invoices[0]?.id ?? null);
  const { data, isPending, isError, error, refetch } = useCorrections(invoiceId);
  const id = useId();
  const items = document.invoices.map((invoice, index) => ({
    value: invoice.id,
    label: `${index + 1}. ${recordTitle(invoice)}${formatRecordLocation(invoice) ? `, ${formatRecordLocation(invoice)}` : ""}`,
  }));

  if (document.invoices.length === 0) {
    return (
      <EmptyState
        icon={PencilLine}
        title="No records to correct"
        description="Corrections appear here once records have been extracted and edited."
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {document.invoices.length > 1 ? (
        <Field className="max-w-sm">
          <FieldLabel htmlFor={`${id}-record`}>Record</FieldLabel>
          <Select items={items} value={invoiceId} onValueChange={(next) => typeof next === "string" && setSelected(next)}>
            <SelectTrigger id={`${id}-record`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {items.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>
      ) : null}
      {isError && !data ? (
        <QueryError error={error} title="Corrections could not be loaded" onRetry={() => void refetch()} />
      ) : isPending ? (
        <Skeleton className="h-32 w-full" />
      ) : data.items.length === 0 ? (
        <EmptyState
          icon={PencilLine}
          title="No corrections yet"
          description="Saved changes are listed here with who made them, when and why."
        />
      ) : (
        <div className="overflow-hidden rounded-lg border bg-card">
          <Table data-testid="corrections-table">
            <TableHeader>
              <TableRow className="bg-muted/50 hover:bg-muted/50">
                <TableHead>Field</TableHead>
                <TableHead>Before</TableHead>
                <TableHead>After</TableHead>
                <TableHead>Comment</TableHead>
                <TableHead>By</TableHead>
                <TableHead>When</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((correction) => (
                <TableRow key={correction.id}>
                  <TableCell className="font-medium">{fieldLabel(correction.field)}</TableCell>
                  <TableCell className="figures text-muted-foreground line-through decoration-muted-foreground/60">
                    {formatAuditValue(correction.old_value)}
                  </TableCell>
                  <TableCell className="figures">{formatAuditValue(correction.new_value)}</TableCell>
                  <TableCell className="max-w-64 whitespace-normal">{correction.comment || EMPTY}</TableCell>
                  <TableCell>{correction.corrected_by}</TableCell>
                  <TableCell className="text-muted-foreground">
                    <time dateTime={correction.created_at}>{formatDateTime(correction.created_at)}</time>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

export function DocumentHistory({ document }: { document: DocumentDetail }) {
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <section aria-labelledby="history-processing" className="flex flex-col gap-3">
        <h2 id="history-processing" className="font-semibold">
          Processing history
        </h2>
        <EventsTimeline document={document} />
      </section>
      <section aria-labelledby="history-corrections" className="flex flex-col gap-3">
        <h2 id="history-corrections" className="font-semibold">
          Corrections
        </h2>
        <CorrectionsAudit document={document} />
      </section>
    </div>
  );
}
