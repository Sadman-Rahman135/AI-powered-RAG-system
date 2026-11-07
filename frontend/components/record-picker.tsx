"use client";

/** Find and pick one invoice record or document to compare. */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { SearchIcon, X } from "lucide-react";
import { useId, useState } from "react";

import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage } from "@/lib/api";
import { DOCUMENT_STATUS_LABELS, FILE_KIND_LABELS, formatCount, formatDate, formatMoney, labelFor } from "@/lib/format";
import { useDebouncedValue } from "@/lib/hooks";
import { queryKeys } from "@/lib/queries";
import type { Document, Invoice } from "@/lib/types";
import { cn } from "@/lib/utils";

export type PickerKind = "invoices" | "documents";

interface PickerOption {
  id: string;
  title: string;
  subtitle: string;
  detail: string;
  mono: boolean;
}

function invoiceOption(invoice: Invoice): PickerOption {
  return {
    id: invoice.id,
    title: invoice.invoice_number ?? "No invoice number",
    subtitle: invoice.supplier_name ?? "Unknown supplier",
    detail: [formatMoney(invoice.total, invoice.currency), formatDate(invoice.invoice_date), invoice.filename].join(", "),
    mono: invoice.invoice_number !== null,
  };
}

function documentOption(document: Document): PickerOption {
  return {
    id: document.id,
    title: document.filename,
    subtitle: `${labelFor(FILE_KIND_LABELS, document.file_kind)}, ${formatCount(document.invoice_count)} ${
      document.invoice_count === 1 ? "record" : "records"
    }`,
    detail: labelFor(DOCUMENT_STATUS_LABELS, document.status),
    mono: false,
  };
}

function useOptions(kind: PickerKind, query: string) {
  return useQuery({
    queryKey: queryKeys.picker(kind, query),
    queryFn: async ({ signal }) =>
      kind === "invoices"
        ? (await api.listInvoices({ q: query || undefined, limit: 8 }, signal)).items.map(invoiceOption)
        : (await api.listDocuments({ q: query || undefined, limit: 8, sort: "-created_at" }, signal)).items.map(documentOption),
    placeholderData: keepPreviousData,
  });
}

function useSelected(kind: PickerKind, id: string | null) {
  return useQuery({
    queryKey: ["picker-selected", kind, id],
    queryFn: async ({ signal }) =>
      kind === "invoices"
        ? invoiceOption(await api.getInvoice(id as string, signal))
        : documentOption(await api.getDocument(id as string, signal)),
    enabled: id !== null,
  });
}

interface RecordPickerProps {
  side: "left" | "right";
  label: string;
  kind: PickerKind;
  value: string | null;
  /** The other side's selection, which cannot be picked again. */
  excludeId: string | null;
  onChange: (id: string | null) => void;
}

export function RecordPicker({ side, label, kind, value, excludeId, onChange }: RecordPickerProps) {
  const [text, setText] = useState("");
  const query = useDebouncedValue(text.trim(), 250);
  const options = useOptions(kind, query);
  const selected = useSelected(kind, value);
  const id = useId();
  const noun = kind === "invoices" ? "record" : "document";

  return (
    <section
      aria-labelledby={`${id}-title`}
      data-testid={`compare-${side}`}
      className="flex min-w-0 flex-col gap-3 rounded-lg border bg-card p-4"
    >
      <h2 id={`${id}-title`} className="font-semibold">
        {label}
      </h2>

      {value ? (
        selected.isPending ? (
          <Skeleton className="h-16 w-full" />
        ) : selected.data ? (
          <div className="flex items-start justify-between gap-3 rounded-md border border-primary/40 bg-accent/40 p-3" data-testid={`compare-${side}-selected`}>
            <OptionText option={selected.data} />
            <Button type="button" variant="ghost" size="icon-sm" aria-label={`Choose a different ${noun}`} onClick={() => onChange(null)}>
              <X aria-hidden="true" />
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-2 text-sm">
            <p className="text-destructive">This {noun} could not be loaded: {errorMessage(selected.error)}</p>
            <Button type="button" variant="outline" size="sm" onClick={() => onChange(null)}>
              Choose another {noun}
            </Button>
          </div>
        )
      ) : (
        <>
          <InputGroup>
            <InputGroupAddon>
              <SearchIcon aria-hidden="true" />
            </InputGroupAddon>
            <InputGroupInput
              type="search"
              value={text}
              onChange={(event) => setText(event.target.value)}
              aria-label={`Find the ${label.toLowerCase()}`}
              placeholder={kind === "invoices" ? "Invoice number, supplier or file name" : "File name"}
              data-testid={`compare-${side}-search`}
            />
          </InputGroup>
          {options.isError && !options.data ? (
            <p className="text-sm text-destructive">{errorMessage(options.error)}</p>
          ) : options.isPending ? (
            <Skeleton className="h-32 w-full" />
          ) : options.data.length === 0 ? (
            <p className="text-sm text-muted-foreground">Nothing matches “{query}”.</p>
          ) : (
            <ul aria-label={`Matching ${noun}s`} className={cn("flex flex-col divide-y rounded-md border", options.isPlaceholderData && "opacity-60")}>
              {options.data.map((option) => {
                const taken = option.id === excludeId;
                return (
                  <li key={option.id}>
                    <button
                      type="button"
                      disabled={taken}
                      onClick={() => onChange(option.id)}
                      data-testid="compare-option"
                      className="flex w-full items-start justify-between gap-3 px-3 py-2 text-left outline-none hover:bg-muted focus-visible:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <OptionText option={option} />
                      {taken ? <span className="shrink-0 text-xs text-muted-foreground">Already chosen</span> : null}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </>
      )}
    </section>
  );
}

function OptionText({ option }: { option: PickerOption }) {
  return (
    <span className="flex min-w-0 flex-col">
      <span className={cn("truncate font-medium", option.mono && "figures")}>{option.title}</span>
      <span className="truncate text-sm text-muted-foreground">{option.subtitle}</span>
      <span className="truncate text-xs text-muted-foreground">{option.detail}</span>
    </span>
  );
}
