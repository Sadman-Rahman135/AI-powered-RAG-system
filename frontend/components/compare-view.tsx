"use client";

/**
 * Compare any two invoice records (or two documents) field by field — for example a PDF invoice against its row
 * in an Excel ledger. The selection lives in the URL (`type`, `left`, `right`) so other pages can link here with
 * one side chosen.
 */
import { ArrowLeftRight, GitCompareArrows } from "lucide-react";
import { useSearchParams } from "next/navigation";

import { ComparisonResult } from "@/components/comparison-result";
import { PageHeader } from "@/components/page-header";
import { QueryError } from "@/components/query-state";
import { RecordPicker, type PickerKind } from "@/components/record-picker";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useCrossVerify } from "@/lib/queries";

const KINDS: Array<{ value: PickerKind; label: string }> = [
  { value: "invoices", label: "Invoice records" },
  { value: "documents", label: "Documents" },
];

export function CompareView() {
  const searchParams = useSearchParams();
  const kind: PickerKind = searchParams.get("type") === "documents" ? "documents" : "invoices";
  const left = searchParams.get("left");
  const right = searchParams.get("right");
  const compare = useCrossVerify();
  const noun = kind === "invoices" ? "record" : "document";

  function update(changes: Partial<Record<"type" | "left" | "right", string | null>>) {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(changes)) {
      if (value) params.set(key, value);
      else params.delete(key);
    }
    compare.reset();
    window.history.replaceState(null, "", `/compare?${params.toString()}`);
  }

  function run() {
    if (!left || !right) return;
    compare.mutate(
      kind === "invoices"
        ? { left_invoice_id: left, right_invoice_id: right }
        : { left_document_id: left, right_document_id: right },
    );
  }

  const ready = left !== null && right !== null && left !== right;

  return (
    <>
      <PageHeader
        title="Compare"
        description="Check two invoices against each other field by field, for example a PDF invoice against its row in a spreadsheet ledger."
      />

      <div className="flex flex-wrap items-center gap-3">
        <span id="compare-kind-label" className="text-sm font-medium">
          Compare
        </span>
        <ToggleGroup
          aria-labelledby="compare-kind-label"
          data-testid="compare-kind"
          variant="outline"
          size="sm"
          spacing={0}
          value={[kind]}
          onValueChange={(next) => {
            const value = next[0] as PickerKind | undefined;
            if (value && value !== kind) update({ type: value, left: null, right: null });
          }}
        >
          {KINDS.map((option) => (
            <ToggleGroupItem key={option.value} value={option.value}>
              {option.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <div className="grid items-start gap-4 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
        <RecordPicker
          key={`left-${kind}`}
          side="left"
          label={`First ${noun}`}
          kind={kind}
          value={left}
          excludeId={right}
          onChange={(id) => update({ left: id })}
        />
        <ArrowLeftRight aria-hidden="true" className="hidden size-5 self-center text-muted-foreground md:block" />
        <RecordPicker
          key={`right-${kind}`}
          side="right"
          label={`Second ${noun}`}
          kind={kind}
          value={right}
          excludeId={left}
          onChange={(id) => update({ right: id })}
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" disabled={!ready || compare.isPending} onClick={run} data-testid="compare-submit">
          {compare.isPending ? <Spinner data-icon="inline-start" /> : <GitCompareArrows aria-hidden="true" data-icon="inline-start" />}
          Compare {noun}s
        </Button>
        {!ready ? (
          <p className="text-sm text-muted-foreground">
            {left && left === right ? `Choose two different ${noun}s.` : `Choose a ${noun} on each side.`}
          </p>
        ) : null}
      </div>

      {compare.isError ? <QueryError error={compare.error} title="The comparison failed" onRetry={run} /> : null}
      {compare.data ? (
        <section aria-label="Comparison result" aria-live="polite">
          <ComparisonResult result={compare.data} />
        </section>
      ) : null}
    </>
  );
}
