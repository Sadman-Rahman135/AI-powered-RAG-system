"use client";

/**
 * Editable view of one extracted invoice record.
 *
 * Each field shows its confidence, source and the verbatim evidence it was read from. Edits are tracked against
 * the record as loaded: Save sends only the fields that changed (and the full line-item list only if any line
 * changed), the server re-validates, and 422 messages are shown next to the fields they concern. Approval is
 * offered only when there are no unsaved edits.
 *
 * The parent should key this component by the record's `id` and `updated_at`, so a saved record (new
 * `updated_at`) starts from fresh state.
 */
import { CheckCheck, CircleAlert, Plus, RotateCcw, TextQuote, Trash2 } from "lucide-react";
import { useId, useMemo, useState, type FormEvent } from "react";

import { ConfidenceBadge } from "@/components/confidence-badge";
import { SEVERITY_ICONS } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Field, FieldError, FieldGroup, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableFooter, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { errorMessage, isApiError } from "@/lib/api";
import { compareDecimals, sumDecimals } from "@/lib/decimal";
import {
  DOCUMENT_TYPE_LABELS,
  SOURCE_LABELS,
  fieldLabel,
  formatDate,
  formatDateTime,
  formatMoney,
  humanize,
  labelFor,
  recordTitle,
} from "@/lib/format";
import type { EditableField, FieldMeta, Invoice, InvoicePatch, Issue, LineItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import { fieldErrorsFromApiError, type FieldErrors } from "@/lib/validation-errors";

type FieldKind = "text" | "date" | "money" | "rate" | "currency" | "document_type";

interface FieldSpec {
  name: EditableField;
  kind: FieldKind;
}

const DETAIL_FIELDS: FieldSpec[] = [
  { name: "invoice_number", kind: "text" },
  { name: "document_type", kind: "document_type" },
  { name: "invoice_date", kind: "date" },
  { name: "due_date", kind: "date" },
  { name: "supplier_name", kind: "text" },
  { name: "customer_name", kind: "text" },
  { name: "currency", kind: "currency" },
];

const AMOUNT_FIELDS: FieldSpec[] = [
  { name: "subtotal", kind: "money" },
  { name: "discount", kind: "money" },
  { name: "shipping", kind: "money" },
  { name: "tax", kind: "money" },
  { name: "tax_rate", kind: "rate" },
  { name: "total", kind: "money" },
];

const ALL_FIELDS = [...DETAIL_FIELDS, ...AMOUNT_FIELDS];
const FIELD_NAMES = new Set<string>(ALL_FIELDS.map((f) => f.name));
const LINE_PARTS = ["description", "quantity", "unit_price", "amount"] as const;
type LinePart = (typeof LINE_PARTS)[number];

const DOCUMENT_TYPE_ITEMS = Object.entries(DOCUMENT_TYPE_LABELS).map(([value, label]) => ({ value, label }));

/** Matches within this absolute difference count as equal (backend `AMOUNT_TOLERANCE`). */
const AMOUNT_TOLERANCE = "0.02";

type Draft = Record<EditableField, string>;

interface LineDraft extends Record<LinePart, string> {
  key: number;
}

let lineKeySeed = 0;
const nextLineKey = () => ++lineKeySeed;

/** DOM id of a field's input, so validation issues elsewhere on the page can focus it. */
export function fieldInputId(invoiceId: string, field: string): string {
  const lineItem = /^line_items\.(\d+)\.(\w+)$/.exec(field);
  if (lineItem) return `invoice-${invoiceId}-line-${lineItem[1]}-${lineItem[2]}`;
  if (field === "line_items") return `invoice-${invoiceId}-line-0-description`;
  return `invoice-${invoiceId}-${field}`;
}

function toDraft(invoice: Invoice): Draft {
  return Object.fromEntries(ALL_FIELDS.map(({ name }) => [name, invoice[name] ?? ""])) as Draft;
}

function toLineDrafts(items: readonly LineItem[]): LineDraft[] {
  return items.map((item) => ({
    key: nextLineKey(),
    description: item.description ?? "",
    quantity: item.quantity ?? "",
    unit_price: item.unit_price ?? "",
    amount: item.amount ?? "",
  }));
}

/** The value that would be stored: trimmed, blank as `null`, currency upper-cased. */
function canonical(name: string, value: string | null | undefined): string | null {
  const trimmed = (value ?? "").trim();
  if (trimmed === "") return null;
  return name === "currency" ? trimmed.toUpperCase() : trimmed;
}

function canonicalLine(line: Record<LinePart, string | null | undefined>): LineItem {
  return {
    description: canonical("description", line.description),
    quantity: canonical("quantity", line.quantity),
    unit_price: canonical("unit_price", line.unit_price),
    amount: canonical("amount", line.amount),
  };
}

const isBlankLine = (line: LineItem) => LINE_PARTS.every((part) => line[part] === null);

const sameLine = (a: LineItem | undefined, b: LineItem | undefined) =>
  a !== undefined && b !== undefined && LINE_PARTS.every((part) => a[part] === b[part]);

function sameLines(a: readonly LineItem[], b: readonly LineItem[]): boolean {
  return a.length === b.length && a.every((line, i) => sameLine(line, b[i]));
}

/** Only what changed: scalar fields that differ, the full line-item list if any line differs, the comment. */
export function buildInvoicePatch(invoice: Invoice, draft: Draft, lines: readonly LineDraft[], comment: string): InvoicePatch {
  const fields: InvoicePatch["fields"] = {};
  for (const { name } of ALL_FIELDS) {
    const next = canonical(name, draft[name]);
    if (next !== canonical(name, invoice[name])) fields[name] = next;
  }
  const nextLines = lines.map(canonicalLine).filter((line) => !isBlankLine(line));
  const originalLines = invoice.line_items.map(canonicalLine);
  return {
    fields,
    line_items: sameLines(nextLines, originalLines) ? null : nextLines,
    comment: comment.trim() || null,
  };
}

interface InvoiceEditorProps {
  invoice: Invoice;
  onSave: (patch: InvoicePatch) => Promise<unknown>;
  onApprove: (comment: string | null) => Promise<unknown>;
}

export function InvoiceEditor({ invoice, onSave, onApprove }: InvoiceEditorProps) {
  const [draft, setDraft] = useState<Draft>(() => toDraft(invoice));
  const [lines, setLines] = useState<LineDraft[]>(() => toLineDrafts(invoice.line_items));
  const [comment, setComment] = useState("");
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [pending, setPending] = useState<"save" | "approve" | null>(null);
  const commentId = useId();

  const original = useMemo(() => toDraft(invoice), [invoice]);
  const originalLines = useMemo(() => invoice.line_items.map(canonicalLine), [invoice.line_items]);
  const patch = buildInvoicePatch(invoice, draft, lines, comment);
  const dirtyFields = new Set(Object.keys(patch.fields));
  const dirtyCount = dirtyFields.size + (patch.line_items ? 1 : 0);
  const dirty = dirtyCount > 0;
  const busy = pending !== null;
  const reviewed = invoice.review_status === "reviewed";

  const issuesByField = useMemo(() => {
    const map = new Map<string, Issue[]>();
    for (const issue of invoice.issues) {
      const key = issue.field && (FIELD_NAMES.has(issue.field) || issue.field.startsWith("line_items")) ? issue.field : "";
      const bucket = key.startsWith("line_items") ? "line_items" : key;
      map.set(bucket, [...(map.get(bucket) ?? []), issue]);
    }
    return map;
  }, [invoice.issues]);

  function clearError(key: string) {
    setFieldErrors((previous) => {
      if (!(key in previous)) return previous;
      const next = { ...previous };
      delete next[key];
      return next;
    });
  }

  function setField(name: EditableField, value: string) {
    setDraft((previous) => ({ ...previous, [name]: name === "currency" ? value.toUpperCase() : value }));
    clearError(name);
  }

  function setLinePart(index: number, part: LinePart, value: string) {
    setLines((previous) => previous.map((line, i) => (i === index ? { ...line, [part]: value } : line)));
    clearError(`line_items.${index}.${part}`);
  }

  /** Adding or removing a line shifts row indices, so per-line server errors no longer point at the right row. */
  function clearLineErrors() {
    setFieldErrors((previous) => {
      const keys = Object.keys(previous).filter((key) => key.startsWith("line_items"));
      if (keys.length === 0) return previous;
      const next = { ...previous };
      for (const key of keys) delete next[key];
      return next;
    });
  }

  function addLine() {
    setLines((previous) => [
      ...previous,
      { key: nextLineKey(), description: "", quantity: "", unit_price: "", amount: "" },
    ]);
    clearLineErrors();
  }

  function removeLine(index: number) {
    setLines((previous) => previous.filter((_, i) => i !== index));
    clearLineErrors();
  }

  function discard() {
    setDraft(toDraft(invoice));
    setLines(toLineDrafts(invoice.line_items));
    setFieldErrors({});
    setFormError(null);
  }

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dirty || busy) return;
    setPending("save");
    setFormError(null);
    setFieldErrors({});
    try {
      await onSave(patch);
      setComment("");
    } catch (error) {
      if (isApiError(error)) setFieldErrors(fieldErrorsFromApiError(error));
      setFormError(errorMessage(error));
    } finally {
      setPending(null);
    }
  }

  async function handleApprove() {
    if (dirty || busy || reviewed) return;
    setPending("approve");
    setFormError(null);
    try {
      await onApprove(comment.trim() || null);
      setComment("");
    } catch (error) {
      setFormError(errorMessage(error));
    } finally {
      setPending(null);
    }
  }

  const rowProps = (spec: FieldSpec) => ({
    spec,
    inputId: fieldInputId(invoice.id, spec.name),
    value: draft[spec.name],
    original: original[spec.name],
    meta: invoice.field_meta[spec.name],
    issues: issuesByField.get(spec.name) ?? [],
    error: fieldErrors[spec.name],
    dirty: dirtyFields.has(spec.name),
    onChange: setField,
  });

  const recordIssues = issuesByField.get("") ?? [];
  const hasFieldErrors = Object.keys(fieldErrors).length > 0;

  return (
    <form
      data-testid="invoice-editor"
      aria-label={`Fields of ${recordTitle(invoice)}`}
      onSubmit={handleSave}
      noValidate
      className="flex flex-col gap-6"
    >
      <fieldset disabled={busy} className="flex min-w-0 flex-col gap-6">
        {recordIssues.length > 0 ? <IssueNotes issues={recordIssues} className="rounded-lg border bg-muted/40 p-3" /> : null}

        <div className="grid gap-x-8 gap-y-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <FieldSet className="min-w-0">
            <FieldLegend variant="label">Details</FieldLegend>
            <FieldGroup className="grid gap-x-4 gap-y-4 sm:grid-cols-2">
              {DETAIL_FIELDS.map((spec) => (
                <FieldRow key={spec.name} {...rowProps(spec)} />
              ))}
            </FieldGroup>
          </FieldSet>
          <FieldSet className="min-w-0">
            <FieldLegend variant="label">Amounts</FieldLegend>
            <FieldGroup className="gap-4">
              {AMOUNT_FIELDS.map((spec) => (
                <FieldRow key={spec.name} {...rowProps(spec)} />
              ))}
            </FieldGroup>
          </FieldSet>
        </div>

        <LineItemsEditor
          invoiceId={invoice.id}
          lines={lines}
          originalLines={originalLines}
          currency={canonical("currency", draft.currency)}
          subtotal={canonical("subtotal", draft.subtotal)}
          issues={issuesByField.get("line_items") ?? []}
          errors={fieldErrors}
          onChange={setLinePart}
          onAdd={addLine}
          onRemove={removeLine}
        />
      </fieldset>

      <div className="flex flex-col gap-4 border-t pt-4">
        {formError ? (
          <Alert variant="destructive">
            <CircleAlert aria-hidden="true" />
            <AlertTitle>{formError}</AlertTitle>
            {hasFieldErrors ? <AlertDescription>Fix the marked fields and save again.</AlertDescription> : null}
          </Alert>
        ) : null}

        <Field>
          <FieldLabel htmlFor={commentId}>Comment (optional)</FieldLabel>
          <Textarea
            id={commentId}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            rows={2}
            disabled={busy}
            placeholder="Why the change, for the audit trail. For example: checked against PO 7781."
          />
        </Field>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground" aria-live="polite">
            {dirty ? `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}` : "No unsaved changes"}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="ghost" onClick={discard} disabled={!dirty || busy}>
              <RotateCcw aria-hidden="true" data-icon="inline-start" />
              Discard changes
            </Button>
            <Button type="submit" variant={dirty ? "default" : "outline"} disabled={!dirty || busy} data-testid="save-invoice">
              {pending === "save" ? <Spinner data-icon="inline-start" /> : null}
              Save changes
            </Button>
            <Button
              type="button"
              variant={dirty ? "outline" : "default"}
              disabled={dirty || busy || reviewed}
              onClick={handleApprove}
              data-testid="approve-invoice"
            >
              {pending === "approve" ? (
                <Spinner data-icon="inline-start" />
              ) : (
                <CheckCheck aria-hidden="true" data-icon="inline-start" />
              )}
              {reviewed ? "Approved" : "Approve"}
            </Button>
          </div>
        </div>
        {reviewed ? (
          <p className="text-sm text-muted-foreground">
            Reviewed by {invoice.reviewed_by ?? "a reviewer"} on {formatDateTime(invoice.reviewed_at)}.
          </p>
        ) : dirty ? (
          <p className="text-sm text-muted-foreground">Save or discard your changes before approving.</p>
        ) : null}
      </div>
    </form>
  );
}

interface FieldRowProps {
  spec: FieldSpec;
  inputId: string;
  value: string;
  original: string;
  meta: FieldMeta | undefined;
  issues: Issue[];
  error: string | undefined;
  dirty: boolean;
  onChange: (name: EditableField, value: string) => void;
}

function FieldRow({ spec, inputId, value, original, meta, issues, error, dirty, onChange }: FieldRowProps) {
  const label = fieldLabel(spec.name);
  const errorId = `${inputId}-error`;
  const notesId = `${inputId}-notes`;
  const describedBy = [error ? errorId : null, dirty || issues.length > 0 ? notesId : null].filter(Boolean).join(" ") || undefined;
  const shared = {
    id: inputId,
    "data-testid": `field-input-${spec.name}`,
    "aria-invalid": error ? true : undefined,
    "aria-describedby": describedBy,
  };

  let control;
  switch (spec.kind) {
    case "document_type":
      control = (
        <Select
          items={DOCUMENT_TYPE_ITEMS}
          value={value}
          onValueChange={(next) => {
            if (typeof next === "string") onChange(spec.name, next);
          }}
        >
          <SelectTrigger {...shared} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {DOCUMENT_TYPE_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      );
      break;
    case "rate":
      control = (
        <InputGroup>
          <InputGroupInput
            {...shared}
            value={value}
            inputMode="decimal"
            autoComplete="off"
            className="figures text-right"
            onChange={(event) => onChange(spec.name, event.target.value)}
          />
          <InputGroupAddon align="inline-end">%</InputGroupAddon>
        </InputGroup>
      );
      break;
    default:
      control = (
        <Input
          {...shared}
          type={spec.kind === "date" ? "date" : "text"}
          value={value}
          inputMode={spec.kind === "money" ? "decimal" : undefined}
          autoComplete="off"
          spellCheck={false}
          maxLength={spec.kind === "currency" ? 3 : undefined}
          className={cn((spec.kind === "money" || spec.kind === "currency" || spec.name === "invoice_number") && "figures", spec.kind === "money" && "text-right")}
          onChange={(event) => onChange(spec.name, event.target.value)}
        />
      );
  }

  const previous = original === "" ? "empty" : spec.kind === "date" ? formatDate(original) : original;

  return (
    <Field
      data-testid="invoice-field"
      data-field={spec.name}
      data-dirty={dirty ? "true" : "false"}
      data-invalid={error ? true : undefined}
      className={cn("gap-1.5 border-l-2 border-transparent pl-2.5", dirty && "border-dirty")}
    >
      <div className="flex min-h-5 flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <FieldLabel htmlFor={inputId}>{label}</FieldLabel>
        <FieldProvenance meta={meta} label={label} />
      </div>
      {control}
      {spec.kind === "date" && value ? (
        // Browsers show date inputs in their own locale order; spell the month out so 03/05 is never ambiguous.
        <p className="text-xs text-muted-foreground">{formatDate(value)}</p>
      ) : null}
      {dirty || issues.length > 0 ? (
        <div id={notesId} className="flex flex-col gap-1">
          {dirty ? (
            <p className="text-xs text-muted-foreground">
              Edited. Was <span className="figures">{previous}</span>
            </p>
          ) : null}
          <IssueNotes issues={issues} />
        </div>
      ) : null}
      {error ? <FieldError id={errorId}>{error}</FieldError> : null}
    </Field>
  );
}

function FieldProvenance({ meta, label }: { meta: FieldMeta | undefined; label: string }) {
  if (!meta) return null;
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-muted-foreground">{labelFor(SOURCE_LABELS, meta.source)}</span>
      <ConfidenceBadge value={meta.confidence} />
      {meta.evidence || meta.notes.length > 0 ? <EvidenceTip meta={meta} label={label} /> : null}
    </div>
  );
}

function EvidenceTip({ meta, label }: { meta: FieldMeta; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Button type="button" variant="ghost" size="icon-xs" aria-label={`Evidence for ${label}`} />}
      >
        <TextQuote aria-hidden="true" />
      </TooltipTrigger>
      <TooltipContent side="top" align="end" className="max-w-sm flex-col items-start gap-1 text-left">
        {meta.evidence ? <p className="figures break-words whitespace-pre-wrap">“{meta.evidence}”</p> : null}
        {meta.page != null ? <p>Page {meta.page}</p> : null}
        {meta.notes.map((note) => (
          <p key={note}>{humanize(note)}</p>
        ))}
      </TooltipContent>
    </Tooltip>
  );
}

const ISSUE_TEXT: Record<string, string> = {
  error: "text-tone-danger-foreground",
  warning: "text-tone-warning-foreground",
  info: "text-tone-info-foreground",
};

function IssueNotes({ issues, className }: { issues: readonly Issue[]; className?: string }) {
  if (issues.length === 0) return null;
  return (
    <ul className={cn("flex flex-col gap-1", className)}>
      {issues.map((issue) => {
        const Icon = SEVERITY_ICONS[issue.severity] ?? SEVERITY_ICONS.info;
        return (
          <li key={issue.id} className={cn("flex items-start gap-1.5 text-xs", ISSUE_TEXT[issue.severity] ?? ISSUE_TEXT.info)}>
            <Icon aria-hidden="true" className="mt-px size-3.5 shrink-0" />
            <span>{issue.message}</span>
          </li>
        );
      })}
    </ul>
  );
}

interface LineItemsEditorProps {
  invoiceId: string;
  lines: readonly LineDraft[];
  originalLines: readonly LineItem[];
  currency: string | null;
  subtotal: string | null;
  issues: readonly Issue[];
  errors: FieldErrors;
  onChange: (index: number, part: LinePart, value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
}

const PART_LABEL: Record<LinePart, string> = {
  description: "description",
  quantity: "quantity",
  unit_price: "unit price",
  amount: "amount",
};

function LineItemsEditor({
  invoiceId,
  lines,
  originalLines,
  currency,
  subtotal,
  issues,
  errors,
  onChange,
  onAdd,
  onRemove,
}: LineItemsEditorProps) {
  const sum = sumDecimals(lines.map((line) => line.amount));
  const comparison = sum !== null && subtotal !== null ? compareDecimals(sum, subtotal, AMOUNT_TOLERANCE) : null;
  const tableError = errors.line_items;

  return (
    <FieldSet className="min-w-0">
      <FieldLegend variant="label">Line items</FieldLegend>
      <IssueNotes issues={issues} />
      <div className="overflow-hidden rounded-lg border">
        <Table className="min-w-[640px]">
          <TableHeader>
            <TableRow className="bg-muted/50 hover:bg-muted/50">
              <TableHead className="w-[44%]">Description</TableHead>
              <TableHead className="text-right">Quantity</TableHead>
              <TableHead className="text-right">Unit price</TableHead>
              <TableHead className="text-right">Amount</TableHead>
              <TableHead className="w-10">
                <span className="sr-only">Remove</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {lines.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="py-4 whitespace-normal text-muted-foreground">
                  No line items were found. Add them if the document lists any.
                </TableCell>
              </TableRow>
            ) : (
              lines.map((line, index) => {
                const changed = !sameLine(canonicalLine(line), originalLines[index]);
                return (
                  <TableRow key={line.key} data-dirty={changed ? "true" : "false"} className={cn(changed && "bg-accent/40")}>
                    {LINE_PARTS.map((part) => {
                      const key = `line_items.${index}.${part}`;
                      const error = errors[key];
                      const id = fieldInputId(invoiceId, key);
                      return (
                        <TableCell key={part} className="align-top">
                          <Input
                            id={id}
                            value={line[part]}
                            aria-label={`Line ${index + 1} ${PART_LABEL[part]}`}
                            aria-invalid={error ? true : undefined}
                            aria-describedby={error ? `${id}-error` : undefined}
                            inputMode={part === "description" ? undefined : "decimal"}
                            autoComplete="off"
                            className={cn(part !== "description" && "figures text-right")}
                            onChange={(event) => onChange(index, part, event.target.value)}
                          />
                          {error ? (
                            <p id={`${id}-error`} role="alert" className="mt-1 text-xs whitespace-normal text-destructive">
                              {error}
                            </p>
                          ) : null}
                        </TableCell>
                      );
                    })}
                    <TableCell className="align-top">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Remove line ${index + 1}`}
                        onClick={() => onRemove(index)}
                      >
                        <Trash2 aria-hidden="true" />
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
          <TableFooter>
            <TableRow className="hover:bg-transparent">
              <TableCell colSpan={3} className="text-right text-muted-foreground">
                Sum of amounts
              </TableCell>
              <TableCell className="figures text-right">{formatMoney(sum, currency)}</TableCell>
              <TableCell />
            </TableRow>
          </TableFooter>
        </Table>
      </div>
      {comparison !== null && comparison !== 0 ? (
        <p className="text-xs text-tone-warning-foreground">
          The line amounts add up to {formatMoney(sum, currency)}, not the subtotal of {formatMoney(subtotal, currency)}.
        </p>
      ) : null}
      {tableError ? (
        <p role="alert" className="text-xs text-destructive">
          {tableError}
        </p>
      ) : null}
      <div>
        <Button type="button" variant="outline" size="sm" onClick={onAdd}>
          <Plus aria-hidden="true" data-icon="inline-start" />
          Add line
        </Button>
      </div>
    </FieldSet>
  );
}
