"use client";

/**
 * All documents with status, live progress, record and issue counts. Filters by status, file type and name;
 * polls while anything on the page is queued or processing and announces documents as they finish.
 */
import { FileSearch, FileStack, OctagonX, RotateCw, SearchIcon, Trash2, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

import { PaginationBar } from "@/components/pagination-bar";
import { CompactProgress } from "@/components/processing-progress";
import { EmptyState, QueryError } from "@/components/query-state";
import { StatusBadge } from "@/components/status-badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "@/components/ui/toast";
import { type DocumentListParams, errorMessage } from "@/lib/api";
import {
  DOCUMENT_STATUS_LABELS,
  FILE_KIND_LABELS,
  formatCount,
  formatDateTime,
  formatDuration,
  formatRelativeTime,
  humanize,
  labelFor,
} from "@/lib/format";
import { useAnnouncer, useDebouncedValue } from "@/lib/hooks";
import { useDeleteDocument, useDocuments, useReprocessDocument } from "@/lib/queries";
import { isActiveDocument } from "@/lib/status";
import type { Document, DocumentStatus, FileKind } from "@/lib/types";

const PAGE_SIZE = 20;
const ALL = "all";

const STATUS_ITEMS = [
  { value: ALL, label: "All statuses" },
  ...Object.entries(DOCUMENT_STATUS_LABELS).map(([value, label]) => ({ value, label })),
];
const KIND_ITEMS = [
  { value: ALL, label: "All types" },
  ...Object.entries(FILE_KIND_LABELS).map(([value, label]) => ({ value, label })),
];

/** Announce documents that settle (completed, needs review, failed) between two polls. */
function useAnnounceSettled(items: readonly Document[] | undefined, announce: (message: string) => void) {
  const previous = useRef(new Map<string, DocumentStatus>());
  useEffect(() => {
    if (!items) return;
    const settled: string[] = [];
    for (const document of items) {
      const before = previous.current.get(document.id);
      if (before && before !== document.status && !isActiveDocument(document)) {
        settled.push(`${document.filename}: ${labelFor(DOCUMENT_STATUS_LABELS, document.status).toLowerCase()}`);
      }
      previous.current.set(document.id, document.status);
    }
    if (settled.length > 3) announce(`${settled.length} documents finished processing.`);
    else if (settled.length > 0) announce(`Finished processing. ${settled.join(". ")}.`);
  }, [items, announce]);
}

export function DocumentsTable() {
  const [status, setStatus] = useState<DocumentStatus | "">("");
  const [kind, setKind] = useState<FileKind | "">("");
  const [name, setName] = useState("");
  const [page, setPage] = useState(1);
  const [deletion, setDeletion] = useState<{ document: Document; open: boolean } | null>(null);
  const q = useDebouncedValue(name.trim(), 300);
  const id = useId();
  const { regionRef, announce } = useAnnouncer();

  const params: DocumentListParams = {
    status,
    file_kind: kind,
    q,
    limit: PAGE_SIZE,
    offset: (page - 1) * PAGE_SIZE,
    sort: "-created_at",
  };
  const { data, isPending, isError, error, refetch, isFetching } = useDocuments(params);
  const reprocess = useReprocessDocument();
  useAnnounceSettled(data?.items, announce);

  // A transient confirmation must not survive the page being hidden by client-side navigation.
  useLayoutEffect(() => () => setDeletion(null), []);

  const filtered = status !== "" || kind !== "" || q !== "";

  function reprocessDocument(document: Document) {
    reprocess.mutate(document.id, {
      onSuccess: () => toast.add({ title: "Queued for reprocessing", description: document.filename, type: "success" }),
      onError: (err) => toast.add({ title: "Could not reprocess", description: errorMessage(err), type: "error" }),
    });
  }

  return (
    <section aria-labelledby={`${id}-title`} className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 id={`${id}-title`} className="text-base font-semibold">
          All documents
        </h2>
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          {isFetching && !isPending ? <Spinner className="size-3" /> : null}
          Updates every 2 seconds while documents are processing
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Field className="w-44">
          <FieldLabel htmlFor={`${id}-status`}>Status</FieldLabel>
          <Select
            items={STATUS_ITEMS}
            value={status || ALL}
            onValueChange={(next) => {
              setStatus(next && next !== ALL ? (next as DocumentStatus) : "");
              setPage(1);
            }}
          >
            <SelectTrigger id={`${id}-status`} className="w-full" data-testid="documents-filter-status">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {STATUS_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>
        <Field className="w-40">
          <FieldLabel htmlFor={`${id}-kind`}>File type</FieldLabel>
          <Select
            items={KIND_ITEMS}
            value={kind || ALL}
            onValueChange={(next) => {
              setKind(next && next !== ALL ? (next as FileKind) : "");
              setPage(1);
            }}
          >
            <SelectTrigger id={`${id}-kind`} className="w-full" data-testid="documents-filter-kind">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {KIND_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>
        <Field className="w-72">
          <FieldLabel htmlFor={`${id}-name`}>File name</FieldLabel>
          <InputGroup>
            <InputGroupAddon>
              <SearchIcon aria-hidden="true" />
            </InputGroupAddon>
            <InputGroupInput
              id={`${id}-name`}
              type="search"
              value={name}
              placeholder="Part of a file name"
              data-testid="documents-filter-name"
              onChange={(event) => {
                setName(event.target.value);
                setPage(1);
              }}
            />
          </InputGroup>
        </Field>
      </div>

      <div ref={regionRef} aria-live="polite" role="status" className="sr-only" data-testid="processing-status" />

      {isError && !data ? (
        <QueryError error={error} title="Documents could not be loaded" onRetry={() => void refetch()} />
      ) : isPending ? (
        <TableSkeleton />
      ) : data.items.length === 0 ? (
        filtered ? (
          <EmptyState icon={FileSearch} title="No documents match these filters">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setStatus("");
                setKind("");
                setName("");
                setPage(1);
              }}
            >
              Clear filters
            </Button>
          </EmptyState>
        ) : (
          <EmptyState
            icon={FileStack}
            title="No documents yet"
            description="Upload invoices, receipts or spreadsheets above. Each file is read, its invoice fields are extracted and checked, and anything doubtful lands in the review queue."
          />
        )
      ) : (
        <>
          <div className="overflow-hidden rounded-lg border bg-card">
            <Table data-testid="documents-table">
              <TableHeader>
                <TableRow className="bg-muted/50 hover:bg-muted/50">
                  <TableHead className="w-[30%] min-w-56">File</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead className="text-right">Records</TableHead>
                  <TableHead>Checks</TableHead>
                  <TableHead className="hidden text-right lg:table-cell">Pages</TableHead>
                  <TableHead className="hidden lg:table-cell">Uploaded</TableHead>
                  <TableHead>
                    <span className="sr-only">Actions</span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.items.map((document) => (
                  <DocumentRow
                    key={document.id}
                    document={document}
                    reprocessing={reprocess.isPending && reprocess.variables === document.id}
                    onReprocess={reprocessDocument}
                    onDelete={(doc) => setDeletion({ document: doc, open: true })}
                  />
                ))}
              </TableBody>
            </Table>
          </div>
          <PaginationBar page={page} pageSize={PAGE_SIZE} total={data.total} onPageChange={setPage} noun="documents" />
        </>
      )}

      <DeleteDocumentDialog
        document={deletion?.document ?? null}
        open={deletion?.open ?? false}
        onClose={() => setDeletion((current) => (current ? { ...current, open: false } : current))}
      />
    </section>
  );
}

interface DocumentRowProps {
  document: Document;
  reprocessing: boolean;
  onReprocess: (document: Document) => void;
  onDelete: (document: Document) => void;
}

function DocumentRow({ document, reprocessing, onReprocess, onDelete }: DocumentRowProps) {
  const active = isActiveDocument(document);
  const { error: errors, warning: warnings } = document.issue_counts;

  return (
    <TableRow data-testid="documents-row" data-status={document.status} data-document-id={document.id}>
      <TableCell className="max-w-0 min-w-56 whitespace-normal">
        <Link
          href={`/documents/${document.id}`}
          className="block truncate font-medium text-foreground underline-offset-2 hover:underline"
          title={document.filename}
        >
          {document.filename}
        </Link>
        {document.status === "failed" && (document.error_message || document.error_code) ? (
          <p className="truncate text-xs text-destructive" title={document.error_message ?? undefined}>
            {document.error_message ?? humanize(document.error_code ?? "")}
          </p>
        ) : null}
        {document.duplicate_of_id ? (
          <p className="text-xs text-tone-warning-foreground">
            Same file as{" "}
            <Link href={`/documents/${document.duplicate_of_id}`} className="underline underline-offset-2">
              an earlier upload
            </Link>
          </p>
        ) : null}
      </TableCell>
      <TableCell className="text-muted-foreground">{labelFor(FILE_KIND_LABELS, document.file_kind)}</TableCell>
      <TableCell>
        <StatusBadge status={document.status} />
      </TableCell>
      <TableCell>
        {active ? (
          <CompactProgress status={document.status} job={document.job} />
        ) : (
          <span className="figures text-xs text-muted-foreground">
            {document.processing_ms != null ? formatDuration(document.processing_ms) : "—"}
          </span>
        )}
      </TableCell>
      <TableCell className="figures text-right">{active ? "—" : formatCount(document.invoice_count)}</TableCell>
      <TableCell>
        <IssueCounts errors={errors} warnings={warnings} checked={!active && document.status !== "failed" && document.invoice_count > 0} />
      </TableCell>
      <TableCell className="figures hidden text-right lg:table-cell">
        {document.page_count != null ? formatCount(document.page_count) : "—"}
        {document.ocr_page_count ? (
          <span className="block text-xs text-muted-foreground">{formatCount(document.ocr_page_count)} OCR</span>
        ) : null}
      </TableCell>
      <TableCell className="hidden text-muted-foreground lg:table-cell">
        <time dateTime={document.created_at} title={formatDateTime(document.created_at)}>
          {formatRelativeTime(document.created_at)}
        </time>
      </TableCell>
      <TableCell className="text-right">
        <div className="flex items-center justify-end gap-1">
          {document.status === "failed" ? (
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={reprocessing}
              onClick={() => onReprocess(document)}
              data-testid="reprocess-document"
            >
              {reprocessing ? <Spinner data-icon="inline-start" /> : <RotateCw aria-hidden="true" data-icon="inline-start" />}
              Reprocess
            </Button>
          ) : null}
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={`Delete ${document.filename}`}
            onClick={() => onDelete(document)}
            data-testid="delete-document"
          >
            <Trash2 aria-hidden="true" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}

/** Error and warning counts; "—" when no checks ran (still processing, failed, or no records found). */
function IssueCounts({ errors, warnings, checked }: { errors: number; warnings: number; checked: boolean }) {
  if (errors === 0 && warnings === 0) {
    return checked ? (
      <span className="text-xs text-tone-success-foreground">All passed</span>
    ) : (
      <span className="text-xs text-muted-foreground">—</span>
    );
  }
  return (
    <span className="flex items-center gap-3 text-xs">
      {errors > 0 ? (
        <span className="figures inline-flex items-center gap-1 text-tone-danger-foreground">
          <OctagonX aria-hidden="true" className="size-3.5" />
          {formatCount(errors)}
          <span className="sr-only"> {errors === 1 ? "error" : "errors"}</span>
        </span>
      ) : null}
      {warnings > 0 ? (
        <span className="figures inline-flex items-center gap-1 text-tone-warning-foreground">
          <TriangleAlert aria-hidden="true" className="size-3.5" />
          {formatCount(warnings)}
          <span className="sr-only"> {warnings === 1 ? "warning" : "warnings"}</span>
        </span>
      ) : null}
    </span>
  );
}

function TableSkeleton() {
  return (
    <div className="flex flex-col gap-px overflow-hidden rounded-lg border bg-border" aria-busy="true" aria-label="Loading documents">
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="flex items-center gap-4 bg-card px-3 py-3">
          <Skeleton className="h-4 w-1/4" />
          <Skeleton className="h-4 w-12" />
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-2 w-32" />
          <Skeleton className="ml-auto h-4 w-16" />
        </div>
      ))}
    </div>
  );
}

interface DeleteDocumentDialogProps {
  document: Document | null;
  open: boolean;
  onClose: () => void;
}

function DeleteDocumentDialog({ document, open, onClose }: DeleteDocumentDialogProps) {
  const remove = useDeleteDocument();

  function confirm() {
    if (!document) return;
    remove.mutate(document.id, {
      onSuccess: () => {
        toast.add({ title: "Document deleted", description: document.filename, type: "success" });
        onClose();
      },
      onError: (err) => toast.add({ title: "Could not delete the document", description: errorMessage(err), type: "error" }),
    });
  }

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next && !remove.isPending) onClose();
      }}
    >
      <AlertDialogContent data-testid="delete-dialog">
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {document?.filename ?? "this document"}?</AlertDialogTitle>
          <AlertDialogDescription>
            The original file, its extracted records, corrections and search entries are removed. This cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={remove.isPending}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={remove.isPending}
            onClick={confirm}
            data-testid="confirm-delete"
          >
            {remove.isPending ? <Spinner data-icon="inline-start" /> : null}
            Delete document
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
