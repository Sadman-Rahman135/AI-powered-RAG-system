"use client";

/**
 * One document: header with download and reprocess, live stage progress while processing, and tabs for the
 * extracted records (editable), validation issues, page text, history and cross-verification.
 *
 * URL parameters (read once when the view mounts): `tab`, `invoice` (scroll to that record) and `page` (open the
 * page text there).
 */
import { ChevronLeft, CircleAlert, Download, FileQuestion, Inbox, RotateCw } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { flushSync } from "react-dom";

import { CrossVerificationPanel } from "@/components/cross-verification-panel";
import { DocumentHistory } from "@/components/document-history";
import { fieldInputId } from "@/components/invoice-editor";
import { InvoiceRecordCard } from "@/components/invoice-record-card";
import { IssuesList } from "@/components/issues-list";
import { PageHeader } from "@/components/page-header";
import { PageTextViewer } from "@/components/page-text-viewer";
import { ProcessingProgress } from "@/components/processing-progress";
import { EmptyState, QueryError } from "@/components/query-state";
import { ReprocessDialog } from "@/components/reprocess-dialog";
import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/components/ui/toast";
import { api, errorMessage, isApiError } from "@/lib/api";
import {
  EMPTY,
  FILE_KIND_LABELS,
  formatBytes,
  formatCount,
  formatDateTime,
  formatDuration,
  humanize,
  labelFor,
  recordTitle,
} from "@/lib/format";
import { useDocument, useReprocessDocument } from "@/lib/queries";
import { isActiveDocument } from "@/lib/status";
import type { DocumentDetail, Issue } from "@/lib/types";

const TABS = ["data", "validation", "text", "history", "cross"] as const;
type Tab = (typeof TABS)[number];

const isTab = (value: string | null): value is Tab => value !== null && (TABS as readonly string[]).includes(value);

export function DocumentDetailSkeleton() {
  return (
    <div className="flex flex-col gap-6" aria-busy="true" aria-label="Loading document">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-7 w-80" />
      </div>
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-9 w-[28rem]" />
      <Skeleton className="h-96 w-full" />
    </div>
  );
}

/** Reads the URL and keys the view by it, so following another link to the same document starts fresh. */
export function DocumentDetailRoute({ id }: { id: string }) {
  const searchParams = useSearchParams();
  const tab = searchParams.get("tab");
  const page = Number.parseInt(searchParams.get("page") ?? "", 10);
  return (
    <DocumentDetailView
      key={`${id}?${searchParams.toString()}`}
      id={id}
      initialTab={isTab(tab) ? tab : searchParams.get("page") ? "text" : "data"}
      focusInvoiceId={searchParams.get("invoice")}
      initialPage={Number.isFinite(page) && page > 0 ? page : 1}
    />
  );
}

interface DocumentDetailViewProps {
  id: string;
  initialTab: Tab;
  focusInvoiceId: string | null;
  initialPage: number;
}

function DocumentDetailView({ id, initialTab, focusInvoiceId, initialPage }: DocumentDetailViewProps) {
  const { data: doc, isPending, isError, error, refetch } = useDocument(id);
  const [tab, setTab] = useState<Tab>(initialTab);
  const loaded = doc !== undefined;

  useEffect(() => {
    if (!loaded || !focusInvoiceId) return;
    window.document.getElementById(`record-${focusInvoiceId}`)?.scrollIntoView({ block: "start" });
  }, [loaded, focusInvoiceId]);

  if (isPending) return <DocumentDetailSkeleton />;
  if (isError && !doc) {
    if (isApiError(error) && error.status === 404) {
      return (
        <EmptyState
          icon={FileQuestion}
          title="This document does not exist"
          description="It may have been deleted. Go back to the document list to find another one."
        >
          <Link href="/" className={buttonVariants({ variant: "outline" })}>
            Back to documents
          </Link>
        </EmptyState>
      );
    }
    return <QueryError error={error} title="The document could not be loaded" onRetry={() => void refetch()} />;
  }

  const allIssues = [...doc.document_issues, ...doc.invoices.flatMap((invoice) => invoice.issues)];
  const errorCount = allIssues.filter((issue) => issue.severity === "error").length;
  const active = isActiveDocument(doc);

  const recordLabel = (invoiceId: string | null): string | null => {
    if (!invoiceId) return "Whole document";
    const index = doc.invoices.findIndex((invoice) => invoice.id === invoiceId);
    if (index === -1) return null;
    const title = recordTitle(doc.invoices[index]);
    return doc.invoices.length > 1 ? `Record ${index + 1}: ${title}` : title;
  };

  const goToField = (issue: Issue) => {
    flushSync(() => setTab("data"));
    const target =
      (issue.invoice_id && issue.field && window.document.getElementById(fieldInputId(issue.invoice_id, issue.field))) ||
      (issue.invoice_id && window.document.getElementById(`record-${issue.invoice_id}`));
    if (target instanceof HTMLElement) {
      target.scrollIntoView({ block: "center" });
      target.focus({ preventScroll: true });
    }
  };

  return (
    <>
      <DocumentHeader document={doc} />
      <DocumentFacts document={doc} />

      {active ? <ProcessingProgress status={doc.status} job={doc.job} /> : null}
      {doc.status === "failed" ? (
        <Alert variant="destructive" data-testid="document-error">
          <CircleAlert aria-hidden="true" />
          <AlertTitle>Processing failed{doc.error_code ? `: ${humanize(doc.error_code)}` : ""}</AlertTitle>
          <AlertDescription>
            {doc.error_message ?? "No reason was recorded."} Reprocess the document once the cause is fixed, or
            upload a better copy.
          </AlertDescription>
        </Alert>
      ) : null}

      <Tabs value={tab} onValueChange={(value) => isTab(value as string) && setTab(value as Tab)}>
        <TabsList variant="line" className="w-full flex-wrap justify-start border-b" aria-label="Document sections">
          <TabsTrigger value="data" className="flex-none" data-testid="tab-data">
            Extracted data
            <Badge variant="neutral" className="figures">
              {formatCount(doc.invoices.length)}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="validation" className="flex-none" data-testid="tab-validation">
            Validation
            {allIssues.length > 0 ? (
              <Badge variant={errorCount > 0 ? "danger" : "warning"} className="figures">
                {formatCount(allIssues.length)}
              </Badge>
            ) : null}
          </TabsTrigger>
          <TabsTrigger value="text" className="flex-none" data-testid="tab-text">
            Text
          </TabsTrigger>
          <TabsTrigger value="history" className="flex-none" data-testid="tab-history">
            History
          </TabsTrigger>
          <TabsTrigger value="cross" className="flex-none" data-testid="tab-cross">
            Cross-verification
          </TabsTrigger>
        </TabsList>

        {/* Kept mounted so unsaved edits survive a look at another tab. */}
        <TabsContent value="data" keepMounted className="flex flex-col gap-4 pt-4">
          {doc.invoices.length === 0 ? (
            <EmptyState
              icon={Inbox}
              title={active ? "Extraction has not finished yet" : "No invoice was found in this document"}
              description={
                active
                  ? "Records appear here as soon as processing completes."
                  : "Check the Text tab to see what was read from the file."
              }
            />
          ) : (
            doc.invoices.map((invoice, index) => (
              <InvoiceRecordCard
                key={invoice.id}
                invoice={invoice}
                position={index + 1}
                total={doc.invoices.length}
                highlighted={invoice.id === focusInvoiceId}
              />
            ))
          )}
        </TabsContent>
        <TabsContent value="validation" className="pt-4">
          <IssuesList issues={allIssues} onSelectField={goToField} recordLabel={recordLabel} />
        </TabsContent>
        <TabsContent value="text" className="pt-4">
          <PageTextViewer document={doc} initialPage={initialPage} />
        </TabsContent>
        <TabsContent value="history" className="pt-4">
          <DocumentHistory document={doc} />
        </TabsContent>
        <TabsContent value="cross" className="pt-4">
          <CrossVerificationPanel document={doc} />
        </TabsContent>
      </Tabs>
    </>
  );
}

function DocumentHeader({ document }: { document: DocumentDetail }) {
  const reprocess = useReprocessDocument();
  const active = isActiveDocument(document);
  const [confirming, setConfirming] = useState(false);

  return (
    <PageHeader
      eyebrow={
        <Link
          href="/"
          className="inline-flex w-fit items-center gap-1 rounded-sm text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft aria-hidden="true" className="size-4" />
          Documents
        </Link>
      }
      title={
        <span className="flex flex-wrap items-center gap-3">
          <span className="break-all">{document.filename}</span>
          <StatusBadge status={document.status} />
        </span>
      }
      actions={
        <>
          <a
            href={api.documentFileUrl(document.id)}
            download={document.filename}
            className={buttonVariants({ variant: "outline" })}
            data-testid="download-original"
          >
            <Download aria-hidden="true" data-icon="inline-start" />
            Download original
          </a>
          <Button
            type="button"
            variant="outline"
            disabled={active || reprocess.isPending}
            data-testid="reprocess-document"
            onClick={() => setConfirming(true)}
          >
            {reprocess.isPending ? <Spinner data-icon="inline-start" /> : <RotateCw aria-hidden="true" data-icon="inline-start" />}
            Reprocess
          </Button>
          <ReprocessDialog
            filename={document.filename}
            open={confirming}
            onOpenChange={setConfirming}
            onConfirm={() =>
              reprocess.mutate(document.id, {
                onSuccess: () => toast.add({ title: "Queued for reprocessing", description: document.filename, type: "success" }),
                onError: (error) => toast.add({ title: "Could not reprocess", description: errorMessage(error), type: "error" }),
              })
            }
          />
        </>
      }
    />
  );
}

function DocumentFacts({ document }: { document: DocumentDetail }) {
  const extraction = document.extraction;
  const model = [extraction?.llm_provider, extraction?.llm_model].filter(Boolean).join(" ");
  const facts: Array<{ label: string; value: string; mono?: boolean; title?: string }> = [
    { label: "File type", value: labelFor(FILE_KIND_LABELS, document.file_kind) },
    { label: "Size", value: formatBytes(document.size_bytes) },
    {
      label: "Pages",
      value:
        document.page_count == null
          ? EMPTY
          : `${formatCount(document.page_count)}${document.ocr_page_count ? `, ${formatCount(document.ocr_page_count)} by OCR` : ""}`,
    },
    { label: "Uploaded", value: formatDateTime(document.created_at) },
    { label: "Processed", value: formatDateTime(document.processed_at) },
    { label: "Processing time", value: formatDuration(document.processing_ms) },
    {
      label: "Extraction",
      value: extraction?.method ? `${humanize(extraction.method)}${model ? `, ${model}` : ""}` : EMPTY,
    },
    { label: "SHA-256", value: `${document.sha256.slice(0, 12)}…`, mono: true, title: document.sha256 },
  ];

  return (
    <dl className="flex flex-wrap gap-x-10 gap-y-3 rounded-lg border bg-card p-4 text-sm">
      {facts.map((fact) => (
        <div key={fact.label} className="flex min-w-0 flex-col gap-0.5">
          <dt className="text-xs text-muted-foreground">{fact.label}</dt>
          <dd className={fact.mono ? "figures" : undefined} title={fact.title ?? fact.value}>
            {fact.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
