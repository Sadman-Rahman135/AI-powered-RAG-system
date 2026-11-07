"use client";

/**
 * Multi-file upload with a per-file outcome for every file: queued (with a link), rejected (with the reason),
 * duplicate (with a link to the original), or failed (with the error, and a retry).
 *
 * Files the server would refuse (type, size, empty) are rejected before sending. The rest go out over
 * XMLHttpRequest — for upload progress — as multipart field `files`, at most `maxFilesPerRequest` per request
 * and within the request size cap; larger selections are split into sequential batches. A file over the per-file
 * limit is rejected client-side with the server's wording; if one slips through, the server's per-file
 * `file_too_large` rejection is shown on its row. Milestones are announced in a polite live region.
 */
import { CircleAlert, CircleCheck, CircleX, Clock, Copy, FileUp, RotateCcw, Upload } from "lucide-react";
import Link from "next/link";
import { useEffect, useId, useRef, useState, type DragEvent } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { errorMessage } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import type { UploadItem } from "@/lib/types";
import {
  DEFAULT_ALLOWED_EXTENSIONS,
  DEFAULT_MAX_FILES_PER_REQUEST,
  DEFAULT_MAX_UPLOAD_MB,
  batchBySize,
  describeExtensions,
  perFileProgress,
  precheckFile,
  uploadFiles,
} from "@/lib/upload";
import { cn } from "@/lib/utils";

type RowState = "waiting" | "uploading" | "queued" | "rejected" | "failed";

interface UploadRow {
  key: number;
  name: string;
  size: number;
  state: RowState;
  progress: number;
  message: string | null;
  documentId: string | null;
  duplicateOfId: string | null;
  /** A failed upload whose file is still held, so it can be sent again. */
  retryable: boolean;
}

interface Batch {
  keys: number[];
  files: File[];
}

interface UploadDropzoneProps {
  maxFilesPerRequest?: number;
  maxUploadMb?: number;
  /** Cap of one request body in MiB; batches are also split so their total stays under it. */
  maxRequestMb?: number;
  allowedExtensions?: readonly string[];
  /** Called with the server's per-file results after each batch. */
  onUploaded?: (items: UploadItem[]) => void;
}

export function UploadDropzone({
  maxFilesPerRequest = DEFAULT_MAX_FILES_PER_REQUEST,
  maxUploadMb = DEFAULT_MAX_UPLOAD_MB,
  maxRequestMb,
  allowedExtensions = DEFAULT_ALLOWED_EXTENSIONS,
  onUploaded,
}: UploadDropzoneProps) {
  const [rows, setRows] = useState<UploadRow[]>([]);
  const [dragging, setDragging] = useState(false);
  const [batchInfo, setBatchInfo] = useState({ current: 0, total: 0 });
  const [requestProgress, setRequestProgress] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const queueRef = useRef<Batch[]>([]);
  const runningRef = useRef(false);
  const filesRef = useRef(new Map<number, File>());
  const keyRef = useRef(0);
  const inputId = useId();
  const helpId = useId();

  const accept = allowedExtensions.join(",");
  const limits = { maxUploadMb, allowedExtensions };

  // Closing or reloading the tab mid-upload would drop the batches not yet sent; ask first.
  useEffect(() => {
    if (!running) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [running]);

  function updateRows(keys: ReadonlySet<number>, update: (row: UploadRow) => UploadRow) {
    setRows((previous) => previous.map((row) => (keys.has(row.key) ? update(row) : row)));
  }

  function applyResults(batch: Batch, items: readonly UploadItem[]) {
    // Results come back in the order the files were sent; fall back to matching by name if the counts differ.
    const remaining = [...items];
    const resultFor = (index: number, name: string): UploadItem | undefined => {
      if (items.length === batch.files.length) return items[index];
      const at = remaining.findIndex((item) => item.filename === name);
      return at === -1 ? undefined : remaining.splice(at, 1)[0];
    };
    const byKey = new Map(batch.keys.map((key, index) => [key, resultFor(index, batch.files[index].name)]));
    for (const [key, item] of byKey) {
      if (item) filesRef.current.delete(key);
    }
    setRows((previous) =>
      previous.map((row) => {
        if (!byKey.has(row.key)) return row;
        const item = byKey.get(row.key);
        if (!item) {
          return {
            ...row,
            state: "failed",
            progress: 0,
            message: "The server returned no result for this file.",
            retryable: true,
          };
        }
        if (item.status === "rejected") {
          return {
            ...row,
            state: "rejected",
            progress: 0,
            message: item.error?.message ?? "The server did not accept this file.",
          };
        }
        return {
          ...row,
          state: "queued",
          progress: 1,
          message: null,
          documentId: item.document_id,
          duplicateOfId: item.duplicate_of_id,
        };
      }),
    );
  }

  async function drain() {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    try {
      while (queueRef.current.length > 0) {
        const batch = queueRef.current.shift() as Batch;
        const keys = new Set(batch.keys);
        const sizes = batch.files.map((file) => file.size);
        setBatchInfo((info) => ({ ...info, current: info.current + 1 }));
        setRequestProgress(0);
        updateRows(keys, (row) => ({ ...row, state: "uploading", progress: 0, message: null }));
        try {
          const response = await uploadFiles(batch.files, {
            onProgress: (loaded, total) => {
              const perFile = perFileProgress(sizes, loaded, total);
              setRequestProgress(total > 0 ? loaded / total : 0);
              setRows((previous) =>
                previous.map((row) => {
                  const index = batch.keys.indexOf(row.key);
                  return index === -1 ? row : { ...row, progress: perFile[index] };
                }),
              );
            },
          });
          applyResults(batch, response.items ?? []);
          onUploaded?.(response.items ?? []);
        } catch (error) {
          const message = errorMessage(error);
          updateRows(keys, (row) => ({ ...row, state: "failed", progress: 0, message, retryable: true }));
        }
      }
    } finally {
      runningRef.current = false;
      setRunning(false);
      setRequestProgress(null);
      setBatchInfo({ current: 0, total: 0 });
    }
  }

  function enqueue(entries: ReadonlyArray<{ key: number; file: File }>) {
    if (entries.length === 0) return;
    const batches = batchBySize(
      entries,
      (entry) => entry.file.size,
      maxFilesPerRequest,
      maxRequestMb === undefined ? undefined : maxRequestMb * 1024 * 1024,
    ).map((group) => ({
      keys: group.map((entry) => entry.key),
      files: group.map((entry) => entry.file),
    }));
    queueRef.current.push(...batches);
    setBatchInfo((info) => ({ ...info, total: info.total + batches.length }));
    void drain();
  }

  function addFiles(files: readonly File[]) {
    if (files.length === 0) return;
    const newRows: UploadRow[] = [];
    const accepted: Array<{ key: number; file: File }> = [];
    for (const file of files) {
      const key = keyRef.current++;
      const reason = precheckFile(file, limits);
      newRows.push({
        key,
        name: file.name,
        size: file.size,
        state: reason ? "rejected" : "waiting",
        progress: 0,
        message: reason,
        documentId: null,
        duplicateOfId: null,
        retryable: false,
      });
      if (!reason) {
        accepted.push({ key, file });
        filesRef.current.set(key, file);
      }
    }
    setRows((previous) => [...newRows, ...previous]);
    enqueue(accepted);
  }

  function retryFailed() {
    const entries = rows.flatMap((row) => {
      const file = row.retryable ? filesRef.current.get(row.key) : undefined;
      return file ? [{ key: row.key, file }] : [];
    });
    if (entries.length === 0) return;
    const keys = new Set(entries.map((entry) => entry.key));
    updateRows(keys, (row) => ({ ...row, state: "waiting", message: null, retryable: false }));
    enqueue(entries);
  }

  function clearList() {
    setRows([]);
    filesRef.current.clear();
  }

  function onDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    if (!dragging) setDragging(true);
  }

  function onDragLeave(event: DragEvent<HTMLDivElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  const counts = countStates(rows);
  const retryable = rows.some((row) => row.retryable);
  const status = statusText(running, counts, batchInfo);

  return (
    <section aria-labelledby={`${inputId}-title`} className="flex flex-col gap-3">
      <div
        data-testid="upload-dropzone"
        data-dragging={dragging ? "true" : "false"}
        onDragEnter={onDragOver}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={cn(
          "flex flex-col items-start gap-3 rounded-lg border border-dashed border-input bg-card px-5 py-4 transition-colors sm:flex-row sm:items-center",
          dragging && "border-primary bg-accent",
        )}
      >
        <div className="flex size-10 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
          <FileUp aria-hidden="true" className="size-5" />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <h2 id={`${inputId}-title`} className="text-sm font-semibold">
            Upload documents
          </h2>
          <p id={helpId} className="text-sm text-muted-foreground">
            Drop invoices, receipts and spreadsheets here: {describeExtensions(allowedExtensions)}, up to {maxUploadMb} MB
            each. Large selections are sent {maxFilesPerRequest} files at a time.
          </p>
        </div>
        <Button type="button" onClick={() => inputRef.current?.click()} aria-describedby={helpId}>
          <Upload aria-hidden="true" data-icon="inline-start" />
          Choose files
        </Button>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          multiple
          accept={accept}
          tabIndex={-1}
          aria-hidden="true"
          className="sr-only"
          data-testid="upload-input"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            addFiles(files);
          }}
        />
      </div>

      <p role="status" aria-live="polite" data-testid="upload-status" className={cn("text-sm", !status && "sr-only")}>
        {status}
      </p>

      {rows.length > 0 ? (
        <div className="flex flex-col gap-2 rounded-lg border bg-card">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
            <p className="text-sm text-muted-foreground">
              {rows.length} file{rows.length === 1 ? "" : "s"}
              {running && requestProgress !== null ? (
                <>
                  {" "}
                  <span data-testid="upload-progress" className="figures">
                    {batchInfo.total > 1 ? `Batch ${batchInfo.current} of ${batchInfo.total}: ` : ""}
                    {Math.round(requestProgress * 100)}%
                  </span>
                </>
              ) : null}
            </p>
            <div className="flex gap-2">
              {retryable && !running ? (
                <Button type="button" variant="outline" size="sm" onClick={retryFailed}>
                  <RotateCcw aria-hidden="true" data-icon="inline-start" />
                  Retry failed
                </Button>
              ) : null}
              {!running ? (
                <Button type="button" variant="ghost" size="sm" onClick={clearList}>
                  Clear list
                </Button>
              ) : null}
            </div>
          </div>
          <ul aria-label="Upload results" className="max-h-80 divide-y overflow-y-auto">
            {rows.map((row) => (
              <UploadRowItem key={row.key} row={row} />
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

type Counts = Record<RowState, number>;

function countStates(rows: readonly UploadRow[]): Counts {
  const counts: Counts = { waiting: 0, uploading: 0, queued: 0, rejected: 0, failed: 0 };
  for (const row of rows) counts[row.state] += 1;
  return counts;
}

function statusText(running: boolean, counts: Counts, batches: { current: number; total: number }): string {
  if (running) {
    const active = counts.waiting + counts.uploading;
    const plural = active === 1 ? "" : "s";
    const batchText = batches.total > 1 ? ` in ${batches.total} batches, batch ${batches.current} of ${batches.total}` : "";
    return `Uploading ${active} file${plural}${batchText}.`;
  }
  const parts = [
    counts.queued ? `${counts.queued} queued` : null,
    counts.rejected ? `${counts.rejected} rejected` : null,
    counts.failed ? `${counts.failed} failed` : null,
  ].filter(Boolean);
  return parts.length > 0 ? `Upload finished: ${parts.join(", ")}.` : "";
}

const STATE_ICON = {
  waiting: Clock,
  uploading: Upload,
  queued: CircleCheck,
  rejected: CircleX,
  failed: CircleAlert,
} as const;

const STATE_ICON_COLOR: Record<RowState, string> = {
  waiting: "text-muted-foreground",
  uploading: "text-tone-info-foreground",
  queued: "text-tone-success-foreground",
  rejected: "text-tone-danger-foreground",
  failed: "text-tone-danger-foreground",
};

function UploadRowItem({ row }: { row: UploadRow }) {
  const Icon = row.duplicateOfId ? Copy : STATE_ICON[row.state];
  const percent = Math.round(row.progress * 100);
  return (
    <li data-testid="upload-row" data-state={row.state} className="flex items-start gap-3 px-4 py-2.5">
      <Icon
        aria-hidden="true"
        className={cn("mt-0.5 size-4 shrink-0", row.duplicateOfId ? "text-tone-warning-foreground" : STATE_ICON_COLOR[row.state])}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex min-w-0 items-baseline gap-2">
          <span className="truncate font-medium" title={row.name}>
            {row.name}
          </span>
          <span className="shrink-0 text-xs text-muted-foreground">{formatBytes(row.size)}</span>
        </div>
        {row.message ? (
          <p className={cn("text-sm", row.state === "rejected" || row.state === "failed" ? "text-destructive" : "text-muted-foreground")}>
            {row.message}
          </p>
        ) : null}
        {row.state === "queued" && row.duplicateOfId ? (
          <p className="text-sm text-tone-warning-foreground">
            Same file as an earlier upload.{" "}
            <Link href={`/documents/${row.duplicateOfId}`} className="font-medium underline underline-offset-2">
              View original
            </Link>
          </p>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {row.state === "uploading" ? (
          <>
            <Progress value={percent} aria-label={`Uploading ${row.name}`} className="w-28" />
            <span className="figures w-9 text-right text-xs text-muted-foreground">{percent}%</span>
          </>
        ) : null}
        {row.state === "waiting" ? <span className="text-xs text-muted-foreground">Waiting</span> : null}
        {row.state === "queued" ? (
          <>
            <Badge variant="success">Queued</Badge>
            {row.documentId ? (
              <Link
                href={`/documents/${row.documentId}`}
                className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                aria-label={`Open ${row.name}`}
              >
                Open
              </Link>
            ) : null}
          </>
        ) : null}
        {row.state === "rejected" ? <Badge variant="danger">Rejected</Badge> : null}
        {row.state === "failed" ? <Badge variant="danger">Upload failed</Badge> : null}
      </div>
    </li>
  );
}
