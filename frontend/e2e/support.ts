/**
 * Shared helpers for the browser end-to-end specs, which run against the live compose stack (UI on :3000 with the
 * same-origin `/api` proxy that injects the API key, so no key is needed here).
 *
 * The database persists between runs and may already hold documents, so specs never assume it is empty: each one
 * uploads its own labelled sample files from `backend/eval/dataset`, identifies them by the ids the upload returns,
 * waits for processing to finish (real OCR and a local LLM: seconds to a minute per document) and deletes what it
 * created afterwards.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, type APIRequestContext, type ConsoleMessage, type Page } from "@playwright/test";

export const DATASET_DIR = process.env.E2E_DATASET_DIR ?? path.resolve(__dirname, "../../backend/eval/dataset");

/** How long a document may take to reach a final status (queue wait + OCR + LLM extraction). */
export const PROCESSING_TIMEOUT_MS = 180_000;

export const FINAL_STATUSES = ["completed", "needs_review", "failed"] as const;

export type DocumentStatus = "queued" | "processing" | "completed" | "needs_review" | "failed";

export interface UploadItem {
  filename: string;
  status: "queued" | "rejected";
  document_id: string | null;
  job_id: string | null;
  duplicate_of_id: string | null;
  error: { code: string; message: string } | null;
}

export interface Issue {
  id: number;
  code: string;
  severity: "error" | "warning" | "info";
  field: string | null;
  message: string;
  invoice_id: string | null;
}

export interface Invoice {
  id: string;
  document_id: string;
  filename: string;
  invoice_number: string | null;
  supplier_name: string | null;
  total: string | null;
  currency: string | null;
  review_status: "auto_approved" | "needs_review" | "reviewed";
  sheet_name: string | null;
  row_number: number | null;
  issues: Issue[];
}

export interface DocumentDetail {
  id: string;
  filename: string;
  sha256: string;
  status: DocumentStatus;
  error_code: string | null;
  error_message: string | null;
  duplicate_of_id: string | null;
  invoice_count?: number;
  invoices: Invoice[];
  document_issues: Issue[];
}

/** Absolute path of a dataset file, e.g. `datasetFile("pdf/t1_digital_01.pdf")`. */
export function datasetFile(relative: string): string {
  return path.join(DATASET_DIR, ...relative.split("/"));
}

export function sha256Of(relative: string): string {
  return createHash("sha256").update(readFileSync(datasetFile(relative))).digest("hex");
}

/** POST the files in one multipart request (field `files`), exactly like the UI does; returns one item per file. */
export async function uploadViaApi(request: APIRequestContext, files: readonly string[]): Promise<UploadItem[]> {
  const form = new FormData();
  for (const relative of files) {
    form.append("files", new Blob([readFileSync(datasetFile(relative))]), path.basename(relative));
  }
  const response = await request.post("/api/v1/documents", { multipart: form });
  expect(response.status(), await response.text()).toBe(202);
  const body = (await response.json()) as { items: UploadItem[] };
  expect(body.items).toHaveLength(files.length);
  return body.items;
}

export async function getDocument(request: APIRequestContext, id: string): Promise<DocumentDetail> {
  const response = await request.get(`/api/v1/documents/${id}`);
  expect(response.ok(), `GET document ${id}: ${response.status()}`).toBeTruthy();
  return (await response.json()) as DocumentDetail;
}

export async function getInvoice(request: APIRequestContext, id: string): Promise<Invoice> {
  const response = await request.get(`/api/v1/invoices/${id}`);
  expect(response.ok(), `GET invoice ${id}: ${response.status()}`).toBeTruthy();
  return (await response.json()) as Invoice;
}

/** Poll the API until the document is completed, needs review or failed. */
export async function waitForFinal(
  request: APIRequestContext,
  id: string,
  timeout = PROCESSING_TIMEOUT_MS,
): Promise<DocumentDetail> {
  let last: DocumentDetail | undefined;
  await expect
    .poll(
      async () => {
        last = await getDocument(request, id);
        return last.status;
      },
      { message: `document ${id} did not finish processing`, timeout, intervals: [1_000, 2_000] },
    )
    .toMatch(/^(completed|needs_review|failed)$/);
  return last as DocumentDetail;
}

/** Upload files and wait until every queued one has finished processing. */
export async function uploadAndProcess(
  request: APIRequestContext,
  files: readonly string[],
): Promise<{ items: UploadItem[]; documents: DocumentDetail[] }> {
  const items = await uploadViaApi(request, files);
  const documents: DocumentDetail[] = [];
  for (const item of items) {
    expect(item.status, `${item.filename}: ${item.error?.message ?? ""}`).toBe("queued");
    documents.push(await waitForFinal(request, item.document_id as string));
  }
  return { items, documents };
}

/**
 * Delete documents this run created. A document whose job is still running cannot be deleted (409), so wait for
 * it to finish first; a document that is already gone (404) is fine.
 */
export async function deleteDocuments(request: APIRequestContext, ids: Iterable<string | null | undefined>): Promise<void> {
  for (const id of new Set([...ids].filter((value): value is string => Boolean(value)))) {
    const existing = await request.get(`/api/v1/documents/${id}`);
    if (existing.status() === 404) continue;
    await waitForFinal(request, id).catch(() => undefined);
    await expect
      .poll(async () => [204, 404].includes((await request.delete(`/api/v1/documents/${id}`)).status()), {
        message: `delete document ${id}`,
        timeout: 60_000,
      })
      .toBe(true);
  }
}

/** Documents (any status) whose stored bytes are identical to a dataset file. */
export async function findDocumentsBySha(request: APIRequestContext, relative: string): Promise<DocumentDetail[]> {
  const sha = sha256Of(relative);
  const response = await request.get("/api/v1/documents", {
    params: { q: path.basename(relative), limit: 100 },
  });
  expect(response.ok()).toBeTruthy();
  const body = (await response.json()) as { items: DocumentDetail[] };
  return body.items.filter((document) => document.sha256 === sha);
}

/**
 * A processed copy of a shared, read-only fixture (the cross-verification ledger): reuse an existing copy when
 * there is one, so no duplicate is created, otherwise upload it. `created` says whether this run owns it.
 */
export async function ensureProcessed(
  request: APIRequestContext,
  relative: string,
): Promise<{ document: DocumentDetail; created: boolean }> {
  const existing = (await findDocumentsBySha(request, relative)).find(
    (document) => document.status === "completed" || document.status === "needs_review",
  );
  if (existing) return { document: await getDocument(request, existing.id), created: false };
  const { documents } = await uploadAndProcess(request, [relative]);
  return { document: documents[0], created: true };
}

/** Collects console errors and uncaught page errors; `ignore` filters known, expected messages. */
export function collectPageErrors(page: Page, ignore: readonly RegExp[] = []): string[] {
  const errors: string[] = [];
  const keep = (text: string) => !ignore.some((pattern) => pattern.test(text));
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error" && keep(message.text())) errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    if (keep(error.message)) errors.push(`pageerror: ${error.message}`);
  });
  return errors;
}

/** The documents-table row of one document. */
export function documentRow(page: Page, documentId: string) {
  return page.locator(`[data-testid="documents-row"][data-document-id="${documentId}"]`);
}

/** The upload-result row of one file in the dropzone's list (newest batch first). */
export function uploadRow(page: Page, filename: string) {
  return page.getByTestId("upload-row").filter({ hasText: filename });
}
