/**
 * Multi-file upload from the dashboard: every file gets its own outcome (queued with a link, rejected with the
 * reason, duplicate notice), and queued documents move through processing to a final status in the live table.
 */
import { expect, test, type Response } from "@playwright/test";

import {
  PROCESSING_TIMEOUT_MS,
  datasetFile,
  deleteDocuments,
  documentRow,
  getDocument,
  uploadRow,
  waitForFinal,
  type UploadItem,
} from "./support";

const DIGITAL_PDF = "pdf/t2_digital_02.pdf";
const EMPTY_FILE = "corrupt/empty.pdf"; // 0 bytes: refused in the browser before sending
const NOT_A_PDF = "corrupt/random_bytes.pdf"; // .pdf name, random content: refused by the server's content sniffing
const TRUNCATED_PDF = "corrupt/truncated.pdf"; // a real PDF header, cut short: accepted, then fails as corrupt_file

const created = new Set<string>();
const ids = { pdf: "", truncated: "" };

test.describe.configure({ mode: "serial" });

test.afterAll(async ({ request }) => {
  await deleteDocuments(request, created);
});

function isUpload(response: Response): boolean {
  return response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/documents";
}

async function uploadItems(response: Response): Promise<UploadItem[]> {
  expect(response.status()).toBe(202);
  const items = ((await response.json()) as { items: UploadItem[] }).items;
  for (const item of items) if (item.document_id) created.add(item.document_id);
  return items;
}

test("a mixed batch shows a per-file outcome and processes to final statuses", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Documents", level: 1 })).toBeVisible();
  await expect(page.getByTestId("documents-table")).toBeVisible();

  const uploaded = page.waitForResponse(isUpload);
  await page
    .getByTestId("upload-input")
    .setInputFiles([DIGITAL_PDF, EMPTY_FILE, NOT_A_PDF, TRUNCATED_PDF].map(datasetFile));
  const items = await uploadItems(await uploaded);

  // The empty file never left the browser: one request carried the other three, in order.
  expect(items.map((item) => item.filename)).toEqual(["t2_digital_02.pdf", "random_bytes.pdf", "truncated.pdf"]);
  const [pdfItem, notPdfItem, truncatedItem] = items;
  expect(pdfItem.status).toBe("queued");
  expect(notPdfItem).toMatchObject({ status: "rejected", document_id: null, error: { code: "unsupported_file_type" } });
  expect(truncatedItem.status).toBe("queued");
  ids.pdf = pdfItem.document_id as string;
  ids.truncated = truncatedItem.document_id as string;

  // Per-file results in the dropzone list.
  await expect(page.getByTestId("upload-status")).toHaveText("Upload finished: 2 queued, 2 rejected.");
  await expect(page.getByTestId("upload-row")).toHaveCount(4);

  const emptyRow = uploadRow(page, "empty.pdf");
  await expect(emptyRow).toHaveAttribute("data-state", "rejected");
  await expect(emptyRow).toContainText("The file is empty.");
  await expect(emptyRow).toContainText("Rejected");

  const notPdfRow = uploadRow(page, "random_bytes.pdf");
  await expect(notPdfRow).toHaveAttribute("data-state", "rejected");
  await expect(notPdfRow).toContainText("Unsupported file type");

  for (const [name, item] of [
    ["t2_digital_02.pdf", pdfItem],
    ["truncated.pdf", truncatedItem],
  ] as const) {
    const row = uploadRow(page, name);
    await expect(row).toHaveAttribute("data-state", "queued");
    await expect(row).toContainText("Queued");
    await expect(row.getByRole("link", { name: `Open ${name}` })).toHaveAttribute("href", `/documents/${item.document_id}`);
  }

  // Live progress in the documents table: both rows appear while still queued or processing, then settle.
  const pdfRow = documentRow(page, pdfItem.document_id as string);
  const truncatedRow = documentRow(page, truncatedItem.document_id as string);
  await expect(pdfRow).toBeVisible();
  const firstSeen = await pdfRow.getAttribute("data-status");
  expect(["queued", "processing"], `first status seen was ${firstSeen}`).toContain(firstSeen);

  await expect(truncatedRow).toHaveAttribute("data-status", "failed", { timeout: PROCESSING_TIMEOUT_MS });
  await expect(truncatedRow.getByTestId("status-badge")).toHaveText("Failed");
  await expect(truncatedRow).toContainText("no page of the PDF could be read");

  await expect(pdfRow).toHaveAttribute("data-status", /^(completed|needs_review)$/, { timeout: PROCESSING_TIMEOUT_MS });
  const pdf = await getDocument(request, pdfItem.document_id as string);
  expect(pdf.invoices.length).toBeGreaterThanOrEqual(1);
  await expect(pdfRow.getByTestId("status-badge")).toHaveAttribute("data-status", pdf.status);
  await expect(pdfRow.locator("td").nth(4)).toHaveText(String(pdf.invoices.length)); // the Records column

  // The failed document explains why on its own page.
  await truncatedRow.getByRole("link", { name: "truncated.pdf" }).click();
  await expect(page).toHaveURL(`/documents/${truncatedItem.document_id}`);
  await expect(page.getByTestId("document-error")).toContainText("Processing failed: Corrupt file");
  await expect(page.getByTestId("document-error")).toContainText("no page of the PDF could be read");
});

test("uploading the same bytes again is accepted and flagged as a duplicate", async ({ page, request }) => {
  expect(ids.pdf, "the first test uploads the original").toBeTruthy();

  await page.goto("/");
  const uploaded = page.waitForResponse(isUpload);
  await page.getByTestId("upload-input").setInputFiles(datasetFile(DIGITAL_PDF));
  const [item] = await uploadItems(await uploaded);

  expect(item.status).toBe("queued");
  expect(item.duplicate_of_id, "same bytes as an earlier upload").toBeTruthy();

  const row = uploadRow(page, "t2_digital_02.pdf");
  await expect(row).toHaveAttribute("data-state", "queued");
  await expect(row).toContainText("Same file as an earlier upload.");
  await expect(row.getByRole("link", { name: "View original" })).toHaveAttribute("href", `/documents/${item.duplicate_of_id}`);

  const tableRow = documentRow(page, item.document_id as string);
  await expect(tableRow).toContainText("Same file as an earlier upload");
  await expect(tableRow).toHaveAttribute("data-status", /^(completed|needs_review)$/, { timeout: PROCESSING_TIMEOUT_MS });

  // Processing reports it as a DUPLICATE_FILE warning on the document.
  const duplicate = await waitForFinal(request, item.document_id as string);
  expect(duplicate.document_issues.map((issue) => issue.code)).toContain("DUPLICATE_FILE");
  await page.goto(`/documents/${item.document_id}?tab=validation`);
  await expect(page.getByTestId("issue-item").filter({ hasText: "This file was already uploaded" })).toBeVisible();
});

test("a document can be deleted from the table after confirming", async ({ page }) => {
  expect(ids.truncated, "the first test uploads truncated.pdf").toBeTruthy();
  await page.goto("/");
  const row = documentRow(page, ids.truncated);
  await expect(row).toBeVisible();

  await row.getByTestId("delete-document").click();
  const dialog = page.getByTestId("delete-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Delete truncated.pdf?");
  await dialog.getByTestId("confirm-delete").click();

  await expect(row).toHaveCount(0);
  await expect(page.getByText("Document deleted")).toBeVisible();
});
