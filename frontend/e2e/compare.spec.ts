/**
 * Cross-verification of a PDF invoice against its row in an Excel ledger: the automatic check on the document's
 * Cross-verification tab, and the field-by-field comparison on the Compare page (records and documents).
 *
 * The ledger is a shared, read-only fixture (an existing processed copy is reused, so no duplicate is created);
 * the PDF is uploaded by this spec and deleted afterwards. Both carry invoice 2024/8001 with identical values.
 */
import { expect, test, type APIRequestContext, type Page, type Response } from "@playwright/test";

import { deleteDocuments, ensureProcessed, uploadAndProcess, type DocumentDetail, type Invoice } from "./support";

const PDF = "pdf/t2_digital_04.pdf"; // EUR amounts: an unambiguous currency symbol
const LEDGER = "ledgers/xv_ledger.xlsx";
const NUMBER = "2024/8001";

let pdf: DocumentDetail;
let ledger: DocumentDetail;
let pdfInvoice: Invoice;
let ledgerRow: Invoice;
const created: string[] = [];

test.describe.configure({ mode: "serial" });

test.beforeAll(async ({ request }) => {
  const shared = await ensureProcessed(request, LEDGER);
  ledger = shared.document;
  if (shared.created) created.push(ledger.id);
  // Uploaded after the ledger, so processing the PDF cross-checks it against the ledger row automatically.
  const { documents } = await uploadAndProcess(request, [PDF]);
  pdf = documents[0];
  created.push(pdf.id);
  pdfInvoice = pdf.invoices[0];
  ledgerRow = ledger.invoices.find((record) => record.invoice_number === NUMBER) as Invoice;
});

test.afterAll(async ({ request }) => {
  await deleteDocuments(request, created);
});

interface StoredComparison {
  status: string;
  trigger: string;
  left: { document_id: string; filename: string };
  right: { document_id: string; filename: string };
  fields: Array<{ field: string; left: string | null; right: string | null; status: string }>;
}

const isCrossVerify = (response: Response) =>
  response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/cross-verify";

/** The stored automatic comparison of the PDF record with the ledger row. */
async function storedComparison(request: APIRequestContext): Promise<StoredComparison> {
  const response = await request.get("/api/v1/cross-verifications", { params: { document_id: pdf.id } });
  expect(response.ok()).toBeTruthy();
  const items = ((await response.json()) as { items: StoredComparison[] }).items;
  const pair = items.find((item) => [item.left.document_id, item.right.document_id].includes(ledger.id));
  expect(pair, "a stored comparison with the ledger").toBeDefined();
  return pair as StoredComparison;
}

/**
 * The comparison table shows every compared field with the server's verdict, the ledger's values, and (for these
 * fixtures) agreement on every field. Mismatches are listed by name so an extraction miss is easy to read.
 */
async function expectFieldByField(page: Page, stored: StoredComparison) {
  const result = page.getByTestId("compare-result");
  const row = (field: string) => result.locator(`[data-testid="compare-field-row"][data-field="${field}"]`);
  await expect(result).toHaveAttribute("data-status", stored.status);
  await expect(result.getByTestId("compare-field-row")).toHaveCount(stored.fields.length);
  expect(stored.fields.map((field) => field.field)).toEqual(expect.arrayContaining(["invoice_number", "supplier_name", "total", "currency"]));
  for (const field of stored.fields) await expect(row(field.field)).toHaveAttribute("data-status", field.status);
  await expect(row("invoice_number")).toHaveAttribute("data-status", "match"); // the pairing key
  await expect(row("invoice_number")).toContainText(NUMBER);
  await expect(row("total")).toContainText("€1,595.26"); // the ledger's value
  await expect(result.getByTestId("compare-line-items")).toBeVisible();

  const differing = stored.fields.filter((field) => field.status !== "match").map((field) => `${field.field}: ${field.left} vs ${field.right}`);
  expect(differing, "fields where the PDF extraction and the ledger row differ").toEqual([]);
  expect(stored.status).toBe("match");
}

/** Search a picker and choose an option once the list shows this query's results (it keeps the previous list while
 * the debounced search runs, so clicking earlier could pick a stale option). */
async function pick(page: Page, side: "left" | "right", query: string, optionText: string) {
  await page.getByTestId(`compare-${side}-search`).fill(query);
  const options = page.getByTestId(`compare-${side}`).getByTestId("compare-option");
  await expect(options.first()).toBeVisible();
  await expect(options.filter({ hasNotText: query })).toHaveCount(0);
  await options.filter({ hasText: optionText }).first().click();
}

test("the fixtures were processed: the PDF invoice and the ledger row share the invoice number", () => {
  expect(pdf.status).not.toBe("failed");
  expect(pdfInvoice?.invoice_number).toBe(NUMBER);
  expect(ledgerRow, `${LEDGER} has a row for ${NUMBER}`).toBeDefined();
  expect(ledgerRow.sheet_name).toBe("International");
});

test("processing cross-checks the PDF against its ledger row automatically", async ({ page, request }) => {
  const stored = await storedComparison(request);
  expect(stored.trigger).toBe("auto");
  await page.goto(`/documents/${pdf.id}`);
  await page.getByTestId("tab-cross").click();

  const check = page.getByTestId("cross-verification").filter({ has: page.locator(`a[href="/documents/${ledger.id}"]`) });
  await expect(check).toHaveCount(1);
  await expect(check).toContainText(`Against xv_ledger.xlsx ${NUMBER}`);
  await expect(check).toContainText("PDF and spreadsheet, checked automatically");
  await expect(check.getByTestId("match-badge").first()).toHaveAttribute("data-status", stored.status);

  // A match is collapsed (a mismatch opens by itself); the field-by-field table opens on request.
  if (stored.status === "match") await check.getByRole("button", { name: "Show fields" }).click();
  await expectFieldByField(page, stored);
  await expect(page.getByTestId("compare-result").getByText("This document")).toBeVisible();
});

test("the Compare page compares the PDF invoice with the ledger row field by field", async ({ page }) => {
  await page.goto("/compare");
  await expect(page.getByTestId("compare-kind").getByRole("button", { name: "Invoice records" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("compare-submit")).toBeDisabled();

  await pick(page, "left", NUMBER, "t2_digital_04.pdf");
  await expect(page).toHaveURL(new RegExp(`left=${pdfInvoice.id}`));
  await expect(page.getByTestId("compare-left-selected")).toContainText(NUMBER);

  // The PDF's own record cannot be picked twice; any copy of the ledger row carries the same values.
  await pick(page, "right", NUMBER, "xv_ledger.xlsx");
  await expect(page).toHaveURL(/[?&]right=[0-9a-f-]{36}/);
  await expect(page.getByTestId("compare-right-selected")).toContainText("xv_ledger.xlsx");
  await expect(page.getByTestId("compare-submit")).toBeEnabled();

  const compared = page.waitForResponse(isCrossVerify);
  await page.getByTestId("compare-submit").click();
  const response = await compared;
  expect(response.status()).toBe(200);
  await expectFieldByField(page, (await response.json()) as StoredComparison);
  await expect(page.getByTestId("compare-result")).toContainText("Compared on request");
  // A pair is stored once: its sides keep the stored orientation (the earlier upload, the ledger, on the left).
  await expect(page.locator("caption")).toHaveText(
    /^Field comparison of (t2_digital_04\.pdf and xv_ledger\.xlsx|xv_ledger\.xlsx and t2_digital_04\.pdf)$/,
  );
});

test("whole documents can be compared too: the matching records are paired", async ({ page }) => {
  await page.goto("/compare");
  await page.getByTestId("compare-kind").getByRole("button", { name: "Documents" }).click();
  await expect(page).toHaveURL(/type=documents/);

  await pick(page, "left", "t2_digital_04", "t2_digital_04.pdf");
  await expect(page).toHaveURL(new RegExp(`left=${pdf.id}`));
  await pick(page, "right", "xv_ledger", "xv_ledger.xlsx");
  await expect(page.getByTestId("compare-right-selected")).toContainText("xv_ledger.xlsx");
  await expect(page.getByTestId("compare-right-selected")).toContainText("XLSX, 20 records");

  const compared = page.waitForResponse(isCrossVerify);
  await page.getByRole("button", { name: "Compare documents" }).click();
  const response = await compared;
  expect(response.status()).toBe(200);
  await expectFieldByField(page, (await response.json()) as StoredComparison);
});
