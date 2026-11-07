/**
 * Human review: a record that needs review is opened from the review queue, a missing required field is filled in
 * and saved (audited, re-validated on the server), and the record is approved.
 *
 * The sample invoice prints no invoice number, so extraction leaves it empty and validation raises the error
 * "Required field 'invoice_number' is missing": a deterministic reason for review that the correction removes.
 */
import { expect, test, type Response } from "@playwright/test";

import { deleteDocuments, getDocument, getInvoice, uploadAndProcess, type DocumentDetail, type Invoice } from "./support";

const SAMPLE = "pdf/t4_digital_02.pdf"; // Konkan Building Supplies, INR, no invoice number on the page
const MISSING_NUMBER = "Required field 'invoice_number' is missing";

let document: DocumentDetail;
let invoice: Invoice;
const newNumber = `KBS-E2E-${Date.now()}`;
const comment = "Invoice number taken from the supplier's cover email (e2e)";

test.describe.configure({ mode: "serial" });

test.beforeAll(async ({ request }) => {
  const { documents } = await uploadAndProcess(request, [SAMPLE]);
  document = documents[0];
  invoice = document.invoices[0];
});

test.afterAll(async ({ request }) => {
  if (document) await deleteDocuments(request, [document.id]);
});

const isPatch = (response: Response) =>
  response.request().method() === "PATCH" && new URL(response.url()).pathname === `/api/v1/invoices/${invoice.id}`;

test("open a record from the review queue, fill in a missing field and save: audited and re-validated", async ({ page, request }) => {
  expect(document.status).toBe("needs_review");
  expect(invoice).toBeDefined();
  expect(invoice.review_status).toBe("needs_review");
  expect(invoice.invoice_number).toBeNull();
  expect(invoice.issues.map((issue) => issue.message)).toContain(MISSING_NUMBER);

  // The review queue lists it (newest first) with its issue codes.
  await page.goto("/review");
  await expect(page.getByRole("heading", { name: "Review queue", level: 1 })).toBeVisible();
  const queueRow = page.getByTestId("review-row").filter({ has: page.locator(`a[href*="${invoice.id}"]`) });
  await expect(queueRow).toBeVisible();
  await expect(queueRow).toContainText("Missing field");
  await queueRow.getByRole("link", { name: "Review" }).click();

  await expect(page).toHaveURL(`/documents/${document.id}?invoice=${invoice.id}`);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("t4_digital_02.pdf");
  const card = page.locator(`#record-${invoice.id}`);
  await expect(card).toHaveAttribute("data-review-status", "needs_review");
  // Every extracted field shows where it came from and how sure the extractor was.
  await expect(card.getByTestId("confidence-badge").first()).toBeVisible();

  // Validation tab: the error is listed, and "Go to" jumps to the empty input.
  await page.getByTestId("tab-validation").click();
  const issueItems = page.getByTestId("issue-item");
  const before = await issueItems.count();
  const missing = issueItems.filter({ hasText: MISSING_NUMBER });
  await expect(missing).toHaveAttribute("data-code", "MISSING_FIELD");
  await expect(page.locator('[data-testid="issue-group"][data-severity="error"]')).toContainText(MISSING_NUMBER);
  await missing.getByRole("button", { name: "Go to Invoice number" }).click();

  const numberInput = card.getByTestId("field-input-invoice_number");
  await expect(numberInput).toBeFocused();
  await expect(numberInput).toHaveValue("");

  // Edit: the field is marked as changed and approval waits for the save.
  await numberInput.fill(newNumber);
  const numberField = card.locator('[data-testid="invoice-field"][data-field="invoice_number"]');
  await expect(numberField).toHaveAttribute("data-dirty", "true");
  await expect(numberField).toContainText("Edited. Was empty");
  await expect(card.getByText("1 unsaved change")).toBeVisible();
  await expect(card.getByTestId("approve-invoice")).toBeDisabled();
  await card.getByLabel("Comment (optional)").fill(comment);

  const patched = page.waitForResponse(isPatch);
  await card.getByTestId("save-invoice").click();
  const response = await patched;
  expect(response.status()).toBe(200);
  const body = (await response.json()) as Invoice;
  expect(body.invoice_number).toBe(newNumber);
  expect(body.issues.map((issue) => issue.message)).not.toContain(MISSING_NUMBER);

  await expect(page.getByText("Changes saved")).toBeVisible();
  await expect(page.getByText(/^Checked again: /)).toBeVisible();

  // The saved value is now a manual one with full confidence.
  await expect(numberInput).toHaveValue(newNumber);
  await expect(numberField).toHaveAttribute("data-dirty", "false");
  await expect(numberField).toContainText("Manual");
  await expect(numberField.getByTestId("confidence-badge")).toContainText("100%");

  // Re-validated issues: the error is gone and the list matches the server.
  await page.getByTestId("tab-validation").click();
  await expect(issueItems.filter({ hasText: MISSING_NUMBER })).toHaveCount(0);
  const refreshed = await getDocument(request, document.id);
  const serverIssues = refreshed.document_issues.length + refreshed.invoices.reduce((sum, record) => sum + record.issues.length, 0);
  await expect(issueItems).toHaveCount(serverIssues);
  expect(serverIssues).toBeLessThan(before);

  // History: the correction is in the audit trail with who, what and why.
  await page.getByTestId("tab-history").click();
  await expect(page.getByTestId("events-timeline")).toBeVisible();
  const audit = page.getByTestId("corrections-table");
  const auditRow = audit.getByRole("row").filter({ hasText: newNumber });
  await expect(auditRow).toHaveCount(1);
  await expect(auditRow).toContainText("Invoice number");
  await expect(auditRow.getByRole("cell").nth(1)).toHaveText("empty"); // the value before
  await expect(auditRow).toContainText(comment);
  await expect(auditRow).toContainText("web-ui");
});

test("approve the corrected record: it becomes reviewed and leaves the queue", async ({ page, request }) => {
  await page.goto(`/documents/${document.id}?invoice=${invoice.id}`);
  const card = page.locator(`#record-${invoice.id}`);
  const approve = card.getByTestId("approve-invoice");
  await expect(approve).toBeEnabled();
  // needs_review while warnings remain; auto_approved if the correction cleared every issue.
  const statusBefore = await card.getAttribute("data-review-status");
  expect(["needs_review", "auto_approved"]).toContain(statusBefore);
  await approve.click();

  await expect(page.getByText("Record approved")).toBeVisible();
  await expect(card).toHaveAttribute("data-review-status", "reviewed");
  await expect(card.getByTestId("review-status-badge")).toHaveAttribute("data-status", "reviewed");
  await expect(approve).toHaveText("Approved");
  await expect(approve).toBeDisabled();
  await expect(card.getByText(/^Reviewed by web-ui on /)).toBeVisible();

  const saved = await getInvoice(request, invoice.id);
  expect(saved.review_status).toBe("reviewed");

  // The approval is part of the audit trail as well.
  await page.getByTestId("tab-history").click();
  const approval = page.getByTestId("corrections-table").getByRole("row").filter({ hasText: "Review status" });
  await expect(approval).toHaveCount(1);
  await expect(approval.getByRole("cell").nth(1)).toHaveText(statusBefore as string);
  await expect(approval.getByRole("cell").nth(2)).toHaveText("reviewed");
  await expect(approval).toContainText("web-ui");

  await page.goto("/review");
  await expect(page.getByTestId("review-row").first().or(page.getByText("Nothing to review"))).toBeVisible();
  await expect(page.locator(`[data-testid="review-row"] a[href*="${invoice.id}"]`)).toHaveCount(0);
});
