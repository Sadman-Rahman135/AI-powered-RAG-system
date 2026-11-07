/**
 * Recorded, narrated walkthrough of the working application (brief deliverable 8). Not part of `npm run test:e2e`:
 * run it with `npm run test:walkthrough` against the live stack. It writes
 *   video/walkthrough.webm                 the video (1440x900, Chromium, slowMo 250 ms, on-screen captions)
 *   video/screenshots/*.png                key steps
 * and prints (and attaches to the report) a timeline of the captions, in video time.
 *
 * Story: dashboard -> upload a batch (digital PDF, scanned JPEG, Excel ledger, a truncated PDF and an empty file)
 * -> live processing -> the PDF invoice: fields with confidence and evidence -> validation issues -> the
 * cross-verification against the ledger shows the ledger row was keyed wrong -> correct the ledger row and approve
 * -> history and audit trail -> approve the PDF -> the scan's OCR text -> keyword then semantic search -> the
 * Compare view of the PDF and its ledger row, field by field.
 *
 * Before recording, earlier copies of these sample files (same bytes) are deleted so the run starts from a known
 * state; documents uploaded during the recording are kept as the walkthrough's end state.
 */
import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";

import { expect, test, type APIRequestContext, type Locator, type Page } from "@playwright/test";

import {
  PROCESSING_TIMEOUT_MS,
  datasetFile,
  deleteDocuments,
  documentRow,
  findDocumentsBySha,
  getDocument,
  getInvoice,
  uploadRow,
  waitForFinal,
  type UploadItem,
} from "./support";

const OUT_DIR = path.resolve(__dirname, "../../video");
const SHOTS_DIR = path.join(OUT_DIR, "screenshots");
const VIEWPORT = { width: 1440, height: 900 };

const PDF = "pdf/t1_digital_03.pdf"; // Pinecrest Fasteners, PFI-25824, CAD 180.04
const SCAN = "images/t2_scan_light_01.jpg"; // a scanned invoice, read by OCR
const LEDGER = "ledgers/xv_ledger.xlsx"; // 20 ledger rows; row PFI-25824 was keyed as 423.47 / 430.39
const TRUNCATED = "corrupt/truncated.pdf"; // a PDF cut short: accepted, then fails safely
const EMPTY = "corrupt/empty.pdf"; // 0 bytes: refused before upload
const NUMBER = "PFI-25824";

/** Vertical band of the viewport that is neither under the sticky header nor under the caption bar. */
const BAND_TOP = 72;
const BAND_BOTTOM_GAP = 160;

const timeline: string[] = [];
let startedAt = 0;
let shot = 0;

test.describe.configure({ mode: "serial" });

// ------------------------------------------------------------------------------------------------ overlay

function stamp(): string {
  const seconds = Math.floor((Date.now() - startedAt) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

/**
 * Installs (once per document, after hydration) the caption bar, a visible cursor that follows the mouse, and room
 * at the bottom of the page so anything can be scrolled clear of the caption. The caption sits bottom-left, away
 * from the toasts at the bottom right.
 */
async function ensureOverlay(page: Page) {
  await page.evaluate((gap) => {
    if (document.getElementById("wt-caption")) return;
    document.body.style.paddingBottom = `${gap + 40}px`;

    const bar = document.createElement("div");
    bar.id = "wt-caption";
    Object.assign(bar.style, {
      position: "fixed",
      left: "24px",
      bottom: "20px",
      width: "min(980px, calc(100vw - 48px))",
      padding: "13px 20px 14px",
      borderRadius: "10px",
      background: "rgba(13, 27, 48, 0.96)",
      color: "#ffffff",
      fontFamily: "inherit",
      boxShadow: "0 10px 30px rgba(0, 0, 0, 0.28)",
      zIndex: "2147483646",
      pointerEvents: "none",
      display: "none",
      lineHeight: "1.35",
    });
    const step = document.createElement("div");
    step.id = "wt-step";
    Object.assign(step.style, { fontSize: "12px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#9cc3ff", marginBottom: "3px" });
    const title = document.createElement("div");
    title.id = "wt-title";
    Object.assign(title.style, { fontSize: "22px", fontWeight: "600" });
    const detail = document.createElement("div");
    detail.id = "wt-detail";
    Object.assign(detail.style, { fontSize: "16px", color: "#d9e4f5", marginTop: "3px" });
    bar.append(step, title, detail);
    document.body.append(bar);

    const cursor = document.createElement("div");
    cursor.id = "wt-cursor";
    Object.assign(cursor.style, {
      position: "fixed",
      left: "0",
      top: "0",
      width: "24px",
      height: "24px",
      marginLeft: "-12px",
      marginTop: "-12px",
      borderRadius: "50%",
      border: "3px solid #f59e0b",
      background: "rgba(245, 158, 11, 0.22)",
      boxShadow: "0 0 0 1px rgba(255, 255, 255, 0.9)",
      zIndex: "2147483647",
      pointerEvents: "none",
      transition: "transform 110ms ease-out",
      transform: "translate(-100px, -100px)",
    });
    document.body.append(cursor);
    const at = { x: -100, y: -100 };
    const place = (scale: number) => {
      cursor.style.transform = `translate(${at.x}px, ${at.y}px) scale(${scale})`;
    };
    window.addEventListener(
      "mousemove",
      (event) => {
        at.x = event.clientX;
        at.y = event.clientY;
        place(1);
      },
      true,
    );
    window.addEventListener("mousedown", () => place(0.6), true);
    window.addEventListener("mouseup", () => place(1), true);
  }, BAND_BOTTOM_GAP);
}

/** Shows a caption and lets the viewer read it; records it in the timeline. */
async function caption(page: Page, title: string, detail = "", dwellMs = 4200) {
  await ensureOverlay(page);
  const at = stamp();
  timeline.push(`${at}  ${title}${detail ? `: ${detail}` : ""}`);
  await page.evaluate(
    ({ title, detail, step }) => {
      (document.getElementById("wt-step") as HTMLElement).textContent = step;
      (document.getElementById("wt-title") as HTMLElement).textContent = title;
      (document.getElementById("wt-detail") as HTMLElement).textContent = detail;
      (document.getElementById("wt-caption") as HTMLElement).style.display = "block";
    },
    { title, detail, step: `DocIntel walkthrough · ${at}` },
  );
  if (dwellMs > 0) await page.waitForTimeout(dwellMs);
}

/** A full-screen title card (opening and closing). */
async function card(page: Page, heading: string, lines: readonly string[], dwellMs: number) {
  await ensureOverlay(page);
  timeline.push(`${stamp()}  [title card] ${heading}`);
  await page.evaluate(
    ({ heading, lines }) => {
      const overlay = document.createElement("div");
      overlay.id = "wt-card";
      Object.assign(overlay.style, {
        position: "fixed",
        inset: "0",
        background: "rgba(13, 27, 48, 0.97)",
        color: "#ffffff",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        gap: "16px",
        zIndex: "2147483645",
        fontFamily: "inherit",
        textAlign: "center",
        padding: "48px",
      });
      const h = document.createElement("div");
      h.textContent = heading;
      Object.assign(h.style, { fontSize: "46px", fontWeight: "700", letterSpacing: "-0.01em" });
      overlay.append(h);
      for (const line of lines) {
        const p = document.createElement("div");
        p.textContent = line;
        Object.assign(p.style, { fontSize: "21px", color: "#c9d8ee", maxWidth: "1000px" });
        overlay.append(p);
      }
      document.body.append(overlay);
      const bar = document.getElementById("wt-caption");
      if (bar) bar.style.display = "none";
    },
    { heading, lines },
  );
  await page.waitForTimeout(dwellMs);
  await page.evaluate(() => document.getElementById("wt-card")?.remove());
}

/** Scroll an element into the visible band, smoothly: centred when it fits, or (`align: "top"`) at the band's top. */
async function bringIntoView(locator: Locator, align: "center" | "top" = "center") {
  await locator.waitFor();
  const moved = await locator.evaluate(
    (element, { top, gap, align }) => {
      const rect = element.getBoundingClientRect();
      const bottom = window.innerHeight - gap;
      const room = bottom - top;
      if (align === "center" && rect.top >= top && rect.bottom <= bottom) return false;
      const offset = align === "center" && rect.height < room ? (room - rect.height) / 2 : 0;
      const target = Math.max(0, window.scrollY + rect.top - top - offset);
      if (Math.abs(target - window.scrollY) < 4) return false;
      window.scrollTo({ top: target, behavior: "smooth" });
      return true;
    },
    { top: BAND_TOP, gap: BAND_BOTTOM_GAP, align },
  );
  if (moved) await locator.page().waitForTimeout(900);
}

/** Outline an element (inside its box, so containers do not clip it) while it is being talked about. */
async function spotlight(locator: Locator, scroll = true) {
  if (scroll) await bringIntoView(locator);
  await locator.evaluate((element) => {
    const el = element as HTMLElement;
    el.dataset.wtSpotlight = "1";
    el.style.outline = "3px solid #f59e0b";
    el.style.outlineOffset = "-2px";
  });
}

async function clearSpotlights(page: Page) {
  await page.evaluate(() => {
    for (const element of document.querySelectorAll<HTMLElement>("[data-wt-spotlight]")) {
      element.style.outline = "";
      element.style.outlineOffset = "";
      delete element.dataset.wtSpotlight;
    }
  });
}

async function screenshot(page: Page, name: string) {
  shot += 1;
  await page.screenshot({ path: path.join(SHOTS_DIR, `${String(shot).padStart(2, "0")}-${name}.png`) });
}

/** Type visibly (slowMo paces each key). */
async function typeInto(locator: Locator, text: string) {
  await bringIntoView(locator);
  await locator.click();
  await locator.fill("");
  await locator.pressSequentially(text);
}

// ------------------------------------------------------------------------------------------------ preparation

async function waitForIdleQueue(request: APIRequestContext) {
  await expect
    .poll(
      async () => {
        const stats = (await (await request.get("/api/v1/stats")).json()) as { documents: { by_status: Record<string, number> } };
        return stats.documents.by_status.queued + stats.documents.by_status.processing;
      },
      { message: "other documents are still being processed", timeout: 300_000, intervals: [2_000] },
    )
    .toBe(0);
}

test.beforeAll(async ({ request }) => {
  mkdirSync(SHOTS_DIR, { recursive: true });
  const earlier = (await Promise.all([PDF, SCAN, LEDGER, TRUNCATED].map((file) => findDocumentsBySha(request, file)))).flat();
  await deleteDocuments(request, earlier.map((document) => document.id));
  await waitForIdleQueue(request);
});

// ------------------------------------------------------------------------------------------------ the walkthrough

test("recorded walkthrough of the working application", async ({ browser, request }, testInfo) => {
  const context = await browser.newContext({ viewport: VIEWPORT, recordVideo: { dir: testInfo.outputPath("video"), size: VIEWPORT } });
  const page = await context.newPage();
  startedAt = Date.now();
  const nav = page.getByRole("navigation", { name: "Main" });

  // 1. Dashboard -------------------------------------------------------------------------------------------------
  await page.goto("/");
  await expect(page.getByTestId("documents-table")).toBeVisible();
  await card(page, "DocIntel", [
    "AI document intelligence for invoices and receipts",
    "PDFs, scans and photos, Excel and CSV  →  OCR  →  structured extraction with a local LLM  →  validation  →  search",
  ], 5000);
  await spotlight(page.getByTestId("stats-strip"));
  await caption(page, "The dashboard", "Live totals: documents, records to review, failures, open checks and processing time. The LLM runs locally.", 5000);
  await clearSpotlights(page);

  // 2. Upload a mixed batch --------------------------------------------------------------------------------------
  await spotlight(page.getByTestId("upload-dropzone"));
  await caption(page, "Upload a mixed batch", "A digital PDF invoice, a scanned JPEG, an Excel ledger, a truncated PDF and an empty file, in one go.", 4500);
  const uploaded = page.waitForResponse(
    (response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/documents",
  );
  await page.getByTestId("upload-input").setInputFiles([PDF, SCAN, LEDGER, TRUNCATED, EMPTY].map(datasetFile));
  const items = ((await (await uploaded).json()) as { items: UploadItem[] }).items;
  await clearSpotlights(page);
  const id = (name: string) => items.find((item) => item.filename === name)?.document_id as string;
  const pdfId = id("t1_digital_03.pdf");
  const scanId = id("t2_scan_light_01.jpg");
  const ledgerId = id("xv_ledger.xlsx");
  const truncatedId = id("truncated.pdf");
  expect([pdfId, scanId, ledgerId, truncatedId].every(Boolean), JSON.stringify(items)).toBe(true);

  await expect(page.getByTestId("upload-status")).toHaveText("Upload finished: 4 queued, 1 rejected.");
  await spotlight(uploadRow(page, "empty.pdf"));
  await caption(page, "One outcome per file", "The empty file is refused at once, with the reason. The other four are queued, each with a link.", 4500);
  await screenshot(page, "upload-results");
  await clearSpotlights(page);

  // 3. Live processing -------------------------------------------------------------------------------------------
  await bringIntoView(page.getByTestId("documents-table"));
  await caption(page, "Processing runs in the background", "Text layer or OCR, LLM extraction, validation, duplicate and cross-checks, search indexing. The table updates live.", 3000);
  await screenshot(page, "processing-live");
  // The story needs the PDF, the ledger and the failed file; the scan may take longer (OCR + LLM) and catches up later.
  const [pdfDoc, ledgerDoc, truncatedDoc] = await Promise.all([pdfId, ledgerId, truncatedId].map((documentId) => waitForFinal(request, documentId)));
  for (const documentId of [pdfId, ledgerId, truncatedId]) {
    await expect(documentRow(page, documentId)).toHaveAttribute("data-status", /^(completed|needs_review|failed)$/, { timeout: 30_000 });
  }
  expect(pdfDoc.status).toBe("needs_review");
  expect(truncatedDoc.status).toBe("failed");
  expect(ledgerDoc.invoices.length).toBe(20);
  const scanNow = await getDocument(request, scanId);
  const scanText =
    scanNow.status === "completed" || scanNow.status === "needs_review"
      ? `Scan: read by OCR, ${scanNow.status === "completed" ? "every check passed" : "needs review"}.`
      : "Scan: still in OCR and extraction, carrying on in the background.";
  await caption(page, "Seconds later", `PDF: needs review. Ledger: ${ledgerDoc.invoices.length} records. Truncated PDF: failed safely, with the reason. ${scanText}`, 2500);
  const scanFinished = scanNow.status === "completed" || scanNow.status === "needs_review";
  for (const documentId of scanFinished ? [truncatedId, ledgerId, scanId, pdfId] : [truncatedId, ledgerId, pdfId]) {
    await spotlight(documentRow(page, documentId));
    await page.waitForTimeout(1400);
  }
  await clearSpotlights(page);

  // 4. The PDF invoice: fields, confidence, evidence --------------------------------------------------------------
  await documentRow(page, pdfId).getByRole("link", { name: "t1_digital_03.pdf" }).click();
  await expect(page).toHaveURL(`/documents/${pdfId}`);
  const pdfCard = page.getByTestId("invoice-card").first();
  await expect(pdfCard).toBeVisible();
  await spotlight(pdfCard.getByTestId("confidence-badge").first(), false);
  await caption(page, "The extracted invoice", "Every field shows where it came from (rules, LLM or both) and a confidence score; the record has an overall score.", 5000);
  await clearSpotlights(page);
  const totalField = pdfCard.locator('[data-testid="invoice-field"][data-field="total"]');
  await spotlight(totalField);
  await pdfCard.getByRole("button", { name: "Evidence for Total" }).hover();
  await expect(page.getByText("Page 1", { exact: true }).last()).toBeVisible();
  await caption(page, "Evidence for each value", "The quote icon shows the exact text the total was read from, and its page. Values the text does not support are rejected.", 5000);
  await screenshot(page, "fields-confidence-evidence");
  await page.mouse.move(720, 160);
  await clearSpotlights(page);

  // 5. Validation issues -----------------------------------------------------------------------------------------
  await page.getByTestId("tab-validation").click();
  await expect(page.getByTestId("issue-item").first()).toBeVisible();
  // The caption names exactly these issues: one error (the ledger mismatch) and one warning (rules vs LLM on the supplier).
  const errorItems = page.locator('[data-testid="issue-group"][data-severity="error"] [data-testid="issue-item"]');
  const warningItems = page.locator('[data-testid="issue-group"][data-severity="warning"] [data-testid="issue-item"]');
  await expect(errorItems).toHaveCount(1);
  await expect(errorItems.first()).toHaveAttribute("data-code", "CROSS_VERIFICATION_MISMATCH");
  await expect(warningItems).toHaveCount(1);
  await expect(warningItems.first()).toHaveAttribute("data-code", "EXTRACTION_DISAGREEMENT");
  await expect(warningItems.first()).toContainText("supplier_name");
  await spotlight(page.locator('[data-testid="issue-group"][data-severity="error"]'));
  await caption(page, "Validation", "An error: the record differs from its row in the Excel ledger. A warning: the rules and the LLM read the supplier name differently.", 5500);
  await screenshot(page, "validation-issues");
  await clearSpotlights(page);

  // 6. Cross-verification: the ledger row was keyed wrong ----------------------------------------------------------
  await page.getByTestId("tab-cross").click();
  await page.mouse.move(720, 160);
  const check = page.getByTestId("cross-verification").first();
  await expect(check.getByTestId("match-badge").first()).toHaveAttribute("data-status", "mismatch");
  const mismatch = check.getByTestId("compare-result");
  await bringIntoView(mismatch);
  await spotlight(mismatch.locator('[data-testid="compare-field-row"][data-field="subtotal"]'), false);
  await spotlight(mismatch.locator('[data-testid="compare-field-row"][data-field="total"]'), false);
  await caption(page, "Cross-verification: PDF vs Excel ledger", "Same invoice number, but subtotal and total differ by 250.35. The PDF matches the printed invoice: the ledger row was keyed wrong.", 6500);
  await screenshot(page, "cross-verification-mismatch");
  await clearSpotlights(page);

  // 7. Correct the ledger row and approve ----------------------------------------------------------------------------
  const rowId = ledgerDoc.invoices.find((record) => record.invoice_number === NUMBER)?.id as string;
  expect(rowId, `ledger row ${NUMBER}`).toBeTruthy();
  await page.goto(`/documents/${ledgerId}?invoice=${rowId}`);
  const rowCard = page.locator(`#record-${rowId}`);
  await expect(rowCard).toBeVisible();
  await ensureOverlay(page);
  await spotlight(rowCard.getByText("Sheet Domestic, row 6"), false);
  await caption(page, "The ledger row", `xv_ledger.xlsx gave ${ledgerDoc.invoices.length} records, one per row. This is ${NUMBER}, sheet Domestic, row 6.`, 4000);
  await clearSpotlights(page);

  const subtotalField = rowCard.locator('[data-testid="invoice-field"][data-field="subtotal"]');
  const rowTotalField = rowCard.locator('[data-testid="invoice-field"][data-field="total"]');
  await typeInto(rowCard.getByTestId("field-input-subtotal"), "173.12");
  await typeInto(rowCard.getByTestId("field-input-total"), "180.04");
  await spotlight(subtotalField, false);
  await spotlight(rowTotalField, false);
  await caption(page, "Correct the two amounts", "Edited fields are marked, with the value they replace.", 3500);
  await clearSpotlights(page);
  const comment = rowCard.getByLabel("Comment (optional)");
  await bringIntoView(comment);
  await comment.fill("Ledger keyed wrong: amounts corrected from the PDF invoice");
  await spotlight(comment, false);
  await caption(page, "Say why", "The comment goes into the audit trail with the change.", 3000);
  await clearSpotlights(page);
  await rowCard.getByTestId("save-invoice").click();
  await expect(page.getByText("Checked again: every check passes.")).toBeVisible();
  // Show the saved amounts while the "Changes saved" toast is still up.
  await bringIntoView(subtotalField, "top");
  await spotlight(subtotalField, false);
  await spotlight(rowTotalField, false);
  await caption(page, "Saved and re-validated on the server", "The values are now manual (100 %). Both records were checked again: the mismatch is gone from the ledger row and from the PDF.", 0);
  await screenshot(page, "correction-saved");
  await page.waitForTimeout(5500);
  await clearSpotlights(page);
  const approveRow = rowCard.getByTestId("approve-invoice");
  await bringIntoView(approveRow);
  await approveRow.click();
  await expect(rowCard).toHaveAttribute("data-review-status", "reviewed");
  await expect(rowCard.getByText(/^Reviewed by web-ui on /)).toBeVisible();
  await spotlight(rowCard.getByText(/^Reviewed by web-ui on /), false);
  await caption(page, "Approved", "The record is marked reviewed, with who approved it and when.", 3500);
  await clearSpotlights(page);

  // 8. History and audit trail ------------------------------------------------------------------------------------
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await page.waitForTimeout(700);
  await page.getByTestId("tab-history").click();
  await expect(page.getByTestId("events-timeline")).toBeVisible();
  await page.getByLabel("Record", { exact: true }).click();
  await page.getByRole("option", { name: new RegExp(NUMBER) }).click();
  const audit = page.getByTestId("corrections-table");
  await expect(audit.getByRole("row", { name: /^Total 430\.39 180\.04 / })).toBeVisible();
  // The correction re-validated the row to auto-approved; the approval records that as its "before".
  await expect(audit.getByRole("row", { name: /^Review status \S+ reviewed / })).toBeVisible();
  await spotlight(audit);
  await caption(page, "History and audit trail", "Processing events on the left. Every correction and the approval on the right: field, before, after, comment, who and when.", 6500);
  await screenshot(page, "history-audit");
  await clearSpotlights(page);

  // 9. Back on the PDF: the cross-check now matches; approve it --------------------------------------------------------
  await page.goto(`/documents/${pdfId}?tab=cross`);
  const fixed = page.getByTestId("cross-verification").first();
  await expect(fixed.getByTestId("match-badge").first()).toHaveAttribute("data-status", "match");
  await spotlight(fixed);
  await caption(page, "Back on the PDF", "The cross-check against the ledger row now matches.", 3500);
  await clearSpotlights(page);
  await page.getByTestId("tab-data").click();
  const approvePdf = page.getByTestId("invoice-card").first().getByTestId("approve-invoice");
  const supplierField = page.getByTestId("invoice-card").first().locator('[data-testid="invoice-field"][data-field="supplier_name"]');
  await expect(supplierField.locator("input")).toHaveValue("Pinecrest Fasteners Inc."); // the name printed on the invoice
  await spotlight(supplierField);
  await caption(page, "Approve the PDF invoice", "Its only warning: the rules and the LLM read the supplier name differently. The name kept, Pinecrest Fasteners Inc., is the one printed on the invoice.", 4500);
  await clearSpotlights(page);
  await bringIntoView(approvePdf);
  await approvePdf.click();
  await expect(page.getByTestId("invoice-card").first()).toHaveAttribute("data-review-status", "reviewed");
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await page.waitForTimeout(700);
  await expect(page.getByRole("heading", { level: 1 }).getByTestId("status-badge")).toHaveAttribute("data-status", "completed");
  await spotlight(page.getByRole("heading", { level: 1 }).getByTestId("status-badge"), false);
  await caption(page, "Reviewed", "With its record approved, the document is complete.", 3000);
  await clearSpotlights(page);

  // 10. The scanned JPEG: OCR -------------------------------------------------------------------------------------
  await nav.getByRole("link", { name: "Documents" }).click();
  await expect(page.getByTestId("documents-table")).toBeVisible();
  const scanRow = documentRow(page, scanId);
  const scanDone = await getDocument(request, scanId);
  if (scanDone.status === "queued" || scanDone.status === "processing") {
    await spotlight(scanRow);
    await caption(page, "The scan is still being read", "OCR and LLM extraction of a photographed or scanned page take longer than a text PDF.", 0);
  }
  await waitForFinal(request, scanId, PROCESSING_TIMEOUT_MS);
  await expect(scanRow).toHaveAttribute("data-status", /^(completed|needs_review)$/, { timeout: 30_000 });
  await spotlight(scanRow);
  await caption(page, "The scanned JPEG is done too", "Read by OCR, extracted and validated like the others.", 3000);
  await clearSpotlights(page);
  await scanRow.getByRole("link", { name: "t2_scan_light_01.jpg" }).click();
  // Client-side navigation keeps the previous document page mounted (hidden): use the visible one.
  await page.getByTestId("tab-text").filter({ visible: true }).click();
  const ocrPage = page.getByTestId("page-text").filter({ visible: true }).first();
  await expect(ocrPage).toContainText("OCR confidence");
  await spotlight(ocrPage.locator("header"));
  await caption(page, "Text read by OCR", "Page text with the method and the OCR confidence: the extraction and the search index work from this.", 5000);
  await clearSpotlights(page);

  // 11. Search: keyword, then semantic ------------------------------------------------------------------------------
  await nav.getByRole("link", { name: "Search" }).click();
  await expect(page.getByTestId("search-input")).toBeVisible();
  const searchButton = page.getByRole("search", { name: "Search documents" }).getByRole("button", { name: "Search", exact: true });
  await typeInto(page.getByTestId("search-input"), "Pinecrest");
  await page.getByTestId("search-mode").getByRole("button", { name: "Keyword", exact: true }).click();
  await searchButton.click();
  await expect(page.getByTestId("search-summary")).toContainText("keyword search");
  await expect(page.getByTestId("search-results").locator("mark").first()).toBeVisible();
  await spotlight(page.getByTestId("search-result").first());
  await caption(page, "Keyword search", "Exact words and numbers, across PDF text, OCR text and ledger rows. Matches are highlighted.", 5500);
  await screenshot(page, "search-keyword");
  await clearSpotlights(page);

  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await typeInto(page.getByTestId("search-input"), "cutting discs for grinders");
  await page.getByTestId("search-mode").getByRole("button", { name: "Semantic", exact: true }).click();
  await searchButton.click();
  await expect(page.getByTestId("search-summary")).toContainText("ranked by similarity");
  const top = page.getByTestId("search-result").first();
  await expect(top).toContainText(NUMBER);
  await spotlight(top);
  await caption(page, "Semantic search", "By meaning: “cutting discs for grinders” ranks the Pinecrest invoice first without naming the supplier.", 5500);
  await screenshot(page, "search-semantic");
  await clearSpotlights(page);

  // 12. Cross-verification view: PDF vs Excel ledger, field by field ---------------------------------------------------
  await nav.getByRole("link", { name: "Compare" }).click();
  await expect(page.getByTestId("compare-left-search")).toBeVisible();
  await caption(page, "Compare any two records", "The PDF invoice on one side, its Excel ledger row on the other.", 2500);
  for (const [side, source] of [
    ["left", "t1_digital_03.pdf"],
    ["right", "xv_ledger.xlsx"],
  ] as const) {
    await typeInto(page.getByTestId(`compare-${side}-search`), NUMBER);
    const options = page.getByTestId(`compare-${side}`).getByTestId("compare-option");
    // The picker keeps its previous list while the debounced search runs: wait for this query's results.
    await expect(options.filter({ hasNotText: NUMBER })).toHaveCount(0);
    await options.filter({ hasText: source }).first().click();
    await expect(page.getByTestId(`compare-${side}-selected`)).toContainText(source);
  }
  await page.getByTestId("compare-submit").click();
  const comparison = page.getByTestId("compare-result");
  await expect(comparison).toHaveAttribute("data-status", "match");
  await spotlight(comparison);
  await caption(page, "Cross-verification view: PDF vs Excel ledger", "Field by field, a tick for every match: after the correction the ledger row agrees with the invoice.", 7000);
  await screenshot(page, "compare-pdf-vs-ledger");
  await clearSpotlights(page);

  await card(page, "End of walkthrough", [
    "Upload  →  OCR and extraction  →  validation  →  cross-verification  →  correction with audit  →  approval  →  search",
    "Everything ran live: RapidOCR, a local Qwen3-4B model, PostgreSQL with pgvector, a FastAPI API and a Next.js UI.",
  ], 5000);
  timeline.push(`${stamp()}  [end]`);

  // Finish the recording and copy the video out of Playwright's output directory.
  const video = page.video();
  expect(video).not.toBeNull();
  await context.close();
  await video?.saveAs(path.join(OUT_DIR, "walkthrough.webm"));
  copyFileSync(path.join(OUT_DIR, "walkthrough.webm"), testInfo.outputPath("walkthrough.webm"));
  await testInfo.attach("timeline.txt", { body: timeline.join("\n"), contentType: "text/plain" });
  // Printed so the run log carries the timestamps for video/README.md.
  console.log(`walkthrough timeline:\n${timeline.join("\n")}`);

  // The end state matches what the video showed.
  expect((await getInvoice(request, rowId)).review_status).toBe("reviewed");
  expect((await getDocument(request, pdfId)).status).toBe("completed");
});
