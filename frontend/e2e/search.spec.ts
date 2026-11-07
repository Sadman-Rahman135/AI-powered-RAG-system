/**
 * Search over the uploaded invoice: keyword, semantic (by meaning, with none of the supplier's words) and
 * filter-only. Matches marked « » by the API are rendered as <mark> elements, never as literal guillemets.
 */
import { expect, test, type Locator, type Page } from "@playwright/test";

import { deleteDocuments, uploadAndProcess, type DocumentDetail, type Invoice } from "./support";

const SAMPLE = "pdf/t3_digital_01.pdf"; // Thames Valley Graphics Ltd -> EDWARDS & SONS LTD, EUR 28,827.65

let document: DocumentDetail;
let invoice: Invoice;

test.describe.configure({ mode: "serial" });

test.beforeAll(async ({ request }) => {
  const { documents } = await uploadAndProcess(request, [SAMPLE]);
  document = documents[0];
  invoice = document.invoices[0];
});

test.afterAll(async ({ request }) => {
  if (document) await deleteDocuments(request, [document.id]);
});

/** Results that open the uploaded document. */
function ourResults(page: Page): Locator {
  return page.getByTestId("search-result").filter({ has: page.locator(`a[href^="/documents/${document.id}"]`) });
}

async function search(page: Page, query: string, mode: "Hybrid" | "Keyword" | "Semantic") {
  const form = page.getByRole("search", { name: "Search documents" });
  await page.getByTestId("search-input").fill(query);
  const modeButton = page.getByTestId("search-mode").getByRole("button", { name: mode, exact: true });
  await modeButton.click();
  await expect(modeButton).toHaveAttribute("aria-pressed", "true");
  await form.getByRole("button", { name: "Search", exact: true }).click();
}

async function expectNoLiteralGuillemets(results: Locator) {
  for (const text of await results.allInnerTexts()) {
    expect(text).not.toMatch(/[«»]/);
  }
}

test("the uploaded invoice was extracted with the supplier the searches look for", () => {
  expect(document.status).not.toBe("failed");
  expect(invoice?.supplier_name).toMatch(/Thames Valley Graphics/i);
});

test("keyword search finds the invoice and highlights the matched words", async ({ page }) => {
  await page.goto("/search");
  await search(page, "Thames Valley", "Keyword");

  await expect(page).toHaveURL(/[?&]mode=keyword/);
  await expect(page.getByTestId("search-summary")).toContainText("keyword search");
  const hits = ourResults(page);
  await expect(hits.first()).toBeVisible();
  const hit = hits.first();
  await expect(hit.getByTestId("match-type")).toHaveText("Keyword");
  await expect(hit.getByTestId("score")).toHaveText(/^\d+\.\d{3}$/);
  await expect(hit).toContainText("Thames Valley Graphics");

  // « » from the API became <mark> elements around the query words.
  const marks = hit.locator("mark");
  await expect(marks.first()).toBeVisible();
  for (const text of await marks.allInnerTexts()) expect(text).toMatch(/^(thames|valley)$/i);
  await expectNoLiteralGuillemets(page.getByTestId("search-results"));

  // The hit opens the record it came from.
  await hit.getByRole("link").first().click();
  await expect(page).toHaveURL(new RegExp(`/documents/${document.id}\\?(invoice|tab)=`));
  await expect(page.getByRole("heading", { level: 1 })).toContainText("t3_digital_01.pdf");
});

test("semantic search finds the invoice by meaning, without its words", async ({ page }) => {
  await page.goto("/search");
  await search(page, "exhibition stand and printed marketing material from a design studio", "Semantic");

  await expect(page).toHaveURL(/[?&]mode=semantic/);
  await expect(page.getByTestId("search-summary")).toContainText("ranked by similarity");
  // Every candidate is listed for semantic search, so being listed proves little: it must rank first.
  const top = page.getByTestId("search-result").first();
  await expect(top.locator(`a[href^="/documents/${document.id}"]`).first()).toBeVisible();
  const hit = ourResults(page).first();
  await expect(hit.getByTestId("match-type")).toHaveText("Semantic");
  await expect(hit.getByTestId("score")).toBeVisible();
  // Any highlighted words in semantic snippets are rendered as marks too.
  await expectNoLiteralGuillemets(page.getByTestId("search-results"));
});

test("filter-only search (blank query) returns the invoice as a filter match", async ({ page }) => {
  await page.goto("/search");
  await page.getByLabel("Supplier", { exact: true }).fill("thames valley graphics");
  await page.getByLabel("Currency", { exact: true }).fill("eur");
  await expect(page.getByLabel("Currency", { exact: true })).toHaveValue("EUR");
  await page.getByLabel("Minimum total", { exact: true }).fill("28000");
  await page.getByRole("search", { name: "Search documents" }).getByRole("button", { name: "Search", exact: true }).click();

  await expect(page).toHaveURL(/[?&]supplier=thames(\+|%20)valley(\+|%20)graphics/);
  await expect(page.getByTestId("search-summary")).toHaveText(/^\d+ results? matching the filters$/);
  const hit = ourResults(page).first();
  await expect(hit).toBeVisible();
  await expect(hit.getByTestId("match-type")).toHaveText("Filter match");
  await expect(hit.getByTestId("score")).toHaveCount(0); // filter matches are not ranked
  await expect(hit).toContainText("€28,827.65");
  await expectNoLiteralGuillemets(page.getByTestId("search-results"));

  // A filter that excludes it leaves it out.
  await page.getByLabel("Minimum total", { exact: true }).fill("30000");
  await page.getByRole("search", { name: "Search documents" }).getByRole("button", { name: "Search", exact: true }).click();
  await expect(page).toHaveURL(/[?&]total_min=30000/);
  await expect(ourResults(page)).toHaveCount(0);
});
