/**
 * Every page loads and renders its main content without console errors, uncaught exceptions or CSP violations:
 * the dashboard, review queue, search (empty and with results), compare, a document with each of its tabs, and
 * the not-found states. Header navigation works client-side.
 */
import { expect, test } from "@playwright/test";

import { collectPageErrors, deleteDocuments, uploadAndProcess } from "./support";

/** The browser logs every non-2xx response; the not-found pages are expected to produce one. */
const NOT_FOUND_RESOURCE = /Failed to load resource: the server responded with a status of 404/;

const PAGES = [
  { path: "/", heading: "Documents", content: "documents-table" },
  { path: "/review", heading: "Review queue", content: null },
  { path: "/search", heading: "Search", content: "search-input" },
  { path: "/search?q=invoice&mode=hybrid", heading: "Search", content: "search-summary" },
  { path: "/compare", heading: "Compare", content: "compare-left" },
] as const;

let documentId: string;
let uploaded: string | null = null;

test.beforeAll(async ({ request }) => {
  // Any finished document will do; upload one only when the database has none.
  const response = await request.get("/api/v1/documents", { params: { limit: 50 } });
  expect(response.ok()).toBeTruthy();
  const items = ((await response.json()) as { items: Array<{ id: string; status: string; invoice_count: number }> }).items;
  const existing = items.find((item) => ["completed", "needs_review"].includes(item.status) && item.invoice_count > 0);
  if (existing) {
    documentId = existing.id;
  } else {
    const { documents } = await uploadAndProcess(request, ["pdf/t1_digital_05.pdf"]);
    documentId = uploaded = documents[0].id;
  }
});

test.afterAll(async ({ request }) => {
  if (uploaded) await deleteDocuments(request, [uploaded]);
});

for (const { path, heading, content } of PAGES) {
  test(`${path} loads without console errors`, async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading, level: 1 })).toBeVisible();
    if (content) await expect(page.getByTestId(content)).toBeVisible();
    else await expect(page.getByTestId("review-table").or(page.getByText("Nothing to review"))).toBeVisible();
    await page.waitForLoadState("networkidle");
    expect(errors).toEqual([]);
  });
}

test("a document page and each of its tabs load without console errors", async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.goto(`/documents/${documentId}`);
  await expect(page.getByTestId("download-original")).toBeVisible();
  await expect(page.getByTestId("invoice-card").first()).toBeVisible();

  await page.getByTestId("tab-validation").click();
  await expect(page.getByTestId("issue-group").first().or(page.getByText(/every check passed/i))).toBeVisible();
  await page.getByTestId("tab-text").click();
  await expect(page.getByTestId("page-text").first()).toBeVisible();
  await page.getByTestId("tab-history").click();
  await expect(page.getByTestId("events-timeline")).toBeVisible();
  await page.getByTestId("tab-cross").click();
  await expect(page.getByTestId("compare-with")).toBeVisible();
  await page.getByTestId("tab-data").click();
  await expect(page.getByTestId("invoice-editor").first()).toBeVisible();

  await page.waitForLoadState("networkidle");
  expect(errors).toEqual([]);
});

test("header navigation moves between pages client-side", async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Main" });
  for (const [label, path, heading] of [
    ["Review queue", "/review", "Review queue"],
    ["Search", "/search", "Search"],
    ["Compare", "/compare", "Compare"],
    ["Documents", "/", "Documents"],
  ] as const) {
    await nav.getByRole("link", { name: label }).click();
    await expect(page).toHaveURL(path);
    await expect(page.getByRole("heading", { name: heading, level: 1 })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test("unknown pages and documents show a not-found state", async ({ page }) => {
  const errors = collectPageErrors(page, [NOT_FOUND_RESOURCE]);
  await page.goto("/no-such-page");
  await expect(page.getByText("This page does not exist")).toBeVisible();

  await page.goto("/documents/00000000-0000-4000-8000-000000000000");
  await expect(page.getByText("This document does not exist")).toBeVisible();
  await page.getByRole("link", { name: "Back to documents" }).click();
  await expect(page.getByRole("heading", { name: "Documents", level: 1 })).toBeVisible();
  expect(errors).toEqual([]);
});
