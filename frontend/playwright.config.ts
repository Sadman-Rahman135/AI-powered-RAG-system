/**
 * Browser end-to-end tests against the running compose stack (`docker compose up`): the UI at E2E_BASE_URL
 * (default http://127.0.0.1:3000) with its same-origin API proxy, the API, the worker with real OCR and the local
 * LLM. Nothing is mocked, so processing waits are long.
 *
 * - `npm run test:e2e`         upload, review, search, compare and smoke specs (project "e2e").
 * - `npm run test:walkthrough` records the narrated walkthrough video into video/ (project "walkthrough").
 *
 * The walkthrough overwrites video/*, so its project exists only when WALKTHROUGH=1 is set or
 * `npm run test:walkthrough` is used. A bare `npx playwright test` therefore runs the e2e specs only.
 *
 * Reports, traces and raw videos go to E2E_OUTPUT_DIR (default: <os temp>/docintel-e2e), outside the repository.
 */
import os from "node:os";
import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000";
const outputRoot = process.env.E2E_OUTPUT_DIR ?? path.join(os.tmpdir(), "docintel-e2e");
// Opt-in, env-based only (workers re-evaluate this file without the CLI args): WALKTHROUGH=1, or running through
// `npm run test:walkthrough` (npm sets npm_lifecycle_event, which workers inherit).
const walkthroughSelected =
  process.env.WALKTHROUGH === "1" || process.env.npm_lifecycle_event === "test:walkthrough";
const viewport = { width: 1440, height: 900 };

export default defineConfig({
  testDir: "./e2e",
  outputDir: path.join(outputRoot, "test-results"),
  // One shared live database and one GPU: run specs one at a time, in file order.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: Boolean(process.env.CI),
  timeout: 300_000,
  expect: { timeout: 15_000 },
  reporter: [["list"], ["html", { open: "never", outputFolder: path.join(outputRoot, "report") }]],
  use: {
    baseURL,
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "e2e",
      testIgnore: /walkthrough\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], viewport },
    },
    ...(walkthroughSelected
      ? [
          {
            name: "walkthrough",
            testMatch: /walkthrough\.spec\.ts/,
            timeout: 900_000,
            use: { ...devices["Desktop Chrome"], viewport, launchOptions: { slowMo: 250 } },
          },
        ]
      : []),
  ],
});
