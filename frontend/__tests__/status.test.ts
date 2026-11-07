import { describe, expect, it } from "vitest";

import { POLL_INTERVAL_MS, isActiveDocument, refetchWhileActive } from "@/lib/status";

import { makeDocument, makeJob } from "./fixtures";

describe("isActiveDocument", () => {
  it("is active while queued or processing", () => {
    expect(isActiveDocument(makeDocument({ status: "queued" }))).toBe(true);
    expect(isActiveDocument(makeDocument({ status: "processing" }))).toBe(true);
  });

  it("is active while its job is still queued or running", () => {
    expect(isActiveDocument(makeDocument({ status: "completed", job: makeJob({ status: "queued" }) }))).toBe(true);
    expect(isActiveDocument(makeDocument({ status: "failed", job: makeJob({ status: "running" }) }))).toBe(true);
  });

  it("is settled once completed, needs review or failed", () => {
    expect(isActiveDocument(makeDocument({ status: "completed" }))).toBe(false);
    expect(isActiveDocument(makeDocument({ status: "needs_review", job: makeJob({ status: "succeeded" }) }))).toBe(false);
    expect(isActiveDocument(makeDocument({ status: "failed", job: makeJob({ status: "failed" }) }))).toBe(false);
  });
});

describe("refetchWhileActive", () => {
  it("polls every 2 s while anything is in flight and stops otherwise", () => {
    expect(POLL_INTERVAL_MS).toBe(2000);
    const interval = refetchWhileActive((items: ReturnType<typeof makeDocument>[] | undefined) => items ?? []);

    expect(interval(undefined)).toBe(false);
    expect(interval([makeDocument({ status: "completed" })])).toBe(false);
    expect(interval([makeDocument({ status: "completed" }), makeDocument({ status: "processing" })])).toBe(2000);
  });
});
