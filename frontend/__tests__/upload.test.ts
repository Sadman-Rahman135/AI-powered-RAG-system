import { describe, expect, it } from "vitest";

import { batchBySize, chunk, perFileProgress, precheckFile } from "@/lib/upload";

describe("chunk", () => {
  it("splits into batches of at most the given size", () => {
    const items = Array.from({ length: 120 }, (_, i) => i);
    expect(chunk(items, 50).map((b) => b.length)).toEqual([50, 50, 20]);
    expect(chunk([], 50)).toEqual([]);
    expect(chunk([1, 2], 50)).toEqual([[1, 2]]);
  });
});

describe("batchBySize", () => {
  const MB = 1024 * 1024;
  const size = (n: number) => n;

  it("splits by file count when no byte cap is known", () => {
    expect(batchBySize([1, 1, 1, 1, 1], size, 2).map((b) => b.length)).toEqual([2, 2, 1]);
  });

  it("splits by total bytes, keeping headroom for multipart overhead", () => {
    // 100 MB cap: three 40 MB files do not fit together (120 MB), two do not either with overhead at 50 MB each.
    expect(batchBySize([40 * MB, 40 * MB, 40 * MB], size, 50, 100 * MB).map((b) => b.length)).toEqual([2, 1]);
    // Exactly the cap leaves no room for framing, so the second file moves to the next request.
    expect(batchBySize([50 * MB, 50 * MB], size, 50, 100 * MB).map((b) => b.length)).toEqual([1, 1]);
  });

  it("applies both limits and preserves order", () => {
    const items = [10 * MB, 10 * MB, 10 * MB, 90 * MB, 1 * MB];
    expect(batchBySize(items, size, 2, 100 * MB)).toEqual([[10 * MB, 10 * MB], [10 * MB], [90 * MB, 1 * MB]]);
  });

  it("still sends a file larger than the cap, alone, so the server can reject it per file", () => {
    expect(batchBySize([1 * MB, 500 * MB, 1 * MB], size, 50, 100 * MB)).toEqual([[1 * MB], [500 * MB], [1 * MB]]);
  });

  it("returns no batches for no files", () => {
    expect(batchBySize([], size, 50, 100 * MB)).toEqual([]);
  });
});

describe("perFileProgress", () => {
  it("maps the request's byte progress onto the files in send order", () => {
    // Multipart overhead is spread proportionally: half the request = the first of two equal files.
    expect(perFileProgress([1000, 1000], 1100, 2200)).toEqual([1, 0]);
    expect(perFileProgress([1000, 1000], 1650, 2200)).toEqual([1, 0.5]);
    expect(perFileProgress([1000, 3000], 2200, 4400)).toEqual([1, 1 / 3]);
  });

  it("is 0 before anything is sent and 1 when everything is sent", () => {
    expect(perFileProgress([10, 20], 0, 40)).toEqual([0, 0]);
    expect(perFileProgress([10, 20], 40, 40)).toEqual([1, 1]);
  });

  it("copes with an unknown total and with empty files", () => {
    expect(perFileProgress([10, 20], 5, 0)).toEqual([0, 0]);
    expect(perFileProgress([0, 10], 10, 10)).toEqual([1, 1]);
  });
});

describe("precheckFile", () => {
  const limits = { maxUploadMb: 200, allowedExtensions: [".pdf", ".png", ".xlsx", ".csv"] };

  it("accepts allowed types regardless of extension case", () => {
    expect(precheckFile(new File(["x"], "SCAN.PDF"), limits)).toBeNull();
    expect(precheckFile(new File(["x"], "ledger.xlsx"), limits)).toBeNull();
  });

  it("explains why a file cannot be uploaded", () => {
    expect(precheckFile(new File(["x"], "virus.exe"), limits)).toMatch(/not a supported file type/i);
    expect(precheckFile(new File(["x"], "no-extension"), limits)).toMatch(/not a supported file type/i);
    expect(precheckFile(new File([], "empty.pdf"), limits)).toMatch(/empty/i);
    const big = new File(["x"], "big.pdf");
    Object.defineProperty(big, "size", { value: 200 * 1024 * 1024 + 1 });
    expect(precheckFile(big, limits)).toMatch(/larger than 200 MB/i);
  });
});
