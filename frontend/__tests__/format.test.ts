import { describe, expect, it } from "vitest";

import {
  EMPTY,
  confidenceBucket,
  fieldLabel,
  formatBytes,
  formatConfidence,
  formatDate,
  formatDuration,
  formatMoney,
  formatPercent,
  humanizeCode,
  plural,
} from "@/lib/format";

describe("formatMoney", () => {
  it("formats a decimal string with its currency", () => {
    expect(formatMoney("1234.5", "EUR")).toBe("€1,234.50");
    expect(formatMoney("110.00", "USD")).toBe("$110.00");
    expect(formatMoney("99", "GBP")).toBe("£99.00");
  });

  it("renders a missing amount as an em dash, never as zero", () => {
    expect(EMPTY).toBe("—");
    expect(formatMoney(null, "EUR")).toBe("—");
    expect(formatMoney(undefined, "EUR")).toBe("—");
    expect(formatMoney("", "EUR")).toBe("—");
    expect(formatMoney("   ", "EUR")).toBe("—");
  });

  it("formats without a symbol when the currency is unknown", () => {
    expect(formatMoney("1234.5", null)).toBe("1,234.50");
    expect(formatMoney("1234.5")).toBe("1,234.50");
  });

  it("keeps negative amounts and credit notes readable", () => {
    expect(formatMoney("-12", "USD")).toBe("-$12.00");
  });

  it("never loses precision to floating point", () => {
    expect(formatMoney("12345678901234567.89", "USD")).toBe("$12,345,678,901,234,567.89");
    expect(formatMoney("0.1234", "USD")).toBe("$0.1234");
    expect(formatMoney("110.0000", "GBP")).toBe("£110.00");
  });

  it("shows values it cannot parse verbatim instead of hiding them", () => {
    expect(formatMoney("12,50", "EUR")).toBe("12,50");
    expect(formatMoney("10", "zz")).toBe("10.00 zz");
  });
});

describe("confidenceBucket", () => {
  it("splits confidence into high / medium / low at 0.9 and the 0.6 review threshold", () => {
    expect(confidenceBucket(1)).toBe("high");
    expect(confidenceBucket(0.9)).toBe("high");
    expect(confidenceBucket(0.89)).toBe("medium");
    expect(confidenceBucket(0.6)).toBe("medium");
    expect(confidenceBucket(0.59)).toBe("low");
    expect(confidenceBucket(0)).toBe("low");
  });

  it("reports unknown when there is no confidence", () => {
    expect(confidenceBucket(null)).toBe("unknown");
    expect(confidenceBucket(undefined)).toBe("unknown");
    expect(confidenceBucket(Number.NaN)).toBe("unknown");
  });

  it("formats confidence as a whole percentage", () => {
    expect(formatConfidence(0.934)).toBe("93%");
    expect(formatConfidence(1)).toBe("100%");
    expect(formatConfidence(null)).toBe("—");
  });
});

describe("dates, sizes, durations", () => {
  it("formats ISO dates with a month name so day/month order is never ambiguous", () => {
    expect(formatDate("2024-03-05")).toBe("5 Mar 2024");
    expect(formatDate("2024-12-31")).toBe("31 Dec 2024");
    expect(formatDate(null)).toBe("—");
    expect(formatDate("not a date")).toBe("not a date");
  });

  it("formats byte sizes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1.2 * 1024 * 1024)).toBe("1.2 MB");
    expect(formatBytes(null)).toBe("—");
  });

  it("formats processing durations", () => {
    expect(formatDuration(850)).toBe("850 ms");
    expect(formatDuration(1800)).toBe("1.8 s");
    expect(formatDuration(125_000)).toBe("2 min 5 s");
    expect(formatDuration(null)).toBe("—");
  });

  it("formats tax rates as percentages", () => {
    expect(formatPercent("20")).toBe("20%");
    expect(formatPercent("7.5")).toBe("7.5%");
    expect(formatPercent(null)).toBe("—");
  });
});

describe("labels", () => {
  it("names invoice fields in plain words", () => {
    expect(fieldLabel("invoice_number")).toBe("Invoice number");
    expect(fieldLabel("tax_rate")).toBe("Tax rate");
    expect(fieldLabel("line_items")).toBe("Line items");
    expect(fieldLabel("something_new")).toBe("Something new");
  });

  it("pluralizes counts", () => {
    expect(plural(1, "error")).toBe("1 error");
    expect(plural(0, "warning")).toBe("0 warnings");
    expect(plural(1209, "page")).toBe("1,209 pages");
    expect(plural(2, "entry", "entries")).toBe("2 entries");
  });

  it("turns issue codes into sentence case", () => {
    expect(humanizeCode("TOTAL_MISMATCH")).toBe("Total mismatch");
    expect(humanizeCode("LINE_ITEMS_SUM_MISMATCH")).toBe("Line items sum mismatch");
    expect(humanizeCode("OCR_LOW_QUALITY")).toBe("OCR low quality");
    expect(humanizeCode("LLM_UNAVAILABLE")).toBe("LLM unavailable");
  });
});
