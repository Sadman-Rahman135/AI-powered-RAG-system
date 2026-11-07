import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { StatsStrip } from "@/components/stats-strip";
import type { Stats } from "@/lib/types";

import { renderWithProviders } from "./test-utils";

const useStats = vi.fn();
vi.mock("@/lib/queries", () => ({ useStats: () => useStats() }));

function stats(llm: Partial<Stats["llm"]> = {}): Stats {
  return {
    documents: { total: 3, by_status: {} },
    invoices: { total: 5, by_review_status: {} },
    issues: { by_code: {}, by_severity: {} },
    processing: { avg_ms: null, p95_ms: null, pages_total: 4 },
    llm: { provider: "local", model: "qwen", fallback_documents: 0, fallback_records: 0, ...llm },
  };
}

describe("StatsStrip LLM fallback warning", () => {
  beforeEach(() => useStats.mockReset());

  it("warns when records fell back to rules", () => {
    useStats.mockReturnValue({ data: stats({ fallback_documents: 2, fallback_records: 7 }), isPending: false, isError: false });
    renderWithProviders(<StatsStrip />);

    expect(screen.getByTestId("llm-fallback-warning")).toHaveTextContent(/fell back to rules on 7 records \(2 documents\)/);
  });

  it("shows nothing when there were no fallbacks", () => {
    useStats.mockReturnValue({ data: stats(), isPending: false, isError: false });
    renderWithProviders(<StatsStrip />);

    expect(screen.queryByTestId("llm-fallback-warning")).toBeNull();
  });
});
