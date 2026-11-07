import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SearchResults } from "@/components/search-view";
import type { SearchHit, SearchResponse } from "@/lib/types";

import { renderWithProviders } from "./test-utils";

function hit(overrides: Partial<SearchHit> = {}): SearchHit {
  return {
    document_id: "doc-1",
    invoice_id: "inv-1",
    page_number: 1,
    score: 0.0328,
    snippet: "Invoice «INV-0042» from Acme",
    match_type: "hybrid",
    filename: "acme.pdf",
    invoice_number: "INV-0042",
    supplier_name: "Acme Freight GmbH",
    customer_name: "Northwind Traders",
    invoice_date: "2024-03-05",
    total: "119.00",
    currency: "EUR",
    review_status: "needs_review",
    ...overrides,
  };
}

function response(items: SearchHit[], overrides: Partial<SearchResponse> = {}): SearchResponse {
  return { items, total: items.length, took_ms: 42, mode: "hybrid", ...overrides };
}

describe("SearchResults", () => {
  it("shows a ranked hit with its match type, score and highlighted snippet", () => {
    renderWithProviders(<SearchResults data={response([hit()])} page={1} stale={false} filterOnly={false} onPageChange={vi.fn()} />);

    const result = screen.getByTestId("search-result");
    expect(within(result).getByTestId("match-type")).toHaveTextContent("Hybrid");
    expect(within(result).getByTestId("score")).toHaveTextContent("0.033");
    expect(result.querySelector("mark")).toHaveTextContent("INV-0042");
  });

  it("renders filter-only hits as a neutral filter match without a relevance score", () => {
    renderWithProviders(
      <SearchResults
        data={response([hit({ match_type: "filter", score: 0, snippet: "" })], { total: 1 })}
        page={1}
        stale={false}
        filterOnly
        onPageChange={vi.fn()}
      />,
    );

    const result = screen.getByTestId("search-result");
    expect(within(result).getByTestId("match-type")).toHaveTextContent("Filter match");
    expect(within(result).queryByTestId("score")).toBeNull();
    expect(screen.getByTestId("search-summary")).toHaveTextContent("1 result matching the filters");
  });

  it("counts semantic and hybrid hits as results, not as relevant matches", () => {
    renderWithProviders(
      <SearchResults data={response([hit(), hit({ invoice_id: "inv-2" })], { total: 57, mode: "semantic" })} page={1} stale={false} filterOnly={false} onPageChange={vi.fn()} />,
    );

    const summary = screen.getByTestId("search-summary");
    expect(summary).toHaveTextContent("57 results");
    expect(summary).toHaveTextContent("ranked by similarity");
    expect(summary).not.toHaveTextContent(/match/i);
  });
});
