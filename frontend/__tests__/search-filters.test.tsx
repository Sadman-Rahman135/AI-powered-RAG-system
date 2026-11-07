import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SearchFilters } from "@/components/search-filters";
import {
  DEFAULT_SEARCH_STATE,
  hasSearchCriteria,
  searchApiParams,
  searchStateFromQuery,
  searchStateToQuery,
  type SearchState,
} from "@/lib/search-params";

import { renderWithProviders } from "./test-utils";

describe("search state <-> query string", () => {
  it("serializes only the criteria that are set, in a stable order", () => {
    const state: SearchState = {
      ...DEFAULT_SEARCH_STATE,
      q: "  acme freight ",
      mode: "keyword",
      supplier: "Acme",
      currency: "eur",
      total_min: "100",
      review_status: "needs_review",
    };

    expect(searchStateToQuery(state)).toBe(
      "q=acme+freight&mode=keyword&supplier=Acme&currency=EUR&total_min=100&review_status=needs_review",
    );
  });

  it("omits the default mode and the first page", () => {
    expect(searchStateToQuery({ ...DEFAULT_SEARCH_STATE, q: "freight" })).toBe("q=freight");
    expect(searchStateToQuery({ ...DEFAULT_SEARCH_STATE, q: "freight", page: 3 })).toBe("q=freight&page=3");
  });

  it("parses a query string back into the same state", () => {
    const state: SearchState = {
      ...DEFAULT_SEARCH_STATE,
      q: "pallet",
      mode: "semantic",
      customer: "Northwind",
      invoice_number: "INV-0042",
      date_from: "2024-01-01",
      date_to: "2024-03-31",
      total_max: "5000",
      file_kind: "xlsx",
      page: 2,
    };

    expect(searchStateFromQuery(searchStateToQuery(state))).toEqual(state);
  });

  it("ignores unknown or invalid values in the URL", () => {
    const state = searchStateFromQuery("mode=fuzzy&review_status=maybe&file_kind=exe&page=-4&utm_source=x");

    expect(state).toEqual(DEFAULT_SEARCH_STATE);
  });

  it("knows when there is something to search for", () => {
    expect(hasSearchCriteria(DEFAULT_SEARCH_STATE)).toBe(false);
    expect(hasSearchCriteria({ ...DEFAULT_SEARCH_STATE, mode: "keyword" })).toBe(false);
    expect(hasSearchCriteria({ ...DEFAULT_SEARCH_STATE, q: " " })).toBe(false);
    expect(hasSearchCriteria({ ...DEFAULT_SEARCH_STATE, supplier: "Acme" })).toBe(true);
  });

  it("maps the state to API parameters with limit and offset", () => {
    expect(searchApiParams({ ...DEFAULT_SEARCH_STATE, q: "acme", page: 3 }, 20)).toEqual({
      q: "acme",
      mode: "hybrid",
      limit: 20,
      offset: 40,
    });
  });
});

describe("SearchFilters", () => {
  it("submits the query, mode and filters; the result serializes to a query string", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(state: SearchState) => void>();
    renderWithProviders(<SearchFilters value={DEFAULT_SEARCH_STATE} onSubmit={onSubmit} />);

    await user.type(screen.getByTestId("search-input"), "pallet transport");
    await user.click(screen.getByRole("button", { name: "Keyword" }));
    await user.type(screen.getByLabelText("Supplier"), "Acme");
    await user.type(screen.getByLabelText("Minimum total"), "100");
    await user.type(screen.getByLabelText("Currency"), "eur");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const submitted = onSubmit.mock.calls[0][0];
    expect(submitted).toMatchObject({ q: "pallet transport", mode: "keyword", supplier: "Acme", total_min: "100", page: 1 });
    expect(searchStateToQuery(submitted)).toBe("q=pallet+transport&mode=keyword&supplier=Acme&currency=EUR&total_min=100");
  });

  it("starts from the current state and resets the filters but keeps the query", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(state: SearchState) => void>();
    renderWithProviders(
      <SearchFilters value={{ ...DEFAULT_SEARCH_STATE, q: "acme", supplier: "Acme", page: 4 }} onSubmit={onSubmit} />,
    );

    expect(screen.getByTestId("search-input")).toHaveValue("acme");
    expect(screen.getByLabelText("Supplier")).toHaveValue("Acme");

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(searchStateToQuery(onSubmit.mock.calls[0][0])).toBe("q=acme");
  });

  it("marks the selected search mode", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SearchFilters value={{ ...DEFAULT_SEARCH_STATE, mode: "semantic" }} onSubmit={vi.fn()} />);

    const group = screen.getByTestId("search-mode");
    expect(screen.getByRole("button", { name: "Semantic" })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "Hybrid" }));
    expect(screen.getByRole("button", { name: "Hybrid" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Semantic" })).toHaveAttribute("aria-pressed", "false");

    // Pressing the active mode again must not leave the search without a mode.
    await user.click(screen.getByRole("button", { name: "Hybrid" }));
    expect(screen.getByRole("button", { name: "Hybrid" })).toHaveAttribute("aria-pressed", "true");
    expect(group).toBeInTheDocument();
  });
});
