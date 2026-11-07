"use client";

/**
 * Search form: query, mode (hybrid / keyword / semantic) and structured invoice filters. Holds a draft until
 * submitted; the page turns the submitted state into the URL (`searchStateToQuery`). Key this component by the
 * current query string so back/forward navigation resets the draft.
 */
import { SearchIcon } from "lucide-react";
import { useId, useState, type ComponentProps, type FormEvent } from "react";

import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { FILE_KIND_LABELS, REVIEW_STATUS_LABELS, SEARCH_MODE_LABELS } from "@/lib/format";
import { SEARCH_MODES, activeFilterCount, clearFilters, type SearchState } from "@/lib/search-params";
import type { SearchMode } from "@/lib/types";

const ANY = "any";

const REVIEW_ITEMS = [
  { value: ANY, label: "Any status" },
  ...Object.entries(REVIEW_STATUS_LABELS).map(([value, label]) => ({ value, label })),
];
const FILE_KIND_ITEMS = [
  { value: ANY, label: "Any type" },
  ...Object.entries(FILE_KIND_LABELS).map(([value, label]) => ({ value, label })),
];

const MODE_HINTS: Record<SearchMode, string> = {
  hybrid: "Words and meaning combined",
  keyword: "Exact words and numbers",
  semantic: "Similar meaning, even with different words",
};

type TextFilter = "supplier" | "customer" | "invoice_number" | "currency" | "date_from" | "date_to" | "total_min" | "total_max";

interface SearchFiltersProps {
  value: SearchState;
  onSubmit: (state: SearchState) => void;
  pending?: boolean;
}

export function SearchFilters({ value, onSubmit, pending = false }: SearchFiltersProps) {
  const [draft, setDraft] = useState<SearchState>(value);
  const id = useId();
  const filters = activeFilterCount(draft);

  function set<K extends keyof SearchState>(key: K, next: SearchState[K]) {
    setDraft((previous) => ({ ...previous, [key]: next }));
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit({ ...draft, page: 1 });
  }

  const textField = (key: TextFilter, label: string, props: ComponentProps<typeof Input> = {}) => (
    <Field>
      <FieldLabel htmlFor={`${id}-${key}`}>{label}</FieldLabel>
      <Input
        id={`${id}-${key}`}
        value={draft[key]}
        autoComplete="off"
        onChange={(event) => set(key, key === "currency" ? event.target.value.toUpperCase() : event.target.value)}
        {...props}
      />
    </Field>
  );

  return (
    <form role="search" aria-label="Search documents" onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <InputGroup className="h-9 min-w-64 flex-1">
            <InputGroupAddon>
              <SearchIcon aria-hidden="true" />
            </InputGroupAddon>
            <InputGroupInput
              data-testid="search-input"
              aria-label="Search text"
              type="search"
              value={draft.q}
              placeholder="Supplier, invoice number, amount or any words in the documents"
              onChange={(event) => set("q", event.target.value)}
            />
          </InputGroup>
          <Button type="submit" size="lg" disabled={pending}>
            Search
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span id={`${id}-mode`} className="text-sm font-medium">
            Match by
          </span>
          <ToggleGroup
            aria-labelledby={`${id}-mode`}
            data-testid="search-mode"
            variant="outline"
            size="sm"
            spacing={0}
            value={[draft.mode]}
            onValueChange={(next) => {
              const mode = next[0] as SearchMode | undefined;
              if (mode && SEARCH_MODES.includes(mode)) set("mode", mode);
            }}
          >
            {SEARCH_MODES.map((mode) => (
              <ToggleGroupItem key={mode} value={mode} title={MODE_HINTS[mode]}>
                {SEARCH_MODE_LABELS[mode]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <span className="text-sm text-muted-foreground">{MODE_HINTS[draft.mode]}</span>
        </div>
      </div>

      <FieldSet>
        <div className="flex items-center justify-between gap-2">
          <FieldLegend variant="label" className="mb-0">
            Filters{filters > 0 ? ` (${filters})` : ""}
          </FieldLegend>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={filters === 0}
            onClick={() => setDraft((previous) => clearFilters(previous))}
          >
            Clear filters
          </Button>
        </div>
        <FieldGroup className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {textField("supplier", "Supplier")}
          {textField("customer", "Customer")}
          {textField("invoice_number", "Invoice number", { className: "figures" })}
          {textField("currency", "Currency", { maxLength: 3, placeholder: "EUR", className: "figures" })}
          <div className="grid grid-cols-2 gap-3">
            {textField("date_from", "Date from", { type: "date" })}
            {textField("date_to", "Date to", { type: "date" })}
          </div>
          <div className="grid grid-cols-2 gap-3">
            {textField("total_min", "Minimum total", { inputMode: "decimal", className: "figures text-right" })}
            {textField("total_max", "Maximum total", { inputMode: "decimal", className: "figures text-right" })}
          </div>
          <Field>
            <FieldLabel htmlFor={`${id}-review`}>Review status</FieldLabel>
            <Select
              items={REVIEW_ITEMS}
              value={draft.review_status || ANY}
              onValueChange={(next) => set("review_status", next && next !== ANY ? (next as SearchState["review_status"]) : "")}
            >
              <SelectTrigger id={`${id}-review`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {REVIEW_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </Field>
          <Field>
            <FieldLabel htmlFor={`${id}-kind`}>File type</FieldLabel>
            <Select
              items={FILE_KIND_ITEMS}
              value={draft.file_kind || ANY}
              onValueChange={(next) => set("file_kind", next && next !== ANY ? (next as SearchState["file_kind"]) : "")}
            >
              <SelectTrigger id={`${id}-kind`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {FILE_KIND_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </Field>
        </FieldGroup>
      </FieldSet>
    </form>
  );
}
