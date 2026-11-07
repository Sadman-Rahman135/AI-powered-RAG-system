import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

interface PaginationBarProps {
  /** 1-based page. */
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  /** Plural noun, e.g. "documents"; `nounOne` is used for a single item. */
  noun?: string;
  nounOne?: string;
  className?: string;
}

/** "21–40 of 1,204 documents" with previous / next; just "8 documents" when everything fits on one page. */
export function PaginationBar({
  page,
  pageSize,
  total,
  onPageChange,
  noun = "items",
  nounOne = noun.replace(/s$/, ""),
  className,
}: PaginationBarProps) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return (
    <nav aria-label="Pagination" className={cn("flex flex-wrap items-center justify-between gap-3", className)}>
      <p className="text-sm text-muted-foreground">
        {pages > 1 ? (
          <>
            <span className="figures">
              {formatCount(first)}–{formatCount(last)}
            </span>{" "}
            of{" "}
          </>
        ) : null}
        <span className="figures">{formatCount(total)}</span> {total === 1 ? nounOne : noun}
      </p>
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
          aria-label="Previous page"
        >
          <ChevronLeft aria-hidden="true" data-icon="inline-start" />
          Previous
        </Button>
        <span className="figures text-sm text-muted-foreground">
          {page} / {pages}
        </span>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={page >= pages}
          onClick={() => onPageChange(page + 1)}
          aria-label="Next page"
        >
          Next
          <ChevronRight aria-hidden="true" data-icon="inline-end" />
        </Button>
      </div>
    </nav>
  );
}
