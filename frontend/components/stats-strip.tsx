"use client";

/**
 * The day at a glance, from `/api/v1/stats`: one ruled band of figures. A cell that needs attention (failures,
 * records to review, work in progress) carries its tone as a top rule.
 */
import Link from "next/link";
import type { ReactNode } from "react";

import { QueryError } from "@/components/query-state";
import type { Tone } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCount, formatDuration, plural } from "@/lib/format";
import { useStats } from "@/lib/queries";
import type { Stats } from "@/lib/types";
import { cn } from "@/lib/utils";

const TONE_RULE: Record<Tone, string> = {
  neutral: "border-t-transparent",
  info: "border-t-tone-info-foreground",
  success: "border-t-tone-success-foreground",
  warning: "border-t-tone-warning-foreground",
  danger: "border-t-tone-danger-foreground",
};

interface Cell {
  key: string;
  label: string;
  value: string;
  detail: ReactNode;
  tone: Tone;
}

function cells(stats: Stats): Cell[] {
  const byStatus = stats.documents.by_status;
  const queued = byStatus.queued ?? 0;
  const processing = byStatus.processing ?? 0;
  const failed = byStatus.failed ?? 0;
  const toReview = stats.invoices.by_review_status.needs_review ?? 0;
  const errors = stats.issues.by_severity.error ?? 0;
  const warnings = stats.issues.by_severity.warning ?? 0;
  const inProgress = queued + processing;

  return [
    {
      key: "documents",
      label: "Documents",
      value: formatCount(stats.documents.total),
      detail: `${plural(stats.processing.pages_total, "page")} read`,
      tone: "neutral",
    },
    {
      key: "in-progress",
      label: "In progress",
      value: formatCount(inProgress),
      detail: inProgress > 0 ? `${formatCount(queued)} waiting, ${formatCount(processing)} processing` : "Nothing waiting",
      tone: inProgress > 0 ? "info" : "neutral",
    },
    {
      key: "review",
      label: "Records to review",
      value: formatCount(toReview),
      detail:
        toReview > 0 ? (
          <Link href="/review" className="text-primary underline-offset-2 hover:underline">
            Open the review queue
          </Link>
        ) : (
          `${plural(stats.invoices.total, "record")} extracted`
        ),
      tone: toReview > 0 ? "warning" : "neutral",
    },
    {
      key: "failed",
      label: "Failed",
      value: formatCount(failed),
      detail: failed > 0 ? "Reprocess them from the list below" : "No failed documents",
      tone: failed > 0 ? "danger" : "neutral",
    },
    {
      key: "issues",
      label: "Open checks",
      value: formatCount(errors + warnings),
      detail: `${plural(errors, "error")}, ${plural(warnings, "warning")}`,
      tone: "neutral",
    },
    {
      key: "speed",
      label: "Average processing",
      value: formatDuration(stats.processing.avg_ms),
      detail: stats.processing.p95_ms != null ? `95% within ${formatDuration(stats.processing.p95_ms)}` : "No timings yet",
      tone: "neutral",
    },
  ];
}

export function StatsStrip() {
  const { data, isPending, isError, error, refetch } = useStats();

  if (isError && !data) {
    return <QueryError error={error} title="Statistics are unavailable" onRetry={() => void refetch()} />;
  }

  return (
    <section aria-label="Summary" className="flex flex-col gap-2">
      <dl
        data-testid="stats-strip"
        className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border sm:grid-cols-3 xl:grid-cols-6"
      >
        {isPending
          ? Array.from({ length: 6 }, (_, i) => (
              <div key={i} className="flex flex-col gap-2 border-t-2 border-t-transparent bg-card p-4">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-7 w-16" />
                <Skeleton className="h-3 w-28" />
              </div>
            ))
          : cells(data).map((cell) => (
              <div
                key={cell.key}
                data-testid={`stat-${cell.key}`}
                className={cn("flex flex-col gap-1 border-t-2 bg-card p-4", TONE_RULE[cell.tone])}
              >
                <dt className="text-xs text-muted-foreground">{cell.label}</dt>
                <dd className="figures text-2xl leading-tight font-semibold">{cell.value}</dd>
                <dd className="text-xs text-muted-foreground">{cell.detail}</dd>
              </div>
            ))}
      </dl>
      {data && data.llm.fallback_records > 0 ? (
        <p data-testid="llm-fallback-warning" className="flex items-center gap-2 text-xs">
          <Badge variant="warning">LLM fallback</Badge>
          <span>
            The language model fell back to rules on {plural(data.llm.fallback_records, "record")} (
            {plural(data.llm.fallback_documents, "document")}).
          </span>
        </p>
      ) : null}
      {data?.llm.provider ? (
        <p className="text-xs text-muted-foreground">
          Extraction model: {data.llm.provider}
          {data.llm.model ? <span className="figures"> {data.llm.model}</span> : null}
        </p>
      ) : null}
    </section>
  );
}
