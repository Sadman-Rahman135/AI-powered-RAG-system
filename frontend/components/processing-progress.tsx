/**
 * Where a document is in the pipeline: stage, page progress and attempt number. The compact variant fits a table
 * cell; the full variant lists every stage in order.
 */
import { Check } from "lucide-react";

import { Progress } from "@/components/ui/progress";
import { STAGE_LABELS, STAGE_ORDER, formatCount, labelFor } from "@/lib/format";
import type { DocumentStatus, Job } from "@/lib/types";
import { cn } from "@/lib/utils";

function pageText(job: Job | null): string | null {
  if (!job || job.progress_total <= 0) return null;
  return `${formatCount(job.progress_current)} of ${formatCount(job.progress_total)} pages`;
}

function percent(job: Job | null): number | null {
  if (!job || job.progress_total <= 0) return null;
  return Math.min(100, Math.round((job.progress_current / job.progress_total) * 100));
}

export function stageText(status: DocumentStatus, job: Job | null): string {
  if (status === "queued" || job?.status === "queued") return "Waiting for a worker";
  if (job?.stage) return labelFor(STAGE_LABELS, job.stage);
  return "Starting";
}

/** One line for tables: stage name, a thin bar and the page count. */
export function CompactProgress({ status, job }: { status: DocumentStatus; job: Job | null }) {
  const value = percent(job);
  const pages = pageText(job);
  if (status === "queued" || job?.status === "queued") {
    return <span className="text-xs text-muted-foreground">Waiting for a worker</span>;
  }
  return (
    <div className="flex w-44 flex-col gap-1">
      <span className="truncate text-xs text-muted-foreground">{stageText(status, job)}</span>
      <Progress
        value={value}
        aria-label={`${stageText(status, job)}${pages ? `, ${pages}` : ""}`}
        className="gap-1"
      />
      {pages ? <span className="figures text-xs text-muted-foreground">{pages}</span> : null}
    </div>
  );
}

/** The ordered pipeline with the current stage highlighted, for the document page. */
export function ProcessingProgress({ status, job }: { status: DocumentStatus; job: Job | null }) {
  const currentIndex = job?.stage ? STAGE_ORDER.indexOf(job.stage) : -1;
  const value = percent(job);
  const pages = pageText(job);
  const queued = status === "queued" || job?.status === "queued";

  return (
    <section aria-label="Processing progress" className="flex flex-col gap-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-semibold">{queued ? "Waiting for a worker" : `${stageText(status, job)}…`}</h2>
        <p className="text-sm text-muted-foreground">
          {pages ? <span className="figures">{pages}</span> : null}
          {job && job.attempt > 1 ? <span> Attempt {job.attempt} of {job.max_attempts}</span> : null}
        </p>
      </div>
      <Progress value={queued ? null : value} aria-label={queued ? "Queued" : stageText(status, job)} />
      <ol className="grid gap-2 text-xs sm:grid-cols-5">
        {STAGE_ORDER.map((stage, index) => {
          const done = currentIndex > index;
          const current = currentIndex === index;
          return (
            <li
              key={stage}
              aria-current={current ? "step" : undefined}
              className={cn(
                "flex items-center gap-1.5 text-muted-foreground",
                current && "font-medium text-foreground",
                done && "text-tone-success-foreground",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "flex size-5 shrink-0 items-center justify-center rounded-full border text-[11px]",
                  current && "border-primary text-primary",
                  done && "border-tone-success-border bg-tone-success",
                )}
              >
                {done ? <Check className="size-3" /> : index + 1}
              </span>
              {STAGE_LABELS[stage]}
              {done ? <span className="sr-only"> (done)</span> : null}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
