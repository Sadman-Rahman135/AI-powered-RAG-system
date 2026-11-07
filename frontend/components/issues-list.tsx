/**
 * Validation issues grouped by severity (errors, then warnings, then info), each with a link to its field.
 */
import { CircleCheck } from "lucide-react";
import { useId } from "react";

import { SEVERITY_ICONS } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { SEVERITY_LABELS, fieldLabel, humanizeCode } from "@/lib/format";
import type { Issue, Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

const SEVERITY_ORDER: Severity[] = ["error", "warning", "info"];

const SEVERITY_TONE = { error: "danger", warning: "warning", info: "info" } as const;

const ICON_COLOR: Record<Severity, string> = {
  error: "text-tone-danger-foreground",
  warning: "text-tone-warning-foreground",
  info: "text-tone-info-foreground",
};

/** Group issues by severity, keeping their order; unknown severities are shown as info rather than dropped. */
export function groupBySeverity(issues: readonly Issue[]): Record<Severity, Issue[]> {
  const groups: Record<Severity, Issue[]> = { error: [], warning: [], info: [] };
  for (const issue of issues) {
    (groups[issue.severity] ?? groups.info).push(issue);
  }
  return groups;
}

interface IssuesListProps {
  issues: readonly Issue[];
  /** Called when the reviewer follows an issue to its field. */
  onSelectField?: (issue: Issue) => void;
  /** Names the record an issue belongs to (useful when a document holds several invoices). */
  recordLabel?: (invoiceId: string | null) => string | null;
  className?: string;
}

export function IssuesList({ issues, onSelectField, recordLabel, className }: IssuesListProps) {
  const baseId = useId();

  if (issues.length === 0) {
    return (
      <Empty className={cn("border", className)}>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <CircleCheck aria-hidden="true" />
          </EmptyMedia>
          <EmptyTitle>No validation issues</EmptyTitle>
          <EmptyDescription>Every check passed: required fields are present and the amounts add up.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  const groups = groupBySeverity(issues);

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {SEVERITY_ORDER.filter((severity) => groups[severity].length > 0).map((severity) => {
        const headingId = `${baseId}-${severity}`;
        const Icon = SEVERITY_ICONS[severity];
        return (
          <section key={severity} data-testid="issue-group" data-severity={severity} aria-labelledby={headingId}>
            <h3 id={headingId} className="mb-2 flex items-center gap-2 text-sm font-semibold">
              {SEVERITY_LABELS[severity].many}
              <Badge variant={SEVERITY_TONE[severity]}>{groups[severity].length}</Badge>
            </h3>
            <ul className="divide-y rounded-lg border bg-card">
              {groups[severity].map((issue) => (
                <li
                  key={issue.id}
                  data-testid="issue-item"
                  data-code={issue.code}
                  className="flex items-start gap-3 px-3 py-2.5"
                >
                  <Icon aria-hidden="true" className={cn("mt-0.5 size-4 shrink-0", ICON_COLOR[severity])} />
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <p className="font-medium">{humanizeCode(issue.code)}</p>
                    <p className="text-muted-foreground">{issue.message}</p>
                    <IssueMeta issue={issue} onSelectField={onSelectField} recordLabel={recordLabel} />
                  </div>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

function IssueMeta({
  issue,
  onSelectField,
  recordLabel,
}: Pick<IssuesListProps, "onSelectField" | "recordLabel"> & { issue: Issue }) {
  const record = recordLabel?.(issue.invoice_id) ?? null;
  if (!record && !issue.field) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
      {record ? <span>{record}</span> : null}
      {issue.field ? (
        onSelectField ? (
          <Button
            type="button"
            variant="link"
            size="xs"
            className="h-auto px-0"
            onClick={() => onSelectField(issue)}
          >
            Go to {fieldLabel(issue.field)}
          </Button>
        ) : (
          <span>Field: {fieldLabel(issue.field)}</span>
        )
      ) : null}
    </div>
  );
}
