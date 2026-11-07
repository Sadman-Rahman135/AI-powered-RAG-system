/**
 * Status chips. Every status pairs a tone with an icon and a word, so it never relies on colour alone.
 */
import {
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleMinus,
  CircleX,
  Clock,
  Info,
  LoaderCircle,
  OctagonX,
  TriangleAlert,
  UserCheck,
  type LucideIcon,
} from "lucide-react";
import type { ComponentProps } from "react";

import { Badge } from "@/components/ui/badge";
import {
  DOCUMENT_STATUS_LABELS,
  REVIEW_STATUS_LABELS,
  SEVERITY_LABELS,
  humanize,
} from "@/lib/format";
import { cn } from "@/lib/utils";

export type Tone = "neutral" | "info" | "success" | "warning" | "danger";

interface Appearance {
  label: string;
  tone: Tone;
  icon: LucideIcon;
  spin?: boolean;
}

interface ToneBadgeProps extends Omit<ComponentProps<typeof Badge>, "variant" | "children"> {
  appearance: Appearance;
}

/** A badge in one of the five status tones, with its icon. */
export function ToneBadge({ appearance, className, ...props }: ToneBadgeProps) {
  const Icon = appearance.icon;
  return (
    <Badge variant={appearance.tone} data-tone={appearance.tone} className={className} {...props}>
      <Icon aria-hidden="true" data-icon="inline-start" className={cn(appearance.spin && "motion-safe:animate-spin")} />
      {appearance.label}
    </Badge>
  );
}

function fallback(value: string): Appearance {
  return { label: humanize(value), tone: "neutral", icon: CircleDashed };
}

const DOCUMENT_STATUS: Record<string, Appearance> = {
  queued: { label: DOCUMENT_STATUS_LABELS.queued, tone: "neutral", icon: Clock },
  processing: { label: DOCUMENT_STATUS_LABELS.processing, tone: "info", icon: LoaderCircle, spin: true },
  completed: { label: DOCUMENT_STATUS_LABELS.completed, tone: "success", icon: CircleCheck },
  needs_review: { label: DOCUMENT_STATUS_LABELS.needs_review, tone: "warning", icon: CircleAlert },
  failed: { label: DOCUMENT_STATUS_LABELS.failed, tone: "danger", icon: CircleX },
};

/** Document processing status: queued, processing, completed, needs review, failed. */
export function StatusBadge({ status, className }: { status: string; className?: string }) {
  return (
    <ToneBadge
      appearance={DOCUMENT_STATUS[status] ?? fallback(status)}
      data-testid="status-badge"
      data-status={status}
      className={className}
    />
  );
}

const REVIEW_STATUS: Record<string, Appearance> = {
  auto_approved: { label: REVIEW_STATUS_LABELS.auto_approved, tone: "success", icon: CircleCheck },
  needs_review: { label: REVIEW_STATUS_LABELS.needs_review, tone: "warning", icon: CircleAlert },
  reviewed: { label: REVIEW_STATUS_LABELS.reviewed, tone: "success", icon: UserCheck },
};

/** Review state of an extracted record. */
export function ReviewStatusBadge({ status, className }: { status: string; className?: string }) {
  return (
    <ToneBadge
      appearance={REVIEW_STATUS[status] ?? fallback(status)}
      data-testid="review-status-badge"
      data-status={status}
      className={className}
    />
  );
}

const SEVERITY: Record<string, Appearance> = {
  error: { label: SEVERITY_LABELS.error.one, tone: "danger", icon: OctagonX },
  warning: { label: SEVERITY_LABELS.warning.one, tone: "warning", icon: TriangleAlert },
  info: { label: SEVERITY_LABELS.info.one, tone: "info", icon: Info },
};

export const SEVERITY_ICONS: Record<string, LucideIcon> = {
  error: OctagonX,
  warning: TriangleAlert,
  info: Info,
};

export function SeverityBadge({ severity, className }: { severity: string; className?: string }) {
  return (
    <ToneBadge
      appearance={SEVERITY[severity] ?? fallback(severity)}
      data-testid="severity-badge"
      data-severity={severity}
      className={className}
    />
  );
}

const MATCH_STATUS: Record<string, Appearance> = {
  match: { label: "Match", tone: "success", icon: CircleCheck },
  mismatch: { label: "Mismatch", tone: "danger", icon: CircleX },
  partial: { label: "Partial match", tone: "warning", icon: CircleAlert },
  missing: { label: "Missing", tone: "neutral", icon: CircleMinus },
};

/** Cross-verification outcome (whole comparison, a field, or the line items). */
export function MatchBadge({ status, className }: { status: string; className?: string }) {
  return (
    <ToneBadge
      appearance={MATCH_STATUS[status] ?? fallback(status)}
      data-testid="match-badge"
      data-status={status}
      className={className}
    />
  );
}
