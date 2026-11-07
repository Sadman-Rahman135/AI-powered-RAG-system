/** Shared error and empty states for data views. */
import { CircleAlert, RotateCw, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { errorMessage, isApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

function explain(error: unknown): string {
  if (isApiError(error)) {
    if (error.status === 401) return "The server refused the request: the API key is missing or wrong.";
    if (error.status === 429) return "Too many requests at once. Wait a moment, then retry.";
  }
  return errorMessage(error);
}

interface QueryErrorProps {
  error: unknown;
  title?: string;
  onRetry?: () => void;
  className?: string;
}

export function QueryError({ error, title = "This could not be loaded", onRetry, className }: QueryErrorProps) {
  return (
    <Alert variant="destructive" className={className}>
      <CircleAlert aria-hidden="true" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{explain(error)}</AlertDescription>
      {onRetry ? (
        <AlertAction>
          <Button type="button" variant="outline" size="sm" onClick={onRetry}>
            <RotateCw aria-hidden="true" data-icon="inline-start" />
            Retry
          </Button>
        </AlertAction>
      ) : null}
    </Alert>
  );
}

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: ReactNode;
  children?: ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, children, className }: EmptyStateProps) {
  return (
    <Empty className={cn("border bg-card", className)}>
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <Icon aria-hidden="true" />
        </EmptyMedia>
        <EmptyTitle>{title}</EmptyTitle>
        {description ? <EmptyDescription>{description}</EmptyDescription> : null}
      </EmptyHeader>
      {children ? <EmptyContent>{children}</EmptyContent> : null}
    </Empty>
  );
}
