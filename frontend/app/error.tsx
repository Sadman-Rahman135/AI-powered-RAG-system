"use client";

import { TriangleAlert } from "lucide-react";

import { EmptyState } from "@/components/query-state";
import { Button } from "@/components/ui/button";

export default function RouteError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <EmptyState
      icon={TriangleAlert}
      title="This page stopped working"
      description="An unexpected error interrupted it. Try again; if it keeps happening, reload the page."
    >
      <Button type="button" variant="outline" onClick={reset}>
        Try again
      </Button>
    </EmptyState>
  );
}
