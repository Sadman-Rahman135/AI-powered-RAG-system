import type { Metadata } from "next";
import { Suspense } from "react";

import { CompareView } from "@/components/compare-view";
import { Skeleton } from "@/components/ui/skeleton";

export const metadata: Metadata = { title: "Compare" };

function CompareSkeleton() {
  return (
    <div className="flex flex-col gap-6" aria-busy="true" aria-label="Loading comparison">
      <Skeleton className="h-7 w-32" />
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    </div>
  );
}

export default function ComparePage() {
  // The chosen records come from the URL, which is read on the client.
  return (
    <Suspense fallback={<CompareSkeleton />}>
      <CompareView />
    </Suspense>
  );
}
