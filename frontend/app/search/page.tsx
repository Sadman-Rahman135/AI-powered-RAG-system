import type { Metadata } from "next";
import { Suspense } from "react";

import { SearchView } from "@/components/search-view";
import { Skeleton } from "@/components/ui/skeleton";

export const metadata: Metadata = { title: "Search" };

function SearchSkeleton() {
  return (
    <div className="flex flex-col gap-6" aria-busy="true" aria-label="Loading search">
      <Skeleton className="h-7 w-32" />
      <Skeleton className="h-56 w-full" />
    </div>
  );
}

export default function SearchPage() {
  // The search state comes from the URL, which is read on the client.
  return (
    <Suspense fallback={<SearchSkeleton />}>
      <SearchView />
    </Suspense>
  );
}
