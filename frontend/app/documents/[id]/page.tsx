import type { Metadata } from "next";
import { Suspense } from "react";

import { DocumentDetailRoute, DocumentDetailSkeleton } from "@/components/document-detail-view";

export const metadata: Metadata = { title: "Document" };

export default function DocumentPage({ params }: { params: Promise<{ id: string }> }) {
  // The id is only known at request time; the skeleton is the prerendered shell.
  return (
    <Suspense fallback={<DocumentDetailSkeleton />}>
      {params.then(({ id }) => (
        <DocumentDetailRoute id={id} />
      ))}
    </Suspense>
  );
}
