/** Processing-state helpers shared by the polling queries. */
import type { Document, Job } from "@/lib/types";

/** How often views refresh while a document or job is still queued or processing. */
export const POLL_INTERVAL_MS = 2000;

export function isActiveJob(job: Pick<Job, "status"> | null | undefined): boolean {
  return job?.status === "queued" || job?.status === "running";
}

/** True while the document (or its latest job) has not reached a final state. */
export function isActiveDocument(document: Pick<Document, "status" | "job">): boolean {
  return document.status === "queued" || document.status === "processing" || isActiveJob(document.job);
}

/**
 * Build a TanStack Query `refetchInterval` callback: poll every {@link POLL_INTERVAL_MS} while any document in the
 * query's data is still in flight, stop once everything has settled.
 */
export function refetchWhileActive<T>(
  select: (data: T | undefined) => ReadonlyArray<Pick<Document, "status" | "job">>,
): (data: T | undefined) => number | false {
  return (data) => (select(data).some(isActiveDocument) ? POLL_INTERVAL_MS : false);
}
