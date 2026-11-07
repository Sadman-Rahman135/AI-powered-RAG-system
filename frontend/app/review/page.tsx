import type { Metadata } from "next";

import { ReviewQueueView } from "@/components/review-queue-view";

export const metadata: Metadata = { title: "Review queue" };

export default function ReviewPage() {
  return <ReviewQueueView />;
}
