import { Badge } from "@/components/ui/badge";
import { confidenceBucket, formatConfidence, type ConfidenceBucket } from "@/lib/format";

const BUCKET_TONE: Record<Exclude<ConfidenceBucket, "unknown">, "success" | "neutral" | "warning"> = {
  high: "success",
  medium: "neutral",
  low: "warning",
};

const BUCKET_WORD: Record<Exclude<ConfidenceBucket, "unknown">, string> = {
  high: "high",
  medium: "medium",
  low: "low",
};

/** Extraction confidence as a percentage, toned by bucket (low = below the 0.6 review threshold). */
export function ConfidenceBadge({ value, className }: { value: number | null | undefined; className?: string }) {
  const bucket = confidenceBucket(value);
  if (bucket === "unknown") return null;
  return (
    <Badge
      variant={BUCKET_TONE[bucket]}
      data-testid="confidence-badge"
      data-bucket={bucket}
      className={className}
      title={`Extraction confidence: ${BUCKET_WORD[bucket]}`}
    >
      <span className="sr-only">Confidence </span>
      <span className="figures">{formatConfidence(value)}</span>
      <span className="sr-only">, {BUCKET_WORD[bucket]}</span>
    </Badge>
  );
}
