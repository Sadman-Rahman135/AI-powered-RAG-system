import { Fragment } from "react";

import { splitHighlights } from "@/lib/highlight";

/** A search snippet with its « » matches rendered as <mark>. Text only — never parsed as HTML. */
export function Snippet({ text, className }: { text: string; className?: string }) {
  return (
    <span className={className}>
      {splitHighlights(text).map((segment, index) =>
        segment.highlighted ? <mark key={index}>{segment.text}</mark> : <Fragment key={index}>{segment.text}</Fragment>,
      )}
    </span>
  );
}
