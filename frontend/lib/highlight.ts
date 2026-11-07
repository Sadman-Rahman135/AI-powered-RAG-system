/**
 * Search snippets mark matches with « and » (PostgreSQL `ts_headline` markers). The snippet is document text, so it
 * is split into plain-text segments and rendered as React text — never as HTML.
 */
export interface HighlightSegment {
  text: string;
  highlighted: boolean;
}

const OPEN = "«";
const CLOSE = "»";

export function splitHighlights(snippet: string): HighlightSegment[] {
  const segments: HighlightSegment[] = [];
  let buffer = "";
  let inMark = false;

  const flush = (highlighted: boolean) => {
    if (buffer) segments.push({ text: buffer, highlighted });
    buffer = "";
  };

  for (const char of snippet) {
    if (char === OPEN) {
      // A nested opening marker is ignored; the current highlight continues.
      if (!inMark) {
        flush(false);
        inMark = true;
      }
      continue;
    }
    if (char === CLOSE) {
      // A closing marker without an opening one is dropped.
      if (inMark) {
        flush(true);
        inMark = false;
      }
      continue;
    }
    buffer += char;
  }
  // An unterminated highlight is shown as plain text rather than highlighting the rest of the snippet.
  flush(false);
  return segments;
}
