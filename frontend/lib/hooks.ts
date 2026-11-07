"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** `value`, updated only after it has stopped changing for `delayMs` (for search-as-you-type requests). */
export function useDebouncedValue<T>(value: T, delayMs = 300): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

/**
 * A polite screen-reader announcer: render `<div ref={regionRef} aria-live="polite" />` (visually hidden) and call
 * `announce(message)`. Writes go straight to the DOM node, so announcing never re-renders the component.
 */
export function useAnnouncer() {
  const regionRef = useRef<HTMLDivElement>(null);
  const announce = useCallback((message: string) => {
    const region = regionRef.current;
    if (!region) return;
    // Clear first so repeating the same message is announced again.
    region.textContent = "";
    window.setTimeout(() => {
      region.textContent = message;
    }, 50);
  }, []);
  return { regionRef, announce };
}
