import { QueryObserver } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import { queryKeys, refreshAfterDocumentChange } from "@/lib/queries";

import { createTestQueryClient } from "./test-utils";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("refreshAfterDocumentChange", () => {
  it("refetches a document list whose first load was already in flight before the change", async () => {
    // The dashboard's first list request can still be running when an upload finishes. That request was answered
    // before the new document existed, so reusing it would leave the new document out of the table.
    const queryClient = createTestQueryClient();
    const firstLoad = deferred<string[]>();
    const answers = [firstLoad.promise, Promise.resolve(["new upload", "older"])];
    let calls = 0;
    const observer = new QueryObserver(queryClient, {
      queryKey: queryKeys.documentList({ limit: 20, offset: 0, sort: "-created_at" }),
      queryFn: () => answers[calls++],
    });
    const unsubscribe = observer.subscribe(() => {});
    expect(calls).toBe(1);

    const refreshed = refreshAfterDocumentChange(queryClient);
    firstLoad.resolve(["older"]);
    await refreshed;
    await new Promise((done) => setTimeout(done, 0));

    expect(calls).toBe(2);
    expect(observer.getCurrentResult().data).toEqual(["new upload", "older"]);
    unsubscribe();
  });
});
