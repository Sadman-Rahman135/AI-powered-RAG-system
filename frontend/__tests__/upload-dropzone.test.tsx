import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { UploadDropzone } from "@/components/upload-dropzone";
import type { UploadItem } from "@/lib/types";

import { FakeXHR } from "./fake-xhr";
import { makeUploadItem } from "./fixtures";
import { renderWithProviders } from "./test-utils";

function pdf(name: string, bytes = 1024): File {
  return new File([new Uint8Array(bytes)], name, { type: "application/pdf" });
}

function queuedFor(names: string[]): { items: UploadItem[] } {
  return {
    items: names.map((filename, i) => makeUploadItem({ filename, document_id: `doc-${filename}-${i}` })),
  };
}

function rows() {
  return screen.getAllByTestId("upload-row");
}

beforeEach(() => {
  FakeXHR.reset();
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("UploadDropzone", () => {
  it("posts the selected files as multipart field `files` to /api/v1/documents", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [pdf("a.pdf"), pdf("b.pdf")]);

    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));
    const xhr = FakeXHR.instances[0];
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe("/api/v1/documents");
    expect(xhr.sentFileNames()).toEqual(["a.pdf", "b.pdf"]);
  });

  it("never puts more than 50 files in one request: 51 files go out as 50 + 1", async () => {
    const user = userEvent.setup();
    const onUploaded = vi.fn();
    renderWithProviders(<UploadDropzone onUploaded={onUploaded} />);
    const files = Array.from({ length: 51 }, (_, i) => pdf(`invoice-${String(i + 1).padStart(2, "0")}.pdf`));

    await user.upload(screen.getByTestId("upload-input"), files);

    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));
    const first = FakeXHR.instances[0];
    expect(first.sentFileNames()).toHaveLength(50);
    expect(screen.getByTestId("upload-status")).toHaveTextContent("2 batches");

    await act(async () => first.respond(202, queuedFor(first.sentFileNames())));

    await waitFor(() => expect(FakeXHR.instances).toHaveLength(2));
    const second = FakeXHR.instances[1];
    expect(second.sentFileNames()).toEqual(["invoice-51.pdf"]);

    await act(async () => second.respond(202, queuedFor(second.sentFileNames())));

    await waitFor(() => expect(rows().filter((r) => r.dataset.state === "queued")).toHaveLength(51));
    expect(onUploaded).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("upload-status")).toHaveTextContent("51 queued");
  });

  it("shows a result for every file: queued, rejected with its reason, and duplicates", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [pdf("new.pdf"), pdf("scan.pdf"), pdf("again.pdf")]);
    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));

    await act(async () =>
      FakeXHR.instances[0].respond(202, {
        items: [
          makeUploadItem({ filename: "new.pdf", document_id: "doc-new" }),
          makeUploadItem({
            filename: "scan.pdf",
            status: "rejected",
            document_id: null,
            job_id: null,
            error: { code: "unsupported_file_type", message: "The file content is not a supported document type." },
          }),
          makeUploadItem({ filename: "again.pdf", document_id: "doc-again", duplicate_of_id: "doc-original" }),
        ],
      }),
    );

    await waitFor(() => expect(rows()[0]).toHaveAttribute("data-state", "queued"));
    const [queued, rejected, duplicate] = rows();

    expect(within(queued).getByRole("link", { name: /open/i })).toHaveAttribute("href", "/documents/doc-new");

    expect(rejected).toHaveAttribute("data-state", "rejected");
    expect(rejected).toHaveTextContent("The file content is not a supported document type.");
    expect(within(rejected).queryByRole("link")).toBeNull();

    expect(duplicate).toHaveTextContent("Same file as an earlier upload");
    expect(within(duplicate).getByRole("link", { name: /original/i })).toHaveAttribute("href", "/documents/doc-original");

    expect(screen.getByTestId("upload-status")).toHaveTextContent("2 queued, 1 rejected");
  });

  it("rejects files the server would refuse before sending anything", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderWithProviders(<UploadDropzone maxUploadMb={200} />);
    const huge = pdf("huge.pdf");
    Object.defineProperty(huge, "size", { value: 201 * 1024 * 1024 });

    await user.upload(screen.getByTestId("upload-input"), [
      pdf("ok.pdf"),
      new File(["MZ"], "setup.exe", { type: "application/octet-stream" }),
      huge,
      pdf("empty.pdf", 0),
    ]);

    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));
    expect(FakeXHR.instances[0].sentFileNames()).toEqual(["ok.pdf"]);

    const byName = (name: string) => rows().find((r) => r.textContent?.includes(name)) as HTMLElement;
    expect(byName("setup.exe")).toHaveAttribute("data-state", "rejected");
    expect(byName("setup.exe")).toHaveTextContent(/not a supported file type/i);
    expect(byName("huge.pdf")).toHaveTextContent(/larger than 200 MB/i);
    expect(byName("empty.pdf")).toHaveTextContent(/empty/i);
  });

  it("does not send anything when every file is refused", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [new File(["x"], "notes.docx")]);

    expect(await screen.findByText(/not a supported file type/i)).toBeInTheDocument();
    expect(FakeXHR.instances).toHaveLength(0);
  });

  it("marks the files of a failed request as failed with the server's message", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [pdf("a.pdf"), pdf("b.pdf")]);
    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));

    await act(async () =>
      FakeXHR.instances[0].respond(429, {
        error: { code: "rate_limited", message: "Too many uploads. Try again in a minute." },
      }),
    );

    await waitFor(() => expect(rows().every((r) => r.dataset.state === "failed")).toBe(true));
    expect(rows()[0]).toHaveTextContent("Too many uploads. Try again in a minute.");
    expect(screen.getByTestId("upload-status")).toHaveTextContent("2 failed");
  });

  it("reports a network failure instead of dropping the files silently", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [pdf("a.pdf")]);
    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));

    await act(async () => FakeXHR.instances[0].failNetwork());

    await waitFor(() => expect(rows()[0]).toHaveAttribute("data-state", "failed"));
    expect(rows()[0]).toHaveTextContent(/could not reach the server/i);
  });

  it("shows upload progress while the request is in flight", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UploadDropzone />);

    await user.upload(screen.getByTestId("upload-input"), [pdf("a.pdf", 1000), pdf("b.pdf", 1000)]);
    await waitFor(() => expect(FakeXHR.instances).toHaveLength(1));

    await act(async () => FakeXHR.instances[0].progress(1100, 2200));

    // The percentage is visual only; the live region announces milestones, not every tick.
    expect(screen.getByTestId("upload-progress")).toHaveTextContent("50%");
    expect(screen.getByTestId("upload-status")).not.toHaveTextContent("50%");
    const [a, b] = rows();
    expect(a).toHaveAttribute("data-state", "uploading");
    expect(within(a).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    expect(within(b).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  });
});
