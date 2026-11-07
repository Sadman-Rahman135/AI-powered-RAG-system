/**
 * Upload helpers: client-side pre-checks that mirror the server's limits, batching, and a multipart upload over
 * XMLHttpRequest (fetch has no upload progress events).
 */
import { API_BASE, ApiError, apiErrorFrom, networkError } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

/** Server limits (`MAX_FILES_PER_REQUEST`, `MAX_UPLOAD_MB`); `/api/v1/config` overrides them when available. */
export const DEFAULT_MAX_FILES_PER_REQUEST = 50;
export const DEFAULT_MAX_UPLOAD_MB = 200;
export const DEFAULT_ALLOWED_EXTENSIONS: readonly string[] = [
  ".pdf",
  ".jpg",
  ".jpeg",
  ".png",
  ".tif",
  ".tiff",
  ".xlsx",
  ".xls",
  ".csv",
];

export interface UploadLimits {
  maxUploadMb: number;
  allowedExtensions: readonly string[];
}

function normalizeExtension(extension: string): string {
  const trimmed = extension.trim().toLowerCase();
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

const EXTENSION_ALIASES: Record<string, string> = { JPEG: "JPG", TIF: "TIFF" };

/** `[".pdf", ".jpg", ".jpeg"]` -> `"PDF or JPG"`, for help text and error messages (aliases shown once). */
export function describeExtensions(extensions: readonly string[]): string {
  const names = [
    ...new Set(
      extensions.map((e) => {
        const name = normalizeExtension(e).slice(1).toUpperCase();
        return EXTENSION_ALIASES[name] ?? name;
      }),
    ),
  ];
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}`;
}

/** Why the server would refuse this file, or `null` when it can be sent. The server still has the final say. */
export function precheckFile(file: File, limits: UploadLimits): string | null {
  const allowed = limits.allowedExtensions.map(normalizeExtension);
  const extension = fileExtension(file.name);
  if (!extension || !allowed.includes(extension)) {
    return `Not a supported file type. Upload ${describeExtensions(limits.allowedExtensions)}.`;
  }
  if (file.size === 0) return "The file is empty.";
  if (file.size > limits.maxUploadMb * 1024 * 1024) {
    return `Larger than ${limits.maxUploadMb} MB, the upload limit.`;
  }
  return null;
}

export function chunk<T>(items: readonly T[], size: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += size) batches.push(items.slice(i, i + size));
  return batches;
}

/** Multipart framing (boundaries and part headers) added per file, and a fixed allowance, when sizing a request. */
const MULTIPART_OVERHEAD_PER_FILE = 2 * 1024;
const MULTIPART_OVERHEAD_FIXED = 64 * 1024;

/**
 * Split items into sequential requests of at most `maxFiles` files whose total size, plus multipart overhead, stays
 * within `maxRequestBytes` (the server's request cap). A file that alone exceeds the cap still gets a request of its
 * own, so the server can answer with its per-file rejection. Order is preserved. Without a byte cap this is `chunk`.
 */
export function batchBySize<T>(
  items: readonly T[],
  sizeOf: (item: T) => number,
  maxFiles: number,
  maxRequestBytes?: number,
): T[][] {
  const budget = maxRequestBytes === undefined ? Infinity : maxRequestBytes - MULTIPART_OVERHEAD_FIXED;
  const batches: T[][] = [];
  let current: T[] = [];
  let used = 0;
  for (const item of items) {
    const cost = sizeOf(item) + MULTIPART_OVERHEAD_PER_FILE;
    if (current.length > 0 && (current.length >= maxFiles || used + cost > budget)) {
      batches.push(current);
      current = [];
      used = 0;
    }
    current.push(item);
    used += cost;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

/**
 * Estimate each file's progress from the progress of the whole multipart request. Files are serialized in order,
 * so the bytes sent so far (scaled from `loaded / total`, which also counts the multipart overhead) fill the files
 * one after another.
 */
export function perFileProgress(sizes: readonly number[], loaded: number, total: number): number[] {
  const fraction = total > 0 ? Math.min(Math.max(loaded / total, 0), 1) : 0;
  const totalBytes = sizes.reduce((sum, size) => sum + size, 0);
  const sent = fraction * totalBytes;
  let offset = 0;
  return sizes.map((size) => {
    const start = offset;
    offset += size;
    if (fraction >= 1) return 1;
    if (size === 0) return sent > start ? 1 : 0;
    return Math.min(Math.max((sent - start) / size, 0), 1);
  });
}

export interface UploadRequestOptions {
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

/** POST the files in one multipart request (field `files`) and resolve with the per-file results. */
export function uploadFiles(files: readonly File[], options: UploadRequestOptions = {}): Promise<UploadResponse> {
  return new Promise<UploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/documents`);
    xhr.setRequestHeader("accept", "application/json");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) options.onProgress?.(event.loaded, event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch {
          reject(new ApiError(xhr.status, "invalid_response", "The server sent a response the app could not read."));
        }
        return;
      }
      reject(apiErrorFrom(xhr.status, xhr.statusText, xhr.responseText));
    };
    xhr.onerror = () => reject(networkError());
    xhr.ontimeout = () => reject(new ApiError(0, "timeout", "The upload took too long and was stopped."));
    xhr.onabort = () => reject(new DOMException("The upload was cancelled.", "AbortError"));
    options.signal?.addEventListener("abort", () => xhr.abort(), { once: true });

    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    xhr.send(form);
  });
}
