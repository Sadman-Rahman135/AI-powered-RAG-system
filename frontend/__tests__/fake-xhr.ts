/** A minimal XMLHttpRequest double: records requests and lets a test drive progress and responses. */
type Listener = (event: ProgressEvent) => void;

export class FakeXHR {
  static instances: FakeXHR[] = [];

  static reset() {
    FakeXHR.instances = [];
  }

  method = "";
  url = "";
  body: Document | XMLHttpRequestBodyInit | null = null;
  requestHeaders: Record<string, string> = {};
  status = 0;
  statusText = "";
  responseText = "";
  readyState = 0;
  aborted = false;
  responseHeaders: Record<string, string> = {};

  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  ontimeout: (() => void) | null = null;

  upload: { onprogress: Listener | null } = { onprogress: null };

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
    this.readyState = 1;
  }

  setRequestHeader(name: string, value: string) {
    this.requestHeaders[name.toLowerCase()] = value;
  }

  getResponseHeader(name: string): string | null {
    return this.responseHeaders[name.toLowerCase()] ?? null;
  }

  send(body: Document | XMLHttpRequestBodyInit | null = null) {
    this.body = body;
    FakeXHR.instances.push(this);
  }

  abort() {
    this.aborted = true;
    this.onabort?.();
  }

  /** Names of the files sent under the multipart field `files`. */
  sentFileNames(): string[] {
    if (!(this.body instanceof FormData)) return [];
    return this.body.getAll("files").map((entry) => (entry instanceof File ? entry.name : String(entry)));
  }

  progress(loaded: number, total: number) {
    this.upload.onprogress?.({ lengthComputable: true, loaded, total } as ProgressEvent);
  }

  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = typeof body === "string" ? body : JSON.stringify(body);
    this.responseHeaders["content-type"] = "application/json";
    this.readyState = 4;
    this.onload?.();
  }

  failNetwork() {
    this.readyState = 4;
    this.status = 0;
    this.onerror?.();
  }
}
