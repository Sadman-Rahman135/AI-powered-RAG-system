/**
 * Map a 422 `validation_error` envelope onto form fields.
 *
 * The contract fixes the envelope (`{"error": {"code", "message", "details"}}`) but not the shape of `details`, so
 * the common shapes are all understood: pydantic-style lists (`[{loc, msg}]`), `{fields: {name: msg}}`,
 * `{errors: [{field, message}]}`, `{field}` and a flat `{name: msg}` map. Line-item errors are keyed
 * `line_items.<index>.<field>`.
 */
import type { ApiError } from "@/lib/api";

export type FieldErrors = Record<string, string>;

const RESERVED_KEYS = new Set(["fields", "errors", "field", "message", "msg", "detail", "details", "code", "type"]);
const LOCATION_PREFIXES = new Set(["body", "query", "path"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asMessage(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (Array.isArray(value)) {
    const parts = value.filter((v): v is string => typeof v === "string" && v.trim() !== "");
    return parts.length ? parts.join(" ") : null;
  }
  return null;
}

function add(errors: FieldErrors, field: string, message: string) {
  errors[field] = errors[field] ? `${errors[field]} ${message}` : message;
}

function fieldFromLocation(location: unknown[]): string | null {
  const parts = location.filter((part) => !(typeof part === "string" && LOCATION_PREFIXES.has(part)));
  if (parts[0] === "fields" && typeof parts[1] === "string") return parts[1];
  if (parts[0] === "line_items") {
    return typeof parts[1] === "number" && typeof parts[2] === "string"
      ? `line_items.${parts[1]}.${parts[2]}`
      : "line_items";
  }
  const last = [...parts].reverse().find((part): part is string => typeof part === "string");
  return last ?? null;
}

function fromList(list: unknown[]): FieldErrors {
  const errors: FieldErrors = {};
  for (const entry of list) {
    if (!isRecord(entry)) continue;
    const message = asMessage(entry.msg) ?? asMessage(entry.message);
    if (!message) continue;
    const field = Array.isArray(entry.loc)
      ? fieldFromLocation(entry.loc)
      : typeof entry.field === "string"
        ? entry.field
        : null;
    if (field) add(errors, field, message);
  }
  return errors;
}

function fromMap(map: Record<string, unknown>, skipReserved: boolean): FieldErrors {
  const errors: FieldErrors = {};
  for (const [field, value] of Object.entries(map)) {
    if (skipReserved && RESERVED_KEYS.has(field)) continue;
    const message = asMessage(value);
    if (message) add(errors, field, message);
  }
  return errors;
}

export function fieldErrorsFromDetails(details: unknown, fallbackMessage: string): FieldErrors {
  if (Array.isArray(details)) return fromList(details);
  if (!isRecord(details)) return {};
  if (isRecord(details.fields)) return fromMap(details.fields, false);
  if (Array.isArray(details.errors)) return fromList(details.errors);
  if (typeof details.field === "string") {
    return { [details.field]: asMessage(details.message) ?? fallbackMessage };
  }
  return fromMap(details, true);
}

/** Field errors carried by a validation error; empty for any other kind of error. */
export function fieldErrorsFromApiError(error: ApiError): FieldErrors {
  if (error.status !== 422 && error.code !== "validation_error") return {};
  return fieldErrorsFromDetails(error.details, error.message);
}
