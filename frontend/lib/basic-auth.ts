/**
 * Optional HTTP basic auth in front of the UI, configured with `UI_BASIC_AUTH=user:pass`.
 *
 * Runs in the Next.js proxy (pages) and in the API route handler, so it only uses Web APIs. Credentials are
 * compared in constant time.
 */
export interface BasicAuthConfig {
  username: string;
  password: string;
}

export const BASIC_AUTH_CHALLENGE = 'Basic realm="DocIntel", charset="UTF-8"';

/** `"user:pass"` -> credentials (the password may contain colons); `null` disables basic auth. */
export function parseBasicAuthConfig(value: string | null | undefined): BasicAuthConfig | null {
  if (!value) return null;
  const colon = value.indexOf(":");
  if (colon <= 0 || colon === value.length - 1) return null;
  return { username: value.slice(0, colon), password: value.slice(colon + 1) };
}

function decodeBase64Utf8(encoded: string): string | null {
  try {
    const binary = atob(encoded);
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

/** Compare two strings without an early exit, so timing does not reveal how much of a secret matched. */
export function timingSafeEqual(a: string, b: string): boolean {
  const encoder = new TextEncoder();
  const left = encoder.encode(a);
  const right = encoder.encode(b);
  const length = Math.max(left.length, right.length);
  let difference = left.length ^ right.length;
  for (let i = 0; i < length; i += 1) {
    difference |= (left[i] ?? 0) ^ (right[i] ?? 0);
  }
  return difference === 0;
}

export function isAuthorized(header: string | null | undefined, config: BasicAuthConfig): boolean {
  if (!header) return false;
  const match = /^basic\s+(\S+)\s*$/i.exec(header.trim());
  if (!match) return false;
  const decoded = decodeBase64Utf8(match[1]);
  if (decoded === null) return false;
  return timingSafeEqual(decoded, `${config.username}:${config.password}`);
}
