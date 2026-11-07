/**
 * Exact arithmetic on plain-notation decimal strings ("1234.50") using BigInt, so money never touches floating
 * point in the browser either.
 */
const DECIMAL_RE = /^([+-])?(\d+)(?:\.(\d+))?$/;

interface Scaled {
  units: bigint;
  scale: number;
}

function parse(value: string): Scaled | null {
  const match = DECIMAL_RE.exec(value.trim());
  if (!match) return null;
  const [, sign, whole, fraction = ""] = match;
  const units = BigInt(whole + fraction) * (sign === "-" ? -1n : 1n);
  return { units, scale: fraction.length };
}

function rescale(value: Scaled, scale: number): bigint {
  return value.units * 10n ** BigInt(scale - value.scale);
}

function format(units: bigint, scale: number): string {
  const negative = units < 0n;
  const digits = (negative ? -units : units).toString().padStart(scale + 1, "0");
  const whole = scale > 0 ? digits.slice(0, -scale) : digits;
  const fraction = scale > 0 ? `.${digits.slice(-scale)}` : "";
  return `${negative ? "-" : ""}${whole}${fraction}`;
}

/** Sum of the non-blank values, at the largest scale among them; `null` if none, or if any is not a number. */
export function sumDecimals(values: ReadonlyArray<string | null | undefined>): string | null {
  const parsed: Scaled[] = [];
  for (const value of values) {
    if (value == null || value.trim() === "") continue;
    const decimal = parse(value);
    if (!decimal) return null;
    parsed.push(decimal);
  }
  if (parsed.length === 0) return null;
  const scale = Math.max(...parsed.map((d) => d.scale));
  const total = parsed.reduce((sum, d) => sum + rescale(d, scale), 0n);
  return format(total, scale);
}

/** -1, 0 or 1 comparing `a` with `b`, treating differences within `tolerance` as equal; `null` if unparseable. */
export function compareDecimals(a: string, b: string, tolerance: string): -1 | 0 | 1 | null {
  const left = parse(a);
  const right = parse(b);
  const limit = parse(tolerance);
  if (!left || !right || !limit) return null;
  const scale = Math.max(left.scale, right.scale, limit.scale);
  const difference = rescale(left, scale) - rescale(right, scale);
  const bound = rescale(limit, scale);
  if (difference > bound) return 1;
  if (difference < -bound) return -1;
  return 0;
}
