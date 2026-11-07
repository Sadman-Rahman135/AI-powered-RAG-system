import { describe, expect, it } from "vitest";

import { isAuthorized, parseBasicAuthConfig } from "@/lib/basic-auth";

function basic(credentials: string): string {
  const bytes = new TextEncoder().encode(credentials);
  return "Basic " + btoa(String.fromCharCode(...bytes));
}

describe("parseBasicAuthConfig", () => {
  it("reads user:pass and allows colons in the password", () => {
    expect(parseBasicAuthConfig("ops:s3cret")).toEqual({ username: "ops", password: "s3cret" });
    expect(parseBasicAuthConfig("ops:a:b:c")).toEqual({ username: "ops", password: "a:b:c" });
  });

  it("is disabled when unset or malformed", () => {
    expect(parseBasicAuthConfig(undefined)).toBeNull();
    expect(parseBasicAuthConfig("")).toBeNull();
    expect(parseBasicAuthConfig("no-colon")).toBeNull();
    expect(parseBasicAuthConfig(":password-only")).toBeNull();
    expect(parseBasicAuthConfig("user:")).toBeNull();
  });
});

describe("isAuthorized", () => {
  const config = { username: "ops", password: "s3cret:x" };

  it("accepts the configured credentials", () => {
    expect(isAuthorized(basic("ops:s3cret:x"), config)).toBe(true);
    expect(isAuthorized("basic " + basic("ops:s3cret:x").slice(6), config)).toBe(true);
  });

  it("rejects anything else", () => {
    expect(isAuthorized(null, config)).toBe(false);
    expect(isAuthorized("", config)).toBe(false);
    expect(isAuthorized(basic("ops:wrong"), config)).toBe(false);
    expect(isAuthorized(basic("other:s3cret:x"), config)).toBe(false);
    expect(isAuthorized("Bearer abc", config)).toBe(false);
    expect(isAuthorized("Basic !!!not-base64!!!", config)).toBe(false);
  });

  it("handles non-ASCII credentials", () => {
    expect(isAuthorized(basic("zoë:pässword"), { username: "zoë", password: "pässword" })).toBe(true);
  });
});
