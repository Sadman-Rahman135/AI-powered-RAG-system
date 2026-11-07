import { describe, expect, it } from "vitest";

import { compareDecimals, sumDecimals } from "@/lib/decimal";

describe("sumDecimals", () => {
  it("adds decimal strings exactly", () => {
    expect(sumDecimals(["80.00", "20.00"])).toBe("100.00");
    expect(sumDecimals(["0.1", "0.2"])).toBe("0.3");
    expect(sumDecimals(["1.005", "2"])).toBe("3.005");
    expect(sumDecimals(["-10.50", "4"])).toBe("-6.50");
    expect(sumDecimals(["99999999999999999.99", "0.01"])).toBe("100000000000000000.00");
  });

  it("ignores blanks and gives up on anything that is not a number", () => {
    expect(sumDecimals(["10", "", null, undefined, " 5 "])).toBe("15");
    expect(sumDecimals([])).toBeNull();
    expect(sumDecimals(["", null])).toBeNull();
    expect(sumDecimals(["10", "1,5"])).toBeNull();
  });
});

describe("compareDecimals", () => {
  it("compares within a tolerance", () => {
    expect(compareDecimals("100.00", "100.01", "0.02")).toBe(0);
    expect(compareDecimals("100.00", "100.03", "0.02")).toBe(-1);
    expect(compareDecimals("100.05", "100", "0.02")).toBe(1);
    expect(compareDecimals("x", "1", "0")).toBeNull();
  });
});
