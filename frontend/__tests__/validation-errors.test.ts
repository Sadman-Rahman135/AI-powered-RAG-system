import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/api";
import { fieldErrorsFromApiError } from "@/lib/validation-errors";

function err(details: unknown, message = "Some values are not valid.") {
  return new ApiError(422, "validation_error", message, details);
}

describe("fieldErrorsFromApiError", () => {
  it("reads pydantic-style error lists", () => {
    expect(
      fieldErrorsFromApiError(
        err([
          { loc: ["body", "fields", "total"], msg: "Not a valid amount", type: "decimal_parsing" },
          { loc: ["body", "line_items", 1, "amount"], msg: "Not a valid amount" },
          { loc: ["body", "comment"], msg: "Too long" },
        ]),
      ),
    ).toEqual({ total: "Not a valid amount", "line_items.1.amount": "Not a valid amount", comment: "Too long" });
  });

  it("reads field maps and error lists", () => {
    expect(fieldErrorsFromApiError(err({ fields: { due_date: "Due date is before the invoice date." } }))).toEqual({
      due_date: "Due date is before the invoice date.",
    });
    expect(fieldErrorsFromApiError(err({ fields: { total: ["Too many decimals.", "Must not be negative."] } }))).toEqual({
      total: "Too many decimals. Must not be negative.",
    });
    expect(fieldErrorsFromApiError(err({ errors: [{ field: "currency", message: "Unknown currency code." }] }))).toEqual({
      currency: "Unknown currency code.",
    });
    expect(fieldErrorsFromApiError(err({ total: "Not a valid amount" }))).toEqual({ total: "Not a valid amount" });
  });

  it("uses the error message for a single named field", () => {
    expect(fieldErrorsFromApiError(err({ field: "invoice_date" }, "Not a valid date."))).toEqual({
      invoice_date: "Not a valid date.",
    });
  });

  it("returns nothing for errors it cannot attribute to a field", () => {
    expect(fieldErrorsFromApiError(err(null))).toEqual({});
    expect(fieldErrorsFromApiError(err("bad"))).toEqual({});
    expect(fieldErrorsFromApiError(err({ limit: 50 }))).toEqual({});
    expect(fieldErrorsFromApiError(new ApiError(409, "already_processing", "Busy", { fields: { total: "x" } }))).toEqual({});
  });
});
