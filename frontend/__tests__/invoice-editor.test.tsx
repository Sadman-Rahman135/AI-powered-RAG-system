import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { InvoiceEditor } from "@/components/invoice-editor";
import { ApiError } from "@/lib/api";
import type { Invoice, InvoicePatch } from "@/lib/types";

import { makeInvoice, makeIssue } from "./fixtures";
import { renderWithProviders } from "./test-utils";

function setup(invoice: Invoice = makeInvoice()) {
  const user = userEvent.setup();
  const onSave = vi.fn<(patch: InvoicePatch) => Promise<unknown>>(async () => invoice);
  const onApprove = vi.fn<(comment: string | null) => Promise<unknown>>(async () => invoice);
  renderWithProviders(<InvoiceEditor invoice={invoice} onSave={onSave} onApprove={onApprove} />);
  return { user, onSave, onApprove };
}

function fieldRow(name: string) {
  return screen.getByTestId("field-input-" + name).closest("[data-testid='invoice-field']") as HTMLElement;
}

describe("InvoiceEditor", () => {
  it("shows the extracted values", () => {
    setup();

    expect(screen.getByTestId("field-input-invoice_number")).toHaveValue("INV-0042");
    expect(screen.getByTestId("field-input-total")).toHaveValue("110.00");
    expect(screen.getByTestId("field-input-invoice_date")).toHaveValue("2024-03-05");
    expect(screen.getByTestId("field-input-discount")).toHaveValue("");
  });

  it("shows confidence and source for each extracted field", () => {
    setup();

    const total = within(fieldRow("total"));
    expect(total.getByTestId("confidence-badge")).toHaveTextContent("93%");
    expect(total.getByTestId("confidence-badge")).toHaveAttribute("data-bucket", "high");
    expect(total.getByText("Rules")).toBeInTheDocument();

    const supplier = within(fieldRow("supplier_name"));
    expect(supplier.getByTestId("confidence-badge")).toHaveAttribute("data-bucket", "low");
    expect(supplier.getByText("LLM")).toBeInTheDocument();
  });

  it("reveals the evidence behind a value", async () => {
    const { user } = setup();

    await user.hover(screen.getByRole("button", { name: "Evidence for Total" }));

    // The evidence is quoted verbatim (Testing Library collapses its double space when matching).
    expect(await screen.findByText(/Total due\s+110\.00/)).toBeInTheDocument();
    expect(screen.getByText(/Page 2/)).toBeInTheDocument();
  });

  it("starts clean, and an edit marks the field dirty and enables Save", async () => {
    const { user } = setup();

    const save = screen.getByTestId("save-invoice");
    expect(save).toBeDisabled();
    expect(fieldRow("total")).toHaveAttribute("data-dirty", "false");

    await user.clear(screen.getByTestId("field-input-total"));
    await user.type(screen.getByTestId("field-input-total"), "120.00");

    expect(fieldRow("total")).toHaveAttribute("data-dirty", "true");
    expect(fieldRow("subtotal")).toHaveAttribute("data-dirty", "false");
    expect(save).toBeEnabled();
    expect(screen.getByText("1 unsaved change")).toBeInTheDocument();
  });

  it("typing a value back to the original clears the dirty mark", async () => {
    const { user } = setup();
    const input = screen.getByTestId("field-input-total");

    await user.clear(input);
    await user.type(input, "110.00");

    expect(fieldRow("total")).toHaveAttribute("data-dirty", "false");
    expect(screen.getByTestId("save-invoice")).toBeDisabled();
  });

  it("submits only the fields that changed", async () => {
    const { user, onSave } = setup();

    await user.clear(screen.getByTestId("field-input-total"));
    await user.type(screen.getByTestId("field-input-total"), "120.00");
    await user.clear(screen.getByTestId("field-input-supplier_name"));
    await user.type(screen.getByTestId("field-input-supplier_name"), "  Acme Freight AG ");
    await user.type(screen.getByLabelText("Comment (optional)"), "Matched to PO 7781");
    await user.click(screen.getByTestId("save-invoice"));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith({
      fields: { supplier_name: "Acme Freight AG", total: "120.00" },
      line_items: null,
      comment: "Matched to PO 7781",
    });
  });

  it("sends null for a cleared field and upper-cases the currency", async () => {
    const { user, onSave } = setup();

    await user.clear(screen.getByTestId("field-input-customer_name"));
    await user.clear(screen.getByTestId("field-input-currency"));
    await user.type(screen.getByTestId("field-input-currency"), "usd");
    await user.click(screen.getByTestId("save-invoice"));

    expect(onSave).toHaveBeenCalledWith({
      fields: { customer_name: null, currency: "USD" },
      line_items: null,
      comment: null,
    });
  });

  it("sends the full line-item list when any line changes", async () => {
    const { user, onSave } = setup();

    const amount = screen.getByLabelText("Line 2 amount");
    await user.clear(amount);
    await user.type(amount, "25.00");
    await user.click(screen.getByTestId("save-invoice"));

    expect(onSave).toHaveBeenCalledWith({
      fields: {},
      line_items: [
        { description: "Pallet transport", quantity: "2", unit_price: "40.00", amount: "80.00" },
        { description: "Handling", quantity: "1", unit_price: "20.00", amount: "25.00" },
      ],
      comment: null,
    });
  });

  it("adds and removes line items", async () => {
    const { user, onSave } = setup();

    await user.click(screen.getByRole("button", { name: "Remove line 1" }));
    await user.click(screen.getByRole("button", { name: "Add line" }));
    await user.type(screen.getByLabelText("Line 2 description"), "Fuel surcharge");
    await user.click(screen.getByTestId("save-invoice"));

    expect(onSave.mock.calls[0][0].line_items).toEqual([
      { description: "Handling", quantity: "1", unit_price: "20.00", amount: "20.00" },
      { description: "Fuel surcharge", quantity: null, unit_price: null, amount: null },
    ]);
  });

  it("shows the server's 422 validation messages next to the fields", async () => {
    const { user, onSave } = setup();
    onSave.mockRejectedValueOnce(
      new ApiError(422, "validation_error", "Some values are not valid.", [
        { loc: ["body", "fields", "total"], msg: "Not a valid amount", type: "decimal_parsing" },
      ]),
    );

    const total = screen.getByTestId("field-input-total");
    await user.clear(total);
    await user.type(total, "12O.00");
    await user.click(screen.getByTestId("save-invoice"));

    expect(await within(fieldRow("total")).findByText("Not a valid amount")).toBeInTheDocument();
    expect(total).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText("Some values are not valid.")).toBeInTheDocument();
    // The edit is kept so the reviewer can fix it.
    expect(total).toHaveValue("12O.00");
    expect(fieldRow("total")).toHaveAttribute("data-dirty", "true");
  });

  it("lists the validation issues of a field under it", () => {
    setup(
      makeInvoice({
        issues: [makeIssue({ field: "total", code: "TOTAL_MISMATCH", message: "Subtotal + tax does not equal the total." })],
      }),
    );

    expect(within(fieldRow("total")).getByText("Subtotal + tax does not equal the total.")).toBeInTheDocument();
  });

  it("requires changes to be saved before approving", async () => {
    const { user, onApprove } = setup();
    const approve = screen.getByTestId("approve-invoice");
    expect(approve).toBeEnabled();

    await user.type(screen.getByTestId("field-input-invoice_number"), "-A");
    expect(approve).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Discard changes" }));
    expect(screen.getByTestId("field-input-invoice_number")).toHaveValue("INV-0042");
    expect(approve).toBeEnabled();

    await user.type(screen.getByLabelText("Comment (optional)"), "Looks right");
    await user.click(approve);
    expect(onApprove).toHaveBeenCalledWith("Looks right");
  });

  it("does not offer approval again for a reviewed record", () => {
    setup(makeInvoice({ review_status: "reviewed", reviewed_by: "ops@example.com", reviewed_at: "2026-11-04T11:00:00Z" }));

    expect(screen.getByTestId("approve-invoice")).toBeDisabled();
    expect(screen.getByText(/ops@example\.com/)).toBeInTheDocument();
  });
});
