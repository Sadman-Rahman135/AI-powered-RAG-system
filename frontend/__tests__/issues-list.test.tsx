import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { IssuesList } from "@/components/issues-list";

import { makeIssue } from "./fixtures";

describe("IssuesList", () => {
  const issues = [
    makeIssue({ code: "LOW_CONFIDENCE", severity: "warning", field: "supplier_name", message: "Supplier read with low confidence." }),
    makeIssue({ code: "TOTAL_MISMATCH", severity: "error", field: "total", message: "Subtotal + tax does not equal the total." }),
    makeIssue({ code: "AMBIGUOUS_DATE", severity: "info", field: "invoice_date", message: "Day and month could be swapped." }),
    makeIssue({ code: "MISSING_FIELD", severity: "error", field: "invoice_number", message: "Invoice number is missing." }),
  ];

  it("groups issues by severity: errors first, then warnings, then info", () => {
    render(<IssuesList issues={issues} />);

    const groups = screen.getAllByTestId("issue-group");
    expect(groups.map((g) => g.getAttribute("data-severity"))).toEqual(["error", "warning", "info"]);

    const errors = within(groups[0]);
    expect(errors.getByRole("heading")).toHaveTextContent("Errors");
    expect(errors.getByRole("heading")).toHaveTextContent("2");
    expect(errors.getAllByTestId("issue-item").map((i) => i.getAttribute("data-code"))).toEqual([
      "TOTAL_MISMATCH",
      "MISSING_FIELD",
    ]);

    expect(within(groups[1]).getByRole("heading")).toHaveTextContent("Warnings");
    expect(within(groups[1]).getByText("Supplier read with low confidence.")).toBeInTheDocument();
    expect(within(groups[2]).getByRole("heading")).toHaveTextContent("Info");
  });

  it("shows the issue code in words next to the message", () => {
    render(<IssuesList issues={[issues[1]]} />);

    const item = screen.getByTestId("issue-item");
    expect(item).toHaveTextContent("Total mismatch");
    expect(item).toHaveTextContent("Subtotal + tax does not equal the total.");
  });

  it("omits severities that have no issues", () => {
    render(<IssuesList issues={[issues[0]]} />);

    const groups = screen.getAllByTestId("issue-group");
    expect(groups).toHaveLength(1);
    expect(groups[0]).toHaveAttribute("data-severity", "warning");
  });

  it("links an issue to its field", async () => {
    const user = userEvent.setup();
    const onSelectField = vi.fn();
    render(<IssuesList issues={issues} onSelectField={onSelectField} />);

    await user.click(screen.getByRole("button", { name: "Go to Total" }));

    expect(onSelectField).toHaveBeenCalledTimes(1);
    expect(onSelectField.mock.calls[0][0]).toMatchObject({ code: "TOTAL_MISMATCH", field: "total" });
  });

  it("names the record an issue belongs to when a labeller is given", () => {
    render(
      <IssuesList
        issues={[makeIssue({ invoice_id: "inv-7" }), makeIssue({ invoice_id: null, field: null, code: "DUPLICATE_FILE", severity: "warning" })]}
        recordLabel={(invoiceId) => (invoiceId ? `Record for ${invoiceId}` : "Whole document")}
      />,
    );

    expect(screen.getByText("Record for inv-7")).toBeInTheDocument();
    expect(screen.getByText("Whole document")).toBeInTheDocument();
  });

  it("says so when there is nothing to review", () => {
    render(<IssuesList issues={[]} />);

    expect(screen.queryAllByTestId("issue-group")).toHaveLength(0);
    expect(screen.getByText("No validation issues")).toBeInTheDocument();
  });
});
