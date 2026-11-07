import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { ReprocessDialog } from "@/components/reprocess-dialog";

function Harness({ onConfirm }: { onConfirm: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Reprocess
      </button>
      <ReprocessDialog filename="inv-1.pdf" open={open} onOpenChange={setOpen} onConfirm={onConfirm} />
    </>
  );
}

describe("ReprocessDialog", () => {
  it("warns that records are replaced and does nothing until confirmed", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(<Harness onConfirm={onConfirm} />);

    await user.click(screen.getByRole("button", { name: "Reprocess" }));

    expect(await screen.findByText(/replaces the extracted records/i)).toBeInTheDocument();
    expect(screen.getByText(/stay in the audit history but are not re-applied/i)).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();

    await user.click(screen.getByTestId("confirm-reprocess"));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("cancelling does not reprocess", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(<Harness onConfirm={onConfirm} />);

    await user.click(screen.getByRole("button", { name: "Reprocess" }));
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(onConfirm).not.toHaveBeenCalled();
  });
});
