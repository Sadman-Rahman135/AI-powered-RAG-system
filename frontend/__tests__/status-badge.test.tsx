import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ReviewStatusBadge, SeverityBadge, StatusBadge } from "@/components/status-badge";

describe("StatusBadge", () => {
  it.each([
    ["queued", "Queued", "neutral"],
    ["processing", "Processing", "info"],
    ["completed", "Completed", "success"],
    ["needs_review", "Needs review", "warning"],
    ["failed", "Failed", "danger"],
  ] as const)("renders %s as '%s' with the %s tone", (status, label, tone) => {
    render(<StatusBadge status={status} />);

    const badge = screen.getByTestId("status-badge");
    expect(badge).toHaveTextContent(label);
    expect(badge).toHaveAttribute("data-status", status);
    expect(badge).toHaveAttribute("data-tone", tone);
  });

  it("pairs the colour with an icon so status is not conveyed by colour alone", () => {
    render(<StatusBadge status="failed" />);

    const badge = screen.getByTestId("status-badge");
    const icon = badge.querySelector("svg");
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute("aria-hidden", "true");
  });

  it("falls back to a neutral, readable label for an unknown status", () => {
    render(<StatusBadge status="paused_by_admin" />);

    const badge = screen.getByTestId("status-badge");
    expect(badge).toHaveTextContent("Paused by admin");
    expect(badge).toHaveAttribute("data-tone", "neutral");
  });
});

describe("ReviewStatusBadge and SeverityBadge", () => {
  it("labels review states", () => {
    render(
      <>
        <ReviewStatusBadge status="auto_approved" />
        <ReviewStatusBadge status="needs_review" />
        <ReviewStatusBadge status="reviewed" />
      </>,
    );

    const badges = screen.getAllByTestId("review-status-badge");
    expect(badges.map((b) => b.textContent)).toEqual(["Auto-approved", "Needs review", "Reviewed"]);
    expect(badges[1]).toHaveAttribute("data-tone", "warning");
  });

  it("labels severities with their tone", () => {
    render(
      <>
        <SeverityBadge severity="error" />
        <SeverityBadge severity="warning" />
        <SeverityBadge severity="info" />
      </>,
    );

    const badges = screen.getAllByTestId("severity-badge");
    expect(badges.map((b) => [b.textContent, b.getAttribute("data-tone")])).toEqual([
      ["Error", "danger"],
      ["Warning", "warning"],
      ["Info", "info"],
    ]);
  });
});
