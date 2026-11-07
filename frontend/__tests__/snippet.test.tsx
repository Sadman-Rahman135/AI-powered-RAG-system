import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Snippet } from "@/components/snippet";
import { splitHighlights } from "@/lib/highlight";

describe("splitHighlights", () => {
  it("splits « » markers into highlighted segments", () => {
    expect(splitHighlights("Total «110.00» due on «5 March»")).toEqual([
      { text: "Total ", highlighted: false },
      { text: "110.00", highlighted: true },
      { text: " due on ", highlighted: false },
      { text: "5 March", highlighted: true },
    ]);
  });

  it("returns plain text untouched", () => {
    expect(splitHighlights("no markers here")).toEqual([{ text: "no markers here", highlighted: false }]);
    expect(splitHighlights("")).toEqual([]);
  });

  it("drops stray or unbalanced markers instead of mis-highlighting", () => {
    expect(splitHighlights("a » b «c")).toEqual([
      { text: "a  b ", highlighted: false },
      { text: "c", highlighted: false },
    ]);
    expect(splitHighlights("«»x")).toEqual([{ text: "x", highlighted: false }]);
  });
});

describe("Snippet", () => {
  it("renders highlights as <mark> elements", () => {
    const { container } = render(<Snippet text="Invoice «INV-0042» from «Acme»" />);

    const marks = Array.from(container.querySelectorAll("mark")).map((m) => m.textContent);
    expect(marks).toEqual(["INV-0042", "Acme"]);
    expect(container).toHaveTextContent("Invoice INV-0042 from Acme");
  });

  it("treats markup inside the document text as text, never as HTML", () => {
    const { container } = render(<Snippet text={'<img src=x onerror="alert(1)"> «<b>bold</b>»'} />);

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(container.querySelector("mark")).toHaveTextContent("<b>bold</b>");
  });
});
