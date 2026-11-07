"""The six invoice/receipt templates: number and date conventions, label vocabularies and page layouts.

T1-T3 are the *dev* split, T4-T6 the *held-out* split (see ``eval.labels``):

====  =============================  ==============================  =========================================
T1    US letter, Helvetica           ``$1,234.56``, ``MM/DD/YYYY``   sales tax, ship-to box, balance due
T2    EU VAT, serif                  ``1.234,56 €``, ``DD.MM.YYYY``  7-column table, IBAN, per-line VAT rate
T3    UK, Times                      ``£1,234.56``, ``5 March 2024`` centred letterhead, VAT @ 20%, sort code
T4    Indian GST, grid               ``₹ 1,23,456.00``, CGST + SGST  HSN column, amount in words, GSTIN
T5    80 mm thermal receipt, mono    ``$12.34`` + ``CAD``            one tall page, barcode, GST + PST/QST/HST
T6    two-column modern, AUD         ``A$1,234.50``                  values above their labels, ABN, BSB
====  =============================  ==============================  =========================================

A template turns an :class:`~eval.synth.data.InvoiceSpec` into HTML blocks on a ``Canvas``; it never decides
*content* (the spec does) — only how it looks, which label words are used and how numbers and dates are written.
"""

from __future__ import annotations

import html as _html
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from app.domain.schemas import DocumentType
from eval.synth.data import InvoiceSpec, Item, Party, TaxLine, plain

if TYPE_CHECKING:
    from eval.synth.render import Canvas, PagePlan

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
PALETTE = ("#1f3a5f", "#0b6e4f", "#8c1c13", "#2d5f8b", "#5b2a86", "#b4530a", "#006d77", "#333333")


def esc(text: str | None) -> str:
    return _html.escape(text or "", quote=False)


def rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


# --------------------------------------------------------------------------------------------------------------------
# number and date formatting
# --------------------------------------------------------------------------------------------------------------------
def group_digits(digits: str, sep: str, *, indian: bool = False) -> str:
    if not sep:
        return digits
    if indian and len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        return sep.join([*parts, tail])
    parts = []
    while len(digits) > 3:
        parts.insert(0, digits[-3:])
        digits = digits[:-3]
    return sep.join([digits, *parts]) if digits else sep.join(parts)


def fmt_number(value: Decimal, *, group: str = ",", dec: str = ".", indian: bool = False, places: int = 2) -> str:
    """``1234.5`` -> ``1,234.50`` / ``1.234,50`` / ``1 234,50`` / ``1,234.50`` (Indian grouping ``12,34,567.00``)."""
    quantum = Decimal(1).scaleb(-places)
    text = f"{abs(value).quantize(quantum):f}"
    whole, _, frac = text.partition(".")
    out = group_digits(whole, group, indian=indian) + (dec + frac if places else "")
    return ("-" if value < 0 else "") + out


def fmt_qty(value: Decimal, *, dec: str = ".", places: int = 0) -> str:
    text = format(plain(value), "f")
    whole, _, frac = text.partition(".")
    frac = frac.ljust(places, "0")
    return whole + (dec + frac if frac else "")


def fmt_rate(rate: Decimal, *, dec: str = ".") -> str:
    return fmt_qty(rate, dec=dec)


def fmt_date(day: date, style: str) -> str:
    d, m, y = day.day, day.month, day.year
    month = MONTHS[m - 1]
    return {
        "us": f"{m:02d}/{d:02d}/{y}",
        "us_long": f"{month} {d}, {y}",
        "us_mon": f"{month[:3]} {d}, {y}",
        "eu_dot": f"{d:02d}.{m:02d}.{y}",
        "eu_slash": f"{d:02d}/{m:02d}/{y}",
        "uk_long": f"{d} {month} {y}",
        "uk_short": f"{d} {month[:3]} {y}",
        "in_slash": f"{d:02d}/{m:02d}/{y}",
        "in_dash": f"{d:02d}-{m:02d}-{y}",
        "in_mon": f"{d:02d}-{month[:3]}-{y}",
        "iso": f"{y}-{m:02d}-{d:02d}",
        "au_long": f"{d} {month} {y}",
        "au_short": f"{d:02d} {month[:3]} {y}",
    }[style]


def number_in_words_inr(amount: Decimal) -> str:
    """``255719.58`` -> ``Rupees Two Lakh Fifty-Five Thousand Seven Hundred Nineteen and Fifty-Eight Paise Only``."""
    ones = (
        "",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Eleven",
        "Twelve",
        "Thirteen",
        "Fourteen",
        "Fifteen",
        "Sixteen",
        "Seventeen",
        "Eighteen",
        "Nineteen",
    )
    tens = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")

    def below_hundred(n: int) -> str:
        return ones[n] if n < 20 else tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")

    def below_thousand(n: int) -> str:
        head = f"{ones[n // 100]} Hundred" if n >= 100 else ""
        rest = below_hundred(n % 100) if n % 100 else ""
        return f"{head} {rest}".strip()

    rupees, paise = divmod(int((abs(amount) * 100).to_integral_value()), 100)
    words: list[str] = []
    for unit, size in (("Crore", 10_000_000), ("Lakh", 100_000), ("Thousand", 1000)):
        part, rupees = divmod(rupees, size)
        if part:
            words.append(f"{below_hundred(part)} {unit}" if part < 100 else f"{below_thousand(part)} {unit}")
    if rupees:
        words.append(below_thousand(rupees))
    text = "Rupees " + (" ".join(words) or "Zero")
    if paise:
        text += f" and {below_hundred(paise)} Paise"
    return text + " Only"


# --------------------------------------------------------------------------------------------------------------------
# label vocabularies (one word picked per document)
# --------------------------------------------------------------------------------------------------------------------
VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "T1": {
        "inv_no": ("Invoice #", "Invoice No.", "Invoice Number", "Invoice No"),
        "inv_date": ("Invoice Date", "Date", "Date Issued", "Issue Date"),
        "due": ("Due Date", "Payment Due", "Pay By", "Due"),
        "terms": ("Terms",),
        "po": ("PO #", "PO Number", "Customer PO"),
        "cust_no": ("Customer ID", "Account #"),
        "svc": ("Ship Date",),
        "pos": ("Place of Supply",),
        "ref": ("Reference",),
        "bill_to": ("Bill To", "Billed To", "Sold To", "Customer", "Invoice To"),
        "desc": ("Description", "Item Description", "Item"),
        "qty": ("Qty", "Quantity", "Qty."),
        "price": ("Rate", "Unit Price", "Price", "Unit Cost"),
        "amount": ("Amount", "Line Total", "Total", "Ext. Price"),
        "subtotal": ("Subtotal", "Sub-total", "Sub Total"),
        "discount": ("Discount", "Less: Discount", "Promo Discount", "Volume Discount"),
        "shipping": ("Shipping", "Shipping & Handling", "Freight", "Delivery"),
        "total": ("Total", "Total Due", "Amount Due", "Balance Due", "Invoice Total"),
    },
    "T2": {
        "inv_no": ("Invoice No.", "Invoice number", "Invoice no:", "Document No."),
        "inv_date": ("Invoice date", "Date", "Date of issue", "Issue date"),
        "due": ("Payment due", "Due date", "Due by", "Payable by"),
        "terms": ("Payment terms",),
        "po": ("Your order", "Order no."),
        "cust_no": ("Customer no.", "Client ref."),
        "svc": ("Delivery date", "Date of supply", "Service date"),
        "pos": ("Place of supply",),
        "ref": ("Our ref.",),
        "bill_to": ("Invoice recipient", "Bill to", "Customer"),
        "desc": ("Description", "Item", "Designation"),
        "qty": ("Qty", "Quantity", "Qty."),
        "price": ("Unit price", "Price", "Net price"),
        "amount": ("Net amount", "Amount", "Total"),
        "subtotal": ("Net total", "Subtotal", "Total net", "Net amount"),
        "discount": ("Discount", "Rebate", "Trade discount"),
        "shipping": ("Shipping costs", "Freight", "Delivery charge", "Carriage"),
        "total": ("Total amount", "Gross total", "Total due", "Invoice total", "Amount payable"),
    },
    "T3": {
        "inv_no": ("Invoice Number", "Invoice No", "Invoice No.", "Our Invoice No"),
        "inv_date": ("Invoice Date", "Date", "Date of Invoice", "Invoice Date"),
        "due": ("Payment Due", "Due Date", "Payment Due Date", "Please pay by"),
        "terms": ("Terms",),
        "po": ("Your Ref", "PO Number", "Order Ref"),
        "cust_no": ("Account No", "Customer Ref"),
        "svc": ("Tax Point", "Date of Supply"),
        "pos": ("Place of Supply",),
        "ref": ("Our Ref",),
        "bill_to": ("To", "Invoice To", "Customer", "Bill To"),
        "desc": ("Description", "Details", "Item"),
        "qty": ("Quantity", "Qty"),
        "price": ("Unit Price", "Price", "Rate"),
        "amount": ("Amount", "Net", "Line Total"),
        "subtotal": ("Sub-total", "Subtotal", "Net Total", "Total Net"),
        "discount": ("Discount", "Less Discount", "Trade Discount"),
        "shipping": ("Delivery", "Carriage", "Postage & Packing", "Delivery Charge"),
        "total": ("Total", "Total Due", "Amount Due", "Total Payable", "Invoice Total"),
    },
    "T4": {
        "inv_no": ("Invoice No.", "Invoice No", "Bill No.", "Tax Invoice No."),
        "inv_date": ("Invoice Date", "Dated", "Date", "Bill Date"),
        "due": ("Due Date", "Payment Due Date", "Due On"),
        "terms": ("Payment Terms", "Credit Terms"),
        "po": ("PO No.", "Buyer's Order No."),
        "cust_no": ("Customer Code",),
        "svc": ("Delivery Date", "Date of Supply"),
        "pos": ("Place of Supply", "State of Supply"),
        "ref": ("Ref No.",),
        "bill_to": ("Bill To", "Buyer (Bill to)", "Billed To", "Consignee"),
        "desc": ("Description of Goods / Services", "Description", "Particulars"),
        "qty": ("Qty", "Quantity"),
        "price": ("Rate", "Rate (per unit)", "Unit Price"),
        "amount": ("Amount", "Taxable Value", "Total"),
        "subtotal": ("Sub Total", "Total Taxable Value", "Taxable Amount", "Total Amount Before Tax"),
        "discount": ("Discount", "Less: Discount", "Trade Discount"),
        "shipping": ("Freight", "Freight Charges", "Packing & Forwarding", "Transport Charges"),
        "total": ("Grand Total", "Invoice Total", "Total Amount", "Net Payable", "Total Invoice Value"),
    },
    "T5": {
        "inv_no": ("RECEIPT #", "TRANS #", "Receipt No:", "Ref #", "INVOICE #"),
        "inv_date": ("Date:", "DATE", "Date/Time:"),
        "due": ("",),
        "terms": ("",),
        "po": ("",),
        "cust_no": ("Member #",),
        "svc": ("",),
        "pos": ("",),
        "ref": ("",),
        "bill_to": ("Customer:", "Member:"),
        "desc": ("",),
        "qty": ("",),
        "price": ("",),
        "amount": ("",),
        "subtotal": ("SUBTOTAL", "Sub-total", "SUB TOTAL"),
        "discount": ("SAVINGS", "DISCOUNT"),
        "shipping": ("DELIVERY",),
        "total": ("TOTAL", "BALANCE DUE", "AMOUNT DUE", "TOTAL DUE"),
    },
    "T6": {
        "inv_no": ("Invoice number", "Invoice no.", "Invoice", "Reference"),
        "inv_date": ("Issued", "Issue date", "Date issued", "Invoice date"),
        "due": ("Due", "Due date", "Payment due", "Pay by"),
        "terms": ("Terms",),
        "po": ("PO number", "Purchase order"),
        "cust_no": ("Client ID",),
        "svc": ("Service period",),
        "pos": ("Place of supply",),
        "ref": ("Project",),
        "bill_to": ("Billed to", "Bill to", "Client", "Invoice to"),
        "desc": ("Description", "Item", "Service"),
        "qty": ("Qty", "Quantity", "Hours / Qty"),
        "price": ("Price", "Rate", "Unit price"),
        "amount": ("Amount", "Total", "Line total"),
        "subtotal": ("Subtotal", "Sub-total", "Net"),
        "discount": ("Discount", "Less discount", "Loyalty discount"),
        "shipping": ("Shipping", "Delivery", "Freight"),
        "total": ("Total", "Total due", "Amount due", "Balance due"),
    },
}

TAX_WORDS: dict[str, tuple[str, ...]] = {
    "tax": ("Sales Tax", "Tax"),
    "state": ("State Tax", "State Sales Tax"),
    "county": ("County Tax", "City Tax"),
    "gst": ("GST",),
    "pst": ("PST",),
    "qst": ("QST", "TVQ"),
    "hst": ("HST",),
    "igst": ("IGST",),
    "cgst": ("CGST",),
    "sgst": ("SGST",),
}
VAT_WORDS = {"de": "MwSt.", "fr": "TVA", "es": "IVA", "it": "IVA", "nl": "BTW"}


@dataclass(frozen=True)
class Col:
    key: str  # desc | qty | unit_price | amount | idx | unit | vat | hsn
    header: str
    width: float
    align: str = "left"


def grid(
    cols: Sequence[Col],
    head: bool,
    rows: Sequence[Sequence[str]],
    *,
    pad: float,
    cls: str = "items",
    border: float = 0.0,
) -> str:
    """A fixed-layout table: a cell's content width is its column width minus padding and (collapsed) border."""

    def cell(tag: str, col: Col, text: str) -> str:
        align = {"right": "text-align:right;", "center": "text-align:center;"}.get(col.align, "")
        return f'<{tag} style="width:{col.width - 2 * pad - border:g}pt;{align}">{text}</{tag}>'

    total = sum(c.width for c in cols)
    out = [f'<table class="{cls}" style="width:{total:g}pt;table-layout:fixed;">']
    if head:
        out.append("<tr>" + "".join(cell("th", c, esc(c.header)) for c in cols) + "</tr>")
    for row in rows:
        out.append("<tr>" + "".join(cell("td", c, t) for c, t in zip(cols, row, strict=True)) + "</tr>")
    out.append("</table>")
    return "".join(out)


def draw_logo(cv: Canvas, x: float, y: float, size: float, color: str, kind: str) -> None:
    """A small vector brand mark (no text, so it never pollutes the text layer)."""
    c = rgb(color)
    if kind == "circle":
        cv.circle(x + size / 2, y + size / 2, size / 2, fill=c)
        cv.circle(x + size / 2, y + size / 2, size / 4, fill=(1, 1, 1))
    elif kind == "square":
        cv.rect(x, y, x + size, y + size, fill=c)
        cv.rect(x + size * 0.28, y + size * 0.28, x + size * 0.72, y + size * 0.72, fill=(1, 1, 1))
    elif kind == "diamond":
        cv.polygon([(x + size / 2, y), (x + size, y + size / 2), (x + size / 2, y + size), (x, y + size / 2)], fill=c)
    else:  # bars
        for i, h in enumerate((0.5, 0.8, 1.0, 0.65)):
            cv.rect(x + i * size / 4, y + size * (1 - h), x + i * size / 4 + size / 5, y + size, fill=c)


def barcode(cv: Canvas, x: float, y: float, width: float, height: float, digits: str) -> None:
    """A Code-128-looking bar pattern derived from the digits (decoration only)."""
    pos = x
    seed = [int(d) for d in digits] or [1]
    i = 0
    while pos < x + width:
        bar = 0.6 + (seed[i % len(seed)] % 4) * 0.5
        cv.rect(pos, y, min(pos + bar, x + width), y + height, fill=(0, 0, 0))
        pos += bar + 0.6 + ((seed[(i + 3) % len(seed)] + i) % 3) * 0.5
        i += 1


# --------------------------------------------------------------------------------------------------------------------
# base template
# --------------------------------------------------------------------------------------------------------------------
class Template:
    id = ""
    split = "dev"
    width = 595.0
    height = 842.0
    margin_left = 42.0
    margin_right = 42.0
    margin_top = 40.0
    margin_bottom = 36.0
    footer_h = 38.0
    single_page = False
    font = "Helvetica"
    size = 9.0
    pad = 4.0  # horizontal cell padding of the item table (pt)
    border = 0.0  # cell border width of the item table (pt)
    dpi_factor = 1.0  # receipts are scanned at a higher resolution than pages
    date_styles: tuple[tuple[str, int], ...] = (("iso", 1),)

    # ----- geometry -------------------------------------------------------------------------------------------
    @property
    def x0(self) -> float:
        return self.margin_left

    @property
    def x1(self) -> float:
        return self.width - self.margin_right

    @property
    def table_x(self) -> tuple[float, float]:
        return self.x0, self.x1

    def bottom(self, height: float) -> float:
        return height - self.margin_bottom

    def page_height(self, spec: InvoiceSpec) -> float:
        return self.height

    # ----- style ----------------------------------------------------------------------------------------------
    def new_style(self, rng: random.Random) -> dict[str, Any]:
        style: dict[str, Any] = {
            "v": {k: rng.choice(w) for k, w in VOCAB[self.id].items()},
            "brand": rng.choice(PALETTE),
            "logo": rng.choice(("circle", "square", "diamond", "bars", None)),
            "date": rng.choices([s for s, _ in self.date_styles], weights=[w for _, w in self.date_styles])[0],
            "page_fmt": rng.choices(("Page {k} of {n}", "Page {k}/{n}"), weights=(75, 25))[0],
            "single_page_no": rng.random() < 0.4,
            "cont_number": rng.random() < 0.7,
            "stamp": rng.choices((None, "PAID", "COPY"), weights=(90, 7, 3))[0],
            "due_first": rng.random() < 0.18,
            "show_terms": rng.random() < 0.35,
            "sym_rows": rng.random() < 0.45,
            "sym_sub": rng.random() < 0.8,
            "code_total": rng.random() < 0.2,
            "rate_fmt": rng.choice(self.rate_formats),
            "vat_word": rng.choice(("VAT", "local")),
            "balance_due": rng.random() < 0.35,
            "tax_name": rng.random(),
        }
        style.update(self.extra_style(rng))
        return style

    rate_formats: tuple[str, ...] = ("{n} ({r}%)",)

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {}

    def css(self, spec: InvoiceSpec) -> str:
        return (
            f"body {{font-family: {self.font}; font-size: {self.size}pt; color: #222;}} "
            "table {border-collapse: collapse; border-spacing: 0;} td, th {vertical-align: top;}"
        )

    # ----- formatting -----------------------------------------------------------------------------------------
    def number_style(self, spec: InvoiceSpec) -> dict[str, Any]:
        return {"group": ",", "dec": "."}

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        return num

    def money(self, spec: InvoiceSpec, value: Decimal, role: str = "row") -> str:
        return self.decorate(spec, fmt_number(value, **self.number_style(spec)), role)

    def date(self, spec: InvoiceSpec, day: date) -> str:
        return fmt_date(day, spec.style["date"])

    def qty(self, spec: InvoiceSpec, value: Decimal) -> str:
        return fmt_qty(value, dec=self.number_style(spec)["dec"])

    def token_needed(self, spec: InvoiceSpec, role: str) -> bool:
        """Whether a currency symbol/code goes on this kind of figure (a missing total forces tokens elsewhere)."""
        if not spec.currency_printed:
            return False
        if role == "total" or not spec.shows("total"):
            return True
        return spec.style["sym_sub"] if role == "sub" else spec.style["sym_rows"]

    # ----- shared content pieces -------------------------------------------------------------------------------
    def title(self, spec: InvoiceSpec) -> str:
        return "CREDIT NOTE" if spec.doc_type is DocumentType.CREDIT_NOTE else "INVOICE"

    def vocab(self, spec: InvoiceSpec, key: str) -> str:
        if key == "inv_no" and spec.doc_type is DocumentType.CREDIT_NOTE:
            return "Credit Note No."
        return spec.style["v"][key]

    def meta_items(self, spec: InvoiceSpec) -> list[tuple[str, str | date]]:
        """(label, value) pairs of the document header; dates stay ``date`` objects (spreadsheets type them)."""
        rows: list[tuple[str, str | date]] = []
        if spec.invoice_number is not None:
            rows.append((self.vocab(spec, "inv_no"), spec.invoice_number))
        date_row: list[tuple[str, str | date]] = (
            [(self.vocab(spec, "inv_date"), spec.invoice_date)] if spec.invoice_date else []
        )
        due_row: list[tuple[str, str | date]] = [(self.vocab(spec, "due"), spec.due_date)] if spec.due_date else []
        if spec.style["due_first"]:
            rows += due_row + date_row
        else:
            rows += date_row
            if spec.style["show_terms"] and spec.due_date is None and self.vocab(spec, "terms"):
                rows.append((self.vocab(spec, "terms"), f"Net {spec.extra['terms_days']}"))
            rows += due_row
        if spec.service_date:
            rows.append((self.vocab(spec, "svc"), spec.service_date))
        if spec.extra["po"]:
            rows.append((self.vocab(spec, "po"), spec.extra["po"]))
        if spec.extra["customer_no"] and spec.customer is not None:
            rows.append((self.vocab(spec, "cust_no"), spec.extra["customer_no"]))
        return rows

    def meta_rows(self, spec: InvoiceSpec) -> list[tuple[str, str]]:
        """``meta_items`` with the dates written the way this template writes them."""
        return [(k, self.date(spec, v) if isinstance(v, date) else v) for k, v in self.meta_items(spec)]

    def sheet_columns(self, spec: InvoiceSpec) -> list[Col]:
        """Item-table columns of a spreadsheet rendering of this document."""
        return self.columns(spec)

    def rate_text(self, spec: InvoiceSpec, rate: Decimal) -> str:
        return fmt_rate(rate, dec=self.number_style(spec)["dec"])

    def tax_name(self, spec: InvoiceSpec, kind: str) -> str:
        if kind == "vat":
            local = VAT_WORDS.get(spec.market) if spec.style["vat_word"] == "local" and self.id == "T2" else None
            return local or "VAT"
        words = TAX_WORDS[kind]
        return words[int(spec.style["tax_name"] * len(words)) % len(words)]

    def tax_label(self, spec: InvoiceSpec, line: TaxLine) -> str:
        name = self.tax_name(spec, line.kind)
        if not spec.rate_printed and line.kind not in {"cgst", "sgst", "igst", "pst", "qst", "hst", "gst"}:
            return name
        return spec.style["rate_fmt"].format(n=name, r=self.rate_text(spec, line.rate))

    def discount_label(self, spec: InvoiceSpec) -> str:
        label = self.vocab(spec, "discount")
        if spec.discount_pct is not None and spec.style["tax_name"] < 0.5:
            return f"{label} ({self.rate_text(spec, spec.discount_pct)}%)"
        return label

    def totals_items(self, spec: InvoiceSpec) -> list[tuple[str, Decimal, str]]:
        """(label, amount, kind) of every line of the totals block, in printed order."""
        rows: list[tuple[str, Decimal, str]] = []
        if spec.subtotal is not None:
            rows.append((self.vocab(spec, "subtotal"), spec.subtotal, "subtotal"))
        if spec.discount is not None:
            rows.append((self.discount_label(spec), spec.discount, "discount"))
        if spec.shipping is not None:
            rows.append((self.vocab(spec, "shipping"), spec.shipping, "shipping"))
        for line in spec.taxes:
            rows.append((self.tax_label(spec, line), line.amount, "tax"))
        if spec.combined_tax_row and spec.taxes and spec.tax_total is not None:
            rate = sum((t.rate for t in spec.taxes), Decimal(0))
            rows.append((f"Total GST @ {self.rate_text(spec, rate)}%", spec.tax_total, "tax_total"))
        if spec.total is not None:
            rows.append((self.vocab(spec, "total"), spec.total, "total"))
            if spec.style["balance_due"] and self.vocab(spec, "total") != "Balance Due" and self.id in {"T1", "T3"}:
                rows.append(("Balance Due", spec.total, "balance"))
        return rows

    def totals_rows(self, spec: InvoiceSpec) -> list[tuple[str, str, str]]:
        """(label, formatted value, kind) of the totals block."""
        return [
            (label, self.money(spec, value, "total" if kind in {"total", "balance"} else "sub"), kind)
            for label, value, kind in self.totals_items(spec)
        ]

    def party_html(self, party: Party, *, name: bool = True, ids: bool = True, contact: bool = True) -> str:
        lines = [f"<b>{esc(party.name)}</b>"] if name and party.name else []
        lines += [esc(line) for line in party.lines]
        if contact and party.contact:
            lines.append(f"Attn: {esc(party.contact)}")
        if party.phone:
            lines.append(f"Tel: {esc(party.phone)}")
        if party.email:
            lines.append(esc(party.email))
        if ids:
            lines += [f"{esc(label)}: {esc(value)}" for label, value in party.ids]
        return "<br>".join(lines)

    def page_label(self, spec: InvoiceSpec, page_no: int, n_pages: int) -> str:
        if n_pages == 1 and not spec.style["single_page_no"]:
            return ""
        return spec.style["page_fmt"].format(k=page_no, n=n_pages)

    # ----- hooks implemented by the concrete templates ---------------------------------------------------------
    def columns(self, spec: InvoiceSpec) -> list[Col]:
        raise NotImplementedError

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        raise NotImplementedError

    def table_html(self, spec: InvoiceSpec, a: int, b: int) -> str:
        rows = spec.rows
        return grid(
            self.columns(spec),
            True,
            [self.cells(spec, it, a + i) for i, it in enumerate(rows[a:b])],
            pad=self.pad,
            border=self.border,
        )

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        raise NotImplementedError

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        raise NotImplementedError

    def draw_continued(self, cv: Canvas, spec: InvoiceSpec) -> None:
        cv.y += cv.html('<div class="cont">Continued on next page</div>', self.x0, self.x1) + 4

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        raise NotImplementedError

    def draw_overlay(self, cv: Canvas, spec: InvoiceSpec, page_no: int) -> None:
        stamp = spec.style["stamp"]
        if stamp:
            cv.stamp(
                stamp,
                self.width * 0.62,
                self.height * 0.42,
                size=54,
                angle=22,
                color=(0.75, 0.1, 0.1) if stamp == "PAID" else (0.3, 0.3, 0.3),
            )

    # ----- page assembly ---------------------------------------------------------------------------------------
    def draw_page(self, cv: Canvas, spec: InvoiceSpec, plan: PagePlan, page_no: int, n_pages: int) -> None:
        cv.y = self.margin_top
        self.draw_header(cv, spec, first=plan.first, page_no=page_no, n_pages=n_pages)
        if plan.stop > plan.start:
            cv.flow(self.table_html(spec, plan.start, plan.stop), *self.table_x, gap=6)
        elif not spec.rows:
            cv.flow(
                '<div class="small">Professional services rendered in accordance with the agreed scope of work.</div>',
                self.x0,
                self.x1,
                gap=14,
            )
        if plan.last:
            self.draw_totals(cv, spec)
        else:
            self.draw_continued(cv, spec)
        self.draw_footer(cv, spec, page_no, n_pages)
        self.draw_overlay(cv, spec, page_no)


# --------------------------------------------------------------------------------------------------------------------
# T1 - US classic
# --------------------------------------------------------------------------------------------------------------------
class T1(Template):
    id = "T1"
    split = "dev"
    width, height = 612.0, 792.0
    margin_left = margin_right = 40.0
    margin_top = 36.0
    margin_bottom = 34.0
    font = "Helvetica"
    size = 9.0
    pad = 5.0
    date_styles = (("us", 70), ("us_mon", 15), ("us_long", 15))
    rate_formats = ("{n} ({r}%)", "{n} ({r}%)", "{n} @ {r}%", "{n} {r}%")

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        if not self.token_needed(spec, role):
            return num
        cur, code_total = spec.currency, spec.style["code_total"]
        if cur == "USD":
            return f"USD {num}" if role == "total" and code_total else f"${num}"
        if role == "total" or not spec.shows("total"):  # CAD: the code is always printed with the total
            return f"${num} CAD"
        return f"${num}"

    def css(self, spec: InvoiceSpec) -> str:
        brand = spec.style["brand"]
        return (
            super().css(spec)
            + f""" .co {{font-size: 16pt; font-weight: bold; color: {brand};}}
 .small {{font-size: 8.2pt; color: #555; line-height: 1.25;}}
 .title {{font-size: 26pt; font-weight: bold; color: {brand}; text-align: right; letter-spacing: 2pt;}}
 table.kv td {{padding: 1.5pt 0;}} td.k {{color: #666; text-align: right; padding-right: 8pt;}}
 td.v {{font-weight: bold; text-align: right;}}
 .lbl {{font-size: 7.5pt; color: #777; letter-spacing: 1.2pt; margin-bottom: 2pt;}}
 .box {{border: 1px solid #b8c0cc; padding: 6pt; background-color: #f6f8fa;}}
 table.items th {{background-color: {brand}; color: #ffffff; font-size: 8.5pt; text-align: left;
   padding: 5pt {self.pad}pt;}}
 table.items td {{padding: 5pt {self.pad}pt; border-bottom: 1px solid #d3d8df;}}
 table.tot td {{padding: 2.5pt 4pt;}}
 tr.total td {{font-weight: bold; font-size: 10.5pt; border-top: 2px solid {brand};}}
 .cont {{text-align: right; font-style: italic; color: #666; font-size: 8pt;}}
 .foot {{font-size: 7.5pt; color: #777;}}"""
        )

    def columns(self, spec: InvoiceSpec) -> list[Col]:
        v = spec.style["v"]
        w = self.x1 - self.x0
        return [
            Col("desc", v["desc"], w - 62 - 84 - 92, "left"),
            Col("qty", v["qty"], 62, "right"),
            Col("unit_price", v["price"], 84, "right"),
            Col("amount", v["amount"], 92, "right"),
        ]

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        return [
            esc(item.description),
            self.qty(spec, item.quantity),
            self.money(spec, item.unit_price),
            self.money(spec, item.amount),
        ]

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        st, sup = spec.style, spec.supplier
        top = cv.y
        if not first:
            name = f"<b>{esc(sup.name)}</b>" if sup.name else ""
            ref = (
                f"Invoice {esc(spec.invoice_number)} (continued)"
                if spec.invoice_number and st["cont_number"]
                else "(continued)"
            )
            h1 = cv.html(f'<div style="font-size:11pt;color:{st["brand"]}">{name}</div>', self.x0, self.x0 + 300, y=top)
            h2 = cv.html(f'<div style="text-align:right;color:#555">{ref}</div>', self.x1 - 220, self.x1, y=top)
            cv.y = top + max(h1, h2) + 10
            return
        lx = self.x0
        if st["logo"]:
            draw_logo(cv, self.x0, top, 38, st["brand"], st["logo"])
            lx = self.x0 + 48
        left = (
            f'<div class="co">{esc(sup.name)}</div>' if sup.name else ""
        ) + f'<div class="small">{self.party_html(sup, name=False, ids=False, contact=False)}</div>'
        if sup.web:
            left += f'<div class="small">{esc(sup.web)}</div>'
        meta = "".join(
            f'<tr><td class="k">{esc(k)}</td><td class="v">{esc(v)}</td></tr>' for k, v in self.meta_rows(spec)
        )
        right = (
            f'<div class="title">{esc(self.title(spec))}</div>'
            f'<table class="kv" style="width:230pt;table-layout:fixed;"><tr><td class="k" style="width:88pt"></td>'
            f'<td class="v" style="width:134pt"></td></tr>{meta}</table>'
        )
        hl = cv.html(left, lx, self.x0 + 300, y=top)
        hr = cv.html(right, self.x1 - 230, self.x1, y=top)
        cv.y = top + max(hl, hr, 40) + 14
        self._draw_parties(cv, spec)

    def _draw_parties(self, cv: Canvas, spec: InvoiceSpec) -> None:
        if spec.customer is None:
            return
        bill = f'<div class="lbl">{esc(self.vocab(spec, "bill_to").upper())}</div>{self.party_html(spec.customer)}'
        boxes = [bill] + (
            [f'<div class="lbl">SHIP TO</div>{self.party_html(spec.ship_to, ids=False)}'] if spec.ship_to else []
        )
        gap, width = 14.0, (self.x1 - self.x0 - 14.0) / 2
        top = cv.y
        heights = [cv.measure(b, 0, width - 12) + 12 for b in boxes]
        tallest = max(heights)
        for i, b in enumerate(boxes):
            x = self.x0 + i * (width + gap)
            cv.rect(x, top, x + width, top + tallest, fill=(0.965, 0.972, 0.98), stroke=(0.72, 0.75, 0.8), width=0.6)
            cv.html(b, x + 6, x + width - 6, y=top + 6)
        cv.y = top + tallest + 14

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        top = cv.y
        rows = self.totals_rows(spec)
        body = "".join(
            f'<tr class="{"total" if kind in {"total", "balance"} else ""}"><td style="width:116pt">{esc(label)}</td>'
            f'<td style="text-align:right;width:92pt">{esc(value)}</td></tr>'
            for label, value, kind in rows
        )
        right = f'<table class="tot" style="width:224pt;table-layout:fixed;">{body}</table>' if rows else ""
        notes = f'<div class="lbl">NOTES</div><div class="small">{esc(spec.extra["note"])}</div>'
        if spec.supplier.bank:
            notes += (
                '<div class="small" style="margin-top:4pt">'
                + " &nbsp;|&nbsp; ".join(f"{esc(k)} {esc(v)}" for k, v in spec.supplier.bank)
                + "</div>"
            )
        hl = cv.html(notes, self.x0, self.x0 + 270, y=top)
        hr = cv.html(right, self.x1 - 224, self.x1, y=top) if right else 0.0
        cv.y = top + max(hl, hr) + 8

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        y = cv.bottom - 24
        cv.line(self.x0, y - 4, self.x1, y - 4, color=(0.75, 0.78, 0.82), width=0.5)
        sup = spec.supplier
        left = "Thank you for your business!" + (f" &nbsp; {esc(sup.name)}" if sup.name else "")
        cv.html(f'<div class="foot">{left}</div>', self.x0, self.x0 + 360, y=y)
        label = self.page_label(spec, page_no, n_pages)
        cv.html(f'<div class="foot" style="text-align:right">{esc(label)}</div>', self.x1 - 140, self.x1, y=y)
        ids = " &nbsp;&middot;&nbsp; ".join(f"{esc(a)} {esc(b)}" for a, b in sup.ids[:1])
        contact = f"{esc(sup.phone)} &nbsp;&middot;&nbsp; {esc(sup.email)} &nbsp;&middot;&nbsp; {ids}"
        cv.html(
            f'<div class="foot">{contact}</div>',
            self.x0,
            self.x1,
            y=y + 10,
        )


# --------------------------------------------------------------------------------------------------------------------
# T2 - EU VAT invoice
# --------------------------------------------------------------------------------------------------------------------
BILINGUAL = {"de": "Rechnung", "fr": "Facture", "es": "Factura", "it": "Fattura", "nl": "Factuur"}


class T2(Template):
    id = "T2"
    split = "dev"
    width, height = 595.0, 842.0
    margin_left, margin_right = 50.0, 46.0
    margin_top = 42.0
    margin_bottom = 38.0
    footer_h = 44.0
    font = "serif"
    size = 9.2
    pad = 3.5
    date_styles = (("eu_dot", 85), ("eu_slash", 15))
    rate_formats = ("{n} {r} %", "{n} {r} %", "{n} {r}%")

    def number_style(self, spec: InvoiceSpec) -> dict[str, Any]:
        return {"group": " " if spec.market == "fr" else ".", "dec": ","}

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        if not self.token_needed(spec, role):
            return num
        return f"EUR {num}" if role == "total" and spec.style["code_total"] else f"{num} €"

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {"bilingual": rng.random() < 0.35, "show_base": rng.random() < 0.3, "sender_line": rng.random() < 0.7}

    def title(self, spec: InvoiceSpec) -> str:
        base = "Credit note" if spec.doc_type is DocumentType.CREDIT_NOTE else "Invoice"
        local = BILINGUAL.get(spec.market)
        if spec.style["bilingual"] and local and spec.doc_type is not DocumentType.CREDIT_NOTE:
            return f"{local} / {base}"
        return base

    def tax_label(self, spec: InvoiceSpec, line: TaxLine) -> str:
        label = super().tax_label(spec, line)
        if spec.style["show_base"] and spec.shows("subtotal"):
            return f"{label} on {self.money(spec, line.base, 'sub')}"
        return label

    def css(self, spec: InvoiceSpec) -> str:
        brand = spec.style["brand"]
        return (
            super().css(spec)
            + f""" .co {{font-size: 16pt; font-weight: bold; color: {brand};}} .small {{font-size: 8pt; color: #555;}}
 .ttl {{font-size: 19pt; font-weight: bold; color: {brand}; margin: 2pt 0 4pt 0;}}
 .sender {{font-size: 6.8pt; color: #666; border-bottom: 1px solid #888; padding-bottom: 1pt; margin-bottom: 4pt;}}
 table.kv td {{padding: 1.5pt 0;}} td.k {{color: #555; padding-right: 6pt;}} td.v {{text-align: right;}}
 table.items th {{background-color: #eceff3; color: #222; font-size: 8.4pt; text-align: left; padding: 4pt {self.pad}pt;
   border-top: 1px solid #555; border-bottom: 1px solid #555;}}
 table.items td {{padding: 4pt {self.pad}pt; border-bottom: 1px solid #d5d9df;}}
 table.tot td {{padding: 2.2pt 4pt;}}
 tr.total td {{font-weight: bold; font-size: 10pt; border-top: 1px solid #222; border-bottom: 3px double #222;}}
 .cont {{text-align: right; font-style: italic; color: #666; font-size: 8pt;}}
 .foot {{font-size: 7.2pt; color: #666;}}"""
        )

    def columns(self, spec: InvoiceSpec) -> list[Col]:
        v = spec.style["v"]
        w = self.x1 - self.x0
        cols = [
            Col("idx", "Pos.", 24, "left"),
            Col("desc", v["desc"], 0, "left"),
            Col("qty", v["qty"], 40, "right"),
            Col("unit", "Unit", 34, "left"),
            Col("unit_price", v["price"], 64, "right"),
        ]
        if spec.taxes:
            cols.append(Col("vat", "VAT %", 40, "right"))
        cols.append(Col("amount", v["amount"], 72, "right"))
        fixed = sum(c.width for c in cols)
        return [Col(c.key, c.header, w - fixed, c.align) if c.key == "desc" else c for c in cols]

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        out = [
            str(index + 1),
            esc(item.description),
            self.qty(spec, item.quantity),
            esc(item.unit),
            self.money(spec, item.unit_price),
        ]
        if spec.taxes:
            rate = spec.tax_rates[item.vat_class] if spec.tax_scheme == "classes" else spec.tax_rates[0]
            out.append(f"{self.rate_text(spec, rate)} %")
        out.append(self.money(spec, item.amount))
        return out

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        st, sup = spec.style, spec.supplier
        top = cv.y
        if not first:
            name = f"<b>{esc(sup.name)}</b>" if sup.name else ""
            ref = (
                f"{esc(self.title(spec))} {esc(spec.invoice_number)} &ndash; continued"
                if spec.invoice_number and st["cont_number"]
                else "continued"
            )
            h1 = cv.html(f'<div style="font-size:11pt;color:{st["brand"]}">{name}</div>', self.x0, self.x0 + 300, y=top)
            h2 = cv.html(f'<div style="text-align:right;color:#555">{ref}</div>', self.x1 - 230, self.x1, y=top)
            cv.y = top + max(h1, h2) + 10
            return
        lx = self.x0
        if st["logo"]:
            draw_logo(cv, self.x0, top, 34, st["brand"], st["logo"])
            lx = self.x0 + 44
        left = f'<div class="co">{esc(sup.name)}</div>' if sup.name else ""
        left += f'<div class="small">{esc(", ".join(sup.lines[:2]))}</div>'
        contact = [f"Tel. {esc(sup.phone)}", esc(sup.email), esc(sup.web)]
        contact += [f"{esc(a)} {esc(b)}" for a, b in sup.ids]
        right = '<div class="small" style="text-align:right">' + "<br>".join(c for c in contact if c.strip()) + "</div>"
        hl = cv.html(left, lx, self.x0 + 300, y=top)
        hr = cv.html(right, self.x1 - 190, self.x1, y=top)
        cv.y = top + max(hl, hr, 34) + 20
        block_top = cv.y
        recipient = ""
        if st["sender_line"] and sup.name:
            recipient += (
                f'<div class="sender">{esc(sup.name)} &middot; {esc(sup.lines[0])} &middot; {esc(sup.lines[1])}</div>'
            )
        if spec.customer is not None:
            caption = f'<div class="small" style="margin-bottom:2pt">{esc(self.vocab(spec, "bill_to"))}</div>'
            recipient += caption + self.party_html(spec.customer, ids=True)
        meta = "".join(
            f'<tr><td class="k">{esc(k)}</td><td class="v">{esc(v)}</td></tr>' for k, v in self.meta_rows(spec)
        )
        meta_html = (
            f'<table class="kv" style="width:200pt;table-layout:fixed;"><tr><td class="k" style="width:96pt"></td>'
            f'<td class="v" style="width:98pt"></td></tr>{meta}</table>'
        )
        h_left = cv.html(recipient, self.x0, self.x0 + 235, y=block_top) if recipient else 0.0
        h_right = cv.html(meta_html, self.x1 - 200, self.x1, y=block_top)
        cv.y = block_top + max(h_left, h_right) + 14
        cv.flow(f'<div class="ttl">{esc(self.title(spec))}</div>', self.x0, self.x1, gap=6)

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        top = cv.y
        rows = self.totals_rows(spec)
        body = "".join(
            f'<tr class="{"total" if kind == "total" else ""}"><td style="width:142pt">{esc(label)}</td>'
            f'<td style="text-align:right;width:84pt">{esc(value)}</td></tr>'
            for label, value, kind in rows
        )
        right = f'<table class="tot" style="width:242pt;table-layout:fixed;">{body}</table>' if rows else ""
        left = f'<div class="small">{esc(spec.extra["note"])}</div>'
        if spec.supplier.bank:
            left += (
                '<div class="small" style="margin-top:5pt"><b>Bank details</b><br>'
                + "<br>".join(f"{esc(k)}: {esc(v)}" for k, v in spec.supplier.bank)
                + "</div>"
            )
        hl = cv.html(left, self.x0, self.x0 + 245, y=top)
        hr = cv.html(right, self.x1 - 242, self.x1, y=top) if right else 0.0
        cv.y = top + max(hl, hr) + 8

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        y = cv.bottom - 30
        cv.line(self.x0, y - 4, self.x1, y - 4, color=(0.6, 0.62, 0.66), width=0.5)
        sup = spec.supplier
        legal = " &middot; ".join(
            x
            for x in (
                esc(sup.name),
                esc(", ".join(sup.lines[:2])),
                esc(sup.phone),
                " ".join(f"{esc(a)} {esc(b)}" for a, b in sup.ids[:1]),
            )
            if x
        )
        cv.html(f'<div class="foot">{legal}</div>', self.x0, self.x1 - 80, y=y)
        cv.html(
            f'<div class="foot" style="text-align:right">{esc(self.page_label(spec, page_no, n_pages))}</div>',
            self.x1 - 80,
            self.x1,
            y=y,
        )


# --------------------------------------------------------------------------------------------------------------------
# T3 - UK traditional
# --------------------------------------------------------------------------------------------------------------------
class T3(Template):
    id = "T3"
    split = "dev"
    width, height = 595.0, 842.0
    margin_left = margin_right = 50.0
    margin_top = 44.0
    margin_bottom = 38.0
    footer_h = 40.0
    font = "Times"
    size = 10.0
    pad = 4.0
    date_styles = (("uk_long", 80), ("uk_short", 10), ("eu_slash", 10))
    rate_formats = ("{n} @ {r}%", "{n} @ {r}%", "{n} ({r}%)", "{n} {r}%")

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        if not self.token_needed(spec, role):
            return num
        sym = "£" if spec.currency == "GBP" else "€"
        return f"{spec.currency} {num}" if role == "total" and spec.style["code_total"] else f"{sym}{num}"

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {"qty_places": rng.choice((0, 0, 2)), "letterhead": rng.choice(("centre", "centre", "left"))}

    def qty(self, spec: InvoiceSpec, value: Decimal) -> str:
        return fmt_qty(value, places=spec.style["qty_places"])

    def css(self, spec: InvoiceSpec) -> str:
        brand = spec.style["brand"]
        return (
            super().css(spec)
            + f""" .co {{font-size: 21pt; font-weight: bold; color: {brand}; text-align: center;}}
 .lh {{font-size: 8.6pt; color: #444; text-align: center; line-height: 1.3;}}
 .small {{font-size: 8.6pt; color: #444;}}
 .ttl {{font-size: 16pt; font-weight: bold; letter-spacing: 3pt; margin: 0 0 6pt 0;}}
 table.kv td {{padding: 1.6pt 0;}} td.k {{padding-right: 8pt; font-weight: bold;}} td.v {{text-align: left;}}
 table.items th {{font-size: 9.4pt; text-align: left; padding: 4pt {self.pad}pt; border-top: 1.5px solid #222;
   border-bottom: 1px solid #222;}}
 table.items td {{padding: 4pt {self.pad}pt; border-bottom: 1px dotted #999;}}
 table.tot td {{padding: 2.4pt 4pt;}}
 tr.total td {{font-weight: bold; font-size: 11pt; border-top: 1.5px solid #222;}}
 .cont {{text-align: right; font-style: italic; font-size: 8.6pt;}}
 .foot {{font-size: 7.8pt; color: #555; text-align: center;}}"""
        )

    def columns(self, spec: InvoiceSpec) -> list[Col]:
        v = spec.style["v"]
        w = self.x1 - self.x0
        unit = (
            f" ({'£' if spec.currency == 'GBP' else '€'})" if spec.currency_printed and spec.style["sym_rows"] else ""
        )
        return [
            Col("desc", v["desc"], w - 70 - 80 - 86, "left"),
            Col("qty", v["qty"], 70, "right"),
            Col("unit_price", v["price"] + unit, 80, "right"),
            Col("amount", v["amount"] + unit, 86, "right"),
        ]

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        return [
            esc(item.description),
            self.qty(spec, item.quantity),
            self.money(spec, item.unit_price),
            self.money(spec, item.amount),
        ]

    def money(self, spec: InvoiceSpec, value: Decimal, role: str = "row") -> str:
        if role == "row" and spec.currency_printed and spec.style["sym_rows"]:
            return fmt_number(value)  # the symbol is in the column header instead
        return super().money(spec, value, role)

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        st, sup = spec.style, spec.supplier
        top = cv.y
        if not first:
            name = f"<b>{esc(sup.name)}</b>" if sup.name else ""
            ref = (
                f"Invoice {esc(spec.invoice_number)} - continued"
                if spec.invoice_number and st["cont_number"]
                else "continued"
            )
            h1 = cv.html(f"<div>{name}</div>", self.x0, self.x0 + 300, y=top)
            h2 = cv.html(f'<div style="text-align:right;font-style:italic">{ref}</div>', self.x1 - 230, self.x1, y=top)
            cv.y = top + max(h1, h2) + 8
            cv.line(self.x0, cv.y - 2, self.x1, cv.y - 2, width=0.6)
            cv.y += 8
            return
        centre = st["letterhead"] == "centre"
        if st["logo"]:
            draw_logo(cv, (self.width - 30) / 2 if centre else self.x0, top, 30, st["brand"], st["logo"])
            top += 36
        lines = ", ".join(sup.lines)
        ids = " &nbsp;|&nbsp; ".join(f"{esc(a)}: {esc(b)}" for a, b in sup.ids)
        head = (
            f'<div class="co" style="text-align:{"center" if centre else "left"}">{esc(sup.name)}</div>'
            if sup.name
            else ""
        )
        align = "center" if centre else "left"
        web = f" &nbsp;|&nbsp; {esc(sup.web)}" if sup.web else ""
        head += (
            f'<div class="lh" style="text-align:{align}">{esc(lines)}<br>Tel: {esc(sup.phone)} &nbsp;|&nbsp; '
            f"{esc(sup.email)}{web}<br>{ids}</div>"
        )
        cv.y = top
        cv.flow(head, self.x0, self.x1, gap=8)
        cv.line(self.x0, cv.y, self.x1, cv.y, width=1.2)
        cv.y += 12
        block_top = cv.y
        left = f'<div class="ttl">{esc(self.title(spec))}</div>'
        if spec.customer is not None:
            caption = f'<div class="small"><b>{esc(self.vocab(spec, "bill_to"))}:</b></div>'
            left += caption + self.party_html(spec.customer, ids=True)
        meta = "".join(
            f'<tr><td class="k">{esc(k)}</td><td class="v">{esc(v)}</td></tr>' for k, v in self.meta_rows(spec)
        )
        right = (
            f'<table class="kv" style="width:215pt;table-layout:fixed;"><tr><td class="k" style="width:110pt"></td>'
            f'<td class="v" style="width:97pt"></td></tr>{meta}</table>'
        )
        hl = cv.html(left, self.x0, self.x0 + 250, y=block_top)
        hr = cv.html(right, self.x1 - 215, self.x1, y=block_top + 4)
        cv.y = block_top + max(hl, hr) + 14

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        top = cv.y
        rows = self.totals_rows(spec)
        body = "".join(
            f'<tr class="{"total" if kind in {"total", "balance"} else ""}"><td style="width:120pt">{esc(label)}</td>'
            f'<td style="text-align:right;width:92pt">{esc(value)}</td></tr>'
            for label, value, kind in rows
        )
        right = f'<table class="tot" style="width:228pt;table-layout:fixed;">{body}</table>' if rows else ""
        left = f'<div class="small"><i>{esc(spec.extra["note"])}</i></div>'
        hl = cv.html(left, self.x0, self.x0 + 240, y=top)
        hr = cv.html(right, self.x1 - 228, self.x1, y=top) if right else 0.0
        cv.y = top + max(hl, hr) + 10
        if spec.supplier.bank:
            bank = "Payment by bank transfer: " + " &nbsp;|&nbsp; ".join(
                f"{esc(k)} {esc(v)}" for k, v in spec.supplier.bank
            )
            cv.flow(f'<div class="small" style="text-align:center">{bank}</div>', self.x0, self.x1, gap=4)

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        y = cv.bottom - 26
        cv.line(self.x0, y - 4, self.x1, y - 4, width=0.5)
        sup = spec.supplier
        reg = f"{esc(sup.name)} &nbsp;|&nbsp; " if sup.name else ""
        cv.html(
            f'<div class="foot">{reg}Registered office: {esc(", ".join(sup.lines[:2]))}</div>', self.x0, self.x1, y=y
        )
        label = self.page_label(spec, page_no, n_pages)
        if label:
            cv.html(f'<div class="foot">{esc(label)}</div>', self.x0, self.x1, y=y + 11)


# --------------------------------------------------------------------------------------------------------------------
# T4 - Indian GST tax invoice (grid)
# --------------------------------------------------------------------------------------------------------------------
class T4(Template):
    id = "T4"
    split = "heldout"
    width, height = 595.0, 842.0
    margin_left = margin_right = 34.0
    margin_top = 32.0
    margin_bottom = 34.0
    footer_h = 40.0
    font = "Helvetica"
    size = 8.6
    pad = 4.0
    border = 0.75
    date_styles = (("in_slash", 65), ("in_dash", 10), ("in_mon", 25))
    rate_formats = ("{n} @ {r}%", "{n} @ {r}%", "{n} ({r}%)", "{n} {r}%")

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {
            "inr": rng.choices(("₹", "Rs.", "INR"), weights=(60, 30, 10))[0],
            "qty_places": rng.choice((0, 2)),
            "copy": rng.choice(("Original for Recipient", "Duplicate for Transporter", "Triplicate for Supplier")),
            "title": rng.choice(("TAX INVOICE", "TAX INVOICE", "INVOICE", "GST INVOICE")),
        }

    def number_style(self, spec: InvoiceSpec) -> dict[str, Any]:
        return {"group": ",", "dec": ".", "indian": True}

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        return f"{spec.style['inr']} {num}" if self.token_needed(spec, role) else num

    def qty(self, spec: InvoiceSpec, value: Decimal) -> str:
        return fmt_qty(value, places=spec.style["qty_places"])

    def title(self, spec: InvoiceSpec) -> str:
        return spec.style["title"]

    def css(self, spec: InvoiceSpec) -> str:
        brand = spec.style["brand"]
        return (
            super().css(spec)
            + f""" .co {{font-size: 14pt; font-weight: bold; color: {brand};}} .small {{font-size: 8pt; color: #444;}}
 .ttl {{text-align: center; font-weight: bold; font-size: 13pt; letter-spacing: 2pt;}}
 .lbl {{font-size: 7.4pt; color: #666; letter-spacing: 0.8pt;}}
 table.kv td {{padding: 1.6pt 0;}} td.k {{color: #444; padding-right: 6pt;}} td.v {{font-weight: bold;}}
 table.items th {{background-color: #eeeeee; font-size: 8.2pt; text-align: left; padding: 3.5pt {self.pad}pt;
   border: 1px solid #555;}}
 table.items td {{padding: 3.5pt {self.pad}pt; border: 1px solid #777;}}
 table.tot td {{padding: 3pt 4pt; border: 1px solid #777;}}
 tr.total td {{font-weight: bold; font-size: 9.6pt; background-color: #eeeeee;}}
 .cont {{text-align: right; font-style: italic; color: #555; font-size: 8pt;}}
 .foot {{font-size: 7.6pt; color: #555;}}"""
        )

    def columns(self, spec: InvoiceSpec) -> list[Col]:
        v = spec.style["v"]
        w = self.x1 - self.x0
        fixed = 24 + 56 + 46 + 68 + 82
        return [
            Col("idx", "Sl.", 24, "center"),
            Col("desc", v["desc"], w - fixed, "left"),
            Col("hsn", "HSN/SAC", 56, "center"),
            Col("qty", v["qty"], 46, "right"),
            Col("unit_price", v["price"], 68, "right"),
            Col("amount", v["amount"], 82, "right"),
        ]

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        return [
            str(index + 1),
            esc(item.description),
            esc(item.hsn or ""),
            self.qty(spec, item.quantity),
            self.money(spec, item.unit_price),
            self.money(spec, item.amount),
        ]

    def meta_items(self, spec: InvoiceSpec) -> list[tuple[str, str | date]]:
        rows = super().meta_items(spec)
        if spec.customer is not None:
            rows.append((self.vocab(spec, "pos"), spec.extra["place_of_supply"]))
        return rows

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        st, sup = spec.style, spec.supplier
        top = cv.y
        if not first:
            name = f"<b>{esc(sup.name)}</b>" if sup.name else ""
            ref = (
                f"{esc(self.title(spec).title())} {esc(spec.invoice_number)} (continued)"
                if spec.invoice_number and st["cont_number"]
                else "(continued)"
            )
            h1 = cv.html(f"<div>{name}</div>", self.x0, self.x0 + 300, y=top)
            h2 = cv.html(f'<div style="text-align:right;color:#555">{ref}</div>', self.x1 - 230, self.x1, y=top)
            cv.y = top + max(h1, h2) + 8
            return
        cv.rect(self.x0, top, self.x1, top + 22, stroke=(0.25, 0.25, 0.25), width=0.8)
        cv.html(f'<div class="ttl">{esc(self.title(spec))}</div>', self.x0, self.x1, y=top + 3)
        cv.html(
            f'<div class="small" style="text-align:right">({esc(st["copy"])})</div>',
            self.x1 - 150,
            self.x1 - 4,
            y=top + 7,
        )
        top += 22
        lx = self.x0 + 8
        if st["logo"]:
            draw_logo(cv, self.x0 + 8, top + 8, 34, st["brand"], st["logo"])
            lx = self.x0 + 52
        company = f'<div class="co">{esc(sup.name)}</div>' if sup.name else ""
        company += f'<div class="small">{esc(", ".join(sup.lines))}<br>' + " &nbsp;|&nbsp; ".join(
            f"{esc(a)}: {esc(b)}" for a, b in sup.ids
        )
        company += f"<br>Tel: {esc(sup.phone)} &nbsp;|&nbsp; {esc(sup.email)}</div>"
        hc = max(cv.html(company, lx, self.x1 - 8, y=top + 6) + 12, 52)
        cv.rect(self.x0, top, self.x1, top + hc, stroke=(0.25, 0.25, 0.25), width=0.8)
        top += hc
        left = ""
        if spec.customer is not None:
            left = f'<div class="lbl">{esc(self.vocab(spec, "bill_to").upper())}</div>{self.party_html(spec.customer)}'
        meta = "".join(
            f'<tr><td class="k">{esc(k)}</td><td class="v">{esc(v)}</td></tr>' for k, v in self.meta_rows(spec)
        )
        right = (
            f'<table class="kv" style="width:236pt;table-layout:fixed;"><tr><td class="k" style="width:104pt"></td>'
            f'<td class="v" style="width:126pt"></td></tr>{meta}</table>'
        )
        split_x = self.x0 + 290
        hl = cv.measure(left, 0, 276) if left else 0.0
        hr = cv.measure(right, 0, 244)
        hi = max(hl, hr) + 12
        cv.rect(self.x0, top, self.x1, top + hi, stroke=(0.25, 0.25, 0.25), width=0.8)
        cv.line(split_x, top, split_x, top + hi, color=(0.25, 0.25, 0.25), width=0.8)
        if left:
            cv.html(left, self.x0 + 6, split_x - 6, y=top + 6)
        cv.html(right, split_x + 6, self.x1 - 6, y=top + 6)
        cv.y = top + hi + 8

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        top = cv.y
        rows = self.totals_rows(spec)
        body = "".join(
            f'<tr class="{"total" if kind == "total" else ""}"><td style="width:141pt">{esc(label)}</td>'
            f'<td style="text-align:right;width:92pt">{esc(value)}</td></tr>'
            for label, value, kind in rows
        )
        right = f'<table class="tot" style="width:250pt;table-layout:fixed;">{body}</table>' if rows else ""
        left = ""
        if spec.total is not None and spec.currency_printed:
            left += f'<div class="small"><b>Amount in words:</b><br>{esc(number_in_words_inr(spec.total))}</div>'
        if spec.supplier.bank:
            left += (
                '<div class="small" style="margin-top:6pt"><b>Bank details</b><br>'
                + "<br>".join(f"{esc(k)}: {esc(v)}" for k, v in spec.supplier.bank)
                + "</div>"
            )
        left += f'<div class="small" style="margin-top:6pt">{esc(spec.extra["note"])}</div>'
        hl = cv.html(left, self.x0, self.x0 + 262, y=top)
        hr = cv.html(right, self.x1 - 250, self.x1, y=top) if right else 0.0
        cv.y = top + max(hl, hr) + 8
        party = f"For {esc(spec.supplier.name)}" if spec.supplier.name else ""
        sign = f'<div class="small" style="text-align:right">{party}<br><br><br>Authorised Signatory</div>'
        cv.flow(sign, self.x1 - 200, self.x1, gap=4)

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        y = cv.bottom - 20
        cv.line(self.x0, y - 4, self.x1, y - 4, color=(0.5, 0.5, 0.5), width=0.5)
        cv.html(
            '<div class="foot" style="text-align:center">This is a Computer Generated Invoice</div>',
            self.x0 + 100,
            self.x1 - 100,
            y=y,
        )
        cv.html(
            f'<div class="foot" style="text-align:right">{esc(self.page_label(spec, page_no, n_pages))}</div>',
            self.x1 - 100,
            self.x1,
            y=y,
        )


# --------------------------------------------------------------------------------------------------------------------
# T5 - 80 mm thermal receipt (CAD)
# --------------------------------------------------------------------------------------------------------------------
class T5(Template):
    id = "T5"
    split = "heldout"
    width, height = 226.0, 600.0
    margin_left = margin_right = 12.0
    margin_top = 14.0
    margin_bottom = 14.0
    footer_h = 0.0
    single_page = True
    font = "Courier"
    size = 7.7
    pad = 1.5
    dpi_factor = 1.5
    date_styles = (("iso", 50), ("us_mon", 30), ("au_short", 20))
    rate_formats = ("{n} {r}%", "{n} {r}%", "{n} ({r}%)")
    BARCODE_BLOCK = 52.0

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {
            "title": rng.choice(("SALES RECEIPT", "CUSTOMER RECEIPT", "RECEIPT")),
            "pay": rng.choice(("VISA", "MASTERCARD", "DEBIT", "INTERAC", "CASH")),
            "stamp": None,
        }

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        if not self.token_needed(spec, role):
            return num
        return f"${num} CAD" if role == "total" else f"${num}"

    def title(self, spec: InvoiceSpec) -> str:
        return spec.style["title"]

    def tax_name(self, spec: InvoiceSpec, kind: str) -> str:
        words = TAX_WORDS[kind]
        return words[0] if kind != "qst" or spec.style["tax_name"] < 0.5 else words[1]

    def meta_items(self, spec: InvoiceSpec) -> list[tuple[str, str | date]]:
        rows: list[tuple[str, str | date]] = []
        if spec.invoice_number is not None:
            rows.append((self.vocab(spec, "inv_no"), spec.invoice_number))
        if spec.invoice_date is not None:
            rows.append((self.vocab(spec, "inv_date"), spec.invoice_date))
        return rows

    def meta_rows(self, spec: InvoiceSpec) -> list[tuple[str, str]]:
        return [
            (k, f"{self.date(spec, v)} {spec.extra['clock']}" if isinstance(v, date) else v)
            for k, v in self.meta_items(spec)
        ]

    def sheet_columns(self, spec: InvoiceSpec) -> list[Col]:
        return [
            Col("desc", "Description", 0),
            Col("qty", "Qty", 0),
            Col("unit_price", "Price", 0),
            Col("amount", "Amount", 0),
        ]

    def css(self, spec: InvoiceSpec) -> str:
        return (
            super().css(spec)
            + """ .store {text-align: center; font-weight: bold; font-size: 11pt; margin-bottom: 2pt;}
 .c {text-align: center;} .b {font-weight: bold;} .dash {border-top: 1px dashed #000; margin: 4pt 0;}
 table.items td {padding: 1pt 1.5pt;} table.tot td {padding: 1pt 1.5pt;}
 tr.total td {font-weight: bold; font-size: 10pt;}
 .sub {font-size: 7pt; color: #333;}"""
        )

    def receipt_html(self, spec: InvoiceSpec) -> str:
        sup, st = spec.supplier, spec.style
        parts: list[str] = []
        if sup.name:
            parts.append(f'<div class="store">{esc(sup.name)}</div>')
        info = [esc(line) for line in sup.lines] + [esc(sup.phone)]
        info += [f"{esc(a)}# {esc(b)}" for a, b in sup.ids] + [f"STORE {spec.extra['store_no']}"]
        parts.append(f'<div class="c">{"<br>".join(info)}</div><div class="dash"></div>')
        parts.append(f'<div class="c b">{esc(self.title(spec))}</div>')
        width = self.x1 - self.x0
        meta = self.meta_rows(spec)
        left_w, right_w = width * 0.42 - 3, width * 0.58 - 3

        def meta_row(left: str, right: str) -> str:
            return (
                f'<tr><td style="width:{left_w:g}pt">{left}</td>'
                f'<td style="width:{right_w:g}pt;text-align:right">{right}</td></tr>'
            )

        meta_html = "".join(meta_row(esc(k), esc(v)) for k, v in meta)
        meta_html += meta_row(f"Cashier: {esc(spec.extra['cashier'])}", f"Reg: {spec.extra['register']}")
        parts.append(f'<table class="items" style="width:{width:g}pt;table-layout:fixed;">{meta_html}</table>')
        if spec.customer is not None:
            parts.append(f"<div>{esc(self.vocab(spec, 'bill_to'))} {esc(spec.customer.name)}</div>")
        parts.append('<div class="dash"></div>')
        cols = [Col("qty", "", 24, "left"), Col("desc", "", width - 24 - 56, "left"), Col("amount", "", 56, "right")]
        rows: list[list[str]] = []
        for it in spec.rows:
            rows.append([self.qty(spec, it.quantity), esc(it.description), self.money(spec, it.amount)])
            if it.price_printed:
                rows.append(
                    [
                        "",
                        f'<span class="sub">{self.qty(spec, it.quantity)} @ {self.money(spec, it.unit_price)}</span>',
                        "",
                    ]
                )
        parts.append(grid(cols, False, rows, pad=self.pad))
        parts.append('<div class="dash"></div>')
        tot = "".join(
            f'<tr class="{"total" if kind == "total" else ""}"><td style="width:{width - 78 - 3:g}pt">{esc(label)}</td>'
            f'<td style="width:{78 - 3:g}pt;text-align:right">{esc(value)}</td></tr>'
            for label, value, kind in self.totals_rows(spec)
        )
        parts.append(f'<table class="tot" style="width:{width:g}pt;table-layout:fixed;">{tot}</table>')
        parts.append('<div class="dash"></div>')
        pay = st["pay"]
        if spec.total is not None and pay != "CASH":
            parts.append(
                f"<div>{pay} ************{spec.extra['card_last4']}<br>AUTH: {spec.extra['auth']} &nbsp; APPROVED</div>"
            )
        elif pay == "CASH":
            parts.append("<div>PAID BY: CASH</div>")
        parts.append(
            '<div class="dash"></div><div class="c">THANK YOU FOR SHOPPING WITH US<br>'
            "RETURNS WITHIN 30 DAYS WITH RECEIPT</div>"
        )
        return "".join(parts)

    def page_height(self, spec: InvoiceSpec) -> float:
        from eval.synth.render import Canvas as _Canvas  # local import: render imports this module

        cv = _Canvas(None, self.width, 3000, self.css(spec), 3000)
        body = cv.measure(self.receipt_html(spec), self.x0, self.x1)
        return max(self.margin_top + body + 6 + self.BARCODE_BLOCK + self.margin_bottom, 260.0)

    def draw_page(self, cv: Canvas, spec: InvoiceSpec, plan: PagePlan, page_no: int, n_pages: int) -> None:
        cv.y = self.margin_top
        cv.flow(self.receipt_html(spec), self.x0, self.x1, gap=6)
        barcode(cv, self.x0 + 18, cv.y, self.x1 - self.x0 - 36, 32, spec.extra["barcode"])
        cv.html(f'<div class="c">{spec.extra["barcode"]}</div>', self.x0, self.x1, y=cv.y + 34)


# --------------------------------------------------------------------------------------------------------------------
# T6 - modern two-column (AUD), values above their labels
# --------------------------------------------------------------------------------------------------------------------
class T6(Template):
    id = "T6"
    split = "heldout"
    width, height = 595.0, 842.0
    margin_left = margin_right = 48.0
    margin_top = 44.0
    margin_bottom = 36.0
    footer_h = 38.0
    font = "Helvetica"
    size = 9.0
    pad = 5.0
    date_styles = (("uk_short", 45), ("au_long", 25), ("au_short", 15), ("iso", 15))
    rate_formats = ("{n} ({r}%)", "{n} {r}%", "{n} ({r}%)")

    def extra_style(self, rng: random.Random) -> dict[str, Any]:
        return {"cur_fmt": rng.choice(("prefix", "suffix"))}

    def decorate(self, spec: InvoiceSpec, num: str, role: str) -> str:
        if not self.token_needed(spec, role):
            return num
        prefix = {"AUD": "A$", "USD": "US$"}.get(spec.currency, "$")
        if spec.style["cur_fmt"] == "prefix":
            return f"{prefix}{num}"
        return f"${num} {spec.currency}" if role == "total" else f"${num}"

    def title(self, spec: InvoiceSpec) -> str:
        return "Credit note" if spec.doc_type is DocumentType.CREDIT_NOTE else "Invoice"

    def css(self, spec: InvoiceSpec) -> str:
        brand = spec.style["brand"]
        return (
            super().css(spec)
            + f""" .co {{font-size: 20pt; font-weight: bold; color: {brand};}}
 .small {{font-size: 8.2pt; color: #666; line-height: 1.3;}}
 .tag {{font-size: 8pt; letter-spacing: 3pt; color: #999;}}
 .big {{font-size: 14pt; font-weight: bold; color: #111;}}
 .cap {{font-size: 7pt; letter-spacing: 1pt; color: #888; margin-top: 2pt;}}
 .lbl {{font-size: 7.4pt; letter-spacing: 1.2pt; color: {brand}; margin-bottom: 3pt;}}
 table.items th {{font-size: 7.4pt; letter-spacing: 0.8pt; color: #777; text-align: left; padding: 6pt {self.pad}pt;
   border-bottom: 1.5px solid #333;}}
 table.items td {{padding: 7pt {self.pad}pt; border-bottom: 1px solid #e3e3e3;}}
 table.tot td {{padding: 3pt 4pt; color: #555;}}
 tr.total td {{font-weight: bold; font-size: 11.5pt; color: {brand}; border-top: 1.5px solid #333; padding-top: 6pt;}}
 .cont {{text-align: right; color: #888; font-size: 8pt;}} .foot {{font-size: 7.4pt; color: #888;}}"""
        )

    def columns(self, spec: InvoiceSpec) -> list[Col]:
        v = spec.style["v"]
        w = self.x1 - self.x0
        return [
            Col("desc", v["desc"].upper(), w - 64 - 90 - 96, "left"),
            Col("qty", v["qty"].upper(), 64, "right"),
            Col("unit_price", v["price"].upper(), 90, "right"),
            Col("amount", v["amount"].upper(), 96, "right"),
        ]

    def cells(self, spec: InvoiceSpec, item: Item, index: int) -> list[str]:
        return [
            esc(item.description),
            self.qty(spec, item.quantity),
            self.money(spec, item.unit_price),
            self.money(spec, item.amount),
        ]

    def meta_blocks(self, spec: InvoiceSpec) -> list[tuple[str, str]]:
        """(value, caption) pairs: the value is printed *above* its caption."""
        blocks: list[tuple[str, str]] = []
        if spec.invoice_number is not None:
            blocks.append((spec.invoice_number, self.vocab(spec, "inv_no")))
        issued = [(self.date(spec, spec.invoice_date), self.vocab(spec, "inv_date"))] if spec.invoice_date else []
        due = [(self.date(spec, spec.due_date), self.vocab(spec, "due"))] if spec.due_date else []
        blocks += (due + issued) if spec.style["due_first"] else (issued + due)
        if spec.total is not None:
            caption = self.vocab(spec, "total") + (f" ({spec.currency})" if spec.currency_printed else "")
            blocks.append((self.money(spec, spec.total, "total"), caption))
        return blocks

    def draw_header(self, cv: Canvas, spec: InvoiceSpec, *, first: bool, page_no: int, n_pages: int) -> None:
        st, sup = spec.style, spec.supplier
        top = cv.y
        if not first:
            name = f'<b style="color:{st["brand"]}">{esc(sup.name)}</b>' if sup.name else ""
            ref = (
                f"{esc(self.title(spec))} {esc(spec.invoice_number)} &middot; continued"
                if spec.invoice_number and st["cont_number"]
                else "continued"
            )
            h1 = cv.html(f"<div>{name}</div>", self.x0, self.x0 + 300, y=top)
            h2 = cv.html(f'<div style="text-align:right;color:#888">{ref}</div>', self.x1 - 230, self.x1, y=top)
            cv.y = top + max(h1, h2) + 12
            return
        lx = self.x0
        if st["logo"]:
            draw_logo(cv, self.x0, top, 30, st["brand"], st["logo"])
            lx = self.x0 + 40
        left = f'<div class="co">{esc(sup.name)}</div>' if sup.name else ""
        left += f'<div class="tag">{esc(self.title(spec).upper())}</div>'
        contact = [esc(sup.phone), esc(sup.email), esc(sup.web)]
        right = '<div class="small" style="text-align:right">' + "<br>".join(c for c in contact if c.strip()) + "</div>"
        hl = cv.html(left, lx, self.x0 + 300, y=top)
        hr = cv.html(right, self.x1 - 180, self.x1, y=top)
        cv.y = top + max(hl, hr, 32) + 22
        blocks = self.meta_blocks(spec)
        if blocks:
            cell = (self.x1 - self.x0) / len(blocks)
            row_h = 0.0
            for i, (value, caption) in enumerate(blocks):
                x = self.x0 + i * cell
                row_h = max(
                    row_h,
                    cv.html(
                        f'<div class="big">{esc(value)}</div><div class="cap">{esc(caption.upper())}</div>',
                        x,
                        x + cell - 8,
                        y=cv.y,
                    ),
                )
            cv.y += row_h + 8
        extras = [
            (self.vocab(spec, k), spec.extra[f]) for k, f in (("po", "po"), ("cust_no", "customer_no")) if spec.extra[f]
        ]
        if spec.service_date:
            extras.append((self.vocab(spec, "svc"), self.date(spec, spec.service_date)))
        if extras:
            cv.flow(
                '<div class="small">'
                + " &nbsp;&middot;&nbsp; ".join(f"{esc(k)}: {esc(v)}" for k, v in extras)
                + "</div>",
                self.x0,
                self.x1,
            )
        cv.y += 8
        cv.line(self.x0, cv.y, self.x1, cv.y, color=(0.85, 0.85, 0.85), width=0.8)
        cv.y += 14
        col = (self.x1 - self.x0 - 24) / 2
        from_html = f'<div class="lbl">FROM</div>{self.party_html(sup, contact=False)}'
        hl = cv.html(from_html, self.x0, self.x0 + col, y=cv.y)
        hr = 0.0
        if spec.customer is not None:
            bill = f'<div class="lbl">{esc(self.vocab(spec, "bill_to").upper())}</div>{self.party_html(spec.customer)}'
            hr = cv.html(bill, self.x0 + col + 24, self.x1, y=cv.y)
        cv.y += max(hl, hr) + 18

    def draw_totals(self, cv: Canvas, spec: InvoiceSpec) -> None:
        top = cv.y
        rows = self.totals_rows(spec)
        body = "".join(
            f'<tr class="{"total" if kind == "total" else ""}"><td style="width:96pt">{esc(label)}</td>'
            f'<td style="text-align:right;width:110pt">{esc(value)}</td></tr>'
            for label, value, kind in rows
        )
        right = f'<table class="tot" style="width:222pt;table-layout:fixed;">{body}</table>' if rows else ""
        left = f'<div class="lbl">NOTES</div><div class="small">{esc(spec.extra["note"])}</div>'
        if spec.supplier.bank:
            left += (
                '<div class="lbl" style="margin-top:8pt">PAYMENT DETAILS</div><div class="small">'
                + "<br>".join(f"{esc(k)} {esc(v)}" for k, v in spec.supplier.bank)
                + (f"<br>Reference {esc(spec.invoice_number)}" if spec.invoice_number else "")
                + "</div>"
            )
        hl = cv.html(left, self.x0, self.x0 + 260, y=top)
        hr = cv.html(right, self.x1 - 222, self.x1, y=top) if right else 0.0
        cv.y = top + max(hl, hr) + 8

    def draw_footer(self, cv: Canvas, spec: InvoiceSpec, page_no: int, n_pages: int) -> None:
        y = cv.bottom - 18
        cv.line(self.x0, y - 6, self.x1, y - 6, color=(0.85, 0.85, 0.85), width=0.8)
        sup = spec.supplier
        left = f"{esc(sup.name)} &middot; " if sup.name else ""
        left += " &middot; ".join(f"{esc(a)} {esc(b)}" for a, b in sup.ids[:1])
        cv.html(f'<div class="foot">{left}</div>', self.x0, self.x1 - 100, y=y)
        cv.html(
            f'<div class="foot" style="text-align:right">{esc(self.page_label(spec, page_no, n_pages))}</div>',
            self.x1 - 100,
            self.x1,
            y=y,
        )


TEMPLATES: dict[str, Template] = {t.id: t for t in (T1(), T2(), T3(), T4(), T5(), T6())}
