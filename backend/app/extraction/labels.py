"""Label vocabularies: how invoices name each field, and how spreadsheets name their columns.

``find_labels`` locates field labels inside a line of text (longest match wins where labels overlap, so
"Sub Total" is a subtotal label, not a total label). Patterns exclude identifier and title look-alikes such as
"Tax Invoice", "VAT No", "GSTIN", "Customer ID" and "Net 30".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

OTHER_DATE = "other_date"  # a date that is not the invoice date (delivery, order, supply, service, period ...)

_ID_WORDS = r"(?:id\b|no\b\.?|nr\b|number|num\b|#|reg\b|registration|code\b|ref\b|account|in\b)"
# Tax qualifiers of a net / gross amount: "Total excl. VAT", "Amount (incl. GST)". The bracketed form only counts
# when it closes right after the tax word: "Total (incl. VAT 20.00)" states the tax amount and keeps its tax label.
_EXCL = r"(?:excl\.?|excluding|exclusive\s*of|before|ex\.?|w/o|without)"
_INCL = r"(?:incl\.?|including|inclusive\s*of|inc\.?|with)"

_RAW_LABELS: dict[str, list[str]] = {
    "invoice_number": [
        r"\b(?:tax\s+)?invoice\s*(?:#|(?:no|number|num|nr|id)\b\.?)",
        r"\binv\.?\s*(?:#|(?:no|number|num)\b\.?)",
        r"\b(?:bill|receipt|document|doc|credit\s*note)\s*(?:#|(?:no|number|num|id)\b\.?)",
        r"\binvoice\s*:",
    ],
    "invoice_date": [
        r"\binvoice\s*date\b",
        r"\bdate\s*of\s*(?:issue|invoice)\b",
        r"\b(?:issue|billing|bill|document|receipt)\s*date\b",
        r"\bissued(?:\s*on)?\b(?!\s*by)",
        r"\bdated\b",
        r"(?<!due\s)(?<!delivery\s)(?<!order\s)(?<!ship\s)(?<!shipping\s)(?<!payment\s)(?<!expiry\s)"
        r"(?<!start\s)(?<!end\s)(?<!dispatch\s)\bdate\b"
        r"(?!\s*(?:due\b|of\s*(?:delivery|order|dispatch|shipment|supply|service)))",
    ],
    # Dates that are printed next to the invoice date but are not it: when goods were supplied or delivered (the
    # VAT tax point / "date of supply"), ordered, serviced, or a period. Never an invoice date.
    OTHER_DATE: [
        r"\b(?:p\.?\s?o\.?|purchase\s*order|order|delivery|delivered|ship(?:ping|ment|ped)?|dispatch(?:ed)?|"
        r"service|supply|expiry|expiration|start|end|travel|event|booking|print(?:ed)?|value|posting|transaction)"
        r"\s*date\b",
        r"\bdate\s*of\s*(?:supply|delivery|order|dispatch|shipment|service)\b",
        r"\btax\s*point(?:\s*date)?\b",
        r"\b(?:billing\s*|service\s*|rental\s*|invoice\s*)?period\b",
        r"\bvalid\s*(?:until|till|through|thru)\b",
    ],
    # (patterns are matched separately: one alternation would let "Date  Due" in "Invoice Date  Due Date" use up
    # the "Due" of the real "Due Date" label)
    "due_date": [
        r"\bdate\s*due\b",
        r"\bdue\s*date\b",
        r"\bpayment\s*due(?:\s*date)?\b",
        r"\bdue\s*(?:by|on)\b",
        r"\bpay(?:able)?\s*by\b",
    ],
    "supplier_name": [
        r"\bfrom\s*:",
        r"\b(?:supplier|vendor|seller|merchant|payee)(?:\s*name)?\b(?!\s*" + _ID_WORDS + r")",
        r"\bsold\s*by\b",
        r"\bbill(?:ed)?\s*by\b",
        r"\bissued\s*by\b",
        r"\binvoice\s*from\b",
        # who is to be paid: "Remit to", "Make cheques payable to", "Pay to" (never the customer)
        r"\bremit(?:tance)?\s*to\b|\b(?:make\s*)?(?:cheques?|checks?|payments?)\s*payable\s*to\b",
        r"\bpay(?:able)?\s*to\b",
    ],
    "customer_name": [
        r"\bbill(?:ed)?[\s-]*to\b",
        r"\binvoice(?:d)?[\s-]*to\b",
        r"\bsold[\s-]*to\b",
        r"\b(?:customer|client|buyer|purchaser)(?:\s*name)?\b(?!\s*" + _ID_WORDS + r")",
        r"\bto\s*:",
        r"\b(?:ship|deliver)[\s-]*to\b",
        r"\b(?:consignee|recipient)\b",
    ],
    "subtotal": [
        r"\bsub[\s-]*total\b(?:\s*\(\s*" + _EXCL + r"\s*(?:vat|tax|gst)\s*\))?",
        r"\bnet\s*(?:amount|total|value|sum)\b",
        r"\btotal\s*" + _EXCL + r"\s*(?:vat|tax|gst)\b|\btotal\s*\(\s*" + _EXCL + r"\s*(?:vat|tax|gst)\s*\)",
        r"\bamount\s*" + _EXCL + r"\s*(?:tax|vat|gst)\b|\bamount\s*\(\s*" + _EXCL + r"\s*(?:tax|vat|gst)\s*\)",
        r"\btaxable\s*(?:amount|value)\b",
        r"\bnet\b(?!\s*(?:\d|payable|due|terms|weight|wt))",
    ],
    "discount": [r"\b(?:less\s*)?discount\b", r"\brebate\b"],
    "shipping": [
        r"\bshipping(?:\s*(?:&|and)\s*handling)?\b",
        r"\bfreight\b",
        r"\bdelivery(?:\s*(?:charges?|fee|cost))?\b(?!\s*(?:date|address|note|to\b))",
        r"\bpostage\b",
        r"\bhandling\b",
        r"\bcarriage\b",
    ],
    "tax": [
        r"\btotal\s*(?:tax|vat|gst)\b|\b(?:(?:sales\s*|value\s*added\s*)?tax|vat|gst)\s*total\b",  # "Tax Total"
        r"\b(?:sales\s*|value\s*added\s*)?tax(?:\s*amount)?\b(?!\s*(?:invoice|exempt|point|" + _ID_WORDS + r"))",
        r"\b(?:vat|gst|hst|pst|qst|cgst|sgst|igst|utgst|iva|mwst|tva|btw|moms)\b(?:\s*amount)?"
        r"(?!\s*" + _ID_WORDS + r")",
    ],
    "total": [
        r"\bgrand\s*total\b",
        r"\btotal\s*(?:amount\s*)?(?:due|payable)\b",
        r"\b(?:net\s*)?(?:amount|balance)\s*(?:due|payable)\b",  # "Net Amount Payable" is one total label
        r"\binvoice\s*(?:total|amount)\b",
        r"\btotal\s*amount\b",
        r"\btotal\s*" + _INCL + r"\s*(?:vat|tax|gst)\b|\btotal\s*\(\s*" + _INCL + r"\s*(?:vat|tax|gst)\s*\)",
        r"\bnet\s*payable\b",
        r"\bamount\s*to\s*pay\b",
        r"\btotal\b(?!\s*(?:tax|vat|gst|excl|excluding|exclusive|before|ex\b|w/o|without|qty|quantity|items?|units|"
        r"hours|pages?|weight|discount))",
        r"\bamount\s*" + _INCL + r"\s*(?:vat|tax|gst)\b|\bamount\s*\(\s*" + _INCL + r"\s*(?:vat|tax|gst)\s*\)",
    ],
    "currency": [r"\bcurrency\b"],
}

FIELD_LABELS: dict[str, list[re.Pattern[str]]] = {
    field: [re.compile(p, re.IGNORECASE) for p in patterns] for field, patterns in _RAW_LABELS.items()
}


@dataclass(frozen=True, slots=True)
class LabelMatch:
    field: str
    start: int
    end: int
    text: str
    priority: int  # index of the matching pattern within the field (lower = more specific)


# Captions that share a money word but name something else ("Shipping Address", "Total Savings", "GST included in
# total"): checked on the text right after (or, for "in total", before) the chosen label.
_NOT_SHIPPING = re.compile(
    r"^\s*(?:address|addr|method|via|terms|date|instructions|details|info|point|mode|carrier|service|status|"
    r"no\b|number|#|id\b|ref|reference|tracking|label|policy|weight|zone|type)",
    re.IGNORECASE,
)
_NOT_TOTAL_AFTER = re.compile(
    r"^\s*(?:savings|saved|pieces|pcs|count|points|kg|kgs|lbs|volume|cartons|packages|boxes|sheets|lines|records)\b",
    re.IGNORECASE,
)
_NOT_TOTAL_BEFORE = re.compile(r"\b(?:in|of|to|from|towards|into)\s+(?:the\s+)?$", re.IGNORECASE)


def _lookalike(text: str, label: LabelMatch) -> bool:
    if label.field == "shipping":
        return bool(_NOT_SHIPPING.match(text[label.end :]))
    if label.field == "total":
        return bool(_NOT_TOTAL_AFTER.match(text[label.end :]) or _NOT_TOTAL_BEFORE.search(text[: label.start]))
    return False


def find_labels(text: str) -> list[LabelMatch]:
    """All field labels in ``text``, left to right, overlaps resolved in favour of the longest label; money words
    that caption something else ("Shipping Address", "Total Savings", "included in total") are dropped."""
    if not text:
        return []
    candidates: list[LabelMatch] = []
    for field, patterns in FIELD_LABELS.items():
        for priority, pattern in enumerate(patterns):
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    candidates.append(LabelMatch(field, m.start(), m.end(), m.group(0), priority))
    candidates.sort(key=lambda c: (-(c.end - c.start), c.priority, c.start))
    chosen: list[LabelMatch] = []
    for c in candidates:
        if all(c.end <= k.start or c.start >= k.end for k in chosen):
            chosen.append(c)
    return sorted((c for c in chosen if not _lookalike(text, c)), key=lambda c: c.start)


def match_label(text: str) -> tuple[str, re.Match[str]] | None:
    """The field whose label starts ``text`` (ignoring leading whitespace/punctuation), e.g. a key cell."""
    if not text:
        return None
    stripped = text.lstrip(" \t*-•:#")
    offset = len(text) - len(stripped)
    labels = [lbl for lbl in find_labels(text) if lbl.start <= offset + 1]
    if not labels:
        return None
    first = labels[0]
    pattern = FIELD_LABELS[first.field][first.priority]
    m = pattern.match(text, first.start)
    return (first.field, m) if m else None


# --------------------------------------------------------------------------------------------------- columns

LEDGER_COLUMNS: dict[str, tuple[str, ...]] = {
    "invoice_number": (
        "invoice no", "invoice number", "invoice #", "invoice num", "inv no", "inv #", "inv number", "invoice id",
        "invoice", "bill no", "bill number", "document no", "doc no", "receipt no", "receipt #",
    ),
    "invoice_date": ("invoice date", "date", "inv date", "bill date", "issue date", "document date", "doc date"),
    "due_date": ("due date", "payment due", "due"),
    "supplier_name": (
        "supplier", "vendor", "supplier name", "vendor name", "seller", "payee", "from", "company", "biller",
    ),
    "customer_name": ("customer", "client", "customer name", "client name", "buyer", "bill to", "sold to"),
    "subtotal": (
        "subtotal", "sub total", "net amount", "net", "amount excl tax", "amount before tax", "taxable amount",
        "net total", "amount excl vat",
    ),
    "discount": ("discount",),
    "shipping": ("shipping", "freight", "delivery"),
    "tax": ("tax", "vat", "gst", "tax amount", "vat amount", "gst amount", "sales tax", "total tax"),
    "total": (
        "total", "grand total", "amount", "total amount", "invoice total", "amount due", "gross amount", "gross",
        "total incl tax", "total incl vat", "invoice amount",
    ),
    "currency": ("currency", "ccy", "cur", "curr"),
}

LINE_ITEM_COLUMNS: dict[str, tuple[str, ...]] = {
    "description": (
        "description", "item", "item description", "items", "product", "service", "details", "particulars",
        "article", "description of goods", "goods", "product description", "service description",
    ),
    "quantity": ("qty", "quantity", "units", "hours", "hrs", "no of units", "nos", "qty.", "pcs", "unit qty"),
    "unit_price": (
        "unit price", "price", "rate", "unit cost", "price per unit", "unit rate", "cost", "each", "unit amount",
        "price/unit", "rate per unit",
    ),
    "amount": (
        "amount", "line total", "total", "ext price", "extended price", "net amount", "value", "line amount",
        "subtotal", "amount (excl tax)",
    ),
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {**LEDGER_COLUMNS, **LINE_ITEM_COLUMNS}


def normalize_header(cell: str) -> str:
    """Lower-case, '#'-preserving, punctuation-free form of a header cell."""
    s = cell.casefold().replace("\n", " ")
    s = re.sub(r"[^\w#/%() ]", " ", s)
    s = re.sub(r"\(\s*[a-z]{3}\s*\)|\(\s*[$€£₹]\s*\)", " ", s)  # "Total (USD)" -> "total"
    return re.sub(r"\s+", " ", s).strip()


def match_column(header: str, columns: Mapping[str, Sequence[str]] = COLUMN_ALIASES) -> str | None:
    """The field a header cell names: exact alias match first, then the longest alias the header starts with."""
    if not header:
        return None
    h = normalize_header(header)
    if not h:
        return None
    for field, aliases in columns.items():
        if h in aliases:
            return field
    best: tuple[int, str] | None = None
    for field, aliases in columns.items():
        for alias in aliases:
            if len(alias) >= 3 and (h.startswith(alias + " ") or h.endswith(" " + alias)) and (
                best is None or len(alias) > best[0]
            ):
                best = (len(alias), field)
    return best[1] if best else None
