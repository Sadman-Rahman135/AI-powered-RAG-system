"""Label vocabulary on top of ``labels.find_labels``, plus the word lists the rules use to classify text.

The shared label table already knows the compound labels (other dates such as "Date of Supply" / "PO Date",
"Date Due", "Pay To" / "Remit To", "BILL-TO", "Tax Total", "Net Amount Payable", "Total (incl. VAT)", "Amount
(excl. VAT)") and drops money look-alikes ("Shipping Address", "Total Savings"). ``find_field_labels`` adds what is
specific to reading a laid-out page:

- "Due:" is a due date; a bare "#" at the start of a cell is an invoice number ("# INV-000123");
- a tax word inside any other qualifier is not a tax line: "Subtotal (excl. VAT)", "Price incl. GST", "Gross (incl
  VAT)"; "Amount net of tax" is a subtotal; "(incl. VAT 20.00)" states the tax and stays a tax label;
- "Discount Total" / "Shipping Total" are one money label, "Total (excl. VAT)" a subtotal;
- matching runs on an OCR-tolerant view of the text ("Tota1", "Subtota1", "lnvoice" read as words).
"""

from __future__ import annotations

import re

from app.extraction.labels import FIELD_LABELS, OTHER_DATE, LabelMatch, find_labels

MONEY_FIELDS = ("subtotal", "discount", "shipping", "tax", "total")
DATE_FIELDS = ("invoice_date", "due_date", OTHER_DATE)
PARTY_FIELDS = ("supplier_name", "customer_name")

_EXTRA: list[tuple[str, int, re.Pattern[str]]] = [
    (field, priority, re.compile(pattern, re.IGNORECASE))
    for field, priority, pattern in (
        ("due_date", 5, r"\bdue\b(?=\s*:)"),  # after every due-date label of the shared table
        ("invoice_number", 9, r"(?:^|(?<=\s))#(?=\s?[A-Za-z0-9/-]*\d)"),  # "# INV-000123" (checked: cell start)
    )
]

_QUALIFIER_GAP = re.compile(
    r"^[\s:]*\(?\s*(?:(?:incl|including|inclusive\s+of|inc|excl|excluding|exclusive\s+of|ex|w/o|without|before|after|"
    r"plus|on|of|net\s+of)\.?\s*)?$",
    re.IGNORECASE,
)
_EXCL_AFTER = re.compile(
    r"^\s*\(?\s*(?:excl|excluding|exclusive\s+of|before|ex|w/o|without|net\s+of)\.?\s*(?:vat|tax|gst|hst|taxes)\b",
    re.IGNORECASE,
)
# "Amount (excl. VAT)" / "Price incl. GST": a tax word inside a qualifier names a net or gross amount, not a tax line.
_TAX_QUALIFIED = re.compile(
    r"(?P<head>\b(?:amount|price|sum|value|net|gross|subtotal|sub\s*total)\b\s*)?\(?\s*"
    r"(?P<q>incl|including|inclusive\s+of|inc|excl|excluding|exclusive\s+of|ex|before|w/o|without|net\s+of)\.?\s*$",
    re.IGNORECASE,
)
_EXCL_WORDS = re.compile(r"^(?:excl|excluding|exclusive|ex|before|w/o|without|net)", re.IGNORECASE)
_PLAIN_TOTAL = re.compile(r"^totals?$", re.IGNORECASE)
# An amount (not a rate) before the qualifier closes: "(incl. VAT 20.00)" states the tax; "(excl. VAT 20%)" does not.
_AMOUNT_AFTER = re.compile(r"^[^)\n%]*?\d[\d.,]*[.,]\d{2}\b(?!\s*%)")


def _requalify_tax(text: str, lab: LabelMatch, prev: LabelMatch | None) -> LabelMatch | None:
    """A tax label right after "excl." / "incl." (with no label of its own before) is the qualifier of a net or
    gross amount: "Amount (excl. VAT)" -> subtotal, "Amount incl. VAT" -> total, "(incl. VAT)" alone -> nothing.
    "(incl. VAT 20.00)" states the tax amount and stays a tax label."""
    window_start = max(prev.end if prev else 0, lab.start - 30)
    m = _TAX_QUALIFIED.search(text, window_start, lab.start)
    if m is None:
        return lab
    start = m.start("head") if m.group("head") else m.start("q")
    if _EXCL_WORDS.match(m.group("q")):
        return LabelMatch("subtotal", start, lab.end, text[start : lab.end], 2)
    if m.group("head"):
        return LabelMatch("total", start, lab.end, text[start : lab.end], 5)
    return lab if _AMOUNT_AFTER.match(text[lab.end :]) else None


def _resolve(candidates: list[LabelMatch]) -> list[LabelMatch]:
    candidates.sort(key=lambda c: (-(c.end - c.start), c.priority, c.start))
    chosen: list[LabelMatch] = []
    for c in candidates:
        if all(c.end <= k.start or c.start >= k.end for k in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.start)


_OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "5": "s", "|": "l"})
_OCR_TOKEN = re.compile(r"[A-Za-z|]*[015|][A-Za-z015|]*")
_OCR_INVOICE = re.compile(r"(?<![A-Za-z])[l1|](?=nvoice)", re.IGNORECASE)


def ocr_label_view(text: str) -> str:
    """``text`` with OCR look-alikes inside words fixed for label matching ("Tota1" -> "Total", "lnvoice" ->
    "Invoice"). Same length, so label offsets still point into the original text; values are read from there."""
    text = _OCR_INVOICE.sub("I", text)
    if not re.search(r"[015|]", text):
        return text

    def fix(m: re.Match[str]) -> str:
        token = m.group(0)
        letters = sum(ch.isalpha() for ch in token)
        return token.translate(_OCR_DIGITS) if letters >= 3 and letters > len(token) - letters else token

    return _OCR_TOKEN.sub(fix, text)


def find_field_labels(text: str) -> list[LabelMatch]:
    """Field labels in ``text`` (left to right), with the fixes listed in the module docstring. Matching runs on
    ``ocr_label_view(text)``; offsets refer to ``text`` itself."""
    if not text:
        return []
    text = ocr_label_view(text)
    candidates = list(find_labels(text))
    for field, priority, pattern in _EXTRA:
        for m in pattern.finditer(text):
            candidates.append(LabelMatch(field, m.start(), m.end(), m.group(0), priority))
    labels = _resolve(candidates)
    out: list[LabelMatch] = []
    for lab in labels:
        prev = out[-1] if out else None
        between = text[prev.end : lab.start] if prev else ""
        if prev is not None and len(between) <= 15 and _QUALIFIER_GAP.match(between):
            if prev.field in MONEY_FIELDS and lab.field == "total" and _PLAIN_TOTAL.match(lab.text):
                out[-1] = LabelMatch(prev.field, prev.start, lab.end, text[prev.start : lab.end], prev.priority)
                continue  # "Discount Total", "Shipping Total" ("Tax Total" is one label in the shared table)
            qualifier = "(" in between or between.strip(" :")
            states_amount = lab.field == "tax" and _AMOUNT_AFTER.match(text[lab.end :])
            if lab.field in MONEY_FIELDS and prev.field in MONEY_FIELDS and qualifier and not states_amount:
                continue  # "Total (incl. VAT)": VAT qualifies the total
        if lab.field == "tax":
            requalified = _requalify_tax(text, lab, prev)
            if requalified is None:
                continue
            lab = requalified
        if lab.field == "total" and _EXCL_AFTER.match(text[lab.end :]):
            lab = LabelMatch("subtotal", lab.start, lab.end, lab.text, 2)  # "Total (excl. VAT)"
        out.append(lab)
    return out


def total_rank(label: LabelMatch, after: str) -> int:
    """Preference among total labels (lower wins): grand total < amount due < balance due < total incl. tax <
    invoice total < plain total."""
    t = re.sub(r"\s+", " ", label.text.lower())
    if "grand" in t:
        return 0
    if "balance" in t:
        return 2
    if "due" in t or "payable" in t or "to pay" in t:
        return 1
    if re.search(r"\b(?:incl|including|inclusive|inc|with)\b", t) or re.match(
        r"^\s*\(?\s*(?:incl|including|inclusive\s+of|inc)\.?\s*(?:vat|tax|gst|hst|taxes)\b", after, re.IGNORECASE
    ):
        return 3
    if "invoice" in t or "amount" in t:
        return 4
    return 5


def tax_kind(label_text: str) -> str:
    """Normalised tax label ("CGST @" -> "cgst", "Sales Tax" -> "sales tax"); aggregate labels -> "total"."""
    t = re.sub(r"[^a-z ]", " ", label_text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if t.startswith("total") or t.endswith("total"):
        return "total"
    return t.replace(" amount", "")


def is_aggregate_tax(label: LabelMatch) -> bool:
    return tax_kind(label.text) == "total"


# ------------------------------------------------------------------------------------------- word lists

TITLE_CORE: dict[str, str] = {
    "credit note": "credit_note",
    "credit memo": "credit_note",
    "credit invoice": "credit_note",
    "gutschrift": "credit_note",
    "receipt": "receipt",
    "invoice": "invoice",
    "bill": "invoice",
    "rechnung": "invoice",
    "facture": "invoice",
    "factura": "invoice",
    "fattura": "invoice",
    "faktura": "invoice",
}
TITLE_WORDS = frozenset(
    {
        "tax", "commercial", "proforma", "pro", "forma", "sales", "cash", "payment", "original", "duplicate",
        "copy", "for", "recipient", "customer", "supplier", "of", "supply", "gst", "vat", "simplified", "official",
        "invoice", "receipt", "credit", "note", "memo", "bill", "rechnung", "facture", "factura", "fattura",
        "faktura", "gutschrift", "and", "&",
    }
)

COMPANY_SUFFIXES = frozenset(
    {
        "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp", "corporation", "co", "company",
        "gmbh", "plc", "pvt", "private", "sa", "sas", "sarl", "bv", "nv", "ag", "kg", "srl", "spa", "pty", "oy",
        "ab", "as", "aps", "kft", "sro", "ug", "ohg", "gbr", "bhd", "sdn", "pte", "lda", "ltda", "cia",
    }
)
BUSINESS_WORDS = frozenset(
    {
        "group", "holdings", "enterprises", "traders", "trading", "industries", "solutions", "services",
        "studio", "studios", "consulting", "partners", "associates", "systems", "technologies", "labs",
        "logistics", "supplies", "store", "shop", "cafe", "restaurant", "bakery", "market", "agency",
    }
)

PAYMENT_WORDS = re.compile(
    r"\b(?:cash|change|tendered|visa|mastercard|amex|debit|credit\s*card|card|paid|payment|balance|tip|gratuity|"
    r"rounding|round\s*off|refund|points|loyalty|approval|auth)\b",
    re.IGNORECASE,
)
ATTENTION_RE = re.compile(r"^\s*(?:attn|attention|c/o|care\s+of|contact|tel|phone|mobile|fax|e-?mail)\b", re.IGNORECASE)
CONTACT_RE = re.compile(
    r"@|\bwww\.|https?://|\.(?:com|net|org|io|co|biz)\b|\b(?:tel|phone|fax|mobile|e-?mail)\b|\+\d", re.IGNORECASE
)
TAX_ID_RE = re.compile(
    r"\b(?:vat|gst|gstin|hst|abn|acn|ein|tin|pan|uen|nzbn|cif|nif|siret|siren|ust|ust-?idnr|tax\s*id|"
    r"tax\s*no|reg(?:istration)?|company\s*(?:no|number|reg)|crn|kvk|iban|swift|bic|sort\s*code|account)\b",
    re.IGNORECASE,
)
STREET_RE = re.compile(
    r"\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|plaza|square|sq|suite|ste|floor|"
    r"fl|level|unit|building|bldg|block|sector|nagar|marg|po\s*box|p\.?\s?o\.?\s*box|box|highway|hwy|court|ct|"
    r"place|pl|parkway|pkwy|terrace|close|crescent|cres|industrial|area|estate|park|strasse|straße|str|weg|"
    r"platz|gasse|allee|rue|calle|via|gracht|plein|laan|plot|apt|apartment)\b\.?",
    re.IGNORECASE,
)
POSTCODE_RE = re.compile(
    r",\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"  # US: ", CA 94103"
    r"|\b[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b"  # UK: "LS1 4AP"
    r"|\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s?\d[ABCEGHJ-NPRSTV-Z]\d\b"  # Canada: "M5H 2M9"
    r"|\b(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+\d{4}\b"  # Australia: "VIC 3000"
    r"|\b\d{4,6}\s+[A-Z][a-zäöüé]+"  # EU / India: "10115 Berlin", "560001 Bengaluru"
    r"|\b\d{6}\b"  # India PIN
)
ID_WORDS_RE = re.compile(
    r"#|\b(?:no|nr|num|number|id|reg|registration|gstin|abn|ein|tin|pan|account|acct|a/c|iban|swift|bic|ref|"
    r"reference|phone|tel|fax|code)\b",
    re.IGNORECASE,
)
KEY_RE = re.compile(r"(?:^|(?<=\s))[A-Za-z][A-Za-z.&/'#-]*(?:\s[A-Za-z][A-Za-z.&/'#-]*){0,3}\s?#?\s*:(?![\d/])")
SECTION_WORDS = frozenset({"details", "detail", "information", "info", "address", "addr", "data"})
CONTINUED_RE = re.compile(r"\bcontinued\b|\bcont'?d\b|\(cont\.?\)", re.IGNORECASE)
PAGE_OF_RE = re.compile(r"\bpage\s*(\d{1,4})\s*(?:of|/)\s*(\d{1,4})\b", re.IGNORECASE)

INVOICE_NUMBER_LABEL_RE = re.compile("|".join(p.pattern for p in FIELD_LABELS["invoice_number"]), re.IGNORECASE)
TOTAL_LABEL_RE = re.compile("|".join(p.pattern for p in FIELD_LABELS["total"]), re.IGNORECASE)
PARTY_LABEL_RE = re.compile(
    "|".join(p.pattern for p in FIELD_LABELS["customer_name"] + FIELD_LABELS["supplier_name"]), re.IGNORECASE
)
