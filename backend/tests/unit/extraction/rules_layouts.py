"""Hand-written invoice layouts for the rules extractor tests, and helpers that render them as ``Line`` objects.

Every layout is plain, column-aligned text. ``as_lines(text, mode)`` renders it three ways so the extractor is
exercised on each kind of input it receives in production:

- ``"text"``: ``rules.lines_from_text`` (pseudo-geometry from character columns);
- ``"pdf"``: one span per word in PDF points (as PyMuPDF ``get_text("words")`` gives them);
- ``"ocr"``: one span per phrase (words joined across single spaces, as OCR text boxes come), with jittered
  baselines and a confidence score.

``span_line`` builds a line from explicitly placed cells for layouts that plain text cannot express (right-aligned
values that do not share character columns with their labels, values printed above their captions...).
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any

from app.domain.schemas import DocumentType
from app.extraction.rules import lines_from_text
from app.ingestion.types import Line, Span

_ADVANCE = 5.4  # PDF points per character column
_LEFT = 36.0
_TOP = 50.0
_LEADING = 13.5
_HEIGHT = 9.5


def _pdf_line(row: str, index: int) -> Line | None:
    y0 = _TOP + index * _LEADING
    spans = [
        Span(m.group(0), _LEFT + m.start() * _ADVANCE, y0, _LEFT + m.end() * _ADVANCE - 0.6, y0 + _HEIGHT)
        for m in re.finditer(r"\S+", row)
    ]
    return Line(spans=spans) if spans else None


def _ocr_line(row: str, index: int, conf: float) -> Line | None:
    jitter = ((index * 7) % 3 - 1) * 0.8
    y0 = _TOP + index * _LEADING + jitter
    spans = [
        Span(m.group(0), _LEFT + m.start() * _ADVANCE, y0, _LEFT + m.end() * _ADVANCE - 0.6, y0 + _HEIGHT, conf=conf)
        for m in re.finditer(r"\S+(?: \S+)*", row)
    ]
    return Line(spans=spans) if spans else None


def as_lines(text: str, mode: str, *, conf: float = 0.93) -> list[Line]:
    """Render column-aligned ``text`` as lines in the given input ``mode`` (text | pdf | ocr)."""
    if mode == "text":
        return lines_from_text(text)
    rows = text.split("\n")
    out: list[Line] = []
    for i, row in enumerate(rows):
        line = _pdf_line(row, i) if mode == "pdf" else _ocr_line(row, i, conf)
        if line is not None:
            out.append(line)
    return out


def pages_as_lines(pages: list[str], mode: str) -> tuple[list[Line], list[int]]:
    """Lines of several pages (each page's y restarts at the top) plus the page number of every line."""
    lines: list[Line] = []
    numbers: list[int] = []
    for number, text in enumerate(pages, start=1):
        page_lines = as_lines(text, mode)
        lines.extend(page_lines)
        numbers.extend([number] * len(page_lines))
    return lines, numbers


def span_line(y: float, *cells: tuple[str, float], conf: float | None = None, char_w: float = 5.4) -> Line:
    """A line from explicitly placed cells ``(text, x0)``; each cell is split into word spans."""
    spans: list[Span] = []
    for text, x0 in cells:
        x = x0
        for word in text.split(" "):
            if word:
                spans.append(Span(word, x, y, x + len(word) * char_w - 0.6, y + _HEIGHT, conf=conf))
            x += (len(word) + 1) * char_w
    return Line(spans=spans)


def item(description: str, quantity: str | None, unit_price: str | None, amount: str | None) -> dict[str, Any]:
    return {
        "description": description,
        "quantity": Decimal(quantity) if quantity is not None else None,
        "unit_price": Decimal(unit_price) if unit_price is not None else None,
        "amount": Decimal(amount) if amount is not None else None,
    }


# ------------------------------------------------------------------------------------------------ layouts

US = """\
ACME Corporation
1200 Market Street, Suite 400
San Francisco, CA 94103
Phone: (415) 555-0100   billing@acme.example
                                                  INVOICE
Invoice #: 10042                         Invoice Date: 03/15/2024
PO Number: PO-7781                       Due Date: 04/14/2024

Bill To:                                 Ship To:
Globex Inc.                              Globex Warehouse
500 Oak Avenue                           77 Dock Road
Springfield, IL 62704                    Springfield, IL 62705

Item   Description                   Qty      Rate       Amount
1      Widget A                       10    $25.00      $250.00
2      Widget B (blue)                 5    $40.00      $200.00
3      Installation service            1   $150.00      $150.00

                                     Subtotal           $600.00
                                     Discount (5%)      -$30.00
                                     Shipping            $25.00
                                     Sales Tax (8.25%)   $47.03
                                     Total              $642.03
                                     Amount Due         $642.03

Payment terms: Net 30. Thank you for your business!
"""

US_EXPECTED: dict[str, Any] = {
    "invoice_number": "10042",
    "invoice_date": date(2024, 3, 15),
    "due_date": date(2024, 4, 14),
    "supplier_name": "ACME Corporation",
    "customer_name": "Globex Inc.",
    "subtotal": Decimal("600.00"),
    "discount": Decimal("30.00"),
    "shipping": Decimal("25.00"),
    "tax": Decimal("47.03"),
    "tax_rate": Decimal("8.25"),
    "total": Decimal("642.03"),
    "currency": "USD",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Widget A", "10", "25.00", "250.00"),
        item("Widget B (blue)", "5", "40.00", "200.00"),
        item("Installation service", "1", "150.00", "150.00"),
    ],
}

EU_VAT = """\
Müller & Schmidt GmbH
Hauptstraße 12, 10115 Berlin, Deutschland
USt-IdNr.: DE123456789

RECHNUNG / INVOICE

Invoice No.: RE-2024-0815
Invoice Date: 05.03.2024
Due Date: 04.04.2024

Customer:
Contoso B.V.
Keizersgracht 100, 1015 Amsterdam, Netherlands

Description                     Qty     Unit Price         Amount
Beratung / Consulting            12        95,00 €     1.140,00 €
Reisekosten / Travel              1       234,56 €       234,56 €

Net Amount                                           1.374,56 €
VAT 19%                                                261,17 €
Total                                                1.635,73 €
"""

EU_VAT_EXPECTED: dict[str, Any] = {
    "invoice_number": "RE-2024-0815",
    "invoice_date": date(2024, 3, 5),
    "due_date": date(2024, 4, 4),
    "supplier_name": "Müller & Schmidt GmbH",
    "customer_name": "Contoso B.V.",
    "subtotal": Decimal("1374.56"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("261.17"),
    "tax_rate": Decimal("19"),
    "total": Decimal("1635.73"),
    "currency": "EUR",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Beratung / Consulting", "12", "95.00", "1140.00"),
        item("Reisekosten / Travel", "1", "234.56", "234.56"),
    ],
}

UK = """\
                                                   TAX INVOICE
Brightwater Design Ltd
Unit 4, Riverside Works, Leeds LS1 4AP
VAT Reg No: GB 123 4567 89

Invoice Number: BWD-1042
Date of Issue: 5 March 2024
Payment Due: 4 April 2024

Invoice To:
Northwind Traders Ltd
Attn: Accounts Payable
22 King Street, Manchester M2 4LQ

Description                                   Qty      Price       Total
Brand identity package                          1    £850.00     £850.00
Business cards (box of 500)                     2     £45.00      £90.00
Website banner set                              3     £60.00     £180.00

Subtotal                                                       £1,120.00
VAT @ 20%                                                        £224.00
Total Due                                                      £1,344.00
"""

UK_EXPECTED: dict[str, Any] = {
    "invoice_number": "BWD-1042",
    "invoice_date": date(2024, 3, 5),
    "due_date": date(2024, 4, 4),
    "supplier_name": "Brightwater Design Ltd",
    "customer_name": "Northwind Traders Ltd",
    "subtotal": Decimal("1120.00"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("224.00"),
    "tax_rate": Decimal("20"),
    "total": Decimal("1344.00"),
    "currency": "GBP",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Brand identity package", "1", "850.00", "850.00"),
        item("Business cards (box of 500)", "2", "45.00", "90.00"),
        item("Website banner set", "3", "60.00", "180.00"),
    ],
}

INDIA_GST = """\
                                TAX INVOICE
Sharma Traders Pvt. Ltd.
12, MG Road, Bengaluru, Karnataka 560001
GSTIN: 29ABCDE1234F1Z5    PAN: ABCDE1234F

Invoice No: ST/2024-25/0042                 Invoice Date: 15/03/2024
Place of Supply: Karnataka                  Due Date: 14/04/2024

Bill To:                                    Ship To:
Kumar Enterprises                           Kumar Enterprises Warehouse
45, Industrial Area, Peenya                 Plot 9, KIADB, Hosur Road
GSTIN: 29PQRSX5678K1Z2

S.No  Description            HSN     Qty        Rate         Amount
1     Steel rods 12mm        7214     10    1,250.00      12,500.00
2     Cement bags (50kg)     2523     50      350.00      17,500.00

                                  Taxable Value         ₹ 30,000.00
                                  CGST @ 9%              ₹ 2,700.00
                                  SGST @ 9%              ₹ 2,700.00
                                  Grand Total           ₹ 35,400.00

Amount in words: Rupees Thirty Five Thousand Four Hundred Only
"""

INDIA_GST_EXPECTED: dict[str, Any] = {
    "invoice_number": "ST/2024-25/0042",
    "invoice_date": date(2024, 3, 15),
    "due_date": date(2024, 4, 14),
    "supplier_name": "Sharma Traders Pvt. Ltd.",
    "customer_name": "Kumar Enterprises",
    "subtotal": Decimal("30000.00"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("5400.00"),
    "tax_rate": None,  # two rates printed (CGST 9% + SGST 9%); the combined 18% is not printed
    "total": Decimal("35400.00"),
    "currency": "INR",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Steel rods 12mm", "10", "1250.00", "12500.00"),
        item("Cement bags (50kg)", "50", "350.00", "17500.00"),
    ],
}

RECEIPT = """\
        MAPLE LEAF CAFE
   88 Queen St W, Toronto ON M5H 2M9
        Tel: (416) 555-0199
     HST Reg #: 81234 5678 RT0001

SALES RECEIPT
Receipt #: 004217
Date: 2024-03-05   10:32 AM

Latte (large)                 5.25
Blueberry muffin              3.75
2 x Croissant                 7.00

SUBTOTAL                     16.00
HST 13%                       2.08
TOTAL                       $18.08

VISA ****1234                18.08
Thank you for visiting!
"""

RECEIPT_EXPECTED: dict[str, Any] = {
    "invoice_number": "004217",
    "invoice_date": date(2024, 3, 5),
    "due_date": None,
    "supplier_name": "MAPLE LEAF CAFE",
    "customer_name": None,
    "subtotal": Decimal("16.00"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("2.08"),
    "tax_rate": Decimal("13"),
    "total": Decimal("18.08"),
    "currency": "CAD",  # "$" on a receipt with HST and a Canadian postcode
    "document_type": DocumentType.RECEIPT,
    "line_items": [
        item("Latte (large)", None, None, "5.25"),
        item("Blueberry muffin", None, None, "3.75"),
        item("Croissant", "2", None, "7.00"),
    ],
}

FROM_TO = """\
INVOICE

From:                                   To:
Pixel Forge Studio LLC                  Initech Corporation
48 Elm Street                           9 Corporate Plaza
Austin, TX 78701                        Dallas, TX 75201
hello@pixelforge.example                Attn: Bill Lumbergh

Invoice No.: PF-2024-031
Issue Date: 2024-02-20
Due Date: 2024-03-21
Currency: USD

Description                                   Hours       Rate        Amount
UI design sprint                                 40      85.00      3,400.00
Usability testing                                12      85.00      1,020.00

Subtotal                                                            4,420.00
Tax (0%)                                                                0.00
Total                                                               4,420.00
"""

FROM_TO_EXPECTED: dict[str, Any] = {
    "invoice_number": "PF-2024-031",
    "invoice_date": date(2024, 2, 20),
    "due_date": date(2024, 3, 21),
    "supplier_name": "Pixel Forge Studio LLC",
    "customer_name": "Initech Corporation",
    "subtotal": Decimal("4420.00"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("0.00"),
    "tax_rate": Decimal("0"),
    "total": Decimal("4420.00"),
    "currency": "USD",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("UI design sprint", "40", "85.00", "3400.00"),
        item("Usability testing", "12", "85.00", "1020.00"),
    ],
}

LABELS_ABOVE = """\
NORTHWIND LOGISTICS PTY LTD
Level 3, 120 Collins Street, Melbourne VIC 3000
ABN 51 824 753 556

TAX INVOICE

Invoice Number     Invoice Date     Due Date       Amount Due
NW-88123           12/03/2024       11/04/2024     AUD 2,563.00

Customer
Fabrikam Pty Ltd
PO Box 77, Sydney NSW 2001

Description                          Qty    Unit Price        Amount
Pallet freight Melbourne-Sydney        6        320.00      1,920.00
Fuel surcharge                         1        210.00        210.00
Tail-lift delivery                     2        100.00        200.00

Subtotal                                                    2,330.00
GST 10%                                                       233.00
Total                                                       2,563.00
"""

LABELS_ABOVE_EXPECTED: dict[str, Any] = {
    "invoice_number": "NW-88123",
    "invoice_date": date(2024, 3, 12),  # day/month ambiguous; AUD documents use day-first dates
    "due_date": date(2024, 4, 11),
    "supplier_name": "NORTHWIND LOGISTICS PTY LTD",
    "customer_name": "Fabrikam Pty Ltd",
    "subtotal": Decimal("2330.00"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("233.00"),
    "tax_rate": Decimal("10"),
    "total": Decimal("2563.00"),
    "currency": "AUD",
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Pallet freight Melbourne-Sydney", "6", "320.00", "1920.00"),
        item("Fuel surcharge", "1", "210.00", "210.00"),
        item("Tail-lift delivery", "2", "100.00", "200.00"),
    ],
}

MULTIPAGE_P1 = """\
Contoso Office Supplies Inc.
1 Microsoft Way, Redmond, WA 98052

INVOICE
Invoice No: CS-55120
Invoice Date: March 18, 2024
Due Date: April 17, 2024

Sold To:
Litware Inc.
300 Pine St, Seattle, WA 98101

Description                        Qty   Unit Price     Line Total
Copy paper A4 (box)                 10        42.00         420.00
Ballpoint pens (pack of 12)         25         6.50         162.50
Stapler heavy duty                   4        18.75          75.00

Continued on next page                                  Page 1 of 2
"""

MULTIPAGE_P2 = """\
Invoice No: CS-55120 (continued)                         Page 2 of 2

Description                        Qty   Unit Price     Line Total
Desk organizer                       6        24.00         144.00
Whiteboard markers (set)            12         9.25         111.00

Subtotal                                                    912.50
Sales Tax 10%                                                91.25
Total                                                     1,003.75
"""

MULTIPAGE_EXPECTED: dict[str, Any] = {
    "invoice_number": "CS-55120",
    "invoice_date": date(2024, 3, 18),
    "due_date": date(2024, 4, 17),
    "supplier_name": "Contoso Office Supplies Inc.",
    "customer_name": "Litware Inc.",
    "subtotal": Decimal("912.50"),
    "discount": None,
    "shipping": None,
    "tax": Decimal("91.25"),
    "tax_rate": Decimal("10"),
    "total": Decimal("1003.75"),
    "currency": None,  # no currency marker anywhere: never defaulted
    "document_type": DocumentType.INVOICE,
    "line_items": [
        item("Copy paper A4 (box)", "10", "42.00", "420.00"),
        item("Ballpoint pens (pack of 12)", "25", "6.50", "162.50"),
        item("Stapler heavy duty", "4", "18.75", "75.00"),
        item("Desk organizer", "6", "24.00", "144.00"),
        item("Whiteboard markers (set)", "12", "9.25", "111.00"),
    ],
}

# name -> (pages, expected)
LAYOUTS: dict[str, tuple[list[str], dict[str, Any]]] = {
    "us": ([US], US_EXPECTED),
    "eu_vat": ([EU_VAT], EU_VAT_EXPECTED),
    "uk": ([UK], UK_EXPECTED),
    "india_gst": ([INDIA_GST], INDIA_GST_EXPECTED),
    "receipt": ([RECEIPT], RECEIPT_EXPECTED),
    "from_to": ([FROM_TO], FROM_TO_EXPECTED),
    "labels_above_values": ([LABELS_ABOVE], LABELS_ABOVE_EXPECTED),
    "multipage_continued": ([MULTIPAGE_P1, MULTIPAGE_P2], MULTIPAGE_EXPECTED),
}


# ------------------------------------------------------------------ further real-world conventions (regressions)

DOTTED_LEADERS = """\
BLUE OCEAN TRADING CO.
Invoice No. ........: BO/7781
Invoice Date .......: 12/04/2024
Customer ...........: Seaside Resorts Ltd
Amount (excl. VAT) .: 1,000.00
VAT 15% ............: 150.00
Total Amount Due ...: 1,150.00 ZAR
"""

FOOTER_LETTERHEAD = """\
Bill To:
Woodgrove Bank
1 Finance Plaza, Boston, MA 02110

Invoice Number: WB-5521
Invoice Date: 02/20/2024

Description                    Hours     Rate       Amount
Audit support                     10   150.00     1,500.00

Total                                             1,500.00

Northwind Consulting LLC | 200 Main St, Boston MA | billing@northwind.example
"""

TALLY_STYLE = """\
                              TAX INVOICE
Invoice No.                 Dated
SI/2024-25/0087             5-Mar-2024
Seller                                      Buyer (Bill to)
Ganesh Hardware Stores                      Lakshmi Builders
22 Market Road, Pune 411002                 Plot 7, MIDC, Pune 411019

Sl No.  Description of Goods      HSN/SAC   Quantity      Rate    per        Amount
1       PVC Pipes 2 inch          3917      40 Nos       120.00   Nos      4,800.00
                                  CGST                                       432.00
                                  SGST                                       432.00
                                  Total                                 ₹ 5,664.00
"""

HASH_NUMBER = """\
Zylker Electronics Hub                                         TAX INVOICE
14B, Northern Street                                          # INV-000123
New York 10001                                                Balance Due
                                                              $1,180.00
Bill To                                     Invoice Date :  05 Mar 2024
Ms. Mary D. Dunton                          Due Date :      05 Mar 2024
"""

OCR_TYPOS = """\
Bluebird Supplies Ltd
lnvoice No: BB-4410
Description           Qty   Price    Amount
Paper reams             5    4.00     20.00
Subtota1                              20.00
VAT 20%                                4.00
Tota1                                 24.00
"""

ITEMS_ABOVE_TAX_LINE = """\
INVOICE #INV-1001                                   Date: 2024-01-15
From: Bright Star Media Ltd                         To: Orion Retail Plc
Services rendered in January                                  £2,400.00
VAT (20%)                                                       £480.00
Total                                                         £2,880.00
"""

CANADA_GST_QST = """\
Northern Lights Outfitters
1200 Rue Sainte-Catherine, Montreal QC H3B 1K9
Invoice: NL-2024-55
Sold To: Caribou Lodge
Subtotal                                      600.00
GST 5%                                         30.00
QST 9.975%                                     59.85
Total                                       $689.85
"""

SPACE_GROUPED_EURO = """\
Invoice No: F-2024-118
BILL-TO: Société Générale de Test SARL
Subtotal                      13 734,56 €
VAT 20%                        2 746,91 €
Total                         16 481,47 €
"""
