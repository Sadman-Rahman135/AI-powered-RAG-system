"""Rendering: invoice specs -> PDF pages via PyMuPDF HTML boxes, pagination, rasterization and PDF/TIFF assembly.

Layout model. A page is a :class:`Canvas` on which a template places HTML blocks (``insert_htmlbox``) at absolute
columns and stacks them vertically. The same drawing code runs in *dry* mode (no page; blocks are only laid out with
``Story.place``) to measure heights, which is how long item tables are paginated: header on page 1, a compact
"continued" header on the following pages, ``Page k of n`` footers and the totals on the last page.

Fonts are PyMuPDF's built-in ones only (Nimbus Sans/Roman/Mono, Charis SIL, and Noto Serif as the fallback for glyphs
such as the rupee sign), so the dataset renders identically on any machine and text extraction is clean.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pymupdf
from PIL import Image

from eval.synth.data import InvoiceSpec
from eval.synth.templates import TEMPLATES, Template

BODY_MARGIN_CSS = "body {margin:1px;}"  # insert_htmlbox always prepends this; dry measuring must match it
MAX_BLOCK_HEIGHT = 20000.0
PDF_DATE = "D:20240101000000Z"
PDF_METADATA = {
    "producer": "eval-synth",
    "creator": "eval-synth",
    "author": "",
    "subject": "",
    "keywords": "",
    "creationDate": PDF_DATE,
    "modDate": PDF_DATE,
}


class LayoutOverflow(RuntimeError):
    """A block did not fit where the pagination plan put it (a bug in a template's height model)."""


class Canvas:
    """A page (or, in dry mode, nothing) plus a vertical cursor; templates place HTML blocks on it."""

    def __init__(self, page: pymupdf.Page | None, width: float, height: float, css: str, bottom: float) -> None:
        self.page = page
        self.width = width
        self.height = height
        self.css = css
        self.bottom = bottom
        self.y = 0.0

    @property
    def dry(self) -> bool:
        return self.page is None

    def measure(self, html: str, x0: float, x1: float, css: str = "") -> float:
        """Height of ``html`` laid out in the column ``[x0, x1]``."""
        story = pymupdf.Story(html=html, user_css=BODY_MARGIN_CSS + self.css + css)
        _, filled = story.place(pymupdf.Rect(0, 0, x1 - x0, MAX_BLOCK_HEIGHT))
        return float(filled[3])

    def html(self, html: str, x0: float, x1: float, *, y: float | None = None, css: str = "") -> float:
        """Draw ``html`` in the column ``[x0, x1]`` at ``y`` (default: the cursor); return its height."""
        top = self.y if y is None else y
        if self.page is None:
            return self.measure(html, x0, x1, css)
        rect = pymupdf.Rect(x0, top, x1, max(self.bottom, top + 1))
        spare, _ = self.page.insert_htmlbox(rect, html, css=self.css + css, scale_low=1)
        if spare < 0:
            raise LayoutOverflow(f"block of {html[:60]!r}... does not fit at y={top:.0f} (bottom {self.bottom:.0f})")
        return rect.height - spare

    def flow(self, html: str, x0: float, x1: float, *, gap: float = 0.0, css: str = "") -> float:
        """Draw at the cursor and move the cursor below the block (plus ``gap``)."""
        height = self.html(html, x0, x1, css=css)
        self.y += height + gap
        return height

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        fill: tuple[float, ...] | None = None,
        stroke: tuple[float, ...] | None = None,
        width: float = 0.6,
        dashes: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        if self.page is not None:
            self.page.draw_rect(
                pymupdf.Rect(x0, y0, x1, y1),
                color=stroke,
                fill=fill,
                width=width,
                dashes=dashes,
                fill_opacity=opacity,
                stroke_opacity=opacity,
            )

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: tuple[float, ...] = (0.2, 0.2, 0.2),
        width: float = 0.6,
        dashes: str | None = None,
    ) -> None:
        if self.page is not None:
            self.page.draw_line((x0, y0), (x1, y1), color=color, width=width, dashes=dashes)

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: tuple[float, ...]) -> None:
        if self.page is not None:
            self.page.draw_polyline(list(points), color=None, fill=fill, closePath=True)

    def circle(self, cx: float, cy: float, radius: float, *, fill: tuple[float, ...]) -> None:
        if self.page is not None:
            self.page.draw_circle((cx, cy), radius, color=None, fill=fill)

    def stamp(self, text: str, cx: float, cy: float, *, size: float, angle: float, color: tuple[float, ...]) -> None:
        """A rotated, translucent rubber-stamp word (e.g. PAID) centred on (cx, cy)."""
        if self.page is None:
            return
        font = pymupdf.Font("hebo")
        width = font.text_length(text, fontsize=size)
        origin = pymupdf.Point(cx - width / 2, cy + size / 3)
        self.page.insert_text(
            origin,
            text,
            fontsize=size,
            fontname="hebo",
            color=color,
            fill_opacity=0.35,
            morph=(pymupdf.Point(cx, cy), pymupdf.Matrix(angle)),
            overlay=True,
        )


# --------------------------------------------------------------------------------------------------------------------
# pagination
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PagePlan:
    start: int  # first item row on the page
    stop: int  # one past the last item row on the page
    first: bool
    last: bool


def _dry_canvas(tpl: Template, spec: InvoiceSpec, height: float) -> Canvas:
    return Canvas(None, tpl.width, height, tpl.css(spec), tpl.bottom(height))


def plan_document(tpl: Template, spec: InvoiceSpec) -> list[PagePlan]:
    """Split the item table over pages so that every page fits; the totals end up on the last page."""
    rows = spec.rows
    n = len(rows)
    if tpl.single_page:
        return [PagePlan(0, n, True, True)]
    height = tpl.page_height(spec)
    cv = _dry_canvas(tpl, spec, height)
    x0, x1 = tpl.table_x
    cv.y = tpl.margin_top
    tpl.draw_header(cv, spec, first=True, page_no=1, n_pages=1)
    first_top = cv.y
    cv.y = tpl.margin_top
    tpl.draw_header(cv, spec, first=False, page_no=2, n_pages=2)
    cont_top = cv.y
    cv.y = 0.0
    tpl.draw_totals(cv, spec)
    totals_h = cv.y
    cv.y = 0.0
    tpl.draw_continued(cv, spec)
    continued_h = cv.y
    limit = cv.bottom - tpl.footer_h

    def table_height(a: int, b: int) -> float:
        return cv.measure(tpl.table_html(spec, a, b), x0, x1) if b > a else 0.0

    plans: list[PagePlan] = []
    a = 0
    while True:
        top = first_top if not plans else cont_top
        avail = limit - top
        if table_height(a, n) + totals_h + 4 <= avail:
            plans.append(PagePlan(a, n, not plans, True))
            return plans
        # fill the page: grow k while the table (plus the "continued" line) still fits
        k = a
        while k < n and table_height(a, k + 1) + continued_h + 4 <= avail:
            k += 1
        if k >= n:  # every row fits but the totals do not: the last row travels to the totals page
            k = n - 1
        if k <= a:
            k = a + 1  # a single oversized row still goes on a page of its own
        plans.append(PagePlan(a, k, not plans, False))
        a = k


def render_spec(doc: pymupdf.Document, spec: InvoiceSpec) -> tuple[int, int]:
    """Append the pages of one document; returns its 1-based (first, last) page numbers in ``doc``."""
    tpl = TEMPLATES[spec.template]
    plans = plan_document(tpl, spec)
    height = tpl.page_height(spec)
    first_page = doc.page_count + 1
    for number, plan in enumerate(plans, start=1):
        page = doc.new_page(width=tpl.width, height=height)
        cv = Canvas(page, tpl.width, height, tpl.css(spec), tpl.bottom(height))
        tpl.draw_page(cv, spec, plan, number, len(plans))
    return first_page, doc.page_count


# --------------------------------------------------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------------------------------------------------
def pdf_bytes(doc: pymupdf.Document) -> bytes:
    """Serialize deterministically (fixed metadata and file id; duplicate font programs merged)."""
    doc.set_metadata(dict(PDF_METADATA))
    return doc.tobytes(garbage=4, deflate=True, no_new_id=True)


@dataclass(frozen=True)
class RenderedPdf:
    data: bytes
    ranges: list[tuple[int, int]]  # (page_start, page_end) per rendered spec, 1-based inclusive
    included: list[int]  # indices (into the given specs) of the specs that were rendered

    @property
    def pages(self) -> int:
        return self.ranges[-1][1] if self.ranges else 0


def render_pdf(specs: Sequence[InvoiceSpec], *, chunk_pages: int = 40, max_pages: int | None = None) -> RenderedPdf:
    """Render documents one after another into a PDF.

    Every HTML box embeds its fonts, so memory would balloon on a long file; pages are therefore rendered into
    chunks that are saved (which merges the duplicate font programs) and appended to the result. With ``max_pages`` a
    document that would push the file past the limit is skipped (and a later, shorter one may still fit), which lets
    a caller hit an exact page count.
    """
    final = pymupdf.open()
    chunk = pymupdf.open()
    ranges: list[tuple[int, int]] = []
    included: list[int] = []
    offset = 0  # pages already moved from chunk into final

    def flush() -> None:
        nonlocal chunk, offset
        if chunk.page_count:
            with pymupdf.open(stream=chunk.tobytes(garbage=4, deflate=True), filetype="pdf") as merged:
                final.insert_pdf(merged)
            offset += chunk.page_count
        chunk.close()
        chunk = pymupdf.open()

    for index, spec in enumerate(specs):
        first, last = render_spec(chunk, spec)
        if max_pages is not None and offset + last > max_pages:
            chunk.delete_pages(first - 1, last - 1)
            continue
        ranges.append((offset + first, offset + last))
        included.append(index)
        if chunk.page_count >= chunk_pages:
            flush()
        if max_pages is not None and offset + chunk.page_count == max_pages:
            break
    flush()
    restore_ligature_text(final)
    data = pdf_bytes(final)
    final.close()
    return RenderedPdf(data, ranges, included)


def encrypt_pdf(data: bytes, *, user_pw: str, owner_pw: str) -> bytes:
    """RC4-128 encrypted copy (deterministic for a fixed file id); an empty ``user_pw`` leaves it openable."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_RC4_128,
            user_pw=user_pw,
            owner_pw=owner_pw,
            permissions=int(pymupdf.PDF_PERM_PRINT),
            no_new_id=True,
        )


# --------------------------------------------------------------------------------------------------------------------
# rasterization and image containers
# --------------------------------------------------------------------------------------------------------------------
def page_images(pdf: bytes, dpi: int, pages: Sequence[int] | None = None) -> Iterator[Image.Image]:
    """Rasterize pages (0-based indices, default all) of a PDF to RGB images."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for index in pages if pages is not None else range(doc.page_count):
            pix = doc[index].get_pixmap(dpi=dpi, alpha=False)
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@dataclass(frozen=True)
class ScanPage:
    jpeg: bytes
    width_pt: float
    height_pt: float


def image_pdf(pages: Sequence[ScanPage]) -> bytes:
    """A PDF whose pages are nothing but scanned images (no text layer)."""
    doc = pymupdf.open()
    for scan in pages:
        page = doc.new_page(width=scan.width_pt, height=scan.height_pt)
        page.insert_image(page.rect, stream=scan.jpeg)
    data = pdf_bytes(doc)
    doc.close()
    return data


def mixed_pdf(digital: bytes, scans: dict[int, ScanPage]) -> bytes:
    """``digital`` with the given 0-based pages replaced by scanned images (per-page OCR fallback test)."""
    out = pymupdf.open()
    with pymupdf.open(stream=digital, filetype="pdf") as src:
        for index in range(src.page_count):
            if index in scans:
                scan = scans[index]
                page = out.new_page(width=scan.width_pt, height=scan.height_pt)
                page.insert_image(page.rect, stream=scan.jpeg)
            else:
                out.insert_pdf(src, from_page=index, to_page=index)
    data = pdf_bytes(out)
    out.close()
    return data


def tiff_bytes(
    images: Sequence[Image.Image], *, dpi: int, gray: bool = False, compression: str = "tiff_lzw", quality: int = 75
) -> bytes:
    """Multi-frame TIFF, lossless (LZW) or JPEG-compressed like the output of a document scanner."""
    frames = [im.convert("L") if gray else im.convert("RGB") for im in images]
    options: dict[str, int] = {"quality": quality} if compression == "jpeg" else {}
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="TIFF",
        save_all=True,
        append_images=frames[1:],
        compression=compression,
        dpi=(dpi, dpi),
        **options,
    )
    return buffer.getvalue()


def png_bytes(image: Image.Image, *, dpi: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", dpi=(dpi, dpi))
    return buffer.getvalue()


# --------------------------------------------------------------------------------------------------------------------
# text-layer repair
# --------------------------------------------------------------------------------------------------------------------
LIGATURES = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "st", 0xFB06: "st"}
_BFCHAR_BLOCK = re.compile(r"beginbfchar(.*?)endbfchar", re.S)
_BFRANGE_BLOCK = re.compile(r"beginbfrange(.*?)endbfrange", re.S)
_PAIR = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
_TRIPLE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]{4})>")


def _parse_to_unicode(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for block in _BFCHAR_BLOCK.findall(text):
        for src, dst in _PAIR.findall(block):
            mapping[int(src, 16)] = bytes.fromhex(dst).decode("utf-16-be")
    for block in _BFRANGE_BLOCK.findall(text):
        for first, last, dst in _TRIPLE.findall(block):
            base = int(dst, 16)
            for offset, code in enumerate(range(int(first, 16), int(last, 16) + 1)):
                mapping[code] = chr(base + offset)
    return mapping


def _write_to_unicode(mapping: dict[int, str]) -> bytes:
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    items = sorted(mapping.items())
    for i in range(0, len(items), 100):
        chunk = items[i : i + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        lines += [f"<{code:04X}> <{text.encode('utf-16-be').hex().upper()}>" for code, text in chunk]
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return ("\n".join(lines) + "\n").encode("ascii")


def restore_ligature_text(doc: pymupdf.Document) -> int:
    """Make text extraction return ``ffi`` for the ligature glyph MuPDF's HTML engine uses for it.

    The glyphs stay (as in any word-processor PDF); only the fonts' ToUnicode maps change, so ``office`` is extracted
    as ``office`` rather than ``o\ufb03ce``. Returns the number of repaired maps.
    """
    fixed = 0
    for xref in range(1, doc.xref_length()):
        kind, value = doc.xref_get_key(xref, "ToUnicode")
        if kind != "xref":
            continue
        cmap = int(value.split()[0])
        mapping = _parse_to_unicode(doc.xref_stream(cmap).decode("latin-1"))
        if any(ord(ch) in LIGATURES for text in mapping.values() for ch in text):
            repaired = {code: "".join(LIGATURES.get(ord(ch), ch) for ch in text) for code, text in mapping.items()}
            doc.update_stream(cmap, _write_to_unicode(repaired))
            fixed += 1
    return fixed
