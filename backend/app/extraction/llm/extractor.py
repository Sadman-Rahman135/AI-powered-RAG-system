"""LLM extraction: a model copies values out of the document; our code normalises and verifies every one of them.

Per field the model returns the value exactly as printed (or null). This module then

1. cleans the string (U+FFFD from a local server's broken multi-byte output is dropped, blanks and "N/A" are null),
2. parses it with the same parsers the rules extractor uses (``parse_money`` / ``parse_number`` /
   ``parse_percentage`` with the document's decimal separator, ``parse_date`` with the document's day/month order,
   ``detect_currency``, ``clean_text_value``),
3. requires that the copied string really occurs in the source text (the "copy exactly" contract), and
4. runs ``ground_result``, which requires the parsed value itself to be supported by the text.

A numeric date the document leaves ambiguous ("05/03/2024") is read with the same regional convention the rules use
(``SourceIndex.order_prior``) and flagged. An invoice number that is only a short fragment of a longer printed number
is kept but scored below the review threshold. The document is cut to the input budget, which follows the model's
context window (``llm.budget``). When the answer runs out of output tokens (a long items table), the request is
retried once without line items, so the header fields survive and the rules supply the items.

A value that fails any step is ``None``; the original goes to ``result.rejected`` and the field keeps a
``FieldMeta`` with confidence 0 and the ``ungrounded_value_rejected`` note. Nothing a model says reaches the record
unless the document says it too. ``FieldMeta.evidence`` is the copied string.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.domain.schemas import (
    DATE_FIELDS,
    MONEY_FIELDS,
    DocumentType,
    ExtractionResult,
    ExtractionSource,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    LineItem,
    MetaNote,
)
from app.extraction.grounding import SourceIndex, Support, amount_core, ground_result, plain_number_core
from app.extraction.llm.base import LLMError, LLMOutputTruncatedError, LLMProvider
from app.extraction.llm.budget import input_budget
from app.extraction.llm.prompt import (
    HEADER_ONLY_SCHEMA,
    LINE_ITEM_FIELDS,
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    SCALAR_FIELDS,
    SYSTEM_PROMPT,
    mark_pages,
    wrap_document,
)
from app.extraction.normalize import (
    ISO_CURRENCIES,
    clean_text_value,
    detect_currency,
    parse_date,
    parse_money,
    parse_number,
    parse_percentage,
)

logger = logging.getLogger(__name__)

EXACT_CONFIDENCE = 0.85  # the copied string occurs in the document as printed
TOLERANT_CONFIDENCE = 0.75  # found only through OCR-tolerant or normalised matching
PARTIAL_CONFIDENCE = 0.5  # a short fragment of a longer printed number: kept, but below the review threshold
CLASSIFICATION_CONFIDENCE = 0.75  # document_type is a judgement, not a copy
HEAD_SHARE = 0.7  # of an over-long document's budget goes to its beginning, the rest to its end

_NULL_WORDS = frozenset({"", "null", "none", "n/a", "na", "nil", "-", "—", "–"})
_NAME_FIELDS = ("supplier_name", "customer_name")
_CONFIDENCE: dict[Support, float] = {
    "exact": EXACT_CONFIDENCE,
    "tolerant": TOLERANT_CONFIDENCE,
    "partial": PARTIAL_CONFIDENCE,
}


def _clean(raw: object) -> str | None:
    """The model's string for a field with U+FFFD and extra whitespace removed; ``None`` for null-like values."""
    if isinstance(raw, bool) or not isinstance(raw, str | int | float):
        return None
    text = " ".join(str(raw).replace("\N{REPLACEMENT CHARACTER}", "").split())
    return None if text.casefold() in _NULL_WORDS else text


def _rejected_meta() -> FieldMeta:
    return FieldMeta(confidence=0.0, source=ExtractionSource.LLM, notes=[MetaNote.UNGROUNDED_VALUE_REJECTED.value])


@dataclass(frozen=True, slots=True)
class _Parsed:
    value: Any
    tier: Support
    anchor: str  # a snippet of the document that locates the value (for its page number)
    notes: tuple[MetaNote, ...] = ()
    confidence_cap: float = 1.0  # a flag and its score must agree: an inferred value is not a confident one


def _parse_scalar(name: str, text: str, index: SourceIndex) -> _Parsed | None:
    """Normalise ``text`` for field ``name`` and require it to occur in the document; ``None`` if either fails."""
    if name == "invoice_number":
        number = clean_text_value(text)
        support = index.id_support(number) if number else None
        return _Parsed(number, support, number) if number and support else None
    if name in _NAME_FIELDS:
        party = clean_text_value(text)
        support = index.text_support(party) if party else None
        return _Parsed(party, support, party) if party and support else None
    if name in MONEY_FIELDS:
        amount = parse_money(text, index.decimal_hint)
        core = amount_core(text, index.decimal_hint)
        if amount is None or core is None or not index.occurs_number(core):
            return None
        return _Parsed(amount, "exact", core)
    if name == "tax_rate":
        rate = parse_percentage(text)
        if rate is None:
            rate = parse_number(text, index.decimal_hint)  # "19" for a printed "19 %"
        core = plain_number_core(text)
        if rate is None or core is None or not index.occurs_number(core):
            return None
        return _Parsed(rate, "exact", core)
    if name in DATE_FIELDS:
        parsed = parse_date(text, index.date_order or "auto")
        if parsed is None or not index.occurs_literal(parsed.raw):
            return None
        if parsed.ambiguous:  # the day/month order is a guess, so the copy is only tolerantly supported
            value = parsed.value
            if index.order_prior == "DMY":  # the same regional reading as the rules (parse_date defaults to MDY)
                day_first = parse_date(parsed.raw, "DMY")
                value = day_first.value if day_first is not None else value
            return _Parsed(value, "tolerant", parsed.raw, (MetaNote.AMBIGUOUS_DATE_ORDER,))
        return _Parsed(parsed.value, "exact", parsed.raw)
    guess = detect_currency(text, index.default_dollar)  # currency
    code: str | None
    if guess is not None:
        code, marker, cap = guess.code, guess.raw, guess.confidence  # 0.5 for a bare "$", as for the rules
    else:  # a bare ISO code such as "CAD" that detect_currency only trusts next to an amount; grounding checks that
        code, marker, cap = (text.upper() if text.upper() in ISO_CURRENCIES else None), text, 1.0
    if code is None or not index.occurs_currency(marker):
        return None
    return _Parsed(code, "exact", marker, confidence_cap=cap)


def _document_type(raw: object) -> DocumentType:
    if not isinstance(raw, str):
        return DocumentType.UNKNOWN
    try:
        return DocumentType(raw.strip().lower().replace(" ", "_").replace("-", "_"))
    except ValueError:
        return DocumentType.UNKNOWN


def _parse_items(raw_items: object, index: SourceIndex) -> tuple[list[LineItem], list[Any]]:
    """The model's rows as line items, plus whatever was not a usable row (kept for transparency)."""
    if raw_items is None:
        return [], []
    if not isinstance(raw_items, list):
        return [], [raw_items]
    items: list[LineItem] = []
    unusable: list[Any] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            unusable.append(raw)
            continue
        item = _parse_item(raw, index)
        if item is not None:
            items.append(item)
    return items, unusable


def _parse_item(raw: dict[str, Any], index: SourceIndex) -> LineItem | None:
    """A line item from the model's verbatim cells; ``None`` for a row with no cell filled in."""
    cells = {name: _clean(raw.get(name)) for name in LINE_ITEM_FIELDS}
    if all(cell is None for cell in cells.values()):
        return None
    quantity, unit_price, amount = cells["quantity"], cells["unit_price"], cells["amount"]
    return LineItem(
        description=clean_text_value(cells["description"]),
        quantity=parse_number(quantity, index.decimal_hint) if quantity else None,
        unit_price=parse_money(unit_price, index.decimal_hint) if unit_price else None,
        amount=parse_money(amount, index.decimal_hint) if amount else None,
    )


def _page_of(index: SourceIndex, anchors: tuple[str, ...], page_start: int | None) -> int | None:
    if page_start is None:
        return None
    for anchor in anchors:
        position = index.locate(anchor)
        if position is not None:
            return page_start + index.text.count("\f", 0, position)
    return None


def _fit_to_budget(text: str, page_start: int | None, budget: int) -> tuple[str, bool]:
    """The page-marked text for the prompt and whether it was cut to ``budget`` characters.

    A longer document keeps its first 70 % and last 30 % of the budget around an explicit marker; page numbers of
    the kept tail stay right.
    """
    if len(text) <= budget:
        return mark_pages(text, page_start), False
    head_chars = int(budget * HEAD_SHARE)
    tail_start = len(text) - (budget - head_chars)
    tail_page = None if page_start is None else page_start + text.count("\f", 0, tail_start)
    head = mark_pages(text[:head_chars], page_start)
    tail = mark_pages(text[tail_start:], tail_page)
    return f"{head}\n[... {tail_start - head_chars} characters omitted ...]\n{tail}", True


class LLMExtractor:
    """Extract an invoice record from document text with an ``LLMProvider``; stateless and thread-safe.

    Pages of ``text`` are separated by form feeds (``\\f``); with ``page_start`` the prompt carries page markers
    and each field's ``FieldMeta.page`` is set.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_input_chars: int | None = None,
        max_output_tokens: int | None = None,
        default_dollar: str | None = None,
    ) -> None:
        settings = get_settings()
        self._provider = provider
        self._max_input_chars = max_input_chars  # None: the budget follows the model (llm.budget)
        self._max_input_chars_setting = settings.llm_max_input_chars
        self._max_output_tokens = settings.llm_max_output_tokens if max_output_tokens is None else max_output_tokens
        self._default_dollar = settings.default_dollar_currency if default_dollar is None else default_dollar
        # a partial invoice-number match must stay below the review threshold, whatever it is configured to
        self._partial_confidence = min(PARTIAL_CONFIDENCE, max(0.0, settings.review_confidence_threshold - 0.05))

    def input_chars(self) -> int:
        """Characters of document text one request carries (an explicit ``max_input_chars`` wins)."""
        if self._max_input_chars is not None:
            return self._max_input_chars
        return input_budget(
            self._provider, max_input_chars=self._max_input_chars_setting, max_output_tokens=self._max_output_tokens
        ).chars

    def _ask(self, body: str, *, header_only: bool) -> Any:
        return self._provider.complete_json(
            system=SYSTEM_PROMPT,
            user=wrap_document(body, header_only=header_only),
            # a provider may mutate what it is handed; the original must survive
            schema=copy.deepcopy(HEADER_ONLY_SCHEMA if header_only else OUTPUT_SCHEMA),
            max_tokens=self._max_output_tokens,
        )

    def extract(self, text: str, *, page_start: int | None = None) -> ExtractionResult:
        """The grounded record the model read from ``text``; raises ``LLMError`` when the provider fails."""
        started = time.perf_counter()
        body, truncated = _fit_to_budget(text, page_start, self.input_chars())
        header_only = False
        try:
            reply = self._ask(body, header_only=False)
        except LLMOutputTruncatedError:
            # the answer did not fit the output budget - almost always a long items table: keep the header fields
            header_only = True
            reply = self._ask(body, header_only=True)
        if not isinstance(reply, dict):
            raise LLMError(f"{self._provider.name} reply was not a JSON object", transient=False)
        index = SourceIndex(text, default_dollar=self._default_dollar)
        result = ground_result(self._to_result(reply, index, page_start), index)
        if truncated:
            result.warnings.append(ExtractionWarning.LLM_INPUT_TRUNCATED.value)
        logger.info(
            "llm extraction: provider=%s model=%s prompt_version=%s input_chars=%d truncated=%s header_only=%s "
            "fields=%d rejected=%d latency_ms=%d",
            self._provider.name,
            self._provider.model,
            PROMPT_VERSION,
            len(text),
            truncated,
            header_only,
            sum(1 for field_meta in result.meta.values() if field_meta.confidence > 0),
            len(result.rejected),
            (time.perf_counter() - started) * 1000,
        )
        return result

    def _to_result(self, reply: dict[str, Any], index: SourceIndex, page_start: int | None) -> ExtractionResult:
        values: dict[str, Any] = {}
        meta: dict[str, FieldMeta] = {}
        rejected: dict[str, Any] = {}
        for name in SCALAR_FIELDS:
            raw = reply.get(name)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, str | int | float):
                rejected[name] = raw  # an object, list or boolean where a string belongs
                meta[name] = _rejected_meta()
                continue
            text = _clean(raw)
            if text is None:  # blank, "N/A" or only U+FFFD: a null, not an error
                continue
            parsed = _parse_scalar(name, text, index)
            if parsed is None:
                rejected[name] = text
                meta[name] = _rejected_meta()
                continue
            values[name] = parsed.value
            tier_confidence = self._partial_confidence if parsed.tier == "partial" else _CONFIDENCE[parsed.tier]
            meta[name] = FieldMeta(
                confidence=min(tier_confidence, parsed.confidence_cap),
                source=ExtractionSource.LLM,
                evidence=text,
                page=_page_of(index, (text, parsed.anchor), page_start),
                notes=[note.value for note in parsed.notes],
            )

        document_type = _document_type(reply.get("document_type"))
        if document_type is not DocumentType.UNKNOWN:
            meta["document_type"] = FieldMeta(confidence=CLASSIFICATION_CONFIDENCE, source=ExtractionSource.LLM)

        items, unusable = _parse_items(reply.get("line_items"), index)
        if unusable:
            rejected["line_items"] = unusable
        if items:
            meta["line_items"] = FieldMeta(confidence=EXACT_CONFIDENCE, source=ExtractionSource.LLM)

        return ExtractionResult(
            data=InvoiceData(**values, document_type=document_type, line_items=items),
            meta=meta,
            method="llm",
            rejected=rejected,
            llm_provider=self._provider.name,
            llm_model=self._provider.model,
            prompt_version=PROMPT_VERSION,
        )
