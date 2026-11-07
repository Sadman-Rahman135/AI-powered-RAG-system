"""Extraction service: turn one parsed document into its invoice records.

PDFs and images
    The pages are split into invoices (``segment_pages``) and every segment yields exactly one record, even when
    nothing could be read (validation then flags the missing fields). Per segment the deterministic rules run on
    the segment's layout lines. In ``llm`` / ``hybrid`` mode with a provider, the segment text (pages separated by
    form feeds) also goes to the LLM - segments in parallel on ``LLM_CONCURRENCY`` threads, results kept in segment
    order - and the reply is grounded against the segment text. ``hybrid`` merges the grounded LLM result with the
    rules result (with the segment text, for the merge's verbatim and item-table checks); ``llm`` takes the grounded
    LLM result. A segment without any text is not sent to the LLM. How much text one request carries follows the
    model's context window (``llm.budget``).
LLM failures
    Never fail the document: the rules result is used and ``ExtractionWarning.LLM_FAILED`` / ``LLM_REFUSED`` is
    added to its warnings; without a provider ``LLM_UNAVAILABLE`` is (validation turns all three into an
    ``LLM_UNAVAILABLE`` info issue).
Spreadsheets
    Mapped deterministically by ``records_from_sheets`` (ledger rows and invoice-layout sheets); never sent to an
    LLM. A sheet record's page range is the sheet's 1-based position, which is how its text is stored.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.domain.schemas import ExtractionResult, ExtractionWarning, SourceKind
from app.extraction.grounding import ground_result
from app.extraction.llm.base import LLMError, LLMProvider, LLMRefusalError
from app.extraction.llm.extractor import LLMExtractor
from app.extraction.merge import merge_results
from app.extraction.rules import extract_invoice_rules
from app.extraction.segment import Segment, segment_pages
from app.extraction.spreadsheet_mapper import records_from_sheets, render_cell
from app.ingestion.types import Page, ParsedDocument, ProgressCallback, Sheet

logger = logging.getLogger(__name__)

ExtractionMode = Literal["rules", "llm", "hybrid"]
PROGRESS_STAGE = "extracting"

__all__ = ["PROGRESS_STAGE", "ExtractedRecord", "ExtractionMode", "extract_records"]


@dataclass(slots=True)
class ExtractedRecord:
    """One invoice record and where it came from.

    ``page_start``/``page_end`` are the PDF pages (or the sheet's position) it was read from, ``sheet_name`` and
    ``row_number`` (1-based, as Excel shows it) locate spreadsheet records, ``text`` is the source text the record
    was read from (the segment's pages separated by form feeds, the ledger row, or the invoice-layout sheet).
    """

    result: ExtractionResult
    source_kind: SourceKind
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    text: str = ""


def extract_records(
    doc: ParsedDocument,
    *,
    mode: ExtractionMode,
    provider: LLMProvider | None,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    diagnostics: list[str] | None = None,
) -> list[ExtractedRecord]:
    """The invoice records of ``doc`` in document order (see the module docstring).

    ``on_progress(done, total, "extracting")`` is called as segments finish (once for spreadsheets).
    ``diagnostics``, when given, receives short count-only codes about the document (e.g. the spreadsheet
    mapper's ``"ledger_rows_skipped:<sheet index>:<n>"``) - never document text.
    """
    if doc.pages:
        return _page_records(doc.pages, mode, provider, settings, on_progress)
    records = _sheet_records(doc.sheets, settings, diagnostics)
    if on_progress is not None:
        on_progress(len(doc.sheets), len(doc.sheets), PROGRESS_STAGE)
    return records


# ----------------------------------------------------------------------------------------------- sheets


def _row_text(sheet: Sheet, row_number: int) -> str:
    if not 1 <= row_number <= len(sheet.rows):
        return ""
    return "\t".join(render_cell(value) for value in sheet.rows[row_number - 1]).rstrip()


def _sheet_records(
    sheets: Sequence[Sheet], settings: Settings, diagnostics: list[str] | None
) -> list[ExtractedRecord]:
    by_name: dict[str, Sheet] = {}
    for each in sheets:
        by_name.setdefault(each.name, each)
    out: list[ExtractedRecord] = []
    for record in records_from_sheets(
        sheets,
        default_dollar=settings.default_dollar_currency,
        date_order=settings.date_order,
        diagnostics=diagnostics,
    ):
        sheet = by_name.get(record.sheet_name)
        number = sheet.index + 1 if sheet is not None else None
        if sheet is None:
            text = ""
        elif record.row_number is not None:
            text = _row_text(sheet, record.row_number)
        else:
            text = sheet.text
        out.append(
            ExtractedRecord(
                result=record.result,
                source_kind=record.source_kind,
                page_start=number,
                page_end=number,
                sheet_name=record.sheet_name,
                row_number=record.row_number,
                text=text,
            )
        )
    return out


# ------------------------------------------------------------------------------------------------ pages


def _segment_text(segment: Segment) -> str:
    return "\f".join(page.text for page in segment.pages)


def _has_text(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _rules(segment: Segment, settings: Settings) -> ExtractionResult:
    lines = [line for page in segment.pages for line in page.lines]
    page_numbers = [page.number for page in segment.pages for _ in page.lines]
    return extract_invoice_rules(
        lines,
        default_dollar=settings.default_dollar_currency,
        date_order=settings.date_order,
        page_numbers=page_numbers,
    )


def _with_warning(result: ExtractionResult, warning: ExtractionWarning) -> ExtractionResult:
    """A copy of ``result`` that carries ``warning`` (inputs are never mutated)."""
    copy = result.model_copy(deep=True)
    if warning.value not in copy.warnings:
        copy.warnings.append(warning.value)
    return copy


def _llm_read(
    extractor: LLMExtractor, text: str, segment: Segment, default_dollar: str
) -> ExtractionResult | ExtractionWarning:
    """The grounded LLM result for one segment, or the warning that replaces it when the LLM path fails.

    Runs on a pool thread; every failure is turned into a warning here so that one segment cannot sink the others.
    """
    try:
        llm = extractor.extract(text, page_start=segment.page_start)
        return ground_result(llm, text, default_dollar=default_dollar)
    except LLMRefusalError:
        logger.warning("llm declined pages %d-%d; using the rules result", segment.page_start, segment.page_end)
        return ExtractionWarning.LLM_REFUSED
    except LLMError as exc:  # messages carry status codes / error types only (see llm.base)
        logger.warning(
            "llm failed on pages %d-%d (%s, transient=%s); using the rules result",
            segment.page_start,
            segment.page_end,
            exc,
            exc.transient,
        )
        return ExtractionWarning.LLM_FAILED
    except Exception as exc:  # an unexpected error on the LLM path must not fail the document either
        logger.error(
            "llm extraction raised %s on pages %d-%d; using the rules result",
            type(exc).__name__,
            segment.page_start,
            segment.page_end,
        )
        return ExtractionWarning.LLM_FAILED


def _combine(
    mode: ExtractionMode,
    rules: ExtractionResult,
    llm: ExtractionResult | ExtractionWarning,
    text: str,
    settings: Settings,
) -> ExtractionResult:
    if isinstance(llm, ExtractionWarning):
        return _with_warning(rules, llm)
    if mode == "llm":
        return llm
    return merge_results(
        rules,
        llm,
        tolerance=settings.amount_tolerance,
        review_threshold=settings.review_confidence_threshold,
        source=text,  # the text both sides read
        default_dollar=settings.default_dollar_currency,
    )


def _page_records(
    pages: Sequence[Page],
    mode: ExtractionMode,
    provider: LLMProvider | None,
    settings: Settings,
    on_progress: ProgressCallback | None,
) -> list[ExtractedRecord]:
    segments = segment_pages(pages)
    texts = [_segment_text(segment) for segment in segments]
    rules = [_rules(segment, settings) for segment in segments]
    results = list(rules)
    readable = [i for i, text in enumerate(texts) if _has_text(text)]
    total = len(segments)

    if mode != "rules" and provider is None:
        for i in readable:
            results[i] = _with_warning(rules[i], ExtractionWarning.LLM_UNAVAILABLE)
    elif mode != "rules" and readable:
        assert provider is not None
        extractor = LLMExtractor(provider, default_dollar=settings.default_dollar_currency)
        done = total - len(readable)
        workers = max(1, min(settings.llm_concurrency, len(readable)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="docintel-llm") as pool:
            futures = {
                pool.submit(_llm_read, extractor, texts[i], segments[i], settings.default_dollar_currency): i
                for i in readable
            }
            for future in as_completed(futures):
                i = futures[future]
                results[i] = _combine(mode, rules[i], future.result(), texts[i], settings)
                done += 1
                if on_progress is not None:
                    on_progress(done, total, PROGRESS_STAGE)
    if on_progress is not None and (mode == "rules" or provider is None or not readable):
        on_progress(total, total, PROGRESS_STAGE)

    return [
        ExtractedRecord(
            result=result,
            source_kind=SourceKind.INVOICE_DOCUMENT,
            page_start=segment.page_start,
            page_end=segment.page_end,
            text=text,
        )
        for segment, text, result in zip(segments, texts, results, strict=True)
    ]
