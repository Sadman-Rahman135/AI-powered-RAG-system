"""Hand-made labels, predictions and results documents for the evaluation metrics and report tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from app.domain.schemas import DocumentType, InvoiceData, IssueCode, LineItem
from eval.labels import LabeledFile, LabeledRecord
from eval.metrics import FileResult, PredictedRecord, compute_metrics
from eval.run_eval import dataset_summary

D = Decimal
LOCATION_KEYS = ("page_start", "page_end", "sheet_name", "row_number", "source_kind")


def truth_record(**fields: Any) -> LabeledRecord:
    """A clean one-page invoice label with ``fields`` overridden."""
    base: dict[str, Any] = {
        "invoice_number": "INV-001",
        "invoice_date": date(2024, 3, 5),
        "supplier_name": "Acme Corp.",
        "customer_name": "Globex Ltd",
        "subtotal": D("100.00"),
        "tax": D("20.00"),
        "total": D("120.00"),
        "currency": "USD",
        "document_type": DocumentType.INVOICE,
        "line_items": [LineItem(description="Widget", quantity=D("2"), unit_price=D("50.00"), amount=D("100.00"))],
        "page_start": 1,
        "page_end": 1,
    }
    base.update(fields)
    return LabeledRecord(**base)


def pred_record(
    truth: LabeledRecord | None = None, *, issues: list[str] | None = None, review: str | None = None, **fields: Any
) -> PredictedRecord:
    """A prediction equal to ``truth`` (or to the default truth) with ``fields`` overridden."""
    source = (truth or truth_record()).model_dump()
    location = {k: source.pop(k) for k in LOCATION_KEYS}
    for key in list(fields):
        if key in location:
            location[key] = fields.pop(key)
    source.update(fields)
    return PredictedRecord(
        data=InvoiceData(**source),
        issue_codes=issues or [],
        review_status=review or ("needs_review" if issues else "auto_approved"),
        **location,
    )


def labeled_file(
    name: str, records: list[LabeledRecord], *, issues: list[list[IssueCode]] | None = None, **fields: Any
) -> LabeledFile:
    """A dev-split clean PDF label (``issues`` = per-record expected codes)."""
    base: dict[str, Any] = {
        "file": name,
        "kind": "pdf",
        "template": "T1",
        "split": "dev",
        "quality": "clean",
        "expect": "success",
        "records": records,
        "record_issues": issues if issues is not None else [[] for _ in records],
        "pages": 1,
    }
    base.update(fields)
    union = sorted({c for codes in base["record_issues"] for c in codes}, key=lambda c: c.value)
    base.setdefault("expected_issue_codes", union)
    return LabeledFile(**base)


def file_result(name: str, records: list[PredictedRecord], **fields: Any) -> FileResult:
    """A completed one-page file processed in one second."""
    base: dict[str, Any] = {
        "file": name,
        "status": "completed",
        "seconds": 1.0,
        "processing_ms": 900,
        "page_count": 1,
        "records": records,
    }
    base.update(fields)
    return FileResult(**base)


def results_document(
    config: str,
    *,
    total_offset: str = "0",
    seconds: float = 1.0,
    limit: int | None = None,
    embedding: str = "fastembed",
    llm_model: str | None = None,
    scale: dict[str, Any] | None = None,
    seed: int = 42,
    dirty: bool | None = False,
    llm_fallbacks: int = 0,
    server: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A small but complete results document as ``eval.run_eval`` writes it: a dev and a held-out invoice, a
    corrupt file; ``total_offset`` shifts the predicted totals (to make configurations differ), ``llm_fallbacks``
    marks records of the dev file as LLM fallbacks, ``server`` is what an LLM server reported about itself."""
    dev, held = truth_record(invoice_number="DEV-1"), truth_record(invoice_number="HELD-1")
    labels = [
        labeled_file("pdf/dev.pdf", [dev]),
        labeled_file("images/held.jpg", [held], split="heldout", template="T4", kind="jpg", quality="photo"),
        labeled_file(
            "corrupt/broken.pdf",
            [],
            kind="corrupt",
            template=None,
            split="na",
            expect="failure",
            expected_error_code="corrupt_file",
        ),
    ]
    offset = D(total_offset)
    results = [
        file_result(
            "pdf/dev.pdf", [pred_record(dev, total=D("120.00") + offset)], seconds=seconds, llm_fallbacks=llm_fallbacks
        ),
        file_result("images/held.jpg", [pred_record(held, customer_name=None)], seconds=seconds * 4),
        file_result("corrupt/broken.pdf", [], status="failed", error_code="corrupt_file", seconds=0.1),
    ]
    files = []
    for label, res in zip(labels, results, strict=True):
        entry = res.model_dump(mode="json")
        entry.update(kind=label.kind, quality=label.quality, split=label.split, expect=label.expect)
        entry["score"] = {"core_correct": 0, "core_total": 0, "wrong_fields": [], "matched": 1, "missed": 0, "extra": 0}
        files.append(entry)
    return {
        "schema_version": 1,
        "config": {
            "name": config,
            "description": f"{config} description",
            "extraction_mode": "rules" if config == "rules" else "hybrid",
            "llm_provider": "none" if config == "rules" else "local",
            "llm_model": llm_model,
            "embedding_provider": embedding,
            "embedding_model": "BAAI/bge-small-en-v1.5" if embedding == "fastembed" else None,
            "ocr_engine": "rapidocr",
            "ocr_workers": 0,
            "env": {},
            "settings": {"embedding_provider": embedding, "ocr_workers": 0, "anthropic_api_key": None},
            **(server or {}),
        },
        "run": {
            "started_at": "2026-11-05T10:00:00+00:00",
            "finished_at": "2026-11-05T10:05:00+00:00",
            "wall_seconds": 300.0,
            "warmup_seconds": 3.0,
            "processing_seconds": seconds * 5 + 0.1,
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "branch": "feature/docintel",
            "dirty": dirty,
            "dirty_paths": 3 if dirty else (0 if dirty is False else None),
            "dataset": "eval/dataset",
            "labels_seed": seed,
            "labels_files": 164,
            "dataset_summary": dataset_summary(labels),
            "files_evaluated": len(labels) if limit is None else limit,
            "limit": limit,
            "include_scale": scale is not None,
            "scale_only": False,
            "command": f"python -m eval.run_eval --config {config}",
        },
        "machine": {
            "os": "Windows-11",
            "python": "3.12.10",
            "cpu": "Test CPU 8-Core",
            "logical_cpus": 16,
            "physical_cpus": 8,
            "ram_gb": 15.3,
            "gpu": ["Test GPU, 6144 MiB"],
        },
        "metrics": compute_metrics(labels, results),
        "files": files,
        "scale": scale,
    }
