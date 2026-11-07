"""Task-local fixtures for the validation tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from app.domain.schemas import DocumentType, InvoiceData, LineItem


def _consistent_fields() -> dict[str, Any]:
    """A complete invoice whose numbers add up: 2 x 25 + 1 x 50 = 100 subtotal, 10 % tax, 110 total."""
    return {
        "invoice_number": "INV-2024-001",
        "invoice_date": date(2024, 3, 5),
        "due_date": date(2024, 4, 4),
        "supplier_name": "Acme Corp",
        "customer_name": "Globex Ltd",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("10.00"),
        "tax_rate": Decimal("10"),
        "total": Decimal("110.00"),
        "currency": "USD",
        "document_type": DocumentType.INVOICE,
        "line_items": [
            LineItem(description="Widget", quantity=Decimal("2"), unit_price=Decimal("25.00"), amount=Decimal("50.00")),
            LineItem(description="Gadget", quantity=Decimal("1"), unit_price=Decimal("50.00"), amount=Decimal("50.00")),
        ],
    }


@pytest.fixture
def make_invoice() -> Callable[..., InvoiceData]:
    """Factory for a fully consistent invoice; keyword arguments replace individual fields."""

    def factory(**overrides: Any) -> InvoiceData:
        return InvoiceData(**{**_consistent_fields(), **overrides})

    return factory
