"""Small query helpers shared by the list endpoints (ORM expressions with bound parameters only)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement, UnaryExpression
from sqlalchemy.orm import QueryableAttribute


def like_pattern(value: str) -> str:
    """Substring pattern for ``LIKE ... ESCAPE '\\'`` in which ``%``, ``_`` and ``\\`` in ``value`` are literal."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def contains(column: QueryableAttribute[Any], value: str) -> ColumnElement[bool]:
    """Case-insensitive substring match (``ILIKE`` on PostgreSQL, ``lower() LIKE lower()`` elsewhere)."""
    return column.ilike(like_pattern(value), escape="\\")


def sort_order(model: Any, sort: str) -> UnaryExpression[Any]:
    """``"-total"`` -> ``total DESC NULLS LAST``; ``"total"`` -> ``total ASC NULLS LAST``. ``sort`` must already be
    one of the route's allowed values (they are validated as a ``Literal``)."""
    column = getattr(model, sort.removeprefix("-"))
    return column.desc().nulls_last() if sort.startswith("-") else column.asc().nulls_last()
