"""Custom column types."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import LargeBinary
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine


class EmbeddingVector(TypeDecorator[np.ndarray]):
    """A fixed-size float32 vector: pgvector ``vector(dim)`` on PostgreSQL, raw float32 bytes elsewhere.

    Values are accepted as any float sequence and returned as ``numpy.ndarray`` (float32).
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.dim,):
            raise ValueError(f"embedding must have shape ({self.dim},), got {arr.shape}")
        if dialect.name == "postgresql":
            return arr
        return arr.tobytes()

    def process_result_value(self, value: Any, dialect: Dialect) -> np.ndarray | None:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype=np.float32)
        return np.frombuffer(value, dtype=np.float32).copy()
