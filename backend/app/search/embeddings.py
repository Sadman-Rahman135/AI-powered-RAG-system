"""Text embedders behind one small protocol.

``FastEmbedEmbedder`` runs an ONNX sentence-embedding model in-process (production). ``HashingEmbedder`` is a
dependency-free, deterministic stand-in for tests and offline use: signed feature hashing of word tokens and
character 3-grams. Both return L2-normalised ``list[float]`` vectors, so cosine similarity is a dot product.

Stored embeddings must stay comparable with query vectors computed by a later process, so the hashing embedder
never uses Python's ``hash()`` (salted per process) and its scheme is pinned by a golden-vector test.
"""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.config import get_settings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_WORD_RE = re.compile(r"[^\W_]+")
_WORD_WEIGHT = 1.0
_GRAM_WEIGHT = 0.5
_GRAM_SIZE = 3


@runtime_checkable
class Embedder(Protocol):
    """Turns text into fixed-size unit vectors. Documents and queries may use different encoders."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _unit(vector: Any) -> list[float]:
    """L2-normalise as plain Python floats; an all-zero vector stays zero (never NaN)."""
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return (array / norm if norm > 0.0 else array).tolist()


@lru_cache(maxsize=1 << 18)
def _bucket_and_sign(feature: str, dim: int) -> tuple[int, float]:
    """Stable (bucket, sign) for a feature string: blake2b is identical in every process and Python version."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % dim, 1.0 if digest[4] & 1 else -1.0


def _features(text: str) -> Iterator[tuple[str, float]]:
    """Weighted features of a text: each lowercase word plus the 3-grams of the word padded with ``#``."""
    for word in _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold()):
        yield f"w:{word}", _WORD_WEIGHT
        padded = f"#{word}#"
        for start in range(len(padded) - _GRAM_SIZE + 1):
            yield f"g:{padded[start : start + _GRAM_SIZE]}", _GRAM_WEIGHT


class HashingEmbedder:
    """Deterministic bag-of-subwords embedder: related texts share words and character 3-grams, so they land
    close together; unrelated texts are close to orthogonal. Text without any word yields the zero vector."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _embed(self, text: str) -> list[float]:
        buckets: list[int] = []
        weights: list[float] = []
        for feature, weight in _features(text):
            bucket, sign = _bucket_and_sign(feature, self.dim)
            buckets.append(bucket)
            weights.append(sign * weight)
        if not buckets:
            return [0.0] * self.dim
        return _unit(np.bincount(buckets, weights=weights, minlength=self.dim))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    """fastembed ``TextEmbedding`` (ONNX, CPU). The model is loaded on first use, once per instance, so
    constructing the embedder is free and never touches the network.

    ``embed_documents`` embeds the texts in order of length, in small batches, and returns the vectors in input
    order. Every ONNX batch is padded to its longest text, and page windows range from a few words to
    ``CHUNK_MAX_CHARS``, so batching mixed lengths spent most of the time on padding. Measured on a 200-page digital
    PDF (314 page windows of 117-458 tokens, 6-core CPU): embedding took 46.5-47.3 s in unsorted batches of 64 and
    22.6-23.1 s in length-sorted batches of 4 (8: 25.5 s, 16: 27.0 s; 1-2: 17-24 s, less steady), with bit-identical
    vectors; the job's whole indexing stage went from 55.7 s to 29.5 s. ONNX Runtime's own thread default (one per
    physical core) was the fastest setting tried; one thread per logical core was slower.

    Args:
        model: fastembed model name (``EMBEDDING_MODEL``).
        cache_dir: where model files are stored (``EMBEDDING_CACHE_DIR``); fastembed's default when ``None``.
        local_files_only: never download; fail if the model is not already cached (tests, air-gapped hosts).
        batch_size: texts per ONNX batch (of similar length, see above).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        batch_size: int = 4,
    ) -> None:
        self.name = model
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self._local_files_only = local_files_only
        self._batch_size = batch_size
        self._model: Any = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        """Embedding size from fastembed's model registry (no model load needed)."""
        if self._dim is None:
            from fastembed import TextEmbedding

            self._dim = int(TextEmbedding.get_embedding_size(self.name))
        return self._dim

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self.name, cache_dir=self.cache_dir, local_files_only=self._local_files_only
                    )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))  # see the class docstring
        vectors = self._load().embed([texts[index] for index in order], batch_size=self._batch_size)
        result: list[list[float]] = [[] for _ in texts]
        for index, vector in zip(order, vectors, strict=True):
            result[index] = _unit(vector)
        return result

    def embed_query(self, text: str) -> list[float]:
        return _unit(next(iter(self._load().query_embed(text))))


@lru_cache(maxsize=8)
def _build_embedder(provider: str, model: str, dim: int, cache_dir: str | None) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim)
    embedder = FastEmbedEmbedder(model, cache_dir=cache_dir)
    if embedder.dim != dim:
        raise ValueError(f"EMBEDDING_DIM={dim} does not match model {model!r}, which produces {embedder.dim}-d vectors")
    return embedder


def get_embedder() -> Embedder:
    """The configured embedder (``EMBEDDING_PROVIDER``/``EMBEDDING_MODEL``/``EMBEDDING_CACHE_DIR``), one cached
    instance per configuration so the ONNX model is loaded once per process."""
    settings = get_settings()
    cache_dir = None if settings.embedding_cache_dir is None else str(settings.embedding_cache_dir)
    return _build_embedder(settings.embedding_provider, settings.embedding_model, settings.embedding_dim, cache_dir)


__all__ = ["DEFAULT_MODEL", "Embedder", "FastEmbedEmbedder", "HashingEmbedder", "get_embedder"]
