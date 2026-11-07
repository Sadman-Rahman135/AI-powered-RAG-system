"""Embedders (deterministic hashing embedder, lazy fastembed wrapper, cached factory)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config import reset_settings_cache
from app.search import embeddings
from app.search.embeddings import FastEmbedEmbedder, HashingEmbedder, get_embedder

BACKEND_DIR = Path(__file__).resolve().parents[3]


def _cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hashing_embedder_deterministic_normalized() -> None:
    embedder = HashingEmbedder()
    assert embedder.dim == 384
    assert embedder.name == "hashing-384"

    text = "Invoice INV-2024-001 from Acme Corp for office chairs"
    first = embedder.embed_query(text)
    assert first == embedder.embed_query(text)
    assert len(first) == 384
    assert all(isinstance(x, float) for x in first)
    assert math.isclose(float(np.linalg.norm(first)), 1.0, abs_tol=1e-6)

    # documents and queries share one representation; case does not matter
    assert embedder.embed_documents([text, "second"])[0] == first
    assert embedder.embed_query(text.upper()) == first
    assert len(embedder.embed_documents(["a", "b", "c"])) == 3
    assert embedder.embed_documents([]) == []

    related = embedder.embed_query("office chair invoice from Acme")
    unrelated = embedder.embed_query("monthly cloud hosting subscription servers")
    assert _cosine(first, related) > _cosine(first, unrelated) + 0.2


def test_hashing_embedder_is_stable_across_processes() -> None:
    """Python's per-process string hash randomisation must not leak into stored vectors."""
    code = (
        "import json; from app.search.embeddings import HashingEmbedder; "
        "print(json.dumps(HashingEmbedder().embed_query('Invoice INV-2024-001 from Acme Corp')))"
    )
    vectors = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=BACKEND_DIR, env=env, capture_output=True, text=True, check=True
        )
        vectors.append(json.loads(out.stdout))
    assert vectors[0] == vectors[1]
    assert vectors[0] == HashingEmbedder().embed_query("Invoice INV-2024-001 from Acme Corp")


def test_hashing_embedder_golden_vector() -> None:
    """Pins the feature-hashing scheme: stored embeddings must stay comparable with future query vectors.

    Values were derived independently from the documented scheme (blake2b-8 of ``w:<word>`` with weight 1 and
    ``g:<3-gram of #word#>`` with weight 0.5; bucket = first 4 bytes LE mod dim, sign = low bit of byte 4).
    """
    assert HashingEmbedder(dim=8).embed_query("invoice") == pytest.approx(
        [0.0, -0.447214, 0.0, -0.447214, -0.447214, 0.447214, -0.447214, 0.0], abs=1e-6
    )
    expected = [0.182574, 0.0, 0.0, -0.365148, 0.547723, -0.365148, 0.0, -0.182574]
    expected += [0.182574, 0.0, 0.182574, 0.0, -0.365148, 0.0, 0.182574, 0.365148]
    assert HashingEmbedder(dim=16).embed_query("Acme Corp INV-2024-001") == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("text", ["", "   \n\t", "!!! --- ???"])
def test_hashing_embedder_empty_text_is_zero_vector(text: str) -> None:
    vector = HashingEmbedder(dim=16).embed_query(text)
    assert vector == [0.0] * 16
    assert not any(math.isnan(x) for x in vector)
    assert HashingEmbedder(dim=16).embed_documents([text]) == [[0.0] * 16]


def test_hashing_embedder_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(dim=0)


def test_fastembed_embedder_dim() -> None:
    embedder = FastEmbedEmbedder(local_files_only=True)
    assert embedder.dim == 384  # static lookup, no model needed
    assert embedder.name == "BAAI/bge-small-en-v1.5"
    try:
        query = embedder.embed_query("office furniture")
        docs = embedder.embed_documents(["Ergonomic office chairs and standing desks", "Cloud hosting subscription"])
    except Exception as exc:  # any load failure (model not cached, no network) means: skip
        pytest.skip(f"fastembed model unavailable offline: {type(exc).__name__}")
    assert len(query) == 384 and all(len(d) == 384 for d in docs)
    for vector in (query, *docs):
        assert math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-3)
    assert _cosine(query, docs[0]) > _cosine(query, docs[1])


class _FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: records construction and returns un-normalised vectors."""

    instances: list[_FakeTextEmbedding] = []  # noqa: RUF012 - test double

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.calls: list[tuple[str, list[str]]] = []
        self.batch_sizes: list[int] = []
        _FakeTextEmbedding.instances.append(self)

    @classmethod
    def get_embedding_size(cls, _model_name: str) -> int:
        return 4

    def embed(self, texts: list[str], batch_size: int = 256) -> Any:
        self.calls.append(("embed", list(texts)))
        self.batch_sizes.append(batch_size)
        return (np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32) for _ in texts)

    def query_embed(self, query: str) -> Any:
        self.calls.append(("query_embed", [query]))
        return iter([np.array([0.0, 0.0, 6.0, 8.0], dtype=np.float32)])


def test_fastembed_embedder_is_lazy_and_uses_document_and_query_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fastembed

    _FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)

    embedder = FastEmbedEmbedder("some/model", cache_dir=tmp_path / "models", local_files_only=True)
    assert embedder.dim == 4
    assert _FakeTextEmbedding.instances == []  # nothing loaded until the first embedding is requested

    docs = embedder.embed_documents(["one", "two"])
    assert docs == [pytest.approx([0.6, 0.8, 0.0, 0.0])] * 2  # L2-normalised plain lists
    assert all(isinstance(x, float) for x in docs[0])
    assert embedder.embed_documents([]) == []

    query = embedder.embed_query("question")
    assert query == pytest.approx([0.0, 0.0, 0.6, 0.8])

    (model,) = _FakeTextEmbedding.instances  # loaded once, reused
    assert model.model_name == "some/model"
    assert model.kwargs["cache_dir"] == str(tmp_path / "models")
    assert model.kwargs["local_files_only"] is True
    assert model.calls == [("embed", ["one", "two"]), ("query_embed", ["question"])]
    assert model.batch_sizes == [4]  # small batches of similar length (see FastEmbedEmbedder)


class _LengthEmbedding(_FakeTextEmbedding):
    """A fake whose vector says which text it was computed from: [len(text), 1, 0, 0]."""

    def embed(self, texts: list[str], batch_size: int = 256) -> Any:
        self.calls.append(("embed", list(texts)))
        self.batch_sizes.append(batch_size)
        return (np.array([float(len(text)), 1.0, 0.0, 0.0], dtype=np.float32) for text in texts)


def test_fastembed_embedder_embeds_shortest_first_and_answers_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastembed

    _FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(fastembed, "TextEmbedding", _LengthEmbedding)
    texts = ["a fairly long page window", "short", "mid length", "x", "short"]

    vectors = FastEmbedEmbedder("some/model", local_files_only=True).embed_documents(texts)

    (model,) = _FakeTextEmbedding.instances
    assert model.calls == [("embed", ["x", "short", "short", "mid length", "a fairly long page window"])]
    lengths = [round(v[0] / v[1]) for v in vectors]  # undo the normalisation: len(text) / 1
    assert lengths == [len(text) for text in texts]  # every vector back at its text's position


def test_get_embedder_is_hashing_under_test_settings_and_cached() -> None:
    first = get_embedder()
    assert isinstance(first, HashingEmbedder)
    assert first.dim == 384
    assert get_embedder() is first


def test_get_embedder_caches_one_instance_per_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    reset_settings_cache()
    base = get_embedder()
    assert isinstance(base, FastEmbedEmbedder)
    assert base.name == "BAAI/bge-small-en-v1.5"
    assert base.cache_dir == str(tmp_path / "cache")
    assert get_embedder() is base

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    reset_settings_cache()
    other = get_embedder()
    assert isinstance(other, FastEmbedEmbedder)
    assert other is not base and other.name == "BAAI/bge-base-en-v1.5" and other.dim == 768
    assert get_embedder() is other

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    reset_settings_cache()
    hashing = get_embedder()
    assert isinstance(hashing, HashingEmbedder) and hashing.dim == 768


def test_get_embedder_rejects_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("EMBEDDING_DIM", "128")  # bge-small produces 384
    reset_settings_cache()
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        get_embedder()


def test_embedder_module_exposes_protocol() -> None:
    assert isinstance(HashingEmbedder(), embeddings.Embedder)
