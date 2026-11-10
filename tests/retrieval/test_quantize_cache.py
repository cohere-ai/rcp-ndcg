"""A run that declares ``embedding_types: [int8]`` uses another index cache entry than one that does not.

The identity is the cache key: two configurations that quantize differently must never resume one another's
index, even when the served float reply is the same.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcp.retrieval import DenseConfig, ServedEmbedding, retrieve


def test_two_embedding_types_write_two_index_cache_entries(
    dataset: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_json: str,
) -> None:
    monkeypatch.setattr("rcp.support.paths.cache_dir", lambda: tmp_path / "cache")
    budget = {
        "base_url": "fake://seed/7?dim=8",
        "model": "stub",
        "tokenizer": tokenizer_json,
        "max_tokens": 8192,
    }
    plain = DenseConfig(encoder=ServedEmbedding(**budget))
    quantized = DenseConfig(encoder=ServedEmbedding(**budget, embedding_types=["int8"]))

    retrieve(dataset, plain, depth=3)  # type: ignore[arg-type]  # the fixture is a Dataset
    retrieve(dataset, quantized, depth=3)  # type: ignore[arg-type]

    entries = [entry for entry in (tmp_path / "cache" / "indexes").iterdir() if entry.is_dir()]
    assert len(entries) == 2, "the two runs must not share an index cache entry"
