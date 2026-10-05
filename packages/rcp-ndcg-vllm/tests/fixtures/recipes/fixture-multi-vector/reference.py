"""Fixture reference for the late-interaction embedder: one float16 vector per whitespace token."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import DIM, token_id, token_vectors, tokens, vector  # noqa: E402

QUERY_PROMPT = "query: "
DOC_PROMPT = "doc: "

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-multivector-reference@{device}"
    return _loaded


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1 for a multi-vector recipe: the reference ids of the prompted document."""
    del query, instruction
    return [token_id(word) for word in tokens(DOC_PROMPT + document)]


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2: one ``(n_tokens, DIM)`` float16 array per text (per-token vectors, unnormalised rows)."""
    assert _loaded
    prefix = QUERY_PROMPT if role == "query" else DOC_PROMPT
    return [token_vectors(prefix + text, "tok") for text in texts]


def embed_dense(texts: list[str], role: str) -> list[np.ndarray]:
    """The dense shape, unused by this fixture (kept so the module satisfies the reference interface)."""
    assert _loaded
    return [vector(text, f"embed-{role}").astype(np.float16) for text in texts]


__all__ = ["DIM", "DOC_PROMPT", "QUERY_PROMPT", "embed", "embed_dense", "load", "render", "tokens"]
