"""Fixture reference for the dense embedder: prompts are composed client-side, the reference mirrors them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import DIM, vector  # noqa: E402

QUERY_PROMPT = "query: "
DOC_PROMPT = "doc: "

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-embed-reference@{device}"
    return _loaded


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1 for an embedding recipe: the reference ids of the prompted document (the query side is unused)."""
    del query, instruction
    from deterministic import token_id, tokens

    return [token_id(word) for word in tokens(DOC_PROMPT + document)]


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2: one unit vector per text; the reference composes the same prompts the client sends."""
    assert _loaded
    prefix = QUERY_PROMPT if role == "query" else DOC_PROMPT
    return [vector(prefix + text, f"embed-{role}").astype(np.float16) for text in texts]


__all__ = ["DIM", "DOC_PROMPT", "QUERY_PROMPT", "embed", "load", "render"]
