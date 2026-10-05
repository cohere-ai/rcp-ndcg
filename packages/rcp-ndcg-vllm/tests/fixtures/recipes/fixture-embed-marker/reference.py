"""Fixture reference for the dense embedder: prompts are composed client-side, the reference mirrors them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import DIM, FixtureTokenizer, vector  # noqa: E402

MAX_TOKENS = 512  # the recipe's client.max_tokens; the reference pins the same budget

QUERY_PROMPT = "query: "
DOC_PROMPT = "doc: "
SEP_SUFFIX = "<<sep>>"
"""The template's fixed tail: the marker anchor, declared in the recipe as id 49998."""
END_SUFFIX = " [END]"
"""The template's fixed tail: the last-token anchor, reserved and re-attached by the cut."""

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-marker-reference@{device}"
    return _loaded


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1 for an embedding recipe: the reference ids of the prompted document (the query side is unused)."""
    del query, instruction
    from deterministic import reserve_and_append, token_id, tokens

    prefix_ids = [token_id(word) for word in tokens(DOC_PROMPT)]
    suffix_ids = [49998]  # the sep special's reserved id, declared in the recipe as anchor_markers
    content_ids = [token_id(word) for word in tokens(document)]
    return reserve_and_append(prefix_ids, content_ids, suffix_ids, MAX_TOKENS)


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2: one unit vector per text; the reference composes the same prompts the client sends."""
    assert _loaded
    prefix = QUERY_PROMPT if role == "query" else DOC_PROMPT
    return [vector(prefix + text + SEP_SUFFIX, "embed") for text in texts]


__all__ = ["DIM", "DOC_PROMPT", "QUERY_PROMPT", "SEP_SUFFIX", "embed", "load", "render", "render_shape"]


def tokenizer() -> FixtureTokenizer:
    """The fixture tokenizer for stage 1 (a production reference omits this; the harness loads client.tokenizer)."""
    return FixtureTokenizer()


def render_shape(shape: str, query: str, document: str, instruction: str | None) -> list[int]:
    """The per-shape hook: [prefix][content][sep marker], the marker id declared in the recipe."""
    del instruction
    from deterministic import reserve_and_append, token_id, tokens

    heads = {
        "query": [token_id(word) for word in tokens(QUERY_PROMPT)],
        "document": [token_id(word) for word in tokens(DOC_PROMPT)],
    }
    suffix_ids = [49998]  # the sep special's reserved id, declared in the recipe as anchor_markers
    prefix_ids = heads[shape]
    content_ids = [token_id(word) for word in tokens(query if shape == "query" else document)]
    return reserve_and_append(prefix_ids, content_ids, suffix_ids, MAX_TOKENS)
