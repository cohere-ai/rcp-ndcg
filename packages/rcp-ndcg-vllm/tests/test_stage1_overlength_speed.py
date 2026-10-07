"""Runtime bound: the request generator's length-ladder padding is one tokenization, never a growth loop.

The harness sampler's over-length padding (``equivalence.stages._over_length``) has its own bound, pinned in
``tests/test_equivalence.py``; this file bounds the generator's exact-length pad, which grows long seeds.
"""

from __future__ import annotations

import time

from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.observe.requests import _pad_to_tokens
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES

BOUND_S = 5.0
"""The per-call bound (seconds).  The single-pass implementation takes well under a second on the
synthetic document below; the measured growth loop took 122 s."""


def _tokenizer() -> object:
    return tokenizer_of(load_recipe(RECIPES / "fixture-embed"))


def test_generator_padding_of_a_long_synthetic_document_is_one_pass() -> None:
    """``observe.requests._pad_to_tokens`` over the same long document (the length-ladder rows)."""
    tokenizer = _tokenizer()
    long_document = "alpha " * 4000
    target = tokenizer.count(long_document) + 2500
    start = time.perf_counter()
    text = _pad_to_tokens(tokenizer, long_document, target)
    elapsed = time.perf_counter() - start
    assert abs(tokenizer.count(text) - target) <= 2
    assert text.startswith(long_document.rstrip()), "the pad never rewrites the seed"
    assert elapsed < BOUND_S, f"_pad_to_tokens took {elapsed:.1f}s on a long synthetic document (bound {BOUND_S}s)"
