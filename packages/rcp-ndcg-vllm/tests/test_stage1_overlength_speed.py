"""Runtime bounds: amplifying a request to over-length is one tokenization, never a growth loop.

GPU-VALIDATION.md stage 1 pads over-length samples per shape (from the pairs rows, whose source
documents can be long BRIGHT items).  The bounds below fail on a growth loop that repeatedly
tokenizes the whole candidate string (measured: 122 s for one call on the 4000-word synthetic
document below, on the fixture tokenizer) and pass on the offsets-based single-pass pad (one
tokenization of the pool, a bounded verification).
"""

from __future__ import annotations

import time

from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.equivalence.stages import _over_length
from rcp_ndcg_vllm.observe.requests import _pad_to_tokens
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES

BOUND_S = 5.0
"""The per-call bound (seconds).  The single-pass implementation takes well under a second on the
synthetic document below; the measured growth loop took 122 s."""


def _tokenizer() -> object:
    return tokenizer_of(load_recipe(RECIPES / "fixture-embed"))


def test_over_length_padding_of_a_long_synthetic_document_is_one_pass() -> None:
    """``stages._over_length`` over a long word-heavy document (the BRIGHT-long-document shape)."""
    tokenizer = _tokenizer()
    long_document = "alpha " * 4000
    start = time.perf_counter()
    text = _over_length(long_document, 32768, tokenizer, 3)
    elapsed = time.perf_counter() - start
    assert tokenizer.count(text) >= 2 * 32768, "the over-length sample must cross twice the budget"
    assert elapsed < BOUND_S, f"_over_length took {elapsed:.1f}s on a long synthetic document (bound {BOUND_S}s)"


def test_over_length_padding_of_a_long_seed_keeps_the_seed_verbatim() -> None:
    """The padded text is the seed plus whole pad words: a verbatim prefix, never a re-segmentation."""
    tokenizer = _tokenizer()
    seed = "alpha " * 100
    text = _over_length(seed, 128, tokenizer, 0)
    assert text.startswith(seed.rstrip()) and tokenizer.count(text) >= 256


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