"""The engine-side Matryoshka rules at vLLM v0.31.0, on both routes (the study's acceptance item 5).

``/v1/embeddings`` validates the request's ``dimensions`` exactly as ``vllm/pooling_params.py`` does --
``is_matryoshka`` first, then ``1 <= k <= embedding_size``, then membership in ``matryoshka_dimensions`` --
and then slices the full-width vector BEFORE the activation (``seqwise/heads.py``: projector, slice,
``PoolerNormalize``).  ``/pooling`` refuses the per-request field outright
(``pooling/serving.py``: "dimensions is currently not supported").  A fake that draws a fresh k-wide vector
instead of slicing cannot catch a wrong cut order or a wrong set, and a fake that ignores the pooling refusal
would let a client that wrongly sends the field pass conformance and fail on the real engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rcp_ndcg_test.engines import EngineFacts, Exchange, StringsPrompts, VllmEmulator
from tests._engines import observation_corpus

from tests._tokenizers import word_tokenizer

_TEXT = "the a of to in is it"


def _emulator(
    tmp_path: Path,
    *,
    is_matryoshka: bool = True,
    matryoshka_dimensions: tuple[int, ...] | None = (2, 4),
    embedding_size: int | None = 8,
    dim: int = 8,
    exchanges: tuple[Exchange, ...] = (),
) -> VllmEmulator:
    """A tiny emulator whose model facts declare the engine-side Matryoshka gate (surrogates unless observed)."""
    facts = EngineFacts(
        "vllm",
        "0.31.0",
        "tiny",
        "fixtures/Tiny",
        64,
        is_matryoshka=is_matryoshka,
        matryoshka_dimensions=matryoshka_dimensions,
        embedding_size=embedding_size,
    )
    corpus = observation_corpus(tmp_path, exchanges)
    return VllmEmulator.from_corpus(corpus, StringsPrompts(), word_tokenizer(), facts, dim=dim)


def _embedding(emulator: VllmEmulator, **fields: object) -> np.ndarray:
    answer = emulator.answer("/embeddings", "POST", {"model": "tiny", "input": [_TEXT], **fields})
    assert answer.status_code == 200, answer.text
    return np.asarray(answer.json()["data"][0]["embedding"], dtype=np.float64)


def _refusal(emulator: VllmEmulator, path: str, body: dict[str, object]) -> dict[str, object]:
    answer = emulator.answer(path, "POST", {"model": "tiny", "input": [_TEXT], **body})
    assert answer.status_code == 400, answer.text
    return answer.json()["error"]


def test_the_pooling_route_refuses_a_per_request_dimensions(tmp_path: Path) -> None:
    """vLLM's ``/pooling`` serving layer raises "dimensions is currently not supported" for the request
    field, whatever the checkpoint declares (``pooling/serving.py``)."""
    error = _refusal(_emulator(tmp_path), "/pooling", {"dimensions": 4})
    assert "dimensions is currently not supported" in str(error["message"])
    assert error["param"] == "dimensions"


def test_the_pooling_route_refuses_dimensions_even_for_a_matryoshka_checkpoint(tmp_path: Path) -> None:
    """The refusal is the route's, not the checkpoint's: a checkpoint with a declared set still gets it."""
    error = _refusal(_emulator(tmp_path), "/pooling", {"dimensions": 2})
    assert "dimensions is currently not supported" in str(error["message"])


def test_embeddings_refuses_dimensions_without_the_matryoshka_gate(tmp_path: Path) -> None:
    """The checkpoint declares neither ``is_matryoshka`` nor ``matryoshka_dimensions``: vLLM refuses any
    ``dimensions`` (``pooling_params.py`` gate 1)."""
    emulator = _emulator(tmp_path, is_matryoshka=False, matryoshka_dimensions=None)
    error = _refusal(emulator, "/embeddings", {"dimensions": 4})
    assert "does not support Matryoshka embeddings" in str(error["message"])
    assert "dimensions must be unset" in str(error["message"])


def test_embeddings_refuses_an_out_of_range_k(tmp_path: Path) -> None:
    """Gate 2: ``1 <= k <= embedding_size`` (``pooling_params.py``)."""
    emulator = _emulator(tmp_path)
    too_wide = _refusal(emulator, "/embeddings", {"dimensions": 9})
    assert "only supports dimensions in range [1, 8]" in str(too_wide["message"])
    assert "got 9" in str(too_wide["message"])


def test_embeddings_refuses_a_k_outside_the_declared_set(tmp_path: Path) -> None:
    """Gate 3: membership in ``matryoshka_dimensions`` when the checkpoint declares one."""
    error = _refusal(_emulator(tmp_path), "/embeddings", {"dimensions": 3})
    assert "only supports Matryoshka dimensions [2, 4]" in str(error["message"])
    assert "got 3" in str(error["message"])


def test_embeddings_refuses_a_k_wider_than_the_embedding_size_it_reports(tmp_path: Path) -> None:
    """A declared set member above the checkpoint's own width is still out of range: gate 2 runs first."""
    emulator = _emulator(tmp_path, matryoshka_dimensions=(2, 16), embedding_size=8)
    error = _refusal(emulator, "/embeddings", {"dimensions": 16})
    assert "only supports dimensions in range [1, 8]" in str(error["message"])


def test_embeddings_slices_the_full_width_vector_before_the_l2(tmp_path: Path) -> None:
    """The declared ``k`` slices the model's full-width vector and normalises the slice -- cut before L2.

    The full-width surrogate and the ``k`` reply are drawn from the same seed, so the k reply must be
    ``normalize(full[:k])`` exactly: a fresh k-wide draw fails the equality, and a fake that normalises
    before slicing (or skips the renormalisation) ships a non-unit or different vector.
    """
    emulator = _emulator(tmp_path)
    full = _embedding(emulator)
    assert full.shape == (8,)
    cut = _embedding(emulator, dimensions=4)
    assert cut.shape == (4,)
    expected = full[:4] / np.linalg.norm(full[:4])
    assert np.allclose(cut, expected, atol=1e-12)
    assert not np.allclose(cut, full[:4], atol=1e-12), "the slice must be renormalised after the cut"


def test_embeddings_replays_a_recorded_k_reply(tmp_path: Path) -> None:
    """An observed k context replays the recorded k-wide vector (the surrogate path only answers the
    unobserved contexts); the validation still runs before the replay."""
    body = {"model": "tiny", "input": [_TEXT], "dimensions": 4}
    reply = {"object": "list", "data": [{"index": 0, "object": "embedding", "embedding": [0.5, 0.5, 0.5, 0.5]}]}
    emulator = _emulator(tmp_path, exchanges=(Exchange(0, "POST", "/embeddings", body, 200, {}, reply),))
    answer = emulator.answer("/embeddings", "POST", body)
    assert answer.status_code == 200
    assert answer.headers["x-rcp-ndcg-emulator-source"] == "replayed"
    assert answer.json()["data"][0]["embedding"] == [0.5, 0.5, 0.5, 0.5]
    error = emulator.answer("/embeddings", "POST", {**body, "dimensions": 3})
    assert error.status_code == 400 and "only supports Matryoshka dimensions" in error.json()["error"]["message"]


@pytest.mark.parametrize("dimensions", [2, 4])
def test_embeddings_serves_every_declared_k(tmp_path: Path, dimensions: int) -> None:
    assert _embedding(_emulator(tmp_path), dimensions=dimensions).shape == (dimensions,)
