"""A retriever whose optional extra is missing fails with the install command (exit 10), not as a bug."""

from __future__ import annotations

import importlib
import sys

import pytest

from rcp_ndcg.errors import DependencyError, ExitCode, classify


def _reimport(monkeypatch: pytest.MonkeyPatch, module: str, *, blocked: tuple[str, ...]):
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, module, raising=False)
    return importlib.import_module(module)


def test_the_hf_encoder_without_torch_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_dense = _reimport(monkeypatch, "rcp_ndcg.retrieval.hf_dense", blocked=("torch", "transformers"))

    with pytest.raises(Exception) as caught:
        hf_dense.load_model_and_tokenizer("some/model")

    error = classify(caught.value)
    assert isinstance(error, DependencyError)
    assert error.exit_code == ExitCode.DEPENDENCY
    assert "rcp-ndcg[local]" in (error.hint or "")


def test_bm25s_without_the_package_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = _reimport(monkeypatch, "rcp_ndcg.retrieval.sparse", blocked=("bm25s",))

    with pytest.raises(Exception) as caught:
        sparse.build_bm25_index(["a document"], "/nonexistent-index-dir", stemmer=None)

    error = classify(caught.value)
    assert isinstance(error, DependencyError)
    assert "pip install --force-reinstall rcp-ndcg" in (error.hint or "")
