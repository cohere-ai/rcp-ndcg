"""The judge's tokenizer: loading from a local file or the Hub (offline, patched), caching and identity."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from rcp_ndcg.data import tokenizer as tokenizer_module
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import DependencyError, MissingInputError
from tests._tokenizers import save, word_tokenizer


@pytest.fixture(autouse=True)
def _fresh_cache():
    load_tokenizer.cache_clear()
    yield
    load_tokenizer.cache_clear()


def test_a_local_file_or_directory_loads_with_the_file_hash(tmp_path: Path) -> None:
    path = save(word_tokenizer(), tmp_path)
    loaded = load_tokenizer(str(path))
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.identity() == {"name": str(path), "sha256": loaded.sha256}
    assert load_tokenizer(str(tmp_path)).sha256 == loaded.sha256  # the directory holding tokenizer.json
    assert loaded.count("the query, and the passage") == 6


def test_a_tokenizer_is_loaded_once_per_process(tmp_path: Path) -> None:
    path = str(save(word_tokenizer(), tmp_path))
    assert load_tokenizer(path) is load_tokenizer(path)


def test_a_missing_local_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError, match="no tokenizer file"):
        load_tokenizer(str(tmp_path / "absent.json"))


def test_a_hub_id_downloads_tokenizer_json_at_its_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    path = save(word_tokenizer(), tmp_path)
    calls = []

    def download(repo_id: str, filename: str, *, revision: str | None = None) -> str:
        calls.append((repo_id, filename, revision))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert load_tokenizer("org/model@abc123").name == "org/model@abc123"
    assert load_tokenizer("org/model").sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [("org/model", "tokenizer.json", "abc123"), ("org/model", "tokenizer.json", None)]


def test_a_missing_library_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = str(save(word_tokenizer(), tmp_path))
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(DependencyError) as caught:
        tokenizer_module.load_tokenizer(path)
    assert caught.value.hint == 'pip install "rcp-ndcg[hf]"'
