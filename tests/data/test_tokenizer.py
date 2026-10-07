"""The judge's tokenizer: loading from a local file or the Hub (offline, patched), caching and identity."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from rcp_ndcg.data import tokenizer as tokenizer_module
from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer
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


def test_embedded_truncation_and_padding_do_not_cap_the_counts(tmp_path: Path) -> None:
    """A tokenizer.json that embeds truncation/padding (topk-embed-v1-small ships truncation at 1024) must not
    cap every count at the embedded length: the caps are reset at load, as transformers resets them per call."""
    from tokenizers import Tokenizer

    embedded = Tokenizer.from_str(word_tokenizer().backend.to_str())
    embedded.enable_truncation(max_length=4)  # direction Right, the shape the reference ships
    embedded.enable_padding(length=9)  # fixed-length padding, the other silent cap
    path = save(TextTokenizer.from_backend(embedded, name="test/embedded-caps"), tmp_path)
    loaded = load_tokenizer(str(path))
    text = "the relevant document answers the query with evidence and page one"
    assert loaded.count(text) > 4  # the embedded truncation (max_length=4) no longer tops the count out
    assert loaded.count(text) == word_tokenizer().count(text)  # and it is the tokenizer's own count
    assert len(loaded.ids(text)) == loaded.count(text)  # the embedded padding (length=9) added no ids
    assert loaded.offsets(text) == word_tokenizer().offsets(text)


def test_a_backend_handed_over_is_reset_the_same_way() -> None:
    """``from_backend`` funnels through the same load, so an in-memory backend's caps are reset too."""
    from tokenizers import Tokenizer

    embedded = Tokenizer.from_str(word_tokenizer().backend.to_str())
    embedded.enable_truncation(max_length=3)
    wrapped = TextTokenizer.from_backend(embedded, name="test/embedded-caps-memory")
    text = "the relevant document answers the query with evidence"
    assert wrapped.count(text) == word_tokenizer().count(text)
