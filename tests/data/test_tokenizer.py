"""The judge's tokenizer: loading from a local file or the Hub (offline, patched), caching and identity."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from rcp_ndcg.data import tokenizer as tokenizer_module
from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer
from rcp_ndcg.errors import DependencyError, MissingInputError
from tests._tokenizers import framed_bpe_tokenizer, save, word_tokenizer


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
    from huggingface_hub.errors import EntryNotFoundError

    path = save(word_tokenizer(), tmp_path)
    calls = []

    def download(repo_id: str, filename: str, *, revision: str | None = None) -> str:
        calls.append((repo_id, filename, revision))
        if filename != "tokenizer.json":
            raise EntryNotFoundError(f"{filename} absent")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert load_tokenizer("org/model@abc123").name == "org/model@abc123"
    assert load_tokenizer("org/model").sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [
        ("org/model", "tokenizer.json", "abc123"),
        ("org/model", "tokenizer_config.json", "abc123"),
        ("org/model", "added_tokens.json", "abc123"),
        ("org/model", "special_tokens_map.json", "abc123"),
        ("org/model", "tokenizer.json", None),
        ("org/model", "tokenizer_config.json", None),
        ("org/model", "added_tokens.json", None),
        ("org/model", "special_tokens_map.json", None),
    ]


def test_a_hub_sidecar_is_downloaded_and_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub tokenizer's sidecar is fetched at the same revision and applied like a local one."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    path = save(word_tokenizer(), tmp_path)
    sidecar = tmp_path / "config.json"
    sidecar.write_text(json.dumps({"pad_token": "+"}), encoding="utf-8")

    def download(repo_id: str, filename: str, *, revision: str | None = None) -> str:
        if filename == "tokenizer.json":
            return str(path)
        if filename == "tokenizer_config.json":
            return str(sidecar)
        raise EntryNotFoundError(f"{filename} absent")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    loaded = load_tokenizer("org/model@abc123")
    assert loaded.special_text("+") == "+"
    assert loaded.sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_sidecar(directory: Path, filename: str, payload: object) -> None:
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_the_tokenizer_config_sidecar_adds_its_pad_token(tmp_path: Path) -> None:
    """The engine loads ``AutoTokenizer``, which honours ``tokenizer_config.json``'s added tokens; the bare
    ``tokenizer.json`` load must apply the same sidecars, or a text with a space before the pad token counts
    one token client-side and two engine-side (ctxl-1b's ``pad_token: "+"``, GPU-E1)."""
    save(word_tokenizer(), tmp_path)
    _write_sidecar(tmp_path, "tokenizer_config.json", {"pad_token": "+"})
    loaded = load_tokenizer(str(tmp_path))
    assert loaded.special_text("+") == "+"
    assert loaded.count("a + of") == 3
    # The added-token matcher gives ``+`` its own id; without the sidecar it falls back to the UNK token 0.
    assert loaded.ids("a + of") == [2, loaded.special_id("+"), 3]


def test_added_tokens_and_special_tokens_map_sidecars_apply(tmp_path: Path) -> None:
    """Every sidecar ``AutoTokenizer`` reads: ``added_tokens.json`` and ``special_tokens_map.json`` too."""
    save(word_tokenizer(), tmp_path)
    _write_sidecar(tmp_path, "added_tokens.json", {"+": 99})
    _write_sidecar(tmp_path, "special_tokens_map.json", {"additional_special_tokens": ["-"]})
    loaded = load_tokenizer(str(tmp_path))
    assert loaded.special_text("+") == "+"
    assert loaded.special_text("-") == "-"
    assert loaded.count("a + of") == 3
    assert loaded.count("a - of") == 3


def test_added_tokens_decoder_and_extra_special_tokens_apply(tmp_path: Path) -> None:
    """``tokenizer_config.json``'s ``added_tokens_decoder`` (id -> entry) and ``extra_special_tokens``."""
    save(word_tokenizer(), tmp_path)
    _write_sidecar(
        tmp_path,
        "tokenizer_config.json",
        {
            "added_tokens_decoder": {"99": {"content": "+", "special": True}},
            "extra_special_tokens": ["-"],
        },
    )
    loaded = load_tokenizer(str(tmp_path))
    assert loaded.special_text("+") == "+"
    assert loaded.special_text("-") == "-"


def test_a_sidecar_that_adds_a_token_moves_the_identity(tmp_path: Path) -> None:
    """The identity covers the sidecar content exactly when it changes the effective vocabulary: two passes
    whose judges tokenize differently never pool (the module's invariant)."""
    path = save(word_tokenizer(), tmp_path)
    plain = load_tokenizer(str(tmp_path))
    assert plain.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    _write_sidecar(tmp_path, "tokenizer_config.json", {"pad_token": "+"})
    load_tokenizer.cache_clear()
    with_sidecar = load_tokenizer(str(tmp_path))
    assert with_sidecar.sha256 != plain.sha256
    load_tokenizer.cache_clear()
    assert load_tokenizer(str(tmp_path)).sha256 == with_sidecar.sha256  # deterministic across loads


def test_a_sidecar_that_repeats_the_added_vocabulary_keeps_the_identity(tmp_path: Path) -> None:
    """A sidecar that names a token ``tokenizer.json`` already carries as added adds nothing: the digest --
    and every store keyed by it -- stays the same (the operator decision, 2026-10-09)."""
    path = save(framed_bpe_tokenizer(), tmp_path)
    _write_sidecar(tmp_path, "tokenizer_config.json", {"additional_special_tokens": ["<|end_turn|>"]})
    loaded = load_tokenizer(str(tmp_path))
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.special_text("end_turn") == "<|end_turn|>"


def test_a_sidecar_token_already_in_the_base_vocabulary_still_moves_the_identity(tmp_path: Path) -> None:
    """A token that exists in the base vocabulary but is not an *added* token (ctxl's ``+``) becomes an
    added token: the matcher changes even though the vocabulary already held the string, so the digest must
    move -- this is the load-bearing ctxl-1b case."""
    path = save(word_tokenizer(), tmp_path)
    _write_sidecar(tmp_path, "tokenizer_config.json", {"pad_token": "the"})
    loaded = load_tokenizer(str(tmp_path))
    assert "the" in loaded.added_tokens()
    assert loaded.sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("RCP_NDCG_NETWORK_TESTS") != "1", reason="needs the public Hugging Face Hub")
def test_the_ctxl_pad_token_sidecar_matches_the_engine_tokenization() -> None:
    """The measured ctxl-1b divergence (GPU-E1): ``tokenizer_config.json``'s ``pad_token: "+"`` makes
    ``AutoTokenizer`` split ``" +"`` into two tokens; the bare load must do the same."""
    loaded = load_tokenizer(
        "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b@8fd1edf6a98564cb712064f884b8ef7df5c1b876"
    )
    assert loaded.ids("2 + 2") == [17, 220, 10, 220, 17]
    assert loaded.ids("H + ion") == [39, 220, 10, 27672]
