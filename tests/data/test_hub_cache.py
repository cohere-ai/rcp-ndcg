"""Offline runs over a hub cache an online run filled (issue #6).

The hub is faked at the two boundaries production calls read: ``hf_hub_download``
and ``try_to_load_from_cache`` see a staged cache under ``tmp_path`` (the
library's frozen ``HF_HUB_CACHE`` constant is pointed at it), and ``HfApi`` is a
stand-in for revision resolution. Nothing here reaches the network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.data.revisions import resolve_revision
from rcp_ndcg.errors import MissingInputError, ProviderError

REPO = "org/data"
SUBSET = "hr__english"
SHA = "4" * 40


class FakeApi:
    """A stand-in ``HfApi`` that resolves one ref and does *not* record it (pre-1.x behaviour)."""

    def __init__(self, refs: dict[tuple[str, str], str]) -> None:
        self.refs = refs
        self.calls: list[tuple[str, str]] = []

    def dataset_info(self, repo_id: str, revision: str | None = None):
        self.calls.append((repo_id, revision or "main"))
        sha = self.refs.get((repo_id, revision or "main"))
        if sha is None:
            raise OSError(f"no such revision {(repo_id, revision)}")
        return SimpleNamespace(sha=sha)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty hub cache under ``tmp_path``; offline, as the suite's autouse fixture sets it."""
    path = tmp_path / "hub"
    path.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(path))
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(path))
    return path


def stage(
    cache: Path,
    *,
    files: dict[str, pd.DataFrame],
    absent: tuple[str, ...] = (),
    sha: str = SHA,
    refs: dict[str, str] | None = None,
) -> Path:
    """Lay out the cache the way a download leaves it: ``snapshots/<sha>/...``, ``.no_exist`` markers, ``refs``."""
    repo_cache = cache / f"datasets--{REPO.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / sha
    for path, frame in files.items():
        (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(snapshot / path)
    for path in absent:
        (repo_cache / ".no_exist" / sha / path).parent.mkdir(parents=True, exist_ok=True)
        (repo_cache / ".no_exist" / sha / path).touch()
    for ref in refs or {}:
        (repo_cache / "refs" / ref).parent.mkdir(parents=True, exist_ok=True)
        (repo_cache / "refs" / ref).write_text(sha)
    return snapshot


_TABLES = {
    f"{SUBSET}/qrels.parquet": pd.DataFrame({"query-id": ["q1"], "corpus-id": ["a"], "score": [1.0]}),
    f"{SUBSET}/top_ranked.parquet": pd.DataFrame({"query-id": ["q1"], "corpus-ids": [["a", "b"]]}),
    f"{SUBSET}/queries.parquet": pd.DataFrame({"id": ["q1"], "text": ["first"]}),
}


def _online(monkeypatch: pytest.MonkeyPatch, refs: dict[tuple[str, str], str]) -> FakeApi:
    """Serve one ref from a fake Hub and take the suite's offline flag off."""
    import huggingface_hub

    fake = FakeApi(refs)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    return fake


# ---------------------------------------------------------------------------
# The offline run without a revision
# ---------------------------------------------------------------------------


def test_an_online_run_records_the_ref_the_offline_run_resolves(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The online run resolves the branch and records it, so the offline run without --revision resolves too."""
    stage(cache, files=_TABLES, absent=(f"{SUBSET}/excluded.parquet",))
    api = _online(monkeypatch, {(REPO, "main"): SHA})

    resolved = resolve_revision(REPO, None)
    resolve_revision.cache_clear()  # the offline run is another process

    assert resolved.commit == SHA
    assert api.calls == [(REPO, "main")]
    ref = cache / f"datasets--{REPO.replace('/', '--')}" / "refs" / "main"
    assert ref.is_file() and ref.read_text() == SHA

    _offline(monkeypatch)
    dataset = load_dataset(f"hf://{REPO}/{SUBSET}")

    assert dataset.revision == SHA
    assert dataset.qrels == {"q1": {"a": 1.0}}
    assert dataset.excluded == {}


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back to the offline flag the suite's autouse fixture holds (the library freezes it at import)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)


def test_offline_unpinned_without_a_recorded_ref_names_the_revision_fix(cache: Path) -> None:
    """A cache with no ref to resolve says so, with the revision fix, and never reports missing data."""
    stage(cache, files=_TABLES)

    with pytest.raises(MissingInputError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}")

    error = caught.value
    assert error.retryable is False
    assert "--revision" in (error.hint or "")
    assert "full sha" in (error.hint or "")
    assert "does not exist" not in error.message, "the qrels are cached; nothing is missing upstream"
    assert error.details["path"] == f"{SUBSET}/qrels.parquet"


def test_offline_pinned_run_serves_an_absent_optional_table(cache: Path) -> None:
    """The pinned offline run: ``.no_exist`` marks a table the repository has never had; the run loads."""
    stage(cache, files=_TABLES, absent=(f"{SUBSET}/excluded.parquet",))

    dataset = load_dataset(f"hf://{REPO}/{SUBSET}", revision=SHA)

    assert dataset.revision == SHA
    assert dataset.qrels == {"q1": {"a": 1.0}}
    assert dataset.candidates == {"q1": ["a", "b"]}
    assert dataset.excluded == {}


def test_offline_uncached_optional_table_names_the_offline_fix(cache: Path) -> None:
    """An optional table the cache knows nothing about is an error with the offline fix, not a silent absence.

    The revision is already pinned: the hint must not tell the caller to pin what is pinned.
    """
    stage(cache, files={f"{SUBSET}/qrels.parquet": _TABLES[f"{SUBSET}/qrels.parquet"]})

    with pytest.raises(MissingInputError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}", revision=SHA)

    error = caught.value
    assert error.retryable is False
    assert "HF_HUB_OFFLINE" in (error.hint or "")
    assert "--revision" not in (error.hint or ""), "the revision is already pinned; the hint says what is missing"
    assert error.details["path"] == f"{SUBSET}/top_ranked.parquet"


def test_offline_pinned_required_table_the_cache_marks_absent_says_does_not_exist(cache: Path) -> None:
    """A ``.no_exist`` mark is the cache's proof the repository has no such table: the error says so, at any pin."""
    stage(cache, files={}, absent=(f"{SUBSET}/qrels.parquet",))

    with pytest.raises(MissingInputError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}", revision=SHA)

    assert f"{SUBSET}/qrels.parquet does not exist at {SHA}" in caught.value.message
    assert caught.value.hint


def test_offline_corpus_materializes_from_the_snapshot(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline, the repository listing comes from the local snapshot, so a run reads its corpus from the cache."""
    card = (
        "---\nconfigs:\n- config_name: hr__english-corpus\n  data_files:\n  - path: hr__english/corpus/*.parquet\n---\n"
    )
    corpus = pd.DataFrame({"id": ["a", "b"], "title": ["A", ""], "text": ["alpha", "beta"]})
    tables = {**_TABLES, f"{SUBSET}/corpus/part-0.parquet": corpus}
    snapshot = stage(cache, files=tables, absent=(f"{SUBSET}/excluded.parquet",))
    (snapshot / "README.md").write_text(card)
    _offline(monkeypatch)

    dataset = load_dataset(f"hf://{REPO}/{SUBSET}", revision=SHA)

    assert {d: doc.text for d, doc in dataset.corpus.items()} == {"a": "A\n\nalpha", "b": "beta"}


def test_offline_listing_without_a_snapshot_names_the_revision_fix(cache: Path) -> None:
    """With no snapshot to list, the offline failure is the classified cache miss with the revision fix."""
    from rcp_ndcg.data.dataset import _hub_listing

    with pytest.raises(MissingInputError) as caught:
        _hub_listing(REPO, None)

    assert caught.value.retryable is False
    assert "--revision" in (caught.value.hint or "")


def test_an_unreachable_hub_is_a_retryable_provider_error(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub that cannot be reached is not missing data: the failure is retryable and names the endpoint."""
    _online(monkeypatch, {})  # the fake Hub has no ref: every lookup raises
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def unreachable(*args: object, **kwargs: object):
        raise LocalEntryNotFoundError(
            "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
            " in the local cache."
        ) from httpx.ConnectError("connection refused")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unreachable)

    with pytest.raises(ProviderError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}")

    error = caught.value
    assert error.retryable is True
    assert "HF_ENDPOINT" in (error.hint or "")


def test_a_required_table_that_does_not_exist_upstream_says_so_with_what_to_check(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Online, a required table the repository lacks stays a missing input, naming the file and a hint."""
    _online(monkeypatch, {})
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(EntryNotFoundError("404")),
    )

    with pytest.raises(MissingInputError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}")

    error = caught.value
    assert f"{SUBSET}/qrels.parquet does not exist" in error.message
    assert error.hint, "a missing input names what produces it"
    assert error.details["path"] == f"{SUBSET}/qrels.parquet"


def test_a_read_only_cache_skips_the_ref_write(
    cache: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A cache that refuses the ref write costs one debug line: the resolution still returns the commit."""
    import logging

    stage(cache, files=_TABLES, absent=(f"{SUBSET}/excluded.parquet",))
    (cache / f"datasets--{REPO.replace('/', '--')}" / "refs").write_text("a file blocks the refs directory")
    _online(monkeypatch, {(REPO, "main"): SHA})

    with caplog.at_level(logging.DEBUG, logger="rcp_ndcg.data.revisions"):
        resolved = resolve_revision(REPO, None)

    assert resolved.commit == SHA
    assert not (cache / f"datasets--{REPO.replace('/', '--')}" / "refs" / "main").exists()
    assert "Could not record" in caplog.text

    resolve_revision.cache_clear()
    _offline(monkeypatch)
    with pytest.raises(MissingInputError) as caught:
        load_dataset(f"hf://{REPO}/{SUBSET}")

    assert "--revision" in (caught.value.hint or ""), "without the ref the offline run says what to do"
