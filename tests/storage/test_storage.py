"""Tests for :mod:`rcp_ndcg.storage`.

``memory://`` stands in for a real object store: fsspec's in-memory filesystem
is a genuine non-local backend (no directories, protocol-qualified listings),
so it exercises the same code paths as ``gs://`` without the network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rcp_ndcg import storage
from rcp_ndcg.errors import DataError, DependencyError, classify


@pytest.fixture(autouse=True)
def _clean_memory_fs():
    """Reset the shared in-memory filesystem between tests."""
    memory = storage.filesystem("memory://")
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]
    yield
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]


class TestUri:
    @pytest.mark.parametrize(
        "uri, remote",
        [
            ("gs://bucket/key", True),
            ("s3://bucket/key", True),
            ("hf://datasets/org/name", True),
            ("memory://a/b", True),
            ("file:///tmp/x", False),
            ("/tmp/x", False),
            ("relative/path.jsonl", False),
            (r"C:\Users\me\file.txt", False),
        ],
    )
    def test_is_remote(self, uri: str, remote: bool) -> None:
        assert storage.is_remote(uri) is remote

    def test_join_preserves_scheme_separator(self) -> None:
        assert storage.join("gs://bucket", "runs", "abc") == "gs://bucket/runs/abc"
        assert storage.join("gs://bucket/", "/runs/", "abc") == "gs://bucket/runs/abc"

    def test_join_local_uses_pathlib(self) -> None:
        assert storage.join("/tmp/runs", "abc", "manifest.json") == "/tmp/runs/abc/manifest.json"

    def test_parent(self) -> None:
        assert storage.parent("gs://bucket/runs/abc/x.json") == "gs://bucket/runs/abc"

    def test_parent_of_a_root_object_stays_remote(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``parent("memory://x")`` returned ``"memory:/"``, a *local* path, so writing a
        root-level object created a directory literally named ``memory:`` in the cwd."""
        assert storage.parent("memory://corpus.jsonl") == "memory://"
        assert storage.parent("gs://bucket/x.json") == "gs://bucket"
        assert storage.is_remote(storage.parent("memory://corpus.jsonl"))
        monkeypatch.chdir(tmp_path)
        storage.write_text("memory://corpus.jsonl", "payload")
        assert not (tmp_path / "memory:").exists()


class TestReadWrite:
    def test_local_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "doc.json"
        storage.write_text(str(target), '{"a": 1}')
        assert json.loads(storage.read_text(str(target))) == {"a": 1}
        assert target.exists(), "parent directories are created on write"

    def test_remote_roundtrip(self) -> None:
        storage.write_text("memory://runs/abc/manifest.json", '{"run_id": "abc"}')
        assert storage.exists("memory://runs/abc/manifest.json")
        assert json.loads(storage.read_text("memory://runs/abc/manifest.json")) == {"run_id": "abc"}

    def test_missing_remote_object_does_not_exist(self) -> None:
        assert storage.exists("memory://runs/nope.json") is False


class TestTransfer:
    def test_get_downloads_a_remote_object(self, tmp_path: Path) -> None:
        storage.write_text("memory://data/corpus.jsonl", '{"id": 1}\n')

        local = storage.get("memory://data/corpus.jsonl", tmp_path / "corpus.jsonl")
        assert local.read_text(encoding="utf-8") == '{"id": 1}\n'


class TestJsonl:
    """The JSONL readers every source reader goes through, local or remote."""

    ROWS = (
        '{"query": "q1", "query_id": "1", "doc_ids": ["d1"]}\n\n{"query": "q2", "query_id": "2", "doc_ids": ["d2"]}\n'
    )

    def test_blank_lines_are_skipped_locally_and_remotely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.storage.io import iter_jsonl

        monkeypatch.setenv("RCP_NDCG_CACHE_DIR", str(tmp_path / "cache"))
        (tmp_path / "data.jsonl").write_text(self.ROWS, encoding="utf-8")
        storage.write_text("memory://runs/abc/data.jsonl", self.ROWS)

        assert [r.query_id for r in iter_jsonl(tmp_path / "data.jsonl")] == ["1", "2"]
        assert [r.query_id for r in iter_jsonl("memory://runs/abc/data.jsonl")] == ["1", "2"]

    def test_a_bad_line_is_a_data_error_that_names_it(self, tmp_path: Path) -> None:
        """A corpus fails on row 4.3 million; the line number is how you find it."""
        from rcp_ndcg.storage.io import iter_json_lines, iter_jsonl

        path = tmp_path / "data.jsonl"
        path.write_text('{"query": "q1", "query_id": "1", "doc_ids": ["d1"]}\n\nBROKEN\n', encoding="utf-8")
        with pytest.raises(DataError, match=r"data\.jsonl:3: invalid JSON"):
            list(iter_jsonl(path))
        path.write_text('{"query": "q1", "query_id": "1", "doc_ids": ["d1"]}\n\n{"query": "q2"}\n', encoding="utf-8")
        with pytest.raises(DataError, match=r"data\.jsonl:3: not a RankingExample"):
            list(iter_jsonl(path))
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(DataError, match="a JSONL row is a JSON object"):
            list(iter_json_lines(path))


class TestListing:
    def test_recursive_listing_is_fully_qualified(self) -> None:
        storage.write_text("memory://runs/a/x.json", "1")
        storage.write_text("memory://runs/b/y.json", "2")

        found = sorted(storage.ls("memory://runs", recursive=True))

        assert found == ["memory://runs/a/x.json", "memory://runs/b/y.json"]
        assert storage.read_text(found[0]) == "1", "listed URIs feed straight back into read"

    def test_listing_a_missing_prefix_is_empty(self) -> None:
        assert storage.ls("memory://nothing/here") == []


class TestCache:
    @pytest.fixture(autouse=True)
    def _cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RCP_NDCG_CACHE_DIR", str(tmp_path / "cache"))

    def test_local_paths_are_returned_untouched(self, tmp_path: Path) -> None:
        source = tmp_path / "already-local.jsonl"
        source.write_text("{}", encoding="utf-8")
        assert storage.cache(source) == source

    def test_remote_object_is_downloaded_once(self) -> None:
        storage.write_text("memory://corpus.jsonl", '{"id": 1}\n')

        first = storage.cache("memory://corpus.jsonl")
        assert first.read_text(encoding="utf-8") == '{"id": 1}\n'

        # Mutating the cached copy is visible on the second call: it was reused
        # rather than re-downloaded.
        first.write_text("SENTINEL", encoding="utf-8")
        assert storage.cache("memory://corpus.jsonl").read_text(encoding="utf-8") == "SENTINEL"

    def test_changed_remote_object_invalidates_the_cache(self) -> None:
        storage.write_text("memory://corpus.jsonl", "short")
        storage.cache("memory://corpus.jsonl")

        storage.write_text("memory://corpus.jsonl", "a much longer payload")

        assert storage.cache("memory://corpus.jsonl").read_text(encoding="utf-8") == "a much longer payload"

    def test_distinct_uris_never_share_a_cache_file(self) -> None:
        """'/' and ':' once became '_', so .../a/b_c and .../a_b/c evicted each other."""
        assert storage.cache_path_for("gs://YOUR-BUCKET/a/b_c") != storage.cache_path_for("gs://YOUR-BUCKET/a_b/c")
        assert storage.cache_path_for("gs://YOUR-BUCKET/a/b_c").name.endswith("_b_c")
        assert len(storage.cache_path_for("gs://YOUR-BUCKET/" + "x" * 400).name) < 255

    def test_missing_remote_object_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            storage.cache("memory://absent.jsonl")


class TestBackendErrors:
    def test_missing_backend_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fsspec

        def _raise(protocol: str, **kwargs):
            raise ImportError(f"no module for {protocol}")

        monkeypatch.setattr(fsspec, "filesystem", _raise)
        monkeypatch.setattr("rcp_ndcg.storage.core._FS_CACHE", {})

        with pytest.raises(DependencyError, match=r"s3") as missing:
            storage.filesystem("s3://bucket/key")
        assert missing.value.hint == 'pip install "rcp-ndcg[s3]"'

    def test_a_missing_backend_is_a_dependency_error_not_a_bug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("fsspec")
        if importlib.util.find_spec("s3fs") is not None:
            pytest.skip("s3fs is installed")
        monkeypatch.setattr("rcp_ndcg.storage.core._FS_CACHE", {})
        with pytest.raises(Exception) as raised:
            storage.exists("s3://b/k")
        error = classify(raised.value)
        assert error.exit_code == 10 and error.hint == 'pip install "rcp-ndcg[s3]"'
