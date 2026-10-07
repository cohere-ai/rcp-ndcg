"""Tests for :mod:`rcp_ndcg.storage`.

``memory://`` stands in for a real object store: fsspec's in-memory filesystem
is a genuine non-local backend (no directories, protocol-qualified listings),
so it exercises the same code paths as ``gs://`` without the network.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from rcp_ndcg import storage
from rcp_ndcg.errors import ConfigError, DataError, DependencyError, MissingInputError, classify


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


class TestRelative:
    def test_returns_the_path_below_the_root(self, tmp_path: Path) -> None:
        (tmp_path / "root" / "sub").mkdir(parents=True)
        (tmp_path / "root" / "sub" / "x.txt").write_text("x")
        assert storage.relative(tmp_path / "root" / "sub" / "x.txt", tmp_path / "root") == "sub/x.txt"

    def test_refuses_a_parent_escape(self) -> None:
        """``..`` defeated the prefix check and handed the caller a path outside the root."""
        with pytest.raises(DataError, match="not below"):
            storage.relative("/base/root/../../etc/passwd", "/base/root")

    def test_refuses_a_sibling_escape(self, tmp_path: Path) -> None:
        (tmp_path / "root").mkdir()
        (tmp_path / "sibling").mkdir()
        (tmp_path / "sibling" / "x.txt").write_text("x")
        with pytest.raises(DataError, match="not below"):
            storage.relative(tmp_path / "root" / ".." / "sibling" / "x.txt", tmp_path / "root")

    @pytest.mark.parametrize("root", ["memory://bucket/root"])
    def test_refuses_escapes_on_a_remote_spelling_too(self, root: str) -> None:
        """The remote spellings keep (or drop) the scheme in fsspec's own stripping; either way the
        escape is refused, not returned."""
        with pytest.raises(DataError, match="not below"):
            storage.relative(f"{root}/../../etc/passwd", root)
        with pytest.raises(DataError, match="not below"):
            storage.relative(f"{root}/../sibling/x.txt", root)
        assert storage.relative(f"{root}/a/../b.txt", root) == "b.txt"


class TestFileUris:
    """``file://`` is a spelling the product accepts (wheelhouse URLs); the local fast paths
    used to answer it literally (``Path('file:///x')``), so ``exists`` was False for a live
    file and ``makedirs`` grew a junk ``file:`` tree in the working directory."""

    def test_exists_info_and_get_understand_file_uris(self, tmp_path: Path) -> None:
        source = tmp_path / "d1"
        source.mkdir()
        (source / "live.txt").write_text("hello")
        uri = f"file://{source}/live.txt"
        assert storage.exists(uri)
        assert storage.info(uri)["size"] == 5
        target = tmp_path / "out.txt"
        assert storage.get(uri, target) == target
        assert target.read_text(encoding="utf-8") == "hello"

    def test_makedirs_and_open_path_write_the_real_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        storage.makedirs(f"file://{tmp_path}/abs/target/dir")
        assert (tmp_path / "abs" / "target" / "dir").is_dir()
        storage.write_text(f"file://{tmp_path}/abs/target/f.txt", "data")
        assert (tmp_path / "abs" / "target" / "f.txt").read_text(encoding="utf-8") == "data"
        assert not (tmp_path / "file:").exists(), "no junk tree in the working directory"


class TestPublish:
    """The one home of temp-file + rename publication (the media cache and the PDF render
    used to grow their own copies)."""

    def test_a_local_target_is_replaced_whole(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "f.bin"
        storage.publish_bytes(target, b"v1")
        storage.publish_bytes(target, b"v2")
        assert target.read_bytes() == b"v2"
        assert [p.name for p in target.parent.iterdir() if p.name != "f.bin"] == [], "no temp file left behind"

    def test_a_streaming_write_is_renamed_into_place(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "f.bin"
        storage.publish(target, lambda tmp: tmp.write_bytes(b"stream"))
        assert target.read_bytes() == b"stream"

    def test_a_streaming_publish_needs_a_local_target(self) -> None:
        with pytest.raises(ConfigError, match="local"):
            storage.publish("memory://x", lambda tmp: tmp.write_text("x"))

    def test_a_remote_target_publishes_through_the_backend(self) -> None:
        storage.publish_bytes("memory://published.bin", b"payload")
        assert storage.read_bytes("memory://published.bin") == b"payload"


class TestCache:
    @pytest.fixture(autouse=True)
    def _cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RCP_NDCG_CACHE_DIR", str(tmp_path / "cache"))

    def test_local_paths_are_returned_untouched(self, tmp_path: Path) -> None:
        source = tmp_path / "already-local.jsonl"
        source.write_text("{}", encoding="utf-8")
        assert storage.cache(source) == source

    def test_a_file_uri_is_returned_as_its_path(self, tmp_path: Path) -> None:
        source = tmp_path / "already-local.jsonl"
        source.write_text("{}", encoding="utf-8")
        assert storage.cache(f"file://{source}") == source, "not the literal 'file:/...' string, which names no file"

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

    def test_a_same_size_remote_change_invalidates_the_cache(self) -> None:
        """The identity is not size alone: a backend exposing only ``created`` + ``size`` (the
        in-memory stand-in) served the old bytes forever after a same-length overwrite."""
        storage.write_text("memory://corpus.jsonl", "first")
        storage.cache("memory://corpus.jsonl")

        storage.write_text("memory://corpus.jsonl", "secon")  # same length

        assert storage.cache("memory://corpus.jsonl").read_text(encoding="utf-8") == "secon"

    def test_a_backend_without_identity_warns_and_never_reuses(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        # configure_logging (run by other tests) silences the package logger's propagation.
        monkeypatch.setattr(logging.getLogger("rcp_ndcg"), "propagate", True)
        calls: list[str] = []
        real_get = storage.core.get

        def counting_get(remote, local):
            calls.append(str(remote))
            return real_get(remote, local)

        monkeypatch.setattr(storage.core, "get", counting_get)
        monkeypatch.setattr(storage.core, "info", lambda uri: {"name": "no freshness field here"})
        storage.write_text("memory://anonymous.bin", "abc")

        storage.cache("memory://anonymous.bin")
        storage.cache("memory://anonymous.bin")

        assert len(calls) == 2, "with no comparable identity the object is re-downloaded, never trusted"
        assert any("identity" in record.getMessage().lower() for record in caplog.records)

    def test_a_missing_remote_object_is_a_missing_input(self) -> None:
        with pytest.raises(MissingInputError):
            storage.cache("memory://absent.jsonl")

    def test_a_publishing_writer_locks_the_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The payload and its identity sidecar are renamed under one exclusive lock, so two
        ranks resolving the same corpus cannot interleave two publications and leave one
        writer's payload under the other's identity (which the staleness check would then
        validate and serve forever)."""
        uri = "memory://lock.bin"
        storage.write_text(uri, "payload")
        cached = storage.cache_path_for(uri)
        lockfile = cached.with_name(f"{cached.name}.lock")
        seen: dict[str, bool] = {}
        real_get = storage.core.get

        def observing_get(remote, local):
            result = real_get(remote, local)
            with open(lockfile, "a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen["locked_during_publish"] = False
                    fcntl.flock(handle, fcntl.LOCK_UN)
                except BlockingIOError:
                    seen["locked_during_publish"] = True
            return result

        monkeypatch.setattr(storage.core, "get", observing_get)
        storage.cache(uri)
        assert seen["locked_during_publish"], "the publication (payload + sidecar) holds an exclusive lock"
        with open(lockfile, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released after the call
            fcntl.flock(handle, fcntl.LOCK_UN)

    def test_distinct_uris_never_share_a_cache_file(self) -> None:
        """'/' and ':' once became '_', so .../a/b_c and .../a_b/c evicted each other."""
        assert storage.cache_path_for("gs://YOUR-BUCKET/a/b_c") != storage.cache_path_for("gs://YOUR-BUCKET/a_b/c")
        assert storage.cache_path_for("gs://YOUR-BUCKET/a/b_c").name.endswith("_b_c")
        assert len(storage.cache_path_for("gs://YOUR-BUCKET/" + "x" * 400).name) < 255

    def test_the_cache_name_leaks_no_query_string(self) -> None:
        """A presigned URL's signature used to land in the cache file name of a shared directory."""
        name = storage.cache_path_for("https://storage.example/bucket/corpus.jsonl?X-Amz-Signature=SECRET").name
        assert "SECRET" not in name and "Signature" not in name
        assert name.endswith("_corpus.jsonl")


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


class TestPublishConcurrency:
    def test_a_reader_sees_the_old_or_the_new_file_never_a_partial_one(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        target.write_text("old", encoding="utf-8")
        storage.publish(target, lambda tmp: tmp.write_text("new", encoding="utf-8"))
        assert target.read_text(encoding="utf-8") == "new"
        assert [path.name for path in tmp_path.iterdir()] == ["data.json"], "the temp file is gone after the rename"

    def test_concurrent_writers_never_share_a_temp_file(self, tmp_path: Path) -> None:
        import threading

        target = tmp_path / "shared"
        errors: list[BaseException] = []

        def writer(name: str) -> None:
            try:
                storage.publish(target, lambda tmp: tmp.write_text(name, encoding="utf-8"))
            except BaseException as exc:  # noqa: BLE001 - the test reports it
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"w{index}",)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert target.read_text(encoding="utf-8").startswith("w")
        assert [path.name for path in tmp_path.iterdir()] == ["shared"]
