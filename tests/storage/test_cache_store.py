"""The ``CacheStore`` contract: one shared conformance test over both implementations.

A downstream implements ``CacheStore`` for its own artifacts (an object store, a resumable per-example
store); the package ships two -- :class:`~rcp.storage.cache.MemoryCacheStore` for tests and
:class:`~rcp.storage.cache.UriCacheStore` over the URI cache's own entries. :func:`assert_cache_store_contract`
is the one contract both pass, and a downstream's own test suite can call it with its own keys.

The gate: a hit and a miss behave as documented, and an invalidated entry is never returned. The URI store's
own tests cover the freshness half of that -- a source that moved is a miss, never the old bytes.

``memory://`` stands in for a real object store, as in ``test_storage.py``: a genuine non-local backend, so
the URI store runs the same paths as ``gs://`` without the network.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pytest

from rcp import storage
from rcp.errors import ConfigError, MissingInputError
from rcp.storage.cache import (
    CacheStore,
    MemoryCacheStore,
    UriCacheStore,
    cache_key,
    lookup,
)


@pytest.fixture(autouse=True)
def _clean_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fresh cache directory and a fresh in-memory filesystem for every test."""
    monkeypatch.setenv("RCP_NDCG_CACHE_DIR", str(tmp_path / "cache"))
    memory = storage.filesystem("memory://")
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]
    yield
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]


def assert_cache_store_contract(store: CacheStore, keys: list[str], prefix: str) -> None:
    """The contract every :class:`~rcp.storage.cache.CacheStore` keeps, checked over *keys*.

    The caller prepares the keys (a key may name an object the store validates against, as the URI store
    does) and passes a *prefix* that names their namespace, so ``keys`` can be checked exactly. A
    downstream runs this against its own implementation with its own keys.

    Raises:
        AssertionError: the store broke one of the contract's clauses.
    """
    for key in keys:
        assert store.get(key) is None, f"a key that was never put is a miss, got bytes for {key}"
    assert store.get(f"{prefix}/never-put") is None, "a key outside the namespace is a miss too"

    store.put(keys[0], b"first")
    assert store.get(keys[0]) == b"first", "a hit returns the bytes that were put"
    store.put(keys[0], b"second")
    assert store.get(keys[0]) == b"second", "a second put replaces the first value"

    for key in keys[1:]:
        store.put(key, key.encode("utf-8"))
    assert set(store.keys(prefix)) == set(keys), "keys(prefix) lists exactly the entries under the prefix"
    assert set(store.keys(f"{prefix}/rows/")) == {keys[0], keys[1]}, "the match is a plain string prefix"
    assert list(store.keys(f"{prefix}/nothing/")) == [], "an unpopulated prefix lists nothing"

    store.delete(keys[0])
    assert store.get(keys[0]) is None, "an invalidated entry is never returned"
    assert set(store.keys(prefix)) == set(keys[1:]), "a deleted key is gone from keys(prefix)"
    store.delete(keys[0])  # deleting a key that is not there is not an error


class TestTheSharedContract:
    @pytest.mark.parametrize("implementation", ["memory", "uri"])
    def test_both_implementations_keep_the_contract(self, implementation: str) -> None:
        """One contract, both implementations: the shared conformance test the store seam is gated on."""
        store: CacheStore = MemoryCacheStore() if implementation == "memory" else UriCacheStore()
        prefix = "memory://contract"
        keys = [f"{prefix}/rows/000001", f"{prefix}/rows/000002", f"{prefix}/other/000001"]
        for key in keys:
            # The URI store records the source's identity at put time, so a put key needs a source; a
            # memory store ignores it.
            storage.write_text(key, "source")
        assert_cache_store_contract(store, keys, prefix)

    def test_the_implementations_are_structurally_cache_stores(self) -> None:
        assert isinstance(MemoryCacheStore(), CacheStore)
        assert isinstance(UriCacheStore(), CacheStore)

    def test_a_memory_store_is_empty_and_lists_nothing(self) -> None:
        store = MemoryCacheStore()
        assert store.get("anything") is None
        assert list(store.keys()) == []


class _RaisingStore:
    """A downstream store whose backend is down: every ``get`` raises."""

    def get(self, key: str) -> bytes | None:
        raise OSError("backend down")

    def put(self, key: str, value: bytes) -> None:
        raise OSError("backend down")

    def keys(self, prefix: str = "") -> Iterable[str]:
        return []

    def delete(self, key: str) -> None:
        raise OSError("backend down")


class TestTheDefensiveLookup:
    def test_a_store_that_raises_on_get_is_a_logged_miss(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """The gate's second clause: a downstream's failing ``get`` is a miss, not a failure, and logged."""
        # configure_logging (run by other tests) silences the package logger's propagation.
        monkeypatch.setattr(logging.getLogger("rcp"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rcp.storage.cache"):
            assert lookup(_RaisingStore(), "memory://corpus.jsonl") is None
        messages = [record.getMessage() for record in caplog.records]
        assert any("miss" in message.lower() for message in messages), messages
        assert any("backend down" in message for message in messages), "the miss names the store's error"

    def test_the_miss_message_redacts_a_signed_key(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """A key may be a presigned URL: its signature belongs in neither a log line nor an error."""
        monkeypatch.setattr(logging.getLogger("rcp"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rcp.storage.cache"):
            lookup(_RaisingStore(), "https://storage.example/bucket/key?X-Amz-Signature=sekrit-value")
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "sekrit-value" not in messages
        assert "https://storage.example/bucket/key" in messages

    def test_a_working_store_is_read_through_unchanged(self) -> None:
        store = MemoryCacheStore()
        store.put("k", b"value")
        assert lookup(store, "k") == b"value"
        assert lookup(store, "absent") is None


class TestTheUriStore:
    """The existing URI cache as a ``CacheStore``: its entries, its freshness rule, never a stale byte."""

    def test_a_key_the_cache_holds_is_a_hit_and_one_it_does_not_is_a_miss(self) -> None:
        """``get`` is a lookup: it never fetches. A miss means the caller reads through ``cache()``."""
        storage.write_text("memory://corpus.jsonl", "payload")
        store = UriCacheStore()

        assert store.get("memory://corpus.jsonl") is None, "nothing is cached yet: a miss, not a download"
        assert not storage.cache_path_for("memory://corpus.jsonl").exists()

        storage.cache("memory://corpus.jsonl")
        assert store.get("memory://corpus.jsonl") == b"payload", "the entry cache() populated is the hit"

    def test_a_source_that_moved_is_a_miss_never_the_old_bytes(self) -> None:
        """The gate: an invalidated entry is never returned. The entry records the source's identity, so a
        same-size overwrite is a different object, not a hit."""
        storage.write_text("memory://corpus.jsonl", "first")
        storage.cache("memory://corpus.jsonl")
        store = UriCacheStore()
        assert store.get("memory://corpus.jsonl") == b"first"

        storage.write_text("memory://corpus.jsonl", "secon")  # same length, different bytes
        assert store.get("memory://corpus.jsonl") is None, "a moved source invalidates the entry"

        assert storage.cache("memory://corpus.jsonl").read_text(encoding="utf-8") == "secon"
        assert store.get("memory://corpus.jsonl") == b"secon", "the re-fetched entry is the hit"

    def test_a_put_is_the_value_until_the_source_moves(self) -> None:
        """``put`` is the caller's assertion: the entry is fresh for the source's identity as of the put."""
        storage.write_text("memory://derived.bin", "source")
        store = UriCacheStore()

        store.put("memory://derived.bin", b"computed")
        assert store.get("memory://derived.bin") == b"computed"

        storage.write_text("memory://derived.bin", "moved!")  # same length
        assert store.get("memory://derived.bin") is None

    def test_a_put_of_a_missing_source_is_a_missing_input(self) -> None:
        """An entry records the identity it is fresh for; with no object there is nothing to record."""
        with pytest.raises(MissingInputError):
            UriCacheStore().put("memory://absent.bin", b"value")

    def test_an_entry_with_no_comparable_identity_is_never_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend that exposes no freshness field is never trusted, as ``cache()`` says: a put under it
        is written but never returned -- and never listed, since ``keys`` names what the store can serve."""
        monkeypatch.setattr(storage.core, "info", lambda uri: {"name": "no freshness field here"})
        storage.write_text("memory://anonymous.bin", "abc")
        store = UriCacheStore()

        store.put("memory://anonymous.bin", b"value")
        assert store.get("memory://anonymous.bin") is None
        assert list(store.keys("memory://")) == []

    def test_a_miss_never_asks_the_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is cached, so there is no identity to compare: a miss is one stat, not a round trip."""
        calls: list[str] = []
        monkeypatch.setattr(storage.core, "info", lambda uri: calls.append(str(uri)) or {})
        assert UriCacheStore().get("memory://never-read") is None
        assert calls == []

    def test_an_empty_cache_lists_nothing_and_creates_nothing(self) -> None:
        """``keys`` before anything is cached: the cache directory may not exist yet, and asking is not an error."""
        store = UriCacheStore()
        assert list(store.keys()) == []
        assert not storage.cache_path_for("memory://never-read").parent.exists()

    def test_delete_removes_the_entry_so_the_reader_path_refetches(self) -> None:
        """Invalidation is the entry's removal: the payload and both sidecars, so ``cache()`` downloads
        again instead of serving the bytes the caller invalidated."""
        uri = "memory://corpus.jsonl"
        storage.write_text(uri, "first")
        storage.cache(uri)
        entry = storage.cache_path_for(uri)
        assert entry.exists()

        UriCacheStore().delete(uri)

        assert not entry.exists()
        assert not entry.with_name(f"{entry.name}.meta").exists()
        assert not entry.with_name(f"{entry.name}.key").exists()
        assert storage.cache(uri).read_text(encoding="utf-8") == "first", "cache() re-fetches, never a torn read"

    def test_keys_lists_the_entries_the_reader_path_populated(self) -> None:
        """``cache()`` records the key it cached, so a reader's downloads are listable by the store: that is
        what a downstream resuming a pass asks (``keys("run-1/rows/")``)."""
        storage.write_text("memory://run-1/rows/000001", "a")
        storage.write_text("memory://run-1/rows/000002", "b")
        storage.write_text("memory://run-2/rows/000001", "c")
        for uri in ("memory://run-1/rows/000001", "memory://run-1/rows/000002", "memory://run-2/rows/000001"):
            storage.cache(uri)

        store = UriCacheStore()
        assert list(store.keys("memory://run-1/")) == ["memory://run-1/rows/000001", "memory://run-1/rows/000002"]
        assert list(store.keys("memory://run-2/")) == ["memory://run-2/rows/000001"]
        assert list(store.keys("memory://run-3/")) == []

    def test_a_key_that_carries_a_credential_is_served_but_never_written_or_listed(self) -> None:
        """The cache file name keeps a presigned URL's signature out of a shared directory; the key sidecar
        must not put it back. The entry is still a hit -- the digest names it -- and never listable."""
        uri = "memory://bucket/corpus.jsonl?X-Amz-Signature=sekrit-value"
        storage.write_text(uri, "payload")
        storage.cache(uri)

        files = list(storage.cache_path_for(uri).parent.iterdir())
        assert files, "the entry was cached"
        for path in files:
            assert "sekrit-value" not in path.name
            if path.is_file():
                assert "sekrit-value" not in path.read_text(encoding="utf-8", errors="ignore")

        store = UriCacheStore()
        assert store.get(uri) == b"payload", "the entry is served: the digest names it"
        assert list(store.keys("memory://bucket/")) == [], "a credential is never written, so never listed"

    def test_a_local_key_is_cached_like_any_other(self, tmp_path: Path) -> None:
        """A key may name a local file (its identity is size and mtime); the store's entry is in the cache,
        and the file itself is never touched."""
        source = tmp_path / "already-local.jsonl"
        source.write_text("local", encoding="utf-8")
        store = UriCacheStore()

        assert store.get(source) is None
        store.put(source, b"stored")
        assert store.get(source) == b"stored"
        assert source.read_text(encoding="utf-8") == "local", "the store never writes the caller's file"

        store.delete(source)
        assert source.exists(), "delete removes the cache entry, never the caller's file"


class TestTheKey:
    def test_a_key_joins_its_parts_with_a_slash(self) -> None:
        assert cache_key("run-1", "rows", "004001") == "run-1/rows/004001"

    def test_a_part_containing_a_slash_is_refused(self) -> None:
        """A part with a ``/`` would make the namespace ambiguous: ``keys('run-1/rows/1')`` would then
        also match ``run-1/rows/10``, and a resume would skip a row it never wrote."""
        with pytest.raises(ConfigError, match="slash|/"):
            cache_key("run-1/rows", "1")

    def test_an_empty_part_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            cache_key("run-1", "")
        with pytest.raises(ConfigError):
            cache_key()

    def test_a_prefix_ending_in_a_slash_names_one_level(self) -> None:
        """The convention the helper exists for: ``run-1/rows/`` matches the rows, not ``run-1/rows-2/1``."""
        prefix = cache_key("run-1", "rows") + "/"
        assert cache_key("run-1", "rows", "000010").startswith(prefix)
        assert not cache_key("run-1", "rows-2", "1").startswith(prefix)
