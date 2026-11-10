"""Freshness-tracked local cache for remote objects.

Readers call :func:`cache` and get a local path back.  A second call re-uses the
cached copy unless the remote object changed, judged by whatever identity the
backend exposes (GCS generation, S3/Azure etag, otherwise size + mtime).

Both the payload and its metadata sidecar are published under an exclusive lock
on a per-entry lockfile, each through a temp file and one rename (see
:func:`rcp.storage.core.publish`).  Concurrent readers -- the N worker
ranks that each resolve the same corpus at start-up are the usual case --
cannot see a half-written file, and two writers whose backend identities
differ cannot interleave their two renames and leave one writer's payload
under the other's identity: the staleness check compares only the sidecar, so
that state would be validated and served until the remote moved again.

A backend that exposes none of the identity fields is never trusted: the
object is re-downloaded on every call, with a warning -- downloading twice is
a cost, serving stale bytes is a wrong answer.

The module also declares the byte-store contract a downstream implements for its own artifacts -- a
resumable per-example store, an object store of its own, a test double: :class:`CacheStore`, with
:meth:`CacheStore.get` a lookup (``None`` for a key the store does not hold), :meth:`CacheStore.put`
replacing a key's value, :meth:`CacheStore.keys` listing a prefix and :meth:`CacheStore.delete`
invalidating. Two implementations ship: :class:`MemoryCacheStore` and :class:`UriCacheStore`, the entries
:func:`cache` populates. :func:`lookup` reads a store defensively -- a store that raises is a miss, logged,
never a failed run -- and :func:`cache_key` builds a prefix-scannable key. The class docstrings are the
contract: they state what a hit, a miss and an invalidation mean for each implementation.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from rcp.errors import ConfigError, MissingInputError
from rcp.storage import core
from rcp.storage.uri import is_remote, local_path
from rcp.support.identity import hash_text, short
from rcp.support.logging import get_logger
from rcp.support.urls import safe_url

logger = get_logger(__name__)

#: Metadata fields, most to least authoritative, used to decide staleness.
_IDENTITY_FIELDS = ("generation", "etag", "version_id", "mtime", "updated", "size")

#: Substrings matched case-insensitively against every metadata key, so a backend's own spelling of
#: a freshness field (``LastModified``, ``created``, ``ETag``) still pins the identity. A matching
#: field that moves without the content changing only costs re-downloads, never stale bytes.
_IDENTITY_PATTERNS = ("etag", "generation", "version", "mtime", "modified", "created", "updated", "size")


def _cache_dir() -> Path:
    from rcp.support.paths import cache_dir

    return cache_dir()


def _identity(metadata: dict[str, Any]) -> dict[str, str]:
    """Reduce backend metadata to the comparable subset.

    Empty when the backend exposes no freshness field at all; :func:`cache` treats that as "never
    reuse", not as "always reuse".
    """
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        lowered = key.lower()
        if key in _IDENTITY_FIELDS or any(pattern in lowered for pattern in _IDENTITY_PATTERNS):
            out[key] = str(value)
    return out


#: How much of the URI's file name a cache file keeps after the digest, for a person reading the directory.
_READABLE_CHARS = 64


def _readable_name(uri: str) -> str:
    """The URI's last *path* segment, sanitised: no query string (a presigned URL's signature
    belongs in the digest, not in the file name of a shared cache directory)."""
    readable = re.sub(r"[^A-Za-z0-9._-]", "_", urlsplit(uri).path.rstrip("/").rsplit("/", 1)[-1])
    return readable[-_READABLE_CHARS:]


def cache_path_for(uri: str) -> Path:
    """Return the local path *uri* caches to (whether or not it is populated).

    The name is a digest of the whole URI, so two URIs never share a file, followed by the URI's last path segment
    (shortened) so the directory stays readable.
    """
    return _cache_dir() / f"{short(hash_text(uri), 32)}_{_readable_name(uri)}"


#: The sidecars an entry carries beside its payload: the identity it is fresh for, and the key it was cached
#: under (so :meth:`UriCacheStore.keys` can list the cache without knowing every key that was ever asked for).
_IDENTITY_SUFFIX = ".meta"
_KEY_SUFFIX = ".key"


def _identity_path(cached_file: Path) -> Path:
    """The sidecar recording the identity a cache entry is fresh for."""
    return cached_file.with_name(f"{cached_file.name}{_IDENTITY_SUFFIX}")


def _key_path(cached_file: Path) -> Path:
    """The sidecar recording the key a cache entry was cached under."""
    return cached_file.with_name(f"{cached_file.name}{_KEY_SUFFIX}")


def _record_key(cached_file: Path, key: str) -> None:
    """Record *key* beside the entry, so the cache is listable by the key that filled it.

    Written only when it differs, and without a lock: the content is a pure function of the key, so two
    writers publish the same bytes. A payload published before this sidecar is an entry ``keys`` cannot
    name; it is still a hit for :func:`cache` and for :meth:`UriCacheStore.get`.

    A key that carries a credential -- userinfo, a query string (a presigned URL's signature) or a fragment --
    is not recorded at all: the digest already names the entry, and the cache directory is shared, which is
    why :func:`cache_path_for` keeps the query string out of the file name.
    """
    if safe_url(key) != key:
        return
    path = _key_path(cached_file)
    try:
        if path.read_text(encoding="utf-8") == key:
            return
    except OSError:
        pass
    core.publish(path, lambda tmp: tmp.write_text(key, encoding="utf-8"))


def _reusable(cached_file: Path, metadata_path: Path, remote_identity: dict[str, str]) -> bool:
    """Whether the cached pair is still the remote object's: the sidecar's identity, compared."""
    if not (cached_file.exists() and metadata_path.exists()):
        return False
    try:
        cached_identity = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata_path.unlink(missing_ok=True)
        return False
    return cached_identity == remote_identity


@contextlib.contextmanager
def _publication_lock(cached_file: Path):
    """An exclusive lock over one cache entry's publication (payload, then its sidecars).

    A torn pair -- one writer's payload under another writer's identity -- would pass the
    staleness check and be served forever. The lock itself is :func:`rcp.storage.publication_lock`
    (the one home of the discipline); on a platform without advisory locks it is absent, and the payload is
    published before its sidecar, so a torn pair mismatches the sidecar and is re-downloaded on the next
    call rather than served.
    """
    with core.publication_lock(cached_file):
        yield


def cache(uri: str | Path) -> Path:
    """Return a local path holding the contents of *uri*.

    Local paths (plain or ``file://``) are returned untouched -- nothing is copied for a path
    that is already on disk. A remote object is downloaded once and re-used until its backend
    identity (generation, etag, size and mtime) changes.

    Args:
        uri: Local path or remote URI.
    """
    if not is_remote(uri):
        return local_path(uri) or Path(uri)

    text = str(uri)
    cached_file = cache_path_for(text)
    metadata_path = _identity_path(cached_file)

    try:
        remote_identity = _identity(core.info(text))
    except FileNotFoundError as exc:
        raise MissingInputError(
            f"remote object not found: {safe_url(text)}",
            hint="the cache reads an existing object: check the URI (and the backend's credentials)",
        ) from exc
    if not remote_identity:
        logger.warning(
            f"{safe_url(text)} exposes no identity metadata the cache can compare (no etag, generation, "
            "mtime or size); re-downloading on every call"
        )

    if _reusable(cached_file, metadata_path, remote_identity):
        if not _key_path(cached_file).exists():  # an entry cached before the key sidecar existed
            _record_key(cached_file, text)
        return cached_file

    with _publication_lock(cached_file):
        if _reusable(cached_file, metadata_path, remote_identity):
            return cached_file  # a rank that waited for the lock published it meanwhile
        core.publish(cached_file, lambda tmp: core.get(text, tmp))
        if remote_identity:
            core.publish(metadata_path, lambda tmp: tmp.write_text(json.dumps(remote_identity), encoding="utf-8"))
        elif metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
        _record_key(cached_file, text)
    return cached_file


def cache_key(*parts: str) -> str:
    """Join *parts* into a cache key: ``"/"``-separated and prefix-scannable.

    A key is an opaque string to a store; this is the package's convention, so that
    ``keys(cache_key("run-1", "rows") + "/")`` lists a run's rows and nothing else. A prefix that names a
    namespace therefore ends in ``"/"``: the scan is a plain string prefix, so ``"run-1/rows/1"`` also matches
    ``"run-1/rows/10"``.

    Args:
        *parts: The key's segments, e.g. ``("run-1", "rows", "004001")``.

    Returns:
        The key, the parts joined with ``"/"``.

    Raises:
        ConfigError: No part was given, or a part is empty or contains ``"/"`` -- the same key would then
            be spelled two ways, and a prefix scan could not tell the two apart.
    """
    if not parts:
        raise ConfigError(
            "a cache key needs at least one part",
            hint='build one with cache_key("run-1", "rows", "004001")',
        )
    for part in parts:
        if not part:
            raise ConfigError("a cache key part is empty", hint="every part names one level of the key")
        if "/" in part:
            raise ConfigError(
                f"a cache key part contains '/': {part!r}",
                hint='pass the segments separately: cache_key("run-1", "rows", "004001")',
            )
    return "/".join(parts)


@runtime_checkable
class CacheStore(Protocol):
    """The contract a downstream implements to be its own artifact cache.

    A key is an opaque string; the package's own keys are ``"/"``-separated and prefix-scannable
    (:func:`cache_key`), and nothing in the contract requires that -- only that :meth:`keys` returns every key
    that starts with the prefix it is given.

    The guarantees a caller may rely on:

    * :meth:`get` returns the bytes stored under a key, or ``None`` when the store does not hold it: a miss is
      an answer, never an exception. A store may read through to its backend, or not; :class:`UriCacheStore`
      does not, and a miss there means the caller reads through :func:`cache` and puts the result.
    * :meth:`put` replaces whatever the key held. What a value is worth is the store's to document: the URI
      store records the source's identity and serves the value until the source moves.
    * :meth:`keys` lists the keys that start with *prefix*; ``""`` lists all of them. The order is not part of
      the contract.
    * :meth:`delete` removes a key: a later :meth:`get` never returns an invalidated entry, and deleting a key
      that is not there is not an error.
    * a store that cannot answer may raise; :func:`lookup` turns that into a logged miss, which is how a
      caller reads a store whose backend it does not control.

    ``isinstance`` (the protocol is runtime-checkable) checks only that the four methods exist. The contract
    itself is ``assert_cache_store_contract`` in the test suite, which a downstream runs over its own keys.
    """

    def get(self, key: str) -> bytes | None:
        """The bytes stored under *key*, or ``None`` for a key the store does not hold."""
        ...

    def put(self, key: str, value: bytes) -> None:
        """Store *value* under *key*, replacing whatever the key held."""
        ...

    def keys(self, prefix: str = "") -> Iterable[str]:
        """The keys that start with *prefix* (all of them for ``""``)."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key*; deleting a key that is not there is not an error."""
        ...


class MemoryCacheStore:
    """A :class:`CacheStore` in a dictionary: the store the tests use, and a template for a downstream's own.

    Nothing is persisted and nothing is validated: a value put under a key is the value read back until it is
    replaced or deleted. Use it where a real store would add nothing -- a test, a dry run -- and implement
    :class:`CacheStore` against your own backend for anything a crash must survive.
    """

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        """The bytes stored under *key*, or ``None`` when nothing was put there."""
        return self._entries.get(str(key))

    def put(self, key: str, value: bytes) -> None:
        """Store *value* under *key*, replacing the previous value."""
        self._entries[str(key)] = value

    def keys(self, prefix: str = "") -> list[str]:
        """The keys that start with *prefix*, sorted (all of them for ``""``)."""
        return sorted(key for key in self._entries if key.startswith(prefix))

    def delete(self, key: str) -> None:
        """Remove *key*; deleting a key that is not there is not an error."""
        self._entries.pop(str(key), None)


class UriCacheStore:
    """The URI cache as a :class:`CacheStore`: the entries :func:`cache` populates, keyed by URI.

    An entry is the payload file :func:`cache_path_for` names, plus the identity sidecar and the key sidecar
    this module writes beside it -- the same files, the same publication lock and the same freshness rule
    :func:`cache` uses, so a store and a reader share one cache rather than two.

    * :meth:`get` is a lookup and never fetches. It returns the entry's bytes only while the entry's recorded
      identity equals the source's current identity (``rcp.storage.info``, as :func:`cache` compares it), so a
      source that moved is a miss -- never the bytes it used to have. A caller that needs the object reads
      through :func:`cache`, which downloads on a miss and publishes the entry this store then hits.
    * :meth:`put` writes *value* into the entry and records the source's identity as it is now, under the
      publication lock. The value is the caller's assertion -- served until the source moves -- and the
      source object itself is never written. A key whose source does not exist is a
      :class:`~rcp.errors.MissingInputError`: an entry with no identity to be fresh for would never be served.
    * :meth:`keys` lists the entries whose key sidecar starts with *prefix* and that the store can serve --
      payload and identity sidecar both present, the pair :func:`cache` publishes -- so the downloads a reader
      made are listable: ``keys("gs://YOUR-BUCKET/run-1/rows/")`` is what a resumed pass asks to skip the rows
      it already has. An entry cached from a backend with no comparable identity is never listed, as it is
      never served, and neither is one whose key carries a credential (a presigned URL's query string): the
      digest names that entry, and the cache directory is shared.
    * :meth:`delete` removes the entry -- payload and both sidecars -- under the publication lock, so a
      concurrent :func:`cache` sees the whole entry or re-fetches it, and the invalidated bytes are never
      served again.

    A key may name a local path as well as a URI (its identity is size and mtime); the entry lives in the
    cache directory, and neither :meth:`put` nor :meth:`delete` touches the caller's file. :func:`cache`
    returns a local path untouched, so a reader that wants the file should not go through the store.
    """

    def get(self, key: str) -> bytes | None:
        """The cached bytes for *key* while they are still the source's, else ``None`` (a miss, never a fetch)."""
        text = str(key)
        entry = cache_path_for(text)
        if not entry.exists():
            return None  # nothing cached: a miss costs one stat, not a round trip to the backend
        try:
            identity = _identity(core.info(text))
        except FileNotFoundError:
            return None  # the source is gone: the entry is not its value any more
        if not _reusable(entry, _identity_path(entry), identity):
            return None
        try:
            return entry.read_bytes()
        except FileNotFoundError:  # a delete between the check and the read: the hit became a miss
            return None

    def put(self, key: str, value: bytes) -> None:
        """Publish *value* as *key*'s entry, recording the source's identity it is fresh for.

        Raises:
            MissingInputError: The key's source does not exist, so the entry would have no identity to be
                validated against and would never be served.
        """
        text = str(key)
        entry = cache_path_for(text)
        try:
            identity = _identity(core.info(text))
        except FileNotFoundError as exc:
            raise MissingInputError(
                f"cannot cache a value under {safe_url(text)}: the object does not exist",
                hint="an entry records the source's identity to be fresh for: write the object first, "
                "or implement CacheStore for a store of your own",
            ) from exc
        with _publication_lock(entry):
            core.publish_bytes(entry, value)
            if identity:
                core.publish(
                    _identity_path(entry),
                    lambda tmp: tmp.write_text(json.dumps(identity), encoding="utf-8"),
                )
            else:  # no comparable identity: an entry the cache's own rule never trusts
                _identity_path(entry).unlink(missing_ok=True)
            _record_key(entry, text)

    def keys(self, prefix: str = "") -> list[str]:
        """The keys of the entries the store can serve that start with *prefix*, sorted."""
        out: list[str] = []
        for key_file in _cache_dir().glob(f"*{_KEY_SUFFIX}"):
            try:
                key = key_file.read_text(encoding="utf-8")
            except OSError:  # a concurrent delete
                continue
            if not key.startswith(prefix):
                continue
            entry = key_file.with_name(key_file.name[: -len(_KEY_SUFFIX)])
            if not (entry.exists() and _identity_path(entry).exists()):
                continue  # a payload or an identity that is gone is not an entry the store can serve
            out.append(key)
        return sorted(out)

    def delete(self, key: str) -> None:
        """Remove *key*'s entry: its payload and both sidecars, under the publication lock."""
        entry = cache_path_for(str(key))
        with _publication_lock(entry):
            entry.unlink(missing_ok=True)
            _identity_path(entry).unlink(missing_ok=True)
            _key_path(entry).unlink(missing_ok=True)


def lookup(store: CacheStore, key: str) -> bytes | None:
    """Read *key* through *store*, treating a store that raises as a miss.

    A downstream implementation may raise -- a backend outage, an expired credential -- and a cache miss is a
    normal answer: a failed cache read must not fail the run that could compute the value. The miss is logged
    with the store's class and the key, redacted (:func:`rcp.support.urls.safe_url`: a key may be a presigned
    URL).

    Args:
        store: The store to read.
        key: The key to look up.

    Returns:
        The stored bytes, or ``None`` on a miss or a store error.
    """
    try:
        return store.get(key)
    except Exception as exc:  # noqa: BLE001 - a cache read must not fail the caller: it is a miss
        logger.warning(
            f"cache store {type(store).__name__} raised on {safe_url(str(key))} ({exc}); treating it as a miss"
        )
        return None


__all__ = [
    "CacheStore",
    "MemoryCacheStore",
    "UriCacheStore",
    "cache",
    "cache_key",
    "cache_path_for",
    "lookup",
]
