"""Freshness-tracked local cache for remote objects.

Readers call :func:`cache` and get a local path back.  A second call re-uses the
cached copy unless the remote object changed, judged by whatever identity the
backend exposes (GCS generation, S3/Azure etag, otherwise size + mtime).

Both the payload and its metadata sidecar are published under an exclusive lock
on a per-entry lockfile, each through a temp file and one rename (see
:func:`rcp_ndcg.storage.core.publish`).  Concurrent readers -- the N worker
ranks that each resolve the same corpus at start-up are the usual case --
cannot see a half-written file, and two writers whose backend identities
differ cannot interleave their two renames and leave one writer's payload
under the other's identity: the staleness check compares only the sidecar, so
that state would be validated and served until the remote moved again.

A backend that exposes none of the identity fields is never trusted: the
object is re-downloaded on every call, with a warning -- downloading twice is
a cost, serving stale bytes is a wrong answer.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rcp_ndcg.errors import MissingInputError
from rcp_ndcg.storage import core
from rcp_ndcg.storage.uri import is_remote, local_path
from rcp_ndcg.support.identity import hash_text, short
from rcp_ndcg.support.logging import get_logger
from rcp_ndcg.support.urls import safe_url

logger = get_logger(__name__)

#: Metadata fields, most to least authoritative, used to decide staleness.
_IDENTITY_FIELDS = ("generation", "etag", "version_id", "mtime", "updated", "size")

#: Substrings matched case-insensitively against every metadata key, so a backend's own spelling of
#: a freshness field (``LastModified``, ``created``, ``ETag``) still pins the identity. A matching
#: field that moves without the content changing only costs re-downloads, never stale bytes.
_IDENTITY_PATTERNS = ("etag", "generation", "version", "mtime", "modified", "created", "updated", "size")


def _cache_dir() -> Path:
    from rcp_ndcg.support.paths import cache_dir

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
    """An exclusive lock over one cache entry's publication (payload then identity sidecar).

    A torn pair -- one writer's payload under another writer's identity -- would pass the
    staleness check and be served forever. On a platform without advisory locks the lock is
    absent: the payload is published before its sidecar, so a torn pair mismatches the sidecar
    and is re-downloaded on the next call rather than served.
    """
    lock_path = cached_file.with_name(f"{cached_file.name}.lock")
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
    metadata_path = cached_file.with_name(f"{cached_file.name}.meta")

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
        return cached_file

    with _publication_lock(cached_file):
        if _reusable(cached_file, metadata_path, remote_identity):
            return cached_file  # a rank that waited for the lock published it meanwhile
        core.publish(cached_file, lambda tmp: core.get(text, tmp))
        if remote_identity:
            core.publish(metadata_path, lambda tmp: tmp.write_text(json.dumps(remote_identity), encoding="utf-8"))
        elif metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
    return cached_file


__all__ = ["cache", "cache_path_for"]
