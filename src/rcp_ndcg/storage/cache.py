"""Freshness-tracked local cache for remote objects.

Readers call :func:`cache` and get a local path back.  A second call re-uses the
cached copy unless the remote object changed, judged by whatever identity the
backend exposes (GCS generation, S3/Azure etag, otherwise size + mtime).

Both the payload and its metadata sidecar are written to a per-process
temporary file and then atomically renamed.  Concurrent readers -- the N
worker ranks that each resolve the same corpus at start-up are the usual
case -- would otherwise observe a half-written file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from rcp_ndcg.storage import core
from rcp_ndcg.storage.uri import is_remote
from rcp_ndcg.support.identity import hash_text, short

#: Metadata fields, most to least authoritative, used to decide staleness.
_IDENTITY_FIELDS = ("generation", "etag", "ETag", "version_id", "mtime", "updated", "last_modified", "size")


def _cache_dir() -> Path:
    from rcp_ndcg.support.paths import cache_dir

    return cache_dir()


def _identity(metadata: dict[str, Any]) -> dict[str, str]:
    """Reduce backend metadata to the comparable subset."""
    return {key: str(metadata[key]) for key in _IDENTITY_FIELDS if metadata.get(key) is not None}


def _atomic_write(target: Path, write: Any) -> None:
    """Materialise *target* through a temp file in the same directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.{os.getpid()}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


#: How much of the URI's file name a cache file keeps after the digest, for a person reading the directory.
_READABLE_CHARS = 64


def cache_path_for(uri: str) -> Path:
    """Return the local path *uri* caches to (whether or not it is populated).

    The name is a digest of the whole URI, so two URIs never share a file, followed by the URI's last path segment
    (shortened) so the directory stays readable.
    """
    readable = re.sub(r"[^A-Za-z0-9._-]", "_", uri.rstrip("/").rsplit("/", 1)[-1])[-_READABLE_CHARS:]
    return _cache_dir() / f"{short(hash_text(uri), 32)}_{readable}"


def cache(uri: str | Path) -> Path:
    """Return a local path holding the contents of *uri*.

    Local paths are returned untouched -- nothing is copied for a path that is
    already on disk. A remote object is downloaded once and re-used until its
    backend identity (generation, etag, size and mtime) changes.

    Args:
        uri: Local path or remote URI.
    """
    if not is_remote(uri):
        return Path(uri)

    text = str(uri)
    cached_file = cache_path_for(text)
    metadata_path = cached_file.with_name(f"{cached_file.name}.meta")

    try:
        remote_identity = _identity(core.info(text))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"remote object not found: {text}") from exc

    if cached_file.exists() and metadata_path.exists():
        try:
            cached_identity = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata_path.unlink(missing_ok=True)
        else:
            if remote_identity and cached_identity == remote_identity:
                return cached_file

    _atomic_write(cached_file, lambda tmp: core.get(text, tmp))
    if remote_identity:
        _atomic_write(metadata_path, lambda tmp: tmp.write_text(json.dumps(remote_identity), encoding="utf-8"))
    elif metadata_path.exists():
        metadata_path.unlink(missing_ok=True)
    return cached_file


__all__ = ["cache", "cache_path_for"]
