"""Filesystem operations that work the same on local disk and object storage.

Everything routes through fsspec, so ``gs://``, ``s3://``, ``az://``, ``hf://``,
``http://`` and plain paths are interchangeable at every call site.  The
protocol-specific packages are optional extras; asking for a protocol that is
not installed raises a message naming the extra to install rather than an
``ImportError`` from three frames down.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from rcp_ndcg.errors import DependencyError, dependency_error
from rcp_ndcg.storage.uri import is_remote, protocol_of, split_protocol

#: Protocol -> the module that serves it, so a missing backend names the install command
#: (:func:`rcp_ndcg.errors.dependency_error` knows which extra provides which module).
_BACKEND_MODULES: dict[str, str] = {
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "s3": "s3fs",
    "s3a": "s3fs",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
    "adl": "adlfs",
    "hf": "huggingface_hub",
    "http": "aiohttp",
    "https": "aiohttp",
}

_FS_CACHE: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}


def filesystem(uri: str | Path, **options: Any):
    """Return the fsspec filesystem serving *uri*.

    Instances are cached per (protocol, options) because fsspec filesystems
    hold connection pools and credential state that is expensive to rebuild --
    the retrieval and judging loops resolve thousands of paths.
    """
    import fsspec

    protocol = protocol_of(uri)
    key = (protocol, tuple(sorted(options.items())))
    cached = _FS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        fs = fsspec.filesystem(protocol, **options)
    except ValueError as exc:  # fsspec: "Protocol not known"
        raise DependencyError(
            f"no fsspec filesystem is registered for {protocol}:// paths",
            hint=f"install the package that provides it, or register one: fsspec.register_implementation("
            f"{protocol!r}, MyFileSystem) or an 'fsspec.specs' entry point",
        ) from exc
    except ImportError as exc:
        module = _BACKEND_MODULES.get(protocol)
        if module is None:
            raise DependencyError(
                f"no storage backend installed for {protocol}:// paths",
                hint=f"install the fsspec implementation of {protocol}:// paths",
            ) from exc
        raise dependency_error(module, needed_for=f"reading {protocol}:// paths") from exc
    _FS_CACHE[key] = fs
    return fs


def _strip(uri: str | Path) -> str:
    """Return the path fsspec expects (protocol retained; it parses it itself)."""
    return str(uri)


@contextmanager
def open_path(uri: str | Path, mode: str = "rb", **kwargs: Any) -> Iterator[IO[Any]]:
    """Open *uri* for reading or writing, creating parent directories on write.

    ``encoding`` defaults to UTF-8 for text modes so behaviour does not drift
    with the host locale.
    """
    if "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    if any(flag in mode for flag in ("w", "a", "x")):
        makedirs(parent_of(uri))
    fs = filesystem(uri)
    with fs.open(_strip(uri), mode, **kwargs) as handle:
        yield handle


def parent_of(uri: str | Path) -> str:
    """Return the containing directory of *uri*."""
    from rcp_ndcg.storage.uri import parent

    return parent(uri)


def exists(uri: str | Path) -> bool:
    """Return ``True`` if *uri* exists."""
    if not is_remote(uri):
        return Path(uri).exists()
    return bool(filesystem(uri).exists(_strip(uri)))


def makedirs(uri: str | Path) -> None:
    """Create *uri* as a directory, including parents.

    Object stores have no directories; the call is a no-op there, which is why
    writers can call this unconditionally.
    """
    if not is_remote(uri):
        Path(uri).mkdir(parents=True, exist_ok=True)
        return
    fs = filesystem(uri)
    try:
        fs.makedirs(_strip(uri), exist_ok=True)
    except (NotImplementedError, FileExistsError):
        pass


def ls(uri: str | Path, *, recursive: bool = False, files_only: bool = True) -> list[str]:
    """List the children of *uri*.

    Returns fully-qualified URIs (protocol preserved) so results can be fed
    straight back into any other function here.
    """
    fs = filesystem(uri)
    path = _strip(uri).rstrip("/")
    if not fs.exists(path):
        return []
    entries = fs.find(path) if recursive else fs.ls(path, detail=False)
    protocol, _ = split_protocol(uri)
    out: list[str] = []
    for entry in entries:
        text = str(entry)
        if files_only and text.endswith("/"):
            continue
        if protocol and "://" not in text:
            text = f"{protocol}://{text.lstrip('/')}" if protocol != "file" else text
        out.append(text)
    return out


def read_bytes(uri: str | Path) -> bytes:
    """Read *uri* in full."""
    with open_path(uri, "rb") as handle:
        return handle.read()


def read_text(uri: str | Path, encoding: str = "utf-8") -> str:
    """Read *uri* as text."""
    return read_bytes(uri).decode(encoding)


def write_bytes(uri: str | Path, payload: bytes) -> None:
    """Write *payload* to *uri*, creating parents as needed."""
    with open_path(uri, "wb") as handle:
        handle.write(payload)


def write_text(uri: str | Path, payload: str, encoding: str = "utf-8") -> None:
    """Write *payload* to *uri* as text."""
    write_bytes(uri, payload.encode(encoding))


def get(remote: str | Path, local: str | Path) -> Path:
    """Download *remote* to *local* and return the local path."""
    target = Path(local)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not is_remote(remote):
        source = Path(remote)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return target
    filesystem(remote).get_file(_strip(remote), str(target))
    return target


def info(uri: str | Path) -> dict[str, Any]:
    """Return the backend's metadata for *uri* (size, mtime, etag/generation)."""
    if not is_remote(uri):
        stat = Path(uri).stat()
        return {"size": stat.st_size, "mtime": stat.st_mtime}
    return dict(filesystem(uri).info(_strip(uri)))


__all__ = [
    "exists",
    "filesystem",
    "get",
    "info",
    "ls",
    "makedirs",
    "open_path",
    "parent_of",
    "read_bytes",
    "read_text",
    "write_bytes",
    "write_text",
]
