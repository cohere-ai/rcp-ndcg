"""Filesystem operations that work the same on local disk and object storage.

Everything routes through fsspec, so ``gs://``, ``s3://``, ``az://``, ``hf://``,
``http://`` and plain paths are interchangeable at every call site.  The
protocol-specific packages are optional extras; asking for a protocol that is
not installed raises a message naming the extra to install rather than an
``ImportError`` from three frames down.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from rcp_ndcg.errors import ConfigError, DataError, DependencyError, dependency_error
from rcp_ndcg.storage.uri import is_remote, local_path, protocol_of, split_protocol

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
    local = local_path(uri)
    if local is not None:
        return local.exists()
    return bool(filesystem(uri).exists(_strip(uri)))


def makedirs(uri: str | Path) -> None:
    """Create *uri* as a directory, including parents.

    Object stores have no directories; the call is a no-op there, which is why
    writers can call this unconditionally.
    """
    local = local_path(uri)
    if local is not None:
        local.mkdir(parents=True, exist_ok=True)
        return
    fs = filesystem(uri)
    try:
        fs.makedirs(_strip(uri), exist_ok=True)
    except (NotImplementedError, FileExistsError):
        pass


def ls(uri: str | Path, *, recursive: bool = False) -> list[str]:
    """List the children of *uri* (every file below it with ``recursive``).

    Returns fully-qualified URIs (protocol preserved) so results can be fed
    straight back into any other function here; :func:`relative` gives an
    entry's path below *uri*.
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
        if text.endswith("/"):
            continue
        if protocol and "://" not in text:
            text = f"{protocol}://{text.lstrip('/')}" if protocol != "file" else text
        out.append(text)
    return out


def _normalised(fs: Any, text: str) -> str:
    """*text* with dot segments resolved and a trailing slash dropped, in the filesystem's own
    spelling (some backends keep the scheme in ``_strip_protocol``, some drop it; both are
    normalised the same way, so the containment check sees the same shape on both sides). A
    surviving ``..`` can never cross the root, whatever the spelling."""
    protocol, _ = split_protocol(text)
    stripped = str(fs._strip_protocol(text)).rstrip("/")
    if protocol is None or protocol == "file":
        return os.path.normpath(stripped)
    return posixpath.normpath(stripped)


def relative(uri: str | Path, root: str | Path) -> str:
    """The ``/``-separated path of *uri* below the directory *root*, however either is spelled.

    Both are normalised as their filesystem normalises paths (dot segments resolved, a ``file://``
    or ``gs://`` prefix dropped), so an entry of :func:`ls` and the root it was listed from agree,
    and a ``..`` cannot escape the root: ``relative('/base/root/../../etc/passwd', '/base/root')``
    is refused, not returned.

    Raises:
        DataError: *uri* is not below *root*.
    """
    fs = filesystem(root)
    base = _normalised(fs, str(root))
    path = _normalised(fs, str(uri))
    if not path.startswith(f"{base}/"):
        raise DataError(f"{uri} is not below {root}")
    return path[len(base) + 1 :]


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
    source = local_path(remote)
    if source is not None:
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return target
    filesystem(remote).get_file(_strip(remote), str(target))
    return target


def info(uri: str | Path) -> dict[str, Any]:
    """Return the backend's metadata for *uri* (size, mtime, etag/generation)."""
    local = local_path(uri)
    if local is not None:
        stat = local.stat()
        return {"size": stat.st_size, "mtime": stat.st_mtime}
    return dict(filesystem(uri).info(_strip(uri)))


def publish(target: str | Path, write: Callable[[Path], Any]) -> None:
    """Materialise *target* atomically: *write* fills a temp file beside it, then one rename puts it in place.

    A reader of *target* sees either the previous file or the complete new one, never a half-written one
    -- which is what concurrent readers of a media cache, a rendered page or a downloaded corpus would
    otherwise see. The one home of the discipline: writers elsewhere call this instead of growing their
    own temp-file copies.

    Only a local path can publish by rename; a remote target raises, because object stores make each
    object visible whole anyway -- their writers use :func:`publish_bytes` or :func:`write_bytes`.

    Raises:
        ConfigError: *target* is a remote URI.
    """
    path = local_path(target)
    if path is None:
        raise ConfigError(
            f"publish needs a local target, got {target}",
            hint="object stores make each object visible whole: write_bytes them directly",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.{os.getpid()}.", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_bytes(target: str | Path, payload: bytes) -> None:
    """Write *payload* to *target*, atomically on a local filesystem (:func:`publish`).

    A remote URI takes the payload directly: an object store publishes per object, so there is no
    half-written file a concurrent reader could see.
    """
    if is_remote(target):
        write_bytes(target, payload)
        return
    publish(target, lambda tmp: tmp.write_bytes(payload))


__all__ = [
    "exists",
    "filesystem",
    "get",
    "info",
    "ls",
    "makedirs",
    "open_path",
    "parent_of",
    "publish",
    "publish_bytes",
    "read_bytes",
    "read_text",
    "relative",
    "write_bytes",
    "write_text",
]
