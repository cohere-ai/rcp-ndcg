"""URI parsing shared by the storage layer.

One place decides what counts as a remote URI, so callers never grow their own
``startswith("gs://")`` checks.  Windows drive letters (``C:\\...``) are
deliberately *not* treated as protocols.
"""

from __future__ import annotations

import re
from pathlib import Path

from rcp_ndcg.errors import ConfigError

#: A protocol is at least two characters so ``C:\\Users`` stays a local path.
_PROTOCOL_RE = re.compile(r"^(?P<protocol>[A-Za-z][A-Za-z0-9+.\-]+)://(?P<rest>.*)$", re.DOTALL)

#: Protocols whose first path segment is an object, not a bucket or host.
_ROOTLESS = frozenset({"memory"})


def split_protocol(uri: str | Path) -> tuple[str | None, str]:
    """Return ``(protocol, remainder)``; protocol is ``None`` for local paths."""
    text = str(uri)
    match = _PROTOCOL_RE.match(text)
    if match is None:
        return None, text
    return match.group("protocol").lower(), match.group("rest")


def protocol_of(uri: str | Path) -> str:
    """Return the fsspec protocol for *uri* (``"file"`` for local paths)."""
    protocol, _ = split_protocol(uri)
    return protocol or "file"


def is_remote(uri: str | Path) -> bool:
    """Return ``True`` if *uri* names an object on a remote filesystem."""
    protocol, _ = split_protocol(uri)
    return protocol is not None and protocol != "file"


def local_dir(path: str | Path, what: str) -> Path:
    """``path`` as a local path: run directories and judgement stores are written locally, never to a bucket.

    Raises:
        ConfigError: ``path`` is a remote URI (``gs://``, ``s3://``, ...); ``what`` names it in the message.
    """
    if is_remote(path):
        raise ConfigError(
            f"{what} must be a local or shared-filesystem path, not {path}",
            hint="write locally and copy it to the bucket as it grows with --mirror <uri> (or `mirror:` in the config)",
        )
    return Path(path)


def join(base: str | Path, *parts: str) -> str:
    """Join URI segments without collapsing the ``scheme://`` separator.

    ``pathlib`` and ``os.path`` both mangle ``gs://bucket`` (the double slash
    survives only by accident), so remote paths get their own joiner.
    """
    if not is_remote(base):
        return str(Path(base).joinpath(*parts))
    text = str(base).rstrip("/")
    for part in parts:
        text = f"{text}/{str(part).strip('/')}"
    return text


def local_path(uri: str | Path) -> Path | None:
    """*uri* as a local path when it names one (a plain path or a ``file://`` URL), else ``None``.

    The local fast paths route through this, so a ``file://`` spelling gets the same answer as the
    plain path (``Path('file:///x')`` is a relative directory literally named ``file:``) and a remote
    URI never reaches a ``Path``.
    """
    protocol, rest = split_protocol(uri)
    if protocol is None:
        return Path(str(uri))
    if protocol == "file":
        return Path(rest if rest.startswith("/") else f"/{rest}")
    return None


def parent(uri: str | Path) -> str:
    """Return *uri* without its final segment."""
    text = str(uri).rstrip("/")
    file_local = local_path(text)
    if file_local is not None:
        return str(file_local.parent)
    protocol, rest = split_protocol(text)
    rest = rest.lstrip("/")
    if "/" not in rest:
        # An object at the root of a bucket or store: its parent is the root
        # itself, never ``"<scheme>:/"`` (which reads as a *local* path downstream).
        return f"{protocol}://" if protocol in _ROOTLESS else f"{protocol}://{rest}"
    return f"{protocol}://{rest.rpartition('/')[0]}"


__all__ = [
    "is_remote",
    "join",
    "local_dir",
    "local_path",
    "parent",
    "protocol_of",
    "split_protocol",
]
