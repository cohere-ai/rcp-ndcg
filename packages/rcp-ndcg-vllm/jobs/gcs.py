#!/usr/bin/env python3
"""The GCS transfer helper of the node scripts (one home for the gs:// copy/list/remove concept).

The stock engine image ships neither ``gcloud`` nor ``gsutil``; when both are missing, ``wave0.sh``
and ``bootstrap.sh`` source ``gcs.sh``, install ``gcsfs`` into a tools directory **outside the engine
environment** (``pip install --target``, the same way uv gets installed), and run this helper with the
credentials the mounted auth script set up (Application Default Credentials: ``GOOGLE_APPLICATION_CREDENTIALS``,
the gcloud ADC file, or the metadata server). Stdlib besides ``fsspec``/``gcsfs``.

    gcs.py cp SRC DST        one file or a directory (recursive, by the source's kind), either side gs://
    gcs.py ls URI            the entries under a gs:// prefix
    gcs.py rm URI            one object

The functions take the filesystem as an argument, so the tests run them against a fake with the fsspec
API and no network; the CLI builds the real one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["gcs_copy", "gcs_list", "gcs_remove", "main", "make_filesystem"]


def make_filesystem() -> Any:
    """The gs:// filesystem: ``gcsfs`` with Application Default Credentials (``token=None``)."""
    from fsspec import filesystem

    return filesystem("gs", token=None)


def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def _split(uri: str) -> str:
    """The path inside the bucket (gcsfs wants the bare path)."""
    return uri[len("gs://") :] if _is_gcs(uri) else uri


def _remote(fs: Any, uri: str) -> bool:
    return _is_gcs(uri)


def gcs_copy(src: str, dst: str, *, fs: Any = None) -> list[str]:
    """Copy one file or directory between gs:// and the local filesystem (either side either way).

    A directory source copies recursively (every file under it, same relative layout); the recursion is
    decided by the source's kind, never by a flag the caller can get wrong. Returns the written target
    paths (for tests and the caller's summary).
    """
    fs = fs or make_filesystem()
    src_remote = _is_gcs(src)
    src_path, dst_path = _split(src), _split(dst)
    src_is_directory = bool(fs.isdir(src_path.rstrip("/"))) if src_remote else Path(src_path).is_dir()

    written: list[str] = []
    if not src_is_directory:
        target = dst_path
        if _is_gcs(dst):
            if fs.isdir(_split(dst).rstrip("/")) or dst.endswith("/"):
                target = f"{dst_path.rstrip('/')}/{os.path.basename(src_path)}"
            fs.put_file(src_path, target)
            return [f"gs://{target}"]
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(src_path, target)
        return [str(target)]

    prefix = src_path.rstrip("/") + "/"
    if src_remote:
        entries = [entry for entry in fs.find(prefix) if not entry.endswith("/")]
    else:
        entries = [str(path).replace(os.sep, "/") for path in Path(prefix).rglob("*") if path.is_file()]
    for entry in entries:
        relative = entry[len(prefix) :] if entry.startswith(prefix) else entry
        target = f"{dst_path.rstrip('/')}/{relative}"
        if _is_gcs(dst):
            fs.put_file(entry, target)
            written.append(f"gs://{target}")
        else:
            local_target = Path(target)
            local_target.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(entry, str(local_target))
            written.append(str(local_target))
    return written


def gcs_list(uri: str, *, fs: Any = None) -> list[str]:
    """The entries under a ``gs://`` prefix, as full ``gs://`` URIs, sorted."""
    fs = fs or make_filesystem()
    prefix = _split(uri).rstrip("/")
    entries = fs.find(prefix)
    return sorted(f"gs://{entry}" for entry in entries if not entry.endswith("/"))


def gcs_remove(uri: str, *, fs: Any = None) -> None:
    """Delete one gs:// object (a file, not a tree: the callers remove what they wrote)."""
    fs = fs or make_filesystem()
    fs.rm(_split(uri))


def main(argv: list[str] | None = None) -> int:
    """The CLI the node scripts call; exit 0 on success, 1 on a failed transfer."""
    parser = argparse.ArgumentParser(prog="gcs.py", description="gs:// copy, list and remove over gcsfs (ADC).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_cp = sub.add_parser("cp", help="copy a file or directory, either side gs://")
    p_cp.add_argument("src")
    p_cp.add_argument("dst")
    p_ls = sub.add_parser("ls", help="list a gs:// prefix")
    p_ls.add_argument("uri")
    p_rm = sub.add_parser("rm", help="delete one gs:// object")
    p_rm.add_argument("uri")
    args = parser.parse_args(argv)
    try:
        if args.command == "cp":
            for target in gcs_copy(args.src, args.dst):
                print(target)
        elif args.command == "ls":
            for entry in gcs_list(args.uri):
                print(entry)
        else:
            gcs_remove(args.uri)
    except Exception as error:  # noqa: BLE001 - the caller sees the one-line failure
        print(f"gcs: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
