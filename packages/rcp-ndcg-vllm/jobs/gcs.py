#!/usr/bin/env python3
"""The GCS transfer helper of the node scripts (one home for the gs:// copy/list/remove concept).

The stock engine image ships neither ``gcloud`` nor ``gsutil``; when both are missing, ``wave0.sh``
and ``bootstrap.sh`` source ``gcs.sh``, install ``gcsfs`` into a tools directory **outside the engine
environment** (``pip install --target``, the same way uv gets installed), and run this helper with the
credentials the mounted auth script set up (Application Default Credentials: ``GOOGLE_APPLICATION_CREDENTIALS``,
the gcloud ADC file, or the metadata server). Stdlib besides ``fsspec``/``gcsfs``.

    gcs.py cp SRC DST [dir|file|auto]
                             one file or a directory, either side gs:// (the kind comes from the caller
                             when declared, else from the source itself)
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


def gcs_copy(src: str, dst: str, *, fs: Any = None, kind: str = "auto") -> list[str]:
    """Copy one file or directory between gs:// and the local filesystem (either side either way).

    A directory source copies recursively (every file under it, same relative layout). The kind comes
    from the caller when it declares one (``dir``/``file`` — the shell dispatch's contract, honoured
    before any inference); ``auto`` decides by the source's kind on disk/service.
    """
    fs = fs or make_filesystem()
    src_remote = _is_gcs(src)
    src_path, dst_path = _split(src), _split(dst)
    if kind == "auto":
        src_is_directory = bool(fs.isdir(src_path.rstrip("/"))) if src_remote else Path(src_path).is_dir()
    elif kind == "dir":
        src_is_directory = True
    else:
        src_is_directory = False

    written: list[str] = []
    if not src_is_directory:
        if _is_gcs(dst):
            target = dst_path
            if fs.isdir(_split(dst).rstrip("/")) or dst.endswith("/"):
                target = f"{dst_path.rstrip('/')}/{os.path.basename(src_path)}"
            if src_remote:
                # gcsfs's put_file opens the source with a local open: a remote source goes through a
                # temporary local file (the CLI branches of gcloud/gsutil copy remote->remote natively).
                import tempfile

                with tempfile.TemporaryDirectory() as work:
                    local = Path(work) / os.path.basename(src_path)
                    fs.get_file(src_path, str(local))
                    fs.put_file(str(local), target)
            else:
                fs.put_file(src_path, target)
            return [f"gs://{target}"]
        target = Path(dst_path)
        if target.is_dir() or dst.endswith("/"):
            target = target / os.path.basename(src_path)  # cp semantics: a directory dest keeps the name
        target.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(src_path, str(target))
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
    p_cp.add_argument(
        "kind",
        nargs="?",
        default="auto",
        choices=["dir", "file", "auto"],
        help="the source's kind; auto decides by the source itself",
    )
    p_ls = sub.add_parser("ls", help="list a gs:// prefix")
    p_ls.add_argument("uri")
    p_rm = sub.add_parser("rm", help="delete one gs:// object")
    p_rm.add_argument("uri")
    args = parser.parse_args(argv)
    try:
        if args.command == "cp":
            for target in gcs_copy(args.src, args.dst, kind=args.kind):
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
