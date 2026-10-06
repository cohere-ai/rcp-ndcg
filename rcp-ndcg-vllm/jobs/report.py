#!/usr/bin/env python3
"""The node scripts' report: a tiny stdlib-only JSON report assembler.

``bootstrap.sh`` and ``wave0.sh`` merge step fragments into one report file as the job goes, so a
fail-fast death still leaves the report of everything that ran; the finished report is printed to
stdout and uploaded with the wave's outputs. Stdlib only (it runs on the image's python3 before the
client environment exists).

    report.py init  --file F --schema NAME [--started ISO]
    report.py merge --file F --key K (--fragment FILE | --literal JSON)
    report.py fail  --file F --step NAME --reason LINE
    report.py emit  --file F            (prints the report; the caller uploads it)

``merge`` replaces the key's value whole (a step owns its section); ``fail`` records the one-line
reason of the step that failed, sets ``passed: false`` and leaves everything else as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["emit", "fail", "init_report", "main", "merge"]

SCHEMA_FIELD = "schema"


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path: str, document: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def init_report(file: str, schema: str, started: str | None) -> dict[str, Any]:
    """A fresh report: the schema name, the start time and ``passed: true`` until a step fails."""
    document: dict[str, Any] = {
        SCHEMA_FIELD: schema,
        "started": started or _now(),
        "passed": True,
    }
    _save(file, document)
    return document


def merge(file: str, key: str, fragment: dict[str, Any]) -> dict[str, Any]:
    """Set ``key`` to the fragment (a step owns its section whole) and write the report back.

    A dotted ``key`` (``reach.hub``) merges one level deep, so a step with several probes can own a
    section: ``merge reach.hub`` sets ``report["reach"]["hub"]``, creating ``reach`` when absent.
    """
    document = _load(file)
    parts = key.split(".")
    target = document
    for part in parts[:-1]:
        node = target.setdefault(part, {})
        if not isinstance(node, dict):
            raise SystemExit(f"merge: {part} of {key} is not a JSON object in {file}")
        target = node
    target[parts[-1]] = fragment
    _save(file, document)
    return document


def fail(file: str, step: str, reason: str) -> dict[str, Any]:
    """Record the failing step and its one-line reason; the caller exits non-zero after this."""
    document = _load(file)
    document["passed"] = False
    document["failed_step"] = step
    document["error"] = reason
    _save(file, document)
    return document


def emit(file: str) -> dict[str, Any]:
    """Print the report (the caller uploads it and cats it to the job's stdout)."""
    document = _load(file)
    json.dump(document, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return document


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    """The CLI; exit 2 on a bad request, 0 otherwise (a failed step is recorded, not raised)."""
    parser = argparse.ArgumentParser(prog="report.py", description="Merge JSON fragments into a report file.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="start a fresh report")
    p_init.add_argument("--file", required=True)
    p_init.add_argument("--schema", required=True)
    p_init.add_argument("--started", default=None)
    p_merge = sub.add_parser("merge", help="set one key from a fragment (a file, or --literal JSON)")
    p_merge.add_argument("--file", required=True)
    p_merge.add_argument("--key", required=True)
    p_merge.add_argument("--fragment", default=None)
    p_merge.add_argument("--literal", default=None)
    p_fail = sub.add_parser("fail", help="record the step that failed and why")
    p_fail.add_argument("--file", required=True)
    p_fail.add_argument("--step", required=True)
    p_fail.add_argument("--reason", required=True)
    p_emit = sub.add_parser("emit", help="print the report")
    p_emit.add_argument("--file", required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        init_report(args.file, args.schema, args.started)
    elif args.command == "merge":
        if args.fragment is None and args.literal is None:
            parser.error("merge needs --fragment FILE or --literal JSON")
        fragment = json.loads(args.literal) if args.literal is not None else _load(args.fragment)
        merge(args.file, args.key, fragment)
    elif args.command == "fail":
        fail(args.file, args.step, args.reason)
    else:
        emit(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
