#!/usr/bin/env python3
"""Check ``requirements-constraints.txt`` against ``uv.lock``: the same pins, however uv spells them.

CI and the release both run this script -- one implementation, never the first test at tag time. The comparison
is semantic: ``{name: version}`` of every exact pin. A newer uv may re-serialise markers, quoting, ``# via``
comments and line layout of the same export; that is not a difference. A moved, added or dropped pin is.

Usage::

    python .github/scripts/check_constraints.py                  # fresh lock export vs requirements-constraints.txt
    python .github/scripts/check_constraints.py --write          # regenerate requirements-constraints.txt
    python .github/scripts/check_constraints.py exported committed  # compare two requirements files' pins

Exit code 0: the pins agree. 1: they differ (each difference is printed) or the input is unusable.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["EXPORT_ARGV", "pins"]

ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = ROOT / "requirements-constraints.txt"

#: The export the constraints file is, for the coordinator's extras (``rcp_ndcg.runners.script.COORDINATOR_EXTRAS``;
#: tests/docs/test_packaging.py cross-checks the two). ``uv`` records this very command in the file's header, so the
#: header stays a truthful regeneration instruction and there is one home for the argv.
EXPORT_ARGV = (
    "uv",
    "export",
    "--frozen",
    "--no-hashes",
    "--no-emit-workspace",
    "--no-dev",
    "--extra",
    "calibrate",
    "--extra",
    "hf",
    "--extra",
    "s3",
    "--extra",
    "azure",
    "-o",
    "requirements-constraints.txt",
)

# name, optional [extras], then the specifier: "torch==2.8.0; ..." -> ("torch", "==2.8.0")
_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def pins(text: str) -> dict[str, str]:
    """``{canonical name: version}`` of every exact pin in a requirements file (comments, markers and layout dropped).

    Args:
        text: The contents of a requirements/constraints file.

    Returns:
        One entry per pinned distribution, keys PEP 503 canonical, values the pinned version.

    Raises:
        SystemExit: A non-empty, non-option line does not carry exactly one exact pin.
    """
    out: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue  # option lines (e.g. --index-url) are not pins
        requirement = line.split(";", 1)[0].strip()
        match = _REQUIREMENT_RE.match(requirement)
        if match is None:
            raise SystemExit(f"line {number}: not a requirement: {line!r}")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        specifier = re.sub(r"[\s()]", "", match.group(2))
        clauses = specifier.split(",")
        if len(clauses) != 1 or not clauses[0].startswith("==") or clauses[0].startswith("==="):
            raise SystemExit(f"line {number}: not one exact pin (==version): {line!r}")
        version = clauses[0].removeprefix("==")
        if out.get(name, version) != version:
            raise SystemExit(f"line {number}: {name} pinned to two versions ({out[name]} and {version})")
        out[name] = version
    return out


def main(argv: list[str]) -> int:
    if argv == ["--write"]:
        subprocess.run(EXPORT_ARGV, cwd=ROOT, check=True)
        return 0
    if argv == []:
        with tempfile.TemporaryDirectory() as tmp:
            exported_path = Path(tmp) / "exported.txt"
            # uv publishes the file atomically (replaces it): read it back by name, never through a handle.
            subprocess.run([*EXPORT_ARGV[:-2], "-o", str(exported_path)], cwd=ROOT, check=True)
            fresh = pins(exported_path.read_text(encoding="utf-8"))
        committed = pins(CONSTRAINTS.read_text(encoding="utf-8"))
    elif len(argv) == 2:
        fresh = pins(Path(argv[0]).read_text(encoding="utf-8"))
        committed = pins(Path(argv[1]).read_text(encoding="utf-8"))
    else:
        raise SystemExit(__doc__)
    drift = [
        f"{name}: lock's export has {fresh.get(name)}, the file has {committed.get(name)}"
        for name in sorted(set(fresh) | set(committed))
        if fresh.get(name) != committed.get(name)
    ]
    if drift:
        print("requirements-constraints.txt is not the lock's export (regenerate with --write):")
        print("\n".join(f"  {line}" for line in drift))
        return 1
    print(f"{len(committed)} pins agree with the lock's export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
