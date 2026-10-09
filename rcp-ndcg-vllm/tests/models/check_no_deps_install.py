"""Simulate the engine environment's ``pip install --no-deps`` freeze check.

The binding rule for the plugin wheel: installing it with
``pip install --no-deps`` into an environment that already has vLLM must
change ``pip freeze`` by exactly this one distribution — nothing pulled in,
nothing upgraded.  The GPU wave runs this check around the real install into
the engine environment (GPU-VALIDATION.md, engine environment item 1); this
script is the CPU simulation and the same reusable check:

    python tests/check_no_deps_install.py <wheel> [--python <interpreter>]

It creates a fresh venv (offline: ``pip install --no-index --no-deps``),
records ``pip freeze`` before and after, and asserts the delta is exactly the
wheel's distribution.  It also asserts the wheel is pure Python (no compiled
extensions) and targets ``py3-none-any`` — the properties that make
``--no-deps`` safe in the engine image.

Exit 0 and a one-line summary on success; exit 1 with the offending diff on
failure.  Stdlib only, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

__all__ = [
    "assert_pure_python",
    "check_install",
    "main",
    "pick_base_python",
    "wheel_requirement",
]

PY3_NONE_ANY = "py3-none-any"
FORBIDDEN_SUFFIXES = (".so", ".pyd", ".dylib", ".dll", ".exe")


def _run(cmd: list[str]) -> str:
    """Run a command, return stdout; raise with stderr on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result.stdout or ""


def wheel_dist_info(wheel: Path, filename: str) -> str:
    """A file's text from the wheel's single ``<dist>.dist-info`` directory."""
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            info.filename
            for info in archive.infolist()
            if info.filename.count("/") == 1 and info.filename.endswith(".dist-info/" + filename)
        ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} in {wheel.name}, got {matches}")
    with zipfile.ZipFile(wheel) as archive:
        return archive.read(matches[0]).decode()


def wheel_requirement(wheel: Path) -> tuple[str, str]:
    """(wheel tag, freeze line) from the wheel's metadata.

    The freeze line is the exact ``pip freeze`` entry a successful install
    adds, e.g. ``rcp-ndcg-vllm-topk==0.0.1``.
    """
    tag = wheel_dist_info(wheel, "WHEEL").split("Tag: ", 1)[1].splitlines()[0]
    metadata = wheel_dist_info(wheel, "METADATA")
    fields = dict(line.split(": ", 1) for line in metadata.splitlines() if ": " in line and not line.startswith(" "))
    return tag, f"{fields['Name']}=={fields['Version']}"


def assert_pure_python(wheel: Path) -> str:
    """The wheel is ``py3-none-any``, carries no compiled extension and
    declares only the stock image's own dependencies (pydantic, PyYAML).

    Returns the wheel tag on success; raises RuntimeError otherwise.  A
    compiled artifact (or a platform-specific tag) would break the engine
    install; a declared dependency would make a ``--no-deps`` install
    incomplete, which is exactly the regression class this check exists to
    catch (the freeze delta alone cannot: ``--no-deps`` never installs
    dependencies).
    """
    tag, _ = wheel_requirement(wheel)
    if tag != PY3_NONE_ANY:
        raise RuntimeError(f"wheel tag is {tag!r}, expected {PY3_NONE_ANY!r}")
    with zipfile.ZipFile(wheel) as archive:
        binaries = [info.filename for info in archive.infolist() if info.filename.endswith(FORBIDDEN_SUFFIXES)]
    if binaries:
        raise RuntimeError(f"wheel contains compiled artifacts: {binaries}")
    metadata = wheel_dist_info(wheel, "METADATA")
    requires_dist = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    # The stock vLLM image ships pydantic and PyYAML: declaring exactly those is what --no-deps assumes
    # (anything else would be left uninstalled in the engine image).
    image_ships = {"pydantic", "pyyaml"}
    for line in requires_dist:
        if "; extra ==" in line:
            continue  # an extra is opt-in: --no-deps never installs it
        requirement = line.split(":", 1)[1].split(";")[0].strip()
        name = re.split(r"[<>=!~(\[ ]", requirement, maxsplit=1)[0].strip().lower()
        if name not in image_ships:
            raise RuntimeError(
                f"wheel declares a dependency the stock image does not ship ({line.strip()!r}); "
                "--no-deps would leave it uninstalled in the engine image"
            )
    return tag


def _has_ensurepip(python: str) -> bool:
    """Whether ``python -m ensurepip`` is usable (uv-managed interpreters ship
    without the pip wheels, so ``venv.create(with_pip=True)`` fails on them)."""
    probe = subprocess.run([python, "-c", "import ensurepip"], capture_output=True, timeout=60)
    return probe.returncode == 0


def pick_base_python(candidates: list[str] | None = None) -> str:
    """Return the first candidate interpreter whose ensurepip works.

    Falls back to conventional system interpreters when the running Python is
    a uv-managed build without bundled pip (a venv from it cannot install
    pip, and the whole point of the check is a real ``pip install``).
    """
    if candidates is None:
        import shutil

        candidates = [sys.executable, shutil.which("python3") or "/usr/bin/python3"]
    for python in candidates:
        if _has_ensurepip(python):
            return python
    raise RuntimeError(
        "no interpreter with a working ensurepip found (tried: "
        f"{candidates}); pass --python <interpreter> with a standard build"
    )


def _canonical_name(requirement_line: str) -> str:
    """The PEP 503 canonical distribution name of a ``pip freeze`` line
    (``name==version``, or ``name @ file://...`` for a direct-path install)."""
    name = requirement_line.split("==")[0].split("@")[0].strip().rstrip("#")
    return name.replace("_", "-").lower()


def check_install(wheel: Path, venv_dir: Path, base_python: str | None = None) -> str:
    """Install ``wheel`` with ``--no-deps`` into a fresh venv; return the
    freeze line, which must be the only change to ``pip freeze``.

    ``--no-index`` makes the install offline: a wheel that needed a dependency
    would fail here instead of silently reaching for the network.  The venv is
    created from ``base_python`` (default: a probed interpreter whose
    ensurepip works).
    """
    tag, requirement = wheel_requirement(wheel)
    if tag != PY3_NONE_ANY:
        raise RuntimeError(f"wheel tag is {tag!r}, expected {PY3_NONE_ANY!r}")
    base = base_python or pick_base_python()
    _run([base, "-m", "venv", str(venv_dir)])
    pip = [
        str(venv_dir / "bin" / "python"),
        "-m",
        "pip",
        "--disable-pip-version-check",
    ]

    freeze_before = set(_run([*pip, "freeze"]).splitlines())
    _run([*pip, "install", "--no-index", "--no-deps", str(wheel)])
    freeze_after = set(_run([*pip, "freeze"]).splitlines())

    added = freeze_after - freeze_before
    removed = freeze_before - freeze_after
    if removed:
        raise RuntimeError(f"--no-deps install removed or upgraded: {sorted(removed)}")
    if len(added) != 1 or _canonical_name(next(iter(added))) != _canonical_name(requirement):
        raise RuntimeError(
            f"--no-deps install changed pip freeze by {sorted(added)}, "
            f"expected exactly the one distribution {requirement!r}"
        )
    return requirement


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, run the check, print the verdict."""
    docline = (__doc__ or "").splitlines()[0]
    parser = argparse.ArgumentParser(description=docline)
    parser.add_argument("wheel", type=Path, help="path to the built wheel")
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "base interpreter for the fresh venv (default: the running "
            "Python, falling back to a system Python whose ensurepip works)"
        ),
    )
    args = parser.parse_args(argv)

    assert_pure_python(args.wheel)
    base_python = args.python or pick_base_python()
    with tempfile.TemporaryDirectory(prefix="rcp-vllm-topk-freeze-") as tmp:
        requirement = check_install(args.wheel, Path(tmp) / "venv", base_python)
    print(f"OK: --no-deps install changed pip freeze by exactly [{requirement}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
