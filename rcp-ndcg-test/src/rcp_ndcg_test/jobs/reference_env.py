"""The wave's reference environments: one venv per family, keyed by the family's ``reference.lock``.

Owner decision 35: a wave builds **one reference environment per family**, not one per pod.  Each
environment is a venv over the engine image's torch/CUDA (``--system-site-packages``) with the family's
own lock pins installed into it (they take precedence over the image's copies), so one recipe's
transformers pin can no longer break the wave and every variant of a family reuses the same environment.
The venv's identity is the lock's SHA-256; the bootstrap builds each needed family once per pod and the
wave runner resolves ``<reference-root>/<family>/bin/python`` per recipe.

The exception is ``# own-torch: true`` in a family's ``reference.in`` (with its evidence): that family's
venv is built without ``--system-site-packages`` and the lock's own torch pin is the environment's.

This module owns the two pieces both the bootstrap and the tests need:

- :func:`family_rows` -- the wave's families, resolved from the recipe ids through the product's own
  loader (one home per concept: never a YAML read here), each with its lock path and own-torch flag;
- :func:`import_problems` -- the post-install check the bootstrap runs in the client environment against
  the family's python: torch imports (and, under own-torch, is the lock's pin), every pinned
  distribution is installed at its pinned version, and every pin imports.

``families`` and ``check`` are the CLI the bootstrap mounts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from rcp_ndcg_test.errors import HarnessError

from .reference_deps import canonical
from .reference_lock import LOCK_SCHEMA, _lock_header, _pinned_versions

__all__ = [
    "LockInfo",
    "family_rows",
    "import_problems",
    "main",
    "parse_lock",
    "reference_python",
]

_IMPORT_NAMES: dict[str, str] = {
    "flash-attn": "flash_attn",
    "flash-linear-attention": "fla",
    "huggingface-hub": "huggingface_hub",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "qwen-vl-utils": "qwen_vl_utils",
    "sentence-transformers": "sentence_transformers",
}
"""Distribution name -> module name where they differ (the check imports every pin)."""

_PROBE = r"""
import importlib
import importlib.metadata as metadata
import json
import sys

spec = json.loads(sys.argv[1])
problems = []
versions = {}
try:
    import torch

    versions["torch"] = torch.__version__
    versions["torch_cuda"] = torch.version.cuda
except Exception as error:  # noqa: BLE001 - the probe records the failure
    problems.append(f"torch does not import: {type(error).__name__}: {error}")
    versions["torch"] = None
    versions["torch_cuda"] = None


def release(version):
    return version.split("+", 1)[0].strip()


for name, pin in spec["pins"].items():
    try:
        seen = metadata.version(name)
    except metadata.PackageNotFoundError:
        problems.append(f"{name}=={pin} is not installed")
        continue
    versions[name] = seen
    if release(seen) != release(pin):
        problems.append(f"{name} is installed at {seen}, the lock pins {pin}")
    module = spec["imports"].get(name, name.replace("-", "_"))
    try:
        importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - the check fails loudly
        problems.append(f"{name}=={pin} does not import ({module}): {type(error).__name__}: {error}")
own_torch = spec["own_torch"]
if own_torch and "torch" in spec["pins"] and versions.get("torch") is not None:
    if release(versions["torch"]) != release(spec["pins"]["torch"]):
        problems.append(
            f"the own-torch family must see torch {spec['pins']['torch']}, the venv sees {versions['torch']}"
        )
print(json.dumps({"versions": versions, "problems": problems}))
"""


@dataclass(frozen=True)
class LockInfo:
    """One family's parsed ``reference.lock``.

    Attributes:
        family: The family directory name.
        image: The engine image the lock was generated against.
        own_torch: Whether the family declares its own torch stack (the exception).
        pins: ``{canonical distribution name: version}`` of the install pins.
        image_constraints: ``{canonical name: version}`` the family floors were checked against.
        sha256: The lock file's SHA-256 (the environment's identity).
    """

    family: str
    image: str
    own_torch: bool
    pins: dict[str, str]
    image_constraints: dict[str, str]
    sha256: str


def parse_lock(path: str | Path) -> LockInfo:
    """Parse a family's ``reference.lock``.

    Inputs: the lock's path.  Output: a :class:`LockInfo` (schema checked, family taken from the header,
    the install pins and the image constraints read from it).  Raises :class:`HarnessError` when the file
    is unreadable, the schema is not :data:`~rcp_ndcg_test.jobs.reference_lock.LOCK_SCHEMA`, or the
    family is missing.  Units: none.
    """
    import hashlib

    lock_path = Path(path)
    try:
        text = lock_path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"reference_env: {lock_path} cannot be read: {error}") from error
    header = _lock_header(text)
    if header.get("rcp-reference-lock") != LOCK_SCHEMA:
        raise HarnessError(
            f"reference_env: {lock_path} is not a {LOCK_SCHEMA} lock (schema "
            f"{header.get('rcp-reference-lock')!r}); regenerate it with jobs.reference_lock"
        )
    family = header.get("family")
    if not family:
        raise HarnessError(f"reference_env: {lock_path} names no family in its header")
    constraints: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("# image-constraint:"):
            continue
        body = line.removeprefix("# image-constraint:").strip()
        name, _, version = body.partition("==")
        constraints[canonical(name.strip())] = version.split(" ", 1)[0].strip()
    return LockInfo(
        family=family,
        image=header.get("image", ""),
        own_torch=header.get("own-torch", "false").lower() == "true",
        pins=_pinned_versions(text),
        image_constraints=constraints,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def family_rows(
    recipes_root: str | Path | None, recipe_ids: list[str]
) -> tuple[list[tuple[str, Path, bool]], dict[str, str]]:
    """The wave's families in first-appearance order, with their lock paths and own-torch flags.

    Inputs: the recipes root (default: the package's) and the wave's recipe ids.  Output:
    ``(rows, failures)`` -- one ``(family, lock path, own_torch)`` row per family the wave needs, and
    ``{recipe id: load error}`` for the ids that do not resolve (the wave reports them as failed rows;
    one failing recipe never stops the family selection).  Raises :class:`HarnessError` when the
    recipes root does not exist or a family has no ``reference.lock``.  Units: none.
    """
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    if not root.is_dir():
        raise HarnessError(f"reference_env: no recipes root at {root}")
    rows: list[tuple[str, Path, bool]] = []
    failures: dict[str, str] = {}
    seen: set[str] = set()
    for recipe_id in recipe_ids:
        try:
            recipe = load_recipe(recipe_id, root=root)
        except Exception as error:  # noqa: BLE001 - one failing recipe never stops the selection
            failures[recipe_id] = f"{type(error).__name__}: {error}"
            continue
        directory = recipe._dir
        if directory is None:  # pragma: no cover - load_recipe sets it
            failures[recipe_id] = "the recipe was loaded without a directory"
            continue
        family = directory.name
        if family in seen:
            continue
        seen.add(family)
        lock = directory / "reference.lock"
        if not lock.is_file():
            raise HarnessError(
                f"reference_env: family {family} has no reference.lock beside its reference.py; generate it "
                "with python -m rcp_ndcg_test.jobs.reference_lock"
            )
        rows.append((family, lock, parse_lock(lock).own_torch))
    return rows, failures


def reference_python(reference_root: str | Path, family: str) -> Path:
    """The python of a family's reference environment: ``<reference_root>/<family>/bin/python``."""
    return Path(reference_root) / family / "bin" / "python"


def environment_facts(recipe_dir: str | Path | None, reference_root: str | Path | None) -> dict[str, object]:
    """The reference environment facts ``equivalence.json`` records for one recipe (decision 35 item 5).

    Inputs: the recipe's family directory (the lock lives beside the reference) and the reference root
    the bootstrap built the family venvs under (``None`` when the reference runs outside a family
    environment).  Output: ``{"family": ..., "lock_sha256": ..., "freeze": [...]}`` -- the family
    name, the SHA-256 of its ``reference.lock`` (the environment's identity) and, when the bootstrap
    recorded it under ``<reference_root>/<family>/freeze.txt``, the venv's ``pip freeze`` lines.  Units:
    none.
    """
    import hashlib

    facts: dict[str, object] = {}
    if recipe_dir is None:
        return facts
    directory = Path(recipe_dir)
    family = directory.name
    facts["family"] = family
    lock = directory / "reference.lock"
    if lock.is_file():
        facts["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    if reference_root is not None:
        freeze = Path(reference_root) / family / "freeze.txt"
        if freeze.is_file():
            facts["freeze"] = [line.strip() for line in freeze.read_text(encoding="utf-8").splitlines() if line.strip()]
    return facts


def import_problems(python: str | Path, lock: LockInfo) -> tuple[list[str], dict[str, object]]:
    """The post-install check, run in ``python`` (the family venv), with the family's name on every line.

    Inputs: the family venv's python and its parsed lock.  Output: ``(problems, facts)`` -- the check
    failures (each prefixed with the family name) and the recorded versions (``torch``, ``torch_cuda``
    and one entry per pin).  The check imports torch (under ``own-torch`` it must be the lock's pin) and
    every pinned distribution (its installed version and its module import).  Raises
    :class:`HarnessError` when the probe subprocess itself fails (no python, no JSON).  Units: none.
    """
    spec = {
        "pins": lock.pins,
        "own_torch": lock.own_torch,
        "imports": {name: _IMPORT_NAMES.get(name, name.replace("-", "_")) for name in lock.pins},
    }
    completed = subprocess.run(
        [str(python), "-c", _PROBE, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise HarnessError(
            f"the reference environment check failed in {python} (exit {completed.returncode}): "
            f"{completed.stderr.strip()[-2000:]}"
        )
    try:
        document = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise HarnessError(f"the reference environment check wrote no JSON: {completed.stdout[-500:]!r}") from error
    problems = [f"family {lock.family}: {problem}" for problem in document.get("problems", [])]
    return problems, dict(document.get("versions", {}))


def _read_ids(value: str) -> list[str]:
    """The ``--recipes`` argument: ``@file`` (one id per line) or a comma-separated list."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise HarnessError(f"reference_env: no wave list at {path}")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``families`` (the wave's family rows) and ``check`` (the post-install import check)."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.jobs.reference_env", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    families = sub.add_parser("families", help="the wave's families, one 'family lock own_torch' row each")
    families.add_argument("--recipes-root", default=None)
    families.add_argument("--recipes", default=None, help="@file or a comma-separated list")
    families.add_argument("--family", default=None, help="one family directory name (no wave list)")
    check = sub.add_parser("check", help="check a family venv against its lock")
    check.add_argument("--python", required=True)
    check.add_argument("--lock", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "families":
            if arguments.family:
                root = Path(arguments.recipes_root) if arguments.recipes_root else default_recipes_root()
                lock = root / arguments.family / "reference.lock"
                if not lock.is_file():
                    raise HarnessError(f"reference_env: family {arguments.family} has no reference.lock under {root}")
                info = parse_lock(lock)
                print(f"{info.family}\t{lock}\t{'true' if info.own_torch else 'false'}")
                return 0
            if not arguments.recipes:
                parser.error("families needs --recipes or --family")
            rows, failures = family_rows(arguments.recipes_root, _read_ids(arguments.recipes))
            for recipe_id, error in failures.items():
                print(f"reference_env: recipe {recipe_id} does not load: {error}", file=sys.stderr)
            for family, lock, own_torch in rows:
                print(f"{family}\t{lock}\t{'true' if own_torch else 'false'}")
            return 0
        lock = parse_lock(arguments.lock)
        problems, facts = import_problems(arguments.python, lock)
        print(json.dumps({"family": lock.family, "lock_sha256": lock.sha256, "facts": facts}))
        for problem in problems:
            print(f"reference_env: {problem}", file=sys.stderr)
        return 1 if problems else 0
    except HarnessError as error:
        print(f"reference_env: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main() with argv
    raise SystemExit(main())
