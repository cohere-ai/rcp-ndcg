"""Group a wave's recipe ids by their engine image (owner decision 38, wired into decision 35).

A GPU job runs one container image: the pod's Python *is* the engine environment, so a wave list that
mixes recipes on different engine images cannot run in one job.  ``embeddinggemma-2`` pins a vLLM nightly
by digest while every other recipe stays on the released image; this module splits the requested variants
into one list per image (first-appearance order, recipe order preserved), and :func:`write_groups` writes
the filtered lists plus the ``wave-images.json`` map the submitter reads.

The grouping uses the product's own loader (one home per concept): a recipe id resolves to its resolved
``engine.image``, so a variant-level override would be honoured if the schema ever allowed one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rcp_ndcg_vllm.recipe import default_recipes_root, resolve_recipe

from rcp_ndcg_test.errors import HarnessError, RecipeError

__all__ = ["GROUP_MAP", "group_ids", "main", "slug_of", "write_groups"]

GROUP_MAP = "wave-images.json"
"""The stage-root map the submitter reads: ``{wave: {image: list file name}}``."""


def slug_of(image: str) -> str:
    """A file-name slug for one engine image (lowercase, runs of non-alphanumerics to one dash, clipped).

    Inputs: the image string (``repository:tag`` or ``repository:tag@sha256:...``).  Output: a slug that
    is stable for the image and short enough for a file name (a digest pin keeps its first 12 hex digits,
    so two digests of one tag never collide).  Units: none.
    """
    digest = ""
    match = re.search(r"@sha256:([0-9a-f]{12,})", image)
    if match:
        digest = "-" + match.group(1)[:12]
        image = image[: match.start()]
    slug = re.sub(r"[^a-z0-9]+", "-", image.lower()).strip("-")
    return (slug[:48] + digest) or "image"


def group_ids(recipe_ids: list[str], recipes_root: str | Path | None = None) -> dict[str, list[str]]:
    """Group recipe ids by their resolved engine image.

    Inputs: the wave's recipe ids and the recipes root (default: the package's).  Output: ``{image:
    [ids]}`` in first-appearance order; each list keeps the wave's order.  Raises
    :class:`HarnessError` when a recipe id does not resolve (the wave's own failure handling cannot
    group a recipe it cannot load, and a silently dropped id would lose a recipe).  Units: none.
    """
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    if not root.is_dir():
        raise HarnessError(f"wavegroups: no recipes root at {root}")
    groups: dict[str, list[str]] = {}
    for recipe_id in recipe_ids:
        try:
            recipe = resolve_recipe(recipe_id, root=root)
        except RecipeError as error:
            raise HarnessError(f"wavegroups: recipe {recipe_id} does not load: {error}") from error
        groups.setdefault(recipe.engine.image, []).append(recipe_id)
    return groups


def write_groups(out_dir: str | Path, wave: str, groups: dict[str, list[str]]) -> dict[str, str]:
    """Write one filtered wave list per engine image; return ``{image: file name}``.

    Inputs: the output directory (the stage's ``wave-lists/``), the wave's name and the groups from
    :func:`group_ids`.  Output: the image -> file-name map (``<wave>.<slug>.txt``, one id per line,
    sorted slugs for determinism); the files land in ``out_dir``.  Units: none.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for image in sorted(groups, key=slug_of):
        name = f"{wave}.{slug_of(image)}.txt"
        (out / name).write_text("".join(f"{recipe_id}\n" for recipe_id in groups[image]), encoding="utf-8")
        mapping[image] = name
    return mapping


def _read_ids(value: str) -> list[str]:
    """The ``--recipes`` argument: ``@file`` (one id per line) or a comma-separated list."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise HarnessError(f"wavegroups: no wave list at {path}")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    """The CLI: group a wave list and print the plan as JSON; ``--out`` also writes the filtered lists.

    With ``--out <wave-lists dir>`` the per-image lists are written and the stage-root
    ``wave-images.json`` is updated (one key per wave), which is what ``rc_build.sh`` stages and
    ``submit.sh`` reads.
    """
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.jobs.wavegroups", description=__doc__)
    parser.add_argument("--wave", required=True, help="the wave's name (the list files' prefix)")
    parser.add_argument("--recipes", required=True, help="@file or a comma-separated list of recipe ids")
    parser.add_argument("--recipes-root", default=None, help="recipe root (default: the package's)")
    parser.add_argument("--out", default=None, help="the stage's wave-lists directory (writes the lists)")
    parser.add_argument("--map-out", default=None, help="the stage root (writes/updates wave-images.json)")
    args = parser.parse_args(argv)
    try:
        groups = group_ids(_read_ids(args.recipes), args.recipes_root)
    except HarnessError as error:
        print(f"wavegroups: {error}", file=sys.stderr)
        return 2
    mapping = write_groups(args.out, args.wave, groups) if args.out else {}
    if args.map_out:
        path = Path(args.map_out) / GROUP_MAP
        document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        document[args.wave] = mapping
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wave": args.wave,
                "groups": [
                    {"image": image, "ids": groups[image], "file": mapping.get(image)}
                    for image in sorted(groups, key=slug_of)
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() with argv
    raise SystemExit(main())
