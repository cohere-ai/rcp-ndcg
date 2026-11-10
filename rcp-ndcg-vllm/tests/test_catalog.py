"""The catalog table in ``README.md`` against ``iter_recipes()``: one pin, so a row cannot drift.

The README's "The recipes" table is the rendered catalog (``docs/reference/recipes.md`` delegates to it).
Every row is checked against the resolved recipe: family, id, model, role, input, the MRL cell, whether the
recipe needs its model plugin, the reference cell and the status state. The test is offline (package data and
PyYAML only).
"""

from __future__ import annotations

import re
from pathlib import Path

from rcp_ndcg_vllm.recipe import iter_families, iter_recipes

README = Path(__file__).resolve().parents[1] / "README.md"
RECIPES_ROOT = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_vllm" / "recipes"

_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
)


def _catalog_rows() -> dict[str, tuple[str, str, str, str, str, str, str, str]]:
    rows: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        family, recipe_id, model, role, inputs, mrl, plugin, reference, status = (
            group.strip() for group in match.groups()
        )
        rows[recipe_id] = (family, model, role, inputs, mrl, plugin, reference, status)
    return rows


def _reference_header(family_id: str) -> str:
    """The first line of a family's ``reference.py``: the reference implementation names its own code path there."""
    return (RECIPES_ROOT / family_id / "reference.py").read_text(encoding="utf-8").splitlines()[0]


def test_the_readme_catalog_matches_iter_recipes() -> None:
    """Every variant -- retrieval and judge -- has exactly one row, and every row names its resolved facts."""
    rows = _catalog_rows()
    recipes = {recipe.id: recipe for recipe in iter_recipes()}
    assert set(rows) == set(recipes), (
        f"catalog rows out of step with iter_recipes(): missing {sorted(set(recipes) - set(rows))}, "
        f"extra {sorted(set(rows) - set(recipes))}"
    )
    families = {family.id: family for family in iter_families()}
    family_of = {variant.id: family.id for family in families.values() for variant in family.variants}
    for recipe_id, (family_id, model, role, inputs, mrl, plugin, reference, status) in rows.items():
        recipe = recipes[recipe_id]
        assert family_id == family_of[recipe_id]
        assert model == recipe.model, f"{recipe_id}: README model {model!r} != recipe {recipe.model!r}"
        assert role == recipe.role, f"{recipe_id}: README role {role!r} != recipe {recipe.role!r}"
        assert inputs == ", ".join(recipe.input), f"{recipe_id}: README input {inputs!r} != {recipe.input}"
        assert (plugin != "—") == (recipe.serve.plugin is not None), (
            f"{recipe_id}: README plugin {plugin!r} does not match serve.plugin {recipe.serve.plugin!r}"
        )
        assert status == recipe.status.state, f"{recipe_id}: README status {status!r} != {recipe.status.state!r}"
        # The reference cell: a judge recipe carries no equivalence reference ("—"); a retrieval family's
        # reference.py names its own code path in its header -- the paper's code (ported from experiments/paper/)
        # or the model card's published usage.
        if role == "judge":
            assert reference == "—", f"{recipe_id}: a judge recipe's reference cell is {reference!r}"
        else:
            assert reference in {"paper", "card"}, f"{recipe_id}: unknown reference cell {reference!r}"
            from_the_paper = "the paper" in _reference_header(family_id)
            assert (reference == "paper") == from_the_paper, (
                f"{recipe_id}: reference {reference!r} does not match the family reference's header"
            )
        # The MRL cell is free-form prose (a set, a range, "none" or "—"); it must name the declared kind
        # when the recipe declares one, and the declared dims must all appear.
        kind = recipe.client.get("mrl_kind")
        if kind is not None:
            assert kind in mrl, f"{recipe_id}: README MRL {mrl!r} does not name mrl_kind {kind!r}"
            dims = tuple(recipe.client.get("mrl_dims") or ()) + tuple(recipe.client.get("mrl_range") or ())
            for value in dims:
                assert str(value) in mrl, f"{recipe_id}: README MRL {mrl!r} omits {value}"
        else:
            assert mrl, f"{recipe_id}: README MRL cell is empty"


_WORKSPACE_MODULES = frozenset({"rcp_ndcg", "rcp_core", "rcp_ndcg_vllm", "rcp_ndcg_test"})
_WORKSPACE_DISTRIBUTIONS = frozenset(name.replace("_", "-") for name in _WORKSPACE_MODULES)


def test_no_reference_subprocess_imports_the_product() -> None:
    """A family's ``reference.py`` is a dependency-free reference subprocess (AGENTS.md): it must not
    import the product. A workspace import makes the reference check the client against itself, so the
    reference loads the checkpoint's own code path instead (vllm-6)."""
    import ast

    offenders: list[str] = []
    for family in iter_families():
        path = RECIPES_ROOT / family.id / "reference.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                if name.split(".", 1)[0] in _WORKSPACE_MODULES:
                    offenders.append(f"{family.id}/reference.py:{node.lineno}: {name}")
    assert not offenders, "reference subprocesses must not import the product:\n" + "\n".join(offenders)


def test_no_reference_environment_pins_a_workspace_distribution() -> None:
    """A family's ``reference.in`` must not pin a workspace distribution: the pin couples the reference
    environment to the release under test (the staged wheelhouse carries the workspace wheels), so the
    reference could never run against another checkout (vllm-6)."""
    offenders: list[str] = []
    for family in iter_families():
        path = RECIPES_ROOT / family.id / "reference.in"
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            requirement = line.split("#", 1)[0].strip()
            if requirement and requirement.split("==", 1)[0].strip().lower() in _WORKSPACE_DISTRIBUTIONS:
                offenders.append(f"{family.id}/reference.in:{number}: {requirement}")
    assert not offenders, "reference.in files must not pin a workspace distribution:\n" + "\n".join(offenders)


#: The owner's explicit disposition list for 0.0.1 (AGENTS.md): a ``failed`` variant ships only when named.
_OWNER_DISPOSED = frozenset({"topk-embed-v1-small", "topk-embed-v1-xsmall"})


def test_every_shipped_recipe_is_verified_or_owner_disposed() -> None:
    """AGENTS.md: "status: unverified does not ship in a tag, and a failed variant ships only when the owner's
    explicit disposition list names it".

    Two rules, both read off a status block: the state it records and the image it names.  First, a recipe
    may claim ``verified`` only on the image it pins (``status.image == engine.image``): the 0.33 pin bump
    sets every recipe to ``unverified``, and writing ``verified`` before a 0.33 wave has run is the
    false-evidence path the plan forbids, so the goldens and this rule would freeze it.  Second, a ``failed``
    variant must be named by the owner's list.  The shipped states are therefore
    ``verified``-on-its-own-image plus that list; a recipe whose status block still names another image is
    waiting for its wave and is ``unverified``.

    The exactness is one-sided while the waves are outstanding: ``unshipped <= _OWNER_DISPOSED`` becomes
    ``==`` once W13 restores the topk variants' recorded ``failed`` (the pin bump set them ``unverified``
    too).  Until then, a recipe that flips to ``failed`` without being owner-named still reds here.
    """
    recipes = {recipe.id: recipe for recipe in iter_recipes()}
    for recipe in recipes.values():
        if recipe.status.state == "verified":
            assert recipe.status.image == recipe.engine.image, (
                f"{recipe.id}: claims verified on {recipe.status.image!r} while pinning "
                f"{recipe.engine.image!r}: no wave has run on the pinned image"
            )
    # Waiting for its wave is ``unverified`` and only that: a recipe that flips to ``failed`` is a
    # disposition decision, never a pending one, so it falls through to the owner-list check below.
    pending = {recipe.id for recipe in recipes.values() if recipe.status.state == "unverified"}
    unshipped = {recipe.id for recipe in recipes.values() if recipe.status.state != "verified"} - pending
    assert unshipped <= set(_OWNER_DISPOSED), (
        f"unshipped recipes that the owner's disposition list does not name: {sorted(unshipped)}"
    )
