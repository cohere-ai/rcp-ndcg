"""The committed wave lists: generated from the shipped recipes, never hand-maintained (owner decision).

The lists' one home is ``rcp-ndcg-test/wave-lists/``; ``rc_build.sh`` stages them as
``<stage>/wave-lists/``.  ``all-retrieval.txt`` lists every shipped recipe id and is pinned to
:func:`rcp_ndcg_vllm.recipe.iter_recipes` here, so a recipe added or removed without regenerating the
list turns this file red.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rcp_ndcg_test.jobs.wavelist import ALL_RETRIEVAL, DEFAULT_WAVE_LISTS, main, parse_ids, write_wave_lists
from rcp_ndcg_vllm.recipe import iter_recipes

from tests.conftest import RECIPES

WAVE_LISTS = Path(__file__).resolve().parents[1] / "wave-lists"
"""The lists' one home in the checkout (``DEFAULT_WAVE_LISTS`` resolves there from the repo root)."""


def _ids(path: Path) -> list[str]:
    return parse_ids(f"@{path}")


def test_the_committed_all_retrieval_list_is_current() -> None:
    """The committed ``all-retrieval.txt`` is exactly the shipped recipe ids, sorted: the catalog and the
    list move together (regenerate with ``python -m rcp_ndcg_test.jobs.wavelist --out rcp-ndcg-test/wave-lists``)."""
    path = WAVE_LISTS / ALL_RETRIEVAL
    assert path.is_file(), f"the committed wave list is missing: {path}"
    assert _ids(path) == sorted(recipe.id for recipe in iter_recipes())
    assert _ids(path), "the list names no recipe"


def test_write_wave_lists_generates_from_the_recipe_catalog(tmp_path: Path) -> None:
    """The generator reads the recipe catalog through ``iter_recipes``: a recipe added to the root appears
    in the generated list, one removed disappears -- the list is never hand-maintained."""
    root = tmp_path / "recipes"
    shutil.copytree(RECIPES, root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", root.parent / "deterministic.py")
    out = tmp_path / "wave-lists"
    written = write_wave_lists(out, root)
    assert written == [out / ALL_RETRIEVAL]
    expected = sorted(recipe.id for recipe in iter_recipes(root))
    assert _ids(written[0]) == expected
    # One more recipe (a copy of the fixture with its id changed) joins the list.
    added = root / "added-recipe"
    shutil.copytree(root / "fixture-embed", added)
    recipe_yaml = added / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("id: fixture-embed", "id: added-recipe"), encoding="utf-8"
    )
    write_wave_lists(out, root)
    assert _ids(out / ALL_RETRIEVAL) == sorted([*expected, "added-recipe"])


def test_write_wave_lists_refuses_an_empty_recipe_root(tmp_path: Path) -> None:
    """A root with no recipe is an error, never an empty committed list (a wave list that names nothing
    would fail every wave it is used for, silently at generation time)."""
    empty = tmp_path / "recipes"
    empty.mkdir()
    with pytest.raises(Exception, match="no recipes"):
        write_wave_lists(tmp_path / "wave-lists", empty)


def test_a_duplicate_recipe_id_is_refused() -> None:
    """A duplicate id would collapse two runs into one result row and a passing run could mask a failing
    one (the wave keys results by id); the wave request is refused before any engine starts."""
    from rcp_ndcg_test.errors import HarnessError
    from rcp_ndcg_test.jobs.wavelist import load_wave

    with pytest.raises(HarnessError, match="duplicate recipe id"):
        load_wave(["fixture-embed", "fixture-embed"], RECIPES)


def test_the_wavelist_cli_regenerates_the_committed_lists(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m rcp_ndcg_test.jobs.wavelist --out <dir>`` writes the lists and reports the count; a
    missing recipe root exits 2."""
    assert main(["--out", str(tmp_path / "lists")]) == 0
    assert _ids(tmp_path / "lists" / ALL_RETRIEVAL) == sorted(recipe.id for recipe in iter_recipes())
    assert "ids" in capsys.readouterr().out
    assert main(["--out", str(tmp_path / "lists"), "--recipes-root", str(tmp_path / "nowhere")]) == 2


def test_the_default_home_is_the_tooling_package() -> None:
    """The lists' default home is ``rcp-ndcg-test/wave-lists`` (decision 20's tooling home); this test's own
    WAVE_LISTS is that directory resolved from the checkout, so the two can never drift apart."""
    assert DEFAULT_WAVE_LISTS.as_posix() == "rcp-ndcg-test/wave-lists"
    assert WAVE_LISTS.name == "wave-lists" and WAVE_LISTS.parent.name == "rcp-ndcg-test"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
