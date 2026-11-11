"""The four module-hygiene decisions, each with the evidence its item asks for.

* ``data/preprocess.py`` -- **kept** and documented as the one import path (item 1): the module says so, and
  every name it exports is a home module's own object, never a second definition.
* ``eval/mteb/`` -- **kept** in the product behind the ``[mteb]`` extra (item 2): under the extra it imports and
  its mteb-dependent branch (the abstention nAUCs) resolves; a bare install skips this test, as CI's gated job
  opens it.
* ``judging/tokens.py`` -- **kept** beside ``TextBudget`` (item 3), because an estimate runs where no tokenizer
  can count; the reason is a docs line (``docs/concepts/text-budgets.md``), so this module has no test for it.
* ``inference/recipes.py`` -- the soft dependency on ``rcp-vllm`` is a **declared extra** (item 4), named in one
  place (the error map) and in the refusal's install line, which says how to install it.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]

#: The homes ``rcp.data.preprocess`` re-exports from (the module docstring lists them name by name).
PREPROCESS_HOMES = (
    "rcp.data.census",
    "rcp.data.postprocess",
    "rcp.data.text_budget",
    "rcp.data.text_policy",
    "rcp.storage.census",
)


def test_the_preprocess_facade_is_kept_and_documented_as_the_one_import_path() -> None:
    """Item 1: kept, not deleted, and the kept module is a facade -- it defines nothing of its own.

    Deleting it would have moved the surface, but it is the import path the curated pages and the pinned API
    page teach, so the decision is to keep it and say so; the test pins both halves of that decision.
    """
    facade = importlib.import_module("rcp.data.preprocess")
    assert "the one import path" in (facade.__doc__ or ""), (
        "rcp.data.preprocess must say it is the one documented import path (the keep decision)"
    )
    for name in facade.__all__:
        value = getattr(facade, name)
        assert any(getattr(importlib.import_module(home), name, None) is value for home in PREPROCESS_HOMES), (
            f"{name} is defined in the facade, not re-exported from a home"
        )


def test_the_mteb_integration_imports_and_uses_mteb_under_its_extra() -> None:
    """Item 2: kept in the product, gated by the ``[mteb]`` extra -- under it, the integration imports and the
    mteb-only branch runs (mteb's abstention nAUCs beside the float-gain nDCG)."""
    pytest.importorskip("mteb")
    module = importlib.import_module("rcp.eval.mteb")
    assert callable(module.get_tasks)
    scores = module.ndcg_float_scores({"q": {"d": 1.0}}, {"q": {"d": 1.0}}, k_values=(1,))
    assert scores["ndcg_float_at_1"] == 1.0
    assert set(scores) > {"ndcg_float_at_1"}, "the mteb-only abstention nAUCs are missing under the extra"


def test_the_serving_recipes_are_a_declared_extra() -> None:
    """Item 4: ``rcp`` declares the sibling it lazily imports, as the ``vllm`` extra, and the error map names
    that extra as its one home (so ``rcp doctor`` reports it and a missing import has an install line)."""
    from rcp.errors import EXTRA_FOR_MODULE

    manifest = tomllib.loads((ROOT / "packages" / "rcp" / "pyproject.toml").read_text(encoding="utf-8"))
    extras = manifest["project"]["optional-dependencies"]
    assert "vllm" in extras, "rcp lazily imports rcp_vllm, so the dependency must be declared as an extra"
    assert {Requirement(spec).name for spec in extras["vllm"]} == {"rcp-vllm"}
    assert EXTRA_FOR_MODULE["rcp_vllm"] == "vllm"


def test_the_refusal_names_the_declared_extra_and_its_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 4: absent, the sibling is refused with the install line of the extra -- never a bare ImportError."""
    from rcp.errors import ConfigError
    from rcp.inference import recipes

    monkeypatch.setitem(sys.modules, "rcp_vllm", None)  # make the lazy import raise ModuleNotFoundError
    with pytest.raises(ConfigError, match="needs the serving recipes") as excinfo:
        recipes.available_recipe_ids()
    hint = excinfo.value.hint or ""
    assert 'pip install "rcp[vllm]"' in hint, hint
