"""The adopt-a-model doctrine matches the recipes and the loader it describes.

The page is the doctrine an author copies when they adopt a checkpoint: its sampling rule, its gate-override
semantics and its wire descriptions must be what the shipped recipes and the recipe loader actually do.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_vllm.recipe import iter_recipes, resolve_recipe

from tests.docs._markdown import ROOT, paragraph

PAGE = ROOT / "docs" / "how-to" / "adopt-a-model.md"


def _sampling(recipe) -> tuple:  # noqa: ANN001 - a resolved Recipe's client is a plain mapping
    extra = recipe.client.get("extra_body") or {}
    return (recipe.client.get("temperature"), extra.get("top_p"), extra.get("top_k"))


def test_the_judge_sampling_doctrine_is_the_recipes_rule() -> None:
    """The page's sampling bullet states the rule the ten judge recipes follow, not a fixed default triple."""
    bullet = paragraph(PAGE.read_text(encoding="utf-8"), "**Sampling is declared, not inherited.**")
    judges = [recipe for recipe in iter_recipes() if recipe.role == "judge"]
    assert len(judges) == 10
    assert len({_sampling(recipe) for recipe in judges}) >= 3, "a fixed default triple would misdescribe these"
    assert "generation_config" in bullet, "the rule is the checkpoint's own generation_config.json"
    assert "HF defaults" not in bullet, "the engine's HF defaults are not what the recipes send"


@pytest.mark.parametrize("recipe_id", ["pplx-embed-v2-late-9b"])
def test_the_gate_override_doctrine_matches_the_loader(recipe_id: str) -> None:
    """A set per-size gate replaces the family's or the published default; the page says so."""
    bullet = paragraph(PAGE.read_text(encoding="utf-8"), "**A measured bound.**")
    assert "never loosen" not in bullet, "the loader replaces in either direction; it has no tighten-only rule"
    assert "replaces" in bullet
    assert resolve_recipe(recipe_id).gates.vec_min_cosine == 0.978  # the shipped override replaces the default


def test_the_late_interaction_wire_matches_the_recipes() -> None:
    """The wire bullets name the shapes the shipped embedders and late-interaction recipes declare."""
    text = PAGE.read_text(encoding="utf-8")
    dense = paragraph(text, "One vector per text")
    late = paragraph(text, "One L2-normalized vector per kept token")
    shapes = {
        recipe.client.get("request_shape") or "text" for recipe in iter_recipes() if recipe.role == "multi_vector"
    }
    assert shapes == {"text", "token_ids"}
    assert "`text`" in dense, "most dense recipes send the default text wire"
    assert "`text`" in late and "`token_ids`" in late, "both shipped late-interaction wires are named"
    assert "the wire is `token_ids` (the client tokenizes; the engine tokenizes nothing)" not in late
