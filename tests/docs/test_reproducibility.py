"""REPRODUCIBILITY's judge section points at the shipped judges.

The paper's judges are ``rcp-ndcg-vllm`` recipes now; the package's ``judging/judges/`` directory holds only the
hosted ``gpt5_hosted`` vendor profile, so a reader sent there finds no paper judge.
"""

from __future__ import annotations

from rcp_ndcg.judging.judges import judge_names
from tests.docs._markdown import ROOT

PAGE = ROOT / "REPRODUCIBILITY.md"
OLD_HOME = "judging/judges/"


def test_the_judge_section_names_the_recipe_route() -> None:
    """The section names `--judge recipe:<id>` and the recipes, not the directory that no longer holds them."""
    text = PAGE.read_text(encoding="utf-8")
    section = text[text.index("## 3. Re-judge a pool") : text.index("## Deviations")]
    assert "recipe:" in section, "the judge route is the recipe pointer"
    assert "rcp-ndcg-vllm" in section, "the judges ship in rcp-ndcg-vllm"
    assert OLD_HOME not in section, f"the section still sends readers to {OLD_HOME}"
    assert list(judge_names()) == ["gpt5_hosted"], "the package holds only the hosted profile"


def test_the_ledger_lists_the_judge_side_divergences() -> None:
    """g-judges-6 / decision 14(ii): the ledger promises 'every cause of a number that moves', yet the
    judge-side divergences lived only in recipe YAML notes.  The Judges entry lists the output caps against
    the paper's run-level ``max_response_tokens``, the generation-defaults sampling decision, the Qwen3.6
    revision pin and the SGLang to vLLM move -- and every judge recipe whose notes declare a divergence is
    named in it."""
    from rcp_ndcg_vllm.recipe import iter_recipes

    text = PAGE.read_text(encoding="utf-8")
    ledger = text[text.index("## Deviations from the paper's code") :]
    assert "### Judges" in ledger, "the ledger has no Judges entry (g-judges-6)"
    entry = ledger.split("### Judges", 1)[1].split("\n## ", 1)[0]
    for required in ("12288", "16384", "8192", "generation", "revision", "SGLang", "vLLM"):
        assert required in entry, f"the Judges entry must name {required!r}"
    divergent = sorted(
        recipe.id
        for recipe in iter_recipes()
        if recipe.role == "judge" and "DECLARED DIVERGENCE" in (recipe.notes or "")
    )
    assert divergent, "the judge recipes with declared divergences must load"
    for recipe_id in divergent:
        assert recipe_id in entry, f"{recipe_id} declares a divergence the ledger must list"
