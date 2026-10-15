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
