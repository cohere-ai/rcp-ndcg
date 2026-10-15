"""The image_processor families in the judging docs are the shipped ones.

The judge config's `image_processor` Literal is the authority; the judges page's field row, the
preprocessing page's processor-family table and the `JudgeConfig` docstring must list every member, and the
table must name the shipped judges under their family.
"""

from __future__ import annotations

from typing import get_args

from rcp_ndcg.data.resolution import ImageProcessor
from rcp_ndcg.judging.client import JudgeConfig
from tests.docs._markdown import ROOT

FAMILIES = get_args(ImageProcessor)
JUDGES_PAGE = ROOT / "docs" / "concepts" / "judges.md"
PREPROCESSING = ROOT / "docs" / "concepts" / "preprocessing.md"


def test_the_judges_field_row_lists_every_family() -> None:
    """The `image_processor` row names every Literal member."""
    row = next(
        line
        for line in JUDGES_PAGE.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `image_processor` |")
    )
    missing = [family for family in FAMILIES if f"`{family}`" not in row]
    assert missing == [], f"the image_processor row omits {missing}"


def test_the_processor_family_table_lists_every_family_and_its_judges() -> None:
    """The table has a row per family, and the shipped judges are named under theirs."""
    text = PREPROCESSING.read_text(encoding="utf-8")
    rows = {
        family: next((line for line in text.splitlines() if line.startswith(f"| `{family}` |")), None)
        for family in FAMILIES
    }
    missing = [family for family, row in rows.items() if row is None]
    assert missing == [], f"the processor-family table omits {missing}"
    for model in ("Qwen3.5-397B", "Qwen3.6-27B", "Qwen3.8-27B", "Qwen3.8-Flash-Next"):
        assert model in rows["qwen3_vl"], f"the qwen3_vl row does not name {model}"
    assert "Gemma 4" in rows["gemma4"], "the gemma4 row does not name its judges"


def test_the_judge_config_docstring_lists_every_family() -> None:
    """The `image_processor` docstring names every Literal member."""
    doc = JudgeConfig.__doc__ or ""
    missing = [family for family in FAMILIES if f"``{family}``" not in doc]
    assert missing == [], f"the JudgeConfig docstring omits {missing}"
