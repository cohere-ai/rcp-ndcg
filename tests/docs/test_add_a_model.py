"""The add-a-model page's worked example, override list and plugin prose match the schema.

The page is the mechanics an author copies: its worked-example comment must be a segment list the schema
accepts, its per-size override list must be the whole whitelist the loader allows, and its plugin paragraph
must name every folded plugin the wheel registers.
"""

from __future__ import annotations

import re

from rcp_ndcg_vllm.recipe import PER_VARIANT_SERVE_FIELDS

from tests.docs._markdown import ROOT, paragraph

PAGE = ROOT / "docs" / "how-to" / "add-a-model.md"
RECIPES = ROOT / "docs" / "reference" / "recipes.md"

#: The checkpoint families whose recipes declare ``serve.plugin`` with a registered architecture.
FOLDED_PLUGINS = ("topk-embed-v1", "pplx-embed-v1", "pplx-embed-v2", "embeddinggemma-2")


def test_the_worked_example_segments_are_what_the_schema_accepts() -> None:
    """The comment's segment keys are `fixed`/`content`, and each part validates through ``Segment``."""
    from rcp_ndcg.data.templates import Segment

    comment = next(
        line
        for line in PAGE.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#") and "anchor: last" in line
    )
    keys = set(re.findall(r"\{(\w+):", comment))
    assert keys <= {"fixed", "content"}, f"the schema refuses these segment keys: {sorted(keys - {'fixed', 'content'})}"
    parts = re.findall(r"\{(\w+):\s*\"?([^\",}]*)\"?\}", comment)
    assert parts, f"no segment list in the comment: {comment}"
    for key, value in parts:
        Segment.model_validate({key: value})


def test_the_per_size_override_list_is_the_whole_whitelist() -> None:
    """Every per-variant serve override the loader allows, and the per-variant licence, is listed."""
    bullet = paragraph(PAGE.read_text(encoding="utf-8"), "1. Append a `variants` row")
    missing = [field for field in PER_VARIANT_SERVE_FIELDS if field not in bullet]
    assert "licence" in bullet
    assert missing == [], f"the per-size override list omits {missing}"


def test_the_plugin_paragraphs_name_every_folded_plugin() -> None:
    """Both plugin paragraphs name the topk, pplx-v1, pplx-v2 and embeddinggemma-2 folded plugins."""
    paragraphs = {
        "docs/how-to/add-a-model.md": paragraph(PAGE.read_text(encoding="utf-8"), "folded models"),
        "docs/reference/recipes.md": paragraph(RECIPES.read_text(encoding="utf-8"), "fold into"),
    }
    for path, text in paragraphs.items():
        missing = [name for name in FOLDED_PLUGINS if name not in text]
        assert missing == [], f"{path} names no plugin for {missing}"
