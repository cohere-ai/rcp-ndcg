"""The anchor audit of ``anchor: last_content`` (a model that pools the last real token of raw text).

The product's definition (:data:`rcp_ndcg.data.templates.AnchorKind`): the anchor is the last kept *content*
token -- the shape ends on its content span, a cut keeps a content prefix, so that token always survives -- and the
fixed segments (a head marker such as jina-embeddings-v5's ``Query:``) are still reserved and audited.  The audit
asserts both on every captured body: the head marker opens the render as the engine reads it, and the render ends
on a content token (then the post-processor's tokens, when the shape declares them).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence import stages as stages_module

from rcp_ndcg.data.tokenizer import load_tokenizer
from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs
from tests.test_equivalence import REFERENCE_PYTHON, _rebased

TOK = load_tokenizer(str(TOKENIZER))


def _last_content_recipe(tmp_path: Path, *, add_special_tokens: bool) -> Any:
    """``fixture-embed`` re-declared as a last-content pooler: ``doc:`` and its separator space as two fixed head
    segments, the document last (no fixed tail); the fixture reference renders ``doc: <document>``."""
    name = f"last-content-{str(add_special_tokens).lower()}"
    root = tmp_path / name
    directory = root / "recipes" / name
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", root / "deterministic.py")
    source = (RECIPES / "fixture-embed" / "reference.py").read_text(encoding="utf-8")
    (directory / "reference.py").write_text(source.replace('SUFFIX = " [END]"', 'SUFFIX = ""'), encoding="utf-8")
    manifest = _rebased((RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8"), name)
    old = '      - {fixed: "doc: "}\n      - {content: document}\n      - {fixed: " [END]"}\n    anchor: last\n'
    assert old in manifest
    new = (
        '      - {fixed: "doc:"}\n      - {fixed: " "}\n      - {content: document}\n    anchor: last_content\n'
        f"    add_special_tokens: {str(add_special_tokens).lower()}\n"
    )
    (directory / "recipe.yaml").write_text(manifest.replace(old, new), encoding="utf-8")
    return load_recipe(directory)


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_stage1_passes_a_last_content_recipe(tmp_path: Path, add_special_tokens: bool) -> None:
    """The client's captured requests -- the over-length samples it cut included -- pass the audit."""
    recipe = _last_content_recipe(tmp_path, add_special_tokens=add_special_tokens)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=3)
    audit = document["anchor_check"]
    assert audit["anchor"] == "last_content"
    assert audit["passed"] is True, audit["failures"][:1]
    assert audit["checked"] >= 4
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_the_last_content_audit_fails_a_render_that_lost_its_head_or_its_content(
    tmp_path: Path, add_special_tokens: bool
) -> None:
    """The audit is real: a body without its head marker, a body whose content is gone (the last token is then
    the frame's) and a body missing the post-processor's tail fail."""
    recipe = _last_content_recipe(tmp_path, add_special_tokens=add_special_tokens)

    def audit(body: Any) -> bool:
        probe = {"rows": [{"shapes": {"document": {"texts": [body]}}, "cuts": 0, "over_cap": False}]}
        return bool(stages_module._anchor_check(recipe, probe, TOK)["passed"])

    whole = "doc: Paris is the capital of France."
    assert audit(whole) is True
    assert audit(TOK.ids(whole, add_special_tokens=add_special_tokens)) is True
    assert audit("Paris is the capital of France.") is False, "the head marker was cut"
    assert audit("doc: ") is False, "no content token: the model would pool the frame"
    if add_special_tokens:
        assert audit(TOK.ids(whole, add_special_tokens=False)) is False, "the post-processor's tail is missing"
    else:
        assert audit(TOK.ids("Paris is the capital of France.", add_special_tokens=False)) is False
