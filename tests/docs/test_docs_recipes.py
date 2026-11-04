"""Every YAML family snippet on the add-a-model page is a real family: ``load_family`` validates it.

The page's sketch families are complete (no abridged ellipses): each block is written into a family
directory named after its ``id``, with the files it references (a stub ``reference.py``, the declared chat
template), and loaded through the package's own ``load_family``/``resolve_recipe`` -- so a snippet the
schema refuses cannot ship in the docs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.docs._markdown import ROOT, code_blocks

PAGE = ROOT / "docs" / "how-to" / "add-a-model.md"

#: The number words a docs page spells a small count with (the serve page states the derived-chunk count;
#: the 0.33 pin's chunk floor reaches sixteen recipes, so the table runs past ten).
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
).split()

REFERENCE_STUB = '''"""The docs snippet's reference stub: the guide documents the contract, this file satisfies it."""

from __future__ import annotations


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _vllm_src() -> str:
    """The vllm package's ``src`` (it sits outside the root uv workspace): put it on the path for the test."""
    return str(ROOT / "rcp-ndcg-vllm" / "src")


def _recipe_blocks(page: Path):
    return [b for b in code_blocks(page.read_text(encoding="utf-8")) if b.language == "yaml"]


@pytest.mark.parametrize("block", _recipe_blocks(PAGE), ids=lambda b: f"line{b.line}")
def test_every_recipe_snippet_loads(block: Any, tmp_path: Path) -> None:  # noqa: ANN401 - pytest block object
    """One docs snippet == one loadable recipe directory."""
    import sys

    import yaml

    sys.path.insert(0, _vllm_src())
    from rcp_ndcg_vllm import load_family, resolve_recipe

    data = yaml.safe_load(block.text)
    assert isinstance(data, dict), "a family snippet is a YAML mapping"
    directory = tmp_path / str(data["id"])
    directory.mkdir()
    (directory / "family.yaml").write_text(block.text, encoding="utf-8")
    (directory / "reference.py").write_text(REFERENCE_STUB, encoding="utf-8")
    template = (data.get("serve") or {}).get("chat_template")
    if template:
        (directory / template).write_text("SYSTEM: {{ query }}\nUSER: {{ document }}\nASSISTANT:", encoding="utf-8")
    family = load_family(directory)
    assert family.id == data["id"]
    first = family.variants[0].id
    assert resolve_recipe(first, root=tmp_path).id == first


def test_the_reproducibility_page_counts_the_recipe_statuses() -> None:
    """Two recipes ship with status ``failed`` (the topk-embed-v1 variants), so the page may not call all
    44 recipes validated end to end on GPU.

    The serve page's count of the recipes that render the derived ``--max-num-batched-tokens`` is the
    catalog's count too: a recipe added, or its budget moved to ``serve.max_model_len``, changes it."""
    from collections import Counter

    from rcp_ndcg_vllm.recipe import iter_recipes, serve_argv

    states = Counter(recipe.status.state for recipe in iter_recipes())
    text = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "each validated end to end on GPU" not in text, "the page claims every recipe was validated"
    assert f"{states['verified']} verified end to end on GPU" in text
    assert f"{states['failed']} shipped with status failed" in text

    # No recipe declares serve.max_num_batched_tokens (ServeConfig has no such field), so the flag in the
    # rendered argv is the value _deployment_values derives at serve.max_model_len.
    derived = sum(
        1
        for recipe in iter_recipes()
        if "--max-num-batched-tokens" in serve_argv(recipe, port=8000, served_model_name=recipe.id)
    )
    serve_page = (ROOT / "docs" / "how-to" / "serve-a-model.md").read_text(encoding="utf-8")
    stated = re.search(r"the (\w+) affected recipes", serve_page)
    assert stated is not None, "the serve page states how many recipes render the derived chunk"
    assert _NUMBER_WORDS.index(stated.group(1)) == derived, (
        f"the serve page says {stated.group(1)!r} affected recipes; the catalog renders the derived "
        f"--max-num-batched-tokens for {derived}"
    )


def test_the_ledger_fingerprint_column_is_the_tree_today() -> None:
    """The GPU ledger's fingerprint column is "the behaviour fingerprint the tree computes today" (the key a
    wave's evidence must carry), so every cell is ``behaviour_fingerprint(recipe)``'s first 12 hex digits.
    A recipe field the key covers moving (a serve value, the image, a patch) reds its row instead of leaving
    a superseded key in the page."""
    from rcp_ndcg_test.fingerprint import behaviour_fingerprint
    from rcp_ndcg_vllm.recipe import iter_recipes

    # The tokenizer bytes are pinned by the recipe's revision: the golden records their SHA-256 (the key's
    # one input that needs a file the tree does not carry), so the key recomputes offline.
    golden = ROOT / "rcp-ndcg-test" / "tests" / "recipes" / "golden"
    page = (ROOT / "docs" / "reference" / "gpu-validation.md").read_text(encoding="utf-8")
    row = re.compile(r"^\| `([^`]+)` \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| `([0-9a-f]+)…` \|", re.MULTILINE)
    cells = dict(row.findall(page))
    recipes = {recipe.id: recipe for recipe in iter_recipes()}
    assert set(cells) == set(recipes), "one ledger row per shipped recipe"
    stale: dict[str, tuple[str, str]] = {}
    for recipe_id, cell in cells.items():
        recorded = json.loads((golden / f"{recipe_id}.json").read_text(encoding="utf-8"))
        tokenizer_sha = recorded["fingerprint"]["inputs"]["tokenizer_sha256"]
        today = behaviour_fingerprint(recipes[recipe_id], tokenizer_sha256_value=tokenizer_sha)[: len(cell)]
        if today != cell:
            stale[recipe_id] = (cell, today)
    assert not stale, f"ledger rows whose fingerprint cell is not the tree's key (cell, today): {stale}"
