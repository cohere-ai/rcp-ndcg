"""Every YAML recipe snippet on the add-a-model page is a real family: ``load_recipe`` validates it.

The page's sketch recipes are complete (no abridged ellipses): each block is written into a recipe directory
named after its ``id`` as ``family.yaml`` (decision 34), with the files it references (a stub
``reference.py``, the declared chat template), and loaded through the package's own ``load_recipe`` -- so a
snippet the schema refuses cannot ship in the docs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.docs._markdown import ROOT, code_blocks

PAGE = ROOT / "docs" / "how-to" / "add-a-model.md"

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
    from rcp_ndcg_vllm import load_recipe

    data = yaml.safe_load(block.text)
    assert isinstance(data, dict), "a recipe snippet is a YAML mapping"
    directory = tmp_path / str(data["id"])
    directory.mkdir()
    (directory / "family.yaml").write_text(block.text, encoding="utf-8")
    (directory / "reference.py").write_text(REFERENCE_STUB, encoding="utf-8")
    template = (data.get("serve") or {}).get("chat_template")
    if template:
        (directory / template).write_text("SYSTEM: {{ query }}\nUSER: {{ document }}\nASSISTANT:", encoding="utf-8")
    recipe = load_recipe(directory)
    assert recipe.id == data["id"]
