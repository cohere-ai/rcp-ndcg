"""The shipped recipes' prose, held to the code it describes: budget quotes, plugin claims, citations.

A recipe's ``family.yaml`` and ``reference.py`` are package data a reader resolves; a number or a claim
the resolved contract contradicts is a defect even when no behaviour moves (the review's g-ref-text-3).
This module is the cross-family home for the mechanically checkable part of that rule: a quoted client
right-cut is a declared client budget, a recipe that declares a plugin never says "no plugin", a citation
to a paper retrieval config names keys the config declares, and the stale claims the review found are
gone.  Offline by construction: the recipe package's data and the checkout's ``experiments/`` files.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from rcp_ndcg_vllm.recipe import default_recipes_root

RECIPES = default_recipes_root()
REPO_ROOT = Path(__file__).resolve().parents[3]


def _family_dirs() -> list[Path]:
    return [path for path in sorted(RECIPES.iterdir()) if (path / "family.yaml").is_file()]


def _normalised(text: str) -> str:
    """``text`` with every run of whitespace collapsed: a claim may wrap across YAML comment lines."""
    return " ".join(text.split())


def test_a_recipe_that_declares_a_plugin_never_claims_an_unplugged_image() -> None:
    """The plugin wheel carries the engine patches; a recipe that declares ``serve.plugin`` does not run
    on an unplugged image, so its prose must not claim one (zembed-1's notes said "on the unmodified
    vllm/vllm-openai:v0.31.0 image, no plugin" while its serve block declared ``plugin: rcp-ndcg-vllm``
    with the pooling-full-context patch)."""
    unplugged = re.compile(r"unmodified [^\n]*image,? no plugin", re.IGNORECASE)
    offenders = [
        family.name
        for family in _family_dirs()
        if (yaml.safe_load((family / "family.yaml").read_text(encoding="utf-8")).get("serve") or {}).get("plugin")
        is not None
        and unplugged.search((family / "family.yaml").read_text(encoding="utf-8"))
    ]
    assert not offenders, f"these recipes declare serve.plugin and claim an unplugged image: {offenders}"


_QUOTED_CUT = re.compile(r"(\d+)-token (?:right cut|right cuts|budget)")


def test_a_quoted_client_right_cut_is_a_declared_client_budget() -> None:
    """A family's prose that quotes the client's right cut or budget with a number must quote one of the
    family's declared client budgets: pplx-embed-v2-context said 262142 while the client declares 131070
    (262144 is the model-side limit, not the client's budget)."""
    offenders: list[str] = []
    for family in _family_dirs():
        yaml_path = family / "family.yaml"
        client = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("client") or {}
        declared = {value for value in client.values() if isinstance(value, int) and not isinstance(value, bool)}
        for path in [yaml_path, *sorted(family.glob("*.py"))]:
            for match in _QUOTED_CUT.finditer(path.read_text(encoding="utf-8")):
                if int(match.group(1)) not in declared:
                    offenders.append(f"{path.relative_to(RECIPES)}: {match.group(0)}")
    assert not offenders, f"a quoted client budget is not declared by the client block: {offenders}"


_PAPER_RETRIEVAL_CITATION = re.compile(r"(experiments/paper/retrieval/[A-Za-z0-9_.-]+\.yaml):(\d+)(?:-(\d+))?")
_CLAIM = re.compile(r"(?<![\w/.])([a-z_]+): ")


def test_a_citation_to_a_paper_retrieval_config_names_keys_the_config_carries() -> None:
    """The octen family cites ``experiments/paper/retrieval/octen.yaml`` for the paper's request shape.
    When the mapping form landed, ``pooling`` and ``doc_prompt`` left the config (the recipe's template
    and the reference declare them) and ``batch_size`` moved to line 22; a citation's parenthesised
    ``key: value`` claims must appear in the cited lines.  Only the retrieval configs are checked: their
    keys are the contract (a served config's claim can describe the paper's old in-process form)."""
    offenders: list[str] = []
    for family in _family_dirs():
        for path in [family / "family.yaml", *sorted(family.glob("*.py"))]:
            for line in path.read_text(encoding="utf-8").splitlines():
                for match in _PAPER_RETRIEVAL_CITATION.finditer(line):
                    lines = (REPO_ROOT / match.group(1)).read_text(encoding="utf-8").splitlines()
                    first = int(match.group(2))
                    last = int(match.group(3) or match.group(2))
                    block = "\n".join(lines[first - 1 : last])
                    missing = [
                        key
                        for key in _CLAIM.findall(line[match.end() :])
                        if not re.search(rf"^\s*{key}:", block, re.MULTILINE)
                    ]
                    if missing:
                        offenders.append(f"{path.relative_to(RECIPES)}: {match.group(0)} does not declare {missing}")
    assert not offenders, f"a citation names keys the cited config does not carry: {offenders}"


#: The review's stale prose, by file: each claim the code no longer matches.  Normalised whitespace, so a
#: claim wrapped across comment lines is still found.
STALE_CLAIMS = {
    "ctxl-rerank-v2-instruct-multilingual/reference.py": "dtype (bfloat16, the pipeline's)",
    "ctxl-rerank-v2-instruct-multilingual/reference.in": (
        "precision class closed by serve.hf_overrides head_dtype: model"
    ),
    "qwen3-reranker/reference.py": "flash attention 2 on CUDA",
    "pplx-embed-v2-context/family.yaml": "int8-valued vectors survive the cast exactly",
}


def test_the_stale_text_reference_claims_are_gone() -> None:
    """The claims the review found contradicted, each checked against the code that superseded it:
    ctxl's reference loads the engine's served dtype (float32 for 1b/2b), not the pipeline's bfloat16,
    and its reference.in no longer cites the removed ``head_dtype`` override; qwen3-reranker's score
    loads the recipe's declared ``attn_implementation`` (sdpa), not flash attention 2; and
    pplx-embed-v2-context's served head L2-normalises, so the fp16 cast is not exact on the wire."""
    offenders: list[str] = []
    for relative, stale in STALE_CLAIMS.items():
        text = _normalised((RECIPES / relative).read_text(encoding="utf-8"))
        if stale in text:
            offenders.append(f"{relative}: still claims {stale!r}")
    assert not offenders, "\n".join(offenders)


def test_the_flash_next_variant_note_names_its_real_quantisation() -> None:
    """The qwen3.8-flash-next-nvfp4 checkpoint's ``hf_quant_config.json`` declares quant_algo
    MIXED_PRECISION (the family's own sources line says so), and vLLM v0.31.0's override loop replaces a
    declared ``--quantization modelopt_fp4`` with ``modelopt_mixed`` before the mismatch check: the
    declared flag is inert and the engine runs modelopt_mixed.  The variant note must say so, never claim
    auto-detection maps the checkpoint's NVFP4 quant_algo to the same method, and the family note must not
    attribute the KV dtype to a paper that never ran this model."""
    family = yaml.safe_load((RECIPES / "qwen3.8-flash-next" / "family.yaml").read_text(encoding="utf-8"))
    variant = next(item for item in family["variants"] if item["id"] == "qwen3.8-flash-next-nvfp4")
    note = _normalised(variant["notes"])
    assert "MIXED_PRECISION" in note and "modelopt_mixed" in note, "the note names the engine's real method"
    assert "NVFP4 quant_algo to the same method" not in note
    assert "the paper set none" not in _normalised(family["notes"]), "the paper never ran this model"
