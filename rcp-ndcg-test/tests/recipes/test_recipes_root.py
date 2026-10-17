"""The shipped recipes root: every family and variant loads and validates against the product's configs.

The per-family modules pin each variant's full contract (one module per family, parametrized over its
variants, two mutants red per family); this module is the cross-family smoke test the root itself was
reserved for: every variant ``iter_recipes()`` yields loads, its id is its family's variant id, and its
``client`` block constructs the product's endpoint model for its role -- so a family that ships a
recipe the product refuses cannot pass unnoticed, and a family id can never be served.
"""

from __future__ import annotations

import re

from rcp_ndcg_vllm.recipe import default_recipes_root, iter_families, iter_recipes

from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.judging import JudgeConfig

_ENDPOINTS = {
    "embed": EmbeddingEndpoint,
    "multi_vector": PoolingEndpoint,
    "rerank": RerankEndpoint,
    "judge": JudgeConfig,
}


def test_the_recipes_root_loads_clean() -> None:
    """Every variant of every shipped family loads, its client block is the product's endpoint config, and
    every embedding/multi-vector variant declares its MRL kind (owner decision 39: a variant never ships
    without one, whether ``truncation``, ``projection`` or ``none``)."""
    recipes = iter_recipes()
    assert recipes, "the shipped recipes root holds no variant"
    variant_ids = {variant.id for family in iter_families() for variant in family.variants}
    for recipe in recipes:
        endpoint = _ENDPOINTS[recipe.role].model_validate(recipe.client)
        assert endpoint.model == recipe.id
        assert endpoint.revision == recipe.revision
        assert recipe.id in variant_ids
        if recipe.role in ("embed", "multi_vector"):
            assert recipe.client.get("mrl_kind") in ("truncation", "projection", "none"), (
                f"{recipe.id}: every embedding/multi-vector variant declares its MRL kind"
            )
        # decision 15: only a judge recipe carries no reference
        assert (recipe.reference is None) == (recipe.role == "judge"), recipe.id


def test_family_ids_are_not_recipe_ids() -> None:
    """A family id names a directory and is never a served id (decision 34)."""
    variant_ids = {recipe.id for recipe in iter_recipes()}
    multi_variant = {family.id for family in iter_families() if len(family.variants) > 1}
    # a single-variant family may share its name with its only variant (jina-reranker-v3)
    assert not (multi_variant & variant_ids), sorted(multi_variant & variant_ids)
    assert variant_ids, "no variant ids"
    assert default_recipes_root().is_dir()


def test_every_commented_instruction_none_states_the_drop_policy() -> None:
    """``instruction: none`` DROPS a dataset task instruction by declaration -- the client returns the
    prepared content, and only an UNDECLARED mode raises -- so the comment beside the field must state
    the drop and must not promise a refusal (or a reference that raises).  The wording lives in each
    family.yaml's own comment; this smoke test holds every commented ``instruction: none`` line to the
    policy, and a comment that mentions a dataset task instruction must say it is dropped."""
    offenders: list[str] = []
    for directory in sorted(default_recipes_root().iterdir()):
        family = directory / "family.yaml"
        if not family.is_file():
            continue
        lines = family.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = re.match(r"\s*instruction:\s*none\b(.*)$", line)
            if match is None or "#" not in match.group(1):
                continue  # an uncommented declaration: no wording to hold
            block = [match.group(1).split("#", 1)[1]]
            for follow in lines[index + 1 : index + 3]:
                if follow.lstrip().startswith("#"):
                    block.append(follow.split("#", 1)[1])
                else:
                    break
            comment = " ".join(part.strip() for part in block)
            if "refused" in comment or "raises on one" in comment:
                offenders.append(f"{directory.name}: promises a refusal: {comment}")
            if "dataset task" in comment and "dropped" not in comment.lower():
                offenders.append(f"{directory.name}: does not say the instruction is dropped: {comment}")
    assert not offenders, "instruction: none drops by declaration:\n" + "\n".join(offenders)
