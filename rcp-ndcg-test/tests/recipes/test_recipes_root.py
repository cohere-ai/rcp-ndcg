"""The shipped recipes root: every family and variant loads and validates against the product's configs.

The per-family modules pin each variant's full contract (one module per family, parametrized over its
variants, two mutants red per family); this module is the cross-family smoke test the root itself was
reserved for: every variant ``iter_recipes()`` yields loads, its id is its family's variant id, and its
``client`` block constructs the product's endpoint model for its role -- so a family that ships a
recipe the product refuses cannot pass unnoticed, and a family id can never be served.
"""

from __future__ import annotations

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
    """Every variant of every shipped family loads and its client block is the product's endpoint config."""
    recipes = iter_recipes()
    assert recipes, "the shipped recipes root holds no variant"
    variant_ids = {variant.id for family in iter_families() for variant in family.variants}
    for recipe in recipes:
        endpoint = _ENDPOINTS[recipe.role].model_validate(recipe.client)
        assert endpoint.model == recipe.id
        assert endpoint.revision == recipe.revision
        assert recipe.id in variant_ids
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
