"""Serving recipes, the equivalence harness, the engine recorder and the GPU wave runner for vLLM.

The package is installed into an engine image (or any Python that talks to one) and stands alone:
``rcp-ndcg`` and ``rcp_ndcg_vllm`` never import each other; they meet at the engine's URL.
"""

from __future__ import annotations

from .errors import HarnessError, RecipeError
from .recipe import (
    PINNED_POOLER_CONFIG_FIELDS,
    BlockingSpec,
    ClientConfig,
    EngineSpec,
    Gates,
    Recipe,
    ReferenceSpec,
    Resources,
    ServeConfig,
    StatusSpec,
    TemplateSegment,
    TemplateSpec,
    client_config,
    default_recipes_root,
    effective_embed_dtype,
    iter_recipes,
    load_recipe,
    recipe_json_schema,
    serve_argv,
)

__version__ = "0.0.1"

__all__ = [
    "BlockingSpec",
    "ClientConfig",
    "EngineSpec",
    "Gates",
    "HarnessError",
    "PINNED_POOLER_CONFIG_FIELDS",
    "Recipe",
    "RecipeError",
    "ReferenceSpec",
    "Resources",
    "ServeConfig",
    "StatusSpec",
    "TemplateSegment",
    "TemplateSpec",
    "client_config",
    "default_recipes_root",
    "effective_embed_dtype",
    "iter_recipes",
    "load_recipe",
    "recipe_json_schema",
    "serve_argv",
]
