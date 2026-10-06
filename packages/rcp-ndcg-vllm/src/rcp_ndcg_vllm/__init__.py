"""The serving-recipes harness for vLLM: recipes, equivalence, the engine recorder and the wave runner.

The recipe's ``client`` block is the product's endpoint config
(:class:`rcp_ndcg.inference.config.EmbeddingEndpoint`, ``PoolingEndpoint`` or ``RerankEndpoint``): one schema,
the product's. Stage 1 fits every sampled prompt with the product's
:func:`rcp_ndcg.data.preprocess.fit`; stage 2 sends through the product's role clients
(:mod:`rcp_ndcg.inference.clients`); the reference runs as a subprocess in its own environment.
"""

from __future__ import annotations

from .errors import HarnessError, RecipeError
from .recipe import (
    PINNED_POOLER_CONFIG_FIELDS,
    ClientEndpoint,
    EngineSpec,
    Gates,
    Recipe,
    ReferenceSpec,
    Resources,
    ServeConfig,
    StatusSpec,
    client_config,
    default_recipes_root,
    iter_recipes,
    load_recipe,
    recipe_json_schema,
    serve_argv,
)

__version__ = "0.0.1"

__all__ = [
    "ClientEndpoint",
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
    "client_config",
    "default_recipes_root",
    "iter_recipes",
    "load_recipe",
    "recipe_json_schema",
    "serve_argv",
]
