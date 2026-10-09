"""The lean vLLM serving package: the GPU-validated serving recipes (package data) and
``rcp-ndcg-vllm serve <recipe-id>``.

The shipped recipes declare how each model is served and how ``rcp-ndcg`` reads it back; their ``client``
block is the product's endpoint config, kept as plain data here and validated by ``rcp-ndcg`` when it reads it
(a resolution through ``recipe: <id>``). The folded model plugins (``rcp_ndcg_vllm.models``) register their
architectures with vLLM lazily through the one ``vllm.general_plugins`` entry point.

The wheel is pure Python and declares only dependencies the stock vLLM image already ships (pydantic, PyYAML),
so ``pip install --no-deps rcp-ndcg-vllm`` prepares an engine image without touching its pins (``pip freeze``
then differs by exactly this wheel). Importing this package never imports torch or vLLM.

Public names (pinned by ``tests/contract``): :class:`~rcp_ndcg_vllm.recipe.Family`,
:class:`~rcp_ndcg_vllm.recipe.Recipe`, :class:`~rcp_ndcg_vllm.recipe.Variant`,
:func:`~rcp_ndcg_vllm.recipe.load_family`, :func:`~rcp_ndcg_vllm.recipe.load_recipe`,
:func:`~rcp_ndcg_vllm.recipe.resolve_recipe`, :func:`~rcp_ndcg_vllm.recipe.iter_families`,
:func:`~rcp_ndcg_vllm.recipe.iter_recipes` and :func:`~rcp_ndcg_vllm.recipe.serve_argv` of
:mod:`rcp_ndcg_vllm.recipe`, the ``rcp-ndcg-vllm`` console tree (``serve``, ``--dry-run``) and the exported
recipe and family schema files (``schema/recipe.schema.json``, ``schema/family.schema.json``). Everything else
in this package is internal.
"""

from __future__ import annotations

from .errors import RecipeError as RecipeError
from .recipe import (
    Family,
    Recipe,
    Variant,
    iter_families,
    iter_recipes,
    load_family,
    load_recipe,
    resolve_recipe,
    serve_argv,
)

__version__ = "0.0.1"

__all__ = [
    "Family",
    "Recipe",
    "RecipeError",
    "Variant",
    "iter_families",
    "iter_recipes",
    "load_family",
    "load_recipe",
    "resolve_recipe",
    "serve_argv",
]
