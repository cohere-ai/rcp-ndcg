"""The verified fake engines' test wiring: one recipe id -> its registered emulator.

The emulators (``rcp_ndcg.testing.engines``) are product code and know nothing of the harness; the
recipes (``rcp_ndcv-vllm``) name the tokenizer, the template and the listwise prompt builder. This is
the one adapter between them (test-side): it loads the recipe through the harness package, resolves the
recipe's real tokenizer from the corpora's vendored store, picks the prompt derivation the role and
scoring name, builds the emulator over the recipe's corpus and registers it for
``fake://vllm-0.31.0/<recipe>``. The wiring mirrors what a golden-replay or conformance run needs --
and what an out-of-tree ``rcp_ndcg.emulators`` entry point would construct for its own models.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENGINES_ROOT = ROOT / "tests" / "contract" / "engines"
RECIPES_ROOT = ROOT / "packages" / "rcp-ndcg-vllm" / "recipes"
_VLLM_SRC = str(ROOT / "packages" / "rcp-ndcg-vllm" / "src")


def harness() -> Any:
    """Import ``rcp_ndcg_vllm`` (outside the uv workspace: its ``src`` goes on the path) and register the
    vendored tokenizer stores, once."""
    if _VLLM_SRC not in sys.path:
        sys.path.insert(0, _VLLM_SRC)
    from rcp_ndcg_vllm.fingerprint import use_tokenizer_store

    for store in ENGINES_ROOT.glob("*/_tokenizers"):
        use_tokenizer_store(store)
    return __import__("rcp_ndcg_vllm.recipe", fromlist=["load_recipe"])


def load_recipe(recipe_id: str) -> Any:
    """The recipe, loaded through the harness's own schema and the product's validation."""
    module = harness()
    return module.load_recipe(RECIPES_ROOT / recipe_id)


def corpus_of(recipe: Any) -> Any:
    """The committed corpus of the recipe's behaviour fingerprint."""
    from rcp_ndcg.testing.engines import load_corpus

    module = harness()
    from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint

    _ = module
    fingerprint = behaviour_fingerprint(recipe)
    return load_corpus(ENGINES_ROOT / "vllm-0.31.0" / recipe.id / fingerprint)


def prompt_strategy(recipe: Any, tokenizer: Any) -> Any:
    """The recipe's engine-prompt derivation: input strings for the pooling routes (the route tokenizes
    the request's strings), the recipe's pair template for a pointwise reranker, and the recipe
    reference's N-passage builder for a listwise one (the engine renders one prompt per request)."""
    from rcp_ndcg.testing.engines import EnginePrompts, PairPrompts, StringsPrompts

    if recipe.role == "embed":
        return StringsPrompts(slot="vector")
    if recipe.role == "multi_vector":
        return StringsPrompts(slot="token_vector")
    if getattr(recipe.client, "listwise", False):
        return EnginePrompts(builder=_listwise_builder(recipe))
    return PairPrompts(template=recipe.client.template, tokenizer=tokenizer)


def _listwise_builder(recipe: Any):
    """The listwise prompt builder the recipe itself declares (its ``reference.py``'s verbatim port of
    the checkpoint's prompt function): one home for the prompt shape, the emulator reads it from there."""
    directory = RECIPES_ROOT / recipe.id
    path = directory / (recipe.reference.entry or "reference.py")
    spec = importlib.util.spec_from_file_location(f"rcp_recipe_reference_{recipe.id.replace('-', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "format_docs_prompt", None)
    if builder is None:
        raise AssertionError(f"{path} declares no format_docs_prompt; a listwise recipe's reference does")

    def build(query: str, documents: list[str]) -> str:
        return builder(query, documents)

    return build


@functools.cache
def _build_emulator(recipe_id: str) -> Any:
    """The one emulator of ``recipe_id``, built once per process (facts from the recipe and its corpus)."""
    harness()
    from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint, load_recipe_tokenizer

    from rcp_ndcg.testing.engines import EngineFacts, VllmEmulator, registry

    recipe = load_recipe(recipe_id)
    tokenizer = load_recipe_tokenizer(recipe)
    corpus = corpus_of(recipe)
    facts = EngineFacts(
        engine_name=str(corpus.engine["name"]),
        engine_version=str(corpus.engine["version"]),
        served_name=recipe.id,
        model_root=recipe.model,
        max_model_len=recipe.serve.max_model_len,
    )
    emulator = VllmEmulator.from_corpus(corpus, prompt_strategy(recipe, tokenizer), tokenizer, facts)
    assert emulator.verified is not None
    emulator.require_verified_for(
        recipe.id, recipe.revision, behaviour_fingerprint(recipe), str(corpus.engine["version"])
    )
    registry.register(emulator)
    return emulator


def emulator_for(recipe_id: str) -> Any:
    """The registered emulator of ``recipe_id``: one per recipe per process, and re-registered on every
    call (the registry may have been reset between tests)."""
    from rcp_ndcg.testing.engines import registry

    emulator = _build_emulator(recipe_id)
    registry.register(emulator)
    return emulator