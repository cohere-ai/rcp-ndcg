"""The folded model plugins: one ``vllm.general_plugins`` entry point registering each architecture lazily.

vLLM loads this group in every engine process at startup and calls :func:`register`, which guards the engine
version once, applies the opted-in engine patches of :mod:`rcp_vllm.patches` (the one place the engine
process reads ``RCP_NDCG_VLLM_PATCHES``), and registers each folded model's architecture and config class.
Every model class is registered as a lazy ``"module:Class"`` string,
so the registry imports ``rcp_vllm.models.topk.model`` /
``rcp_vllm.models.pplx.model`` only when a request for the architecture actually needs it; importing this
package (or executing :func:`register`) never imports torch or vLLM here — the model modules do, inside vLLM.
The topk and pplx distributions folded into this one wheel (layout-move item 3): there is a single
``vllm.general_plugins`` entry point (``rcp-vllm``) and a single version guard
(:mod:`rcp_vllm.models.version_guard`).
"""

from __future__ import annotations

import sys
from typing import NoReturn

LAZY_MODEL_MODULES: tuple[str, ...] = (
    "rcp_vllm.models.embedding_gemma2.attention",
    "rcp_vllm.models.keep_pooler",
    "rcp_vllm.models.pplx.config",
    "rcp_vllm.models.pplx.hf_config",
    "rcp_vllm.models.pplx.late",
    "rcp_vllm.models.pplx.late_data",
    "rcp_vllm.models.pplx.model",
    "rcp_vllm.models.pplx.pooler",
    "rcp_vllm.models.pplx.pooling_core",
    "rcp_vllm.models.token_pooler",
    "rcp_vllm.models.topk.config",
    "rcp_vllm.models.topk.model",
    "rcp_vllm.models.topk.pooling",
    "rcp_vllm.models.topk.plugin",
    "rcp_vllm.models.topk.weights",
)
"""The registry-lazy model modules: they import vLLM/torch by design, and only vLLM imports them (as the
``module:Class`` strings the entry point registers). Everything else -- including this entry-point callable
and the one version guard -- must import clean (``tests/test_no_torch.py`` pins it; the surface walk skips
these). :mod:`rcp_vllm.models.keep_rule` is the one keyed module that is NOT lazy: it imports neither
vLLM nor torch (the declared rules and the positions they keep are pure data and pure Python), so the
harness scan may import it, and it is keyed by the fingerprint all the same."""

PLUGIN_ENGINE_MODULES: tuple[str, ...] = (
    "rcp_vllm.models",
    "rcp_vllm.models.version_guard",
    "rcp_vllm.models.pplx",
    "rcp_vllm.models.pplx.config",
    "rcp_vllm.models.pplx.hf_config",
    "rcp_vllm.models.topk",
    "rcp_vllm.models.topk.config",
    "rcp_vllm.models.topk.plugin",
    "rcp_vllm.patches",
)
"""The engine-side modules every plugin recipe's engine imports at registration, whatever its architecture:
:func:`register`'s entry-point callable, the one version guard, the package inits, the config
registrations and the patch applier.  ``pplx.config``/``topk.config`` are registered by their
``register_*`` calls at startup.  A recipe that declares ``serve.plugin`` keys these beside its
architectures' modules (``rcp-fp/4``); a change to one of them moves every plugin recipe, because every
plugin engine imports it.  The patch modules themselves are keyed per recipe, by its declared
``serve.patches``: a patch is imported unconditionally but applied only when opted in."""

ARCHITECTURE_MODULES: dict[str, tuple[str, ...]] = {
    # The contextual chunk model (pplx-embed-v2-context-9b-preview): the model class imports its pooler
    # and the pooling core (the config registration is shared, in PLUGIN_ENGINE_MODULES), the pooler
    # uses the plugin's one token-embed head construction, and the id-level contract and the per-chunk
    # segmentation live in the pure-Python chunk rule (the pooling core and the observation corpus's
    # acceptance check both read it).
    "PplxContextualModel": (
        "rcp_vllm.models.pplx.chunk_rule",
        "rcp_vllm.models.pplx.model",
        "rcp_vllm.models.pplx.pooler",
        "rcp_vllm.models.pplx.pooling_core",
        "rcp_vllm.models.token_pooler",
    ),
    # The late-interaction sibling (pplx-embed-v2-late-0.6b/-9b): its model class, the Dense-head mapping,
    # and the keep-rule machinery its pooler applies (the declared rules and the pool that drops them).
    "Qwen3_5Model": (
        "rcp_vllm.models.pplx.late",
        "rcp_vllm.models.pplx.late_data",
        "rcp_vllm.models.keep_rule",
        "rcp_vllm.models.keep_pooler",
        "rcp_vllm.models.token_pooler",
    ),
    # The topk multimodal late-interaction model: its model and weight mapping (its config registration is
    # shared, in PLUGIN_ENGINE_MODULES) and the same keep-rule machinery (its image allowlist).
    "TopkEmbedModel": (
        "rcp_vllm.models.topk.model",
        "rcp_vllm.models.topk.weights",
        "rcp_vllm.models.keep_rule",
        "rcp_vllm.models.keep_pooler",
        "rcp_vllm.models.token_pooler",
    ),
    # A config-only registration (the pplx-embed-v1 family): the plugin registers the transformers config
    # class so config.json parses locally, while the stock Qwen3ForCausalLM converted to pooling serves the
    # model -- so the modules are the shared config registration, keyed for every plugin recipe anyway.
    "PplxV1Config": ("rcp_vllm.models.pplx.hf_config",),
}
"""Every architecture (and config-only registration) this wheel registers, mapped to the engine-side
modules that implement it.

One home: the registration constants (``PLUGIN_ARCHITECTURE``, ``LATE_ARCHITECTURE``,
``topk.plugin.MODEL_ARCHITECTURE``; ``PplxV1Config`` is the config-only class ``register_pplx`` registers
for the pplx-embed-v1 family) are the truth, and ``tests/test_plugin_modules.py`` pins the keys against
them and the values against the lazy-import contract.  The behaviour fingerprint hashes these modules for
a recipe that declares the registration (``plugin_sha256.<module>``), so a change that can move that
architecture's output moves the recipes that declare it -- and no other recipe's key.

The embeddinggemma-2 fold's two registrations (``EmbeddingGemma2Config`` and ``EmbeddingGemma2Model``)
were deleted with the fold: the pinned image's registry carries the architecture and its transformers the
classes, and the recipe is a patch-only plugin carrier now (``serve.plugin_architectures`` empty,
``serve.patches: [torch-sdpa-attention]``), so the patch name is what keys the engine code it runs.
"""

__all__ = ["ARCHITECTURE_MODULES", "LAZY_MODEL_MODULES", "PLUGIN_ENGINE_MODULES", "register"]


def register() -> None:
    """Apply the opted-in engine patches, then register every folded architecture with the running vLLM
    (idempotent; runs once per engine process).

    The version guard runs before the registries are touched, so an unsupported engine fails with an explicit
    message instead of importing classes whose contracts may have moved. The patches apply before the
    architectures are registered: a patch is the code that makes an architecture importable or runnable
    (the embeddinggemma-2 recipe's ``torch-sdpa-attention`` registers the SDPA backend its
    ``--attention-backend`` pin selects), so it must never land after the registration that needs it.

    Raises:
        RuntimeError: the running vLLM is outside the validated range
            (:mod:`rcp_vllm.models.version_guard`), or a folded package's registration failed. The
            failure is printed with the package's name before it is re-raised: vLLM's entry-point loader
            catches it and only logs the traceback, and the packages are registered in turn, so an engine
            whose log does not name the package would start half-registered.
    """
    from .version_guard import require_vllm_version

    require_vllm_version()

    from rcp_vllm.patches import apply_opted_in_patches

    apply_opted_in_patches()

    from .pplx import register_pplx
    from .topk.plugin import register_topk

    try:
        register_topk()
    except Exception as error:
        _refuse("rcp_vllm.models.topk", error)
    try:
        register_pplx()
    except Exception as error:
        _refuse("rcp_vllm.models.pplx", error)


def _refuse(package: str, error: Exception) -> NoReturn:
    """Print a folded package's registration failure with its name, then re-raise it.

    vLLM's entry-point loader catches what :func:`register` raises and only logs the traceback, and the two
    packages register in turn: the refusal has to name the package in the log, or the engine starts
    half-registered with no line saying which half.
    """
    print(
        f"rcp-vllm: registering {package} failed ({type(error).__name__}: {error}); the engine would start "
        f"with {package}'s architectures unregistered",
        file=sys.stderr,
    )
    raise error
