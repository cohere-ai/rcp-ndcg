"""The folded model plugins: one ``vllm.general_plugins`` entry point registering each architecture lazily.

vLLM loads this group in every engine process at startup and calls :func:`register`, which guards the engine
version once, registers each folded model's architecture and config class, and applies the opted-in engine
patches of :mod:`rcp_ndcg_vllm.patches` (the one place the engine process reads ``RCP_NDCG_VLLM_PATCHES``).
Every model class is registered as a lazy ``"module:Class"`` string, so the registry imports
``rcp_ndcg_vllm.models.topk.model`` /
``rcp_ndcg_vllm.models.pplx.model`` only when a request for the architecture actually needs it; importing this
package (or executing :func:`register`) never imports torch or vLLM here — the model modules do, inside vLLM.
The topk and pplx distributions folded into this one wheel (layout-move item 3): there is a single
``vllm.general_plugins`` entry point (``rcp-ndcg-vllm``) and a single version guard
(:mod:`rcp_ndcg_vllm.models.version_guard`).
"""

from __future__ import annotations

LAZY_MODEL_MODULES: tuple[str, ...] = (
    "rcp_ndcg_vllm.models.pplx.config",
    "rcp_ndcg_vllm.models.pplx.hf_config",
    "rcp_ndcg_vllm.models.pplx.late",
    "rcp_ndcg_vllm.models.pplx.late_data",
    "rcp_ndcg_vllm.models.pplx.model",
    "rcp_ndcg_vllm.models.pplx.pooler",
    "rcp_ndcg_vllm.models.pplx.pooling_core",
    "rcp_ndcg_vllm.models.topk.config",
    "rcp_ndcg_vllm.models.topk.model",
    "rcp_ndcg_vllm.models.topk.pooling",
    "rcp_ndcg_vllm.models.topk.plugin",
    "rcp_ndcg_vllm.models.topk.weights",
)
"""The registry-lazy model modules: they import vLLM/torch by design, and only vLLM imports them (as the
``module:Class`` strings the entry point registers). Everything else -- including this entry-point callable
and the one version guard -- must import clean (``tests/test_no_torch.py`` pins it; the surface walk skips
these)."""

PLUGIN_ENGINE_MODULES: tuple[str, ...] = (
    "rcp_ndcg_vllm.models",
    "rcp_ndcg_vllm.models.version_guard",
    "rcp_ndcg_vllm.models.pplx",
    "rcp_ndcg_vllm.models.pplx.config",
    "rcp_ndcg_vllm.models.pplx.hf_config",
    "rcp_ndcg_vllm.models.topk",
    "rcp_ndcg_vllm.models.topk.config",
    "rcp_ndcg_vllm.models.topk.plugin",
    "rcp_ndcg_vllm.patches",
)
"""The engine-side modules every plugin recipe's engine imports at registration, whatever its architecture:
:func:`register`'s entry-point callable, the one version guard, the package inits, the two config
registrations and the patch applier.  A recipe that declares ``serve.plugin`` keys these beside its
architectures' modules (``rcp-fp/4``); a change to one of them moves every plugin recipe, because every
plugin engine imports it.  The patch modules themselves are keyed per recipe, by its declared
``serve.patches``: a patch is imported unconditionally but applied only when opted in."""

ARCHITECTURE_MODULES: dict[str, tuple[str, ...]] = {
    # The contextual chunk model (pplx-embed-v2-context-9b-preview): the model class imports its pooler
    # and the pooling core (the config registration is shared, in PLUGIN_ENGINE_MODULES).
    "PplxContextualModel": (
        "rcp_ndcg_vllm.models.pplx.model",
        "rcp_ndcg_vllm.models.pplx.pooler",
        "rcp_ndcg_vllm.models.pplx.pooling_core",
    ),
    # The late-interaction sibling (pplx-embed-v2-late-0.6b): its model class and the Dense-head mapping.
    "Qwen3_5Model": (
        "rcp_ndcg_vllm.models.pplx.late",
        "rcp_ndcg_vllm.models.pplx.late_data",
    ),
    # The topk multimodal late-interaction model: its model and weight mapping (its config registration is
    # shared, in PLUGIN_ENGINE_MODULES).
    "TopkEmbedModel": (
        "rcp_ndcg_vllm.models.topk.model",
        "rcp_ndcg_vllm.models.topk.weights",
    ),
}
"""Every architecture this wheel registers, mapped to the engine-side modules that implement it.

One home: the registration constants (``PLUGIN_ARCHITECTURE``, ``LATE_ARCHITECTURE``,
``topk.plugin.MODEL_ARCHITECTURE``) are the truth, and ``tests/test_plugin_modules.py`` pins the keys
against them and the values against :data:`LAZY_MODEL_MODULES`.  The behaviour fingerprint hashes these
modules for a recipe that declares the architecture (``plugin_sha256.<module>``), so a change that can move
that architecture's output moves the recipes that declare it -- and no other recipe's key."""

__all__ = ["ARCHITECTURE_MODULES", "LAZY_MODEL_MODULES", "PLUGIN_ENGINE_MODULES", "register"]


def register() -> None:
    """Register every folded architecture with the running vLLM and apply the opted-in engine patches
    (idempotent; runs once per engine process).

    The version guard runs before the registries are touched, so an unsupported engine fails with an explicit
    message instead of importing classes whose contracts may have moved.

    Raises:
        RuntimeError: the running vLLM is outside the validated range
            (:mod:`rcp_ndcg_vllm.models.version_guard`).
    """
    from .version_guard import require_vllm_version

    require_vllm_version()

    from .pplx import register_pplx
    from .topk.plugin import register_topk

    register_topk()
    register_pplx()

    from rcp_ndcg_vllm.patches import apply_opted_in_patches

    apply_opted_in_patches()
