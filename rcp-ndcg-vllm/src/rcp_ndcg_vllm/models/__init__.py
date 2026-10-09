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
    "rcp_ndcg_vllm.models.pplx.late_pooler",
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

__all__ = ["LAZY_MODEL_MODULES", "register"]


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
