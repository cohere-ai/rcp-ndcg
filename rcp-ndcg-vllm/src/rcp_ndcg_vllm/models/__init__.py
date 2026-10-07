"""The folded model plugins: one ``vllm.general_plugins`` entry point registering each architecture lazily.

vLLM loads this group in every engine process at startup and calls :func:`register`, which guards the engine
version once and registers each folded model's architecture and config class. Every model class is registered
as a lazy ``"module:Class"`` string, so the registry imports ``rcp_ndcg_vllm.models.topk.model`` /
``rcp_ndcg_vllm.models.pplx.model`` only when a request for the architecture actually needs it; importing this
package (or executing :func:`register`) never imports torch or vLLM here — the model modules do, inside vLLM.
The topk and pplx distributions folded into this one wheel (layout-move item 3): there is a single
``vllm.general_plugins`` entry point (``rcp-ndcg-vllm``) and a single version guard
(:mod:`rcp_ndcg_vllm.models.version_guard`).
"""

from __future__ import annotations

__all__ = ["LAZY_MODEL_MODULES", "register"]

LAZY_MODEL_MODULES: tuple[str, ...] = (
    "rcp_ndcg_vllm.models.pplx.config",
    "rcp_ndcg_vllm.models.pplx.model",
    "rcp_ndcg_vllm.models.pplx.pooler",
    "rcp_ndcg_vllm.models.pplx.pooling_core",
    "rcp_ndcg_vllm.models.topk",
    "rcp_ndcg_vllm.models.topk.config",
    "rcp_ndcg_vllm.models.topk.model",
    "rcp_ndcg_vllm.models.topk.plugin",
    "rcp_ndcg_vllm.models.topk.pooling",
    "rcp_ndcg_vllm.models.topk.weights",
)
"""The registry-lazy model modules: they import vLLM/torch (and the version guard refuses outside an engine)
by design, and only vLLM imports them -- as the ``module:Class`` strings the entry point registers. Importing
every OTHER module of this package needs neither torch nor vLLM (test_no_torch, and the contract walks)."""


def register() -> None:
    """Register every folded architecture with the running vLLM (idempotent; runs once per engine process).

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
