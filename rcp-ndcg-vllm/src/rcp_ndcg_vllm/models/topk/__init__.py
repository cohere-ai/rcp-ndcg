"""vLLM general plugin for ``topk-io/topk-embed-v1-small``.

The distribution registers the checkpoint's architecture (``TopkEmbedModel``)
with vLLM's model registry through a ``vllm.general_plugins`` entry point, so
the unmodified ``vllm/vllm-openai`` image serves the model after
``pip install --no-deps <wheel>``.

Importing this package never imports vLLM or torch: the entry-point loader and
the CPU-only tests run where vLLM is absent.  The model class lives in
``rcp_ndcg_vllm.models.topk.model`` and is imported lazily by the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.0.1"

__all__ = ["TopkEmbedModel", "__version__", "register"]

if TYPE_CHECKING:
    # Re-exported lazily (see __getattr__); these imports are type-only so the
    # package import itself needs neither vLLM nor torch.
    from .model import TopkEmbedModel
    from .plugin import register_topk as register


def __getattr__(name: str):  # noqa: ANN202 - PEP 562 module hook
    """Resolve lazily so importing the package needs neither vLLM nor torch."""
    if name == "register":
        from .plugin import register_topk as register

        return register
    if name == "TopkEmbedModel":
        from .model import TopkEmbedModel

        return TopkEmbedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
