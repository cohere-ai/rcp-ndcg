"""The embeddinggemma-2 fold patch: the transformers classes the digest-pinned engine nightly lacks.

E2 r1: ``google/embeddinggemma-2`` on the digest-pinned nightly failed at serve -- the image's
transformers does not know the checkpoint's ``model_type: embedding_gemma2``, so ``AutoConfig``
refuses ``config.json`` and vLLM's own ``embedding_gemma2`` model module cannot import the
``transformers.models.embedding_gemma2.*`` classes it needs. The owner's 2026-10-10 decision:
backport the architecture into the engine environment through the plugin. This patch installs
:func:`rcp_ndcg_vllm.models.embedding_gemma2.install_fold`: it loads transformers 5.19.0's three
``embedding_gemma2`` modules (shipped in this wheel under
``rcp_ndcg_vllm/models/embedding_gemma2/fold/``) under their upstream module names and registers the
config class with ``AutoConfig`` and the processor and video-processor classes with
``AutoProcessor`` / ``AutoVideoProcessor``.

Opt-in: the embeddinggemma-2 recipe declares ``serve.patches: [embeddinggemma2-transformers-fold]``,
so only that engine process installs the fold and only that recipe's fingerprint keys this module
and the fold's files. It is inert when the running transformers already carries the classes.

Removal condition: delete this patch (the recipe's ``serve.patches`` entry with it) when
``engine.image`` moves to a vLLM image whose transformers ships ``embedding_gemma2``.
"""

from __future__ import annotations

__all__ = ["PATCH_NAME", "apply"]

#: The opt-in name (``serve.patches``; the recipe schema validates it against ``PATCH_NAMES``).
PATCH_NAME = "embeddinggemma2-transformers-fold"


def apply() -> None:
    """Install the folded transformers classes into the running engine (idempotent)."""
    from rcp_ndcg_vllm.models.embedding_gemma2 import install_fold  # noqa: PLC0415

    install_fold()
