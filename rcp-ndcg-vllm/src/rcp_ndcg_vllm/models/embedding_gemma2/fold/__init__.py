"""The folded transformers ``embedding_gemma2`` modules (see the parent package's docstring).

The three files here are transformers 5.19.0's ``models/embedding_gemma2/{configuration,processing,
video_processing}_embedding_gemma2.py``, each with a header naming the upstream file; the parent
package loads them by path under their upstream module names, so their relative imports resolve
inside transformers. They import transformers (and torch, through it), so they are
registry-lazy: ``rcp_ndcg_vllm.models.LAZY_MODEL_MODULES`` keeps the no-torch harness scan from
importing them.
"""

from __future__ import annotations

__all__: list[str] = []
