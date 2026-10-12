"""The folded ``embedding_gemma2`` modules (see the parent package's docstring).

The four ``*_embedding_gemma2.py`` files here are transformers 5.19.0's ``models/embedding_gemma2/``
modules (configuration, modeling, processing, video processing) and ``embedding_gemma2.py`` is vLLM
nightly ``8cbd5d03``'s ``model_executor/models/embedding_gemma2.py``, each with a header naming the
upstream file. The parent package loads them by path under their upstream module names, so their
relative imports resolve inside transformers.

Importing this package installs them under those upstream names **before any folded module body
runs**: vLLM imports the folded model module as the registered ``"module:Class"`` string, and that
module's first statements are ``from transformers.models.embedding_gemma2...`` -- the pinned image's
transformers does not carry the package, so those imports resolve only if the fold is installed
first, in every process that imports the model (E2 round 5: the engine core imported the model with
the plugin loaded but the fold's patch not applied there, and died with
``ModuleNotFoundError: No module named 'transformers.models.embedding_gemma2'``). The installation is
the parent package's :func:`~rcp_ndcg_vllm.models.embedding_gemma2.install_fold`: idempotent, and
inert where the running transformers already carries the classes. It also sets the folded classes on
this package, because it declares this package as their ``__module__`` (transformers'
``_LazyAutoMapping.register`` refuses a key whose ``__module__`` starts with ``transformers.``): a
folded class's declared module must carry it, or the checkpoint's config cannot be pickled when vLLM
spawns the engine core. Where transformers is not importable at all this package imports clean (the
vendored model's own ``from transformers ...`` raises the library's message), so the no-transformers
CPU environments can import it.

The folded modules import transformers (and torch, through it), so they are registry-lazy:
``rcp_ndcg_vllm.models.LAZY_MODEL_MODULES`` keeps the no-torch harness scan from importing them.
"""

from __future__ import annotations

import importlib.util

__all__: list[str] = []

if importlib.util.find_spec("transformers") is not None:
    from rcp_ndcg_vllm.models.embedding_gemma2 import install_fold

    install_fold()
