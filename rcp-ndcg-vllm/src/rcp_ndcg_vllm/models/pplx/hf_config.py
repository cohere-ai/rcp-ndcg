"""The checkpoint's configuration class, registered locally with transformers.

The checkpoint's ``config.json`` carries ``auto_map`` pointing its ``AutoConfig`` at
remote code (``configuration_pplx_contextual.PplxContextualConfig``).  With
``--trust-remote-code`` the engine downloads and executes that remote module while
resolving the configuration; without the flag it dies with the "contains custom code"
error.  This module therefore restates the remote config class locally and
``rcp_ndcg_vllm.models.pplx.register`` adds it to transformers' ``AutoConfig`` registry.  With
``model_type`` registered locally, transformers takes the explicit-local-code path --
``auto_factory`` skips the remote branch whenever the registered class's ``__module__``
is not ``transformers.*`` (the same mechanism
``rcp_ndcg_vllm.models.topk.config.TopkEmbedConfig`` rides) -- so neither ``--trust-remote-code``
nor a runtime code download is needed: the engine parses ``config.json`` into the same
object the remote code builds.

The class below restates ``configuration_pplx_contextual.py`` at the pinned revision
b667039ee8b438a6350fbc91bbcecd86f9d363ba (lines 6-27) line for line; every field and
default is the remote code's.  The boundary marker's default literal is an angle-bracket
token and is therefore *built here from characters*, never typed
(``chr(60) + "..." + chr(124) + ">"``) -- the value equals the checkpoint's
``config.json`` ``boundary_marker`` at the pinned revision byte for byte (read from that file on
2026-10-07; the file itself is not available offline). The contract test pins the built string
against the pplx recipe's reference constant ``BOUNDARY_MARKER``, which carries the same
config.json value.

vLLM consumes ``text_config``/``vision_config`` (through transformers' own Qwen3.5
machinery, as the class subclasses ``Qwen3_5Config`` exactly like the remote one) and
``embedding_dim``; the prefix/marker/cap fields are parsed so the config object is
faithful for inspection and for the remote code's own calls.
"""

from __future__ import annotations

from typing import Any

from transformers import Qwen3_5Config

__all__ = ["PplxContextualConfig", "BOUNDARY_MARKER_TEXT"]

#: The chunk boundary marker's literal text (an added token; referred to by name
#: ``chunk_sep`` everywhere else).  Built from characters per the house rule that
#: angle-bracket special tokens are never typed out in source.
BOUNDARY_MARKER_TEXT = chr(60) + "|chunk_sep" + chr(124) + ">"


class PplxContextualConfig(Qwen3_5Config):
    """``pplx_contextual_qwen3_5``: a Qwen3.5 config plus the retrieval knobs.

    A faithful restatement of the checkpoint's remote ``PplxContextualConfig``
    (``configuration_pplx_contextual.py:6-27`` at the pinned revision): the ``model_type``
    and the Qwen3.5 sub-config machinery, plus the prefixes and caps the model's
    ``prepare_inputs`` reads.  The ``__post_init__`` behaviour is the remote ``__init__``
    sequence after ``super().__init__(**kwargs)`` (the caps feed
    ``max_position_embeddings``; caching is off on the text config).
    """

    model_type = "pplx_contextual_qwen3_5"

    def __init__(
        self,
        embedding_dim: int = 2048,
        query_prefix: str = "[Q] ",
        document_prefix: str = "[D] ",
        boundary_marker: str = BOUNDARY_MARKER_TEXT,
        query_length: int = 32768,
        document_length: int = 32768,
        **kwargs: Any,
    ) -> None:
        """Mirror the remote ``__init__`` field for field (see the module docstring)."""
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.boundary_marker = boundary_marker
        self.query_length = query_length
        self.document_length = document_length
        self.max_position_embeddings = document_length
        self.text_config.use_cache = False
