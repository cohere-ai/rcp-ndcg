"""The checkpoints' configuration classes, registered locally with transformers.

Two pplx checkpoints carry a custom ``model_type`` whose ``config.json`` points
``AutoConfig`` at remote code; both are restated here and
``rcp_ndcg_vllm.models.pplx.register`` adds them to transformers' ``AutoConfig`` registry.
With ``model_type`` registered locally, transformers takes the explicit-local-code
path -- ``auto_factory`` skips the remote branch whenever the registered class's
``__module__`` is not ``transformers.*`` (the same mechanism
``rcp_ndcg_vllm.models.topk.config.TopkEmbedConfig`` rides) -- so neither
``--trust-remote-code`` nor a runtime code download is needed: the engine parses
``config.json`` into the same object the remote code builds.

``PplxContextualConfig`` restates ``configuration_pplx_contextual.py`` at the pinned
revision b667039ee8b438a6350fbc91bbcecd86f9d363ba (lines 6-27) line for line; every field
and default is the remote code's.  The boundary marker's default literal is an
angle-bracket token and is therefore *built here from characters*, never typed
(``chr(60) + "..." + chr(124) + ">"``) -- the value equals the checkpoint's
``config.json`` ``boundary_marker`` at the pinned revision byte for byte (read from that
file on 2026-10-07; the file itself is not available offline). The contract test pins the
built string against the pplx recipe's reference constant ``BOUNDARY_MARKER``, which
carries the same config.json value.

``PplxV1Config`` restates ``configuration.py`` of the pplx-embed-v1 family at its pinned
revisions (2c4d510dd4a732063c31a0f70193e35067b51fd8 and
06456497a00540a582918fe8dcd3a5eabb207772; the file is byte-identical at both): a plain
``Qwen3Config`` subclass whose only delta is the ``model_type`` name.  The checkpoint's
``modeling.PPLXQwen3Model`` is likewise ``Qwen3Model`` with bidirectional attention (its
``post_init`` flips every layer's ``is_causal`` and its ``forward`` builds an all-to-all
mask), which the recipe expresses on the stock engine with
``serve.hf_overrides {architectures: [Qwen3ForCausalLM], is_causal: false}`` -- this class
only makes the config parse locally, so no remote code runs and
``serve.trust_remote_code`` stays false.
"""

from __future__ import annotations

from typing import Any

from transformers import Qwen3_5Config, Qwen3Config

__all__ = ["BOUNDARY_MARKER_TEXT", "PplxContextualConfig", "PplxV1Config"]

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


class PplxV1Config(Qwen3Config):
    """``bidirectional_pplx_qwen3``: the pplx-embed-v1 family's Qwen3 config.

    A faithful restatement of the checkpoint's remote ``configuration.py`` at the pinned
    revisions: the remote module imports ``Qwen3Config`` and changes only the
    ``model_type``.  Every other field (``use_bidirectional_attention``,
    ``max_position_embeddings``, the per-layer ``layer_types``) parses through the
    inherited Qwen3 machinery; the recipe's ``serve.hf_overrides`` carries the
    bidirectional contract the remote ``modeling.py`` implements.
    """

    model_type = "bidirectional_pplx_qwen3"
