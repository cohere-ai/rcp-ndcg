"""The checkpoint's configuration class, registered locally with transformers.

The checkpoint's ``config.json`` carries ``auto_map`` pointing its
``AutoConfig`` at remote code (``modeling_topk_embed.TopkEmbedConfig``), whose
module imports ``.hf_backbone`` and, through it, ``fla``
(flash-linear-attention) at module import — a package the engine image does
not ship.  With ``--trust-remote-code`` on the stock image, transformers
executes that remote module while resolving the configuration, so
``vllm serve`` dies with ``ModuleNotFoundError: No module named 'fla'`` before
the model class is ever constructed; without the flag it dies with the
"contains custom code" error.  This module therefore restates the remote
config class locally and ``plugin.register`` adds it to transformers'
``AutoConfig`` registry.  With ``model_type`` registered locally, transformers
takes the explicit-local-code path — ``auto_factory`` skips the remote branch
whenever the registered class's ``__module__`` is not ``transformers.*``
(``explicit_local_code``, every transformers 5.x from 5.10 to 5.17) — so the
engine never executes the checkpoint's remote code: no ``fla`` requirement, no
runtime code download, and ``--trust-remote-code`` becomes unnecessary (and
harmless: registration wins under either flag value).

The class below restates ``modeling_topk_embed.py:12-29`` at the pinned
revision e54485ebab921f2c18c4d092b3f4c40dcca26781, line for line; every field
and default is the remote code's.  ``modeling_topk_embed.py`` itself is never
imported at serve time (vLLM resolves the model class from its own registry),
so this is the only config surface the engine consumes.
"""

from __future__ import annotations

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

__all__ = ["TopkEmbedConfig", "MODEL_TYPE"]

MODEL_TYPE = "topk_embed"


class TopkEmbedConfig(Qwen3_5Config):
    """``topk_embed``: a Qwen3.5 multimodal config plus the retriever knobs.

    A faithful restatement of the checkpoint's remote ``TopkEmbedConfig``
    (modeling_topk_embed.py:12-29): ``model_type`` and ``sub_configs`` so
    ``text_config`` / ``vision_config`` parse through transformers' own
    Qwen3.5 machinery, and the retrieval knobs the checkpoint adds at the top
    level.  vLLM consumes only ``dim`` (the projection output size, via
    ColQwen3_5Model) and the sub-configs; the remaining knobs are parsed so
    the config object is faithful for inspection (the prompts, the skip ids
    and the caps are recipe/client concerns, not engine ones).
    """

    model_type = MODEL_TYPE
    sub_configs = Qwen3_5Config.sub_configs
    dim: int = 1024
    output_dim: int | None = None
    normalize: bool = True
    query_template: str = "Query: "
    document_prompt: str = "Document: "
    image_token_budget: int = 1280
    scoring_skip_ids: list[int] | None = None

    def __post_init__(self, **kwargs) -> None:
        """Mirror the remote ``__post_init__`` (modeling_topk_embed.py:23-29):
        default ``output_dim`` to ``dim``, keep the MRL bound honest, and
        disable caching on the text config the way the checkpoint ships."""
        super().__post_init__(**kwargs)
        if self.output_dim is None:
            self.output_dim = self.dim
        if not 0 < self.output_dim <= self.dim:
            raise ValueError(f"output_dim must be in 1..{self.dim}")
        self.text_config.use_cache = False
