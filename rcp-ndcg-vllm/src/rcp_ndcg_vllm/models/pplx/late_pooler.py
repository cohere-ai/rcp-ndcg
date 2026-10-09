"""The pplx-late plugin's pooler: per-token vectors with the declared keep-rule applied engine-side.

vLLM's own ``TokenPooler`` runs two stages: a *pooling method* over the batch's hidden states
(``TokenPoolingFn``), then a *head* (projector, optional Matryoshka cut, activation). The late-interaction
recipes serve ``token_embed`` through the stock ``AllPool`` method -- one vector per prompt token -- and
this module replaces it with ``PplxLateKeepPool``: the same ``AllPool`` batch work (the per-sequence split,
the chunked-prefill caching) followed by the declared keep-rule, so the vectors at the rule's excluded
positions never reach the head and the wire carries only kept vectors.

The rule itself lives in :mod:`rcp_ndcg_vllm.models.pplx.late_keep` (the declared ids and the positions
they keep); this module is the vLLM-side adapter: it asks vLLM's pooling runner for the prompt token ids
(``PoolingParamsUpdate.requires_token_ids``, which makes the runner build their CPU copy -- the v1 runner
leaves ``prompt_token_ids`` ``None``, so ``get_prompt_token_ids()`` would raise) and indexes each
sequence's hidden states with them. The head is vLLM's own ``TokenEmbeddingPoolerHead``, exactly as
``pooler_for_token_embed`` builds it (the checkpoint's projection as the projector, ``PoolerNormalize()``
unless the request sends ``use_activation: false``), so every request-level knob behaves as it does for the
in-tree token poolers.
"""

from __future__ import annotations

__all__ = ["PplxLateKeepPool", "build_late_pooler"]

from collections.abc import Sequence, Set

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.tokwise.heads import TokenEmbeddingPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import (
    AllPool,
    TokenPoolingMethodOutputItem,
)
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from rcp_ndcg_vllm.models.pplx.late_keep import declared_skip_ids, kept_positions


class PplxLateKeepPool(AllPool):
    """Per-token pooling over the flattened batch, with the declared keep-rule's positions dropped.

    ``AllPool`` (vLLM ``tokwise/methods.py``) does the batch work -- splitting the flattened hidden states
    per request and caching chunked-prefill pieces until a sequence is complete; this subclass indexes each
    finished sequence by the positions the declared rule keeps. The rule is the engine's half of the
    recipe's declaration (``serve.hf_overrides.document_skip_token_ids``); an empty rule keeps every
    position, which is what the stock pooler returns.
    """

    def __init__(self, *, skip_ids: Sequence[int] = ()) -> None:
        super().__init__()
        self._skip_ids = tuple(skip_ids)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        # The rule reads the prompt token ids; this flag is what makes vLLM's pooling runner build their
        # CPU copy (PoolingParamsUpdate.requires_token_ids).
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data = super().forward(hidden_states, pooling_metadata)
        if not self._skip_ids:
            return pooled_data
        token_ids_cpu = pooling_metadata.get_prompt_token_ids_cpu()
        pooled: list[TokenPoolingMethodOutputItem] = []
        for data, token_ids in zip(pooled_data, token_ids_cpu, strict=True):
            # Unfinished chunked prefill: AllPool already returned None for the sequence.
            if data is None:
                pooled.append(None)
                continue
            positions = kept_positions([int(token) for token in token_ids.tolist()], self._skip_ids)
            index = torch.tensor(positions, dtype=torch.long, device=data.device)
            pooled.append(data[index])
        return pooled


def build_late_pooler(model_config: ModelConfig, *, projector: torch.nn.Module) -> TokenPooler:
    """The late model's ``TokenPooler``: the keep pool over the batch + vLLM's token-embed head.

    Mirrors vLLM's ``pooler_for_token_embed`` (``tokwise/poolers.py``) with ``PplxLateKeepPool`` in place of
    the configured token pooling method: ``head_dtype`` (fp32 for pooling models) casts the vectors, the
    projector applies the checkpoint's trained head, and ``activation=PoolerNormalize()`` L2-normalises
    unless the request sends ``use_activation: false``. The rule is read from the served model's HF config
    (:func:`~rcp_ndcg_vllm.models.pplx.late_keep.declared_skip_ids`); the keep pool's method is the
    ``token_embed`` per-token contract the late-interaction recipes serve.

    Args:
        model_config: vLLM's ``ModelConfig``; ``head_dtype`` and the declared rule are read.

        projector: The model's trained projection (``ColQwen3_5Model.custom_text_proj``), registered on the
            model so the checkpoint's Dense head loads into it.

    Raises:
        ValueError: ``hf_overrides.document_skip_token_ids`` is not a list of token ids.
    """
    head = TokenEmbeddingPoolerHead(
        head_dtype=model_config.head_dtype,
        projector=projector,
        activation=PoolerNormalize(),
    )
    return TokenPooler(pooling=PplxLateKeepPool(skip_ids=declared_skip_ids(model_config)), head=head)
