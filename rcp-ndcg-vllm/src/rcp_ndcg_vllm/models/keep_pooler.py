"""The plugin's keep pooler: per-token vectors with the declared keep-rules applied engine-side.

vLLM's own ``TokenPooler`` runs two stages: a *pooling method* over the batch's hidden states
(``TokenPoolingFn``), then a *head* (projector, optional Matryoshka cut, activation). Both late-interaction
families serve ``token_embed`` through the stock ``AllPool`` method -- one vector per prompt token -- and this
module replaces it with ``KeepPool``: the same ``AllPool`` batch work (the per-sequence split, the
chunked-prefill caching) followed by the recipe's declared keep-rules, so the vectors at the rules' excluded
positions never reach the head and the wire carries only kept vectors.

The rules themselves live in :mod:`rcp_ndcg_vllm.models.keep_rule` (the declared ids, their gates and the
positions they keep); this module is the vLLM-side adapter: it asks vLLM's pooling runner for the prompt
token ids (``PoolingParamsUpdate.requires_token_ids``, which makes the runner build their CPU copy -- the v1
runner leaves ``prompt_token_ids`` ``None``, so ``get_prompt_token_ids()`` would raise) and indexes each
sequence's hidden states with them. The head is the plugin's one head construction
(:func:`~rcp_ndcg_vllm.models.token_pooler.token_embed_pooler`, shared with the contextual pooler), so every
request-level knob behaves as it does for the in-tree token poolers.
"""

from __future__ import annotations

__all__ = ["KeepPool", "build_keep_pooler"]

from collections.abc import Sequence, Set

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import (
    AllPool,
    TokenPoolingMethodOutputItem,
)
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from rcp_ndcg_vllm.models.keep_rule import (
    declared_keep_ids,
    declared_skip_ids,
    declared_skip_prefix_id,
    kept_positions,
)
from rcp_ndcg_vllm.models.token_pooler import token_embed_pooler


class KeepPool(AllPool):
    """Per-token pooling over the flattened batch, with the declared rules' excluded positions dropped.

    ``AllPool`` (vLLM ``tokwise/methods.py``) does the batch work -- splitting the flattened hidden states
    per request and caching chunked-prefill pieces until a sequence is complete; this subclass indexes each
    finished sequence by the positions the declared rules keep: the media allowlist for a row that carries
    one of its ids, the text skip rule (gated by the document role prefix) for every other document row, and
    every position for a query. The rules are the engine's half of the recipe's declarations
    (``serve.hf_overrides.document_skip_token_ids`` / ``document_keep_token_ids`` and the skip rule's
    ``document_skip_prefix_token_id``); with no rule declared the class keeps every position, which is what
    the stock pooler returns.
    """

    def __init__(
        self,
        *,
        skip_ids: Sequence[int] = (),
        document_prefix_id: int | None = None,
        keep_ids: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self._skip_ids = tuple(skip_ids)
        self._document_prefix_id = document_prefix_id
        self._keep_ids = tuple(keep_ids)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        # The rules read the prompt token ids; this flag is what makes vLLM's pooling runner build their
        # CPU copy (PoolingParamsUpdate.requires_token_ids).
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data = super().forward(hidden_states, pooling_metadata)
        if not self._skip_ids and not self._keep_ids:
            return pooled_data
        token_ids_cpu = pooling_metadata.get_prompt_token_ids_cpu()
        pooled: list[TokenPoolingMethodOutputItem] = []
        for data, token_ids in zip(pooled_data, token_ids_cpu, strict=True):
            # Unfinished chunked prefill: AllPool already returned None for the sequence.
            if data is None:
                pooled.append(None)
                continue
            positions = kept_positions(
                [int(token) for token in token_ids.tolist()],
                self._skip_ids,
                document_prefix_id=self._document_prefix_id,
                keep_ids=self._keep_ids,
            )
            index = torch.tensor(positions, dtype=torch.long, device=data.device)
            pooled.append(data[index])
        return pooled


def build_keep_pooler(model_config: ModelConfig, *, projector: torch.nn.Module) -> TokenPooler:
    """A late-interaction model's ``TokenPooler``: the keep pool over the batch + the plugin's head.

    The rules are read from the served model's HF config
    (:func:`~rcp_ndcg_vllm.models.keep_rule.declared_skip_ids`, its document role prefix
    ``document_skip_prefix_token_id`` and the media allowlist
    :func:`~rcp_ndcg_vllm.models.keep_rule.declared_keep_ids`); the head comes from the plugin's one
    construction (:func:`~rcp_ndcg_vllm.models.token_pooler.token_embed_pooler`), and the keep pool's method
    is the ``token_embed`` per-token contract the late-interaction recipes serve.

    Args:
        model_config: vLLM's ``ModelConfig``; ``head_dtype`` and the declared rules are read.

        projector: The model's trained projection (``ColQwen3_5Model.custom_text_proj`` for the pplx-late
            checkpoints, ``custom_text_proj`` -- the renamed ``head.`` -- for topk), registered on the model
            so the checkpoint's head loads into it.

    Raises:
        ValueError: a declared rule is not a list of token ids, or the TEXT skip rule is declared without
            ``document_skip_prefix_token_id`` (that rule is document-side: without the role gate it would
            drop a query prompt's positions too, diverging from the checkpoint's own mask and the
            reference's ``encode_query``).
    """
    skip_ids = declared_skip_ids(model_config)
    document_prefix_id = declared_skip_prefix_id(model_config)
    keep_ids = declared_keep_ids(model_config)
    if skip_ids and document_prefix_id is None:
        raise ValueError(
            "the served model's hf_overrides declares document_skip_token_ids without "
            "document_skip_prefix_token_id: the keep-rule is document-side, and without the document role "
            "prefix it would drop a query prompt's positions too (the checkpoint's mask declares "
            "skiplist_tasks: ['document'])"
        )
    pooling = KeepPool(skip_ids=skip_ids, document_prefix_id=document_prefix_id, keep_ids=keep_ids)
    return token_embed_pooler(model_config, projector=projector, pooling=pooling)
