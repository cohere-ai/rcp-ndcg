"""Plugin tests that need vLLM importable (the engine image's environment).

These skip with a clear reason wherever vLLM cannot import on CPU (the rcp-ndcg dev
venv carries torch only); on the GPU wave they run against the image's vLLM 0.31.0.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "vllm",
    reason="vLLM is not importable in this environment (the rcp-ndcg dev venv carries "
    "torch only, by design); the wired-pooler and model-class checks run on the GPU "
    "wave, whose engine image has vLLM 0.31.0",
)

from rcp_ndcg_vllm.models.pplx.pooler import PplxChunkPool, build_pooler  # noqa: E402  (conftest inserts src/)
from rcp_ndcg_vllm.models.pplx.pooling_core import (  # noqa: E402
    BOUNDARY_TOKEN_ID,
    DOCUMENT_PREFIX_TOKEN_IDS,
    QUERY_PREFIX_TOKEN_ID,
    PplxInt8Projection,
)
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler  # noqa: E402
from vllm.pooling_params import PoolingParams  # noqa: E402
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates  # noqa: E402

torch.manual_seed(20261006)

HIDDEN = 8
EMBED = 4
_TASK = "token_embed"

_NO_ENGINE_CONFIG = {"scheduler_config": {"enable_chunked_prefill": False, "async_scheduling": False}}


def _no_engine_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the engine's vllm config around ``AllPool``'s constructor.

    ``AllPool`` reads the current vllm config at construction (chunked prefill, async
    scheduling); a scheduler with chunked prefill off is what this model's serving
    config produces (encoder-only pooling disables chunked prefill). vLLM's own tests
    patch the same module global.
    """
    import types

    import vllm.model_executor.layers.pooler.tokwise.methods as methods_module

    config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(**_NO_ENGINE_CONFIG["scheduler_config"]))
    monkeypatch.setattr(methods_module, "get_current_vllm_config", lambda: config)


def _wired_metadata(pool: PplxChunkPool, sequences: list[list[int]]) -> PoolingMetadata:
    """A real ``PoolingMetadata`` for the given per-sequence token ids, as the v1 runner builds it."""
    params = PoolingParams(task=_TASK)
    # The same instance-method call the engine's pooling runner makes per request
    # (pooling_runner.add_request); it must be called on the instance.
    pool.get_pooling_updates(_TASK).apply(params)
    assert params.requires_token_ids is True

    max_len = max(len(s) for s in sequences)
    ids_cpu = torch.zeros((len(sequences), max_len), dtype=torch.int64)
    for row, ids in enumerate(sequences):
        ids_cpu[row, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
    scheduled = torch.tensor([len(s) for s in sequences], dtype=torch.int32)
    metadata = PoolingMetadata(
        prompt_lens=torch.tensor([len(s) for s in sequences], dtype=torch.int64),
        prompt_token_ids=None,  # the v1 runner leaves the device tensor unset
        prompt_token_ids_cpu=ids_cpu,
        pooling_params=[params] * len(sequences),
        pooling_states=[PoolingStates() for _ in sequences],
    )
    metadata.build_pooling_cursor(
        scheduled.numpy(),
        seq_lens_cpu=torch.tensor([len(s) for s in sequences], dtype=torch.int64),
        device=torch.device("cpu"),
    )
    return metadata


def _wired_forward(monkeypatch: pytest.MonkeyPatch, sequences: list[list[int]], hidden: torch.Tensor):
    _no_engine_config(monkeypatch)
    pool = PplxChunkPool()
    metadata = _wired_metadata(pool, sequences)
    return pool(hidden, metadata)


def test_wired_pooler_matches_the_token_space_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_ids = [
        DOCUMENT_PREFIX_TOKEN_IDS[0],
        DOCUMENT_PREFIX_TOKEN_IDS[1],
        11,
        12,
        13,
        BOUNDARY_TOKEN_ID,
        14,
        15,
    ]
    query_ids = [QUERY_PREFIX_TOKEN_ID, 21, 22]
    hidden = torch.randn(len(doc_ids) + len(query_ids), HIDDEN)
    outputs = _wired_forward(monkeypatch, [doc_ids, query_ids], hidden)

    assert outputs[0].shape == (2, HIDDEN), "one vector per chunk for the document"
    assert outputs[1].shape == (1, HIDDEN), "one vector for the query"
    assert torch.allclose(outputs[0][0], hidden[2:5].float().mean(0), atol=1e-6)
    assert torch.allclose(outputs[0][1], hidden[6:8].float().mean(0), atol=1e-6)
    # The query pools over the whole sequence, the [Q] prefix token included.
    assert torch.allclose(outputs[1][0], hidden[8:11].float().mean(0), atol=1e-6)


def test_wired_pooler_passes_the_warmup_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_engine_config(monkeypatch)
    hidden = torch.randn(6, HIDDEN)
    pool = PplxChunkPool()
    metadata = _wired_metadata(pool, [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    outputs = pool(hidden, metadata)
    assert all(out.shape == (1, HIDDEN) for out in outputs)


def test_build_pooler_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_engine_config(monkeypatch)
    import types

    fake_model_config = types.SimpleNamespace(
        head_dtype=torch.float32,
        pooler_config=types.SimpleNamespace(task=_TASK),
    )
    projector = PplxInt8Projection(HIDDEN, EMBED, dtype=torch.float32)
    pooler = build_pooler(fake_model_config, projector=projector)
    assert isinstance(pooler, TokenPooler)
    assert pooler.get_supported_tasks() == {_TASK}
    assert pooler.head.projector is projector
    updates = pooler.get_pooling_updates(_TASK)
    assert updates.requires_token_ids is True


def test_model_class_attributes_and_mapper() -> None:
    from rcp_ndcg_vllm.models.pplx.model import PplxContextualForPooling

    assert PplxContextualForPooling.is_pooling_model is True
    assert PplxContextualForPooling.default_seq_pooling_type == "CLS"
    assert PplxContextualForPooling.default_tok_pooling_type == "ALL"
    mapper = PplxContextualForPooling.hf_to_vllm_mapper
    assert mapper.map_name("language_model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )
    assert mapper.map_name("model.language_model.layers.1.mlp.down_proj.weight") == (
        "model.layers.1.mlp.down_proj.weight"
    )
    assert mapper.map_name("visual.blocks.0.attn.q.weight") is None
    assert mapper.map_name("mtp.fc.weight") is None
    assert mapper.map_name("contextual_projection.weight") == "contextual_projection.weight"


def test_registration_against_the_real_registry() -> None:
    import vllm
    from rcp_ndcg_vllm.models.pplx import PLUGIN_ARCHITECTURE, register
    from rcp_ndcg_vllm.models.version_guard import require_vllm_version

    version = require_vllm_version()
    assert version == (0, 31), f"the engine image pinned 0.31.x; found {vllm.__version__}"
    register()
    assert PLUGIN_ARCHITECTURE in vllm.ModelRegistry.get_supported_archs()
