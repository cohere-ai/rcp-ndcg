"""The local configuration class and its AutoConfig registration.

Skips where transformers cannot be imported (the CPU dev environment has no
transformers; the engine image and the reference environment do, and the
round-2 verifiers run it there in a scratch venv).

The behavioural core — transformers' AutoConfig must take the locally
registered class instead of executing the checkpoint's remote config code
(whose module imports ``fla``, absent from the engine image) — is proven
end-to-end by ``test_registered_config_wins_over_remote_code`` with a local
directory that carries the checkpoint's own ``auto_map``.
"""

# ruff: isort: skip  (the transformers import guard must precede the imports below)

from __future__ import annotations

import json
from pathlib import Path

import pytest

TRANSFORMERS_MISSING_REASON = (
    "transformers is not importable on this CPU environment; the config-class "
    "tests run on the engine image (GPU wave T0) and in the round-2 "
    "verifier's scratch venv"
)

# The module under test imports transformers (the config class subclasses
# transformers' Qwen3_5Config), so this file skips where transformers is
# absent; the imports below must follow that guard.
pytest.importorskip("transformers", reason=TRANSFORMERS_MISSING_REASON)

import rcp_ndcg_vllm.models.topk.config as config_module  # noqa: E402
from rcp_ndcg_vllm.models.topk.config import MODEL_TYPE, TopkEmbedConfig  # noqa: E402

TINY_VISION_CONFIG = {
    "model_type": "qwen3_5",
    "depth": 1,
    "hidden_size": 8,
    "num_heads": 2,
    "intermediate_size": 8,
    "patch_size": 2,
    "spatial_merge_size": 1,
    "out_hidden_size": 8,
    "num_position_embeddings": 4,
    "temporal_patch_size": 1,
    "in_channels": 1,
}


def tiny_text_config(use_cache: bool = True) -> dict:
    """A minimal qwen3_5_text sub-config (same architecture shape, tiny dims)."""
    return {
        "model_type": "qwen3_5_text",
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "vocab_size": 16,
        "is_causal": False,
        "use_cache": use_cache,
    }


def write_tiny_checkpoint(tmp_path: Path, config: dict) -> Path:
    """Write a minimal checkpoint directory (config.json only)."""
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="module")
def auto_config():
    """transformers' AutoConfig (skips the module where transformers is absent)."""
    return pytest.importorskip("transformers").AutoConfig


@pytest.fixture
def registered() -> None:
    """Register the config the way the entry-point callable does (idempotent)."""
    from transformers import AutoConfig

    AutoConfig.register(MODEL_TYPE, TopkEmbedConfig, exist_ok=True)


def test_config_class_mirrors_the_remote_fields() -> None:
    """The restated class carries the remote TopkEmbedConfig's ``model_type``
    and retrieval knobs with the same defaults (modeling_topk_embed.py:12-22)."""
    assert config_module.MODEL_TYPE == "topk_embed"
    assert TopkEmbedConfig.model_type == MODEL_TYPE
    assert TopkEmbedConfig.dim == 1024  # the remote default; the checkpoint's
    # own config.json overrides it with 2048
    assert TopkEmbedConfig.output_dim is None
    assert TopkEmbedConfig.normalize is True
    assert TopkEmbedConfig.query_template == "Query: "
    assert TopkEmbedConfig.document_prompt == "Document: "
    assert TopkEmbedConfig.image_token_budget == 1280
    assert TopkEmbedConfig.scoring_skip_ids is None


def test_registered_config_wins_over_remote_code(auto_config, tmp_path: Path, registered: None) -> None:
    """With the config registered, AutoConfig resolves a local directory whose
    config.json carries the checkpoint's auto_map WITHOUT executing remote
    code — even with trust_remote_code=True: transformers' explicit-local-code
    path wins (auto_factory skips the remote branch when the registered
    class's __module__ is not transformers.*).  The remote module the auto_map
    names does not exist here, so any attempt to execute it fails this test
    loudly instead of silently using remote code."""
    tiny = {
        "architectures": ["TopkEmbedModel"],
        "model_type": MODEL_TYPE,
        "auto_map": {"AutoConfig": "modeling_topk_embed.TopkEmbedConfig"},
        "dim": 8,
        "text_config": tiny_text_config(),
        "vision_config": TINY_VISION_CONFIG,
    }
    config = auto_config.from_pretrained(write_tiny_checkpoint(tmp_path, tiny), trust_remote_code=True)
    assert type(config) is TopkEmbedConfig, (
        "transformers executed the checkpoint's remote config code instead of the plugin's registered class"
    )


def test_registered_config_parses_the_checkpoint_knobs(auto_config, tmp_path: Path, registered: None) -> None:
    """The registered class parses the checkpoint's retrieval knobs and applies
    the remote __post_init__ semantics: output_dim defaults to dim,
    text_config.use_cache is forced off, the knobs carry the config's values."""
    tiny = {"model_type": MODEL_TYPE, "dim": 16, "text_config": tiny_text_config(), "vision_config": TINY_VISION_CONFIG}
    config = auto_config.from_pretrained(write_tiny_checkpoint(tmp_path, tiny))
    assert type(config) is TopkEmbedConfig
    assert config.output_dim == 16
    assert config.dim == 16
    assert config.text_config.use_cache is False
    assert config.normalize is True
    assert config.query_template == "Query: "


def test_registered_config_rejects_mrl_wider_than_dim(auto_config, tmp_path: Path, registered: None) -> None:
    """The remote config's MRL bound (output_dim in 1..dim) is enforced
    (modeling_topk_embed.py:27-28)."""
    import pytest

    tiny = {
        "model_type": MODEL_TYPE,
        "dim": 8,
        "output_dim": 9,
        "text_config": tiny_text_config(),
        "vision_config": TINY_VISION_CONFIG,
    }
    with pytest.raises(ValueError, match="output_dim"):
        auto_config.from_pretrained(write_tiny_checkpoint(tmp_path, tiny))
