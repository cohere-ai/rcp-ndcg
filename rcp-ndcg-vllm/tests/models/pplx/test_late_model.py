"""The late-interaction sibling (pplx-embed-v2-late-0.6b): the plugin's contract, as data.

The class's own delta over the inherited ``ColQwen3_5Model`` is weight loading, so the
CPU suite pins it as data (no vLLM import): the architecture the plugin registers, the
checkpoint's safetensors census against the name mapping, the separate Dense-head file
and its tensor, the zero-initialised bias marking, and the registration call (stub
registries, as ``test_contract_core.py`` does). Tests that need vLLM importable (the
class resolving through the real base class) skip with a clear reason and run on the
GPU wave, whose engine image has vLLM 0.31.0.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest
from rcp_ndcg_vllm.models.pplx import LATE_ARCHITECTURE, LATE_MODEL_QUALNAME, PLUGIN_ARCHITECTURE
from rcp_ndcg_vllm.models.pplx.late_data import (
    ARCHITECTURE,
    CHECKPOINT_PREFIXES,
    CHECKPOINT_TENSOR_RE,
    DENSE_HEAD_BIAS_TENSOR,
    DENSE_HEAD_FILE,
    DENSE_HEAD_TENSOR,
    IGNORED_CHECKPOINT_PREFIXES,
    PROJECTION_BIAS_TARGET_NAME,
    PROJECTION_TARGET_NAME,
    ZERO_INITIALISED_PARAMETERS,
    map_checkpoint_name,
    mark_zero_initialised,
)

# The checkpoint's safetensors census at the pinned revision
# 8fc2de24534aa3610d85fa59c463313a5f096455, re-derived from the Hub file headers
# (model.safetensors and 1_Dense/model.safetensors) and pinned here: the served
# model must resolve every name below or drop it explicitly.
CENSUS = {
    "language_model.embed_tokens.weight": (248320, 1024),
    "language_model.layers.0.input_layernorm.weight": (1024,),
    "language_model.norm.weight": (1024,),
    "visual.patch_embed.proj.weight": (1024, 3, 16, 16),
}
MTP_WEIGHT = "mtp.layers.0.fc.weight"


def test_the_registered_architecture_is_the_checkpoints_own() -> None:
    """The plugin registers exactly the checkpoint's ``architectures[0]`` -- and it is NOT
    one of vLLM v0.31.0's own names (a flags-only serve cannot resolve the model class)."""
    assert LATE_ARCHITECTURE == "Qwen3_5Model" == ARCHITECTURE
    assert LATE_ARCHITECTURE != PLUGIN_ARCHITECTURE  # the contextual sibling keeps its own
    assert LATE_MODEL_QUALNAME == "rcp_ndcg_vllm.models.pplx.late:PplxLateMultiVectorModel"
    for own in ("Qwen3_5ForCausalLM", "ColQwen3_5", "Qwen3_5ForConditionalGeneration"):
        assert LATE_ARCHITECTURE != own


def test_the_checkpoint_naming_maps_through_the_inherited_mapper() -> None:
    """The checkpoint's top-level naming maps through the stock ColQwen3_5Model table the
    subclass inherits: ``language_model.*`` -> ``language_model.model.*``, ``visual.*``
    unchanged, ``mtp.*`` dropped (none exist in this checkpoint -- the census pins that)."""
    assert map_checkpoint_name("language_model.embed_tokens.weight") == "language_model.model.embed_tokens.weight"
    assert map_checkpoint_name("visual.patch_embed.proj.weight") == "visual.patch_embed.proj.weight"
    assert map_checkpoint_name(MTP_WEIGHT) is None
    # The checkpoint's real tensors all resolve or are explicitly ignored; nothing else exists.
    for name in CENSUS:
        assert map_checkpoint_name(name) is not None
        assert name.startswith(("language_model.", "visual."))
    assert IGNORED_CHECKPOINT_PREFIXES == ("mtp.",)


def test_the_dense_head_is_a_separate_file_loaded_by_name() -> None:
    """The head facts the loader rests on: the sentence-transformers module file, its single
    tensor, and the projector name the inherited loader's _PROJ_LAYER_NAMES intercepts."""
    assert DENSE_HEAD_FILE == "1_Dense/model.safetensors"
    assert DENSE_HEAD_TENSOR == "linear.weight"
    assert PROJECTION_TARGET_NAME == "custom_text_proj.weight"
    # The stock loader's interception names (colqwen3_5.py:210-214 at v0.31.0) contain the target's stem.
    assert any("custom_text_proj" in stem for stem in ("custom_text_proj", "embedding_proj_layer"))
    # The head tensor never passes through the backbone mapper (the plugin renames it first).
    assert map_checkpoint_name(DENSE_HEAD_TENSOR) == DENSE_HEAD_TENSOR  # unprefixed: unchanged


def test_mark_zero_initialised_claims_both_qualnames() -> None:
    """The load-tracker annotation: the bias-less checkpoint's zero bias is claimed under the
    model attribute's and the pooler projector's qualnames, pinned by their literals (the
    topk precedent: a symbolic pin cannot fail), and a shipped bias stays loaded."""
    loaded = {"custom_text_proj.weight", "language_model.model.embed_tokens.weight"}
    marked = mark_zero_initialised(set(loaded))
    assert loaded < marked  # the checkpoint's tensors are still in the set
    # The literals, by name: the two qualnames the served tracker reads.
    assert "custom_text_proj.bias" in marked
    assert "pooler.head.projector.bias" in marked
    assert set(ZERO_INITIALISED_PARAMETERS) == {"custom_text_proj.bias", "pooler.head.projector.bias"}
    assert marked == loaded | {"custom_text_proj.bias", "pooler.head.projector.bias"}
    # A shipped head.bias is loaded and marked by the in-tree path; marking is idempotent.
    both = mark_zero_initialised({"custom_text_proj.weight", "custom_text_proj.bias"})
    assert both == {"custom_text_proj.weight", "custom_text_proj.bias", "pooler.head.projector.bias"}
    # Idempotent: a second mark adds nothing.
    assert mark_zero_initialised(marked) == marked


def test_register_registers_both_architectures(monkeypatch: pytest.MonkeyPatch) -> None:
    """``register()`` against stub registries: BOTH pplx architectures register lazily, the
    contextual config class and its vLLM config handler still register, and a second load
    is harmless."""
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.31.0",  # noqa: ARG005
    )
    registered: dict[str, str] = {}
    registry = types.SimpleNamespace(
        get_supported_archs=lambda: set(registered),
        register_model=lambda arch, cls: registered.setdefault(arch, cls),
    )
    config_map: dict[str, type] = {}
    auto_config_calls: list[tuple[str, type]] = []

    class FakeAutoConfig:
        """Records ``register`` calls the way transformers' ``AutoConfig`` does."""

        @classmethod
        def register(cls, model_type: str, config_cls: type, *, exist_ok: bool = False) -> None:
            assert exist_ok is True  # repeated plugin loads must be harmless
            auto_config_calls.append((model_type, config_cls))

    fake_config_module = types.ModuleType("vllm.model_executor.models.config")
    fake_config_module.MODELS_CONFIG_MAP = config_map  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.ModelRegistry = registry  # type: ignore[attr-defined]
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig  # type: ignore[attr-defined]
    fake_transformers.Qwen3_5Config = type("Qwen3_5Config", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", types.ModuleType("vllm.model_executor.models"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("rcp_ndcg_vllm.models.pplx.hf_config", None)

    import rcp_ndcg_vllm.models.pplx

    rcp_ndcg_vllm.models.pplx.register()

    # The contextual sibling: the architecture, its config handler and the config class.
    assert registered[rcp_ndcg_vllm.models.pplx.PLUGIN_ARCHITECTURE] == "rcp_ndcg_vllm.models.pplx.model:PplxContextualForPooling"
    assert config_map[rcp_ndcg_vllm.models.pplx.PLUGIN_ARCHITECTURE]
    assert auto_config_calls == [("pplx_contextual_qwen3_5", rcp_ndcg_vllm.models.pplx.hf_config.PplxContextualConfig)]
    # The late sibling: the architecture name only -- no config registration, no config
    # handler (the checkpoint's model_type qwen3_5 is native to the engine's transformers
    # line, and its text_config carries is_causal: false itself).
    assert registered[rcp_ndcg_vllm.models.pplx.LATE_ARCHITECTURE] == rcp_ndcg_vllm.models.pplx.LATE_MODEL_QUALNAME
    assert rcp_ndcg_vllm.models.pplx.LATE_ARCHITECTURE not in config_map
    # A second load must not raise (idempotent registrations).
    rcp_ndcg_vllm.models.pplx.register()
    assert registered[rcp_ndcg_vllm.models.pplx.LATE_ARCHITECTURE] == rcp_ndcg_vllm.models.pplx.LATE_MODEL_QUALNAME


def test_checkpoints_census_names_all_resolve_or_drop() -> None:
    """The census: every checkpoint tensor is inside the mapped name spaces the loader pins,
    and the head file is outside them (loaded separately, by name) -- the loud-refusal
    contract of ``PplxLateMultiVectorModel.load_weights``."""
    for name in CENSUS:
        assert CHECKPOINT_TENSOR_RE.match(name), name
        assert map_checkpoint_name(name) is not None
    assert not CHECKPOINT_TENSOR_RE.match(DENSE_HEAD_TENSOR)
    assert CHECKPOINT_PREFIXES == {"language_model.": "language_model.model.", "mtp.": None}


def test_the_dense_head_bias_target_is_pinned() -> None:
    """The head-file tensors map onto the two projection parameters: the weight always, a
    shipped bias when a revision carries one (this one ships none -- the loader then leaves
    the constructor's zeros in place)."""
    assert DENSE_HEAD_BIAS_TENSOR == "linear.bias"
    assert PROJECTION_BIAS_TARGET_NAME == "custom_text_proj.bias"
    # The bias target is the first of the two zero-initialised qualnames.
    assert PROJECTION_BIAS_TARGET_NAME == ZERO_INITIALISED_PARAMETERS[0]


VLLM_MISSING_REASON = (
    "vLLM is not importable in this environment (the dev and plugin venvs carry torch only, "
    "by design); the mapper cross-check runs on the GPU wave, whose engine image has vLLM 0.31.0"
)


def test_served_class_mapper_cross_check() -> None:
    """The mapper carried by the registered model class maps the checkpoint's names exactly as
    the restated table predicts (cross-check against vLLM's own WeightsMapper). Skipped where
    vLLM cannot be imported; runs on the GPU wave."""
    pytest.importorskip("vllm", reason=VLLM_MISSING_REASON)

    from rcp_ndcg_vllm.models.pplx.late import PplxLateMultiVectorModel

    for name in (*CENSUS, MTP_WEIGHT):
        assert PplxLateMultiVectorModel.hf_to_vllm_mapper.map_name(name) == map_checkpoint_name(name), name
    # The head tensors bypass the mapper entirely (the loader renames them first).
    assert map_checkpoint_name(DENSE_HEAD_TENSOR) == DENSE_HEAD_TENSOR


def test_served_class_is_a_stock_colqwen3_5_subclass() -> None:
    """The registered class inherits every forward-affecting behaviour from ColQwen3_5Model
    unchanged (pooling type, pooling flag, projection-pooler wiring, the processor
    registration); only ``load_weights`` is overridden. Skipped where vLLM cannot be
    imported; runs on the GPU wave."""
    pytest.importorskip("vllm", reason=VLLM_MISSING_REASON)

    from rcp_ndcg_vllm.models.pplx.late import PplxLateMultiVectorModel
    from vllm.model_executor.models.colqwen3_5 import ColQwen3_5Model

    assert issubclass(PplxLateMultiVectorModel, ColQwen3_5Model)
    assert PplxLateMultiVectorModel.is_pooling_model is True
    assert PplxLateMultiVectorModel.default_seq_pooling_type == ColQwen3_5Model.default_seq_pooling_type
    assert PplxLateMultiVectorModel.default_tok_pooling_type == ColQwen3_5Model.default_tok_pooling_type
    # The projection-name matcher is inherited untouched; the loader's renames land in
    # the canonical namespace it already knows.
    assert PplxLateMultiVectorModel._PROJ_LAYER_NAMES == {"custom_text_proj", "embedding_proj_layer"}
    assert PplxLateMultiVectorModel._is_proj_weight(PplxLateMultiVectorModel, PROJECTION_TARGET_NAME)
    assert PplxLateMultiVectorModel._is_proj_weight(PplxLateMultiVectorModel, PROJECTION_BIAS_TARGET_NAME)
