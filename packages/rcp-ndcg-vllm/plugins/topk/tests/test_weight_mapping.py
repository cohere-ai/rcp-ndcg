"""The checkpoint-to-vLLM weight-name mapping.

The mapping is the plugin's entire delta from the native ColQwen3_5Model, so
the tests pin it against the checkpoint's real tensor list (fixture from the
safetensors header, 618 names).  Where vLLM is importable, the test also
cross-checks the mapper the served model class actually carries against the
restated table, so a vLLM release that changes the upstream mappings fails
here instead of silently corrupting weight loading.
"""

from __future__ import annotations

import pytest
from conftest import CENSUS_PREFIX_COUNTS  # noqa: I001 - conftest prepends src/
from rcp_ndcg_vllm_topk import weights

VLLM_MISSING_REASON = (
    "vLLM is not importable on this CPU environment; the served model "
    "class's mapper is cross-checked on the engine image (GPU wave T0)"
)


def _prefix(name: str, depth: int = 2) -> str:
    return ".".join(name.split(".")[:depth])


def test_census_prefix_counts(
    checkpoint_tensor_names: list[str],
) -> None:
    """The fixture is the checkpoint's census: one projection tensor and the
    two backbone trees (safetensors header at the pinned revision)."""
    counts: dict[str, int] = {}
    for name in checkpoint_tensor_names:
        if name == "head.weight":
            counts["head.weight"] = counts.get("head.weight", 0) + 1
        else:
            key = _prefix(name)
            counts[key] = counts.get(key, 0) + 1
    assert counts == CENSUS_PREFIX_COUNTS


def test_plugin_delta_is_only_the_projection_rename(
    checkpoint_tensor_names: list[str],
) -> None:
    """The plugin's own table entry is the projection rename alone: the rest of
    the table restates the upstream Qwen3-VL mapping, and every checkpoint name
    lands where the served model expects it."""
    assert weights.PROJECTION_SOURCE_PREFIX == "head."
    assert weights.PROJECTION_TARGET_PREFIX == "custom_text_proj."
    upstream = {
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
        "mtp.": None,
    }
    assert {k: v for k, v in weights.CHECKPOINT_TO_VLLM_PREFIXES.items() if k != "head."} == upstream
    assert weights.map_checkpoint_name("head.weight") == "custom_text_proj.weight"


def test_mapped_names_follow_the_upstream_shapes(
    checkpoint_tensor_names: list[str],
) -> None:
    """Every checkpoint name maps exactly as the served model's module tree
    expects: the Qwen3-VL convention renames, the projection rename, and
    nothing else moves."""
    for name in checkpoint_tensor_names:
        mapped = weights.map_checkpoint_name(name)
        assert mapped is not None
        if name.startswith("model.language_model."):
            assert mapped == "language_model.model." + name.removeprefix("model.language_model."), name
        elif name.startswith("model.visual."):
            assert mapped == "visual." + name.removeprefix("model.visual."), name
        elif name == "head.weight":
            assert mapped == "custom_text_proj.weight"
        else:  # nothing else in this checkpoint; pass through, never drop
            assert mapped == name, name


def test_mapped_names_are_vllm_module_paths(
    checkpoint_tensor_names: list[str],
) -> None:
    """After mapping, every name lives under one of the module paths the
    served model exposes (custom_text_proj, language_model.model, visual)."""
    roots = {"custom_text_proj.", "language_model.model.", "visual."}
    for name in checkpoint_tensor_names:
        mapped = weights.map_checkpoint_name(name)
        assert mapped is not None
        assert mapped.startswith(tuple(roots)), (name, mapped)


def test_ignored_prefix_is_dropped() -> None:
    """A tensor under an ignored prefix (none in this checkpoint) is dropped,
    matching the upstream mapper's ``None`` entries."""
    assert weights.map_checkpoint_name("mtp.layers.0.fc.weight") is None
    assert "mtp." in weights.IGNORED_CHECKPOINT_PREFIXES


def test_unmapped_names_pass_through() -> None:
    """A name outside every table entry is kept unchanged (future-proofing:
    a new prefix must fail the loader loudly, not vanish)."""
    assert weights.map_checkpoint_name("something.else.weight") == ("something.else.weight")


def test_mapped_names_are_unique(
    checkpoint_tensor_names: list[str],
) -> None:
    """The mapping is injective on the checkpoint (no two tensors land on the
    same vLLM name; a collision would overwrite a loaded weight)."""
    mapped = [mapped for name in checkpoint_tensor_names if (mapped := weights.map_checkpoint_name(name)) is not None]
    assert len(mapped) == len(set(mapped)) == 618


def test_served_class_mapper_cross_check(
    checkpoint_tensor_names: list[str],
) -> None:
    """The mapper carried by the registered model class maps all 618 checkpoint
    names exactly as the restated table predicts (cross-check against vLLM's
    own WeightsMapper).  Skipped where vLLM cannot be imported."""
    pytest.importorskip("vllm", reason=VLLM_MISSING_REASON)

    from rcp_ndcg_vllm_topk.model import TopkEmbedModel

    for name in checkpoint_tensor_names:
        assert TopkEmbedModel.hf_to_vllm_mapper.map_name(name) == (weights.map_checkpoint_name(name)), name


def test_served_class_is_a_stock_colqwen3_5_subclass() -> None:
    """The registered class inherits every forward-affecting behaviour from
    ColQwen3_5Model unchanged (pooling type, pooling flag, projection-pooler
    wiring); only ``hf_to_vllm_mapper`` is overridden."""
    pytest.importorskip("vllm", reason=VLLM_MISSING_REASON)

    from rcp_ndcg_vllm_topk.model import TopkEmbedModel
    from vllm.model_executor.models.colqwen3_5 import ColQwen3_5Model

    assert issubclass(TopkEmbedModel, ColQwen3_5Model)
    assert TopkEmbedModel.is_pooling_model is True
    assert TopkEmbedModel.default_seq_pooling_type == (ColQwen3_5Model.default_seq_pooling_type)
    assert TopkEmbedModel.default_tok_pooling_type == (ColQwen3_5Model.default_tok_pooling_type)
    # The projection-name matcher is inherited untouched; the mapping puts the
    # checkpoint name into the canonical namespace the matcher already knows.
    # (`self` only carries the class-level `_PROJ_LAYER_NAMES` lookup, so
    # passing the class keeps the check construction-free — the instance
    # method cannot be called unbound with just the name.)
    assert TopkEmbedModel._PROJ_LAYER_NAMES == {
        "custom_text_proj",
        "embedding_proj_layer",
    }
    assert TopkEmbedModel._is_proj_weight(TopkEmbedModel, "custom_text_proj.weight")
    assert TopkEmbedModel._is_proj_weight(TopkEmbedModel, "custom_text_proj.bias")
    # The checkpoint's original projection name matches nothing (it flows to
    # the AutoWeightsLoader, where the mapper renames it):
    assert not TopkEmbedModel._is_proj_weight(TopkEmbedModel, "head.weight")
