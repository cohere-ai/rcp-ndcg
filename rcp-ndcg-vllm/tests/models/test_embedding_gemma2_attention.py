"""The plugin's ``TORCH_SDPA`` encoder-only backend (the embeddinggemma-2 recipe's engine backend).

E2 round 10: the digest-pinned nightly refuses every backend the recipe tried and dies inside
FLEX_ATTENTION's own ``torch.compile``/CUDAGraph interaction, so the plugin registers its own class
under ``Backend.TORCH_SDPA`` (the member the engine's own error names) and the recipe pins
``--attention-backend TORCH_SDPA``. The backend imports vLLM and torch, which this CPU suite does not
carry: the tests install a fake ``vllm`` surface shaped like the pinned nightly's interface
(``vllm/v1/attention/backend.py``, ``vllm/v1/attention/backends/registry.py``) and use the real torch
for the numeric checks -- the packed-request boundaries, the grouped-query attention and the sliding
window. The real interface is exercised against the image's own vLLM by the lane's interface read; the
GPU wave is the served confirmation.
"""

from __future__ import annotations

import sys
import types
from enum import StrEnum
from typing import Any

import pytest
import torch
from rcp_ndcg_vllm.patches import PATCH_NAMES, PATCHES_ENV, apply_opted_in_patches

BACKEND_MODULE = "rcp_ndcg_vllm.models.embedding_gemma2.attention"
BACKEND_QUALNAME = f"{BACKEND_MODULE}.TorchSdpaAttentionBackend"
PATCH_NAME = "torch-sdpa-attention"


class _FakeAttentionType(StrEnum):
    DECODER = "decoder"
    ENCODER_ONLY = "encoder_only"


class _FakeAttentionBackend:
    """The slice of the interface the backend class overrides (``vllm/v1/attention/backend.py``).

    ``supports_head_size`` mirrors the real base: an empty declared list means no cap.
    """

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported = cls.get_supported_head_sizes()
        return (not supported) or head_size in supported


class _FakeAttentionMetadata:
    """The marker base the metadata dataclass extends."""


class _FakeAttentionMetadataBuilder[M]:
    """The builder base: the constructor stores what the engine hands it (the real one does more)."""

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device


class _FakeAttentionImpl:
    """The impl base (the real one's ``__new__`` reads distributed state; the tests construct eagerly)."""


class _FakeCommonAttentionMetadata:
    """The engine's per-batch metadata (the builder copies three fields out of it)."""

    def __init__(
        self,
        *,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        num_actual_tokens: int,
    ) -> None:
        self.query_start_loc = query_start_loc
        self.num_reqs = num_reqs
        self.num_actual_tokens = num_actual_tokens


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install the fake ``vllm`` interface, import the backend module through it, and restore.

    The registration recorder is what the registration test asserts against: the real
    ``register_backend(backend, class_path)`` only stores the path in ``_ATTN_OVERRIDES`` (the lane's
    image read exercises the real one).
    """
    registered: list[tuple[Any, str]] = []

    class _FakeBackendEnum(StrEnum):
        TORCH_SDPA = ""

    def register_backend(backend: Any, class_path: str) -> None:
        registered.append((backend, class_path))

    backend_module = types.ModuleType("vllm.v1.attention.backend")
    backend_module.AttentionBackend = _FakeAttentionBackend
    backend_module.AttentionMetadata = _FakeAttentionMetadata
    backend_module.AttentionMetadataBuilder = _FakeAttentionMetadataBuilder
    backend_module.AttentionImpl = _FakeAttentionImpl
    backend_module.AttentionType = _FakeAttentionType
    backend_module.CommonAttentionMetadata = _FakeCommonAttentionMetadata
    registry_module = types.ModuleType("vllm.v1.attention.backends.registry")
    registry_module.AttentionBackendEnum = _FakeBackendEnum
    registry_module.register_backend = register_backend

    packages = {
        "vllm": {},
        "vllm.v1": {},
        "vllm.v1.attention": {},
        "vllm.v1.attention.backends": {},
    }
    installed: dict[str, types.ModuleType] = {}
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = []
        installed[name] = module
    installed["vllm.v1.attention.backend"] = backend_module
    installed["vllm.v1.attention.backends.registry"] = registry_module
    for name, module in installed.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, BACKEND_MODULE, raising=False)

    import importlib

    module = importlib.import_module(BACKEND_MODULE)
    return {"module": module, "registered": registered, "enum": _FakeBackendEnum}


def test_the_backend_imports_and_declares_the_encoder_only_contract(fake_vllm: dict[str, Any]) -> None:
    """The class imports against the engine's interface and clears exactly the model's attention shape:
    no head-size cap (256 and 512), encoder-only attention, a sliding window and non-causal attention."""
    backend = fake_vllm["module"].TorchSdpaAttentionBackend
    assert issubclass(backend, _FakeAttentionBackend)
    assert backend.get_name() == "TORCH_SDPA"
    assert backend.get_supported_head_sizes() == [], "no cap: the full layers are head_dim 512"
    assert backend.supports_head_size(256) and backend.supports_head_size(512)
    assert backend.supports_attn_type(_FakeAttentionType.ENCODER_ONLY)
    assert not backend.supports_attn_type(_FakeAttentionType.DECODER)
    assert backend.supports_sliding_window() and backend.supports_non_causal()
    assert backend.forward_includes_kv_cache_update is False
    assert torch.bfloat16 in backend.supported_dtypes
    assert backend.get_impl_cls().__name__ == "TorchSdpaAttentionImpl"
    assert backend.get_builder_cls().__name__ == "TorchSdpaAttentionMetadataBuilder"


def test_the_registration_overrides_the_torch_sdpa_enum_member(fake_vllm: dict[str, Any]) -> None:
    """``register_torch_sdpa_backend`` hands the engine the class path the enum member resolves to:
    the plugin's own class, named from its module (the one home of the path)."""
    fake_vllm["module"].register_torch_sdpa_backend()
    assert fake_vllm["registered"] == [(fake_vllm["enum"].TORCH_SDPA, BACKEND_QUALNAME)]


def test_the_patch_is_declared_and_registers_the_backend(fake_vllm: dict[str, Any]) -> None:
    """The opt-in name is shipped, and applying it (the engine's ``RCP_NDCG_VLLM_PATCHES`` path) reaches
    the registration; an engine that did not opt in registers nothing."""
    assert PATCH_NAME in PATCH_NAMES
    assert apply_opted_in_patches(environ={PATCHES_ENV: PATCH_NAME}) == (PATCH_NAME,)
    assert fake_vllm["registered"] == [(fake_vllm["enum"].TORCH_SDPA, BACKEND_QUALNAME)]


def test_the_builder_copies_the_request_boundaries(fake_vllm: dict[str, Any]) -> None:
    """The metadata is the batch's packed-request boundaries, taken from the engine's common metadata."""
    module = fake_vllm["module"]
    builder = module.TorchSdpaAttentionMetadataBuilder(None, ["layer"], None, torch.device("cpu"))
    common = _FakeCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
    )
    metadata = builder.build(0, common)
    assert metadata.query_start_loc.tolist() == [0, 3, 5]
    assert metadata.num_reqs == 2 and metadata.num_actual_tokens == 5


def _impl(module: Any, num_heads: int = 4, num_kv_heads: int = 2, sliding_window: int | None = None) -> Any:
    return module.TorchSdpaAttentionImpl(
        num_heads,
        8,
        1.0,
        num_kv_heads,
        None,
        sliding_window,
        "auto",
        None,
        _FakeAttentionType.ENCODER_ONLY,
    )


def _packed(module: Any, sliding_window: int | None = None) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A two-request packed batch (3 then 2 tokens, 4 query heads / 2 kv heads, head_dim 8)."""
    torch.manual_seed(0)
    query = torch.randn(5, 4, 8)
    key = torch.randn(5, 2, 8)
    value = torch.randn(5, 2, 8)
    metadata = module.TorchSdpaAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32), num_reqs=2, num_actual_tokens=5
    )
    return _impl(module, sliding_window=sliding_window), metadata, query, key, value


def _manual(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int | None = None,
) -> torch.Tensor:
    """The reference: per request, non-causal SDPA at scale 1.0, GQA by ``repeat_interleave``."""
    expected = torch.empty_like(query)
    for start, end in ((0, 3), (3, 5)):
        q = query[start:end].transpose(0, 1)
        k = key[start:end].transpose(0, 1).repeat_interleave(2, dim=0)
        v = value[start:end].transpose(0, 1).repeat_interleave(2, dim=0)
        scores = q @ k.transpose(-2, -1)
        if window is not None:
            positions = torch.arange(end - start)
            keep = (positions[:, None] - positions[None, :]).abs() <= window
            scores = scores.masked_fill(~keep, float("-inf"))
        expected[start:end] = (torch.softmax(scores, dim=-1) @ v).transpose(0, 1)
    return expected


def test_the_forward_matches_the_manual_reference(fake_vllm: dict[str, Any]) -> None:
    """One SDPA call per request over the packed q/k/v: the output equals the manual non-causal
    grouped-query reference at the model's scale (1.0)."""
    module = fake_vllm["module"]
    impl, metadata, query, key, value = _packed(module)
    output = torch.empty(5, 4, 8)
    impl.forward(None, query, key, value, torch.empty(0), metadata, output)
    assert torch.allclose(output, _manual(query, key, value), atol=1e-6)


def test_the_requests_do_not_attend_each_other(fake_vllm: dict[str, Any]) -> None:
    """The batch is packed along the token dimension: rewriting request 0's tokens must not move
    request 1's output (the boundary split is the only thing keeping them apart)."""
    module = fake_vllm["module"]
    impl, metadata, query, key, value = _packed(module)
    output = torch.empty(5, 4, 8)
    impl.forward(None, query, key, value, torch.empty(0), metadata, output)
    rewritten = query.clone()
    rewritten[0] += 100.0
    other = torch.empty(5, 4, 8)
    impl.forward(None, rewritten, key, value, torch.empty(0), metadata, other)
    assert torch.allclose(other[3:], output[3:], atol=1e-6)


def test_the_sliding_window_masks_exactly_the_declared_band(fake_vllm: dict[str, Any]) -> None:
    """vLLM's window W masks ``|i - j| <= W - 1`` (the vendored model passes the checkpoint's 512 + 1),
    so a window of 4 keeps distance <= 3 and masks the rest -- exactly the manual band."""
    module = fake_vllm["module"]
    impl, metadata, query, key, value = _packed(module, sliding_window=4)
    output = torch.empty(5, 4, 8)
    impl.forward(None, query, key, value, torch.empty(0), metadata, output)
    assert torch.allclose(output, _manual(query, key, value, window=3), atol=1e-6)


def test_the_forward_fills_the_profiling_output(fake_vllm: dict[str, Any]) -> None:
    """A profiling run has no metadata; the impl zeroes the output (the values are unused)."""
    module = fake_vllm["module"]
    impl, _metadata, query, key, value = _packed(module)
    output = torch.full((5, 4, 8), 7.0)
    impl.forward(None, query, key, value, torch.empty(0), None, output)
    assert torch.count_nonzero(output) == 0
