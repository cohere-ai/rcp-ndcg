"""The plugin's contract in pure torch, proven without vLLM.

The reference side of the equivalence is an **independent oracle**: it re-derives the
reference implementation's pools from the chunk *strings* (char spans, as the remote
code's ``prepare_inputs``/``_pool`` build them, per the r-pplx research draft), while the
plugin's :func:`rcp_ndcg_vllm.models.pplx.pooling_core.pool_sequence` sees only the token ids that
cross the wire. The two must agree token for token — that agreement is the serving
contract. The tiny "backbone" that produces the hidden states is a random embedding plus
a small block, shared by both sides, so the comparison isolates the plugin's pooling and
head on a randomly initialised configuration of the model's architecture.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest
import torch

# Imported after the pytest block: conftest.py has already put this package's
# src/ tree on sys.path, so no in-file statement precedes these imports.
from rcp_ndcg_vllm.models.pplx import PLUGIN_ARCHITECTURE, PLUGIN_NAME
from rcp_ndcg_vllm.models.pplx.pooling_core import (
    BOUNDARY_TOKEN_ID,
    DOCUMENT_PREFIX_TOKEN_IDS,
    QUERY_PREFIX_TOKEN_ID,
    PplxInt8Projection,
    pool_sequence,
)
from rcp_ndcg_vllm.models.version_guard import (
    SUPPORTED_VLLM_MAX,
    SUPPORTED_VLLM_MIN,
    checked_vllm_version,
    require_vllm_version,
)

HIDDEN = 8
EMBED = 4
VOCAB = 96

torch.manual_seed(20261006)


# ---------------------------------------------------------------------------
# The tiny configuration of the model's architecture, and the wire contract.
# ---------------------------------------------------------------------------
class TinyTokenizer:
    """A character tokenizer that keeps the checkpoint's added-token contract.

    Every chunk character is one token; ``<|chunk_sep|>`` is the boundary id; the
    ``[D] `` prefix renders as the two literal prefix ids and ``[Q] `` as the one
    prefix id — the measured behaviour of the checkpoint's tokenizer that forces the
    token-id contract.
    """

    def __init__(self) -> None:
        self._next = 1

    def char_id(self, char: str) -> int:
        return 10 + (ord(char) % 80)

    def tokenize_document(self, chunks: list[str]) -> list[int]:
        ids = list(DOCUMENT_PREFIX_TOKEN_IDS)
        for index, chunk in enumerate(chunks):
            if index:
                ids.append(BOUNDARY_TOKEN_ID)
            ids.extend(self.char_id(c) for c in chunk)
        return ids

    def tokenize_query(self, query: str) -> list[int]:
        return [QUERY_PREFIX_TOKEN_ID, *(self.char_id(c) for c in query)]


class TinyBackbone(torch.nn.Module):
    """A stand-in backbone: random embedding plus a small block, shared by both sides."""

    def __init__(self, tokenizer: TinyTokenizer) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.block = torch.nn.Linear(HIDDEN, HIDDEN)
        with torch.no_grad():
            # The contract ids must produce real hidden states; anything above VOCAB
            # would index out of range in the stand-in.
            self.embed.weight.uniform_(-1.0, 1.0)

    def forward(self, ids: list[int]) -> torch.Tensor:
        id_tensor = torch.tensor([self._stand_in(i) for i in ids], dtype=torch.int64)
        return torch.tanh(self.block(self.embed(id_tensor)))

    @staticmethod
    def _stand_in(token_id: int) -> int:
        """Fold any token id into the stand-in vocab (the contract ids included)."""
        return token_id % VOCAB


def reference_pool(tokenizer: TinyTokenizer, backbone: TinyBackbone, chunks: list[str]) -> torch.Tensor:
    """The reference implementation's document pooling, from the chunk strings.

    Char spans per chunk (cursor over the joined text, prefix and markers excluded),
    converted to token spans through the char tokenizer's offsets — the reference
    draft's ``chunk_token_spans`` logic — then span means, then the int8 head.
    """
    prefix_chars = len("[D] ")
    cursor = prefix_chars
    char_spans = []
    for index, chunk in enumerate(chunks):
        if index:
            cursor += len("<|chunk_sep|>")
        char_spans.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)

    ids = tokenizer.tokenize_document(chunks)
    hidden = backbone(ids).to(torch.float32)
    # Character-level tokenizer, so each non-added token covers exactly one character of
    # the joined text — except the boundary marker, which is one token spanning its 13
    # characters; prefix and markers are excluded from the chunk spans either way.
    marker_chars = len("<|chunk_sep|>")
    rows = []
    for start, end in char_spans:
        token_positions = []
        offset = prefix_chars
        for token_index, token_id in enumerate(ids[2:], start=2):
            if token_id == BOUNDARY_TOKEN_ID:
                offset += marker_chars
                continue
            if start <= offset < end:
                token_positions.append(token_index)
            offset += 1
        if token_positions:
            rows.append(hidden[token_positions].mean(0))
        else:
            rows.append(torch.zeros(HIDDEN, dtype=torch.float32))
    return torch.stack(rows)


def reference_query_pool(tokenizer: TinyTokenizer, backbone: TinyBackbone, query: str) -> torch.Tensor:
    """The reference implementation's query pooling: mean over everything, prefix in."""
    ids = tokenizer.tokenize_query(query)
    hidden = backbone(ids).to(torch.float32)
    return hidden.mean(0, keepdim=True)


class ReferenceHead(torch.nn.Module):
    """The reference head, written out from the model card, sharing the plugin's weights."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        projected = pooled @ self.weight.T
        return torch.round(torch.tanh(projected) * 127.0).clamp(-128.0, 127.0)


# ---------------------------------------------------------------------------
# Equivalence: plugin pooling + head vs the reference oracle.
# ---------------------------------------------------------------------------
def test_document_chunks_match_the_reference_char_span_pools() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    chunks = ["hello world", "second chunk here", "a third one"]
    ids = tokenizer.tokenize_document(chunks)

    hidden = backbone(ids)
    plugin_rows = pool_sequence(hidden, torch.tensor(ids, dtype=torch.int64))
    reference_rows = reference_pool(tokenizer, backbone, chunks)

    head_weight = torch.randn(EMBED, HIDDEN, dtype=torch.float32)
    plugin_head = PplxInt8Projection(HIDDEN, EMBED, dtype=torch.float32)
    with torch.no_grad():
        plugin_head.linear.weight.copy_(head_weight)
    reference_head = ReferenceHead(head_weight)

    assert plugin_rows.shape == reference_rows.shape == (len(chunks), HIDDEN)
    assert torch.allclose(plugin_rows, reference_rows, atol=1e-6), (
        "the plugin's segmentation from token ids disagrees with the reference's char-span pooling"
    )
    plugin_embedded = plugin_head(plugin_rows)
    reference_embedded = reference_head(reference_rows)
    assert torch.allclose(plugin_embedded, reference_embedded, atol=1e-5), (
        "the plugin's int8 head disagrees with the reference head on shared weights"
    )
    assert plugin_embedded.abs().max() <= 127.0
    assert torch.equal(plugin_embedded, torch.round(plugin_embedded))


def test_single_chunk_document_has_no_marker_and_matches() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    chunks = ["only one chunk"]
    ids = tokenizer.tokenize_document(chunks)
    assert BOUNDARY_TOKEN_ID not in ids  # the audit's unstated case: no marker to segment on
    plugin_rows = pool_sequence(backbone(ids), torch.tensor(ids, dtype=torch.int64))
    reference_rows = reference_pool(tokenizer, backbone, chunks)
    assert torch.allclose(plugin_rows, reference_rows, atol=1e-6)


def test_empty_chunk_yields_the_reference_zero_vector() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    chunks = ["filled", "", "tail"]
    ids = tokenizer.tokenize_document(chunks)
    plugin_rows = pool_sequence(backbone(ids), torch.tensor(ids, dtype=torch.int64))
    reference_rows = reference_pool(tokenizer, backbone, chunks)
    assert plugin_rows.shape == (3, HIDDEN)
    assert torch.all(plugin_rows[1] == 0), "an empty chunk must pool to the zero vector"
    assert torch.allclose(plugin_rows, reference_rows, atol=1e-6)


def test_query_mean_includes_the_prefix_token() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    query = "what drives breakthroughs"
    ids = tokenizer.tokenize_query(query)
    hidden = backbone(ids)
    plugin_rows = pool_sequence(hidden, torch.tensor(ids, dtype=torch.int64))
    reference = reference_query_pool(tokenizer, backbone, query)
    assert plugin_rows.shape == (1, HIDDEN)
    assert torch.allclose(plugin_rows, reference, atol=1e-6)


def test_role_disambiguation_and_refusals() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    hidden = backbone(tokenizer.tokenize_query("abc"))

    # The query prefix wins over everything else.
    with pytest.raises(ValueError, match="token ids"):
        pool_sequence(hidden, torch.tensor([5, 6, 7]))
    # An incomplete document prefix is a refusal, not a guess.
    with pytest.raises(ValueError, match="token ids"):
        pool_sequence(hidden, torch.tensor([DOCUMENT_PREFIX_TOKEN_IDS[0]]))
    # Misaligned ids never pool a shifted sequence.
    with pytest.raises(ValueError, match="align"):
        pool_sequence(hidden, torch.tensor([QUERY_PREFIX_TOKEN_ID, 1, 2]))


def test_chunk_text_containing_the_marker_splits_a_documented_contract() -> None:
    """The declared limitation: the id wire cannot tell an in-chunk marker from a boundary.

    A chunk whose text is exactly the marker string renders as id 248079 and the pooler
    segments there — the reference's char-span pooling would keep the token inside the
    chunk. The README declares the recipe's chunker must not emit the marker as chunk
    content; this test pins what the pooler does anyway, so a change to the segmentation
    or to the contract note cannot pass silently.
    """
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    # The real tokenizer renders the chunk text "<|chunk_sep|>" as the added id 248079
    # (it is an added token, not a special one), so the wire ids are: prefix, "a",
    # marker (the boundary), marker (the chunk's own content).
    ids = [*DOCUMENT_PREFIX_TOKEN_IDS, tokenizer.char_id("a"), BOUNDARY_TOKEN_ID, BOUNDARY_TOKEN_ID]
    hidden = backbone(ids)
    rows = pool_sequence(hidden, torch.tensor(ids, dtype=torch.int64))
    # The client sent 2 chunks ("a", "<|chunk_sep|>") and the reference would return 2
    # rows (the marker token pooled inside the second chunk); the id wire cannot tell
    # that marker from a boundary, so the pooler sees three segments — "a", then two
    # empty ones (zero vectors). The declared contract: such chunk text is refused by
    # the recipe's chunker, never sent.
    assert rows.shape == (3, HIDDEN)
    assert torch.allclose(rows[0], hidden[2:3].to(torch.float32).mean(0), atol=1e-6)
    assert torch.all(rows[1] == 0)
    assert torch.all(rows[2] == 0)


def test_pooler_warmup_dummy_is_single_span() -> None:
    tokenizer = TinyTokenizer()
    backbone = TinyBackbone(tokenizer)
    hidden = backbone([1, 2, 3])
    rows = pool_sequence(hidden, torch.zeros(3, dtype=torch.int64))
    assert rows.shape == (1, HIDDEN)
    assert torch.allclose(rows[0], hidden.to(torch.float32).mean(0), atol=1e-6)


def test_int8_head_is_created_in_head_dtype() -> None:
    head = PplxInt8Projection(HIDDEN, EMBED, dtype=torch.float32)
    assert head.linear.weight.dtype == torch.float32
    assert head.linear.bias is None
    saturated = head(torch.full((1, HIDDEN), 40.0))
    assert float(saturated.detach().abs().max()) == 127.0  # tanh saturates, clamp keeps the sign


# ---------------------------------------------------------------------------
# Version guard.
# ---------------------------------------------------------------------------
def test_version_guard_refuses_outside_the_validated_range(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in ("0.30.2", "0.32.0", "1.0.0", "0.29"):
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name, _b=bad: _b,  # noqa: ARG005
        )
        with pytest.raises(RuntimeError, match="vLLM >= 0.31"):
            require_vllm_version()


def test_version_guard_accepts_the_pinned_range(monkeypatch: pytest.MonkeyPatch) -> None:
    for good in ("0.31.0", "0.31.5", "0.31.0rc1", "0.31.0.dev123"):
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name, _g=good: _g,  # noqa: ARG005
        )
        assert require_vllm_version() == (0, 31)
        assert checked_vllm_version() == (0, 31)


def test_version_guard_bounds_match_the_brief() -> None:
    assert SUPPORTED_VLLM_MIN == (0, 31)
    assert SUPPORTED_VLLM_MAX == (0, 32)


# ---------------------------------------------------------------------------
# Entry-point registration (stub registry: no vLLM needed).
# ---------------------------------------------------------------------------
def test_distribution_declares_the_general_plugins_entry_point() -> None:
    eps = importlib.metadata.entry_points()
    # The distribution may not be installed in this environment; when it is (the GPU
    # wave installs the wheel), the entry point must be discoverable and named right.
    group = [ep for ep in eps.select(group="vllm.general_plugins") if ep.value == "rcp_ndcg_vllm.models.pplx:register"]
    if not group:
        pytest.skip(
            "rcp-ndcg-vllm-pplx is not installed in this environment; run the "
            "entry-point assertion against the built wheel (the freeze test installs it)"
        )
    assert group[0].name == PLUGIN_NAME


def test_register_registers_model_and_config_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``register_pplx()`` against a stub registry: the exact lazy path and config binding."""
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
    fake_config_module = types.ModuleType("vllm.model_executor.models.config")
    fake_config_module.MODELS_CONFIG_MAP = config_map  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.ModelRegistry = registry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", types.ModuleType("vllm.model_executor.models"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.config", fake_config_module)

    import rcp_ndcg_vllm.models.pplx

    rcp_ndcg_vllm.models.pplx.register_pplx()

    assert registered == {PLUGIN_ARCHITECTURE: "rcp_ndcg_vllm.models.pplx.model:PplxContextualForPooling"}
    assert config_map[PLUGIN_ARCHITECTURE].__name__ == "PplxContextualConfig"

    # Re-entrant: a second call must not raise and must not double-register.
    rcp_ndcg_vllm.models.pplx.register_pplx()
    assert registered == {PLUGIN_ARCHITECTURE: "rcp_ndcg_vllm.models.pplx.model:PplxContextualForPooling"}


def test_register_refuses_a_vllm_outside_the_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.30.0")  # noqa: ARG005
    import rcp_ndcg_vllm.models.pplx

    with pytest.raises(RuntimeError, match="vllm/vllm-openai:v0.31.0"):
        rcp_ndcg_vllm.models.pplx.register_pplx()


def test_config_handler_forces_bidirectional_on_both_configs() -> None:
    from rcp_ndcg_vllm.models.pplx.config import PplxContextualConfig

    class FakeModelConfig:
        hf_config: types.SimpleNamespace
        hf_text_config: types.SimpleNamespace

    model_config = FakeModelConfig()
    model_config.hf_config = types.SimpleNamespace(is_causal=True)
    model_config.hf_text_config = types.SimpleNamespace(is_causal=True)
    PplxContextualConfig.verify_and_update_model_config(model_config)  # type: ignore[arg-type]
    assert model_config.hf_config.is_causal is False
    assert model_config.hf_text_config.is_causal is False
