"""The engine-side keep-rules: the declared rules read from the engine config, the positions they keep, and
the pooler that applies them.

The rules' home is the recipe (``client.document_skip_token_ids`` / ``client.media_keep_token_ids``,
rendered into the engine's ``serve.hf_overrides`` and cross-checked by the loader); the plugin reads them
from the served model's HF config and drops the excluded positions before the vectors leave the pooler, so
the wire carries only kept vectors. The text skip rule is document-side (pplx-embed-v2-late's mask declares
``skiplist_tasks: ["document"]``), so its engine half also declares the document role prefix
(``document_skip_prefix_token_id``): a row that does not open with it is a query prompt and keeps every
position. The media allowlist is gated by its own presence: a row carrying one of its ids is a media
document (topk-embed-v1's image render) and keeps only the allowlist's positions.

The vLLM-side pooler is exercised against a stubbed vLLM surface (the plugin venv carries no vLLM by
design): the stubs are the tiny pieces ``keep_pooler``/``token_pooler`` import, with the one behaviour under
test -- the per-sequence split of the hidden states -- kept faithful, so the rules' application (which
positions reach the head) is a real assertion, not a source-string pin.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from rcp_ndcg_vllm.models.keep_rule import (
    declared_keep_ids,
    declared_skip_ids,
    declared_skip_prefix_id,
    kept_positions,
)

#: The plugin modules that import vLLM: re-imported under the stubs and dropped again afterwards, so a stub
#: never leaks into another test's imports.
_VLLM_DEPENDENT = (
    "rcp_ndcg_vllm.models.keep_pooler",
    "rcp_ndcg_vllm.models.token_pooler",
    "rcp_ndcg_vllm.models.pplx.pooler",
)


class TestKeptPositions:
    """The pure rule: the positions whose token id is not in the declared skip list, ascending."""

    def test_the_declared_ids_are_dropped_and_the_rest_kept(self) -> None:
        ids = [248078, 248053, 11, 248056, 11, 248054]
        assert kept_positions(ids, [11]) == [0, 1, 3, 5]
        assert kept_positions(ids, []) == [0, 1, 2, 3, 4, 5]

    def test_an_image_render_keeps_its_structural_positions_unless_the_rule_names_them(self) -> None:
        """The plugin sees the render's own ids: a media document's head and vision markers stay unless the
        declared rule lists them -- the checkpoint's punctuation mask never does, while a rule that named a
        structural id would drop exactly that position (the plugin's contract is by token id, nothing else)."""
        render = [248078, 248053, 248056, 248056, 248054]
        assert kept_positions(render, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 25, 26]) == [0, 1, 2, 3, 4]
        assert kept_positions(render, [248053, 248054]) == [0, 2, 3]

    def test_a_repeated_id_drops_at_every_position(self) -> None:
        assert kept_positions([7, 7, 8], [7]) == [2]

    def test_a_row_fully_inside_the_rule_keeps_nothing(self) -> None:
        assert kept_positions([11, 12], [11, 12]) == []

    def test_an_empty_row_keeps_nothing(self) -> None:
        assert kept_positions([], [11]) == []

    def test_a_query_row_keeps_every_position(self) -> None:
        """The rule is document-side: with the document prefix declared, a row that does not open with it
        (the checkpoint's ``[Q] `` prompt, 248077) keeps every position -- the reference's ``encode_query``
        keeps them whole, and the checkpoint's mask declares ``skiplist_tasks: ["document"]``."""
        query = [248077, 3710, 557, 30, 11]
        assert kept_positions(query, [11, 30], document_prefix_id=248078) == [0, 1, 2, 3, 4]
        # A document row (the ``[D] `` prefix) still drops the rule's positions.
        assert kept_positions([248078, 3710, 30, 11], [11, 30], document_prefix_id=248078) == [0, 1]
        # An empty row is not a document either.
        assert kept_positions([], [11], document_prefix_id=248078) == []

    def test_a_media_row_keeps_only_the_allowlist_positions(self) -> None:
        """The media allowlist: a row carrying one of its ids is a media document (topk-embed-v1's image
        render: the chat template's wrapper, then the image-patch token, then a caption), and only the
        allowlist's positions are kept -- the wrapper and the caption drop, exactly what that checkpoint's
        reference keeps (``topk_embed_st.py:_image_row``: ``keep = ids == image_token_id``)."""
        image = 248056
        render = [248045, 846, 198, 248053, image, image, 248054, 248046, 198]
        assert kept_positions(render, [0, 1, 2], keep_ids=[image]) == [4, 5]
        captioned = [*render, 11, 22, 11]
        assert kept_positions(captioned, [11], keep_ids=[image]) == [4, 5], "the caption is not in the allowlist"
        # A row without an allowlist id follows the text rule (here: position 1 is skipped).
        assert kept_positions([846, 11, 22], [11], keep_ids=[image]) == [0, 2]
        # The allowlist takes precedence over the text rule when both are declared.
        assert kept_positions(render, [image], keep_ids=[image]) == [4, 5]

    def test_a_row_with_no_allowlist_id_follows_the_text_rule(self) -> None:
        """A text document (no allowlist id) still drops the text rule's positions; an empty allowlist is
        no allowlist at all."""
        assert kept_positions([7, 11, 9], [11], keep_ids=[248056]) == [0, 2]
        assert kept_positions([7, 11, 9], [11], keep_ids=[]) == [0, 2]


class TestDeclaredSkipIds:
    """The rule as the engine config carries it: ``hf_overrides.document_skip_token_ids``."""

    class _Config:
        def __init__(self, **fields: object) -> None:
            self.__dict__.update(fields)

    class _ModelConfig:
        def __init__(self, hf_config: object) -> None:
            self.hf_config = hf_config

    def test_an_undeclared_rule_is_empty(self) -> None:
        assert declared_skip_ids(self._ModelConfig(self._Config())) == ()

    def test_the_declared_list_is_read_as_ints(self) -> None:
        model_config = self._ModelConfig(self._Config(document_skip_token_ids=[0, 11, 248056]))
        assert declared_skip_ids(model_config) == (0, 11, 248056)

    def test_a_malformed_declaration_is_refused_by_name(self) -> None:
        model_config = self._ModelConfig(self._Config(document_skip_token_ids="0,11"))
        with pytest.raises(ValueError, match="document_skip_token_ids"):
            declared_skip_ids(model_config)
        model_config = self._ModelConfig(self._Config(document_skip_token_ids=["zero"]))
        with pytest.raises(ValueError, match="document_skip_token_ids"):
            declared_skip_ids(model_config)

    def test_the_document_gate_is_read_from_the_engine_half(self) -> None:
        assert declared_skip_prefix_id(self._ModelConfig(self._Config())) is None
        model_config = self._ModelConfig(self._Config(document_skip_prefix_token_id=248078))
        assert declared_skip_prefix_id(model_config) == 248078

    def test_a_malformed_document_gate_is_refused_by_name(self) -> None:
        model_config = self._ModelConfig(self._Config(document_skip_prefix_token_id="[D] "))
        with pytest.raises(ValueError, match="document_skip_prefix_token_id"):
            declared_skip_prefix_id(model_config)

    def test_the_media_allowlist_is_read_from_the_engine_half(self) -> None:
        assert declared_keep_ids(self._ModelConfig(self._Config())) == ()
        model_config = self._ModelConfig(self._Config(document_keep_token_ids=[248056]))
        assert declared_keep_ids(model_config) == (248056,)

    def test_a_malformed_allowlist_is_refused_by_name(self) -> None:
        for value in ("248056", [True], ["x"]):
            model_config = self._ModelConfig(self._Config(document_keep_token_ids=value))
            with pytest.raises(ValueError, match="document_keep_token_ids"):
                declared_keep_ids(model_config)


def _stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the tiny vLLM surface ``keep_pooler``/``token_pooler``/``pplx.pooler`` import (the plugin venv
    carries no vLLM).

    ``AllPool.forward`` is faithful to vLLM's per-sequence split (``torch.split`` on the scheduled lengths)
    and ``PoolingMetadata`` carries the CPU id row and the prompt lengths, so the pooler's own work -- which
    positions reach the head -- is what the tests below assert.
    """
    import torch

    class AllPool:
        def __init__(self) -> None:
            self.clone_finished = False

        def forward(self, hidden_states: Any, pooling_metadata: Any) -> list[Any]:
            lengths = [int(length) for length in pooling_metadata.prompt_lens]
            rows: list[Any] = list(torch.split(hidden_states, lengths))
            # The unfinished chunked-prefill rows AllPool returns None for (the metadata's own marker).
            unfinished = getattr(pooling_metadata, "unfinished", None)
            if unfinished is not None:
                rows = [None if flag else row for flag, row in zip(unfinished, rows, strict=True)]
            return rows

    class TokenPooler:
        def __init__(self, pooling: Any, head: Any = None) -> None:
            self.pooling = pooling
            self.head = head

    class TokenEmbeddingPoolerHead:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class PoolingParamsUpdate:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class PoolingMetadata:
        def __init__(self, rows: list[list[int]], unfinished: list[bool] | None = None) -> None:
            # The real metadata holds one padded (n_seq, max_len) CPU tensor and slices each row to its
            # prompt length in ``get_prompt_token_ids_cpu`` (vllm/v1/pool/metadata.py:100-115).
            width = max(len(row) for row in rows)
            padded = [row + [0] * (width - len(row)) for row in rows]
            self.prompt_token_ids_cpu = torch.tensor(padded, dtype=torch.long)
            self.prompt_lens = [len(row) for row in rows]
            self.unfinished = unfinished

        def get_prompt_token_ids_cpu(self) -> list[Any]:
            return [self.prompt_token_ids_cpu[i, :num] for i, num in enumerate(self.prompt_lens)]

    class PoolerNormalize:
        pass

    class TokenPoolingMethod:
        pass

    class TokenPoolingMethodOutputItem:
        pass

    class ModelConfig:
        pass

    modules: dict[str, types.ModuleType] = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.tasks": types.ModuleType("vllm.tasks"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.pooler": types.ModuleType("vllm.model_executor.layers.pooler"),
        "vllm.model_executor.layers.pooler.activations": types.ModuleType(
            "vllm.model_executor.layers.pooler.activations"
        ),
        "vllm.model_executor.layers.pooler.tokwise": types.ModuleType("vllm.model_executor.layers.pooler.tokwise"),
        "vllm.model_executor.layers.pooler.tokwise.heads": types.ModuleType(
            "vllm.model_executor.layers.pooler.tokwise.heads"
        ),
        "vllm.model_executor.layers.pooler.tokwise.methods": types.ModuleType(
            "vllm.model_executor.layers.pooler.tokwise.methods"
        ),
        "vllm.model_executor.layers.pooler.tokwise.poolers": types.ModuleType(
            "vllm.model_executor.layers.pooler.tokwise.poolers"
        ),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.pool": types.ModuleType("vllm.v1.pool"),
        "vllm.v1.pool.metadata": types.ModuleType("vllm.v1.pool.metadata"),
    }
    modules["vllm.config"].ModelConfig = ModelConfig  # type: ignore[attr-defined]
    modules["vllm.tasks"].PoolingTask = str  # type: ignore[attr-defined]
    modules["vllm.model_executor.layers.pooler"].PoolingParamsUpdate = PoolingParamsUpdate  # type: ignore[attr-defined]
    modules["vllm.model_executor.layers.pooler.activations"].PoolerNormalize = PoolerNormalize  # type: ignore[attr-defined]
    modules["vllm.model_executor.layers.pooler.tokwise.heads"].TokenEmbeddingPoolerHead = (  # type: ignore[attr-defined]
        TokenEmbeddingPoolerHead
    )
    methods = modules["vllm.model_executor.layers.pooler.tokwise.methods"]
    methods.AllPool = AllPool  # type: ignore[attr-defined]
    methods.TokenPoolingMethod = TokenPoolingMethod  # type: ignore[attr-defined]
    methods.TokenPoolingMethodOutputItem = TokenPoolingMethodOutputItem  # type: ignore[attr-defined]
    modules["vllm.model_executor.layers.pooler.tokwise.poolers"].TokenPooler = TokenPooler  # type: ignore[attr-defined]
    modules["vllm.v1.pool.metadata"].PoolingMetadata = PoolingMetadata  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in _VLLM_DEPENDENT:
        sys.modules.pop(name, None)


@pytest.fixture
def stubbed_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stubbed vLLM surface, dropped again after the test (a stub must not leak into another import)."""
    _stub_vllm(monkeypatch)
    try:
        yield
    finally:
        for name in _VLLM_DEPENDENT:
            sys.modules.pop(name, None)


def _model_config(**hf_fields: object) -> Any:
    """A stand-in for vLLM's ``ModelConfig``: ``head_dtype`` and the declared HF-config fields."""
    hf_config = types.SimpleNamespace(**hf_fields)
    return types.SimpleNamespace(head_dtype=None, hf_config=hf_config)


def test_the_pooler_drops_the_rules_positions_and_spares_a_query(stubbed_vllm: None) -> None:
    """The engine-side application, behaviourally: a document row keeps the positions the rule keeps, a query
    row keeps every position, an empty rule returns the stock pooler's rows, and an unfinished chunked-prefill
    sequence stays ``None``."""
    import torch
    from rcp_ndcg_vllm.models.keep_pooler import KeepPool

    hidden = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    pool = KeepPool(skip_ids=(11,), document_prefix_id=248078)
    metadata = sys.modules["vllm.v1.pool.metadata"].PoolingMetadata(
        [[248078, 11, 7, 11, 9], [248077, 11, 7, 11, 9], [248078, 7, 9]]
    )
    kept, query, shorter = pool.forward(hidden, metadata)
    assert kept.tolist() == [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]  # positions 0, 2, 4 of the document row
    assert query.tolist() == hidden[5:10].tolist(), "a query row keeps every position"
    assert shorter.tolist() == [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]

    plain = KeepPool(skip_ids=())
    rows = plain.forward(hidden, metadata)
    assert len(rows) == 3 and len(rows[0]) == 5, "an empty rule returns the AllPool rows unchanged"


def test_the_pooler_keeps_only_the_media_allowlist_positions(stubbed_vllm: None) -> None:
    """The media allowlist, behaviourally: a row carrying the image-patch token keeps only those positions
    (the wrapper, the caption and the query rows are untouched by the allowlist)."""
    import torch
    from rcp_ndcg_vllm.models.keep_pooler import KeepPool

    image = 248056
    hidden = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    pool = KeepPool(keep_ids=(image,))
    metadata = sys.modules["vllm.v1.pool.metadata"].PoolingMetadata(
        [[248053, image, image, 11, 22], [846, 11, 22, 33, 44]]
    )
    media, query = pool.forward(hidden, metadata)
    assert media.tolist() == [[2.0, 3.0], [4.0, 5.0]], "only the image-patch positions survive"
    assert query.tolist() == hidden[5:10].tolist(), "a row without an allowlist id keeps everything"


@pytest.mark.parametrize("unfinished", [[True], [False]])
def test_an_unfinished_chunked_prefill_sequence_stays_none(stubbed_vllm: None, unfinished: list[bool]) -> None:
    import torch
    from rcp_ndcg_vllm.models.keep_pooler import KeepPool

    pool = KeepPool(skip_ids=(11,), document_prefix_id=248078)
    metadata = sys.modules["vllm.v1.pool.metadata"].PoolingMetadata([[248078, 11, 7]], unfinished=unfinished)
    (pooled,) = pool.forward(torch.arange(6, dtype=torch.float32).reshape(3, 2), metadata)
    if unfinished[0]:
        assert pooled is None, "an unfinished sequence stays None (AllPool's own contract)"
    else:
        assert pooled.tolist() == [[0.0, 1.0], [4.0, 5.0]], "the finished row keeps positions 0 and 2"


def test_the_rule_needs_the_document_gate(stubbed_vllm: None) -> None:
    """A text skip rule without the document role prefix is refused: it would drop a query prompt's positions
    too. A media allowlist needs no prefix (its presence is its own gate)."""
    import torch
    from rcp_ndcg_vllm.models.keep_pooler import build_keep_pooler

    with pytest.raises(ValueError, match="document_skip_prefix_token_id"):
        build_keep_pooler(_model_config(document_skip_token_ids=[11]), projector=torch.nn.Identity())

    pooler = build_keep_pooler(
        _model_config(document_skip_token_ids=[11], document_skip_prefix_token_id=248078),
        projector=torch.nn.Identity(),
    )
    assert pooler.pooling._skip_ids == (11,)
    assert pooler.pooling._document_prefix_id == 248078

    topk = build_keep_pooler(_model_config(document_keep_token_ids=[248056]), projector=torch.nn.Identity())
    assert topk.pooling._skip_ids == ()
    assert topk.pooling._document_prefix_id is None
    assert topk.pooling._keep_ids == (248056,)
