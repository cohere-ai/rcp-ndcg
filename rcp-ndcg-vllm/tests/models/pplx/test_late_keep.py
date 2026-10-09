"""The pplx-late engine-side keep-rule: the declared rule read from the engine config and the positions it
keeps, as data.

The rule's home is the recipe (``client.document_skip_token_ids``, rendered into the engine's
``serve.hf_overrides.document_skip_token_ids`` and cross-checked by the loader); the plugin reads it from
the served model's HF config and drops the excluded positions before the vectors leave the pooler, so the
wire carries only kept vectors. The two functions below are vLLM-free on purpose: the CPU suite pins the
rule's semantics with fake token-id rows, and the vLLM-side pooler that calls them is pinned by source and
by the GPU wave.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_vllm.models.pplx.late_keep import declared_skip_ids, kept_positions


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


def test_the_late_pooler_applies_the_declared_rule() -> None:
    """The vLLM-side wiring is pinned by source where vLLM cannot be imported (the plugin venv carries no
    vLLM by design): the late model builds the keep pooler when the rule is declared, and the pooler indexes
    the per-sequence hidden states with :func:`kept_positions`. The behaviour itself is the GPU wave's (the
    engine image has vLLM 0.31.0)."""
    from pathlib import Path

    import rcp_ndcg_vllm.models.pplx as package

    root = Path(package.__file__).parent
    pooler = (root / "late_pooler.py").read_text(encoding="utf-8")
    keep = (root / "late_keep.py").read_text(encoding="utf-8")
    model = (root / "late.py").read_text(encoding="utf-8")
    assert "kept_positions" in pooler and "declared_skip_ids" in pooler
    assert "build_late_pooler" in model and "declared_skip_ids" in model
    # The key is read in exactly one place: the keep module's own accessor (the model class only asks it).
    assert "SKIP_IDS_KEY" in keep and "SKIP_IDS_KEY" not in model and "SKIP_IDS_KEY" not in pooler
