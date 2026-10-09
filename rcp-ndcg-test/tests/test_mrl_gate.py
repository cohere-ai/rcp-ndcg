"""Stage 2 gates every declared ``k`` ex-post from one full-width run (the study's acceptance item 6).

The served run and the reference run are both full width (the recipe's selection is stripped for the
served pass); stage 2 applies the product's one MRL head to both sides per declared ``k`` and then the
ordinary per-vector/per-token cosine gate, so one forward pass gates the full width and every declared
cut.  ``k`` is not part of any stored-output key: the stored full-width vectors serve every ``k``, and the
head's semantics are versioned by the gating code (``MRL_GATE_VERSION``).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.equivalence import run, stage2_scores
from rcp_ndcg_test.equivalence.stages import MRL_GATE_VERSION
from rcp_ndcg_vllm.recipe import load_recipe

from rcp_ndcg.errors import ConfigError
from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub, write_pairs

REFERENCE_PYTHON = sys.executable


def load(recipe_id: str) -> Any:
    return load_recipe(RECIPES / recipe_id)


def _pairs(tmp_path: Path) -> Path:
    return write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])


def test_stage2_gates_every_declared_k_from_one_full_width_run(tmp_path: Path) -> None:
    """The declared set [2, 4, 8] produces the full-width row plus one row per ``k``, all from one pass."""
    recipe = load("fixture-embed-mrl")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    assert [row["mrl_dim"] for row in document["gates"]] == [None, 2, 4, 8]
    assert all(row["passed"] for row in document["gates"]), document["gates"]
    assert {entry["mrl_dim"] for entry in document["per_vector"]} == {None, 2, 4, 8}
    assert document["mrl_gate_version"] == MRL_GATE_VERSION


def test_stage2_strips_the_engine_side_selection_for_the_full_width_run(tmp_path: Path) -> None:
    """The recipe selects ``dimensions: 4``; the served pass must ask full width (the engine's cut would
    discard every other ``k``), so no captured ``/v1/embeddings`` request carries ``dimensions``."""
    recipe = load("fixture-embed-mrl")
    assert recipe.client.get("dimensions") == 4
    recorder: list[dict[str, Any]] = []
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(
            recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url, recorder=recorder
        )
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    bodies = [entry["request_body"] for entry in recorder if str(entry["url"]).endswith("/v1/embeddings")]
    assert bodies, "the embed client must have sent requests"
    assert all("dimensions" not in body for body in bodies), bodies[0]


def test_stage2_strips_the_client_side_selection_and_gates_the_full_width(tmp_path: Path) -> None:
    """The pooling recipe selects ``mrl_dim: 4``; the served pass must be the 8-wide full width, or the
    ``k=8`` gate could not apply the head at all (the head refuses a ``k`` wider than the vectors)."""
    recipe = load("fixture-multi-vector-mrl")
    assert recipe.client.get("mrl_dim") == 4
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    assert [row["mrl_dim"] for row in document["gates"]] == [None, 2, 4, 8]


def test_stage2_uses_the_product_head_for_a_k_wider_than_the_vectors(tmp_path: Path) -> None:
    """A declared ``k`` above the checkpoint's width is refused by the product's head, not silently
    sliced to the shorter width: the stage fails loudly instead of gating a shape it never produced."""
    source = RECIPES / "fixture-embed-mrl"
    root = tmp_path / "wide"
    directory = root / "recipes" / "fixture-embed-mrl-wide"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", root / "deterministic.py")
    (directory / "reference.py").write_text((source / "reference.py").read_text(encoding="utf-8"), encoding="utf-8")
    manifest = (source / "family.yaml").read_text(encoding="utf-8")
    manifest = (
        manifest.replace("id: fixture-embed-mrl", "id: fixture-embed-mrl-wide")
        .replace("tokenizer: ../../tokenizer.json", f"tokenizer: {TOKENIZER}")
        .replace("mrl_dims: [2, 4, 8]", "mrl_dims: [2, 4, 16]")
    )
    (directory / "family.yaml").write_text(manifest, encoding="utf-8")
    recipe = load_recipe(directory)
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        with pytest.raises(ConfigError, match="wider than the vectors"):
            stage2_scores(recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()


def test_the_report_names_every_gated_k(tmp_path: Path) -> None:
    """``EQUIVALENCE.md`` carries one gate row per ``k`` (and the full-width row)."""
    recipe = load("fixture-embed-mrl")
    out = tmp_path / "out"
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = run(
            recipe,
            base_url=engine.base_url,
            pairs_path=str(_pairs(tmp_path)),
            out_dir=str(out),
            stages=[2],
            reference_python=REFERENCE_PYTHON,
        )
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    markdown = (out / "EQUIVALENCE.md").read_text(encoding="utf-8")
    assert "k=2" in markdown and "k=4" in markdown and "k=8" in markdown


def test_stage2_gates_a_ranges_endpoints_and_the_run_selection(tmp_path: Path) -> None:
    """A range declaration cannot be enumerated: the gate covers its two endpoints AND the run's selection,
    which the served client no longer carries (the full-width pass strips it)."""
    source = RECIPES / "fixture-embed-mrl"
    root = tmp_path / "range"
    directory = root / "recipes" / "fixture-embed-mrl-range"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", root / "deterministic.py")
    (directory / "reference.py").write_text((source / "reference.py").read_text(encoding="utf-8"), encoding="utf-8")
    manifest = (source / "family.yaml").read_text(encoding="utf-8")
    manifest = (
        manifest.replace("id: fixture-embed-mrl", "id: fixture-embed-mrl-range")
        .replace("tokenizer: ../../tokenizer.json", f"tokenizer: {TOKENIZER}")
        .replace("mrl_dims: [2, 4, 8]", "mrl_range: [2, 8]")
    )
    (directory / "family.yaml").write_text(manifest, encoding="utf-8")
    recipe = load_recipe(directory)
    assert recipe.client.get("dimensions") == 4
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    assert [row["mrl_dim"] for row in document["gates"]] == [None, 2, 8, 4]


def test_stage2_reports_the_base_vector_count_and_the_per_k_comparisons(tmp_path: Path) -> None:
    """``n_vectors`` stays the base vectors compared (the full-width rows); ``n_comparisons`` counts every
    per-k comparison, so a multi-k summary never conflates the two units."""
    recipe = load("fixture-embed-mrl")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, _pairs(tmp_path), REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document["passed"] is True, document["gates"]
    assert document["n_vectors"] == sum(1 for entry in document["per_vector"] if entry["mrl_dim"] is None)
    assert document["n_comparisons"] == len(document["per_vector"])
    assert document["n_comparisons"] == 4 * document["n_vectors"]


def test_mrl_gate_builds_the_product_head_from_the_projection_declaration() -> None:
    """The projection branch configures the product's own :class:`MrlHead` with the declared source (the
    harness never slices a projection-kind checkpoint)."""
    from rcp_ndcg_test.equivalence.stages import _mrl_gate

    from rcp_ndcg.data.mrl import MrlProjection
    from rcp_ndcg.inference.config import EmbeddingEndpoint

    config = EmbeddingEndpoint(
        base_url="http://127.0.0.1:8100/v1",
        model="fixture",
        tokenizer="fixtures/tokenizer.json",
        max_tokens=64,
        mrl_kind="projection",
        mrl_dims=(4,),
        mrl_projection=MrlProjection(source="hf://org/model@revision/projections.safetensors"),
    )
    head, dims = _mrl_gate(config, 4)
    assert head is not None and head.kind == "projection" and head.dims == (4,)
    assert head.projection is not None and head.projection.source.endswith("projections.safetensors")
    assert dims == (None, 4)
