"""Which rows the harness reports instead of gating: exactly those the client changed (decision 9).

A row is reported non-gating, under the recipe's declared over-cap deviation, exactly when the client changed
what it sends relative to the uncut input -- whatever made it: the budget counted with the frame (a content under
``max_tokens`` whose framed request is over it), a query settled at its share inside a pair the budget would take
whole, or a declared per-shape cap.  A row the client sent uncut gates exactly.  The harness reads that from the
client's own processing records (one per changed row, each change named by its mechanism), never by recomputing
the client's cut or by comparing the content count with the budget.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts, stage2_scores
from rcp_ndcg_vllm.equivalence.fitting import client_template

from rcp_ndcg.data.tokenizer import load_tokenizer
from tests.conftest import RECIPES, TOKENIZER, start_stub, write_pairs
from tests.test_equivalence import REFERENCE_PYTHON, _rebased

TOK = load_tokenizer(str(TOKENIZER))
DEVIATION = "over_cap_cut_differs"


def _words(count: int, word: str = "cities") -> str:
    """A text of ``count`` repetitions of ``word`` (the fixture tokenizer's whole words)."""
    return " ".join([word] * count)


def _text_with_content_tokens(low: int, high: int) -> str:
    """A text whose own token count lies in ``[low, high]`` (the fixture tokenizer)."""
    text = "cities"
    while TOK.count(text) < low:
        text += " cities"
    assert TOK.count(text) <= high, (low, high, TOK.count(text))
    return text


def _deviating(recipe: Any, deviation: str | None = DEVIATION) -> Any:
    """``recipe`` with ``reference.known_deviations`` set to ``[deviation]`` (``None``: none declared)."""
    deviations = [deviation] if deviation else []
    return recipe.model_copy(update={"reference": recipe.reference.model_copy(update={"known_deviations": deviations})})


def _embed_with_a_query_shape(tmp_path: Path) -> Any:
    """``fixture-embed`` plus a query shape capped by its own ``query_max_tokens`` (16): the fixture reference
    renders ``query: <query>`` for it, uncut."""
    directory = tmp_path / "scratch" / "recipes" / "embed-query"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "scratch" / "deterministic.py")
    shutil.copy(RECIPES / "fixture-embed" / "reference.py", directory / "reference.py")
    manifest = _rebased((RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8"), "embed-query")
    manifest = manifest.replace(
        "  template:\n    document:",
        '  query_max_tokens: 16\n  template:\n    query:\n      - {fixed: "query: "}\n      - {content: query}\n'
        "    document:",
    )
    (directory / "recipe.yaml").write_text(manifest, encoding="utf-8")
    return load_recipe(directory)


def _rerank_frame_only_row() -> dict[str, Any]:
    """A pair whose content (query + document) fits ``max_tokens`` (160) while the framed request does not."""
    recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    template = client_template(recipe)
    query = "capital of france"
    document = "cities"
    while TOK.count(query + document + " cities") <= 158:
        document += " cities"
    framed = TOK.count(template.render("pair", TOK, query=query, document=document), add_special_tokens=True)
    assert TOK.count(query) + TOK.count(document) <= 160 < framed, "content under the budget, the frame over it"
    return {"query": query, "documents": [document]}


def _rerank_share_row() -> dict[str, Any]:
    """A pair the budget takes whole (well under 160) whose query exceeds its share (48): the client settles it."""
    query = _text_with_content_tokens(52, 60)
    return {"query": query, "documents": ["cities and rivers"]}


def _rerank_with_a_document_cap() -> Any:
    """``fixture-rerank-pointwise`` with a per-document cap of 24 tokens beside its pair budget (H3): every
    :data:`UNDER_CAP` document stays under it."""
    recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    return recipe.model_copy(update={"client": {**recipe.client, "document_max_tokens": 24}})


def _embed_frame_only_row() -> dict[str, Any]:
    """A document whose content fits ``max_tokens`` (128) while ``doc: <document> [END]`` does not."""
    document = _text_with_content_tokens(125, 128)
    recipe = load_recipe(RECIPES / "fixture-embed")
    framed = TOK.count(client_template(recipe).render("document", TOK, document=document), add_special_tokens=True)
    assert framed > 128
    return {"query": "capital of france", "documents": [document]}


UNDER_CAP = {
    "query": "capital of france",
    "documents": ["cities and rivers in europe", "another document about rivers", "a b c"],
}
"""A row every shape sends uncut (three documents, so stage 2 has a per-query Kendall tau to gate)."""


def _case(name: str, tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    """The recipe and the over-cap row of one case."""
    if name == "rerank-frame-only":
        return load_recipe(RECIPES / "fixture-rerank-pointwise"), _rerank_frame_only_row()
    if name == "rerank-query-share":
        return load_recipe(RECIPES / "fixture-rerank-pointwise"), _rerank_share_row()
    if name == "rerank-document-share":
        return _rerank_with_a_document_cap(), {"query": "capital of france", "documents": [_words(6)]}
    if name == "embed-frame-only":
        return load_recipe(RECIPES / "fixture-embed"), _embed_frame_only_row()
    assert name == "embed-query-share"
    return _embed_with_a_query_shape(tmp_path), {"query": _text_with_content_tokens(20, 30), "documents": ["x"]}


EXPECTED = {
    "rerank-frame-only": "budget_cut",
    "rerank-query-share": "query_share",
    "rerank-document-share": "document_share",
    "embed-frame-only": "budget_cut",
    "embed-query-share": "budget_cut",  # the embed query shape's whole budget is its query_max_tokens
}
"""The mechanism the client's processing record names for each case's changed row."""

CASES = ["rerank-frame-only", "rerank-query-share", "rerank-document-share", "embed-frame-only", "embed-query-share"]


@pytest.mark.parametrize("case", CASES)
def test_stage1_reports_every_row_the_client_changed_and_gates_the_rest(tmp_path: Path, case: str) -> None:
    """Under the declared deviation the changed row is reported (non-gating) and the uncut row is compared;
    without it, the changed row gates -- and fails, since the reference renders the uncut input."""
    recipe, over = _case(case, tmp_path)
    pairs = write_pairs(tmp_path / "pairs.jsonl", [over, UNDER_CAP])
    document = stage1_prompts(_deviating(recipe), pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    render = document["render_check"]
    assert render["passed"] is True, render["failures"][:1]
    assert {row["index"] for row in render["over_cap"]["rows"]} == {0}, "only the changed row is reported"
    assert render["over_cap"]["gating"] is False
    mechanisms = {
        name for row in render["over_cap"]["rows"] for change in row["changes"] for name in change["mechanisms"]
    }
    assert mechanisms == {EXPECTED[case]}, "the report names the client's own mechanism"
    assert render["rows"] >= 2, "the uncut row was rendered and compared"

    gated = stage1_prompts(_deviating(recipe, None), pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    failures = gated["render_check"]["failures"]
    assert gated["render_check"]["passed"] is False
    assert {failure["index"] for failure in failures} == {0}, "the uncut row matched exactly"


def test_stage1_an_uncut_row_gates_exactly_under_the_deviation(tmp_path: Path) -> None:
    """A declared deviation never excuses a row the client sent uncut: a reference that diverges on it fails."""
    directory = tmp_path / "divergent" / "recipes" / "divergent"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "divergent" / "deterministic.py")
    source = (RECIPES / "fixture-embed" / "reference.py").read_text(encoding="utf-8")
    (directory / "reference.py").write_text(source.replace('SUFFIX = " [END]"', 'SUFFIX = " [END] "'), encoding="utf-8")
    manifest = (RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8")
    (directory / "recipe.yaml").write_text(_rebased(manifest, "divergent"), encoding="utf-8")
    pairs = write_pairs(tmp_path / "pairs.jsonl", [UNDER_CAP])
    render = stage1_prompts(_deviating(load_recipe(directory)), pairs, REFERENCE_PYTHON, over_length_per_shape=1)[
        "render_check"
    ]
    assert render["passed"] is False
    assert "over_cap" not in render


@pytest.mark.parametrize("case", ["rerank-frame-only", "rerank-query-share", "rerank-document-share"])
def test_stage2_rerank_reports_every_pair_the_client_changed(tmp_path: Path, case: str) -> None:
    """Stage 2 classifies on the same census rows: a frame-only overflow and a share settlement (which changes
    every pair of its call) are reported, the uncut row's pairs gate."""
    recipe, over = _case(case, tmp_path)
    pairs = write_pairs(tmp_path / "pairs.jsonl", [over, UNDER_CAP])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(_deviating(recipe), pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert {entry["query_index"] for entry in document["over_cap"]["pairs"]} == {0}
    assert {entry["query_index"] for entry in document["per_document"]} == {1}, "the uncut row gates"
    assert document["passed"] is True, document["gates"]


@pytest.mark.parametrize("case", ["embed-frame-only", "embed-query-share"])
def test_stage2_vectors_report_every_text_the_client_changed(tmp_path: Path, case: str) -> None:
    """The vector stage too: the changed text is reported, the uncut texts gate."""
    recipe, over = _case(case, tmp_path)
    pairs = write_pairs(tmp_path / "pairs.jsonl", [over, UNDER_CAP])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(_deviating(recipe), pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    reported = {entry["referent"] for entry in document["over_cap"]["pairs"]}
    shape = "query" if case == "embed-query-share" else "document"
    assert reported == {f"row 0 {shape} 0"}, reported
    assert all(entry["over_cap"] is False for entry in document["per_vector"])
    assert document["passed"] is True, document["per_vector"][:2]


def test_the_rerank_audit_holds_every_document_span_to_its_declared_cap(tmp_path: Path) -> None:
    """H3 on the wire: the over-length samples (padded in the document span) ship within the declared cap, and
    a captured document span over it fails the audit, as a query span over its share does."""
    from rcp_ndcg_vllm.equivalence import stages as stages_module

    recipe = _rerank_with_a_document_cap()
    pairs = write_pairs(tmp_path / "pairs.jsonl", [UNDER_CAP])
    audit = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)["anchor_check"]
    assert audit["passed"] is True, audit["failures"][:1]
    over = {"query": "capital of france", "queries": ["capital of france"], "documents": [_words(6)]}
    probe = {"rows": [{"shapes": {"pair": over}, "cuts": 1, "over_cap": True}]}
    failures = stages_module._anchor_check(recipe, probe, TOK)["failures"]
    assert [failure["check"] for failure in failures] == ["document_share"]


def _divergent(tmp_path: Path, recipe_id: str, old: str, new: str) -> Any:
    """``recipe_id`` copied with a reference that diverges on every text (``old`` replaced by ``new``)."""
    name = f"{recipe_id}-divergent"
    root = tmp_path / name
    directory = root / "recipes" / name
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", root / "deterministic.py")
    for extra in (RECIPES / recipe_id).glob("*.jinja"):
        shutil.copy(extra, directory / extra.name)
    source = (RECIPES / recipe_id / "reference.py").read_text(encoding="utf-8")
    assert source.count(old) == 1
    (directory / "reference.py").write_text(source.replace(old, new), encoding="utf-8")
    manifest = (RECIPES / recipe_id / "recipe.yaml").read_text(encoding="utf-8")
    manifest = _rebased(manifest, name)
    (directory / "recipe.yaml").write_text(manifest, encoding="utf-8")
    return load_recipe(directory)


@pytest.mark.parametrize(
    ("recipe_id", "old", "new"),
    [
        ("fixture-embed", 'SUFFIX = " [END]"', 'SUFFIX = " [END] "'),
        (
            "fixture-rerank-pointwise",
            '"documents": [str(document) for document in row["documents"]],',
            '"documents": [str(document) + " " for document in row["documents"]],',
        ),
    ],
)
def test_stage1_gates_every_text_the_client_sent_uncut_beside_a_changed_sibling(
    tmp_path: Path, recipe_id: str, old: str, new: str
) -> None:
    """Gating is per text, never per row: one cut document in a row makes only ITS comparison non-gating. The
    reference diverges on every text; the row's first document is sent uncut, its second is cut -- the first's
    mismatch must fail the render check, the second's is reported."""
    recipe = _deviating(_divergent(tmp_path, recipe_id, old, new))
    row = {"query": "capital of france", "documents": ["cities and rivers", _words(60)]}
    pairs = write_pairs(tmp_path / "pairs.jsonl", [row])
    render = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=1)["render_check"]
    assert render["passed"] is False, "the uncut document's mismatch gates"
    gated = render["failures"]
    assert gated and all(failure.get("span", "document 0") == "document 0" for failure in gated), gated
    if recipe_id == "fixture-rerank-pointwise":
        reported = [m["span"] for entry in render["over_cap"]["rows"] for m in entry["mismatches"]]
        assert reported == ["document 1"], reported


def test_a_record_under_no_position_changes_every_input_of_its_call() -> None:
    """A processing record whose id names no position (the media fit's owner fallback, the role's name) is not
    attributable to one input: every input of the call is reported changed -- never a crash, never a pass."""
    from types import SimpleNamespace

    from rcp_ndcg_vllm.equivalence import stages as stages_module

    from rcp_ndcg.data.preprocess import ProcessingRecord

    record = ProcessingRecord(corpus="rerank", input_id="rerank", shape="pair", mechanisms=("media_drop",))
    client = SimpleNamespace(processing=[record])
    assert stages_module._changed_rows(client, 0, 3) == [True, True, True]
