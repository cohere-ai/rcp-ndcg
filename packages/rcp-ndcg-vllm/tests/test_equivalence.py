"""Stage 1 and stage 2 through the product's role clients: the captured wire, the audit and the gates.

The client's captured requests are what every stage-1 check audits (no harness-side re-derivation); stage 2
sends through the same clients with the recipe's real budget and gates the answers against the reference
subprocess's outputs.  The harness process never imports torch or transformers.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts, stage2_scores
from rcp_ndcg_vllm.equivalence.gates import ResolvedGates
from rcp_ndcg_vllm.equivalence.metrics import stage3_metrics

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub, write_pairs

REFERENCE_PYTHON = sys.executable


def load(recipe_id: str) -> Any:
    return load_recipe(RECIPES / recipe_id)


def _rebased(manifest: str, new_id: str) -> str:
    """A fixture manifest copied to a scratch directory: its id and its recipe-relative tokenizer rebased."""
    return (
        manifest.replace("id: fixture-rerank-pointwise", f"id: {new_id}")
        .replace("id: fixture-embed", f"id: {new_id}")
        .replace("tokenizer: ../../tokenizer.json", f"tokenizer: {TOKENIZER}")
    )


@pytest.mark.parametrize(
    "recipe_id",
    [
        "fixture-embed",
        "fixture-embed-cls",
        "fixture-embed-edge",
        "fixture-embed-marker",
        "fixture-multi-vector",
        "fixture-rerank-pointwise",
        "fixture-rerank-listwise",
    ],
)
def test_stage1_passes_for_every_anchor_kind_with_the_reference_render(tmp_path: Path, recipe_id: str) -> None:
    """The client's captured requests, the reference subprocess render and the template check all agree."""
    recipe = load(recipe_id)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=3)
    assert document["client"]["exchanges"] > 0, recipe_id
    assert document["anchor_check"]["passed"] is True, (recipe_id, document["anchor_check"]["failures"][:1])
    assert document["render_check"]["passed"] is True, (recipe_id, document["render_check"]["failures"][:1])
    if recipe.serve.chat_template is not None:
        assert document["template_render_check"]["passed"] is True, recipe_id


def test_stage1_render_check_catches_a_divergent_reference(tmp_path: Path) -> None:
    """The render check is a real comparison: a one-character divergent render fails, and so does an
    under-rendering reference (a missing declared shape)."""
    load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    source = (RECIPES / "fixture-embed" / "reference.py").read_text(encoding="utf-8")
    manifest = (RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8")
    for name, mutation in (
        ("divergent", ('PREFIX = "doc: "', 'PREFIX = "doc:  "')),
        ("empty", ('output_result = {"rows": rows}', 'output_result = {"rows": []}')),
    ):
        # The fixture references import their deterministic helpers two levels up: mirror that layout
        # (scratch/<name>/<id>/reference.py, scratch/deterministic.py).
        directory = tmp_path / name / "recipes" / name
        directory.mkdir(parents=True)
        shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / name / "deterministic.py")
        old, new = mutation
        assert old in source, name
        (directory / "reference.py").write_text(source.replace(old, new), encoding="utf-8")
        (directory / "recipe.yaml").write_text(_rebased(manifest, name), encoding="utf-8")
        document = stage1_prompts(load_recipe(directory), pairs, REFERENCE_PYTHON, over_length_per_shape=1)
        assert document["render_check"]["passed"] is False, name
        assert document["render_check"]["failures"], name


def test_stage1_audits_the_clients_settled_query(tmp_path: Path) -> None:
    """The settle-once rule, audited on the captured wire: one query span per row, within its declared share."""
    from rcp_ndcg_vllm.equivalence.wire import role_client

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load("fixture-rerank-pointwise")
    pairs = write_pairs(
        tmp_path / "pairs.jsonl",
        [{"query": "What is the capital of France?", "documents": ["short. " * 80, "Paris is the capital of France."]}],
    )
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    client, capture = role_client(recipe, None)
    client.rerank("What is the capital of France?", ["short. " * 80, "Paris is the capital of France."])
    queries = [capture.texts(exchange).get("query") for exchange in capture.exchanges]
    assert queries and len(set(queries)) == 1, f"one settled span per row, got {len(set(queries))}"
    tokenizer = load_tokenizer(str(TOKENIZER))
    assert tokenizer.count(str(queries[0])) <= recipe.client.query_max_tokens


def test_stage1_engine_tokenize_check_runs_against_the_stub(tmp_path: Path) -> None:
    """The engine's /tokenize must agree with the recipe tokenizer's ids (R29); without an engine: not_run."""
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage1_prompts(recipe, pairs, None, base_url=engine.base_url, over_length_per_shape=2)
        assert document["engine_tokenize_check"]["status"] == "run"
        assert document["engine_tokenize_check"]["passed"] is True
        assert document["engine_tokenize_check"]["checked"] > 0
    finally:
        engine.stop()
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None  # not_run is neutral, never passed


def test_engine_tokenize_check_fails_when_the_engine_tokenizes_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutant engine whose /tokenize disagrees with the recipe tokenizer fails the check (R29)."""
    import httpx

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load("fixture-embed")
    tokenizer = load_tokenizer(str(TOKENIZER))
    text = "doc: Paris is the capital of France. [END]"
    real_ids = tokenizer.ids(text, add_special_tokens=True)

    def drifted_post(url: str, **_: Any) -> Any:
        return httpx.Response(200, json={"tokens": real_ids[:-1], "count": len(real_ids) - 1})

    monkeypatch.setattr(httpx, "post", drifted_post)
    probe = {"rows": [{"shapes": {"document": {"texts": [text]}}, "cuts": 0, "over_cap": False}]}
    check = stages_module_check(recipe, probe, tokenizer)
    assert check is not None and check["passed"] is False and check["failures"]


def stages_module_check(recipe: Any, probe: dict[str, Any], tokenizer: Any) -> Any:
    """The /tokenize check, driven directly (the mutant-engine path the suite exercises)."""
    from rcp_ndcg_vllm.equivalence import stages as stages_module

    return stages_module._engine_tokenize_check(recipe, probe, tokenizer, "http://engine")


def test_stage1_without_a_reference_python_reports_not_run(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert document["render_check"]["status"] == "not_run"
    assert document["anchor_check"]["passed"] is True


def test_stage1_carves_over_cap_rows_out_of_the_render_check_when_declared(tmp_path: Path) -> None:
    """With anchor_drop_over_cap declared, over-cap pairs-file rows are reported non-gating in stage 1 too."""
    recipe = load("fixture-rerank-pointwise")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]})}
    )
    over_cap_row = {"query": "over the cap " * 40, "documents": ["document tokens"]}
    pairs = write_pairs(tmp_path / "pairs.jsonl", [over_cap_row, *sample_pairs(1)])
    document = stage1_prompts(deviating, pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    render = document["render_check"]
    assert render["status"] == "run"
    assert render.get("over_cap", {}).get("n_rows", 0) >= 1
    assert render["over_cap"]["gating"] is False
    assert render["passed"] is True, render["failures"][:1]


def test_stage1_compares_shaped_pairs_rows(tmp_path: Path) -> None:
    """A pairs-file row carrying the documented per-row ``shape`` field is compared, not silently dropped."""
    recipe = load("fixture-embed")
    shaped = {"query": "capital of france", "documents": ["document 0 about cities"], "shape": "document"}
    pairs = write_pairs(tmp_path / "pairs.jsonl", [shaped])
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    render = document["render_check"]
    assert render["status"] == "run"
    assert render["rows"] >= 1, "the shaped row must be sent to the reference and compared"


def test_stage2_rerank_through_the_product_client(tmp_path: Path) -> None:
    """Stage 2 sends through the product's RerankClient (the recipe's real budget) and gates the scores."""
    recipe = load("fixture-rerank-pointwise")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["passed"] is True, document["gates"]
        assert document["abs_delta_max"] == 0.0
        assert document["kendall_tau_median"] == 1.0
        assert document["over_cap"]["n_pairs"] == 0
    finally:
        engine.stop()


def test_stage2_embed_through_the_product_client_compares_every_text(tmp_path: Path) -> None:
    """Stage 2 sends through the product's EmbeddingClient: every document's vectors compare."""
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["passed"] is True, document["per_vector"][:2]
        assert all(entry["within"] for entry in document["per_vector"])
    finally:
        engine.stop()


def test_stage2_multi_vector_compares_every_text_and_fails_on_noise(tmp_path: Path) -> None:
    """The multi_vector stage compares every text's token rows -- and the gates fail a noisy stub."""
    recipe = load("fixture-multi-vector")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["passed"] is True, document["per_vector"][:2]
        assert sum(entry["referent"].startswith("row 0 document") for entry in document["per_vector"]) >= 2
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.2", "--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=noisy.base_url)
        assert document["passed"] is False
    finally:
        noisy.stop()


def test_stage2_gates_only_under_cap_pairs_when_the_deviation_is_declared(tmp_path: Path) -> None:
    """With anchor_drop_over_cap, over-cap pairs (the client's own census) are non-gating; the rest gate."""
    recipe = load("fixture-rerank-pointwise")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]})}
    )
    pairs = write_pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query": "over the cap", "documents": ["long document tokens " * 400]},
            *sample_pairs(2)[:1],
        ],
    )
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(deviating, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["over_cap"]["n_pairs"] >= 1, "the long pair must classify over cap"
        assert document["over_cap"]["known_deviation"] is True
        assert document["passed"] is True, document["per_document"][:1]
        assert [entry for entry in document["per_document"] if not entry["over_cap"]]
        assert document["over_cap"]["pairs"], "the carved-out pair is named with its delta"
        assert document["over_cap"]["pairs"][0]["abs_delta"] > 0.05, (
            "the over-cap pair would have failed the gates (the reference scores the whole document)"
        )
    finally:
        engine.stop()
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["passed"] is False
        assert document["over_cap"]["known_deviation"] is False
        assert document["over_cap"]["pairs"] and document["over_cap"]["pairs"][0]["within"] is False
    finally:
        engine.stop()


def test_stage2_carves_a_chunked_over_cap_document_out_when_declared(tmp_path: Path) -> None:
    """A chunked over-cap document's census rows carry <original>#<chunk>: the carve-out classifies them."""
    recipe = load("fixture-rerank-pointwise")
    from rcp_ndcg.data.preprocess import ChunkPolicy

    chunking = recipe.model_copy(
        update={
            "reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]}),
            "client": recipe.client.model_copy(
                update={"on_overflow": "chunk", "chunk": ChunkPolicy(max_tokens=40, overlap_tokens=0)}
            ),
        }
    )
    pairs = write_pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query": "chunk me", "documents": ["long document tokens " * 200]},
            *sample_pairs(2)[:1],
        ],
    )
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(chunking, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["over_cap"]["n_pairs"] >= 1, "the chunked document must classify over cap from its origin"
        assert document["passed"] is True, document["per_document"][:1]
        assert [entry for entry in document["per_document"] if not entry["over_cap"]], "the under-cap row gates"
    finally:
        engine.stop()


def test_stage2_raises_a_typed_error_on_a_short_reference(tmp_path: Path) -> None:
    """A reference that emits fewer rows (or fewer scores) than the pairs file is a typed error, never a
    silent truncation or an untyped zip failure.  The mutations fire the typed raises (the layout mirrors the
    fixture's so the reference subprocess runs at all), asserted by their messages."""
    from rcp_ndcg_vllm.errors import HarnessError

    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    full = (RECIPES / "fixture-rerank-pointwise" / "reference.py").read_text(encoding="utf-8")
    scores_line = 'rows.append({"index": index, "scores": [score(folded, document) for document in row["documents"]]})'
    mutations = {
        "fewer-rows": (
            full.replace("for index, row in enumerate(pairs):", "for index, row in enumerate(pairs[:1]):"),
            "silently drop the later rows",
        ),
        "short-scores": (
            full.replace(scores_line, scores_line.replace("]})", "][:-1]})")),
            "scores align to the documents as given",
        ),
    }
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        for name, (source, message) in mutations.items():
            # The fixture layout, mirrored: <scratch>/<name>/recipes/<id>/reference.py two levels under
            # <scratch>/<name>/, next to deterministic.py -- the mutations must reach the harness, not die
            # in the subprocess.
            directory = tmp_path / name / "recipes" / name
            directory.mkdir(parents=True)
            shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / name / "deterministic.py")
            (directory / "reference.py").write_text(source, encoding="utf-8")
            manifest = (RECIPES / "fixture-rerank-pointwise" / "recipe.yaml").read_text(encoding="utf-8")
            (directory / "recipe.yaml").write_text(_rebased(manifest, name), encoding="utf-8")
            shutil.copy(RECIPES / "fixture-rerank-pointwise" / "template.jinja", directory / "template.jinja")
            with pytest.raises(HarnessError, match=message):
                stage2_scores(load_recipe(directory), pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()


def test_stage2_vector_over_cap_carve_out_when_declared(tmp_path: Path) -> None:
    """The vector stage honours the declared deviation too: the texts the client cut are non-gating."""
    recipe = load("fixture-embed")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]})}
    )
    pairs = write_pairs(
        tmp_path / "pairs.jsonl",
        [{"query": "short", "documents": ["over the cap " * 200]}],
    )
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(deviating, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        assert document["over_cap"]["n_pairs"] >= 1, "the long document must be cut by the client"
        assert document["over_cap"]["known_deviation"] is True
    finally:
        engine.stop()


def test_the_public_run_orchestrator_writes_the_report(tmp_path: Path) -> None:
    """The public `run()` runs the stages, writes equivalence.json and EQUIVALENCE.md, and returns the verdict."""
    from rcp_ndcg_vllm.equivalence import run

    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    out = tmp_path / "out"
    document = run(recipe, base_url=None, pairs_path=str(pairs), out_dir=str(out), stages=[1])
    assert document["passed"] is True, document["stage1"]["failures"][:1]
    report = out / "equivalence.json"
    markdown = out / "EQUIVALENCE.md"
    assert report.is_file() and markdown.is_file()
    assert "Stage 1" in markdown.read_text(encoding="utf-8")


def test_stage3_metrics_compares_served_against_reference(tmp_path: Path) -> None:
    """Stage 3 shells out to `rcp-ndcg eval score`; identical rankings give delta 0 and a pass."""
    pytest.importorskip("rcp_ndcg")
    from rcp_ndcg.data import Rankings

    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    for system in ("served", "reference"):
        Rankings.from_orders({"q1": ["a", "b", "c"], "q2": ["b", "a", "c"]}, system=system).save(
            rankings_dir / f"toy.{system}.jsonl"
        )
    dataset = [
        {
            "query_id": "q1",
            "query": "q1",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 1.0, "b": 0.5, "c": 0.1},
        },
        {
            "query_id": "q2",
            "query": "q2",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 0.2, "b": 0.9, "c": 0.0},
        },
    ]
    (rankings_dir / "toy.dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8"
    )
    gates = ResolvedGates(
        prob_p99_abs=0.02,
        prob_max_abs=0.05,
        logit_rel_abs=0.05,
        cos_max_abs=0.01,
        vec_min_cosine=0.999,
        tau_min=0.98,
        metrics_max_abs=2e-3,
        embed_dtype="float16",
    )
    document = stage3_metrics(rankings_dir, gates)
    assert document["passed"] is True


class CountingWords:
    """A whitespace-word tokenizer that counts its calls (the sampler's TokenizerAdapter surface)."""

    def __init__(self) -> None:
        self.calls = 0
        self.char_work = 0

    def count(self, text: str) -> int:
        """Tokens of ``text`` -- one per word -- and the recorded re-tokenization work, in characters."""
        self.calls += 1
        self.char_work += len(text)
        return len(text.split()) or 1


def test_over_length_padding_is_bounded_and_over_budget() -> None:
    """The sampler pads to >= 2x the budget in whole words with a RUNTIME BOUND (the hang fix).

    A 32768-token budget once re-tokenized the growing text at every step (one count per ~7 removed
    words), and one stage-1 sample of 10 over-length texts sat in ``tokenizer.count`` for minutes.  The
    bound is what makes the six network-gated files finish: at most 8 measured passes, so the counted
    characters stay a small multiple of the padded length instead of quadratic.
    """
    from rcp_ndcg_vllm.equivalence.stages import _over_length

    for budget in (64, 2048, 32768):
        tokenizer = CountingWords()
        seed = "How fast does light travel in a vacuum?"
        text = _over_length(seed, budget, tokenizer, 2)
        assert tokenizer.count(text) >= budget * 2  # the contract: at least twice the budget, still text
        assert text.startswith(seed)  # seed preserved, padding appended
        assert all(word == "pad2" for word in text.split()[len(seed.split()) :])  # whole words of the marker
        assert tokenizer.calls <= 12, (budget, tokenizer.calls)  # bounded pass count, not a re-tokenizing loop
        assert tokenizer.char_work <= 24 * len(text), (budget, tokenizer.char_work, len(text))


class CeilingWords(CountingWords):
    """A whitespace-word tokenizer whose count saturates at a ceiling (an embedded truncation, recipe G5)."""

    def __init__(self, ceiling: int) -> None:
        super().__init__()
        self.ceiling = ceiling

    def count(self, text: str) -> int:
        """Tokens of ``text``, never more than the ceiling."""
        return min(super().count(text), self.ceiling)


def test_over_length_padding_refuses_a_counter_that_never_reaches_the_target() -> None:
    """A tokenizer whose count stops at a ceiling below twice the budget cannot yield an over-length sample.

    The bounded sampler must say so instead of returning a text it never measured over the target: such a
    sample would audit an uncut input as if it were over the cap (nothing passes silently).
    """
    from rcp_ndcg_vllm.equivalence.stages import _over_length
    from rcp_ndcg_vllm.errors import HarnessError

    tokenizer = CeilingWords(ceiling=1024)
    with pytest.raises(HarnessError, match=r"2048 tokens.*1024"):
        _over_length("How fast does light travel in a vacuum?", 1024, tokenizer, 0)
    assert tokenizer.calls <= 12  # the refusal comes after the bounded passes, not after a hang
