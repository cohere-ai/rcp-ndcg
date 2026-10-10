"""Stage 1 and stage 2 through the product's role clients: the captured wire, the audit and the gates.

The client's captured requests are what every stage-1 check audits (no harness-side re-derivation); stage 2
sends through the same clients with the recipe's real budget and gates the answers against the reference
subprocess's outputs.  The harness process never imports torch or transformers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.equivalence import fitting, stage1_prompts, stage2_scores
from rcp_ndcg_test.equivalence.gates import ResolvedGates
from rcp_ndcg_test.equivalence.metrics import stage3_metrics
from rcp_ndcg_vllm import load_recipe

from tests.conftest import RECIPES, TOKENIZER, hub_cache, sample_pairs, start_stub, write_pairs

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
    manifest = (RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8")
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
        (directory / "family.yaml").write_text(_rebased(manifest, name), encoding="utf-8")
        document = stage1_prompts(load_recipe(directory), pairs, REFERENCE_PYTHON, over_length_per_shape=1)
        assert document["render_check"]["passed"] is False, name
        assert document["render_check"]["failures"], name


def _stub_with_recipe(recipe: Any, *flags: str) -> Any:
    """The stub engine started with the recipe's own serve argv (its served chat template) plus flags."""
    from rcp_ndcg_vllm.recipe import serve_argv

    argv = serve_argv(recipe, port=0, served_model_name=recipe.id)
    argv = ["127.0.0.1" if value == "0.0.0.0" else value for value in argv[argv.index(recipe.model) + 1 :]]
    return start_stub("--tokenizer", str(TOKENIZER), *argv, *flags)


def _recipe_flags(recipe: Any) -> list[str]:
    """The recipe's serve argv flags (everything after the model), as the stub's own arguments."""
    from rcp_ndcg_vllm.recipe import serve_argv

    argv = serve_argv(recipe, port=0, served_model_name=recipe.id)
    return ["127.0.0.1" if value == "0.0.0.0" else value for value in argv[argv.index(recipe.model) + 1 :]]


def test_the_stub_counts_a_rerank_pairs_rendered_prompt(tmp_path: Path) -> None:
    """The CPU stub's ``/rerank`` usage counts the served chat template's render of each pair, not the bare
    spans: stage 1's prompt-token probe passes against it for a rerank recipe with a served template, and a
    stub that counts the spans (the pre-fix behaviour) fails the probe -- which is what pins the stub's
    count (the probe is the only check that reads a rerank engine's usage)."""
    recipe = load("fixture-rerank-pointwise")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    engine = _stub_with_recipe(recipe)
    try:
        document = stage1_prompts(recipe, pairs, None, base_url=engine.base_url, over_length_per_shape=1)
    finally:
        engine.stop()
    check = document["engine_prompt_tokens_check"]
    assert check["status"] == "run" and check["passed"] is True, check["failures"][:2]
    assert check["checked"] > 0

    # The mutant: the pre-fix stub counted `count(query) + count(document)` per pair (no template).
    source = (Path(__file__).resolve().parent / "stub_engine.py").read_text(encoding="utf-8")
    old = 'if not path:\n            return f"{query} {document}"'
    assert old in source
    mutant_source = source.replace(old, 'if path or True:\n            return f"{query} {document}"')
    mutant = tmp_path / "stub_engine_mutant.py"
    mutant.write_text(mutant_source, encoding="utf-8")
    # The stub puts its own ``fixtures/`` directory on the path for the deterministic helpers: mirror it.
    (tmp_path / "fixtures").mkdir(exist_ok=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "fixtures" / "deterministic.py")
    from tests.conftest import StubEngine

    process = subprocess.Popen(
        [sys.executable, str(mutant), "--port", "0", "--tokenizer", str(TOKENIZER), *_recipe_flags(recipe)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    line = process.stdout.readline().decode() if process.stdout else ""
    assert line.startswith("RCPS_STUB_PORT="), line
    drifted = StubEngine(process, int(line.strip().split("=", 1)[1]))
    try:
        document = stage1_prompts(recipe, pairs, None, base_url=drifted.base_url, over_length_per_shape=1)
    finally:
        drifted.stop()
    check = document["engine_prompt_tokens_check"]
    assert check["passed"] is False and check["failures"], check


def test_stage1_render_check_covers_every_document_of_a_row(tmp_path: Path) -> None:
    """Review A4: the render comparison used to compare only the first text per (row, shape), so a row's
    second and later documents were never held to the reference.  A reference that diverges on the SECOND
    document alone must fail the check, and the failure must name that document (with the pre-fix harness the
    second document never reached the reference and the check passed)."""
    source = (RECIPES / "fixture-embed" / "reference.py").read_text(encoding="utf-8")
    manifest = (RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8")
    directory = tmp_path / "second-document" / "recipes" / "second-document"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "second-document" / "deterministic.py")
    old = "return prefix + document + suffix"
    assert old in source
    mutated = source.replace(old, 'return prefix + document + suffix + (" X" if "MARK" in document else "")')
    (directory / "reference.py").write_text(mutated, encoding="utf-8")
    (directory / "family.yaml").write_text(_rebased(manifest, "second-document"), encoding="utf-8")
    rows = [{"query": "the query", "documents": ["a plain document", "a document MARK here"]}]
    pairs = write_pairs(tmp_path / "pairs.jsonl", rows)
    document = stage1_prompts(load_recipe(directory), pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    check = document["render_check"]
    assert check["rows"] == len(rows[0]["documents"]), "the reference must be asked to render every document"
    assert check["passed"] is False, check
    assert [failure.get("document") for failure in check["failures"]] == [1], check["failures"]


def test_stage1_audits_the_clients_settled_query(tmp_path: Path) -> None:
    """The settle-once rule, audited on the captured wire: one query span per row, within its declared share."""
    from rcp_ndcg_test.equivalence.wire import role_client

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
    assert tokenizer.count(str(queries[0])) <= recipe.client.get("query_max_tokens")


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
    from rcp_ndcg_test.equivalence import stages as stages_module

    return stages_module._engine_tokenize_check(recipe, probe, tokenizer, "http://engine")


def test_stage1_without_a_reference_python_reports_not_run(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert document["render_check"]["status"] == "not_run"
    assert document["anchor_check"]["passed"] is True


@pytest.mark.parametrize("deviation", ["anchor_drop_over_cap", "over_cap_cut_differs"])
def test_stage1_carves_over_cap_rows_out_of_the_render_check_when_declared(tmp_path: Path, deviation: str) -> None:
    """With an over-cap deviation declared, over-cap pairs-file rows are reported non-gating in stage 1 too."""
    recipe = load("fixture-rerank-pointwise")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": [deviation]})}
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


@pytest.mark.parametrize("deviation", ["anchor_drop_over_cap", "over_cap_cut_differs"])
def test_stage2_gates_only_under_cap_pairs_when_the_deviation_is_declared(tmp_path: Path, deviation: str) -> None:
    """With an over-cap deviation, over-cap pairs (the client's own census) are non-gating; the rest gate."""
    recipe = load("fixture-rerank-pointwise")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": [deviation]})}
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
            "client": {**recipe.client, "on_overflow": "chunk", "chunk": ChunkPolicy(max_tokens=40, overlap_tokens=0)},
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
    from rcp_ndcg_test.errors import HarnessError

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
            manifest = (RECIPES / "fixture-rerank-pointwise" / "family.yaml").read_text(encoding="utf-8")
            (directory / "family.yaml").write_text(_rebased(manifest, name), encoding="utf-8")
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


def test_a_cuda_reference_recipe_refuses_a_cpu_stage_2(tmp_path: Path) -> None:
    """GPU-E1: every reference ran on CPU.  A recipe that declares ``reference.device: cuda`` refuses a
    CPU reference run with the way out -- the wave runner gives each reference a GPU of its own, and a
    CPU run compares a bf16 engine against the wrong precision (or cannot run at all)."""
    from rcp_ndcg_test.equivalence import run
    from rcp_ndcg_test.errors import HarnessError

    base = load("fixture-embed")
    recipe = base.model_copy(update={"reference": base.reference.model_copy(update={"device": "cuda"})})
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    with pytest.raises(HarnessError, match="reference.device: cuda"):
        run(
            recipe,
            base_url="http://127.0.0.1:9/v1",  # never reached: the refusal fires first
            pairs_path=str(pairs),
            out_dir=str(tmp_path / "out"),
            stages=[2],
            reference_python=sys.executable,
            device="cpu",
        )
    # Stage 1 alone still runs: its render is tokenizer work, no model, no device need.
    document = run(
        recipe, base_url=None, pairs_path=str(pairs), out_dir=str(tmp_path / "out1"), stages=[1], device="cpu"
    )
    assert document["stage1"]["passed"] is True


def test_run_records_the_reference_device_and_its_gpu(tmp_path: Path) -> None:
    """The device the reference ran on (and the GPU it was pinned to) is recorded in the report document
    and in equivalence.json (GPU-E1: the wave's CPU references were invisible in the reports)."""
    import json

    from rcp_ndcg_test.equivalence import run

    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = run(
        recipe, base_url=None, pairs_path=str(pairs), out_dir=str(tmp_path / "out"), stages=[1], device="cuda",
        reference_gpu=5,
    )  # fmt: skip
    assert document["device"] == "cuda"
    assert document["reference_gpu"] == 5
    written = json.loads((tmp_path / "out" / "equivalence.json").read_text(encoding="utf-8"))
    assert written["device"] == "cuda" and written["reference_gpu"] == 5


def test_the_reference_subprocess_is_pinned_to_its_own_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reference subprocess runs with CUDA_VISIBLE_DEVICES set to its assigned GPU (never the
    engine's, which holds 90 % of its memory); without a pin nothing is set."""

    from rcp_ndcg_test.equivalence.reference import run_reference

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    entry = tmp_path / "env_reference.py"
    entry.write_text(
        "import argparse, json, os\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--mode"); p.add_argument("--pairs"); p.add_argument("--out")\n'
        'p.add_argument("--tokenizer"); p.add_argument("--device", default="cpu"); p.add_argument("--recipe")\n'
        "a = p.parse_args()\n"
        'json.dump({"cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}, open(a.out, "w"))\n'
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    document = run_reference(
        sys.executable,
        str(entry),
        mode="render",
        pairs_path=pairs,
        out_path=tmp_path / "out.json",
        tokenizer_spec="",
        recipe=load("fixture-embed"),
        cuda_visible_devices="3",
    )
    assert document["cuda"] == "3"
    unpinned = run_reference(
        sys.executable,
        str(entry),
        mode="render",
        pairs_path=pairs,
        out_path=tmp_path / "out2.json",
        tokenizer_spec="",
        recipe=load("fixture-embed"),
    )
    assert unpinned["cuda"] is None  # no pin: the child inherits the runner's environment


def test_the_public_run_orchestrator_writes_the_report(tmp_path: Path) -> None:
    """The public `run()` runs the stages, writes equivalence.json and EQUIVALENCE.md, and returns the verdict."""
    from rcp_ndcg_test.equivalence import run

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
    from rcp_ndcg_test.equivalence.stages import _over_length

    for budget in (64, 2048, 32768):
        tokenizer = CountingWords()
        seed = "How fast does light travel in a vacuum?"
        text = _over_length(seed, budget, tokenizer, 2)
        assert tokenizer.count(text) >= budget * 2  # the contract: at least twice the budget, still text
        assert text.startswith(seed)  # seed preserved, padding appended
        assert all(word == "pad2" for word in text.split()[len(seed.split()) :])  # whole words of the marker
        assert tokenizer.calls <= 12, (budget, tokenizer.calls)  # bounded pass count, not a re-tokenizing loop
        assert tokenizer.char_work <= 24 * len(text), (budget, tokenizer.char_work, len(text))


def test_over_length_padding_without_a_budget_pads_past_twice_the_default() -> None:
    """A recipe without ``client.max_tokens`` still gets an over-length sample: the sampler's default budget
    is 128 tokens, so the sample reaches at least 256 (twice the default, as with a declared budget)."""
    from rcp_ndcg_test.equivalence.stages import _over_length

    tokenizer = CountingWords()
    text = _over_length("How fast does light travel in a vacuum?", None, tokenizer, 0)
    assert tokenizer.count(text) >= 256
    assert tokenizer.calls <= 12


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
    from rcp_ndcg_test.equivalence.stages import _over_length
    from rcp_ndcg_test.errors import HarnessError

    tokenizer = CeilingWords(ceiling=1024)
    with pytest.raises(HarnessError, match=r"2048 tokens.*1024"):
        _over_length("How fast does light travel in a vacuum?", 1024, tokenizer, 0)
    assert tokenizer.calls <= 12  # the refusal comes after the bounded passes, not after a hang


def _with_client(recipe: Any, **updates: Any) -> Any:
    """``recipe`` with its client config updated (e.g. another wire ``request_shape``): the plain client
    block merged; the product's endpoint model validates the block when the harness's role client reads it."""
    return recipe.model_copy(update={"client": {**recipe.client, **updates}})


@pytest.mark.parametrize("recipe_id", ["fixture-embed", "fixture-embed-cls", "fixture-embed-marker"])
def test_stage1_audits_token_ids_bodies_on_the_sent_ids(tmp_path: Path, recipe_id: str) -> None:
    """G1: a ``request_shape: token_ids`` client sends ``{"input": [[ids]]}``; the audit reads those ids as
    sent (they already carry the anchor edge and the post-processor's tokens) -- and a cut edge fails it."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = _with_client(load(recipe_id), request_shape="token_ids")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    audit = document["anchor_check"]
    assert audit["passed"] is True, audit["failures"][:1]
    assert audit["checked"] > 0
    tokenizer = load_tokenizer(str(TOKENIZER))
    sent = tokenizer.ids("doc: Paris is the capital of France. [END]", add_special_tokens=True)
    if recipe_id == "fixture-embed-cls":
        sent = tokenizer.ids(tokenizer.special_text("cls") + "doc: Paris is the capital.")
    elif recipe_id == "fixture-embed-marker":
        sent = tokenizer.ids("doc: Paris is the capital." + tokenizer.special_text("sep"))
    whole = {"rows": [{"shapes": {"document": {"texts": [sent]}}, "cuts": 0, "over_cap": False}]}
    assert stages_module._anchor_check(recipe, whole, tokenizer)["passed"] is True
    cut = sent[1:] if recipe_id == "fixture-embed-cls" else sent[:-1]
    broken = {"rows": [{"shapes": {"document": {"texts": [cut]}}, "cuts": 0, "over_cap": False}]}
    assert stages_module._anchor_check(recipe, broken, tokenizer)["passed"] is False


def test_stage1_render_check_compares_token_ids_bodies_on_the_reference_ids(tmp_path: Path) -> None:
    """G1: a ``token_ids`` client sends ids, the reference renders text: the render check compares the sent
    ids with the reference text's ids (the product tokenizer, the shape's ``add_special_tokens`` flag), so a
    faithful reference passes stage 1 and a one-character divergent one still fails (``dog:`` for ``doc:``;
    a doubled space would not do: this tokenizer splits on whitespace, so its ids -- what the engine reads --
    are the same)."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    recipe = _with_client(load("fixture-embed"), request_shape="token_ids")
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    render = document["render_check"]
    assert render["status"] == "run" and render["rows"] > 0
    assert render["passed"] is True, render["failures"][:1]
    assert document["passed"] is True
    source = (RECIPES / "fixture-embed" / "reference.py").read_text(encoding="utf-8")
    directory = tmp_path / "divergent" / "recipes" / "divergent"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "divergent" / "deterministic.py")
    assert 'PREFIX = "doc: "' in source
    (directory / "reference.py").write_text(source.replace('PREFIX = "doc: "', 'PREFIX = "dog: "'), encoding="utf-8")
    manifest = (RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8")
    (directory / "family.yaml").write_text(_rebased(manifest, "divergent"), encoding="utf-8")
    divergent = _with_client(load_recipe(directory), request_shape="token_ids")
    render = stage1_prompts(divergent, pairs, REFERENCE_PYTHON, over_length_per_shape=1)["render_check"]
    assert render["passed"] is False
    failure = render["failures"][0]
    assert failure["served_ids_head"] != failure["reference_ids_head"]


def test_engine_tokenize_check_does_not_post_token_ids_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """G1: a ``token_ids`` body is read by the engine as sent -- the /tokenize check posts no text for it and
    reports ``not_run`` (never a vacuous pass)."""
    import httpx

    from rcp_ndcg.data.tokenizer import load_tokenizer

    def no_post(url: str, **_: Any) -> Any:
        raise AssertionError("a token_ids body has no text to /tokenize")

    monkeypatch.setattr(httpx, "post", no_post)
    recipe = _with_client(load("fixture-embed"), request_shape="token_ids")
    probe = {"rows": [{"shapes": {"document": {"texts": [[5, 6, 7]]}}, "cuts": 0, "over_cap": False}]}
    check = stages_module_check(recipe, probe, load_tokenizer(str(TOKENIZER)))
    assert check is not None and check["status"] == "not_run" and check["passed"] is None


def test_stage1_audits_messages_bodies_and_fails_an_audit_that_checked_nothing(tmp_path: Path) -> None:
    """G2: a ``messages`` client's bodies extract as their message texts (media parts as placeholders), so the
    audit checks every captured input; an audit that checked zero inputs fails instead of passing vacuously."""
    from rcp_ndcg_core.content import TEXT_JOIN
    from rcp_ndcg_test.equivalence import stages as stages_module
    from rcp_ndcg_test.equivalence.wire import Capture

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = _with_client(load("fixture-embed"), request_shape="messages")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    audit = document["anchor_check"]
    assert audit["checked"] == len(sample_pairs()[0]["documents"]) + 2, audit
    assert audit["passed"] is True, audit["failures"][:1]
    # A batch: one conversation per item (a list of messages alone is ONE conversation, one embedding).
    body = {
        "messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "doc: a caption"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "text", "text": " [END]"},
                    ],
                }
            ],
            [{"role": "user", "content": "doc: plain [END]"}],
            [{"role": "user", "content": ["doc: bare", " [END]"]}],  # bare strings: text parts, as vLLM reads them
        ]
    }
    capture = Capture(recipe)
    capture.exchanges = [{"url": "http://engine/v1/embeddings", "status": 200, "request_body": body}]
    texts = capture.texts(capture.exchanges[0])
    # A message's text parts join as the engine joins them (vLLM's chat_utils: "\n"), the product's TEXT_JOIN.
    assert TEXT_JOIN == "\n"
    assert texts["input"] == ["doc: a caption\n [END]", "doc: plain [END]", "doc: bare\n [END]"]
    assert texts["media"] == [["image_url"], [], []]
    one = capture.texts({"request_body": {"messages": body["messages"][1] + body["messages"][2]}})
    assert one["input"] == ["doc: plain [END]\ndoc: bare\n [END]"], "one conversation is one input"
    assert stages_module._captured_heads(capture, [])["first"][0]["media"] == [["image_url"], [], []]  # reported
    empty = {"rows": [{"shapes": {"document": {"texts": []}}, "cuts": 0, "over_cap": False}]}
    nothing = stages_module._anchor_check(recipe, empty, load_tokenizer(str(TOKENIZER)))
    assert nothing["checked"] == 0 and nothing["passed"] is False
    assert nothing["failures"][0]["check"] == "nothing_checked"


def _byte_level_bpe(path: Path) -> Path:
    """A small byte-level BPE tokenizer (GPT-2's pre-tokenizer), trained in-test and written to ``path``: a
    space joins the word after it (``" document"`` is one token), so a frame ending in a space merges into
    the content that follows it -- the mergey tokenizers of the served models, offline."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore[assignment]
    backend.decoder = decoders.ByteLevel()  # type: ignore[assignment]
    texts = [row["query"] for row in sample_pairs()] + [text for row in sample_pairs() for text in row["documents"]]
    corpus = [f"doc: {text}" for text in texts]  # the framed texts: " document" is learned as one token
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<|cls|>", "<|sep|>", "<|end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(corpus * 20, trainer)
    backend.save(str(path))
    return path


def test_stage1_head_edge_tolerates_the_join_merge_on_a_byte_level_bpe(tmp_path: Path) -> None:
    """G3: on a byte-level BPE the head's trailing space merges into the first content token (``"doc: "`` ends
    in ``Ġ`` alone, ``"doc: document"`` reads ``Ġdocument``). The ``anchor: first`` audit compares the head
    up to that join -- the frame's whitespace the fit verifies on the assembled render -- so a whole head
    passes, and a head whose own tokens changed still fails."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    from rcp_ndcg.data.tokenizer import load_tokenizer

    bpe = _byte_level_bpe(tmp_path / "tokenizer.json")
    tokenizer = load_tokenizer(str(bpe))
    cls = tokenizer.special_text("cls")
    assert tokenizer.ids(cls + "doc: ")[:3] == tokenizer.ids(cls + "doc: document")[:3]
    assert tokenizer.ids(cls + "doc: ")[3] != tokenizer.ids(cls + "doc: document")[3]  # the join merged
    recipe = _with_client(load("fixture-embed-cls"), tokenizer=str(bpe))
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    for request_shape in ("text", "token_ids"):
        shaped = _with_client(recipe, request_shape=request_shape)
        audit = stage1_prompts(shaped, pairs, None, over_length_per_shape=2)["anchor_check"]
        assert audit["passed"] is True, (request_shape, audit["failures"][:1])
        assert audit["checked"] == len(sample_pairs()[0]["documents"]) + 2
    # no cls, another head, a cut head, a head that lost its colon (its join then reads Ġdocument)
    for broken in ("doc: document 0", cls + "dog: document 0", cls + "do", cls + "doc document 0"):
        for body in (broken, tokenizer.ids(broken)):
            probe = {"rows": [{"shapes": {"document": {"texts": [body]}}, "cuts": 0, "over_cap": False}]}
            check = stages_module._anchor_check(recipe, probe, tokenizer)
            assert check["passed"] is False, (broken, type(body).__name__)


_QWEN_STYLE_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
    r"|\s+(?!\S)|\s+"
)
"""The pre-tokenizer split of the Qwen-family byte-level BPEs: one punctuation character joins the LETTERS
after it (the second alternative), so a head ending in ``:`` merges its last token into the content."""

_JOIN_CONTENTS = ("Paris is big", "über alles", "中国的首都", "123 apples", "(parens)", "\nnewline first",
                  " leading space", "🙂 emoji", "Ünïcödé", "-dash", "'s owner", "")  # fmt: skip


def _qwen_style_bpe(path: Path) -> Path:
    """A small byte-level BPE with the Qwen-family pre-tokenizer split, trained in-test on framed texts; its
    post-processor prepends ``<|cls|>`` (so an ``add_special_tokens: true`` head edge opens with it)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore[assignment]
        [
            pre_tokenizers.Split(Regex(_QWEN_STYLE_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()  # type: ignore[assignment]
    documents = [text for row in sample_pairs() for text in row["documents"]]
    heads = ("Query:", "Query: ", "doc: ", "Instruct: x\nQuery:")
    corpus = [head + content for head in heads for content in (*_JOIN_CONTENTS, *documents)]
    trainer = trainers.BpeTrainer(
        vocab_size=800,
        special_tokens=["<|cls|>", "<|sep|>", "<|end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(corpus * 30, trainer)
    cls_id = backend.token_to_id("<|cls|>")
    backend.post_processor = processors.TemplateProcessing(  # type: ignore[assignment]
        single="<|cls|> $A", special_tokens=[("<|cls|>", cls_id)]
    )
    backend.save(str(path))
    return path


@pytest.mark.parametrize("head", ["Query:", "Instruct: x\nQuery:", "Query: ", "doc: "])
def test_stage1_head_edge_is_the_heads_own_tokens_in_the_assembled_render(tmp_path: Path, head: str) -> None:
    """G3: the ``anchor: first`` edge is the tokens lying wholly inside the head's characters of the assembled
    render, so a head whose last token merges into the content -- whitespace on a GPT-2 BPE, or ``:`` under
    the Qwen-family split (``:Paris``) -- passes on text and token_ids bodies for any content. A text body
    that lost or changed a head character fails; a token_ids body (no text on the wire) fails when a head
    token no content can merge away changed -- the merging join itself is the render check's to compare."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    from rcp_ndcg.data.tokenizer import load_tokenizer

    bpe = _qwen_style_bpe(tmp_path / "tokenizer.json")
    tokenizer = load_tokenizer(str(bpe))
    base = load("fixture-embed-cls")
    spec = {**base.client["template"], "document": [{"fixed": head}, {"content": "document"}]}
    spec["add_special_tokens"] = True  # the post-processor's <|cls|> opens the edge, then the head's tokens
    recipe = _with_client(base, tokenizer=str(bpe), template=spec)
    template = fitting.client_template(recipe)
    assert tokenizer.ids("x", add_special_tokens=True)[0] == tokenizer.special_id("cls")

    def audit(body: str | list[int]) -> dict[str, Any]:
        probe = {"rows": [{"shapes": {"document": {"texts": [body]}}, "cuts": 0, "over_cap": False}]}
        return stages_module._anchor_check(recipe, probe, tokenizer)

    for content in _JOIN_CONTENTS:
        render = template.render("document", tokenizer, document=content)
        for body in (render, tokenizer.ids(render, add_special_tokens=True)):
            check = audit(body)
            assert check["passed"] is True, (content, type(body).__name__, check["failures"][:1])
    rendered_head = template.render("document", tokenizer, document="")
    assert audit(rendered_head[:-1] + "Paris")["passed"] is False  # the head's last character, cut
    changed = rendered_head.replace("Query", "Quarry").replace("doc", "dog") + "Paris"
    for body in (changed, tokenizer.ids(changed, add_special_tokens=True)):
        assert audit(body)["passed"] is False, type(body).__name__


def test_stage1_token_ids_head_edge_is_common_to_every_join_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G3: a token_ids body's head edge is the head tokens wholly inside the head in the assembled render with
    EVERY join probe (their common prefix), and a token ending exactly at the head's end is inside it."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    from rcp_ndcg.data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(str(_byte_level_bpe(tmp_path / "tokenizer.json")))
    assert stages_module._stable_head_tokens(tokenizer, "doc:") == tokenizer.ids("doc:")  # ":" ends at the end
    assert stages_module._stable_head_tokens(tokenizer, "doc: ") == tokenizer.ids("doc:")  # the space merges
    monkeypatch.setattr(stages_module, "_JOIN_PROBES", (".",))
    assert stages_module._stable_head_tokens(tokenizer, "doc: ") == tokenizer.ids("doc: ")  # "." keeps the space
    monkeypatch.setattr(stages_module, "_JOIN_PROBES", (".", "a"))
    assert stages_module._stable_head_tokens(tokenizer, "doc: ") == tokenizer.ids("doc:")  # "a" takes it


def test_stage1_marker_audit_is_not_masked_by_the_post_processor() -> None:
    """A marker the client dropped is missing even when the shape's post-processor appends the same special
    (``add_special_tokens: true``): the audit counts markers in the sent content, without the post-processor's
    tokens -- a text body tokenized without them, a token_ids body with them stripped from its edges."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    from rcp_ndcg.data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(str(TOKENIZER))
    base = load("fixture-embed-marker")
    spec = {
        **base.client["template"],
        "document": [{"fixed": "doc: "}, {"content": "document"}, {"fixed": "{special:end}"}],
    }
    spec["anchor_markers"] = ["end"]
    spec["add_special_tokens"] = True
    recipe = _with_client(base, template=spec)
    end = tokenizer.special_id("end")
    assert tokenizer.ids("doc: x", add_special_tokens=True)[-1] == end  # the post-processor appends it too

    def audit(body: str | list[int]) -> bool:
        probe = {"rows": [{"shapes": {"document": {"texts": [body]}}, "cuts": 0, "over_cap": False}]}
        return stages_module._anchor_check(recipe, probe, tokenizer)["passed"]

    whole = "doc: Paris is the capital." + tokenizer.special_text("end")
    cut = "doc: Paris is the"  # the client dropped the marker
    assert audit(whole) is True
    assert audit(tokenizer.ids(whole, add_special_tokens=True)) is True
    assert audit(cut) is False
    assert audit(tokenizer.ids(cut, add_special_tokens=True)) is False


_CHAT_TEMPLATE = (
    "{%- for message in messages -%}doc: {% for part in message.content -%}"
    "{%- if part.type == 'text' %}{{ part.text }}{% endif -%}{%- endfor %} [END]{%- endfor -%}"
)
"""A served chat template that frames one user turn exactly as ``fixture-embed`` declares it (``doc: ... [END]``)."""


def _messages_recipe(tmp_path: Path, chat_template: str, *, client_extra: str = "") -> Any:
    """``fixture-embed`` on the ``messages`` route, served with ``chat_template`` (the engine's frame); the
    ``client_extra`` YAML lines join the client block."""
    directory = tmp_path / "scratch" / "recipes" / "embed-messages"
    directory.mkdir(parents=True)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "scratch" / "deterministic.py")
    shutil.copy(RECIPES / "fixture-embed" / "reference.py", directory / "reference.py")
    (directory / "chat.jinja").write_text(chat_template, encoding="utf-8")
    manifest = _rebased((RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8"), "embed-messages")
    manifest = manifest.replace("  chat_template: null", "  chat_template: chat.jinja").replace(
        "  api: openai_embeddings", "  api: openai_embeddings\n  request_shape: messages" + client_extra
    )
    (directory / "family.yaml").write_text(manifest, encoding="utf-8")
    return load_recipe(directory)


def test_stage1_messages_route_is_framed_once_by_the_served_chat_template(tmp_path: Path) -> None:
    """H5: the messages route sends the content; the served chat template, rendered over the captured
    conversations as the engine renders them, must equal the declared template's render exactly -- one frame.
    A template that frames the turn twice fails the check (as the client's old framed messages did)."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    recipe = _messages_recipe(tmp_path, _CHAT_TEMPLATE)
    document = stage1_prompts(recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=2)
    check = document["template_render_check"]
    assert check["passed"] is True and check["checked"] > 0, check["failures"][:1]
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["passed"] is True

    twice = _messages_recipe(tmp_path / "twice", _CHAT_TEMPLATE.replace("doc: ", "doc: doc: "))
    check = stage1_prompts(twice, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["passed"] is False
    failure = check["failures"][0]
    assert failure["engine_head"].startswith("doc: doc: ") and failure["declared_head"].count("doc: ") == 1


def test_stage1_messages_route_renders_the_declared_generation_prompt(tmp_path: Path) -> None:
    """A served chat template whose closing frame renders only under ``add_generation_prompt`` (the
    assistant header of Qwen3-VL-Embedding's template): the engine renders each captured request with the
    flag it carries (vLLM's default false when absent), so without the declared flag the served render
    misses the declared tail and the check fails; with ``add_generation_prompt: true`` it is the frame."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    template = _CHAT_TEMPLATE.replace(" [END]", "{% if add_generation_prompt %} [END]{% endif %}")
    without = _messages_recipe(tmp_path / "without", template)
    check = stage1_prompts(without, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["passed"] is False and not check["failures"][0]["engine_head"].endswith("[END]")
    declared = _messages_recipe(tmp_path / "declared", template, client_extra="\n  add_generation_prompt: true")
    assert declared.client.get("add_generation_prompt") is True
    check = stage1_prompts(declared, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["passed"] is True and check["checked"] > 0, check["failures"][:1]


def _hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str], absent: tuple[str, ...] = ()
) -> None:
    """An offline Hub cache holding ``fixtures/DenseEmbedder`` at fixture-embed's pinned revision with ``files``
    (``absent``: the files it records as not in the repository)."""
    hub_cache(
        tmp_path, monkeypatch, "fixtures/DenseEmbedder", "0123456789abcdef0123456789abcdef01234567", files, absent
    )


def test_stage1_messages_route_render_checks_the_checkpoints_own_chat_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``serve.chat_template`` the engine frames the messages route with the checkpoint's own chat
    template at the pinned revision: the harness reads that file (the Hub cache, or a pinned fetch) and
    render-checks against it -- run and passed when it frames the declared template once, failed when it does
    not, and failed (never passed, never silently skipped) when the file cannot be resolved."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    recipe = _messages_recipe(tmp_path / "own", _CHAT_TEMPLATE)
    recipe = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"chat_template": None})})
    _hub_cache(tmp_path / "good", monkeypatch, {"chat_template.jinja": _CHAT_TEMPLATE})
    check = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["status"] == "run" and check["passed"] is True and check["checked"] > 0, check
    assert check["template"] == "fixtures/DenseEmbedder@0123456789abcdef0123456789abcdef01234567:chat_template.jinja"
    assert check["template_sha256"] == hashlib.sha256(_CHAT_TEMPLATE.encode("utf-8")).hexdigest()

    _hub_cache(tmp_path / "twice", monkeypatch, {"chat_template.jinja": _CHAT_TEMPLATE.replace("doc: ", "doc: doc: ")})
    assert stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]["passed"] is False

    config = json.dumps({"chat_template": _CHAT_TEMPLATE})
    absent = ("chat_template.jinja", "chat_template.json")
    _hub_cache(tmp_path / "config", monkeypatch, {"tokenizer_config.json": config}, absent)
    check = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["passed"] is True and check["template"].endswith(":tokenizer_config.json"), check

    # Not cached and no Hub to ask is not "absent": the checkpoint may ship a chat_template.jinja, so rendering
    # tokenizer_config.json's template in its place would check the wrong template.
    _hub_cache(tmp_path / "unknown", monkeypatch, {"tokenizer_config.json": config})
    check = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["status"] == "unresolved" and check["passed"] is False, check
    assert "chat_template.jinja" in check["failures"][0]["note"], check

    _hub_cache(tmp_path / "empty", monkeypatch, {})
    check = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["status"] == "unresolved" and check["passed"] is False, check


def test_stage2_on_the_messages_route_through_the_stub(tmp_path: Path) -> None:
    """Stage 2 on the ``messages`` route: the client sends each item's content as one conversation, and the
    stub, like vLLM's chat path, frames it with the served chat template (the request's
    ``add_generation_prompt``, false by default) before embedding. A template that frames the declared turn
    once passes the gates; one that frames it twice changes what the model reads, and the vectors fail."""
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    for name, chat_template, passes in (
        ("once", _CHAT_TEMPLATE, True),
        ("twice", _CHAT_TEMPLATE.replace("doc: ", "doc: doc: "), False),
    ):
        recipe = _messages_recipe(tmp_path / name, chat_template)
        assert recipe._dir is not None
        engine = start_stub("--tokenizer", str(TOKENIZER), "--chat-template", str(recipe._dir / "chat.jinja"))
        try:
            document = stage2_scores(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
        finally:
            engine.stop()
        assert document["passed"] is passes, (name, document["per_vector"][:2])
        assert document["n_vectors"] == sum(len(row["documents"]) for row in sample_pairs()[:2])


def test_a_checkpoint_template_read_error_is_unresolved_never_a_fall_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an absent file falls through to the next template source: any other failure reading
    ``chat_template.jinja`` (a broken cache, a refused download) fails the check as unresolved, never
    renders ``tokenizer_config.json``'s template in its place."""
    import huggingface_hub

    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    recipe = _messages_recipe(tmp_path / "own", _CHAT_TEMPLATE)
    recipe = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"chat_template": None})})
    _hub_cache(
        tmp_path / "config", monkeypatch, {"tokenizer_config.json": json.dumps({"chat_template": _CHAT_TEMPLATE})}
    )
    real = huggingface_hub.hf_hub_download

    def broken(repo_id: str, filename: str, **kwargs: Any) -> str:
        if filename == "chat_template.jinja":
            raise OSError("the cache's blob is unreadable")
        return real(repo_id, filename, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", broken)
    check = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)["template_render_check"]
    assert check["status"] == "unresolved" and check["passed"] is False, check
    assert "chat_template.jinja" in check["failures"][0]["note"]


def test_the_stage_2_defaults_are_the_published_ones_and_an_override_wins() -> None:
    """``resolve_gates`` on a recipe with no ``gates`` block is the published tolerance set -- the release's
    guards. A single relaxed default is invisible today (mutating ``_TAU_MIN`` 0.98 -> 0.5 survives every
    gate-referencing test), so every default is pinned here in one table; a declared field overrides only
    itself."""
    from rcp_ndcg_test.equivalence.gates import resolve_gates

    recipe = load("fixture-embed")
    assert recipe.gates.model_dump(exclude_none=True) == {}, "fixture-embed declares no gate overrides"
    gates = resolve_gates(recipe)
    assert gates.model_dump() == {
        "prob_p99_abs": 0.02,
        "prob_max_abs": 0.05,
        "logit_rel_abs": 0.05,
        "cos_max_abs": 0.01,
        "vec_min_cosine": 1.0 - 1e-3,
        "tau_min": 0.98,
        "metrics_max_abs": 2e-3,
        "embed_dtype": "float16",
    }
    declared = recipe.model_copy(update={"gates": recipe.gates.model_copy(update={"tau_min": 0.5})})
    overridden = resolve_gates(declared)
    assert overridden.tau_min == 0.5
    assert overridden.model_dump(exclude={"tau_min"}) == gates.model_dump(exclude={"tau_min"}), (
        "one declared field overrides itself, not its neighbours"
    )


def test_a_width_mismatch_gates_stage_2_with_both_widths() -> None:
    """A served vector whose width differs from the reference's is a named gate failure carrying both
    widths -- never a ValueError out of the cosine (MRL's declared dimension makes this live: a served
    truncation against a full-width reference, or the other way round)."""
    from rcp_ndcg_test.equivalence import stages as stages_module
    from rcp_ndcg_test.equivalence.gates import resolve_gates

    recipe = load("fixture-embed")
    gates = resolve_gates(recipe)
    per_vector: list[dict[str, Any]] = []
    stages_module._compare_shape(recipe, [[[1.0, 0.0]]], [[1.0, 0.0, 0.0]], 0, "document", per_vector, gates)
    assert per_vector and per_vector[0]["within"] is False
    assert per_vector[0]["cosine"] is None
    assert "2-wide" in per_vector[0]["note"] and "3-wide" in per_vector[0]["note"]
    summary = stages_module._vector_summary(recipe, per_vector, gates)
    assert summary["passed"] is False


def test_the_mean_anchor_audit_checks_the_content_and_the_fixed_edges() -> None:
    """``anchor: mean`` has no anchor token, so the audit checks what a cut must keep: the declared fixed
    edges (the shape's head here) and at least one content token between them.  A body missing the head, or
    one whose cut emptied the content, fails -- the audit is not vacuous for the shipped mean-anchor
    recipes (pplx-embed-v1, embeddinggemma-2, topk-embed-v1, pplx-embed-v2-late)."""
    from rcp_ndcg_test.equivalence import stages as stages_module
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of

    recipe = load("fixture-multi-vector")  # template: "doc: " + content, anchor: mean
    assert fitting.client_template(recipe).anchor == "mean"
    tokenizer = tokenizer_of(recipe)
    good = {"rows": [{"shapes": {"document": {"texts": ["doc: paris"]}}}]}
    check = stages_module._anchor_check(recipe, good, tokenizer)
    assert check["passed"] is True and check["checked"] == 1, check["failures"]
    headless = {"rows": [{"shapes": {"document": {"texts": ["paris"]}}}]}
    check = stages_module._anchor_check(recipe, headless, tokenizer)
    assert check["passed"] is False and check["failures"][0]["check"] == "mean_head"
    emptied = {"rows": [{"shapes": {"document": {"texts": ["doc: "]}}}]}
    check = stages_module._anchor_check(recipe, emptied, tokenizer)
    assert check["passed"] is False and check["failures"][0]["check"] == "mean_content"
