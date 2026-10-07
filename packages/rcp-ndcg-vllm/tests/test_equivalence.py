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


def _with_client(recipe: Any, **updates: Any) -> Any:
    """``recipe`` with its client config updated (e.g. another wire ``request_shape``), revalidated."""
    client = type(recipe.client)(**{**recipe.client.model_dump(), **updates})
    return recipe.model_copy(update={"client": client})


@pytest.mark.parametrize("recipe_id", ["fixture-embed", "fixture-embed-cls", "fixture-embed-marker"])
def test_stage1_audits_token_ids_bodies_on_the_sent_ids(tmp_path: Path, recipe_id: str) -> None:
    """G1: a ``request_shape: token_ids`` client sends ``{"input": [[ids]]}``; the audit reads those ids as
    sent (they already carry the anchor edge and the post-processor's tokens) -- and a cut edge fails it."""
    from rcp_ndcg_vllm.equivalence import stages as stages_module

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
    manifest = (RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8")
    (directory / "recipe.yaml").write_text(_rebased(manifest, "divergent"), encoding="utf-8")
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
    from rcp_ndcg_vllm.equivalence import stages as stages_module
    from rcp_ndcg_vllm.equivalence.wire import Capture

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = _with_client(load("fixture-embed"), request_shape="messages")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    audit = document["anchor_check"]
    assert audit["checked"] == len(sample_pairs()[0]["documents"]) + 2, audit
    assert audit["passed"] is True, audit["failures"][:1]
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "doc: a caption"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": " [END]"},
                ],
            },
            {"role": "user", "content": "doc: plain [END]"},
        ]
    }
    capture = Capture(recipe)
    capture.exchanges = [{"url": "http://engine/v1/embeddings", "status": 200, "request_body": body}]
    texts = capture.texts(capture.exchanges[0])
    assert texts["input"] == ["doc: a caption [END]", "doc: plain [END]"]
    assert texts["media"] == [["image_url"], []]
    assert stages_module._captured_heads(capture, [])["first"][0]["media"] == [["image_url"], []]  # reported
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
    from rcp_ndcg_vllm.equivalence import stages as stages_module

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
    for broken in ("doc: document 0", cls + "dog: document 0", cls + "do"):  # no cls, another head, a cut head
        for body in (broken, tokenizer.ids(broken)):
            probe = {"rows": [{"shapes": {"document": {"texts": [body]}}, "cuts": 0, "over_cap": False}]}
            check = stages_module._anchor_check(recipe, probe, tokenizer)
            assert check["passed"] is False, (broken, type(body).__name__)
