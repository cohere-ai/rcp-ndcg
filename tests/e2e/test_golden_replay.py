"""The golden replays (GPU-VALIDATION.md items 3-4): suite minis run through rcp-ndcg's full retrieval
and rerank path against the emulators, their nDCG@10 and RCP-nDCG@10 pinned to 1e-9.

**What the numbers are: regression pins, not independent GPU numbers.** The shakedown recorded no
subset run, so ``tests/e2e/golden/goldens.json`` holds the metrics this code computed from the
recorded corpus (``RCP_UPDATE_GOLDENS=1`` recomputes them; every entry says ``kind:
regression-pin``). They catch any change of the replayed path -- fit, template render, transport,
adapter parse, index, rerank, evaluation -- but they are not evidence that the path reproduces a GPU
run. The RC0 subset corpus (the full served exchanges of one real NanoBEIR and one ViDoRe subset)
replaces the fixtures and turns the pins into the GPU run's numbers.

Every request input of these runs is **observed** in the recipe's corpus, so the replay is exact; the
suite fails if any answer was a surrogate (``x-rcp-ndcg-emulator-source``), and fails when the
observations are deleted. Texts are the recorded prompts' sources. The RCP gains are **synthetic
declared fixture values** ranked opposite the model's order (the judged-best document is the one the
model ranks last), so both metrics move whenever a rank moves.

* NanoBEIR-shaped: the retrieval view over two documents (qwen3-embedding-0.6b: the corpus observes
  two distinct prompts, the instructed query and the bare question, so the documents are the bare
  question and the instructed string as a document -- documents render bare) and the rerank view
  (qwen3-reranker-0.6b).
* ViDoRe-shaped: the rerank view only (qwen3-vl-reranker-2b over the recorded text pool). **Waiver**:
  the retrieval view needs page-image embeddings and the provisional corpus observes none (its VL
  embedder saw text only); ``test_the_vidore_retrieval_view_waiver_holds`` fails, naming the work, as
  soon as a corpus with an image input lands.

Regenerate: ``RCP_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/e2e/test_golden_replay.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._engines import emulator_for, harness, load_recipe

GOLDEN = Path(__file__).resolve().parent / "golden" / "goldens.json"
ENGINES_ROOT = Path(__file__).resolve().parents[1] / "contract" / "engines"
EPSILON = 1e-9
PROVENANCE = (
    "computed by tests/e2e/test_golden_replay.py from the provisional shakedown corpus, which recorded no "
    "subset run: a regression pin of the replayed path, not an independent GPU-run number"
)

INSTRUCTED = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QUESTION = "What is the capital of France?"

#: The golden minis. Texts are the shakedown corpus's recorded request texts; ``gains`` are synthetic
#: declared fixture values reversed against the model's order, ``qrels`` the fixture's relevance.
CASES = {
    "nanobeir": {
        "retrieval": {
            "encoder": "qwen3-embedding-0.6b",
            "query_text": QUESTION,
            # d0: the bare question (observed, request #3); d1: the instructed query string sent as a
            # document (documents render bare, so it is the observed request #1). The model ranks d1
            # first; the gains judge d0 best.
            "doc_texts": [QUESTION, INSTRUCTED + QUESTION],
            "gains": [0.75, 0.25],
            "qrels": [1, 0],
        },
        "rerank": {
            "model": "qwen3-reranker-0.6b",
            "query_text": QUESTION,
            "doc_texts": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
            "gains": [0.25, 0.75],
            "qrels": [1, 0],
        },
    },
    "vidore": {
        "retrieval": None,  # waived: no page-image observation (module docstring)
        "rerank": {
            "model": "qwen3-vl-reranker-2b",
            "query_text": QUESTION,
            "doc_texts": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
            "gains": [0.25, 0.75],
            "qrels": [1, 0],
        },
    },
}


def _tokenizer_file(recipe_id: str, tmp_path: Path) -> Path:
    """The recipe's real ``tokenizer.json``, materialised from the corpora's vendored store (the client
    counts its budget with the same file the fingerprint hashes)."""
    recipe = load_recipe(recipe_id)
    spec = recipe.client.tokenizer
    harness()
    from rcp_ndcg_vllm.fingerprint import _store_lookup

    data, _ = _store_lookup(str(spec))
    assert data is not None, spec
    path = tmp_path / f"{recipe_id}-tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _dataset(row: dict, tmp_path: Path, name: str):
    from rcp_ndcg.data import load_dataset

    path = tmp_path / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": row["query_text"],
                    "doc_ids": [f"d{index}" for index in range(len(row["doc_texts"]))],
                    "docs": row["doc_texts"],
                    "qrels": {f"d{index}": int(row["qrels"][index]) for index in range(len(row["doc_texts"]))},
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )
    return load_dataset(f"jsonl:{path}")


def _gains(row: dict) -> dict:
    """The fixture's synthetic declared gains (reversed against the model's order)."""
    return {"q1": {f"d{index}": float(gain) for index, gain in enumerate(row["gains"])}}


def run_retrieval_view(recipe_id: str, row: dict, tmp_path: Path) -> tuple[object, str]:
    """The retrieval view (the paper's embedders): index and search a subset corpus through the recipe's
    emulator -> :class:`~rcp_ndcg.data.Rankings`."""
    from rcp_ndcg_vllm.recipe import client_config

    from rcp_ndcg.retrieval import index as build_index
    from rcp_ndcg.retrieval import search
    from rcp_ndcg.retrieval.config import DenseConfig, ServedEmbedding

    recipe = load_recipe(recipe_id)
    harness()
    data = client_config(recipe, base_url=f"fake://vllm-0.31.0/{recipe_id}")
    data["tokenizer"] = str(_tokenizer_file(recipe_id, tmp_path))
    encoder = ServedEmbedding(**data)
    dataset = _dataset(row, tmp_path, f"{recipe_id}-retrieval")
    built = build_index(dataset, DenseConfig(encoder=encoder), out=tmp_path / "index")
    rankings = search(built, dataset, depth=10)
    return rankings, list(rankings.systems)[0]


def run_rerank_view(recipe_id: str, row: dict, tmp_path: Path) -> tuple[object, str]:
    """The reranking view (the paper's rerankers over a released pool): rerank the ranked pool through
    the recipe's emulator -> :class:`~rcp_ndcg.data.Rankings`."""
    from rcp_ndcg_vllm.recipe import client_config

    from rcp_ndcg.data import Rankings
    from rcp_ndcg.retrieval import rerank
    from rcp_ndcg.retrieval.config import ServedReranker

    recipe = load_recipe(recipe_id)
    harness()
    data = client_config(recipe, base_url=f"fake://vllm-0.31.0/{recipe_id}")
    data["tokenizer"] = str(_tokenizer_file(recipe_id, tmp_path))
    reranker = ServedReranker(**data)
    dataset = _dataset(row, tmp_path, f"{recipe_id}-rerank")
    pool = Rankings.from_orders({"q1": [doc for doc in reversed([f"d{i}" for i in range(len(row["doc_texts"]))])]})
    result = rerank(dataset, pool, reranker, depth=10)
    return result, list(result.systems)[0]


def score(rankings, system: str, row: dict, tmp_path: Path) -> dict[str, float]:
    """The two goldens of one run: ``qrel_ndcg@10`` (nDCG@10) and ``rcp_ndcg@10`` (RCP-nDCG@10)."""
    from rcp_ndcg.eval import evaluate

    report = evaluate(
        rankings,
        dataset=_dataset(row, tmp_path, "scored"),
        gains=_gains(row),
        metrics=("rcp_ndcg", "qrel_ndcg"),
        k=10,
        bootstrap=0,
    )
    return {"qrel_ndcg@10": report.value(system, "qrel_ndcg", 10), "rcp_ndcg@10": report.value(system, "rcp_ndcg", 10)}


def _load_goldens() -> dict:
    if GOLDEN.is_file():
        return json.loads(GOLDEN.read_text(encoding="utf-8"))
    return {}


def _fingerprints(case: str) -> dict[str, str]:
    out = {}
    for row in CASES[case].values():
        if isinstance(row, dict):
            recipe_id = row.get("encoder") or row.get("model")
            emulator = emulator_for(recipe_id)
            assert emulator.verified is not None
            out[recipe_id] = emulator.verified.behaviour_fingerprint
    return out


def _store(values: dict) -> None:
    goldens = _load_goldens()
    goldens.update(values)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recipes(case: str) -> list[str]:
    return [row.get("encoder") or row.get("model") for row in CASES[case].values() if isinstance(row, dict)]


def run_case(case: str, tmp_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """One golden mini through the full path: the metrics per view, and the provenance of every reply
    this run received per recipe (``replayed``/``surrogate``/...), from the emulators' answer logs."""
    starts = {recipe_id: len(emulator_for(recipe_id).answer_log) for recipe_id in _recipes(case)}
    results = {}
    for view, runner in (("retrieval", run_retrieval_view), ("rerank", run_rerank_view)):
        row = CASES[case][view]
        if row is None:
            continue
        recipe_id = row.get("encoder") or row.get("model")
        rankings, system = runner(recipe_id, row, tmp_path / view)
        (tmp_path / view).mkdir(exist_ok=True)
        results[view] = score(rankings, system, row, tmp_path / view)
    answers = {recipe_id: emulator_for(recipe_id).answer_log[start:] for recipe_id, start in starts.items()}
    return results, answers


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_golden_replay_reproduces_the_pinned_metrics(case: str, tmp_path: Path) -> None:
    """One suite mini through the full path reproduces its regression pins to 1e-9 -- and **only**
    replays answered it (every input observed), or the test fails."""
    results, answers = run_case(case, tmp_path)
    for recipe_id, kinds in answers.items():
        assert kinds and set(kinds) == {"replayed"}, f"{recipe_id}: {sorted(set(kinds))} reached the golden run"

    fingerprint = _fingerprints(case)
    if os.environ.get("RCP_UPDATE_GOLDENS"):
        _store({case: {"kind": "regression-pin", "provenance": PROVENANCE, "fingerprints": fingerprint, **results}})
        pytest.skip(f"goldens regenerated for {case}")
    goldens = _load_goldens()
    assert goldens.get(case), f"no goldens for {case}; regenerate with RCP_UPDATE_GOLDENS=1"
    assert goldens[case]["kind"] == "regression-pin" and goldens[case]["provenance"] == PROVENANCE
    assert goldens[case]["fingerprints"] == fingerprint, (
        "the goldens were recorded against other corpora (a behaviour fingerprint moved): "
        "re-record the corpus at RC0, then regenerate the goldens"
    )
    assert set(results) == {view for view in ("retrieval", "rerank") if view in goldens[case]}
    for view in results:
        for metric, value in results[view].items():
            expected = goldens[case][view][metric]
            assert value is not None and abs(value - expected) <= EPSILON, (
                f"{case}/{view}/{metric}: {value} != golden {expected} (beyond {EPSILON})"
            )


def test_a_golden_run_without_its_observations_is_caught(tmp_path: Path) -> None:
    """The golden is not vacuous: with every observation deleted the run is answered by surrogates, and
    the observed-inputs guard the golden test asserts catches it."""
    case = "nanobeir"
    saved = {recipe_id: emulator_for(recipe_id).observations for recipe_id in _recipes(case)}
    try:
        for recipe_id in saved:
            emulator_for(recipe_id).observations = {}
        _, answers = run_case(case, tmp_path)
    finally:
        for recipe_id, observations in saved.items():
            emulator_for(recipe_id).observations = observations
    assert all(kinds and set(kinds) == {"surrogate"} for kinds in answers.values()), answers


def _key_of(emulator, prompt: str) -> str:
    (key,) = [key for key in emulator.observations if json.loads(key)[0] == prompt]
    return key


def test_the_retrieval_golden_ranks_by_the_vectors(tmp_path: Path) -> None:
    """The retrieval golden moves with the vectors: two documents, the model ranks the judged-best one
    last (reversed gains), and moving one document's vector moves both metrics."""
    from dataclasses import replace

    row = CASES["nanobeir"]["retrieval"]
    assert len(row["doc_texts"]) >= 2
    pinned = _load_goldens()["nanobeir"]["retrieval"]
    assert pinned["qrel_ndcg@10"] < 1.0 and pinned["rcp_ndcg@10"] < 1.0, "the model's order is not the judged order"

    emulator = emulator_for(row["encoder"])
    query_key, bare_key = _key_of(emulator, INSTRUCTED + QUESTION), _key_of(emulator, QUESTION)
    saved = emulator.observations
    query_vector = saved[query_key][0].vector
    emulator.observations = {**saved, bare_key: (replace(saved[bare_key][0], vector=query_vector),)}
    try:
        rankings, system = run_retrieval_view(row["encoder"], row, tmp_path / "moved")
        moved = score(rankings, system, row, tmp_path / "moved")
    finally:
        emulator.observations = saved
    assert moved["qrel_ndcg@10"] != pytest.approx(pinned["qrel_ndcg@10"], abs=EPSILON), moved
    assert moved["rcp_ndcg@10"] != pytest.approx(pinned["rcp_ndcg@10"], abs=EPSILON), moved


def test_the_vidore_retrieval_view_waiver_holds() -> None:
    """The documented waiver, as a tripwire on the corpus (never on recipe semantics): the ViDoRe
    retrieval view needs page-image embeddings, and no committed corpus of the VL embedder observes an
    image input. When one does, this fails and names the work."""
    from rcp_ndcg.testing.engines import find_corpora, load_corpus

    corpora = find_corpora(ENGINES_ROOT, recipe_id="qwen3-vl-embedding-2b")
    assert corpora, "the VL embedder has no committed corpus"
    for directory in corpora:
        text = json.dumps([exchange.request_body for exchange in load_corpus(directory).exchanges])
        assert "image" not in text and "data:" not in text, (
            f"{directory} observes an image input: build the ViDoRe retrieval-view golden "
            "(tests/e2e/test_golden_replay.py, RCP_UPDATE_GOLDENS=1) and drop this waiver"
        )


def test_the_golden_replay_goes_red_when_a_rerank_score_is_perturbed(tmp_path: Path) -> None:
    """Mutation (GPU-VALIDATION item 5): one perturbed rerank score moves the ranking and the golden
    metric with it -- the golden replay catches a changed score, not only a changed order file."""
    case = "nanobeir"
    row = CASES[case]["rerank"]
    recipe_id = row["model"]
    emulator = emulator_for(recipe_id)
    before = _load_goldens()[case]["rerank"]

    # perturb the replayed score of the Berlin pair prompt past the Paris one's: the ranking flips
    perturbed = dict(emulator.observations)
    flipped_map = {}
    for key, history in perturbed.items():
        rows = [
            replace_score(observation, observation.score + 2.0)
            if observation.score is not None and "Berlin" in key
            else observation
            for observation in history
        ]
        flipped_map[key] = tuple(rows)
    emulator.observations = flipped_map
    try:
        (tmp_path / "rerank").mkdir(parents=True, exist_ok=True)
        rankings, system = run_rerank_view(recipe_id, row, tmp_path / "rerank")
        after = score(rankings, system, row, tmp_path)
    finally:
        emulator.observations = perturbed
    assert after["qrel_ndcg@10"] != pytest.approx(before["qrel_ndcg@10"], abs=EPSILON)
    assert after["rcp_ndcg@10"] != pytest.approx(before["rcp_ndcg@10"], abs=EPSILON)


def replace_score(observation, score: float):
    from dataclasses import replace

    return replace(observation, score=score)
