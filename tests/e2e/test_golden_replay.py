"""The golden replays (GPU-VALIDATION.md items 3-4): one NanoBEIR-shaped and one ViDoRe-shaped subset
run through rcp-ndcg's full retrieval and rerank path against the emulators reproduce the GPU run's
metrics to 1e-9.

Every request input of these runs is **observed** in the recipe's corpus (the shakedown's recorded
exchanges), so the replay is exact and the metrics are the GPU run's numbers recomputed end to end:
fit, template render, transport, adapter parse, index, rerank and evaluation. The suite fails if any
answer was a surrogate.

The fixtures here are the shakedown's provisional minis (``tests/e2e/golden/``): the query, pages and
pool texts are the recorded prompts' sources. The RCP gains are **synthetic declared fixture values**
deliberately ranked opposite the models' scores, so the metric is sensitive to any order change; the
qrels keep the texts' honest relevance. The numbers are **regression pins generated from the recorded
corpus** (``RCP_UPDATE_GOLDENS=1`` recomputes them from the same corpus the emulators replay -- the
shakedown corpus recorded no subset run), not an independent GPU run's outputs: the RC0 subset corpus
(the full served exchanges of one real NanoBEIR and one ViDoRe subset) replaces the fixtures and
becomes the GPU run's numbers. The ViDoRe mini runs the rerank view only (its retrieval view is a
tripwire-pinned recipe gap).

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

#: The two golden minis. Texts are exactly the shakedown corpus's recorded request texts; the gains are
#: synthetic declared fixture values (formal-judgement stand-ins with the ranks reversed against the
#: model scores), the qrels the honest relevance of the texts to the query.
CASES = {
    "nanobeir": {
        "view_query": "What is the capital of France?",
        "retrieval": {
            "encoder": "qwen3-embedding-0.6b",
            "query_text": "What is the capital of France?",
            "doc_texts": ["What is the capital of France?"],
        },
        "rerank": {
            "model": "qwen3-reranker-0.6b",
            "query_text": "What is the capital of France?",
            "doc_texts": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        },
    },
    "vidore": {
        "view_query": "What is the capital of France?",
        # the retrieval view of ViDoRe is not built here: the VL embedder's template declares no
        # ``query`` shape (its one frame is the model's default prompt), and the product's search
        # renders queries as ``query`` -- a recipe-side declaration this lane deliberately does not
        # invent. ``test_the_vidore_retrieval_view_gap_is_declared`` pins the gap (it fails and names
        # the work when the shape lands), and the ViDoRe golden covers the rerank view over the
        # recorded pool.
        "retrieval": None,
        "rerank": {
            "model": "qwen3-vl-reranker-2b",
            "query_text": "What is the capital of France?",
            "doc_texts": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
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
                    "qrels": {f"d{index}": int(index == 0) for index in range(len(row["doc_texts"]))},
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )
    return load_dataset(f"jsonl:{path}")


def _gains(row: dict) -> dict:
    """The fixture's synthetic declared gains: reversed against the models' score order (the last
    document is the judged-best), so nDCG moves whenever a rank moves."""
    count = len(row["doc_texts"])
    return {"q1": {f"d{index}": 0.75 if index == count - 1 else 0.25 for index in range(count)}}


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


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_golden_replay_reproduces_the_metrics(case: str, tmp_path: Path) -> None:
    """One suite mini through the full path reproduces the goldens to 1e-9 -- and **only** answer
    replays (every input observed), or the run is marked and the test fails."""
    emulator_for(CASES[case]["rerank"]["model"])
    if CASES[case]["retrieval"] is not None:
        emulator_for(CASES[case]["retrieval"]["encoder"])
    for _view, _row in CASES[case].items():
        if isinstance(_row, dict):
            emulator_for(_row.get("encoder") or _row.get("model")).clear_answer_log()
    results = {}
    for view, runner in (("retrieval", run_retrieval_view), ("rerank", run_rerank_view)):
        row = CASES[case][view]
        if row is None:
            continue
        recipe_id = row.get("encoder") or row.get("model")
        rankings, system = runner(recipe_id, row, tmp_path / view)
        (tmp_path / view).mkdir(exist_ok=True)
        results[view] = score(rankings, system, row, tmp_path)

    # a test that asserts numbers can only use observed inputs: every answer of THIS run was a replay
    for _view, row in CASES[case].items():
        if not isinstance(row, dict):
            continue
        recipe_id = row.get("encoder") or row.get("model")
        answers = getattr(emulator_for(recipe_id), "answer_log", [])
        assert answers and set(answers) == {"replayed"}, f"{recipe_id}: surrogate answers reached the golden run"

    fingerprint = _fingerprints(case)
    if os.environ.get("RCP_UPDATE_GOLDENS"):
        _store({case: {"fingerprints": fingerprint, **results}})
        pytest.skip(f"goldens regenerated for {case}")
    goldens = _load_goldens()
    assert goldens.get(case), f"no goldens for {case}; regenerate with RCP_UPDATE_GOLDENS=1"
    assert goldens[case]["fingerprints"] == fingerprint, (
        "the goldens were recorded against other corpora (a behaviour fingerprint moved): "
        "re-record the corpus at RC0, then regenerate the goldens"
    )
    for view in results:
        for metric, value in results[view].items():
            expected = goldens[case][view][metric]
            assert value is not None and abs(value - expected) <= EPSILON, (
                f"{case}/{view}/{metric}: {value} != golden {expected} (beyond {EPSILON})"
            )


def test_the_vidore_retrieval_view_gap_is_declared() -> None:
    """A tripwire, never a silent skip: ViDoRe's retrieval golden is missing because the VL embedder's
    template declares no ``query`` shape. When the recipe declares one (fam-vl's call), this fails and
    names the work: build the ViDoRe retrieval view golden and regenerate."""
    recipe = load_recipe("qwen3-vl-embedding-2b")
    assert recipe.client.template is not None
    assert "query" not in recipe.client.template.shapes(), (
        "qwen3-vl-embedding-2b now declares a query shape: add the ViDoRe retrieval-view golden "
        "(tests/e2e/test_golden_replay.py, RCP_UPDATE_GOLDENS=1)"
    )


def test_the_golden_replay_goes_red_when_a_rerank_score_is_perturbed(tmp_path: Path) -> None:
    """Mutation (GPU-VALIDATION item 5): one perturbed rerank score moves the ranking and the golden
    metric with it -- the golden replay catches a changed score, not only a changed order file."""
    case = "nanobeir"
    row = CASES[case]["rerank"]
    recipe_id = row["model"]
    emulator = emulator_for(recipe_id)
    goldens = _load_goldens()
    before = goldens[case]["rerank"]

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
