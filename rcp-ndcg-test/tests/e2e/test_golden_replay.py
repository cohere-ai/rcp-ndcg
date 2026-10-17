"""The golden replays (docs/how-to/use-verified-fake-engines.md): suite minis run through rcp-ndcg's full retrieval
and rerank path against the emulators, their nDCG@10 and RCP-nDCG@10 pinned to 1e-9.

**What the numbers are: regression pins, not independent GPU numbers.**
``tests/e2e/golden/goldens.json`` holds the metrics this code computed from the release corpora's
recorded exchanges (``RCP_UPDATE_GOLDENS=1`` recomputes them; every entry says ``kind: regression-pin``
and carries the behaviour fingerprints that pin the corpora it was computed from -- the entries were
regenerated from the release recordings at ``392bd61``, so they are no longer the provisional shakedown
corpus's numbers). They catch any change of the replayed path -- fit, template render, transport,
adapter parse, index, rerank, evaluation -- but they are the emulators' replay of the recorded GPU
exchanges, never an independent GPU-run measurement of the path.

Every request input of these runs is **observed** in the recipe's corpus, so the replay is exact; the
suite fails if any answer was a surrogate (``x-rcp-ndcg-emulator-source``), and fails when the
observations are deleted. Texts are the recorded prompts' sources. The RCP gains are **synthetic
declared fixture values**, and how they sit against the model's order is per view, not one property:
the NanoBEIR retrieval view orders its documents by the model's own recorded vectors with the gain-best
LAST (``_ordered_by_model``) and the NanoBEIR rerank view's corpus order happens to put the gain-best
last too, so both metrics move whenever a rank moves there. The ViDoRe rerank pin is the other way
round: its documents are in the rerank corpus's own order, which the model ranks with the gain-best
FIRST (``rcp_ndcg@10 = 1.0``) while the qrel-relevant document is second (``qrel_ndcg@10 = 0.6309``).

* NanoBEIR-shaped: the retrieval view over two documents (qwen3-embedding-0.6b: the corpus observes
  two distinct prompts, the instructed query and the bare question, so the documents are the bare
  question and the instructed string as a document -- documents render bare) and the rerank view
  (zerank-2-reranker over the same recorded text pool; every committed corpus is a release recording
  and current -- ``tests/conformance/stale.json`` ships empty, so the qwen3-reranker corpora are not
  declared stale).
* ViDoRe-shaped: the rerank view only (zerank-1-small-reranker over the recorded text pool; the VL
  reranker's corpus is a current release recording too, and the pin replays the pool through the
  reranker it was computed with). **Waiver**:
  the retrieval view needs ViDoRe page-image embeddings and no committed corpus observes a
  ViDoRe-sourced image. The VL embedder's corpus does observe media (measured over its committed
  records: 13 requests carry an image part, 4 a video container), but they are the observation plan's
  synthetic media buckets, whose qrels and documents are not ViDoRe's.
  ``test_the_vidore_retrieval_view_waiver_holds`` keys on the waiver's ACTUAL condition -- a
  ViDoRe-sourced image input -- not on images in general, and fails, naming the work, as soon as one
  is observed.

Regenerate: ``RCP_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/e2e/test_golden_replay.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from tests._engines import corpus_of, emulator_for, harness, load_recipe

GOLDEN = Path(__file__).resolve().parent / "golden" / "goldens.json"
ENGINES_ROOT = Path(__file__).resolve().parents[2] / "corpora"
EPSILON = 1e-9
PROVENANCE = (
    "computed by tests/e2e/test_golden_replay.py from the release corpora's recorded exchanges (the "
    "fingerprints below pin them), replayed through the recipes' emulators: a regression pin of the "
    "replayed path, not an independent GPU-run number"
)

INSTRUCTED = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QUESTION = "What is the capital of France?"


def _key_of(emulator: Any, prompt: str) -> str:
    """One observation key whose prompt part is ``prompt`` (the replay key's first element). A corpus may
    record the same prompt for more than one request, so the first match is returned."""
    for key in emulator.observations:
        if json.loads(key)[0] == prompt:
            return key
    raise AssertionError(f"no observation carries the prompt {prompt!r}")


def _recorded_bodies(recipe_id: str) -> list[dict[str, Any]]:
    """The request bodies one recipe's corpus recorded, in record order."""
    from rcp_ndcg_test.engines import exchanges_of

    corpus = corpus_of(load_recipe(recipe_id))
    return [dict(exchange.request_body or {}) for exchange in exchanges_of(corpus)]


def _embed_case_texts(encoder: str) -> tuple[str, list[str]]:
    """One observed query text and two observed document texts from the encoder's corpus: the client
    renders a query with the instruction and a document bare, so the recorded instructed prompts give the
    query text back (minus the prefix) and the recorded bare prompts are the document texts. The release
    recordings' prompts are not the shakedown's, so the case reads them instead of hard-coding them."""
    inputs = [
        item
        for body in _recorded_bodies(encoder)
        for item in (body.get("input") if isinstance(body.get("input"), list) else [body.get("input")])
        if isinstance(item, str)
    ]
    instructed = [text for text in inputs if text.startswith(INSTRUCTED)]
    bare = [text for text in inputs if not text.startswith(INSTRUCTED)]
    assert instructed and len(bare) >= 2, (
        f"{encoder}: the corpus records {len(instructed)} instructed and {len(bare)} bare"
    )
    return instructed[0][len(INSTRUCTED) :], bare[:2]


def _rerank_case_texts(model: str) -> tuple[str, list[str]]:
    """One observed query and two observed documents from the reranker's corpus (its first recorded
    request that carries both)."""
    for body in _recorded_bodies(model):
        query, documents = body.get("query"), body.get("documents")
        if isinstance(query, str) and isinstance(documents, list) and len(documents) >= 2:
            return query, [str(document) for document in documents[:2]]
    raise AssertionError(f"{model}: the corpus records no query with two documents")


_EMBED_QUERY, _EMBED_DOCS_RAW = _embed_case_texts("qwen3-embedding-0.6b")
_RERANK_QUERY, _RERANK_DOCS = _rerank_case_texts("zerank-2-reranker")
_VIDORE_QUERY, _VIDORE_DOCS = _rerank_case_texts("zerank-1-small-reranker")


def _ordered_by_model(encoder: str, query_text: str, doc_texts: list[str]) -> list[str]:
    """The documents ordered so the model's OWN recorded vectors put the LAST one first: the fixture's d0 is
    judged best (gains ``[0.75, 0.25]``) and relevant (qrels ``[1, 0]``) while the model ranks d1 first, so
    both pinned metrics are below 1.0 -- the property the mutation test needs, and it holds whatever texts
    the corpus recorded."""
    import numpy as np

    emulator = emulator_for(encoder)

    def score(text: str) -> float:
        query = np.asarray(emulator.observations[_key_of(emulator, INSTRUCTED + query_text)][0].vector, dtype=float)
        vector = np.asarray(emulator.observations[_key_of(emulator, text)][0].vector, dtype=float)
        return float(query @ vector)

    return sorted(doc_texts, key=score)  # the model's worst first, its best last (the fixture's d1)


_EMBED_DOCS = _ordered_by_model("qwen3-embedding-0.6b", _EMBED_QUERY, _EMBED_DOCS_RAW)

#: The golden minis. Texts are the CURRENT corpus's recorded request texts (the release recordings' prompts
#: are not the shakedown's, so they are read, never hard-coded); ``gains`` are synthetic declared fixture
#: values and ``qrels`` the fixture's relevance -- how the gains sit against the model's order is per view
#: (the module docstring: NanoBEIR pins the gain-best last, the ViDoRe rerank pin first).
CASES = {
    "nanobeir": {
        "retrieval": {
            "encoder": "qwen3-embedding-0.6b",
            "query_text": _EMBED_QUERY,
            "doc_texts": _EMBED_DOCS,
            "gains": [0.75, 0.25],
            "qrels": [1, 0],
        },
        "rerank": {
            "model": "zerank-2-reranker",
            "query_text": _RERANK_QUERY,
            "doc_texts": _RERANK_DOCS,
            "gains": [0.25, 0.75],
            "qrels": [1, 0],
        },
    },
    "vidore": {
        "retrieval": None,  # waived: no page-image observation (module docstring)
        "rerank": {
            "model": "zerank-1-small-reranker",
            "query_text": _VIDORE_QUERY,
            "doc_texts": _VIDORE_DOCS,
            "gains": [0.25, 0.75],
            "qrels": [1, 0],
        },
    },
}


def _tokenizer_file(recipe_id: str, tmp_path: Path) -> Path:
    """The recipe's real ``tokenizer.json``, materialised from the corpora's vendored store (the client
    counts its budget with the same file the fingerprint hashes)."""
    recipe = load_recipe(recipe_id)
    spec = recipe.client.get("tokenizer")
    harness()
    from rcp_ndcg_test.fingerprint import stored_tokenizer

    data, _ = stored_tokenizer(str(spec))
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
    """The fixture's synthetic declared gains, ``qrels``' other half (see the module docstring for how they
    sit against each view's model order)."""
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


def test_the_retrieval_golden_ranks_by_the_vectors(tmp_path: Path) -> None:
    """The retrieval golden moves with the vectors: two documents, the model ranks the judged-best one
    last (reversed gains), and moving one document's vector moves both metrics."""
    from dataclasses import replace

    row = CASES["nanobeir"]["retrieval"]
    assert len(row["doc_texts"]) >= 2
    pinned = _load_goldens()["nanobeir"]["retrieval"]
    assert pinned["qrel_ndcg@10"] < 1.0 and pinned["rcp_ndcg@10"] < 1.0, "the model's order is not the judged order"

    emulator = emulator_for(row["encoder"])
    query_key, bare_key = _key_of(emulator, INSTRUCTED + row["query_text"]), _key_of(emulator, row["doc_texts"][0])
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
    retrieval view needs ViDoRe page-image embeddings, and no committed corpus of the VL embedder
    observes one. The tripwire keys on the waiver's ACTUAL condition -- a ViDoRe-sourced image input --
    not on images in general: the observation plan's synthetic media buckets do observe images, but
    their qrels and documents are not ViDoRe's, so they cannot feed the ViDoRe retrieval-view golden.
    When a ViDoRe-sourced image is observed, this fails and names the work."""
    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.engines import find_corpora

    corpora = find_corpora(ENGINES_ROOT, recipe_id="qwen3-vl-embedding-2b")
    assert corpora, "the VL embedder has no committed corpus"
    for directory in corpora:
        for record in load_corpus(directory).records:
            source = (record.get("inputs") or {}).get("source") or {}
            if not isinstance(source, dict):
                continue
            names = " ".join(str(source.get(field) or "") for field in ("suite", "subset", "file")).lower()
            if "vidore" not in names:
                continue
            body = json.dumps(record.get("request") or {})
            assert "image" not in body and "data:" not in body, (
                f"{directory} observes a ViDoRe-sourced image input: build the ViDoRe retrieval-view "
                "golden (tests/e2e/test_golden_replay.py, RCP_UPDATE_GOLDENS=1) and drop this waiver"
            )


def test_the_golden_replay_goes_red_when_a_rerank_score_is_perturbed(tmp_path: Path) -> None:
    """Mutation (docs/how-to/release-candidates.md): one perturbed rerank score moves the ranking and the golden
    metric with it -- the golden replay catches a changed score, not only a changed order file."""
    case = "nanobeir"
    row = CASES[case]["rerank"]
    recipe_id = row["model"]
    emulator = emulator_for(recipe_id)
    before = _load_goldens()[case]["rerank"]

    # perturb the replayed score of the judged-best document's prompt past the other's: the ranking flips
    # (the case's texts are the corpus's own, so the perturbed key is derived, never named)
    judged_best = row["doc_texts"][1]  # gains [0.25, 0.75]: the second document is judged best
    perturbed = dict(emulator.observations)
    flipped_map = {}
    for key, history in perturbed.items():
        rows = [
            replace_score(observation, observation.score + 2.0)
            if observation.score is not None and judged_best in key
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
