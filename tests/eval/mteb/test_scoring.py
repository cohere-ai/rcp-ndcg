"""Scoring inside mteb (`rcp_ndcg.eval.mteb.stored`): a `SearchProtocol` serving stored `Rankings` to
`mteb.evaluate`, a `ModelMeta` from our model identity, and the round trips: the integer nDCG@10 equals our
evaluator's (mteb breaks ties by doc id; so does the nanobeir protocol), and `TaskResult` files land in mteb's
`ResultCache` layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

MTEB = pytest.importorskip("mteb")

from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.abstasks.task_metadata import TaskMetadata  # noqa: E402
from mteb.cache import ResultCache  # noqa: E402

from rcp_ndcg.data import Rankings  # noqa: E402
from rcp_ndcg.data.dataset import Dataset  # noqa: E402
from rcp_ndcg.data.io.mteb import MtebWriter  # noqa: E402
from rcp_ndcg.errors import ConfigError, DataError  # noqa: E402
from rcp_ndcg.eval.evaluate import evaluate  # noqa: E402
from rcp_ndcg.eval.mteb import model_meta, stored_rankings_model  # noqa: E402

SUBSET = "NanoArguAnaRetrieval"
TASK = "NanoArguAnaRCPReranking"


def a_dataset() -> Dataset:
    """One query's pool with tied scores, so the tie rule is exercised, and every query positively labelled."""
    return Dataset.from_records(
        name=SUBSET,
        protocol="nanobeir",
        queries=[{"query_id": "q1", "text": "what is a tortoise"}, {"query_id": "q2", "text": "how long"}],
        corpus=[
            {"doc_id": "d1", "text": "a tortoise is a reptile"},
            {"doc_id": "d2", "text": "unrelated passage"},
            {"doc_id": "d3", "text": "tortoises can live over a century"},
        ],
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2},
            {"query_id": "q1", "doc_id": "d2", "grade": 0},
            {"query_id": "q1", "doc_id": "d3", "grade": 1},
            {"query_id": "q2", "doc_id": "d3", "grade": 1},
            {"query_id": "q2", "doc_id": "d2", "grade": 0},
        ],
        candidates={"q1": ["d1", "d2", "d3"], "q2": ["d3", "d2"]},
    )


def a_run() -> Rankings:
    return Rankings.from_records(
        [
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d1", "score": 0.9},
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d2", "score": 0.5},
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d3", "score": 0.5},
            {"system": "m", "dataset": SUBSET, "query_id": "q2", "doc_id": "d3", "score": 0.8},
            {"system": "m", "dataset": SUBSET, "query_id": "q2", "doc_id": "d2", "score": 0.5},
        ]
    )


def a_task(folder: str) -> Any:
    """A plain mteb retrieval task whose dataset is the written directory (the republished layout)."""

    class Local(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=TASK,
            description="a local round-trip task",
            dataset={"path": folder, "revision": "main"},
            type="Retrieval",
            eval_langs={SUBSET: ["eng-Latn"]},
            main_score="ndcg_at_10",
            category="t2t",
        )

    return Local()


@pytest.fixture
def task(tmp_path: Path) -> Any:
    out = str(tmp_path / "data")
    MtebWriter().write_dataset(a_dataset(), out, subset=SUBSET)
    return a_task(out)


def test_model_meta_carries_our_identity() -> None:
    meta = model_meta("org/model", "abc123", n_parameters=7_000_000, license="apache-2.0")
    assert meta.name == "org/model" and meta.revision == "abc123"
    assert meta.n_parameters == 7_000_000
    assert meta.loader is None  # the model is not loadable from mteb; the scores are served as stored


def test_model_meta_name_needs_the_org_prefix() -> None:
    with pytest.raises(ConfigError, match="org/model"):
        model_meta("model", "abc123")


def test_model_meta_unknown_fields_are_refused() -> None:
    with pytest.raises(ConfigError, match="no_such_field"):
        model_meta("org/model", "abc123", no_such_field=1)


def test_model_meta_defaults_cover_every_required_field() -> None:
    """Every ModelMeta field without a default must be satisfied by `model_meta`'s undeclared-field defaults,
    so a caller who declares only what they know still gets a valid meta (mteb adds a required field -> this
    fails until the defaults follow)."""
    from mteb.models.model_meta import ModelMeta

    from rcp_ndcg.eval.mteb import stored

    required = {name for name, field in ModelMeta.model_fields.items() if field.is_required()} - {"name", "revision"}
    assert required <= set(stored._META_DEFAULTS), sorted(required - set(stored._META_DEFAULTS))
    assert model_meta("org/model", "abc123") is not None


def test_stored_rankings_scored_by_mteb_give_our_integer_ndcg_at_10(task: Any, tmp_path: Path) -> None:
    model = stored_rankings_model(a_run(), model_meta("org/model", "abc123"))
    cache = ResultCache(tmp_path / "cache")

    result = mteb_evaluate(model, task, cache, tmp_path / "preds")

    ours = evaluate(a_run(), dataset=a_dataset(), metrics=("qrel_ndcg",), k=10, bootstrap=0)
    ours_at_10 = ours.summary[0].value
    theirs_at_10 = result.task_results[0].scores["test"][0]["ndcg_at_10"]
    assert theirs_at_10 == pytest.approx(ours_at_10), f"mteb {theirs_at_10} != ours {ours_at_10}"


def test_tied_scores_score_identically_inside_mteb(task: Any) -> None:
    """The gate's tie check: q1 ranks d2 and d3 tied; mteb (pytrec_eval) and the nanobeir protocol both break
    the tie by document id descending, so the per-query values agree (d3 wins the tie; d2 is its tie group's
    mean on ours is not used here -- the integer protocol has no group means)."""
    model = stored_rankings_model(a_run(), model_meta("org/model", "abc123"))
    result = mteb_evaluate(model, task, ResultCache(Path(task.metadata.dataset["path"]) / ".." / "cache"))

    ours = evaluate(a_run(), dataset=a_dataset(), metrics=("qrel_ndcg",), k=10, bootstrap=0)
    per_query_ours = {v.query_id: v.value for v in ours.per_query if v.metric == "qrel_ndcg"}
    assert per_query_ours["q1"] is not None
    assert result.task_results[0].scores["test"][0]["ndcg_at_10"] == pytest.approx(ours.summary[0].value)


def test_the_task_result_lands_in_mteb_s_cache_layout(task: Any, tmp_path: Path) -> None:
    meta = model_meta("org/model", "abc123")
    model = stored_rankings_model(a_run(), meta)
    cache = ResultCache(tmp_path / "cache")

    mteb_evaluate(model, task, cache, tmp_path / "preds")

    loaded = cache.load_task_result(TASK, meta)
    assert loaded is not None and loaded.task_name == TASK
    path = cache.get_task_result_path(task_name=TASK, model_name=meta)
    assert path.is_file()
    assert path.parts[-4:-1] == ("results", "org__model", "abc123")  # {results}/{org__model}/{revision}/{Task}.json
    assert (path.parent / "model_meta.json").is_file()
    assert (path.parent / "run_settings.jsonl").is_file()
    document = json.loads(path.read_text())
    assert document["task_name"] == TASK


def test_the_predictions_file_is_written_by_mteb(task: Any, tmp_path: Path) -> None:
    model = stored_rankings_model(a_run(), model_meta("org/model", "abc123"))
    preds = tmp_path / "preds"

    mteb_evaluate(model, task, ResultCache(tmp_path / "cache"), preds)

    document = json.loads((preds / f"{TASK}_predictions.json").read_text())
    assert document["mteb_model_meta"] == {"model_name": "org/model", "revision": "abc123"}
    assert document[SUBSET]["test"]["q1"] == {"d1": 0.9, "d2": 0.5, "d3": 0.5}


def assert_same_ndcg(ours: float | None, theirs: float) -> None:
    """mteb rounds its mean to 5 decimals; the nanobeir protocol rounds per query before its mean. Same
    number modulo those two rounding conventions (at most half a unit in the 5th decimal)."""
    assert ours is not None
    assert theirs == pytest.approx(ours, abs=1e-5), f"mteb {theirs} != ours {ours}"


def test_a_run_scoring_outside_the_pool_is_restricted_to_it(tmp_path: Path) -> None:
    """The reranking view scores the pool only: a doc outside `top_ranked` must not leak into the run."""
    dataset = a_dataset()
    run = Rankings.from_records(
        [
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d1", "score": 1.0},
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d2", "score": 0.9},
            {"system": "m", "dataset": SUBSET, "query_id": "q1", "doc_id": "d9", "score": 2.0},  # not pooled
            {"system": "m", "dataset": SUBSET, "query_id": "q2", "doc_id": "d3", "score": 0.8},
            {"system": "m", "dataset": SUBSET, "query_id": "q2", "doc_id": "d2", "score": 0.5},
        ]
    )
    out = str(tmp_path / "data")
    MtebWriter().write_dataset(dataset, out, subset=SUBSET)
    model = stored_rankings_model(run, model_meta("org/model", "abc123"))

    result = mteb_evaluate(model, a_task(out), ResultCache(tmp_path / "cache"))

    ours = evaluate(run, dataset=dataset, metrics=("qrel_ndcg",), k=10, bootstrap=0)
    assert_same_ndcg(ours.summary[0].value, result.task_results[0].scores["test"][0]["ndcg_at_10"])


def test_a_run_of_another_subset_is_refused_not_silently_served(tmp_path: Path) -> None:
    """nanobeir subsets share query ids: serving another subset's scores for this task's queries would read as
    plausible, wrong numbers. The run's own dataset column names its subset; a mismatch is a DataError."""
    other = Rankings.from_records(
        [
            {"system": "m", "dataset": "NanoFEVERRetrieval", "query_id": "q1", "doc_id": "d1", "score": 2.0},
            {"system": "m", "dataset": "NanoFEVERRetrieval", "query_id": "q1", "doc_id": "d2", "score": 1.0},
        ]
    )
    model = stored_rankings_model(other, model_meta("org/model", "abc123"))
    with pytest.raises(DataError, match="NanoFEVERRetrieval"):
        model.search({"q1": "text"}, task_metadata=None, hf_split="test", hf_subset="NanoArguAnaRetrieval", top_k=10)


def test_an_unnamed_single_dataset_run_is_served_for_the_task_s_subset() -> None:
    run = Rankings.from_scores({"q1": {"d1": 1.0}})  # names no dataset
    model = stored_rankings_model(run, model_meta("org/model", "abc"))
    scores = model.search(
        {"q1": "text"}, task_metadata=None, hf_split="test", hf_subset="default", top_k=10, top_ranked={"q1": ["d1"]}
    )
    assert scores == {"q1": {"d1": 1.0}}


def test_the_cap_and_its_tie_rule_apply_to_served_scores() -> None:
    """The served scores are capped at mteb's own cap (here: top_k=5 of 8 scored), and the documents that keep
    their place at the cap are the ones the tie rule of `rank_by_score` (doc id descending) ranks first."""
    scores = {f"d{i:02d}": 1.0 for i in range(8)}  # all tied
    run = Rankings.from_scores({"q1": scores}, dataset="ds")
    model = stored_rankings_model(run, model_meta("org/model", "abc"))
    served = model.search({"q1": "text"}, task_metadata=None, hf_split="test", hf_subset="ds", top_k=5)
    assert sorted(served["q1"]) == ["d03", "d04", "d05", "d06", "d07"]  # the highest doc ids win the ties


def test_the_cap_never_exceeds_the_predictions_file_s_1000() -> None:
    scores = {f"d{i:04d}": float(2000 - i) for i in range(1500)}
    run = Rankings.from_scores({"q1": scores}, dataset="ds")
    model = stored_rankings_model(run, model_meta("org/model", "abc"))
    served = model.search({"q1": "text"}, task_metadata=None, hf_split="test", hf_subset="ds", top_k=5000)
    assert len(served["q1"]) == 1000  # mteb's own cap, whatever the task's k values are
    assert served["q1"]["d0000"] == 2000.0


def mteb_evaluate(model: Any, task: Any, cache: ResultCache, preds: Path | None = None) -> Any:
    import mteb

    return mteb.evaluate(
        model,
        task,
        cache=cache,
        prediction_folder=preds,
        encode_kwargs={},
        show_progress_bar=False,
    )
