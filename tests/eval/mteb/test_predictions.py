"""`Rankings.save(format="mteb")`: the `{Task}_predictions.json` file mteb's evaluator writes, from stored
rankings: `{"mteb_model_meta": {model_name, revision}, subset: {split: {qid: {did: score}}}}` (mteb
``abstask._save_task_predictions``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp_ndcg.data import Rankings
from rcp_ndcg.errors import ConfigError, DataError


def run(dataset: str = "NanoArguAnaRetrieval", **overrides) -> Rankings:
    records = [
        {"query_id": "q1", "doc_id": "d1", "score": 2.0},
        {"query_id": "q1", "doc_id": "d2", "score": 1.0},
        {"query_id": "q2", "doc_id": "d3", "score": 0.5},
    ]
    for record in records:
        record.update(overrides)
        record["dataset"] = dataset
    return Rankings.from_records(records)


def save(rankings: Rankings, folder: Path, **kwargs) -> str:
    defaults = dict(
        task="NanoArguAnaRCPReranking",
        qrels={"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}},
        model_name="org/model",
        model_revision="abc123",
    )
    defaults.update(kwargs)
    return rankings.save(str(folder), format="mteb", **defaults)


def test_the_file_is_mteb_s_predictions_layout(tmp_path: Path) -> None:
    out = save(run(), tmp_path)
    assert Path(out).name == "NanoArguAnaRCPReranking_predictions.json"
    document = json.loads(Path(out).read_text())
    assert document == {
        "mteb_model_meta": {"model_name": "org/model", "revision": "abc123"},
        "NanoArguAnaRetrieval": {"test": {"q1": {"d1": 2.0, "d2": 1.0}, "q2": {"d3": 0.5}}},
    }


def test_an_unnamed_dataset_writes_the_default_subset(tmp_path: Path) -> None:
    rankings = Rankings.from_scores({"q1": {"d1": 1.0}})
    out = save(rankings, tmp_path, qrels={"q1": {"d1": 1}})
    document = json.loads(Path(out).read_text())
    assert set(document) == {"mteb_model_meta", "default"}
    assert document["default"] == {"test": {"q1": {"d1": 1.0}}}


def test_a_query_without_qrels_is_dropped_declared_policy(tmp_path: Path) -> None:
    """mteb raises on a result for a query that has no qrels, so the file may not carry one."""
    out = save(run(), tmp_path, qrels={"q1": {"d1": 2, "d2": 0}})  # q2 is not in the qrels
    document = json.loads(Path(out).read_text())
    assert "q2" not in document["NanoArguAnaRetrieval"]["test"]


def test_every_qrels_query_is_present(tmp_path: Path) -> None:
    """A labelled query the run did not rank makes the file lie about its coverage: refused, naming the query."""
    with pytest.raises(DataError, match="q9"):
        save(run(), tmp_path, qrels={"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}, "q9": {"d1": 1}})


def test_an_empty_qrels_dict_is_skipped_like_the_evaluator_skips_it(tmp_path: Path) -> None:
    """mteb drops queries whose qrels are empty (`_filter_queries_without_positives`), so they are not part of
    "every filtered query"."""
    out = save(run(), tmp_path, qrels={"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}, "q3": {}})
    document = json.loads(Path(out).read_text())
    assert set(document["NanoArguAnaRetrieval"]["test"]) == {"q1", "q2"}


def test_at_most_1000_documents_per_query(tmp_path: Path) -> None:
    scores = {f"d{i:04d}": float(1000 - i) for i in range(1500)}
    rankings = Rankings.from_scores({"q1": scores}, dataset="ds")
    out = save(rankings, tmp_path, task="T", qrels={"q1": {"d0000": 1}}, model_name="o/m", model_revision="r")
    document = json.loads(Path(out).read_text())
    docs = document["ds"]["test"]["q1"]
    assert len(docs) == 1000
    assert max(docs, key=docs.get) == "d0000"  # the best-scored documents survive the cap


def test_tied_scores_at_the_cap_break_by_document_id_descending(tmp_path: Path) -> None:
    scores = {f"d{i:04d}": 1.0 for i in range(1200)}
    rankings = Rankings.from_scores({"q1": scores}, dataset="ds")
    out = save(rankings, tmp_path, task="T", qrels={"q1": {"d1199": 1}}, model_name="o/m", model_revision="r")
    docs = json.loads(Path(out).read_text())["ds"]["test"]["q1"]
    assert len(docs) == 1000
    assert min(docs) == "d0200" and max(docs) == "d1199"  # the highest doc ids win the ties


def test_multi_system_rankings_need_a_system(tmp_path: Path) -> None:
    rankings = Rankings.concat([run(), Rankings(run().to_pandas().assign(system="other"))])
    with pytest.raises(DataError, match="system"):
        save(rankings, tmp_path)
    out = save(rankings, tmp_path, system="system")
    assert Path(out).is_file()


def test_the_required_arguments_are_refused_when_missing(tmp_path: Path) -> None:
    rankings = run()
    with pytest.raises(ConfigError, match="task"):
        rankings.save(str(tmp_path), format="mteb", qrels={}, model_name="o/m", model_revision="r")
    with pytest.raises(ConfigError, match="qrels"):
        rankings.save(str(tmp_path), format="mteb", task="T", model_name="o/m", model_revision="r")
    with pytest.raises(ConfigError, match="model_name"):
        rankings.save(str(tmp_path), format="mteb", task="T", qrels={}, model_revision="r")
    with pytest.raises(ConfigError, match="model_revision"):
        rankings.save(str(tmp_path), format="mteb", task="T", qrels={}, model_name="o/m")


def test_the_split_is_the_caller_s(tmp_path: Path) -> None:
    out = save(run(), tmp_path, split="train")
    document = json.loads(Path(out).read_text())
    assert document["NanoArguAnaRetrieval"]["train"]["q1"]["d1"] == 2.0
