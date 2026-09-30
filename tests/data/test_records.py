"""In-memory input: ``Dataset.from_records`` and ``Rankings.from_records`` read plain records and validate them."""

from __future__ import annotations

import pandas as pd
import pytest

import rcp_ndcg as rcp
from rcp_ndcg.data import Dataset, DocumentRow, QrelRow, QueryRow, RankingRow, Rankings
from rcp_ndcg.errors import DataError

QUERIES = [{"query_id": "q1", "text": "tides"}, {"query_id": "q2", "text": "moon"}]
CORPUS = [{"doc_id": d, "text": f"text of {d}"} for d in ("a", "b", "c")]
QRELS = [
    {"query_id": "q1", "doc_id": "a", "grade": 2.0, "gain": 0.9, "theta": 2.0},
    {"query_id": "q1", "doc_id": "b", "grade": 0.0, "gain": 0.1, "theta": -1.5},
    {"query_id": "q2", "doc_id": "c", "grade": 1.0},
]


def _dataset(**changes: object) -> Dataset:
    arguments: dict[str, object] = {"queries": QUERIES, "corpus": CORPUS, "qrels": QRELS, **changes}
    return Dataset.from_records(name="mem", **arguments)  # type: ignore[arg-type]


def test_a_dataset_from_records_holds_its_tables_without_files() -> None:
    dataset = _dataset(candidates={"q1": ["a", "b"], "q2": ["c", "a"]}, excluded={"q2": ["b"]})

    assert dataset.uri is None
    assert dataset.queries["q1"] == QueryRow(query_id="q1", text="tides")
    assert dataset.corpus["c"].text == "text of c"
    assert dataset.qrels == {"q1": {"a": 2.0, "b": 0.0}, "q2": {"c": 1.0}}
    assert dataset.gains == {"q1": {"a": 0.9, "b": 0.1}}
    assert dataset.thetas == {"q1": {"a": 2.0, "b": -1.5}}
    assert dataset.candidates == {"q1": ["a", "b"], "q2": ["c", "a"]}
    assert dataset.excluded == {"q2": ["b"]}


def test_row_models_and_dicts_are_both_records() -> None:
    dataset = Dataset.from_records(
        name="mem",
        queries=[QueryRow(query_id="q1", text="tides")],
        corpus=[DocumentRow(doc_id="a", text="x")],
        qrels=[QrelRow(query_id="q1", doc_id="a", grade=1)],
    )
    assert dataset.qrels == {"q1": {"a": 1.0}}


def test_an_in_memory_dataset_scores_like_a_loaded_one() -> None:
    dataset = _dataset(qrels=QRELS[:2])
    rankings = Rankings.from_records(
        pd.DataFrame({"query_id": ["q1", "q1"], "doc_id": ["b", "a"], "score": [2.0, 1.0]}).to_dict("records")
    )
    report = rcp.evaluate(rankings, dataset=dataset, k=2, bootstrap=0)
    assert report.gains_source == "dataset"
    assert 0.0 < report.value("system") < 1.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"queries": [{"query_id": "q1", "txt": "x"}]}, r"queries\[0\]: txt: unknown key \(did you mean 'text'\?\)"),
        ({"qrels": [{"query_id": "q1", "doc_id": "a"}]}, r"qrels\[0\]: grade: Field required"),
        ({"qrels": [{"query_id": "q1", "doc_id": "a", "grade": 1, "gain": 1.5}]}, r"qrels\[0\]: gain"),
        ({"qrels": [{"query_id": "q1", "doc_id": "a", "grade": float("nan")}]}, r"grade: not finite"),
        ({"queries": [*QUERIES, {"query_id": "q1"}]}, "query_id 'q1' appears twice"),
        ({"qrels": [QRELS[0], QRELS[0]]}, "labelled twice"),
        ({"qrels": [{"query_id": "q9", "doc_id": "a", "grade": 1}]}, "not in queries"),
        ({"qrels": [{"query_id": "q1", "doc_id": "zz", "grade": 1}]}, "not in the corpus"),
        ({"candidates": {"q1": ["a", "a"]}}, "lists"),
        ({"candidates": {"q1": "a"}}, "not a list"),
        ({"queries": ["q1"]}, r"queries\[0\] is a str"),
    ],
)
def test_malformed_records_are_refused_with_the_record_named(changes: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        _dataset(**changes)


def test_a_record_error_carries_the_field_and_a_did_you_mean() -> None:
    with pytest.raises(DataError) as caught:
        Rankings.from_records([{"query-id": "q1", "doc_id": "a", "score": 1.0}])
    details = caught.value.details
    assert details["index"] == 0
    problems = {problem["field"]: problem for problem in details["errors"]}
    assert problems["query-id"]["did_you_mean"] == "query_id"
    assert problems["query_id"]["problem"] == "Field required"


def test_rankings_from_records_validate_strictly() -> None:
    rankings = Rankings.from_records(
        [RankingRow(query_id="q1", doc_id="a", score=1.0, system="bm25"), {"query_id": 1, "doc_id": 2, "score": 3}]
    )
    assert rankings.systems == ["bm25", "system"]
    assert rankings.for_query("1", system="system") == {"2": 3.0}
    with pytest.raises(DataError, match="not finite"):
        Rankings.from_records([{"query_id": "q1", "doc_id": "a", "score": float("inf")}])
    with pytest.raises(DataError, match="duplicate score"):
        Rankings.from_records([{"query_id": "q1", "doc_id": "a", "score": 1.0}] * 2)


def test_short_reprs() -> None:
    dataset = _dataset(candidates={"q1": ["a", "b"]})
    rankings = Rankings.from_scores({"q1": {"a": 1.0}}, system="bm25")
    assert repr(dataset) == "Dataset('mem', protocol=None, 2 labelled queries, 1 pools, 2 gains)"
    assert repr(rankings) == "Rankings(1 scores, 1 queries, systems=[bm25])"
    assert dataset.parts == (dataset,)


def test_validate_is_a_library_call_over_any_dataset_and_rankings() -> None:
    """The checks behind `data validate` run on an in-memory dataset, with no command line."""
    dataset = _dataset(
        qrels=[*QRELS[:2], {"query_id": "q2", "doc_id": "c", "grade": 1.0, "gain": 1.0}],
        candidates={"q1": ["a", "b", "c"], "q2": ["c"]},
        excluded={"q1": ["a"]},
    )
    rankings = Rankings.from_scores({"q2": {"c": 2.0, "a": 1.0}, "q9": {"a": 1.0}}, system="mine")

    report = rcp.data.validate(dataset, rankings)

    assert (report.ok, report.errors, report.systems) == (False, ["EXCLUDED_RELEVANT", "UNKNOWN_QUERIES"], ["mine"])
    counts = {check.code: (check.count, check.examples) for check in report.checks}
    assert counts["UNJUDGED_CANDIDATES"] == (1, ["q1/c"])
    assert counts["OUTSIDE_POOL"] == (1, ["q2/a"])
    assert counts["UNKNOWN_QUERIES"] == (1, ["q9"])
