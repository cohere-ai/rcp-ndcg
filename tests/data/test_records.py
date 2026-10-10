"""In-memory input: ``Dataset.from_records`` and ``Rankings.from_records`` read plain records and validate them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rcp_ndcg_core.records import Document, Query

import rcp_ndcg as rcp
from rcp_ndcg.data import Dataset, QrelRow, RankingRow, Rankings
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
    assert dataset.queries["q1"] == Query(query_id="q1", text="tides")
    assert dataset.corpus["c"].text == "text of c"
    assert dataset.qrels == {"q1": {"a": 2.0, "b": 0.0}, "q2": {"c": 1.0}}
    assert dataset.gains == {"q1": {"a": 0.9, "b": 0.1}}
    assert dataset.thetas == {"q1": {"a": 2.0, "b": -1.5}}
    assert dataset.candidates == {"q1": ["a", "b"], "q2": ["c", "a"]}
    assert dataset.excluded == {"q2": ["b"]}


def test_records_and_dicts_are_both_records() -> None:
    dataset = Dataset.from_records(
        name="mem",
        queries=[Query(query_id="q1", text="tides")],
        corpus=[Document(doc_id="a", text="x")],
        qrels=[QrelRow(query_id="q1", doc_id="a", grade=1)],
    )
    assert dataset.qrels == {"q1": {"a": 1.0}}


def test_the_dataset_holds_the_core_records() -> None:
    """The pipeline's real in-memory model, not a compatibility row: the tables are the records."""
    dataset = _dataset()

    assert isinstance(dataset.queries["q1"], Query)
    assert isinstance(dataset.corpus["a"], Document)
    assert dataset.queries["q1"].query_id == "q1"
    assert dataset.corpus["a"].doc_id == "a"


def test_the_compatibility_row_models_are_gone() -> None:
    with pytest.raises(ImportError):
        from rcp_ndcg.data import DocumentRow  # noqa: F401
    with pytest.raises(ImportError):
        from rcp_ndcg.data import QueryRow  # noqa: F401


def test_numeric_ids_are_coerced_to_strings() -> None:
    """The row models' rule, moved into the records: a numeric id reads as its string form."""
    dataset = Dataset.from_records(
        name="mem",
        queries=[{"query_id": 1, "text": "tides"}],
        corpus=[{"doc_id": 2, "text": "x"}],
        qrels=[{"query_id": 1, "doc_id": 2, "grade": 1}],
    )

    assert list(dataset.queries) == ["1"]
    assert list(dataset.corpus) == ["2"]


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


def test_an_excluded_pooled_document_without_a_positive_label_is_not_excluded_relevant() -> None:
    """``_positive`` filters ``g > 0`` so an excluded zero-label pool document (a pooled but
    unjudged one) does not raise EXCLUDED_RELEVANT; the filter is what keeps the check honest."""
    dataset = _dataset(candidates={"q1": ["a", "b"]}, excluded={"q1": ["b"]})  # b's grade is 0

    report = rcp.data.validate(dataset)

    assert report.ok
    assert not any(check.code == "EXCLUDED_RELEVANT" for check in report.checks)


def test_validate_checks_the_rows_that_rank_the_dataset() -> None:
    """A suite-wide file validated against one subset: the other subsets' rows are not this dataset's."""
    biology = Dataset.from_records(
        name="biology",
        qrels=[{"query_id": "0", "doc_id": "a1", "grade": 1}, {"query_id": "0", "doc_id": "a2", "grade": 0}],
        candidates={"0": ["a1", "a2"]},
    )
    suite_wide = Rankings.from_records(
        [
            {"dataset": "biology", "query_id": "0", "doc_id": "a1", "score": 2.0},
            {"dataset": "earth_science", "query_id": "0", "doc_id": "b1", "score": 2.0},
            {"dataset": "earth_science", "query_id": "7", "doc_id": "b9", "score": 1.0},
        ]
    )

    assert rcp.data.validate(biology, suite_wide).checks == []
    other = rcp.data.validate(biology, Rankings.from_scores({"0": {"b1": 1.0}}, dataset="earth_science"))
    assert other.errors == ["NO_RANKINGS"]


# ---------------------------------------------------------------------------
# Title, subset/split/task, provenance: the data-model fields (workstream 10)
# ---------------------------------------------------------------------------


def test_a_document_carries_its_title_as_a_field_and_the_body_in_text() -> None:
    from rcp_ndcg_core.records import Document

    document = Document(doc_id="d1", title="Tortoises", text="a tortoise is a reptile")

    assert document.title == "Tortoises"
    assert document.text == "a tortoise is a reptile", "the body stays in text; nothing joins at read time"
    assert Document(doc_id="d1", text="body").title is None
    assert Document(doc_id="d1", text="body").model_dump(exclude_none=True) == {"id": "d1", "text": "body"}


def test_a_document_title_round_trips_through_jsonl(tmp_path: Path) -> None:
    from rcp_ndcg_core.records import Document

    path = tmp_path / "corpus.jsonl"
    path.write_text(Document(doc_id="d1", title="T", text="body").model_dump_json(exclude_none=True) + "\n")

    from rcp_ndcg.storage.io import iter_jsonl

    (restored,) = iter_jsonl(str(path), example_class=Document, forbid_extra=True)
    assert restored.title == "T" and restored.text == "body"


def test_a_corpus_record_carries_its_title() -> None:
    dataset = _dataset(corpus=[{"doc_id": "a", "title": "Alpha", "text": "alpha body"}, *CORPUS[1:]])

    assert dataset.corpus["a"].title == "Alpha"
    assert dataset.corpus["a"].text == "alpha body"
    assert dataset.corpus["b"].title is None


def test_a_dataset_records_subset_split_and_task_with_their_defaults() -> None:
    dataset = _dataset()

    assert dataset.subset == "default"
    assert dataset.split == "test"
    assert dataset.task is None
    assert dataset.task_instruction is None
    assert dataset.provenance is None


def test_from_records_takes_subset_split_task_and_task_instruction() -> None:
    dataset = Dataset.from_records(
        name="mem",
        queries=QUERIES,
        corpus=CORPUS,
        qrels=QRELS,
        subset="en",
        split="dev",
        task="MIRACLRetrieval",
        task_instruction={"query": "Given a query, retrieve passages"},
    )

    assert (dataset.subset, dataset.split, dataset.task) == ("en", "dev", "MIRACLRetrieval")
    assert dataset.task_instruction == {"query": "Given a query, retrieve passages"}
    assert dataset.export_key == ("MIRACLRetrieval", "en", "dev")
    assert _dataset().export_key == ("mem", "default", "test"), "the task falls back to the dataset name"


def test_a_task_instruction_must_name_its_sides() -> None:
    with pytest.raises(DataError, match="query.*document"):
        _dataset(task_instruction={"passage": "no such side"})
