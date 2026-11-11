"""The metric seam: the ``rcp.metrics`` group, the shipped three and the k rule.

The shipped three stay the registry's first members and are scored through it, so a plugin and a built-in
resolve through one lookup and one duplicate-name policy. A plugin is scored on the dataset's qrels: at every
requested cutoff when its ``k_required`` is true, once (at ``k=0``, "no cutoff") when it is false.

The last test pins the four shipped suites' reports byte for byte against the base tree's output
(``fixtures/suite_reports/``), so the seam cannot move a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp.data import Dataset, Rankings
from rcp.errors import ConfigError, DataError
from rcp.eval import METRICS, Metric, evaluate, metric_class, registered_metrics
from tests._helpers._metrics import MeanAveragePrecision, install_metrics, metric_entry

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "_helpers" / "fixtures" / "paper_protocol_queries.json"
GOLDEN = Path(__file__).parent / "fixtures" / "suite_reports"
SUITES = ("nanobeir", "bright", "vidore", "trecdl")


def _fixture() -> tuple[Dataset, Rankings]:
    """One query, two relevant documents of three; ``oracle`` ranks them first, ``late`` puts a miss first."""
    dataset = Dataset(
        name="toy",
        qrels={"q0": {"a": 1.0, "b": 0.0, "c": 1.0}},
        candidates={"q0": ["a", "b", "c"]},
    )
    rankings = Rankings.concat(
        [
            Rankings.from_scores({"q0": {"a": 3.0, "c": 2.0, "b": 1.0}}, system="oracle"),
            Rankings.from_scores({"q0": {"b": 3.0, "a": 2.0, "c": 1.0}}, system="late"),
        ]
    )
    return dataset, rankings


def test_the_shipped_three_start_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch)

    assert METRICS == ("rcp_ndcg", "qrel_ndcg", "count_ndcg")
    assert registered_metrics() == METRICS, "the shipped tuple is the registry's start"
    for name in METRICS:
        assert issubclass(metric_class(name), Metric)


def test_a_plugin_registers_and_scores_a_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("map", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    assert registered_metrics() == (*METRICS, "map")
    assert metric_class("map") is MeanAveragePrecision

    report = evaluate(rankings, dataset=dataset, metrics=["map"], k=[2], bootstrap=0)

    assert report.metrics == ["map"] and report.gains_source == "none"
    assert report.value("oracle", "map", k=2) == pytest.approx(1.0), "both relevant documents at the top"
    assert report.value("late", "map", k=2) == pytest.approx(0.25), "(1/2) / min(2 relevant, k=2)"


def test_a_plugin_that_takes_a_cutoff_is_scored_at_every_requested_one(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("map", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["map"], k=[1, 2], bootstrap=0)

    assert {(row.metric, row.k) for row in report.per_query} == {("map", 1), ("map", 2)}
    assert report.value("oracle", "map", k=1) == pytest.approx(1.0)
    assert report.value("late", "map", k=1) == pytest.approx(0.0), "the only slot holds a miss"


def test_a_plugin_that_ignores_the_cutoff_is_scored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("mrr", "tests._helpers._metrics:MeanReciprocalRank"))
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["mrr"], k=[1, 2], bootstrap=0)

    assert {row.k for row in report.per_query} == {0}, "one row, and k=0 says the metric takes no cutoff"
    assert {(row.metric, row.k) for row in report.summary} == {("mrr", 0)}, "one summary row per system"
    assert report.value("oracle", "mrr") == pytest.approx(1.0)
    assert report.value("late", "mrr") == pytest.approx(0.5), "the first relevant document is second"


def test_the_shipped_metrics_are_scored_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin that takes ``rcp_ndcg`` makes the shipped name ambiguous: the refusal is the one policy's, not a
    silent choice of the built-in (which is what scoring the built-ins directly would do)."""
    install_metrics(monkeypatch, metric_entry("rcp_ndcg", "tests._helpers._metrics:CollidingRcp"))
    dataset, _ = _fixture()
    rankings = Rankings.from_scores({"q0": {"a": 1.0}}, system="s")

    with pytest.raises(ConfigError, match="more than one installed distribution") as caught:
        metric_class("rcp_ndcg")
    message = str(caught.value)
    assert "rcp.eval.metrics:RcpNdcg" in message and "tests._helpers._metrics:CollidingRcp" in message
    assert registered_metrics() == METRICS, "the listing names each name once: the collision is not listed twice"
    with pytest.raises(ConfigError, match="more than one installed distribution"):
        evaluate(rankings, dataset=dataset, metrics=["rcp_ndcg"], gains=dataset.qrels, bootstrap=0)


def test_a_plugin_whose_name_disagrees_with_its_entry_point_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry-point name is the metric name: a class that calls itself something else is refused, so a
    report's metric name always resolves back to the class that scored it."""
    install_metrics(monkeypatch, metric_entry("misnamed", "tests._helpers._metrics:MeanAveragePrecision"))

    with pytest.raises(ConfigError, match="must agree") as caught:
        metric_class("misnamed")

    assert "'misnamed'" in str(caught.value) and "'map'" in str(caught.value), "both names are in the message"


def test_an_unknown_metric_names_the_registered_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("map", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    with pytest.raises(ConfigError, match="unknown metrics") as caught:
        evaluate(rankings, dataset=dataset, metrics=["nope"], bootstrap=0)

    assert "map" in str(caught.value), "the installed plugins are listed, so a typo is fixable"


def test_an_entry_point_that_is_not_a_metric_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("wrong", "rcp.results:ResultRecord"))

    with pytest.raises(ConfigError, match="not a Metric"):
        metric_class("wrong")


def test_the_registry_reads_the_one_group_through_the_shared_scan() -> None:
    """The packaging scan resolves the module's ``group=`` constant: the seventh group is a real read, not a
    string in a docstring."""
    from rcp.eval import metrics
    from tests.contract.surface import entry_point_groups

    assert entry_point_groups(Path(metrics.__file__)) == {"rcp.metrics"}


@pytest.mark.parametrize("suite", SUITES)
def test_the_shipped_suite_report_is_byte_identical_to_the_base(suite: str) -> None:
    """The four suites' reports were captured from the base tree; the seam must not move a byte of them."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    queries = [query for query in fixture["queries"] if query["suite"] == suite]
    dataset = Dataset(
        name=suite,
        qrels={query["query_id"]: query["qrels"] for query in queries},
        gains={query["query_id"]: query["gains"] for query in queries},
        candidates={query["query_id"]: query["candidates"] for query in queries if query.get("candidates")} or None,
        excluded={query["query_id"]: list(query["excluded"]) for query in queries if query.get("excluded")},
    )
    rankings = Rankings.from_scores({query["query_id"]: query["scores"] for query in queries}, system="system")

    report = evaluate(rankings, dataset=dataset, protocol=suite, k=fixture["k"], bootstrap=0)

    payload = (report.to_json(indent=2) + "\n").encode("utf-8")
    assert payload == (GOLDEN / f"{suite}.json").read_bytes()


def test_a_duplicate_metric_name_is_scored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name given twice is one metric, never two rows: the CLI's ``--metric`` joins ``--metrics``, so a
    name can arrive twice."""
    install_metrics(monkeypatch)
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["qrel_ndcg", "qrel_ndcg"], k=[1], bootstrap=0)

    assert report.metrics == ["qrel_ndcg"]
    assert len([row for row in report.summary if row.metric == "qrel_ndcg"]) == 2, "one per system"


def test_a_plugin_that_returns_a_non_number_is_a_data_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam's contract is ``float | None``; a plugin that returns something else is named, not crashed on."""
    install_metrics(monkeypatch, metric_entry("not_a_number", "tests._helpers._metrics:NotANumber"))
    dataset, rankings = _fixture()

    with pytest.raises(DataError, match="not_a_number.*float"):
        evaluate(rankings, dataset=dataset, metrics=["not_a_number"], k=[1], bootstrap=0)


def test_a_plugin_value_that_is_undefined_does_not_warn_about_qrels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin returns ``None`` for an undefined query; the qrel-nDCG warning must not claim it."""
    install_metrics(monkeypatch, metric_entry("map", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset = Dataset(name="toy", qrels={"q0": {"a": 0.0, "b": 0.0}}, candidates={"q0": ["a", "b"]})
    rankings = Rankings.from_scores({"q0": {"a": 1.0, "b": 0.0}}, system="s")

    report = evaluate(rankings, dataset=dataset, metrics=["map"], k=[1], bootstrap=0)

    assert report.warnings == []
    assert report.value("s", "map") is None, "no relevant document: the value is undefined, not 0"


def test_a_plugin_value_of_nan_is_undefined_not_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nan`` is the contract's other spelling of undefined: the row drops out as ``None``, with no warning,
    where ``inf`` and a non-number are refused. Pinned because :meth:`rcp.eval.Metric.score` promises it."""
    install_metrics(monkeypatch, metric_entry("nan_value", "tests._helpers._metrics:NanValue"))
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["nan_value"], k=[1], bootstrap=0)

    assert report.warnings == []
    assert report.value("oracle", "nan_value") is None
    assert {(row.metric, row.value) for row in report.summary} == {("nan_value", None)}
    assert {row.num_queries for row in report.summary} == {0}, "a NaN row is no value, not a value of NaN"
