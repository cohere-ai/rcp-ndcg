"""The metric seam: the ``rcp.metrics`` group, the shipped ten and the k rule.

The shipped ten stay the registry's first members and are scored through it, so a plugin and a built-in
resolve through one lookup and one duplicate-name policy. A plugin is scored on the dataset's qrels: at every
requested cutoff when its ``k_required`` is true, once (at ``k=0``, "no cutoff") when it is false.

The seven corpus metrics are selectable by name too, and each one's value is its ``rcp_core.metric`` function's,
scored under the metric's own published tie rule rather than the protocol's. The three that need an input the
seam does not carry are not names here.

The last test pins the four shipped suites' reports byte for byte against the base tree's output
(``fixtures/suite_reports/``), so the seam cannot move a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_core import metric as core_metric

from rcp.data import Dataset, Rankings
from rcp.errors import ConfigError, DataError, ExitCode
from rcp.eval import METRICS, Metric, evaluate, metric_class, registered_metrics, sensitivity
from tests._helpers._metrics import MeanAveragePrecision, install_metrics, metric_entry

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "_helpers" / "fixtures" / "paper_protocol_queries.json"
GOLDEN = Path(__file__).parent / "fixtures" / "suite_reports"
SUITES = ("nanobeir", "bright", "vidore", "trecdl")

#: The seven corpus metrics of :mod:`rcp_core.metric` the seam scores: name -> the function it calls.
CORPUS_METRICS = {
    "ndcg_float": core_metric.ndcg_float,
    "map": core_metric.map,
    "recall": core_metric.recall,
    "precision": core_metric.precision,
    "success": core_metric.success,
    "mrr": core_metric.mrr,
    "recall_cap": core_metric.recall_cap,
}

#: The three the seam cannot score: each needs an input one query's labels and ranking do not carry.
NOT_SEAM_METRICS = ("hole", "paired_accuracy", "coverage")

#: The specification's corpus: ``q1``'s tie straddles ``k=2`` and ``q2``'s tie sits at the top.
TIE_QRELS = {"q1": {"a": 2.0, "b": 1.0, "c": 0.0, "d": 1.0, "e": 0.0}, "q2": {"b": 1.0, "c": 1.0, "d": 0.0}}
TIE_SCORES = {
    "q1": {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.4, "e": 0.2},
    "q2": {"a": 0.8, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2},
}


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


def test_the_shipped_ten_start_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch)

    assert METRICS == ("rcp_ndcg", "qrel_ndcg", "count_ndcg", *CORPUS_METRICS)
    assert registered_metrics() == METRICS, "the shipped tuple is the registry's start"
    for name in METRICS:
        assert issubclass(metric_class(name), Metric)


def test_a_shipped_corpus_metric_is_selectable_and_is_its_function_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each of the seven is a name in the registry, and its value is ``rcp_core.metric``'s own: one body."""
    install_metrics(monkeypatch)
    dataset = Dataset(name="toy", qrels=TIE_QRELS)
    rankings = Rankings.from_scores(TIE_SCORES, system="tied")

    report = evaluate(rankings, dataset=dataset, metrics=list(CORPUS_METRICS), k=[2], bootstrap=0)

    assert report.metrics == list(CORPUS_METRICS)
    for name, function in CORPUS_METRICS.items():
        assert report.value("tied", name, k=2) == pytest.approx(function(TIE_QRELS, TIE_SCORES, k=2)), name


def test_a_corpus_metrics_function_does_not_bind_the_instance() -> None:
    """A corpus metric's ``function`` is a plain function on the class, not a descriptor that binds ``self``.

    Python 3.14's ``functools.partial`` implements the descriptor protocol, so a ``partial`` stored as a class
    attribute is read through the instance as ``partial(function, self)``: :meth:`_CorpusMetric.score`'s
    ``self.function({query: labels}, {query: ordered}, k=k)`` then passes three positional arguments to a corpus
    function that takes two, and every one of the seven raises ``TypeError`` on 3.14 (green on 3.12, which is
    why the interpreter's change reached ``main``). ``staticmethod`` is the declaration that the attribute is a
    plain function; this test pins it where 3.12 cannot show the defect.
    """
    for name, function in CORPUS_METRICS.items():
        metric = metric_class(name)()
        stored = vars(type(metric))["function"]
        assert isinstance(stored, staticmethod), (
            f"{name}.function is a bare {type(stored).__name__}: on Python 3.14 an instance access binds self "
            "as a third positional argument"
        )
        # What an instance access goes through and what a class access goes through must be one object: a 3.14
        # partial returns ``partial(function, self)`` from the first and the partial itself from the second.
        assert stored.__get__(metric, type(metric)) is stored.__get__(None, type(metric)), "no instance is bound"
        assert metric.function(TIE_QRELS, TIE_SCORES, k=2) == pytest.approx(function(TIE_QRELS, TIE_SCORES, k=2))


def test_a_corpus_metric_keeps_its_own_tie_rule_under_another_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protocol cannot choose a corpus metric's tie rule: ``map`` orders a tie by document id (0.4166...)
    and ``mrr`` keeps the run's order (0.5), while the protocol's own rule is ``group_mean``."""
    install_metrics(monkeypatch)
    dataset = Dataset(name="toy", qrels=TIE_QRELS)
    rankings = Rankings.from_scores(TIE_SCORES, system="tied")

    report = evaluate(rankings, dataset=dataset, metrics=["map", "mrr"], protocol="plain", k=[1, 2], bootstrap=0)

    assert report.protocol.ties == "group_mean", "the protocol's rule is the one neither metric may take"
    assert report.value("tied", "map", k=2) == pytest.approx(0.41666666666666663)
    assert report.value("tied", "map", k=2) != pytest.approx(0.5833333333333333), "that is the stable order"
    assert report.value("tied", "mrr", k=1) == pytest.approx(0.5), "q2's tie keeps a before b: the hit is second"
    assert report.value("tied", "mrr", k=1) != pytest.approx(1.0), "that is the document-id order"


def test_the_input_order_metrics_score_the_run_order_not_the_pool_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mrr`` and ``recall_cap`` order a tie by the run's insertion order, which a judged pool's own order does
    not supply: the pool lists the only relevant document first (1.0), the run lists it second (0.0)."""
    install_metrics(monkeypatch)
    qrels = {"q1": {"b": 1.0}}
    scores = {"q1": {"a": 0.5, "b": 0.5}}
    dataset = Dataset(name="toy", qrels=qrels, candidates={"q1": ["b", "a"]})
    rankings = Rankings.from_scores(scores, system="tied")

    report = evaluate(rankings, dataset=dataset, metrics=["mrr", "recall_cap"], protocol="plain", k=[1], bootstrap=0)

    assert core_metric.mrr(qrels, scores, k=1) == 0.0, "the core reads the run's order: b is ranked second"
    assert report.value("tied", "mrr", k=1) == pytest.approx(0.0), "the seam is the core's value"
    assert report.value("tied", "mrr", k=1) != pytest.approx(1.0), "that is the pool's order, which is no ranking"
    assert report.value("tied", "recall_cap", k=1) == pytest.approx(0.0), "the same rule, the same value"
    assert report.value("tied", "recall_cap", k=1) != pytest.approx(1.0), "that is the pool's order"


def test_sensitivity_names_a_shipped_corpus_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sensitivity(metric=...)`` reads a metric the report holds: a corpus metric is one, and the pair here
    separates (``bad`` loses half a MAP@2 on every query, by a different amount on each)."""
    install_metrics(monkeypatch)
    qrels = {f"q{i}": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0} for i in range(6)}
    dataset = Dataset(name="toy", qrels=qrels)
    good = {query_id: {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0} for query_id in qrels}
    bad = {
        query_id: {"a": 4.0, "c": 3.0, "b": 2.0, "d": 1.0} if i % 2 else {"c": 4.0, "a": 3.0, "b": 2.0, "d": 1.0}
        for i, query_id in enumerate(qrels)
    }
    rankings = Rankings.concat([Rankings.from_scores(good, system="good"), Rankings.from_scores(bad, system="bad")])

    report = evaluate(rankings, dataset=dataset, metrics=["map"], protocol="plain", k=[2], bootstrap=0)

    assert report.value("good", "map", k=2) == pytest.approx(1.0)
    assert report.value("bad", "map", k=2) == pytest.approx(0.375), "half the queries lose half a hit"
    assert sensitivity(report, metric="map", k=2) == pytest.approx(1.0)


def test_a_labelled_query_the_run_does_not_rank_is_scored_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam's denominator is the dataset's labelled queries: an unranked one is 0.0 and the report warns,
    where the standalone function divides by the queries the run carries."""
    install_metrics(monkeypatch)
    dataset = Dataset(name="toy", qrels=TIE_QRELS)
    rankings = Rankings.from_scores({"q1": TIE_SCORES["q1"]}, system="partial")

    report = evaluate(rankings, dataset=dataset, metrics=["map"], k=[2], bootstrap=0)

    ranked = core_metric.map({"q1": TIE_QRELS["q1"]}, {"q1": TIE_SCORES["q1"]}, k=2)
    assert report.value("partial", "map", k=2) == pytest.approx(ranked / 2), "q1's value, then q2's 0.0"
    assert ranked / 2 != pytest.approx(core_metric.map(TIE_QRELS, TIE_SCORES, k=2)), "the function's own mean"
    assert [warning.code for warning in report.warnings] == ["UNRANKED_QUERIES"], "the report says it scored 0"
    assert report.value("partial", "map", k=2) is not None, "an unranked labelled query is a 0.0, not a dropped row"


def test_the_metrics_that_need_more_than_a_query_are_not_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """``hole``, ``paired_accuracy`` and ``coverage`` are public in ``rcp_core.metric`` and not selectable here."""
    install_metrics(monkeypatch)
    dataset = Dataset(name="toy", qrels=TIE_QRELS)
    rankings = Rankings.from_scores(TIE_SCORES, system="tied")

    for name in NOT_SEAM_METRICS:
        assert name not in registered_metrics()
        assert callable(getattr(core_metric, name)), "public where it lives"
        with pytest.raises(ConfigError, match="unknown metrics") as caught:
            evaluate(rankings, dataset=dataset, metrics=[name], k=[2], bootstrap=0)
        assert "ndcg_float" in str(caught.value), "the refusal names the registered set"


def test_an_unknown_metric_class_names_the_registered_set() -> None:
    """``metric_class`` is the lookup both ``evaluate`` and the CLI go through: a typo names every metric."""
    with pytest.raises(ConfigError, match="unknown metric") as caught:
        metric_class("mapp")

    message = str(caught.value)
    for name in METRICS:
        assert name in message, f"{name} is registered, so the refusal lists it"


def test_a_plugin_registers_and_scores_a_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("toy_metric", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    assert registered_metrics() == (*METRICS, "toy_metric")
    assert metric_class("toy_metric") is MeanAveragePrecision

    report = evaluate(rankings, dataset=dataset, metrics=["toy_metric"], k=[2], bootstrap=0)

    assert report.metrics == ["toy_metric"] and report.gains_source == "none"
    assert report.value("oracle", "toy_metric", k=2) == pytest.approx(1.0), "both relevant documents at the top"
    assert report.value("late", "toy_metric", k=2) == pytest.approx(0.25), "(1/2) / min(2 relevant, k=2)"


def test_a_plugin_that_takes_a_cutoff_is_scored_at_every_requested_one(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("toy_metric", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["toy_metric"], k=[1, 2], bootstrap=0)

    assert {(row.metric, row.k) for row in report.per_query} == {("toy_metric", 1), ("toy_metric", 2)}
    assert report.value("oracle", "toy_metric", k=1) == pytest.approx(1.0)
    assert report.value("late", "toy_metric", k=1) == pytest.approx(0.0), "the only slot holds a miss"


def test_a_plugin_that_ignores_the_cutoff_is_scored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("toy_metric_no_cutoff", "tests._helpers._metrics:MeanReciprocalRank"))
    dataset, rankings = _fixture()

    report = evaluate(rankings, dataset=dataset, metrics=["toy_metric_no_cutoff"], k=[1, 2], bootstrap=0)

    assert {row.k for row in report.per_query} == {0}, "one row, and k=0 says the metric takes no cutoff"
    one_row = {("toy_metric_no_cutoff", 0)}
    assert {(row.metric, row.k) for row in report.summary} == one_row, "one summary row per system"
    assert report.value("oracle", "toy_metric_no_cutoff") == pytest.approx(1.0)
    assert report.value("late", "toy_metric_no_cutoff") == pytest.approx(0.5), "the first relevant document is second"


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

    message = str(caught.value)
    assert "'misnamed'" in message and "'toy_metric'" in message, "both names are in the message"


def test_an_unknown_metric_names_the_registered_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install_metrics(monkeypatch, metric_entry("toy_metric", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset, rankings = _fixture()

    with pytest.raises(ConfigError, match="unknown metrics") as caught:
        evaluate(rankings, dataset=dataset, metrics=["nope"], bootstrap=0)

    assert "toy_metric" in str(caught.value), "the installed plugins are listed, so a typo is fixable"


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


def test_a_metric_that_refuses_the_input_is_a_data_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corpus metric refuses a negative qrel grade with a ``ValueError``: the seam translates it, so the
    caller gets the contract's ``DataError`` (exit 12) naming the metric and the query, not an internal error."""
    install_metrics(monkeypatch)
    dataset = Dataset(name="toy", qrels={"q1": {"a": -1.0, "b": 1.0}})
    rankings = Rankings.from_scores({"q1": {"a": 2.0, "b": 1.0}}, system="s")

    with pytest.raises(DataError, match="'mrr'.*'q1'.*the qrel grade must be at least 0") as caught:
        evaluate(rankings, dataset=dataset, metrics=["mrr"], protocol="plain", k=[1], bootstrap=0)

    assert caught.value.exit_code == ExitCode.DATA, "a refused input is DATA/12, not INTERNAL/1"


def test_a_plugin_value_that_is_undefined_does_not_warn_about_qrels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin returns ``None`` for an undefined query; the qrel-nDCG warning must not claim it."""
    install_metrics(monkeypatch, metric_entry("toy_metric", "tests._helpers._metrics:MeanAveragePrecision"))
    dataset = Dataset(name="toy", qrels={"q0": {"a": 0.0, "b": 0.0}}, candidates={"q0": ["a", "b"]})
    rankings = Rankings.from_scores({"q0": {"a": 1.0, "b": 0.0}}, system="s")

    report = evaluate(rankings, dataset=dataset, metrics=["toy_metric"], k=[1], bootstrap=0)

    assert report.warnings == []
    assert report.value("s", "toy_metric") is None, "no relevant document: the value is undefined, not 0"


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
