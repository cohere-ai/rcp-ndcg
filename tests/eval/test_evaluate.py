"""``evaluate``, ``compare``, ``sensitivity`` and ``explain`` through the public ``rcp_ndcg.eval`` API."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rcp_ndcg_core import gain, ndcg
from rcp_ndcg_core.protocol import score_query
from scipy.stats import ttest_rel

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.eval import EvalReport, compare, evaluate, explain, sensitivity

LOG2_3 = math.log2(3)
ITEMS = {"gamma": [1.0, 1.2, 0.8, 1.1], "beta": [-1.0, -0.5, 0.0, 0.5]}
QUERIES = [f"q{i}" for i in range(6)]
THETAS = {q: {"d1": 1.5, "d2": 0.5, "d3": -0.5, "d4": -1.5} for q in QUERIES}
GAINS = {q: {d: gain(t, ITEMS) for d, t in docs.items()} for q, docs in THETAS.items()}


def _dataset(**fields) -> Dataset:
    defaults = {
        "name": "toy",
        "qrels": {q: {"d1": 2.0, "d2": 1.0, "d3": 0.0, "d4": 0.0} for q in QUERIES},
        "candidates": {q: ["d1", "d2", "d3", "d4"] for q in QUERIES},
    }
    return Dataset(**{**defaults, **fields})


def _rankings() -> Rankings:
    """``good`` ranks by theta, ``bad`` reverses it, ``mixed`` is good on the even queries only."""
    good = {q: {"d1": 4.0, "d2": 3.0, "d3": 2.0, "d4": 1.0} for q in QUERIES}
    bad = {q: {"d1": 1.0, "d2": 2.0, "d3": 3.0, "d4": 4.0} for q in QUERIES}
    mixed = {q: good[q] if int(q[1:]) % 2 == 0 else {"d2": 4.0, "d1": 3.0, "d3": 2.0, "d4": 1.0} for q in QUERIES}
    return Rankings.concat(
        [Rankings.from_scores(scores, system=name) for name, scores in (("good", good), ("bad", bad), ("mixed", mixed))]
    )


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return evaluate(_rankings(), dataset=_dataset(), gains=GAINS, k=[2, 4], bootstrap=200)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_every_value_is_the_protocols_score(report: EvalReport) -> None:
    rankings, dataset = _rankings(), _dataset()
    for row in report.per_query:
        labels = GAINS[row.query_id] if row.metric == "rcp_ndcg" else dataset.qrels[row.query_id]
        expected = score_query(
            rankings.for_query(row.query_id, system=row.system),
            labels,
            protocol="plain",
            k=row.k,
            metric=row.metric,
        )
        assert row.value == expected
    assert report.value("good", "rcp_ndcg", k=4) == pytest.approx(1.0)
    assert report.value("bad", "qrel_ndcg", k=2) == 0.0


def test_the_summary_is_the_mean_per_dataset_then_over_datasets() -> None:
    small = Dataset(name="small", qrels={"q0": {"d1": 1.0}}, gains={"q0": {"d1": 1.0, "d2": 0.0}})
    rankings = Rankings.concat(
        [
            Rankings.from_scores(_rankings().queries(system="mixed"), system="s", dataset="toy"),
            Rankings.from_scores({"q0": {"d2": 2.0, "d1": 1.0}}, system="s", dataset="small"),
        ]
    )
    suite = Dataset(name="suite", subsets=(_dataset(gains=GAINS), small))

    result = evaluate(rankings, dataset=suite, metrics=["rcp_ndcg"], k=2, bootstrap=0)

    per_dataset = {row.dataset: row.value for row in result.per_dataset}
    toy_mean = np.mean([row.value for row in result.per_query if row.dataset == "toy"])
    assert per_dataset == {"toy": pytest.approx(toy_mean), "small": pytest.approx(1 / LOG2_3)}
    assert result.value("s") == pytest.approx((toy_mean + 1 / LOG2_3) / 2), "datasets weigh equally"
    assert result.gains_source == "dataset", "released gains when none are passed"


def test_gains_come_from_the_argument_then_a_calibration_then_the_dataset() -> None:
    dataset = _dataset(gains={q: {"d4": 1.0} for q in QUERIES})
    calibration = SimpleNamespace(gains=lambda: GAINS)

    passed = evaluate(_rankings(), dataset=dataset, gains=GAINS, metrics=["rcp_ndcg"], bootstrap=0)
    calibrated = evaluate(_rankings(), dataset=dataset, gains=calibration, metrics=["rcp_ndcg"], bootstrap=0)
    released = evaluate(_rankings(), dataset=dataset, metrics=["rcp_ndcg"], bootstrap=0)

    assert (passed.gains_source, calibrated.gains_source, released.gains_source) == ("gains", "calibration", "dataset")
    assert passed.value("good") == calibrated.value("good") == pytest.approx(1.0)
    assert released.value("bad") == pytest.approx(1.0), "the released gains favour d4"


def _calibration(thetas: dict[str, dict[str, dict[str, float]]]):
    """A :class:`~rcp_ndcg.calibration.Calibration` holding ``{dataset: {query_id: {doc_id: theta}}}``."""
    from rcp_ndcg_core.schemas import ItemParams

    from rcp_ndcg.calibration import Calibration
    from rcp_ndcg.calibration.fit import ThetaRow

    rows = tuple(
        ThetaRow(dataset=name, query_id=q, doc_id=d, theta=t, theta_se=None, source="fit")
        for name, queries in thetas.items()
        for q, docs in queries.items()
        for d, t in docs.items()
    )
    return Calibration(
        mode="rubric_only",
        items=ItemParams(**ITEMS),
        queries={},
        thetas=rows,
        families={},
        judge_severity={},
        coverage={},
        diagnostics={},
        identity={},
    )


def test_a_calibration_of_several_datasets_scores_every_part_of_a_suite() -> None:
    """Its gains() keys ``<dataset>||<query_id>``; the report once came back empty without an error."""
    suite = Dataset(name="suite", subsets=(_dataset(name="a"), _dataset(name="b")))
    rankings = Rankings.concat(
        [
            Rankings.from_scores(_rankings().queries(system="good"), system="good", dataset="a"),
            Rankings.from_scores(_rankings().queries(system="bad"), system="good", dataset="b"),
        ]
    )

    result = evaluate(rankings, dataset=suite, gains=_calibration({"a": THETAS, "b": THETAS}), k=4, bootstrap=0)

    per_dataset = {(row.dataset, row.metric): row.value for row in result.per_dataset}
    assert per_dataset[("a", "rcp_ndcg")] == pytest.approx(1.0) and per_dataset[("b", "rcp_ndcg")] < 1.0
    assert result.gains_source == "calibration"
    with pytest.raises(DataError, match="no labelled query"):
        evaluate(rankings, dataset=suite, gains=_calibration({"c": THETAS, "d": THETAS}), k=4)


def test_rcp_ndcg_never_falls_back_to_integer_qrels() -> None:
    with pytest.raises(DataError, match="integer qrels are not RCP gains"):
        evaluate(_rankings(), dataset=_dataset())
    with pytest.raises(DataError, match=r"outside \[0, 1\]"):
        evaluate(_rankings(), dataset=_dataset(), gains=_dataset().qrels)
    only_qrels = evaluate(_rankings(), dataset=_dataset(), metrics=["qrel_ndcg"], bootstrap=0)
    assert only_qrels.gains_source == "none" and only_qrels.metrics == ["qrel_ndcg"]


def test_count_ndcg_scores_the_count_gains() -> None:
    count = {q: {"d1": 0.6, "d2": 0.4} for q in QUERIES}
    result = evaluate(_rankings(), dataset=_dataset(), metrics=["count_ndcg"], count_gains=count, k=2, bootstrap=0)

    assert result.value("good", "count_ndcg") == 1.0
    with pytest.raises(DataError, match="count_gains"):
        evaluate(_rankings(), dataset=_dataset(), metrics=["count_ndcg"])


def test_what_the_numbers_do_not_show_is_a_typed_warning() -> None:
    dataset = _dataset(qrels={q: {"d3": 0.0} for q in QUERIES})
    partial = Rankings.from_scores({"q0": {"d1": 1.0}}, system="partial")

    result = evaluate(partial, dataset=dataset, gains=GAINS, k=2, bootstrap=0)

    assert {w.code for w in result.warnings} == {"UNRANKED_QUERIES", "NO_POSITIVE_QRELS"}
    assert result.value("partial", "qrel_ndcg") is None
    assert [r.value for r in result.per_query if r.metric == "rcp_ndcg" and r.query_id == "q1"] == [0.0]


def test_the_protocol_decides_pools_and_ties() -> None:
    tied = Rankings.from_scores({q: {"d1": 1.0, "d2": 1.0, "d3": 0.0, "d4": 0.0, "x": 5.0} for q in QUERIES})
    no_pools = _dataset(candidates=None)

    nanobeir = evaluate(tied, dataset=_dataset(), gains=GAINS, protocol="nanobeir", k=1, metrics=["rcp_ndcg"])
    plain = evaluate(tied, dataset=_dataset(), gains=GAINS, protocol="plain", k=1, metrics=["rcp_ndcg"])

    assert nanobeir.value("system") == pytest.approx(GAINS["q0"]["d2"] / GAINS["q0"]["d1"]), "pool only, id ties"
    assert plain.value("system") == 0.0, "the unjudged x leads when nothing restricts to the pool"
    with pytest.raises(DataError, match="judged pool"):
        evaluate(tied, dataset=no_pools, gains=GAINS, protocol="nanobeir")


def test_bad_arguments_are_config_errors() -> None:
    for kwargs, match in (
        ({"suite": "nanobeir", "dataset": _dataset()}, "not both"),
        ({}, "suite= or dataset="),
        ({"dataset": _dataset(), "k": 0}, "positive"),
        ({"dataset": _dataset(), "metrics": ["mrr"]}, "unknown metrics"),
        ({"dataset": _dataset(), "protocol": "msmarco"}, "unknown protocol"),
    ):
        with pytest.raises(ConfigError, match=match):
            evaluate(_rankings(), gains=GAINS, **kwargs)


def test_the_report_serialises_and_tabulates(report: EvalReport) -> None:
    payload = json.loads(report.to_json())

    assert payload["schema"] == "rcp-ndcg.eval-report.v1"
    assert EvalReport.model_validate(payload).model_dump() == report.model_dump()
    assert len(report.to_pandas()) == 3 * 6 * 2 * 2
    assert list(report.to_pandas("summary").columns)[:4] == ["system", "metric", "k", "value"]


def test_the_leaderboard_is_the_wide_table_and_its_mean_is_the_summary(report: EvalReport) -> None:
    k = report.k[0]
    board = report.leaderboard("rcp_ndcg", k=k)

    assert list(board.columns)[-1] == "mean"
    assert sorted(board.index) == sorted(report.systems)
    assert list(board["mean"]) == sorted(board["mean"], reverse=True)
    for system in report.systems:
        assert board.loc[system, "mean"] == pytest.approx(report.value(system, "rcp_ndcg", k))
    assert repr(report).startswith(f"EvalReport(protocol={report.protocol.name}, {len(report.systems)} systems")


def test_the_summary_interval_is_seeded_and_brackets_the_value() -> None:
    first = evaluate(_rankings(), dataset=_dataset(), gains=GAINS, bootstrap=300, seed=3)
    again = evaluate(_rankings(), dataset=_dataset(), gains=GAINS, bootstrap=300, seed=3)

    assert first.summary == again.summary
    for row in first.summary:
        assert row.ci_low - 1e-12 <= row.value <= row.ci_high + 1e-12


def test_evaluate_reproduces_the_papers_per_query_values() -> None:
    """The paper anchors of ``tests/core/test_protocol.py``, scored through ``evaluate``."""
    fixture = json.loads((Path(__file__).parents[1] / "core" / "fixtures" / "paper_protocol_queries.json").read_text())
    for query in fixture["queries"]:
        dataset = Dataset(
            name=query["dataset"],
            qrels={query["query_id"]: query["qrels"]},
            gains={query["query_id"]: query["gains"]},
            candidates={query["query_id"]: query["candidates"]} if query.get("candidates") else None,
            excluded={query["query_id"]: list(query["excluded"])},
        )
        rankings = Rankings.from_scores({query["query_id"]: query["scores"]})

        result = evaluate(rankings, dataset=dataset, protocol=query["suite"], k=fixture["k"], bootstrap=0)

        assert result.value("system", "rcp_ndcg") == pytest.approx(
            query["expected"]["rcp_ndcg"], abs=query["rcp_ndcg_tolerance"]
        )
        assert result.value("system", "qrel_ndcg") == pytest.approx(query["expected"]["qrel_ndcg"], abs=1e-12)


# ---------------------------------------------------------------------------
# compare and sensitivity
# ---------------------------------------------------------------------------


def test_deltas_are_b_minus_a_with_the_papers_paired_t_test(report: EvalReport) -> None:
    result = compare(report, baseline="bad", k=4, bootstrap=500)

    pair = next(p for p in result.pairs if p.system_b == "good")
    good = [r.value for r in report.per_query if (r.system, r.metric, r.k) == ("good", "rcp_ndcg", 4)]
    mixed = [r.value for r in report.per_query if (r.system, r.metric, r.k) == ("mixed", "rcp_ndcg", 4)]
    mixed_pair = next(p for p in result.pairs if p.system_b == "mixed")
    assert [(p.system_a, p.system_b) for p in result.pairs] == [("bad", "good"), ("bad", "mixed")]
    assert pair.delta == pytest.approx(report.value("good", k=4) - report.value("bad", k=4))
    assert pair.delta > 0 and pair.ci_low - 1e-12 <= pair.delta <= pair.ci_high + 1e-12
    assert pair.significant and pair.p_value == 0.0, "scipy's test: a constant gain on every query"
    assert mixed_pair.p_value == pytest.approx(
        ttest_rel(
            mixed, [r.value for r in report.per_query if (r.system, r.metric, r.k) == ("bad", "rcp_ndcg", 4)]
        ).pvalue
    )
    assert good != mixed


def test_all_pairs_follow_the_report_order_and_list_sign_flips() -> None:
    qrels = {q: {"d2": 1.0} for q in QUERIES}  # the humans prefer d2, the gains prefer d1
    result = evaluate(_rankings(), dataset=_dataset(qrels=qrels), gains=GAINS, k=1, bootstrap=0)

    comparison = compare(result, bootstrap=0)

    assert [(p.system_a, p.system_b) for p in comparison.pairs] == [
        ("good", "bad"),
        ("good", "mixed"),
        ("bad", "mixed"),
    ]
    flips = next(p for p in comparison.pairs if p.system_b == "mixed").sign_flips
    assert {f.query_id for f in flips} == {"q1", "q3", "q5"}
    assert all(f.delta_rcp_ndcg < 0 < f.delta_qrel_ndcg for f in flips)
    assert list(comparison.to_pandas().columns)[:3] == ["system_a", "system_b", "value_a"]


def test_compare_refuses_what_it_cannot_do(report: EvalReport) -> None:
    with pytest.raises(DataError, match="pass k="):
        compare(report)
    with pytest.raises(ConfigError, match="baseline"):
        compare(report, baseline="nobody", k=2)
    with pytest.raises(DataError, match="count_ndcg"):
        compare(report, metric="count_ndcg", k=2)


def test_sensitivity_is_the_share_of_pairs_a_paired_t_test_separates(report: EvalReport) -> None:
    values = {
        s: [r.value for r in report.per_query if (r.system, r.metric, r.k) == (s, "qrel_ndcg", 4)]
        for s in ("good", "bad", "mixed")
    }
    pairs = (("good", "bad"), ("good", "mixed"), ("bad", "mixed"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.mean([ttest_rel(values[a], values[b]).pvalue < 0.05 for a, b in pairs])

    assert sensitivity(report, metric="qrel_ndcg", k=4) == pytest.approx(expected)


def test_sensitivity_is_the_mean_over_datasets_of_each_datasets_share_of_separated_pairs() -> None:
    """The paper's definition: per dataset, the share of all system pairs whose paired t-test has p < alpha (a pair
    without two shared queries is not separated), then the mean over datasets."""
    from rcp_ndcg_core.protocol import PROTOCOLS

    from rcp_ndcg.eval import QueryValue, SummaryValue

    x_a = [0.1, 0.2, 0.3, 0.4]
    per_system = {
        ("a", "x"): x_a,
        ("a", "y"): [v + d for v, d in zip(x_a, [0.5, 0.52, 0.49, 0.51], strict=True)],  # separated from x and z
        ("a", "z"): [v + d for v, d in zip(x_a, [0.01, -0.02, 0.015, -0.005], strict=True)],  # not from x
        ("b", "x"): [0.2, 0.5, 0.3],
        ("b", "y"): [0.25, 0.45, 0.33],  # not separated from x
        ("b", "z"): [0.9],  # one shared query: no test, not separated
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert ttest_rel(per_system[("a", "x")], per_system[("a", "y")]).pvalue < 0.05
        assert ttest_rel(per_system[("a", "y")], per_system[("a", "z")]).pvalue < 0.05
        assert ttest_rel(per_system[("a", "x")], per_system[("a", "z")]).pvalue >= 0.05
        assert ttest_rel(per_system[("b", "x")], per_system[("b", "y")]).pvalue >= 0.05
    rows = [
        QueryValue(system=s, dataset=d, query_id=f"q{i}", metric="qrel_ndcg", k=10, value=v)
        for (d, s), values in per_system.items()
        for i, v in enumerate(values)
    ]
    summary = [
        SummaryValue(system=s, metric="qrel_ndcg", k=10, value=0.0, ci_low=None, ci_high=None, num_queries=0,
                     num_datasets=2)
        for s in ("x", "y", "z")
    ]  # fmt: skip
    report = EvalReport(protocol=PROTOCOLS["plain"], gains_source="none", metrics=["qrel_ndcg"], k=[10],
                        per_query=rows, per_dataset=[], summary=summary)  # fmt: skip

    # a: 2 of 3 pairs, b: 0 of 3 pairs; the mean over the two datasets is 1/3.
    assert sensitivity(report, metric="qrel_ndcg") == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_shows_each_systems_top_k_with_criteria_and_splits_the_gap(report: EvalReport) -> None:
    result = explain(report, "q1", calibration=_calibration({"toy": THETAS}), k=2)

    good, bad, mixed = result.systems
    assert [d.doc_id for d in good.top] == ["d1", "d2"] and [d.doc_id for d in bad.top] == ["d4", "d3"]
    first = good.top[0]
    assert (first.gain, first.grade, first.theta) == (GAINS["q1"]["d1"], 2.0, 1.5)
    assert sum(c.contribution for c in first.criteria) == pytest.approx(first.gain)
    assert good.values["rcp_ndcg@4"] == 1.0
    delta = next(d for d in result.deltas if d.system_b == "mixed")
    assert delta.total == pytest.approx(ndcg(["d2", "d1", "d3", "d4"], GAINS["q1"], k=2) - 1.0)
    assert delta.selection == pytest.approx(0.0) and delta.ordering == pytest.approx(delta.total)


def test_explain_needs_a_report_from_evaluate(report: EvalReport) -> None:
    with pytest.raises(DataError, match="use a report returned by evaluate"):
        explain(EvalReport.model_validate_json(report.to_json()), "q1")
    with pytest.raises(DataError, match=r"query 'nope' is not in the report; known: q0, q1, q2"):
        explain(report, "nope")


def test_a_non_finite_score_is_a_data_error_not_a_crash() -> None:
    """The metric's bare ValueError once reached the CLI as exit 1, "a bug": rankings refuse the score."""
    with pytest.raises(DataError, match="not finite") as caught:
        Rankings.from_scores({"q0": {"d1": float("nan"), "d2": 1.0}}, system="s")

    assert caught.value.details["query_id"] == "q0"
