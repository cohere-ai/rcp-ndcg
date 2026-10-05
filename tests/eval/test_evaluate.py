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
    with pytest.raises(DataError, match="has none of them"):
        evaluate(rankings, dataset=suite, gains=_calibration({"c": THETAS, "d": THETAS}), k=4)


def _bright_like() -> Dataset:
    """Two subsets whose query ids coincide ("0", "1") over disjoint corpora, as BRIGHT's subsets do."""
    parts = tuple(
        Dataset(
            name=name,
            qrels={q: {f"{name}-a": 1.0, f"{name}-b": 0.0} for q in ("0", "1")},
            gains={q: {f"{name}-a": 0.9, f"{name}-b": 0.1} for q in ("0", "1")},
        )
        for name in ("biology", "earth_science")
    )
    return Dataset(name="bright-like", subsets=parts)


def _oracle(*, divided: bool) -> Rankings:
    return Rankings.from_records(
        {"query_id": q, "doc_id": f"{name}-{d}", "score": score, **({"dataset": name} if divided else {})}
        for name in ("biology", "earth_science")
        for q in ("0", "1")
        for d, score in (("a", 2.0), ("b", 1.0))
    )


def test_rankings_without_a_dataset_are_refused_over_subsets_that_share_query_ids() -> None:
    """Undivided, both subsets' documents merged into one ranking per query id and an oracle scored far below 1."""
    suite = _bright_like()

    with pytest.raises(DataError, match="name no dataset") as refused:
        evaluate(_oracle(divided=False), dataset=suite, k=1, bootstrap=0)

    assert refused.value.details["subsets"] == ["biology", "earth_science"]
    assert "--subset" in refused.value.cli_hint and "`dataset` column" in refused.value.hint
    assert evaluate(_oracle(divided=True), dataset=suite, k=1, bootstrap=0).value("system") == 1.0
    biology_run = Rankings.from_scores(_oracle(divided=True).queries(dataset="biology"))  # a TREC run of one subset
    one_subset = evaluate(biology_run, dataset=suite.subsets[0], k=1, bootstrap=0)
    assert one_subset.value("system") == 1.0, "a subset scored alone reads the undivided rows"


def test_rankings_without_a_dataset_score_a_suite_whose_query_ids_are_unique() -> None:
    parts = (_dataset(name="a"), _dataset(name="b", qrels={"p0": {"d1": 1.0}}, candidates={"p0": ["d1"]}))
    rankings = Rankings.from_scores({**_rankings().queries(system="good"), "p0": {"d1": 1.0}})

    result = evaluate(rankings, dataset=Dataset(name="s", subsets=parts), metrics=["qrel_ndcg"], k=4, bootstrap=0)

    assert result.value("system", "qrel_ndcg") == 1.0


def test_a_calibration_of_one_dataset_scores_only_the_subset_of_that_name() -> None:
    suite = _bright_like()
    thetas = {q: {"biology-a": 2.0, "biology-b": -2.0} for q in ("0", "1")}

    result = evaluate(_oracle(divided=True), dataset=suite, gains=_calibration({"biology": thetas}), k=1, bootstrap=0)

    assert {row.dataset for row in result.per_dataset if row.metric == "rcp_ndcg"} == {"biology"}
    assert result.value("system") == 1.0
    released = evaluate(_oracle(divided=True), dataset=suite, k=1, bootstrap=0)
    same_ids = _calibration({"biology": {"0": {"earth_science-a": 2.0}}})  # a document id both subsets could hold
    explained = explain(released, "0", dataset="earth_science", calibration=same_ids)
    assert {doc.doc_id: doc.theta for doc in explained.systems[0].top}["earth_science-a"] is None, (
        "biology's thetas never explain earth_science"
    )
    with pytest.raises(DataError, match="has none of them"):
        evaluate(_oracle(divided=True), dataset=suite, gains=_calibration({"physics": thetas}), k=1)


def test_gains_keyed_by_bare_query_ids_are_refused_over_subsets_that_share_them() -> None:
    suite = _bright_like()
    bare = {"0": {"biology-a": 1.0}}

    with pytest.raises(DataError, match="bare query ids"):
        evaluate(_oracle(divided=True), dataset=suite, gains=bare, k=1)
    keyed = {f"{p.name}/{q}": docs for p in suite.subsets for q, docs in (p.gains or {}).items()}
    assert evaluate(_oracle(divided=True), dataset=suite, gains=keyed, k=1, bootstrap=0).value("system") == 1.0


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
    counts = {row.metric: row.num_queries for row in result.per_dataset}
    summary = {row.metric: row.num_queries for row in result.summary}
    assert counts == summary == {"rcp_ndcg": 6, "qrel_ndcg": 0}, "both count the queries with a defined value"
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
# rankings that match nothing are refused (issue #5)
# ---------------------------------------------------------------------------


def _vidore_like(**fields) -> Dataset:
    """One subset named like ViDoRe's, with pools, qrels and released gains on `corpus-test-` ids."""
    defaults = {
        "name": "hr__english",
        "qrels": {
            "q1": {"corpus-test-1": 1.0, "corpus-test-2": 0.0, "corpus-test-3": 0.0},
            "q2": {"corpus-test-1": 0.0, "corpus-test-2": 1.0, "corpus-test-3": 0.0},
        },
        "gains": {
            "q1": {"corpus-test-1": 0.9, "corpus-test-2": 0.1, "corpus-test-3": 0.0},
            "q2": {"corpus-test-2": 0.9, "corpus-test-1": 0.1, "corpus-test-3": 0.0},
        },
        "candidates": {
            "q1": ["corpus-test-1", "corpus-test-2", "corpus-test-3"],
            "q2": ["corpus-test-2", "corpus-test-1", "corpus-test-3"],
        },  # fmt: skip
    }
    return Dataset(**{**defaults, **fields})


def _pool_orders(dataset: Dataset, *, prefixed: bool = True, dataset_name: str | None = None) -> Rankings:
    """One system ranking every pool in pool order (unprefixed ids when asked)."""
    orders = {
        q: [d if prefixed else d.removeprefix("corpus-test-") for d in docs]
        for q, docs in (dataset.candidates or {}).items()
    }
    return Rankings.from_orders(orders, system="mine", dataset=dataset.name if dataset_name is None else dataset_name)


def test_the_issue_reproducer_ok_scores_and_the_mistakes_are_data_errors() -> None:
    """The issue's three files, offline: `ok` scores 1.0, `wrong_dataset` and `no_prefix` are exit-12 errors."""
    dataset = _vidore_like()
    ok = _pool_orders(dataset)
    wrong_dataset = _pool_orders(dataset, dataset_name="hr")
    no_prefix = _pool_orders(dataset, prefixed=False)

    assert evaluate(ok, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0).value("mine") == 1.0
    for broken, match in ((wrong_dataset, "no rankings of dataset 'hr__english'"),
                          (no_prefix, "no ranked document is in the pools or labels")):  # fmt: skip
        with pytest.raises(DataError, match=match) as caught:
            evaluate(broken, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0)
        assert int(caught.value.exit_code) == 12


def test_rankings_of_another_dataset_name_the_system_and_the_fix() -> None:
    """The `dataset` column holds `hr` where the scored subset is `hr__english`: a DataError naming the system."""
    dataset = _vidore_like()
    wrong = _pool_orders(dataset, dataset_name="hr")

    with pytest.raises(DataError) as caught:
        evaluate(wrong, dataset=dataset, gains=dataset.gains, metrics=["qrel_ndcg"], bootstrap=0)

    message = caught.value.message
    assert message.startswith(
        "system 'mine': no rankings of dataset 'hr__english'; the rankings name the datasets ['hr']"
    )
    assert caught.value.hint is not None and "exact subset name" in caught.value.hint
    assert "hr__english" in caught.value.hint


def test_no_subset_of_a_suite_is_named_is_a_data_error() -> None:
    """A suite scored with a file of another dataset: the message names the suite and its subsets."""
    suite = Dataset(name="vidore", subsets=(_vidore_like(), _vidore_like(name="energy__french")))
    wrong = _pool_orders(_vidore_like(), dataset_name="hr")

    with pytest.raises(DataError, match="no rankings of dataset 'vidore'") as caught:
        evaluate(wrong, dataset=suite, metrics=["qrel_ndcg"], bootstrap=0)

    assert "hr__english" in caught.value.hint and "energy__french" in caught.value.hint


def test_a_file_is_refused_when_any_of_its_systems_matches_nothing() -> None:
    """The checks are per system: a file of a matching and a non-matching system is refused, naming the bad one."""
    dataset = _vidore_like()
    broken = Rankings.from_orders(
        {q: list(docs) for q, docs in (dataset.candidates or {}).items()}, system="broken", dataset="hr"
    )

    with pytest.raises(DataError, match="system 'broken'"):
        evaluate(Rankings.concat([_pool_orders(dataset), broken]), dataset=dataset, gains=dataset.gains,
                 protocol="vidore", bootstrap=0)  # fmt: skip


def test_systems_scores_only_the_named_systems_of_a_file_with_one_broken() -> None:
    """`systems=` scores the named systems of a multi-system file; the broken one no longer fails the command."""
    dataset = _vidore_like()
    broken = Rankings.from_orders(
        {q: list(docs) for q, docs in (dataset.candidates or {}).items()}, system="broken", dataset="hr"
    )
    both = Rankings.concat([_pool_orders(dataset), broken])

    report = evaluate(both, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0, systems=["mine"])

    assert report.systems == ["mine"], "only the named system, in the rankings' order"
    assert report.value("mine") == 1.0
    with pytest.raises(ConfigError, match="systems \\['nobody'\\] are not in the rankings") as caught:
        evaluate(both, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0, systems=["nobody"])
    assert "'mine'" in caught.value.message and "'broken'" in caught.value.message, "the file's systems are listed"
    with pytest.raises(ConfigError, match="names no system"):
        evaluate(both, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0, systems=[])


def test_the_refusal_of_a_multi_system_file_names_the_way_out_for_the_others() -> None:
    """With several systems in the file, the hint says to drop the broken one's rows or score the others."""
    dataset = _vidore_like()
    broken = Rankings.from_orders(
        {q: list(docs) for q, docs in (dataset.candidates or {}).items()}, system="broken", dataset="hr"
    )

    with pytest.raises(DataError) as caught:
        evaluate(Rankings.concat([_pool_orders(dataset), broken]), dataset=dataset, gains=dataset.gains,
                 protocol="vidore", bootstrap=0)  # fmt: skip

    assert caught.value.hint is not None and "exact subset name" in caught.value.hint
    assert "systems=[...]" in caught.value.hint and "drop" in caught.value.hint, "the Python way out"
    assert caught.value.cli_hint is not None and "--system" in caught.value.cli_hint, "the command-line way out"
    no_prefix = Rankings.concat(
        [
            _pool_orders(dataset),
            Rankings.from_orders(
                {q: [d.removeprefix("corpus-test-") for d in docs] for q, docs in (dataset.candidates or {}).items()},
                system="broken",
                dataset=dataset.name,
            ),
        ]
    )
    with pytest.raises(DataError) as no_overlap:
        evaluate(no_prefix, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0, systems=["broken"])
    assert "no ranked document is in the pools or labels" in no_overlap.value.message
    assert "systems=[...]" in (no_overlap.value.hint or ""), "the file still holds the other systems"


def test_a_single_system_file_is_refused_without_the_multi_system_hint() -> None:
    """One system alone: dropping its rows or --system helps nobody, so the hint stays as it was."""
    dataset = _vidore_like()
    broken = Rankings.from_orders(
        {q: list(docs) for q, docs in (dataset.candidates or {}).items()}, system="broken", dataset="hr"
    )

    with pytest.raises(DataError) as caught:
        evaluate(broken, dataset=dataset, gains=dataset.gains, protocol="vidore", bootstrap=0)

    assert caught.value.hint is not None and "exact subset name" in caught.value.hint
    assert "--system" not in caught.value.hint


@pytest.mark.parametrize("protocol", ["plain", "vidore"])
def test_rankings_of_another_corpus_are_refused_under_every_protocol(protocol: str) -> None:
    """Doc ids without the `corpus-test-` prefix match no pool or label: a DataError, under any protocol."""
    dataset = _vidore_like()
    no_prefix = _pool_orders(dataset, prefixed=False)

    with pytest.raises(DataError) as caught:
        evaluate(no_prefix, dataset=dataset, gains=dataset.gains, protocol=protocol, bootstrap=0)

    message = caught.value.message
    assert "no ranked document is in the pools or labels of 'hr__english'" in message
    assert "ranked '1' vs pool 'corpus-test-1'" in message, "one ranked id next to one pool id"
    assert int(caught.value.exit_code) == 12


def test_without_pools_the_labels_are_what_ranked_ids_must_match() -> None:
    dataset = _vidore_like(candidates=None)
    no_prefix = Rankings.from_orders(
        {q: [d.removeprefix("corpus-test-") for d in docs] for q, docs in (dataset.gains or {}).items()},
        system="mine",
        dataset=dataset.name,
    )

    with pytest.raises(DataError, match="vs label 'corpus-test-1'"):
        evaluate(no_prefix, dataset=dataset, gains=dataset.gains, protocol="plain", bootstrap=0)


def test_partial_overlap_still_scores_and_warns_as_before() -> None:
    """A system that ranks some of the dataset keeps its numbers, and nothing new warns."""
    dataset = _vidore_like()
    partial = Rankings.from_orders(
        {"q1": ["corpus-test-1", "corpus-test-2", "corpus-test-3"], "q2": ["other-9"]},
        system="mine",
        dataset=dataset.name,
    )

    report = evaluate(partial, dataset=dataset, gains=dataset.gains, protocol="vidore", k=2, bootstrap=0)

    assert 0.0 < report.value("mine") < 1.0, "q1's ranked pool scores, q2's unjudged id scores 0"
    assert report.warnings == []


def test_unranked_queries_name_the_subsets_they_count() -> None:
    """One subset ranked, one not: the warning keeps its text and names the subset scored 0."""
    suite = Dataset(name="vidore", subsets=(_vidore_like(), _vidore_like(name="energy__french")))
    rankings = _pool_orders(_vidore_like())

    report = evaluate(rankings, dataset=suite, metrics=["qrel_ndcg"], bootstrap=0)

    (warning,) = [w for w in report.warnings if w.code == "UNRANKED_QUERIES"]
    assert warning.message.startswith("mine: 2 labelled queries have no ranking; scored 0")
    assert "subsets: energy__french" in warning.message and "hr__english" not in warning.message
    per_dataset = {row.dataset: row.value for row in report.per_dataset}
    assert per_dataset["hr__english"] == pytest.approx(1.0), "the ranked subset keeps its score"


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


def test_explain_refuses_a_report_that_scored_no_systems() -> None:
    """A report whose metrics matched no labelled query has no systems to explain: a typed refusal, not a crash."""
    empty = evaluate(
        _rankings(), dataset=Dataset(name="gains-only", gains={"q0": {"d1": 1.0}}), metrics=["qrel_ndcg"],
        bootstrap=0,
    )  # fmt: skip

    assert empty.systems == []
    with pytest.raises(DataError, match="scored no systems"):
        explain(empty, "q0")
