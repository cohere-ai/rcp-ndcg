"""The continuous-qrel nDCG: ``rcp_core.metric.ndcg_float`` and ``rcp_core.metric.per_query_ndcg``.

The vectors are the ones frozen in the metric specification (sections 3, 4, 6 and 7). Each query carries one tie,
and the value of the metric changes with the tie rule, so a test that passes under ``"input_order"`` or
``"doc_id_desc"`` would fail here.
"""

from __future__ import annotations

import math

import pytest
from rcp_core import ndcg, ndcg_float, per_query_ndcg

# Query q1: the tie is the class {b, c} at 0.5, and it straddles the k=2 cutoff.
Q1_SCORES = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.4, "e": 0.2}
Q1_FLOAT_QRELS = {"a": 0.9, "b": 0.4, "c": 0.6, "d": 0.2, "e": 0.0}

# Query q2: the tie is the class {a, b} at 0.8; a is not in the qrels at all.
Q2_SCORES = {"a": 0.8, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
Q2_FLOAT_QRELS = {"b": 0.5, "c": 0.4, "d": 0.0}

QRELS = {"q1": Q1_FLOAT_QRELS, "q2": Q2_FLOAT_QRELS}
RESULTS = {"q1": Q1_SCORES, "q2": Q2_SCORES}

# The two per-query values of the corpus at k=2: q1 and q2 of specification 6.5.
Q1_AT_2 = 0.9506530148390225
Q2_AT_2 = 0.5419293804283138


def test_ndcg_float_of_one_query_credits_the_tie_class_mean() -> None:
    """q1 at k=2: the class {b, c} is credited (0.4 + 0.6) / 2 = 0.5 before the cutoff applies."""
    assert ndcg_float({"q1": Q1_FLOAT_QRELS}, {"q1": Q1_SCORES}, k=2) == pytest.approx(Q1_AT_2, rel=1e-12)


def test_the_tie_rule_decides_this_vector() -> None:
    """The same input under the two ordering rules: 0.9013… (insertion order) and 1.0 (document id descending).

    ``"doc_id_desc"`` reaches the maximum because document c, ordered first, carries the larger gain. That is the
    leak the class mean removes, and it is why a wrong tie rule cannot pass the vector above.
    """
    assert ndcg(Q1_SCORES, Q1_FLOAT_QRELS, k=2, ties="input_order") == pytest.approx(0.9013060296780449, rel=1e-12)
    assert ndcg(Q1_SCORES, Q1_FLOAT_QRELS, k=2, ties="doc_id_desc") == pytest.approx(1.0, rel=1e-12)


def test_the_class_mean_is_the_mean_over_every_order_of_the_tie() -> None:
    """The identity a reader can check: with the class inside the cutoff, the credit is the mean of the orders."""
    orders = (
        ndcg(["a", "b", "c", "d", "e"], Q1_FLOAT_QRELS, k=2),
        ndcg(["a", "c", "b", "d", "e"], Q1_FLOAT_QRELS, k=2),
    )
    assert ndcg_float({"q1": Q1_FLOAT_QRELS}, {"q1": Q1_SCORES}, k=2) == pytest.approx(sum(orders) / 2, rel=1e-12)


def test_ndcg_float_of_the_corpus_with_a_tie_at_the_cutoff() -> None:
    """Both queries tie, and the mean is over the two evaluated queries: (0.9506… + 0.5419…) / 2."""
    assert ndcg_float(QRELS, RESULTS, k=2) == pytest.approx(0.7462911976336681, rel=1e-12)
    assert ndcg_float(QRELS, RESULTS, k=5) == pytest.approx(0.899693802011639, rel=1e-12)


def test_ndcg_float_keeps_the_ideal_of_the_whole_labelled_set() -> None:
    """A cutoff deeper than the run changes nothing: the ideal is the query's whole qrel set either way."""
    assert ndcg_float({"q1": Q1_FLOAT_QRELS}, {"q1": Q1_SCORES}, k=5) == pytest.approx(0.9916322409655682, rel=1e-12)


def test_per_query_ndcg_is_the_values_behind_ndcg_float() -> None:
    """The vector of specification 7.5: one list per dataset, the mean of which is ``ndcg_float`` at the same k."""
    assert per_query_ndcg({"d": QRELS}, {"d": RESULTS}, k=2) == {
        "d": [pytest.approx(Q1_AT_2, rel=1e-12), pytest.approx(Q2_AT_2, rel=1e-12)]
    }
    values = per_query_ndcg({"d": QRELS}, {"d": RESULTS}, k=2)["d"]
    assert sum(values) / len(values) == pytest.approx(ndcg_float(QRELS, RESULTS, k=2), rel=1e-12)


def test_per_query_ndcg_skips_a_query_with_no_ranking_and_keeps_an_empty_dataset() -> None:
    """A query in the qrels with no run is not scored; a dataset with nothing scored stays, with an empty list."""
    assert per_query_ndcg({"d": QRELS}, {"d": {"q1": Q1_SCORES}}, k=2) == {"d": [pytest.approx(Q1_AT_2, rel=1e-12)]}
    assert per_query_ndcg({"d": QRELS}, {"other": RESULTS}, k=2) == {"d": []}
    assert per_query_ndcg({}, {}, k=2) == {}


def test_a_query_with_no_positive_float_qrel_scores_zero_and_stays_in_the_mean() -> None:
    qrels = {"q1": Q1_FLOAT_QRELS, "q2": {"b": 0.0}}
    results = {"q1": Q1_SCORES, "q2": Q2_SCORES}
    assert ndcg_float(qrels, results, k=2) == pytest.approx(Q1_AT_2 / 2, rel=1e-12)


def test_ndcg_float_refuses_an_undefined_mean() -> None:
    """0.0 would report "nothing ranked correctly" for a corpus that scored nothing at all."""
    with pytest.raises(ValueError, match="no query to average over"):
        ndcg_float({}, RESULTS, k=2)
    with pytest.raises(ValueError, match="no query to average over"):
        ndcg_float(QRELS, {"other": Q1_SCORES}, k=2)
    with pytest.raises(ValueError, match="no query to average over"):
        ndcg_float(QRELS, {}, k=2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_ndcg_float_refuses_a_non_finite_grade_or_score(bad: float) -> None:
    """A NaN grade flows through the sums and returns a NaN nDCG; a NaN score has no sort position."""
    with pytest.raises(ValueError, match="must be finite"):
        ndcg_float({"q1": {"a": bad}}, {"q1": {"a": 1.0}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        ndcg_float({"q1": {"a": 1.0}}, {"q1": {"a": bad}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        per_query_ndcg({"d": {"q1": {"a": bad}}}, {"d": {"q1": {"a": 1.0}}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        per_query_ndcg({"d": {"q1": {"a": 1.0}}}, {"d": {"q1": {"a": bad}}}, k=1)


def test_ndcg_float_refuses_a_negative_float_qrel() -> None:
    """The gain is the qrel score itself, so a negative one leaves [0, 1]; a sigmoid belongs to dataset creation."""
    with pytest.raises(ValueError, match="at least 0"):
        ndcg_float({"q1": {"a": -0.1}}, {"q1": {"a": 1.0}}, k=1)
    with pytest.raises(ValueError, match="at least 0"):
        per_query_ndcg({"d": {"q1": {"a": -0.1}}}, {"d": {"q1": {"a": 1.0}}}, k=1)


def test_ndcg_float_refuses_a_non_positive_cutoff() -> None:
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be greater than 0"):
            ndcg_float(QRELS, RESULTS, k=k)
        with pytest.raises(ValueError, match="k must be greater than 0"):
            per_query_ndcg({"d": QRELS}, {"d": RESULTS}, k=k)


def test_a_duplicate_document_id_cannot_reach_these_metrics() -> None:
    """The inputs are score mappings, so a document id is a key: it cannot repeat (``ndcg``'s ranking form
    refuses one, and that is where a list of ids can arrive)."""
    with pytest.raises(ValueError, match="duplicate document IDs"):
        ndcg(["a", "a"], {"a": 1.0}, k=2)
