"""The paper's scoring protocol (``rcp_ndcg_core.protocol``) and the Count-nDCG baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rcp_ndcg_core.protocol import PROTOCOLS, Protocol, aggregate, candidate_docs, score_query

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "paper_protocol_queries.json").read_text())
LOG2_3 = math.log2(3)

#: arXiv v1's tie rule per suite (document id for NanoBEIR and BRIGHT, judge-pool order for TREC-DL). The fixture's
#: values are arXiv v2's, printed under the presets' rule (ties credited with their group's mean gain on every suite);
#: the guards below score the fixture under the old rule to show which queries the rule moves.
V1_TIES = {"nanobeir": "doc_id_desc", "bright": "doc_id_desc", "trecdl": "input_order", "vidore": "group_mean"}


def _v1_protocol(query: dict) -> Protocol:
    """The query's suite preset with arXiv v1's tie rule."""
    return PROTOCOLS[query["suite"]].model_copy(update={"ties": V1_TIES[query["suite"]]})


def _score_fixture(query: dict, protocol: Protocol | str, *, excluded: bool = True) -> tuple[float, float]:
    """``(rcp_ndcg, qrel_ndcg)`` of a fixture query."""
    common = {
        "protocol": protocol,
        "candidates": query.get("candidates"),
        "excluded": query["excluded"] if excluded else (),
        "k": FIXTURE["k"],
    }
    return (
        score_query(query["scores"], query["gains"], **common),
        score_query(query["scores"], query["qrels"], metric="qrel_ndcg", **common),
    )


# ---------------------------------------------------------------------------
# Paper anchors: per-query values of the paper's tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", FIXTURE["queries"], ids=lambda q: f"{q['dataset']}:{q['query_id'][:24]}")
def test_reproduces_the_papers_per_query_values(query: dict) -> None:
    rcp, qrel = _score_fixture(query, query["suite"])

    # The fixture states each query's RCP tolerance (see its description); the paper's qrel-nDCG is exact.
    assert rcp == pytest.approx(query["expected"]["rcp_ndcg"], abs=query["rcp_ndcg_tolerance"])
    assert qrel == pytest.approx(query["expected"]["qrel_ndcg"], abs=1e-12)


def _query(query_id: str) -> dict:
    return next(q for q in FIXTURE["queries"] if q["query_id"] == query_id)


@pytest.mark.parametrize(
    ("query_id", "other_rule"),
    [
        ("test-religion-grcrgshwbr-pro03a", "doc_id_desc"),
        ("PLAIN-2051", "doc_id_desc"),
        ("5cd7b83eb30797aa8bf49b4b7a3f6a433aca4a75", "doc_id_desc"),
        ("aops_1959_IMO_Problems/Problem_1", "doc_id_desc"),
        ("359349", "input_order"),
        ("359349", "doc_id_desc"),
        ("query-test-7", "doc_id_desc"),
        ("query-test-7", "input_order"),
    ],
)
def test_the_fixture_queries_depend_on_their_tie_rule(query_id: str, other_rule: str) -> None:
    """Guards the anchors above: with another tie rule (arXiv v1's, or the other order) these queries score
    differently from the presets' group mean."""
    query = _query(query_id)
    paper = PROTOCOLS[query["suite"]]
    other = paper.model_copy(update={"ties": other_rule})

    assert _score_fixture(query, other) != _score_fixture(query, paper)


@pytest.mark.parametrize(
    "query_id",
    [
        "test-religion-grcrgshwbr-pro03a",
        "PLAIN-2051",
        "5cd7b83eb30797aa8bf49b4b7a3f6a433aca4a75",
        "aops_1959_IMO_Problems/Problem_1",
        "359349",
    ],
)
def test_the_presets_score_the_fixtures_tied_queries_by_their_group_mean(query_id: str) -> None:
    """The fixture's NanoBEIR, BRIGHT and TREC-DL queries whose ties reach the cutoff: the preset scores them as an
    explicit group-mean protocol does, whatever order the scores and the pool come in, and no longer as arXiv v1
    printed them."""
    query = _query(query_id)
    group_mean = PROTOCOLS[query["suite"]].model_copy(update={"ties": "group_mean", "name": "explicit"})
    shuffled = {
        **query,
        "scores": dict(reversed(list(query["scores"].items()))),
        "candidates": list(reversed(query["candidates"])) if query.get("candidates") else query.get("candidates"),
    }

    assert _score_fixture(query, query["suite"]) == _score_fixture(query, group_mean)
    assert _score_fixture(shuffled, query["suite"]) == pytest.approx(_score_fixture(query, query["suite"]), abs=1e-12)
    assert _score_fixture(query, query["suite"]) != _score_fixture(query, _v1_protocol(query))


def test_bright_excluded_ids_leave_the_ranking_and_the_ideal() -> None:
    query = _query("aops_1959_IMO_Problems/Problem_1")
    with_rule = _score_fixture(query, "bright")
    without = _score_fixture(query, "bright", excluded=False)

    assert with_rule[0] != pytest.approx(without[0])


def test_bright_scores_unjudged_candidates_as_zero_rather_than_dropping_them() -> None:
    query = _query("TheoremQA_jianyu_xu/Cayley_3.json")
    restricted = PROTOCOLS["bright"].model_copy(update={"restrict_to_candidates": True})

    got = score_query(
        query["scores"],
        query["gains"],
        protocol=restricted,
        candidates=list(query["gains"]),
        excluded=query["excluded"],
    )

    assert got != pytest.approx(query["expected"]["rcp_ndcg"], abs=1e-6)


# ---------------------------------------------------------------------------
# Suite rules on small hand-computed cases
# ---------------------------------------------------------------------------


def test_excluded_documents_are_removed_from_the_ranking_and_both_ideals() -> None:
    scores = {"self": 3.0, "a": 2.0, "b": 1.0}
    gains = {"self": 1.0, "a": 0.5, "b": 0.25}
    qrels = {"self": 1, "b": 1}

    rcp = score_query(scores, gains, protocol="plain", k=2, excluded={"self"})
    qrel = score_query(scores, qrels, protocol="plain", metric="qrel_ndcg", k=2, excluded={"self"})

    assert rcp == pytest.approx(1.0)  # a, b is the ideal order once "self" is gone
    assert qrel == pytest.approx((1 / LOG2_3) / 1.0)


def test_restricting_to_the_judged_pool_drops_other_candidates() -> None:
    scores = {"unjudged": 9.0, "a": 2.0, "b": 1.0}
    gains = {"a": 1.0, "b": 0.5}
    restricted = score_query(scores, gains, protocol="nanobeir", k=2, candidates=["a", "b"])
    unrestricted = score_query(scores, gains, protocol="bright", k=2)

    assert restricted == pytest.approx(1.0)
    assert unrestricted == pytest.approx((1 / LOG2_3) / (1.0 + 0.5 / LOG2_3))


def test_qrel_ideal_counts_positives_outside_the_scored_pool() -> None:
    scores = {"a": 2.0, "b": 1.0}
    qrels = {"a": 1, "elsewhere": 1}

    got = score_query(scores, qrels, protocol="plain", metric="qrel_ndcg", k=2)

    assert got == pytest.approx(1.0 / (1.0 + 1 / LOG2_3))


def test_qrel_ndcg_is_nan_without_positives_and_rounds_like_beir() -> None:
    no_positives = score_query({"a": 1.0}, {"a": 0}, protocol="plain", metric="qrel_ndcg", k=1)
    rounded = score_query({"a": 2.0, "b": 1.0}, {"b": 1}, protocol="bright", metric="qrel_ndcg", k=2)
    unrounded_rcp = score_query({"a": 2.0, "b": 1.0}, {"b": 1.0}, protocol="bright", k=2)

    assert math.isnan(no_positives)
    assert rounded == round(1 / LOG2_3, 5)
    assert unrounded_rcp == 1 / LOG2_3


def test_a_non_finite_qrel_grade_is_refused_not_read_as_absent() -> None:
    """A NaN grade failed ``> 0`` and scored as grade 0 (0.6309297535714575, identical to
    ``{"a": 0.0}``), silently dropping a positive from the ranking and the ideal."""
    qrels = {"a": float("nan"), "b": 1.0}
    with pytest.raises(ValueError, match="gains must be finite"):
        score_query({"a": 2.0, "b": 1.0}, qrels, protocol="plain", metric="qrel_ndcg", k=2)
    with pytest.raises(ValueError, match="gains must be finite"):
        score_query({"a": 2.0, "b": 1.0}, {"a": float("inf"), "b": 1.0}, protocol="plain", metric="qrel_ndcg", k=2)


def test_input_order_ties_follow_the_judged_pool_order() -> None:
    scores = {"b": 1.0, "a": 1.0}
    gains = {"a": 1.0}
    pool_order = Protocol(name="pool-order", ties="input_order", restrict_to_candidates=True)

    assert score_query(scores, gains, protocol=pool_order, k=1, candidates=["a", "b"]) == 1.0
    assert score_query(scores, gains, protocol=pool_order, k=1, candidates=["b", "a"]) == 0.0
    with pytest.raises(ValueError, match="judged pool"):
        score_query(scores, gains, protocol=pool_order)


@pytest.mark.parametrize("preset", sorted(PROTOCOLS))
def test_no_preset_takes_a_tie_order_from_the_ids_or_the_pool(preset: str) -> None:
    """Two documents tied at the cutoff share their mean gain under every preset, whatever their ids and the pool
    order (a pool sorted by relevance would otherwise leak relevance into the tie)."""
    gains = {"a": 1.0, "b": 0.0, "c": 0.5}
    expected = 0.5 / 1.0  # k=1: the tied pair {a, b} is credited its mean gain 0.5; the ideal DCG@1 is 1.0
    for pool in (["a", "b", "c"], ["b", "a", "c"]):
        for scores in ({"a": 2.0, "b": 2.0, "c": 1.0}, {"b": 2.0, "a": 2.0, "c": 1.0}):
            assert score_query(scores, gains, protocol=preset, k=1, candidates=pool) == pytest.approx(expected)


def test_identical_ids_can_be_dropped_without_an_excluded_list() -> None:
    protocol = Protocol(name="beir", ties="doc_id_desc", drop_identical_ids=True)

    assert candidate_docs(protocol, ["q1", "d1"], query_id="q1") == ["d1"]
    assert score_query({"q1": 2.0, "d1": 1.0}, {"q1": 1.0, "d1": 0.5}, protocol=protocol, query_id="q1", k=1) == 1.0


def test_candidate_docs_keeps_the_pool_order_and_drops_excluded_ids() -> None:
    for suite in ("nanobeir", "vidore", "trecdl", "mteb"):
        assert candidate_docs(suite, ["c", "x", "a"], candidates=["a", "b", "c"], excluded={"c"}) == ["a"]
    assert candidate_docs("bright", ["c", "x", "a"], candidates=["a", "b", "c"]) == ["a", "c", "x"]
    with pytest.raises(ValueError, match="unknown protocol"):
        candidate_docs("nope", [])


def test_candidate_docs_deduplicates_instead_of_returning_duplicates() -> None:
    """The entering ids are the ones a ranking may hold, and a ranking may not repeat an id:
    fed straight to ``ndcg``, the duplicated list raised while ``score_query`` silently deduped.
    """
    assert candidate_docs("plain", ["b", "a", "a", "b"]) == ["b", "a"]
    assert candidate_docs("plain", ["b", "a"], candidates=["a", "a", "b"]) == ["a", "b"]
    assert candidate_docs("nanobeir", ["b", "a", "a"], candidates=["a", "b", "c"]) == ["a", "b"]


def test_aggregate_takes_the_mean_per_dataset_then_over_datasets() -> None:
    assert aggregate({"d1": {"q1": 1.0}, "d2": {"q1": 0.0, "q2": 0.5}}) == pytest.approx((1.0 + 0.25) / 2)
    assert aggregate({"d1": {"q1": math.nan, "q2": 0.5}, "d2": {"q1": math.nan}, "d3": {}}) == 0.5
    assert math.isnan(aggregate({"d1": {"q1": math.nan}}))


def test_presets_carry_the_papers_tie_rules() -> None:
    assert {name: p.ties for name, p in PROTOCOLS.items()} == {
        "nanobeir": "group_mean",
        "bright": "group_mean",
        "vidore": "group_mean",
        "trecdl": "group_mean",
        "mteb": "group_mean",
        "plain": "group_mean",
    }
    with pytest.raises(ValueError):
        Protocol(name="x", ties="stable")  # type: ignore[arg-type]


def test_a_preset_name_resolves_to_its_protocol_and_an_unknown_one_is_refused() -> None:
    from rcp_ndcg_core.protocol import PROTOCOLS, resolve_protocol

    assert resolve_protocol("nanobeir") is PROTOCOLS["nanobeir"]
    mine = PROTOCOLS["plain"].model_copy(update={"name": "mine"})
    assert resolve_protocol(mine) is mine
    with pytest.raises(ValueError, match="nanobeir"):
        resolve_protocol("nanobeer")
