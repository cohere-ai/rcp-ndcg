"""The MTEB page's tie claim matches the protocols it describes.

``docs/how-to/mteb-integration.md`` tells users when mteb's integer ``ndcg_at_10`` equals
``rcp_ndcg.eval.evaluate``'s ``qrel_ndcg``.  The two break ties differently outside the ``doc_id_desc``
suites, so the equality must stay scoped to those suites (or to untied scores); this test pins the sentence
and the numbers it rests on (docs-9: the page once claimed the tie rules agree for every suite).
"""

from __future__ import annotations

import pytest
from rcp_ndcg_core.protocol import PROTOCOLS, score_query

from tests.docs._markdown import ROOT, paragraph

PAGE = ROOT / "docs" / "how-to" / "mteb-integration.md"

_TIED = {"a": 1.0, "b": 1.0}
"""Two documents tied on score; ``b`` is the higher id and the only graded one."""

_CANDIDATES = ["a", "b"]
"""The pool order: ``b`` is second, so ``input_order`` ranks it second and ``doc_id_desc`` first."""


def _qrel(protocol: str, scores: dict[str, float] = _TIED) -> float:
    """qrel-nDCG of the tied (or given) ranking under one suite's protocol."""
    return score_query(scores, {"b": 1}, protocol=protocol, metric="qrel_ndcg", candidates=_CANDIDATES, k=2)


def test_the_page_scopes_the_tie_claim_to_the_rule_that_makes_it_true() -> None:
    """The equality sentence names the tie rule it needs and the rule the presets use; the old blanket "the tie
    rules agree" must not come back.

    Under arXiv v2 every preset credits a tie its group's mean gain (mteb's ``ndcg_float_at_k`` rule), so
    ``qrel_ndcg`` equals mteb's integer ``ndcg_at_10`` only under a protocol that breaks ties by document id
    descending - what mteb's integer metrics do - and the page has to say which is which rather than claim
    agreement. arXiv v1 broke ties by document id on NanoBEIR and BRIGHT and by pool order on TREC-DL, which is
    why the earlier version of this test could name those suites.
    """
    claim = paragraph(PAGE.read_text(encoding="utf-8"), "The served scores are")
    ties = {name: protocol.ties for name, protocol in PROTOCOLS.items()}
    assert set(ties.values()) == {"group_mean"}, ties
    assert "doc_id_desc" in claim, "the page does not name the tie rule the equality needs"
    assert "group's mean gain" in claim, "the page does not name the rule every preset now uses"
    assert "differ only in how they round" in claim, "the untied case is part of the claim"
    assert "the tie rules agree" not in claim, "the broad claim is false for every suite's group_mean ties"


def test_the_tied_ranking_numbers_match_the_page_s_scope() -> None:
    """The page's scope is the protocol's: under arXiv v2 every preset credits a tie its group's mean gain, so a
    tied ranking scores the same under every suite, and it is mteb's *integer* metric (which breaks ties by
    document id descending) that differs. Untied scores agree up to the 5-decimal BEIR rounding."""
    same = _qrel("nanobeir")
    for suite in ("bright", "vidore", "trecdl"):
        # The suites round differently (nanobeir/bright to 5 decimals, vidore/trecdl not at all), so the tied
        # value agrees to the coarser rounding rather than exactly.
        assert _qrel(suite) == pytest.approx(same, abs=1e-4), f"{suite} does not credit the tie its group's mean gain"
    # The rule is the group's mean gain, not the document-id order: the tied ranking is strictly below the
    # untied one (a tie at the top of an order-1 gain class cannot borrow the best position).
    assert same < 1.0
    untied = {name: _qrel(name, {"a": 2.0, "b": 1.0}) for name in ("nanobeir", "bright", "vidore", "trecdl")}
    for value in untied.values():
        assert value == pytest.approx(0.6309297535714575)
