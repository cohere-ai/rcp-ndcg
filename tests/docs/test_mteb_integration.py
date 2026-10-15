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


def test_the_page_scopes_the_tie_claim_to_the_doc_id_desc_suites() -> None:
    """The equality sentence names the suites whose protocol breaks ties by document id descending and the
    tie rules that diverge; the old blanket "the tie rules agree" must not come back."""
    claim = paragraph(PAGE.read_text(encoding="utf-8"), "The served scores are")
    doc_id_desc = sorted(name for name, protocol in PROTOCOLS.items() if protocol.ties == "doc_id_desc")
    assert doc_id_desc == ["bright", "nanobeir"], doc_id_desc
    for suite in doc_id_desc:
        assert suite in claim.lower(), f"the page does not scope the equality to {suite}"
    for suite in ("vidore", "trecdl"):
        assert PROTOCOLS[suite].ties in claim, f"the page does not name {suite}'s tie rule"
    assert "when no scores are tied" in claim, "the untied case is part of the claim"
    assert "the tie rules agree" not in claim, "the broad claim is false for the group_mean/input_order suites"


def test_the_tied_ranking_numbers_match_the_page_s_scope() -> None:
    """The page's scope is the protocol's: a tied ranking scores the same under the doc_id_desc suites and
    differently under ViDoRe v3 / TREC-DL; untied scores agree (up to the 5-decimal BEIR rounding)."""
    same = _qrel("nanobeir")
    assert same == pytest.approx(1.0)
    assert _qrel("bright") == pytest.approx(same)
    assert _qrel("vidore") != pytest.approx(same)
    assert _qrel("trecdl") != pytest.approx(same)
    untied = {name: _qrel(name, {"a": 2.0, "b": 1.0}) for name in ("nanobeir", "bright", "vidore", "trecdl")}
    for value in untied.values():
        assert value == pytest.approx(0.6309297535714575)
