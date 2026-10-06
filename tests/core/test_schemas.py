"""Tests for rcp_ndcg_core.schemas — data model invariants."""

from __future__ import annotations

import json

import pytest
from rcp_ndcg_core._records import Document, Query, RankingExample


class TestQuery:
    def test_query_alias_text_and_id(self):
        q = Query(query="q", query_id="q1")
        assert q.text == "q"
        assert q.id == "q1"

    def test_query_construct_by_field_name(self):
        q = Query(text="q", id="q1")
        assert q.text == "q"
        assert q.id == "q1"

    def test_query_format_query_with_instruction(self):
        q = Query(query="find docs", query_id="q1", instruction="Given a claim, find documents that refute the claim")
        assert q.format_query() == "Task: Given a claim, find documents that refute the claim\nQuery: find docs"

    def test_query_format_query_without_instruction(self):
        q = Query(query="find docs", query_id="q1")
        assert q.format_query() == "find docs"

    def test_query_format_query_instruction_none(self):
        q = Query(query="find docs", query_id="q1", instruction=None)
        assert q.format_query() == "find docs"

    def test_query_format_query_instruction_empty_string(self):
        q = Query(query="find docs", query_id="q1", instruction="")
        assert q.format_query() == "find docs"

    def test_query_format_query_instruction_whitespace_only(self):
        q = Query(query="find docs", query_id="q1", instruction="   ")
        assert q.format_query() == "find docs"

    def test_query_format_query_strips_whitespace(self):
        q = Query(query="  find docs  ", query_id="q1", instruction="  Given a query, retrieve docs  ")
        result = q.format_query()
        assert result == "Task: Given a query, retrieve docs\nQuery: find docs"


class TestDocument:
    def test_document_alias_doc_id(self):
        d = Document(text="hello", doc_id="d1")
        assert d.id == "d1"
        assert d.doc_id == "d1"

    def test_document_alias_docno(self):
        d = Document(text="hello", docno="d1")
        assert d.id == "d1"


class TestRankingExample:
    def test_ranking_example_auto_sorts_by_scores(self):
        ex = RankingExample(
            query="q",
            query_id="q1",
            doc_ids=["a", "b", "c"],
            docs=["A", "B", "C"],
            scores=[1.0, 3.0, 2.0],
        )
        assert ex.scores == [3.0, 2.0, 1.0]
        assert ex.doc_ids == ["b", "c", "a"]
        assert ex.docs == ["B", "C", "A"]

    def test_ranking_example_no_scores_preserves_order(self):
        ex = RankingExample(
            query="q",
            query_id="q1",
            doc_ids=["a", "b", "c"],
            docs=["A", "B", "C"],
        )
        assert ex.doc_ids == ["a", "b", "c"]

    def test_ranking_example_serialize_roundtrip(self, ranking_example):
        json_str = ranking_example.model_dump_json()
        restored = RankingExample(**json.loads(json_str))
        assert restored.text == ranking_example.text
        assert restored.doc_ids == ranking_example.doc_ids

    def test_the_query_alias_is_the_text_not_a_second_field(self):
        """``query`` duplicated the aliased ``text``: built with ``text=``, ``query`` stayed
        empty and both keys were written, so the two could disagree in one line."""
        ex = RankingExample(query="AAA", query_id="q1", doc_ids=["a"])
        assert ex.text == "AAA"
        assert json.loads(ex.serialize_jsonl())["text"] == "AAA"
        assert "query" not in json.loads(ex.serialize_jsonl())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("docs", ["A"]),
            ("docs", ["A", "B", "C"]),
            ("scores", [1.0]),
            ("scores", [1.0, 2.0, 3.0]),
        ],
    )
    def test_ranking_example_rejects_misaligned_document_fields(self, field, value):
        with pytest.raises(ValueError, match=rf"{field} must align with doc_ids"):
            RankingExample(
                query="q",
                query_id="q1",
                doc_ids=["a", "b"],
                **{field: value},
            )

    def test_ranking_example_rejects_duplicate_doc_ids(self):
        with pytest.raises(ValueError, match="doc_ids must not contain duplicates"):
            RankingExample(
                query="q",
                query_id="q1",
                doc_ids=["a", "a"],
                docs=["first", "second"],
            )


class TestUnknownFieldPreservation:
    """Dataset-specific fields must survive read -> rerank -> write.

    The context-compression eval sets carry ``answer`` / ``evidence`` /
    ``evidence_intersection`` / ``query_types`` alongside the standard ranking
    fields. Dropping them would make a rescored file unusable by the pipeline
    that produced it.
    """

    def _raw(self) -> dict:
        return {
            "query": "why is the sky blue?",
            "query_id": "q1",
            "doc_ids": ["d1", "d2"],
            "docs": ["scattering", "unrelated"],
            "scores": [0.9, 0.1],
            "qrels": {"d1": 1},
            "answer": "Rayleigh scattering",
            "evidence": [{"text": "shorter wavelengths scatter more", "doc_id": "d1", "weight": 1}],
            "evidence_intersection": [{"doc_id": "d1", "text": "scatter"}],
            "query_types": ["factual"],
        }

    def test_unknown_fields_round_trip(self):
        raw = self._raw()
        ex = RankingExample.model_validate(raw)
        out = json.loads(ex.serialize_jsonl())
        for field in ("answer", "evidence", "evidence_intersection", "query_types"):
            assert out[field] == raw[field], f"{field} was not preserved"

    def test_known_aliases_are_not_duplicated_into_extras(self):
        """``query_id`` is consumed by its alias, not kept twice."""
        ex = RankingExample.model_validate(self._raw())
        assert set(ex.__pydantic_extra__ or {}) == {"answer", "evidence", "evidence_intersection", "query_types"}
        assert ex.id == "q1"
        assert ex.scores == [0.9, 0.1]
