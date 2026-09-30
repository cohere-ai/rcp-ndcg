"""Content model: the multimodal shape and its text-only compatibility."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rcp_ndcg_core._records import Document, Query, RankingExample
from rcp_ndcg_core.content import (
    Content,
    ImagePart,
    MediaRef,
    Modality,
    TextPart,
    VideoPart,
)


class TestMediaRef:
    def test_cache_key_prefers_hash(self):
        assert MediaRef(uri="gs://b/x.png", sha256="deadbeef").cache_key == "deadbeef"

    def test_cache_key_falls_back_to_uri(self):
        assert MediaRef(uri="gs://b/x.png").cache_key == "gs://b/x.png"

    def test_unknown_field_is_rejected(self):
        # A typo in a media field must not be silently retained as metadata.
        with pytest.raises(ValidationError):
            MediaRef(uri="gs://b/x.png", widht=10)


class TestContent:
    def test_text_only(self):
        content = Content.from_text("a passage")
        assert content.text == "a passage"
        assert not content.has_media
        assert content.modality is Modality.TEXT

    def test_image_only_has_no_text(self):
        content = Content.from_image("gs://b/page.png", page=3, width=1700, height=2200)
        assert content.text == ""
        assert content.has_media
        assert not content.has_text
        assert content.modality is Modality.IMAGE
        assert content.media[0].width == 1700

    def test_interleaved_is_multimodal_and_joins_only_text(self):
        content = Content.from_parts(
            [
                TextPart(text="caption above"),
                ImagePart(ref=MediaRef(uri="gs://b/fig.png")),
                TextPart(text="caption below"),
            ]
        )
        assert content.text == "caption above\ncaption below"
        assert content.modality is Modality.MULTIMODAL
        assert len(content.media) == 1

    def test_coerce_accepts_str_content_and_none(self):
        assert Content.coerce("x").text == "x"
        assert Content.coerce(Content.from_text("y")).text == "y"
        assert not Content.coerce(None).has_text and not Content.coerce(None).has_media
        assert Content.coerce([TextPart(text="z")]).text == "z"

    def test_serialises_as_bare_array(self):
        content = Content.from_text("hi")
        assert content.model_dump() == [{"type": "text", "text": "hi"}]

    def test_round_trip_preserves_part_types(self):
        content = Content.from_parts(
            [TextPart(text="t"), ImagePart(ref=MediaRef(uri="u"), page=1), VideoPart(frames=[MediaRef(uri="f0")])]
        )
        again = Content.model_validate_json(content.model_dump_json())
        assert [type(p) for p in again] == [TextPart, ImagePart, VideoPart]

    def test_sequence_protocol(self):
        content = Content.from_parts([TextPart(text="a"), TextPart(text="b")])
        assert len(content) == 2
        assert content[0].text == "a"
        assert [p.text for p in content] == ["a", "b"]
        assert bool(content)
        assert not bool(Content())

    def test_video_flattens_frames_into_media(self):
        content = Content.from_parts(
            [VideoPart(ref=MediaRef(uri="v.mp4"), frames=[MediaRef(uri="f0.jpg"), MediaRef(uri="f1.jpg")])]
        )
        assert [ref.uri for ref in content.media] == ["v.mp4", "f0.jpg", "f1.jpg"]
        assert content.modality is Modality.VIDEO

    def test_video_needs_container_or_frames(self):
        with pytest.raises(ValidationError, match="container"):
            VideoPart()


class TestTruncated:
    def test_no_budget_returns_the_same_object(self):
        content = Content.from_text("abcdef")
        assert content.truncated(None) is content

    def test_text_is_clipped(self):
        assert Content.from_text("abcdef").truncated(3).text == "abc"

    def test_an_image_is_never_clipped(self):
        """Half a page is not a shorter document, it is a different one."""
        content = Content.from_image("gs://b/page.png")
        assert content.truncated(1).media == content.media

    def test_the_budget_spans_the_parts(self):
        """Per-part budgets would let a three-part document spend three times the
        allowance."""
        content = Content.from_parts([TextPart(text="aaaa"), TextPart(text="bbbb")])
        assert content.truncated(7).text == "aaaa\nbb"

    def test_the_result_is_a_verbatim_prefix_of_the_joined_text(self):
        """The joining newlines count, so a cut located in ``text`` (a token boundary) lands where it was found."""
        content = Content.from_parts([TextPart(text="aaaa"), TextPart(text="bbbb"), TextPart(text="cc")])
        for n in range(len(content.text) + 1):
            assert content.text.startswith(content.truncated(n).text)
            assert content.truncated(n).text == content.text[:n].removesuffix("\n")

    def test_an_exhausted_budget_drops_later_text_but_keeps_media(self):
        content = Content.from_parts([TextPart(text="aaaa"), ImagePart(ref=MediaRef(uri="u")), TextPart(text="bbbb")])
        truncated = content.truncated(4)

        assert truncated.text == "aaaa"
        assert len(truncated.media) == 1

    def test_media_only_content_is_untouched(self):
        content = Content.from_image("gs://b/page.png")
        assert content.truncated(0) is content


class TestDocumentAndQuery:
    def test_text_document_unchanged(self):
        doc = Document(doc_id="d1", text="hello")
        assert doc.text == "hello"
        assert doc.media == []
        assert not doc.has_media
        assert doc.as_content.text == "hello"

    def test_docno_alias_still_resolves(self):
        assert Document(docno="d1", text="x").id == "d1"

    def test_image_document(self):
        doc = Document(doc_id="d2", content=Content.from_image("gs://b/1.png"))
        assert doc.text == ""
        assert doc.has_media
        assert doc.modality is Modality.IMAGE

    def test_text_is_derived_from_content_and_cannot_disagree(self):
        # A caller passing both must not end up with a text view that lies.
        doc = Document(doc_id="d3", text="stale", content=Content.from_text("fresh"))
        assert doc.text == "fresh"

    def test_query_format_content_prefixes_instruction_for_image_query(self):
        query = Query(query_id="q1", instruction="Find the invoice", content=Content.from_image("gs://b/q.png"))
        content = query.format_content()
        assert content[0].text == "Task: Find the invoice"
        assert content.has_media

    def test_query_format_content_matches_format_query_for_text(self):
        query = Query(query_id="q1", query="what", instruction="Find it")
        assert query.format_content().text == query.format_query()


class TestRankingExample:
    def test_text_only_serialisation_has_no_content_key(self):
        example = RankingExample(query_id="q", query="q", doc_ids=["a"], docs=["A"])
        assert "contents" not in example.model_dump_json(exclude_none=True)

    def test_docs_is_derived_when_contents_given(self):
        example = RankingExample(
            query_id="q",
            query="q",
            doc_ids=["a", "b"],
            contents=[Content.from_image("gs://b/1.png"), Content.from_text("text b")],
        )
        assert example.docs == ["", "text b"]
        assert not example.has_media  # the query body is plain text

    def test_text_query_with_image_docs_formats_as_text(self):
        """`format_content` must not take the image branch for a text query.

        The two media questions -- does the query carry media, do the documents --
        have separate names precisely because conflating them sent a text query
        with image documents down the interleaved-parts path, changing the string
        that gets embedded.
        """
        example = RankingExample(
            query_id="q",
            query="what is in the chart",
            instruction="find the page",
            doc_ids=["a"],
            contents=[Content.from_image("gs://b/1.png")],
        )
        content = example.format_content()
        assert not content.has_media
        assert content.text == "Task: find the page\nQuery: what is in the chart"

    def test_image_query_keeps_its_image_when_formatted(self):
        example = RankingExample(
            query_id="q",
            content=Content.from_image("gs://b/query.png"),
            instruction="find a similar page",
            doc_ids=["a"],
        )
        content = example.format_content()
        assert content.has_media
        assert content.text == "Task: find a similar page"

    def test_doc_contents_materialises_from_text_only_docs(self):
        example = RankingExample(query_id="q", query="q", doc_ids=["a"], docs=["A"])
        assert [c.text for c in example.doc_contents] == ["A"]

    def test_doc_contents_raises_when_unhydrated(self):
        example = RankingExample(query_id="q", query="q", doc_ids=["a"])
        with pytest.raises(ValueError, match="document bodies"):
            _ = example.doc_contents

    def test_score_sort_permutes_every_aligned_field(self):
        example = RankingExample(
            query_id="q",
            query="q",
            doc_ids=["a", "b", "c"],
            contents=[Content.from_text("A"), Content.from_text("B"), Content.from_text("C")],
            scores=[0.1, 0.9, 0.5],
        )
        assert example.doc_ids == ["b", "c", "a"]
        assert example.docs == ["B", "C", "A"]
        assert [c.text for c in example.contents] == ["B", "C", "A"]

    def test_contents_must_align_with_doc_ids(self):
        with pytest.raises(ValidationError, match="contents must align"):
            RankingExample(query_id="q", query="q", doc_ids=["a", "b"], contents=[Content.from_text("A")])
