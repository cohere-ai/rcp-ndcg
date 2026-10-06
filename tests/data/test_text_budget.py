"""The one text-budget mechanism: template-framed fitting for every served role.

The mechanism (:func:`rcp_ndcg.data.preprocess.fit`) renders a declared template, measures the
template's fixed overhead once per (template, shape), cuts only the content spans within the
remaining budget, re-attaches the template, and records every cut. These tests pin what the review
lanes found: every anchor a model reads its output from survives a cut at its declared position
(the wrapped-prompt defect class: a right cut of the whole rendered prompt loses them), inputs under
budget are byte-identical to the uncut render, the pair split honours ``query_max_tokens``,
chunks carry the full template and their census rows name the ``max`` aggregation, self-hosted
budgets are explicit, and a hosted vendor profile without a tokenizer records ``budget_source:
vendor`` and cuts nothing.
"""

from __future__ import annotations

import logging

import pytest

from rcp_ndcg.data.preprocess import (
    ChunkPolicy,
    TextBudget,
    TextBudgetExceededError,
    TextTruncationCensus,
    fit,
    max_pool_scores_by_document,
)
from rcp_ndcg.data.templates import Segment, TemplateSpec
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import check_declarations, identity_payload
from tests._tokenizers import framed_bpe_tokenizer, spaced_special_tokenizer, word_tokenizer

FRAMED = framed_bpe_tokenizer()
WORDS = word_tokenizer()
SPACED = spaced_special_tokenizer()

#: The two special tokens the framed fixture adds: a template's suffix anchor and the post-processor's,
#: written in templates by name and resolved to these literals from the tokenizer's added tokens.
END_TURN = "<|end_turn|>"
END_OF_TEXT_NAME = "end_of_text"
END_TURN_NAME = "end_turn"

LONG = (
    "the relevant evidence of the query answers with a precise passage one two three four five. " * 12
    + "the document is relevant and the answer is on the page of the passage."
)


def query_template() -> TemplateSpec:
    """Query shape with a trailing suffix anchor (a last-token pooler reads its output there)."""
    return TemplateSpec(query=(Segment(fixed="Query: "), Segment(content="query"), Segment(fixed="{special:end_turn}")))


def document_template() -> TemplateSpec:
    return TemplateSpec(
        document=(Segment(fixed="<doc> "), Segment(content="document"), Segment(fixed=" {special:end_turn}"))
    )


def pair_template() -> TemplateSpec:
    """Pair shape, query first, assistant-suffix anchor last (the pointwise-reranker shape)."""
    return TemplateSpec(
        pair=(
            Segment(fixed="<instruct>: judge the pair\n<query>: "),
            Segment(content="query"),
            Segment(fixed="\n<document>: "),
            Segment(content="document"),
            Segment(fixed="{special:end_turn}"),
        )
    )


def document_first_template() -> TemplateSpec:
    """Pair shape, document first: the query block and the readout are the anchors (the ctxl shape)."""
    return TemplateSpec(
        pair=(
            Segment(fixed="Check whether the document answers the query.\n<document> "),
            Segment(content="document"),
            Segment(fixed="\n<query> "),
            Segment(content="query"),
            Segment(fixed=" ??"),
        )
    )


def marker_template() -> TemplateSpec:
    return TemplateSpec(
        query=(Segment(fixed="Query: "), Segment(content="query"), Segment(fixed="{special:end_turn}")),
        anchor="marker",
        anchor_markers=("end_turn",),
    )


def first_template() -> TemplateSpec:
    return TemplateSpec(document=(Segment(fixed="{special:end_turn} "), Segment(content="document")), anchor="first")


def mean_template() -> TemplateSpec:
    return TemplateSpec(
        document=(Segment(fixed="<doc> "), Segment(content="document"), Segment(fixed=" </doc>")), anchor="mean"
    )


def post_only_template() -> TemplateSpec:
    """No framing at all: the anchor is the tokenizer post-processor's own (the Qwen3-Embedding shape)."""
    return TemplateSpec(document=(Segment(content="document"),))


def budget(template: TemplateSpec | None = None, **fields: object) -> TextBudget:
    """A 24-token budget on the framed fixture's tokens (small enough that ``LONG`` overflows)."""
    fields.setdefault("tokenizer", "test/framed-bpe")
    fields.setdefault("max_tokens", 24)
    if template is not None:
        fields.setdefault("template", template)
    return TextBudget(**fields)  # type: ignore[arg-type]


def engine_ids(rendered: str, template: TemplateSpec, shape: str) -> list[int]:
    """The ids the engine sees: the rendered string under the shape's ``add_special_tokens`` flag."""
    return FRAMED.ids(rendered, add_special_tokens=template.adds_special_tokens(shape))


def ends_with(ids: list[int], tail: list[int]) -> bool:
    return ids[-len(tail) :] == tail if tail else not ids


# ---------------------------------------------------------------------------------------------------------------
# TemplateSpec: the template as data
# ---------------------------------------------------------------------------------------------------------------


class TestTemplateSpec:
    def test_a_template_declares_at_least_one_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            TemplateSpec()

    def test_every_declared_shape_has_a_content_span(self) -> None:
        with pytest.raises(ValueError, match="content"):
            TemplateSpec(query=(Segment(fixed="Query: "),))

    def test_a_span_outside_its_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="document"):
            TemplateSpec(query=(Segment(content="document"),))

    def test_a_segment_is_fixed_or_content_never_both(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Segment(fixed="x", content="query")
        with pytest.raises(ValueError, match="exactly one"):
            Segment()

    def test_an_anchor_of_marker_declares_markers_and_nothing_else_does(self) -> None:
        with pytest.raises(ValueError, match="anchor_markers"):
            TemplateSpec(query=(Segment(content="query"),), anchor="marker")
        with pytest.raises(ValueError, match="anchor_markers"):
            TemplateSpec(query=(Segment(content="query"),), anchor="last", anchor_markers=("end_turn",))

    def test_an_anchor_of_last_ends_fixed_or_declares_add_special_tokens(self) -> None:
        with pytest.raises(ValueError, match="anchor: last"):
            TemplateSpec(query=(Segment(content="query"),), add_special_tokens=False)
        with pytest.raises(ValueError, match="anchor: last"):
            TemplateSpec(query=(Segment(content="query"),), add_special_tokens={"query": False})

    def test_add_special_tokens_is_a_bool_or_per_shape(self) -> None:
        spec = TemplateSpec(
            query=(Segment(content="query"), Segment(fixed=".")),  # ends fixed: safe with no post-processor
            document=(Segment(content="document"),),  # ends on content: only the post-processor anchors it
            add_special_tokens={"query": False, "document": True},
        )
        assert spec.adds_special_tokens("query") is False
        assert spec.adds_special_tokens("document") is True
        with pytest.raises(ValueError, match="pair"):
            TemplateSpec(query=(Segment(content="query"),), add_special_tokens={"pair": False})

    def test_the_template_identity_is_its_canonical_json(self) -> None:
        check_declarations(TemplateSpec)
        check_declarations(Segment)
        one = identity_payload(query_template())
        other = identity_payload(
            TemplateSpec(query=(Segment(fixed="Q: "), Segment(content="query"), Segment(fixed=END_TURN)))
        )
        assert one["query"] != other["query"]
        assert one["anchor"] == "last"

    def test_specials_are_named_and_resolved_from_the_tokenizer(self) -> None:
        rendered = query_template().render("query", FRAMED, query="the query")
        assert rendered == f"Query: the query{END_TURN}"
        with pytest.raises(ConfigError, match="no special token"):
            TemplateSpec(query=(Segment(fixed="{special:no_such_token}"), Segment(content="query"))).render(
                "query", FRAMED, query="q"
            )

    def test_a_special_name_is_resolved_exactly_keeping_its_whitespace(self) -> None:
        """An added token named ``[Q] `` (a trailing space, as pplx-embed's ship) is resolved as written: the
        resolver strips nothing, or the name could not be written at all."""
        spec = TemplateSpec(query=(Segment(fixed="{special:[Q] }"), Segment(content="query")))
        assert spec.render("query", SPACED, query="the query") == "[Q] the query"
        assert spec.render("query", SPACED, query="") == "[Q] "

    def test_an_unknown_special_names_the_nearest_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            TemplateSpec(query=(Segment(fixed="{special:end_tur}"), Segment(content="query"))).render(
                "query", FRAMED, query="q"
            )
        assert "did you mean" in caught.value.hint and "end_turn" in caught.value.hint
        with pytest.raises(ConfigError) as caught:
            TemplateSpec(query=(Segment(fixed="{special:Q}"), Segment(content="query"))).render(
                "query", SPACED, query="q"
            )
        assert "[Q] " in caught.value.hint

    def test_the_overhead_is_the_empty_render_counted_once(self) -> None:
        spec = document_template()
        empty = spec.render("document", FRAMED, document="")
        expected = FRAMED.count(empty, add_special_tokens=spec.adds_special_tokens("document"))
        first = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        second = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        assert first.overhead == second.overhead == expected

    def test_the_render_refuses_an_undeclared_shape(self) -> None:
        with pytest.raises(ConfigError, match="pair"):
            query_template().render("pair", FRAMED, query="q")

    def test_a_last_content_anchor_needs_no_fixed_tail(self) -> None:
        """``anchor: last_content`` (jina-embeddings-v5): the model pools the last real token of raw text, so
        the shape may end on a content span -- which ``anchor: last`` refuses, because there the model reads
        a fixed position. Fixed segments (the head marker) are still reserved and audited; the cut keeps a
        content prefix, so the last kept content token always survives."""
        shape = (Segment(fixed="Query: "), Segment(content="query"))
        with pytest.raises(ValueError, match="anchor: last"):
            TemplateSpec(query=shape, anchor="last", add_special_tokens=False)
        spec = TemplateSpec(query=shape, anchor="last_content", add_special_tokens=False)
        assert spec.shapes() == ("query",)
        # The fixed head marker is still part of the frame: rendered whole, and reserved by the budget.
        assert spec.render("query", FRAMED, query="the query") == "Query: the query"
        result = fit(["the query " * 30], shape="query", budget=budget(spec, max_tokens=12), tokenizer=FRAMED)
        assert result.texts[0].startswith("Query: ")  # the head marker survived the cut
        assert result.texts[0].endswith(result.contents[0])  # and the shape ends on the (kept) content


# ---------------------------------------------------------------------------------------------------------------
# Declared content normalisation: the template's per-shape strip/lowercase, applied by fit
# ---------------------------------------------------------------------------------------------------------------


class TestDeclaredNormalisation:
    def test_the_reference_and_the_engine_see_the_same_normalised_text(self) -> None:
        """G4: the model's wrapper strips the query text and the whole document (topk); Cobble checkpoints
        lowercase. Declared per shape on the template, applied by fit before measuring, so the reference and
        the engine read the same text."""
        spec = TemplateSpec(
            query=(Segment(content="query"),),
            document=(Segment(fixed="Document: "), Segment(content="document")),
            normalize={"query": ("strip",), "document": ("strip", "lowercase")},
        )
        query = fit(["  the QUERY  "], shape="query", budget=budget(spec), tokenizer=FRAMED)
        assert query.contents[0] == "the QUERY"  # stripped, case kept
        assert query.texts[0] == "the QUERY"
        document = fit(["  The DOCUMENT  "], shape="document", budget=budget(spec), tokenizer=FRAMED)
        assert document.contents[0] == "the document"  # stripped and lowercased
        assert document.texts[0] == "Document: the document"

    def test_a_tuple_declares_the_same_ops_for_every_declared_shape(self) -> None:
        spec = TemplateSpec(
            query=(Segment(content="query"),),
            document=(Segment(content="document"),),
            normalize=("strip",),
        )
        assert spec.normalisers("query") == ("strip",)
        assert spec.normalisers("document") == ("strip",)

    def test_a_per_shape_mapping_must_name_every_declared_shape(self) -> None:
        with pytest.raises(ValueError, match="document"):
            TemplateSpec(
                query=(Segment(content="query"),),
                document=(Segment(content="document"),),
                normalize={"query": ("strip",)},
            )
        with pytest.raises(ValueError, match="pair"):
            TemplateSpec(query=(Segment(content="query"),), normalize={"pair": ("strip",), "query": ()})

    def test_normalisation_applies_to_both_pair_spans_before_measuring(self) -> None:
        spec = TemplateSpec(
            pair=(Segment(content="query"), Segment(fixed="\n"), Segment(content="document")),
            normalize=("strip", "lowercase"),
        )
        result = fit(
            [("  The QUERY  ", "  The DOCUMENT ")], shape="pair", budget=budget(spec, max_tokens=64), tokenizer=FRAMED
        )
        assert result.contents[0] == ("the query", "the document")
        assert result.texts[0] == "the query\nthe document"

    def test_the_cut_is_taken_from_the_normalised_text_and_the_row_names_both_ends(self) -> None:
        spec = TemplateSpec(document=(Segment(content="document"),), normalize=("strip", "lowercase"))
        census = TextTruncationCensus()
        result = fit(["  " + LONG], shape="document", budget=budget(spec), tokenizer=FRAMED, census=census)
        assert result.contents[0] == result.contents[0].lower()  # the cut was measured on the normalised text
        row = census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)[0].as_row()
        assert row["original_chars"] == len(LONG) + 2  # the input as given
        assert row["kept_chars"] == len(result.contents[0])  # the normalised, cut text as sent

    def test_the_normalisation_enters_the_template_identity(self) -> None:
        one = identity_payload(TemplateSpec(query=(Segment(content="query"),), normalize=("strip",)))
        other = identity_payload(TemplateSpec(query=(Segment(content="query"),), normalize=("strip", "lowercase")))
        assert one["normalize"] == ["strip"]
        assert other["normalize"] == ["strip", "lowercase"]  # the ops and their order are content


# ---------------------------------------------------------------------------------------------------------------
# Anchors: every declared anchor survives a cut at its declared position
# ---------------------------------------------------------------------------------------------------------------


class TestAnchorsSurviveACut:
    def test_the_suffix_anchor_stays_last(self) -> None:
        spec = document_template()
        result = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        ids = engine_ids(result.texts[0], spec, "document")
        assert len(ids) <= 24
        assert ids[-1] == FRAMED.special_id(END_OF_TEXT_NAME)  # the post-processor's, on top of the template's
        assert ids[-2] == FRAMED.special_id(END_TURN_NAME)  # the suffix anchor, immediately before it

    def test_the_pair_generation_prompt_stays_last(self) -> None:
        spec = pair_template()
        result = fit([("the query", LONG)], shape="pair", budget=budget(spec, max_tokens=64), tokenizer=FRAMED)
        ids = engine_ids(result.texts[0], spec, "pair")
        assert len(ids) <= 64
        assert ids[-2] == FRAMED.special_id(END_TURN_NAME)  # the assistant suffix the score is read from

    def test_a_document_first_pair_keeps_the_query_block(self) -> None:
        spec = document_first_template()
        result = fit([("the query", LONG)], shape="pair", budget=budget(spec, max_tokens=64), tokenizer=FRAMED)
        # The rendered request ends with the intact query block and readout, then the post-processor's anchor:
        # a document-first template makes the query an anchor, so the cut took the document, never the query.
        assert result.texts[0].endswith("\n<query> the query ??")
        assert result.texts[0] == spec.render(
            "pair", FRAMED, query=result.contents[0][0], document=result.contents[0][1]
        )
        assert FRAMED.count(result.contents[0][1]) < FRAMED.count(LONG)  # the document was cut, the query was not
        assert FRAMED.ids(result.texts[0], add_special_tokens=True)[-1] == FRAMED.special_id(END_OF_TEXT_NAME)

    def test_a_first_token_anchor_stays_first(self) -> None:
        spec = first_template()
        result = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        ids = engine_ids(result.texts[0], spec, "document")
        assert ids[0] == FRAMED.special_id(END_TURN_NAME)

    def test_a_marker_anchor_survives(self) -> None:
        spec = marker_template()
        result = fit([LONG], shape="query", budget=budget(spec), tokenizer=FRAMED)
        ids = engine_ids(result.texts[0], spec, "query")
        assert spec.anchor_markers == ("end_turn",)
        assert FRAMED.special_id(END_TURN_NAME) in ids

    def test_a_mean_anchor_keeps_the_frame(self) -> None:
        spec = mean_template()
        result = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        # The frame is reserved whole around the cut: the render is exactly the template around the kept span.
        assert result.texts[0] == spec.render("document", FRAMED, document=result.contents[0])
        assert result.texts[0].startswith("<doc> ")
        assert result.texts[0].endswith(" </doc>")
        assert FRAMED.count(result.contents[0]) < FRAMED.count(LONG)

    def test_a_post_processor_anchor_is_reserved_without_a_template(self) -> None:
        budget_ = TextBudget(tokenizer="test/framed-bpe", max_tokens=24)
        result = fit([LONG], shape="document", budget=budget_, tokenizer=FRAMED)
        assert result.overhead == 1  # the post-processor's end-of-text token, measured on the empty render
        assert FRAMED.ids(result.texts[0], add_special_tokens=True)[-1] == FRAMED.special_id(END_OF_TEXT_NAME)
        assert len(FRAMED.ids(result.texts[0], add_special_tokens=True)) <= 24


class TestByteIdenticalUnderBudget:
    def test_a_query_under_budget_is_the_uncut_render(self) -> None:
        spec = query_template()
        text = "the query"
        result = fit([text], shape="query", budget=budget(spec), tokenizer=FRAMED)
        assert result.texts[0] == spec.render("query", FRAMED, query=text)
        assert result.cuts == ()

    def test_a_pair_under_budget_is_the_uncut_render(self) -> None:
        spec = pair_template()
        pair = ("the query", "the relevant document")
        result = fit([pair], shape="pair", budget=budget(spec, max_tokens=64), tokenizer=FRAMED)
        assert result.texts[0] == spec.render("pair", FRAMED, query=pair[0], document=pair[1])
        assert result.contents[0] == pair
        assert result.ids == ("0",)

    def test_every_assembled_render_fits_the_budget(self) -> None:
        spec = pair_template()
        result = fit([("the query", LONG)], shape="pair", budget=budget(spec, max_tokens=64), tokenizer=FRAMED)
        assert len(engine_ids(result.texts[0], spec, "pair")) <= 64


# ---------------------------------------------------------------------------------------------------------------
# The pair split
# ---------------------------------------------------------------------------------------------------------------


class TestPairSplit:
    def test_the_query_is_cut_to_query_max_tokens_and_the_document_gets_the_rest(self) -> None:
        spec = pair_template()
        pair = (LONG, LONG)
        split = TextBudget(
            tokenizer="test/framed-bpe", max_tokens=64, query_max_tokens=8, template=spec, on_overflow="cut"
        )
        result = fit([pair], shape="pair", budget=split, tokenizer=FRAMED)
        query_final, document_final = result.contents[0]
        assert FRAMED.count(query_final) <= 8
        ids = engine_ids(result.texts[0], spec, "pair")
        assert len(ids) <= 64
        assert ids[-2] == FRAMED.special_id(END_TURN_NAME)
        assert FRAMED.count(document_final) < FRAMED.count(LONG)

    def test_a_query_that_fills_the_budget_refuses_with_a_hint(self) -> None:
        spec = pair_template()
        split = TextBudget(tokenizer="test/framed-bpe", max_tokens=64, template=spec)
        with pytest.raises(TextBudgetExceededError, match="query_max_tokens"):
            fit([(LONG, "the document")], shape="pair", budget=split, tokenizer=FRAMED)

    def test_query_max_tokens_at_or_over_max_tokens_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            TextBudget(tokenizer="t", max_tokens=40, query_max_tokens=40)
        with pytest.raises(ValueError, match="max_tokens"):
            TextBudget(tokenizer="t", max_tokens=40, query_max_tokens=41)


# ---------------------------------------------------------------------------------------------------------------
# Chunking: every chunk carries the full template; the census names the aggregation
# ---------------------------------------------------------------------------------------------------------------


class TestChunking:
    def test_every_chunk_carries_the_full_template_and_names_max_aggregation(self) -> None:
        spec = document_template()
        policy = ChunkPolicy(max_tokens=6, overlap_tokens=0)
        chunked = TextBudget(
            tokenizer="test/framed-bpe", max_tokens=24, template=spec, on_overflow="chunk", chunk=policy
        )
        census = TextTruncationCensus()
        result = fit([LONG], shape="document", budget=chunked, tokenizer=FRAMED, ids=["d0"], corpus="c", census=census)
        assert len(result.texts) > 1
        for out_id, text in zip(result.ids, result.texts, strict=True):
            assert out_id.startswith("d0#")
            # Every chunk carries the full template: the frame opens and the suffix anchor closes each one
            # (string-level: a byte-level join may merge the frame's last token with the content's first).
            assert text.startswith("<doc> ")
            assert text.endswith(" <|end_turn|>")
            assert len(engine_ids(text, spec, "document")) <= 20
        assert "".join(result.contents) == LONG  # overlap 0: the chunks tile the document
        assert result.aggregation == "max"
        rows = [cut.as_row() for cut in census.cuts()]
        assert rows
        assert all(
            row["mechanism"] == "text_budget" and row["aggregation"] == "max" and row["budget_source"] == "tokenizer"
            for row in rows
        )
        scores = {out_id: float(k) for k, out_id in enumerate(result.ids)}
        pooled = max_pool_scores_by_document(scores, result.chunk_mapping)
        assert pooled == {"d0": float(len(result.ids) - 1)}

    def test_an_unsplit_document_keeps_its_id(self) -> None:
        spec = document_template()
        chunked = TextBudget(
            tokenizer="test/framed-bpe",
            max_tokens=64,
            template=spec,
            on_overflow="chunk",
            chunk=ChunkPolicy(max_tokens=10, overlap_tokens=2),
        )
        result = fit(["short", LONG], shape="document", budget=chunked, tokenizer=FRAMED, ids=["a", "b"])
        assert result.ids[0] == "a"
        assert result.ids[1].startswith("b#")
        assert result.chunk_mapping is not None
        assert result.chunk_mapping["a"] == "a"

    def test_a_chunk_geometry_that_cannot_fit_is_refused(self) -> None:
        spec = document_template()
        chunked = TextBudget(
            tokenizer="test/framed-bpe",
            max_tokens=20,
            template=spec,
            on_overflow="chunk",
            chunk=ChunkPolicy(max_tokens=64, overlap_tokens=0),
        )
        with pytest.raises(ConfigError, match="chunk.max_tokens"):
            fit([LONG], shape="document", budget=chunked, tokenizer=FRAMED)

    def test_a_query_is_never_chunked(self) -> None:
        spec = query_template()
        chunked = TextBudget(
            tokenizer="test/framed-bpe",
            max_tokens=24,
            template=spec,
            on_overflow="chunk",
            chunk=ChunkPolicy(max_tokens=6, overlap_tokens=0),
        )
        with pytest.raises(TextBudgetExceededError, match="never"):
            fit([LONG], shape="query", budget=chunked, tokenizer=FRAMED)


# ---------------------------------------------------------------------------------------------------------------
# Fail, cut and census
# ---------------------------------------------------------------------------------------------------------------


class TestFailAndCut:
    def test_a_fail_policy_refuses_an_over_length_input(self) -> None:
        spec = document_template()
        budget_ = TextBudget(tokenizer="test/framed-bpe", max_tokens=24, template=spec, on_overflow="fail")
        with pytest.raises(TextBudgetExceededError):
            fit([LONG], shape="document", budget=budget_, tokenizer=FRAMED)

    def test_a_cut_is_recorded_with_its_budget_source_and_shape(self) -> None:
        census = TextTruncationCensus()
        result = fit([LONG], shape="document", budget=budget(document_template()), tokenizer=FRAMED, census=census)
        assert len(census) == 1
        row = census.cuts()[0].as_row()
        assert row["mechanism"] == "text_budget"
        assert row["budget_source"] == "tokenizer"
        assert row["shape"] == "document"
        assert row["original_tokens"] == FRAMED.count(LONG)
        assert row["kept_tokens"] == FRAMED.count(result.contents[0])
        assert row["tokens_lost"] > 0

    def test_an_under_budget_input_is_not_recorded(self) -> None:
        census = TextTruncationCensus()
        fit(["the query"], shape="document", budget=budget(document_template()), tokenizer=FRAMED, census=census)
        assert len(census) == 0


# ---------------------------------------------------------------------------------------------------------------
# The vendor profile (no tokenizer) and the media hook
# ---------------------------------------------------------------------------------------------------------------


class TestVendorBudget:
    def test_content_goes_uncut_and_the_budget_is_recorded_as_vendor(self, caplog: pytest.LogCaptureFixture) -> None:
        vendor = TextBudget(tokenizer=None, max_tokens=4096)
        census = TextTruncationCensus()
        with caplog.at_level(logging.WARNING, logger="rcp_ndcg.data.preprocess"):
            result = fit([LONG], shape="document", budget=vendor, ids=["d0"], corpus="c", census=census)
        assert result.texts == (LONG,)
        assert result.contents == (LONG,)
        assert result.budget_source == "vendor"
        assert result.overhead is None
        row = census.cuts()[0].as_row()
        assert row["mechanism"] == "text_budget"
        assert row["budget_source"] == "vendor"
        assert row["doc_id"] == "<budget>"
        warnings = [record for record in caplog.records if "vendor" in record.message.lower()]
        assert len(warnings) == 1
        fit([LONG], shape="document", budget=vendor, ids=["d0"], corpus="c")  # the second call warns no more
        assert len([record for record in caplog.records if "vendor" in record.message.lower()]) == 1

    def test_a_vendor_pair_returns_parts_not_rendered_text(self) -> None:
        vendor = TextBudget(tokenizer=None, max_tokens=4096)
        result = fit([("the query", "the document")], shape="pair", budget=vendor)
        assert result.texts == ()
        assert result.contents == (("the query", "the document"),)

    def test_the_vendor_budget_row_is_recorded_when_a_census_arrives_later(self) -> None:
        """The budget row is a fact of the run, not of the first batch: a census attached after an earlier
        call still gets its row (the warning stays once per corpus, the record per census)."""
        vendor = TextBudget(tokenizer=None, max_tokens=4096)
        fit([LONG], shape="document", budget=vendor, corpus="c")  # no census here
        census = TextTruncationCensus()
        fit([LONG], shape="document", budget=vendor, corpus="c", census=census)
        rows = census.cuts()
        assert len(rows) == 1 and rows[0].as_row()["budget_source"] == "vendor"
        fit([LONG], shape="document", budget=vendor, corpus="c", census=census)  # batching: still one row
        assert len(census.cuts()) == 1

    def test_a_budget_without_a_tokenizer_refuses_inert_policies(self) -> None:
        """Nothing is defaulted silently: chunk/fail and a query split are inert without a tokenizer."""
        with pytest.raises(ConfigError, match="uncut"):
            TextBudget(
                tokenizer=None, max_tokens=4096, on_overflow="chunk", chunk=ChunkPolicy(max_tokens=8, overlap_tokens=0)
            )
        with pytest.raises(ConfigError, match="uncut"):
            TextBudget(tokenizer=None, max_tokens=4096, on_overflow="fail")
        with pytest.raises(ConfigError, match="uncut"):
            TextBudget(tokenizer=None, max_tokens=4096, query_max_tokens=64)

    def test_a_budget_without_a_tokenizer_refuses_a_template_it_cannot_render(self) -> None:
        """A hosted profile sends content as stored: a declared template would be silently ignored."""
        with pytest.raises(ConfigError, match="template"):
            TextBudget(tokenizer=None, max_tokens=4096, template=document_template())


# ---------------------------------------------------------------------------------------------------------------
# Per-shape budgets: query_max_tokens caps the query shape on the embedding roles
# ---------------------------------------------------------------------------------------------------------------


class TestPerShapeBudget:
    def test_query_max_tokens_is_the_query_shapes_whole_budget(self) -> None:
        """Per-shape budgets (the topk hand-off): on the query shape ``query_max_tokens`` IS the budget (its
        whole budget there), and ``max_tokens`` keeps capping the document shape."""
        spec = TemplateSpec(
            query=(Segment(fixed="Q: "), Segment(content="query")),
            document=(Segment(fixed="D: "), Segment(content="document")),
        )
        split = TextBudget(tokenizer="test/framed-bpe", max_tokens=64, query_max_tokens=10, template=spec)
        query = fit([LONG], shape="query", budget=split, tokenizer=FRAMED)
        assert len(engine_ids(query.texts[0], spec, "query")) <= 10
        assert FRAMED.count(query.contents[0]) < FRAMED.count(LONG)
        document = fit([LONG], shape="document", budget=split, tokenizer=FRAMED, ids=["d"])
        # The same budget, document shape: capped at max_tokens (64), not at the query's 10.
        assert 10 < len(engine_ids(document.texts[0], spec, "document")) <= 64

    def test_the_census_rows_name_the_shapes_budget(self) -> None:
        census = TextTruncationCensus()
        spec = TemplateSpec(
            query=(Segment(fixed="Q: "), Segment(content="query")),
            document=(Segment(fixed="D: "), Segment(content="document")),
        )
        split = TextBudget(tokenizer="test/framed-bpe", max_tokens=64, query_max_tokens=10, template=spec)
        fit([LONG], shape="query", budget=split, tokenizer=FRAMED, census=census)
        rows = [cut.as_row() for cut in census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)]
        assert rows and rows[0]["budget_tokens"] == 10 and rows[0]["shape"] == "query"
        document_census = TextTruncationCensus()
        fit([LONG], shape="document", budget=split, tokenizer=FRAMED, census=document_census, ids=["d"])
        document_rows = [cut.as_row() for cut in document_census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)]
        assert document_rows[0]["budget_tokens"] == 64

    def test_a_query_shape_refuses_a_budget_the_frame_alone_fills(self) -> None:
        split = TextBudget(tokenizer="test/framed-bpe", max_tokens=64, query_max_tokens=4, template=query_template())
        with pytest.raises(ConfigError, match="4"):
            fit([LONG], shape="query", budget=split, tokenizer=FRAMED)

    def test_an_undeclared_shape_is_a_typed_error_even_with_a_per_shape_flag(self) -> None:
        spec = TemplateSpec(
            query=(Segment(content="query"), Segment(fixed=".")),
            document=(Segment(content="document"),),
            add_special_tokens={"query": False, "document": True},
        )
        with pytest.raises(ConfigError, match="pair"):
            spec.adds_special_tokens("pair")

    def test_a_single_piece_under_chunk_keeps_its_input_id(self) -> None:
        """A pair whose document splits into one piece (an empty document) is one request, not a chunk:
        it keeps the input's id, like chunk_ranking_example keeps an unsplit document."""
        chunked = TextBudget(
            tokenizer="test/framed-bpe",
            max_tokens=64,
            template=pair_template(),
            on_overflow="chunk",
            chunk=ChunkPolicy(max_tokens=6, overlap_tokens=0),
        )
        result = fit([("the query", "")], shape="pair", budget=chunked, tokenizer=FRAMED, ids=["d"])
        assert result.ids == ("d",)
        assert result.chunk_mapping is None
        assert result.aggregation is None

    def test_the_budget_refuses_a_different_tokenizer_than_it_declares(self) -> None:
        vendor_named = TextBudget(tokenizer="test/framed-bpe", max_tokens=24)
        with pytest.raises(ConfigError, match="tokenizer"):
            fit([LONG], shape="document", budget=vendor_named, tokenizer=WORDS)
        same = fit([LONG], shape="document", budget=vendor_named, tokenizer=FRAMED)
        assert same.budget_source == "tokenizer"

    def test_the_budget_refuses_to_run_without_the_tokenizer_it_declares(self) -> None:
        """A budget that declares a tokenizer is measured and cut; handing it no tokenizer is a caller
        error, not a hosted profile -- the vendor path is for budgets that declare none."""
        declared = TextBudget(tokenizer="test/framed-bpe", max_tokens=24)
        with pytest.raises(ConfigError, match="tokenizer"):
            fit([LONG], shape="document", budget=declared, tokenizer=None)

    def test_media_tokens_are_refused_without_a_tokenizer(self) -> None:
        """A hosted profile cannot reserve media it cannot count: the declaration is refused, never ignored."""
        vendor = TextBudget(tokenizer=None, max_tokens=4096)
        with pytest.raises(ConfigError, match="media"):
            fit([LONG], shape="document", budget=vendor, media_tokens=[50])

    def test_the_pair_query_cut_is_verified_against_the_assembled_render(self) -> None:
        """The query span's cut must be measured with the query in its own span: an asymmetric frame whose
        two joins tokenize differently would otherwise ship a render over the budget -- an over-limit
        request reaches the engine, which cuts it itself (the wrapped-prompt defect class again)."""
        spec = TemplateSpec(
            pair=(
                Segment(fixed="q "),
                Segment(content="query"),
                Segment(fixed=" mid"),
                Segment(content="document"),
                Segment(fixed=" end"),
            ),
            add_special_tokens={"pair": False},
        )
        split = TextBudget(
            tokenizer="test/word-level", max_tokens=8, query_max_tokens=6, template=spec, on_overflow="cut"
        )
        result = fit(
            [("relevant relevant relevant relevant relevant five", "")], shape="pair", budget=split, tokenizer=WORDS
        )
        assert len(WORDS.ids(result.texts[0])) <= 8, result.texts[0]

    def test_duplicate_output_ids_are_refused(self) -> None:
        """Two outputs with one id would pool one score over the other (chunk_ranking_example refuses the
        same collision)."""
        chunked = TextBudget(
            tokenizer="test/framed-bpe",
            max_tokens=64,
            template=document_template(),
            on_overflow="chunk",
            chunk=ChunkPolicy(max_tokens=10, overlap_tokens=2),
        )
        with pytest.raises(DataError, match="a#0"):
            # The first input chunks into 'a#0', colliding with the second input's own id.
            fit([LONG, "the query"], shape="document", budget=chunked, tokenizer=FRAMED, ids=["a", "a#0"])


class TestMediaHook:
    def test_media_tokens_are_reserved_and_never_cut(self) -> None:
        spec = document_template()
        without = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED)
        with_media = fit([LONG], shape="document", budget=budget(spec), tokenizer=FRAMED, media_tokens=[4])
        # The media's share came off the content budget, not the frame's: less text is kept than without media,
        # and the assembled render (media ride beside it) still fits the declared budget.
        assert FRAMED.count(with_media.contents[0]) < FRAMED.count(without.contents[0])
        assert len(engine_ids(with_media.texts[0], spec, "document")) <= 24
        assert engine_ids(with_media.texts[0], spec, "document")[-2] == FRAMED.special_id(END_TURN_NAME)

    def test_media_that_fills_the_budget_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="media"):
            fit([LONG], shape="document", budget=budget(document_template()), tokenizer=FRAMED, media_tokens=[24])


# ---------------------------------------------------------------------------------------------------------------
# Identity and the word-level family
# ---------------------------------------------------------------------------------------------------------------


class TestIdentityAndFamilies:
    def test_the_tokenizer_sha256_is_content_and_its_name_runtime(self) -> None:
        check_declarations(TextBudget)
        check_declarations(ChunkPolicy)
        plain = TextBudget(tokenizer="test/framed-bpe", max_tokens=24)
        payload = identity_payload(plain)
        assert "tokenizer" not in payload and payload["max_tokens"] == 24
        assert plain.identity(FRAMED)["tokenizer_sha256"] == FRAMED.sha256
        assert plain.identity(None) == payload  # vendor mode: no file, no hash
        other = TextBudget(tokenizer="test/word-level", max_tokens=24)
        assert identity_payload(other) == payload  # the name is runtime: same numbers, same identity

    def test_the_template_enters_the_identity(self) -> None:
        framed = identity_payload(TextBudget(tokenizer="t", max_tokens=24, template=query_template()))
        assert "template" in framed
        assert "template" not in identity_payload(TextBudget(tokenizer="t", max_tokens=24))

    def test_a_chunk_geometry_is_reused_not_copied(self) -> None:
        policy = ChunkPolicy(max_tokens=6, overlap_tokens=0)
        chunked = TextBudget(
            tokenizer="test/framed-bpe", max_tokens=24, on_overflow="chunk", chunk=policy, template=document_template()
        )
        assert chunked.chunk is policy

    def test_the_word_level_family_cuts_the_same_way(self) -> None:
        spec = TemplateSpec(document=(Segment(fixed="<doc> "), Segment(content="document"), Segment(fixed=" </doc>")))
        budget_ = TextBudget(tokenizer="test/word-level", max_tokens=8, template=spec)
        result = fit([LONG], shape="document", budget=budget_, tokenizer=WORDS)
        assert WORDS.count(result.contents[0]) <= 8
        assert result.texts[0].startswith("<doc> ")
        assert result.texts[0].endswith(" </doc>")


# ---------------------------------------------------------------------------------------------------------------
# The wrapped-prompt defect class: cutting the whole render from the right
# ---------------------------------------------------------------------------------------------------------------


class TestTheWrappedPromptDefectClass:
    def test_cutting_the_whole_render_from_the_right_loses_the_anchor_and_fit_does_not(self) -> None:
        """The mutation: render, then keep the first ``max_tokens`` ids -- the wrapped-prompt defect.

        The anchor test below is red for the defective ids and green for the mechanism's, which is
        what pins this lane's rule: the cut applies to the content spans only, and the template is
        re-attached after the cut.
        """
        spec = pair_template()
        pair = ("the query", LONG)
        # The defective algorithm: render the whole prompt, then keep its first 64 ids (a right cut of the
        # rendered prompt at the cap -- the wrapped-prompt defect, and today's engine-side truncation).
        rendered = spec.render("pair", FRAMED, query=pair[0], document=pair[1])
        defective = FRAMED.ids(rendered, add_special_tokens=spec.adds_special_tokens("pair"))[:64]
        anchor = FRAMED.special_id(END_TURN_NAME)
        assert not ends_with(defective, [anchor])  # RED: the score would be read off an interior token

        pair_budget = TextBudget(tokenizer="test/framed-bpe", max_tokens=64, template=spec, on_overflow="cut")
        result = fit([pair], shape="pair", budget=pair_budget, tokenizer=FRAMED)
        ids = engine_ids(result.texts[0], spec, "pair")
        assert len(ids) <= 64
        assert ids[-2] == anchor  # GREEN: the anchor is at its declared position, before the post-processor's
