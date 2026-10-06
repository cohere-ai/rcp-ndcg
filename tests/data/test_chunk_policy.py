"""Chunking: the token-based policy, the splitter, the per-query producer and the aggregation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart

from rcp_ndcg.data.preprocess import (
    DEFAULT_TEXT_POLICY,
    ChunkPolicy,
    Preprocessing,
    TextPolicy,
    apply_text_policy,
    chunk_ranking_example,
    document_ids_from_chunks,
    max_pool_rubric_window_by_document,
    max_pool_scores_by_document,
    split_into_chunks,
)
from rcp_ndcg.errors import DataError
from tests._tokenizers import byte_bpe_tokenizer, word_tokenizer

WORDS = word_tokenizer()
GEOMETRY = ChunkPolicy(max_tokens=20, overlap_tokens=5)
CHUNK = TextPolicy(on_overflow="chunk", chunk=GEOMETRY)
#: 36 tokens t0..t35 (one word each) with irregular whitespace between them.
TOKENS = [f"t{i}" for i in range(36)]
LONG = "".join(token + ("  " if i % 3 == 0 else "\n" if i % 3 == 1 else " ") for i, token in enumerate(TOKENS)).rstrip()


def _tokens(piece: str) -> list[str]:
    return piece.split()


class TestPolicy:
    def test_the_chunk_mode_needs_a_geometry_and_takes_its_cap_from_it(self) -> None:
        assert CHUNK.max_tokens == 20
        assert CHUNK.chunk is not None and CHUNK.chunk.descriptor == "chunk:20/5:max"
        with pytest.raises(ValidationError, match="needs a chunk geometry"):
            TextPolicy(on_overflow="chunk")
        with pytest.raises(ValidationError, match="must be unset or equal"):
            TextPolicy(on_overflow="chunk", max_tokens=99, chunk=GEOMETRY)
        with pytest.raises(ValidationError, match="'chunk' only"):
            TextPolicy(on_overflow="truncate", chunk=GEOMETRY)
        with pytest.raises(ValidationError, match="must advance"):
            ChunkPolicy(max_tokens=10, overlap_tokens=10)

    def test_the_other_modes_serialise_without_chunk_fields(self) -> None:
        assert DEFAULT_TEXT_POLICY.model_dump(mode="json") == {"on_overflow": "keep", "max_tokens": None}
        loud = TextPolicy(on_overflow="truncate", max_tokens=99)
        assert loud.model_dump(mode="json") == {"on_overflow": "truncate", "max_tokens": 99}

    def test_the_chunk_geometry_is_part_of_the_identity(self) -> None:
        other = TextPolicy(on_overflow="chunk", chunk=ChunkPolicy(max_tokens=20, overlap_tokens=6))
        assert Preprocessing(text=CHUNK).key != Preprocessing(text=other).key
        declared = {"on_overflow": "chunk", "chunk": {"max_tokens": 20, "overlap_tokens": 5}}
        assert TextPolicy.model_validate(declared) == CHUNK

    def test_the_loader_keeps_a_chunked_corpus_whole(self) -> None:
        assert apply_text_policy(LONG, doc_id="d", policy=CHUNK) is LONG


class TestSplitIntoChunks:
    def test_chunks_are_verbatim_slices_with_the_token_overlap(self) -> None:
        chunks = split_into_chunks(LONG, GEOMETRY, WORDS)
        assert [_tokens(chunk) for chunk in chunks] == [TOKENS[0:20], TOKENS[15:35], TOKENS[30:36]]
        assert all(chunk in LONG for chunk in chunks)  # slices of the document, whitespace and all
        assert chunks[0] == LONG[: LONG.index("t19") + 3] and chunks[-1] == LONG[LONG.index("t30") :]
        assert all(WORDS.count(chunk) <= 20 for chunk in chunks)

    def test_sub_word_chunks_stay_verbatim_and_within_the_cap(self) -> None:
        bpe = byte_bpe_tokenizer()
        text = "Unbelievable results: the relevant passage 日本語 🙂 answers the query with evidence. " * 3
        policy = ChunkPolicy(max_tokens=12, overlap_tokens=3)
        chunks = split_into_chunks(text, policy, bpe)
        assert len(chunks) > 2 and chunks[0].startswith("Unbelievable") and text.endswith(chunks[-1])
        position = 0
        for chunk in chunks:
            start = text.index(chunk, max(position - len(chunk), 0))
            assert start <= position  # consecutive chunks overlap or touch: nothing is skipped
            position = start + len(chunk)
            assert bpe.count(chunk) <= 12
        assert position == len(text)

    def test_without_overlap_the_chunks_cover_the_text_between_tokens(self) -> None:
        """The whitespace between two chunks' tokens was once in no chunk."""
        chunks = split_into_chunks(LONG, ChunkPolicy(max_tokens=3, overlap_tokens=0), WORDS)

        assert len(chunks) > 2
        assert "".join(chunks) == LONG

    def test_short_text_is_one_chunk(self) -> None:
        assert split_into_chunks("short", GEOMETRY, WORDS) == ["short"]

    def test_a_one_token_chunk_may_sit_over_the_cap(self) -> None:
        """The one-token boundary (the QA survivor 15): a chunk is at least one document token, so one
        token whose text re-tokenizes longer alone (a multi-byte character split across byte tokens) makes
        a chunk over the cap -- the resolver breaks there instead of shrinking past it (and a multi-byte
        edge character may repeat, the declared duplication-only caveat)."""
        bpe = byte_bpe_tokenizer()
        text = "日 本"
        assert bpe.count("日") > 1  # the single token re-tokenizes longer on its own
        chunks = split_into_chunks(text, ChunkPolicy(max_tokens=1, overlap_tokens=0), bpe)
        assert chunks[0] == "日"  # the first chunk is the single token, whole, over the cap -- declared
        assert bpe.count(chunks[0]) > 1
        assert "".join(chunks) == "日日 本本本"  # coverage with the multi-byte edge repeat, nothing lost


class TestChunkRankingExample:
    def _example(self) -> RankingExample:
        return RankingExample(
            query_id="q1",
            query="q",
            doc_ids=["short", "long", "tail"],
            docs=["a short one", LONG, "also short"],
            scores=[3.0, 2.0, 1.0],
            qrels={"long": 1},
        )

    def test_a_document_over_the_cap_becomes_its_chunks_in_place(self) -> None:
        example = self._example()
        assert dict(zip(example.doc_ids, example.docs, strict=True))["long"] == LONG  # the unchunked row
        chunked = chunk_ranking_example(example, GEOMETRY, WORDS)
        assert chunked.doc_ids == ["short", "long#0", "long#1", "long#2", "tail"]
        assert chunked.chunk_mapping == {
            "short": "short",
            "long#0": "long",
            "long#1": "long",
            "long#2": "long",
            "tail": "tail",
        }
        assert chunked.scores == [3.0, 2.0, 2.0, 2.0, 1.0]
        assert chunked.qrels == {"long": 1}
        # Built fresh, not copied: the per-row caches describe the chunked row.
        assert _tokens(dict(zip(chunked.doc_ids, chunked.docs, strict=True))["long#1"]) == TOKENS[15:35]

    def test_nothing_over_the_cap_leaves_the_example_untouched(self) -> None:
        example = RankingExample(query_id="q", query="q", doc_ids=["a"], docs=["tiny"])
        assert chunk_ranking_example(example, GEOMETRY, WORDS) is example
        assert example.chunk_mapping is None

    def test_media_documents_are_never_split(self) -> None:
        page = Content.from_parts(
            [TextPart(text=LONG), ImagePart(ref=MediaRef(uri="p.png", sha256="0" * 64, mime="image/png"))]
        )
        example = RankingExample(
            query_id="q", query="q", doc_ids=["page", "text"], contents=[page, Content.from_text(LONG)]
        )
        chunked = chunk_ranking_example(example, GEOMETRY, WORDS)
        assert chunked.doc_ids == ["page", "text#0", "text#1", "text#2"]
        assert chunked.contents is not None and chunked.contents[0] == page

    def test_refusals(self) -> None:
        with pytest.raises(DataError, match="hydrate"):
            chunk_ranking_example(RankingExample(query_id="q", query="q", doc_ids=["a"]), GEOMETRY, WORDS)
        already = self._example().model_copy(update={"chunk_mapping": {"short": "short"}})
        with pytest.raises(DataError, match="already carries"):
            chunk_ranking_example(already, GEOMETRY, WORDS)
        clash = RankingExample(query_id="q", query="q", doc_ids=["long", "long#0"], docs=[LONG, "x"])
        with pytest.raises(DataError, match="collide"):
            chunk_ranking_example(clash, GEOMETRY, WORDS)


class TestAggregation:
    MAPPING = {"a#0": "a", "a#1": "a", "b": "b"}

    def test_chunks_map_back_to_their_documents_in_first_chunk_order(self) -> None:
        assert document_ids_from_chunks(["a#0", "b", "a#1"], self.MAPPING) == ["a", "b"]

    def test_scores_pool_to_the_best_chunk(self) -> None:
        assert max_pool_scores_by_document({"a#0": 1.0, "b": 0.5, "a#1": 3.0}, self.MAPPING) == {"a": 3.0, "b": 0.5}

    def test_a_document_passes_a_criterion_when_any_chunk_in_the_window_does(self) -> None:
        window = {"a#0": {"C1": 1, "C2": 0}, "a#1": {"C1": 0, "C2": 1}, "b": {"C1": 0, "C2": 0}}
        assert max_pool_rubric_window_by_document(window, self.MAPPING) == {
            "a": {"C1": 1, "C2": 1},
            "b": {"C1": 0, "C2": 0},
        }
        with pytest.raises(DataError, match="inconsistent"):
            max_pool_rubric_window_by_document({"a#0": {"C1": 1}, "a#1": {"C2": 1}}, self.MAPPING)
        with pytest.raises(DataError, match="no document ID"):
            max_pool_scores_by_document({"c": 1.0}, self.MAPPING)
