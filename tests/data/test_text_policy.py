"""Guards for the declared text truncation policy, in tokens of the judge's tokenizer.

Which cut a document receives must not depend on the corpus's file format: the
policy is one declared object. These tests pin its vocabulary, its refusals, the
single choke-point through which a document may be shortened, and the cut itself:
a verbatim prefix of the document at one of its token boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rcp_ndcg.data.preprocess import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEXT_POLICY,
    DocumentOverCapError,
    TextPolicy,
    TextTruncationCensus,
    apply_text_policy,
    token_prefix,
)
from rcp_ndcg.errors import ConfigError
from tests._tokenizers import byte_bpe_tokenizer, word_tokenizer

WORDS = word_tokenizer()
BPE = byte_bpe_tokenizer()
#: Irregular whitespace, punctuation, multi-byte characters and emoji: what a decode-based cut would not survive.
TEXT = "The  evidence,\n\nof the query:\trelevant!  Ünïcödé 日本語 🙂🙂 answer &amp; page one  two\nthree four five."


class TestPolicyVocabulary:
    def test_default_is_keep(self) -> None:
        """An undeclared policy is ``keep`` -- declared, not accidental."""
        assert DEFAULT_TEXT_POLICY.on_overflow == "keep"
        assert DEFAULT_TEXT_POLICY.max_tokens is None

    def test_a_cutting_mode_defaults_its_cap(self) -> None:
        assert TextPolicy(on_overflow="truncate").max_tokens == DEFAULT_MAX_TOKENS
        assert TextPolicy(on_overflow="fail").max_tokens == DEFAULT_MAX_TOKENS
        assert TextPolicy(on_overflow="truncate", max_tokens=1234).max_tokens == 1234

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="chunky"):
            TextPolicy(on_overflow="chunky")  # type: ignore[arg-type]


class TestPolicyRefusals:
    def test_keep_takes_no_cap(self) -> None:
        with pytest.raises(ValueError, match="'keep' takes no max_tokens"):
            TextPolicy(on_overflow="keep", max_tokens=500)

    @pytest.mark.parametrize("mode", ["truncate", "fail"])
    def test_a_token_limit_without_a_tokenizer_is_refused(self, mode: str) -> None:
        """No character fallback: a policy that cuts needs the judge's tokenizer, even for a short document."""
        with pytest.raises(ConfigError, match="tokenizer") as caught:
            apply_text_policy("short", doc_id="d", policy=TextPolicy(on_overflow=mode, max_tokens=10))  # type: ignore[arg-type]
        assert "judge.tokenizer" in (caught.value.hint or "")

    def test_keep_needs_no_tokenizer(self) -> None:
        text = "z " * 100
        assert apply_text_policy(text, doc_id="d") is text


class TestTokenPrefix:
    @pytest.mark.parametrize("tokenizer", [WORDS, BPE], ids=["word-level", "byte-bpe"])
    def test_a_cut_is_a_verbatim_prefix_within_the_budget(self, tokenizer) -> None:
        total = tokenizer.count(TEXT)
        for budget in range(total + 1):
            prefix = token_prefix(TEXT, budget, tokenizer)
            assert TEXT.startswith(prefix), budget
            assert tokenizer.count(prefix) <= budget, budget
        assert token_prefix(TEXT, total, tokenizer) is TEXT

    def test_the_cut_ends_at_a_token_boundary_and_keeps_the_whitespace_inside(self) -> None:
        # Word-level tokens: The / evidence / , / of / the / query / : -- whitespace belongs to no token.
        assert token_prefix(TEXT, 6, WORDS) == "The  evidence,\n\nof the query"

    def test_a_cut_word_that_retokenizes_longer_moves_back_a_boundary(self) -> None:
        text = "unbelievable unbelievable"
        for budget in range(BPE.count(text)):
            prefix = token_prefix(text, budget, BPE)
            assert text.startswith(prefix) and BPE.count(prefix) <= budget

    def test_the_rendered_form_is_what_is_counted(self) -> None:
        """Counted as the prompt carries it: escaping ``&`` makes more tokens, so the prefix is shorter."""
        text = "the & the & the & the"
        plain = token_prefix(text, 5, WORDS)
        escaped = token_prefix(text, 5, WORDS, rendered=lambda piece: piece.replace("&", "&amp;"))
        assert text.startswith(escaped) and len(escaped) < len(plain)
        assert WORDS.count(escaped.replace("&", "&amp;")) <= 5


class TestApplyTextPolicy:
    def test_truncate_caps_at_the_declared_limit(self) -> None:
        policy = TextPolicy(on_overflow="truncate", max_tokens=10)
        cut = apply_text_policy(TEXT, doc_id="d", policy=policy, tokenizer=WORDS)
        assert TEXT.startswith(cut) and WORDS.count(cut) == 10
        short = "the query"
        assert apply_text_policy(short, doc_id="d", policy=policy, tokenizer=WORDS) is short

    def test_truncate_records_every_cut_with_its_id_tokens_and_chars(self) -> None:
        census = TextTruncationCensus()
        policy = TextPolicy(on_overflow="truncate", max_tokens=10)
        apply_text_policy("the " * 25, doc_id="doc-1", corpus="corpus-a", policy=policy, tokenizer=WORDS, census=census)
        apply_text_policy("a " * 12, doc_id="doc-2", corpus="corpus-a", policy=policy, tokenizer=WORDS, census=census)
        apply_text_policy("short", doc_id="doc-3", corpus="corpus-a", policy=policy, tokenizer=WORDS, census=census)
        assert len(census) == 2  # the short document is not a cut
        rows = [cut.as_row() for cut in census.cuts()]
        assert rows[0] == {
            "corpus": "corpus-a",
            "doc_id": "doc-1",
            "original_tokens": 25,
            "kept_tokens": 10,
            "tokens_lost": 15,
            "original_chars": 100,
            "kept_chars": 39,  # "the " * 9 + "the": the cut ends at the tenth token
            "chars_lost": 61,
            "mechanism": "doc_policy",
            "query_id": None,
        }
        assert rows[1]["doc_id"] == "doc-2"

    def test_fail_mode_refuses_over_cap_and_says_what_to_do(self) -> None:
        policy = TextPolicy(on_overflow="fail", max_tokens=10)
        with pytest.raises(DocumentOverCapError, match="'truncate'"):
            apply_text_policy("a " * 11, doc_id="doc-1", corpus="corpus-a", policy=policy, tokenizer=WORDS)
        text = "a " * 10
        assert apply_text_policy(text, doc_id="doc-1", policy=policy, tokenizer=WORDS) is text

    def test_mechanisms_are_separate_namespaces(self) -> None:
        census = TextTruncationCensus()
        with pytest.raises(ValueError, match="mechanism"):
            census.record(
                corpus="c",
                doc_id="d",
                original_chars=20,
                kept_chars=10,
                original_tokens=10,
                kept_tokens=5,
                mechanism="mystery",
            )


class TestCensusSink:
    def test_sink_file_receives_one_row_per_cut(self, tmp_path: Path) -> None:
        sink = tmp_path / "census.jsonl"
        census = TextTruncationCensus(sink=sink)
        policy = TextPolicy(on_overflow="truncate", max_tokens=10)
        apply_text_policy("a " * 20, doc_id="d1", policy=policy, tokenizer=WORDS, census=census)
        apply_text_policy("b " * 20, doc_id="d2", policy=policy, tokenizer=WORDS, census=census)
        rows = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
        assert [row["doc_id"] for row in rows] == ["d1", "d2"]
        assert rows[0]["tokens_lost"] == 10

    def test_sink_errors_do_not_lose_the_cut(self, tmp_path: Path) -> None:
        # A read-only sink directory must not crash the loader that records.
        sink = tmp_path / "nope" / "census.jsonl"
        census = TextTruncationCensus(sink=sink)
        policy = TextPolicy(on_overflow="truncate", max_tokens=10)
        assert (
            WORDS.count(apply_text_policy("a " * 20, doc_id="d1", policy=policy, tokenizer=WORDS, census=census)) == 10
        )
        assert len(census) == 1
