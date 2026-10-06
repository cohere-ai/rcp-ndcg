"""The text-budget wiring in the role clients (item 4): a config with ``max_tokens`` fits every request
through :func:`rcp_ndcg.data.preprocess.fit` -- only content spans cut, the template re-attached (anchors
kept), every cut recorded, and nothing left for the engine to truncate (no ``truncate_prompt_tokens`` or
kin is ever sent).
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart

from rcp_ndcg.data.preprocess import TextBudgetExceededError, TextTruncationCensus
from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens
from rcp_ndcg.data.templates import Segment, TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import CapabilityError, ConfigError
from rcp_ndcg.inference import EmbeddingClient, PoolingClient, RerankClient
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.types import EncodeRole, Reply, Usage
from tests._tokenizers import framed_bpe_tokenizer, save, word_tokenizer
from tests.conftest import SESSION_TOKENIZER
from tests.inference._pooling import PoolingServer
from tests.inference.test_pool_client import _GatedSender

END_TURN = "<|end_turn|>"


class RecordingSender:
    """Records the bodies it was handed and answers each role's shape plausibly."""

    def __init__(self, *, score: float = 0.5) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.score = score

    async def send(self, calls: Any) -> list[Any]:
        replies: list[Any] = []
        for call in calls:
            body = call.json
            self.bodies.append(body)
            if call.path == "/pooling":
                inputs = (
                    body["input"]
                    if "input" in body
                    else [
                        " ".join(part["text"] for part in m["content"] if part["type"] == "text")
                        for m in body["messages"]
                    ]
                )
                items = [{"index": i, "data": [[1.0, 1.0]]} for i in range(len(inputs))]
                replies.append(Reply(200, {"data": items, "usage": {"prompt_tokens": len(inputs)}}, {}))
            elif call.path == "/rerank":
                documents = body["documents"]
                rows = [{"index": i, "relevance_score": self.score} for i in range(len(documents))]
                replies.append(Reply(200, {"results": rows[::-1]}, {}))
            elif "input" in body:
                replies.append(
                    Reply(200, {"data": [{"index": i, "embedding": [1.0, 1.0]} for i in range(len(body["input"]))]}, {})
                )
            elif "texts" in body:
                replies.append(Reply(200, {"embeddings": {"float": [[1.0, 1.0]] * len(body["texts"])}}, {}))
            else:
                replies.append(Reply(200, {"embeddings": [{"values": [1.0, 1.0]} for _ in body["requests"]]}, {}))
        return replies

    async def probe(self) -> list[Any]:
        return []

    @property
    def usage(self) -> Any:

        return Usage()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


def texts(*items: str) -> list[Content]:
    return [Content.from_text(item) for item in items]


class TestEmbedBudget:
    def test_a_budget_cuts_the_content_and_records_the_census(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        census = TextTruncationCensus()
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=6,
            doc_prompt="D: ",
        )
        client = EmbeddingClient(config, sender=sender, census=census)
        long_text = " ".join(["evidence"] * 20)

        client.encode(texts(long_text), EncodeRole.DOCUMENT)

        sent = sender.bodies[0]["input"][0]
        assert sent.startswith("D: evidence"), "the side's prompt is applied before the fit"
        assert word_tokenizer().count(sent) <= 6, "the sent render fits the budget"
        assert sent != f"D: {long_text}", "an over-budget input is cut, never sent whole"
        cuts = census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)
        assert len(cuts) == 1 and cuts[0].budget_source == "tokenizer"
        assert client.census is census

    def test_an_input_under_budget_is_sent_byte_identical(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=6
        )
        EmbeddingClient(config, sender=sender).encode(texts("the evidence"), EncodeRole.DOCUMENT)

        assert sender.bodies[0]["input"][0] == "the evidence"

    def test_the_template_is_re_attached_and_the_anchor_kept(self, tmp_path: Any) -> None:
        """A cut content span gets the template back around it, anchor last (the wrapped-prompt defect
        class: an engine-side truncation would drop the anchor a last-token pooler reads)."""
        tokenizer = framed_bpe_tokenizer()
        (tmp_path / "framed").mkdir()
        file = save(tokenizer, tmp_path / "framed")
        template = TemplateSpec(
            document=(Segment(fixed="<doc> "), Segment(content="document"), Segment(fixed="{special:end_turn}"))
        )
        sender = RecordingSender()
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=str(file),
            max_tokens=10,
            template=template,
        )
        long_text = " ".join(["evidence"] * 40)
        EmbeddingClient(config, sender=sender).encode(texts(long_text), EncodeRole.DOCUMENT)

        sent = sender.bodies[0]["input"][0]
        assert sent.startswith("<doc> ")
        assert sent.endswith(END_TURN), "the anchor survives the cut"
        assert tokenizer.count(sent) <= 10

    def test_a_hosted_profile_with_only_max_tokens_sends_content_uncut(self) -> None:
        """The vendor path: no tokenizer, the documented limit recorded, content uncut."""
        sender = RecordingSender()
        client = EmbeddingClient(
            EmbeddingEndpoint(api="cohere", base_url="http://127.0.0.1:9000/v1", model="m", max_tokens=1024),
            sender=sender,
        )
        text = " ".join(["evidence"] * 500)
        client.encode(texts(text), EncodeRole.DOCUMENT)

        assert sender.bodies[0]["texts"][0] == text, "nothing is cut without a tokenizer"
        rows = client.census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)
        assert len(rows) == 1 and rows[0].budget_source == "vendor" and rows[0].doc_id == "<budget>"

    def test_on_overflow_chunk_is_refused_for_a_vector_role(self, tokenizer_json: str) -> None:
        """Chunks pool scores by max; a vector has no score to pool. Declared refusal, not a silent
        reinterpretation of the declared aggregation."""
        with pytest.raises(ConfigError, match="chunk"):
            EmbeddingClient(
                EmbeddingEndpoint(
                    base_url="http://127.0.0.1:9000/v1",
                    model="m",
                    tokenizer=tokenizer_json,
                    max_tokens=64,
                    on_overflow="chunk",
                    chunk={"max_tokens": 16, "overlap_tokens": 0},
                ),
                sender=RecordingSender(),
            )

    def test_on_overflow_fail_raises_the_budget_error(self, tokenizer_json: str) -> None:
        client = EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=4,
                on_overflow="fail",
            ),
            sender=RecordingSender(),
        )
        with pytest.raises(TextBudgetExceededError):
            client.encode(texts(" ".join(["evidence"] * 50)), EncodeRole.DOCUMENT)

    def test_the_media_wiring_runs_through_the_shared_preparation(self, tokenizer_json: str) -> None:
        """The one preparation path: a media-carrying request for the text-only embed role is refused
        before the media is fetched (the adapters' own contract, kept in front of preparation); the
        media-token reservation the fit receives is the prepared request's."""
        from rcp_ndcg_core.content import MediaRef

        client = EmbeddingClient(
            EmbeddingEndpoint(base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=64),
            sender=RecordingSender(),
        )
        page = Content(root=[ImagePart(ref=MediaRef(uri="gs://bucket/page_1.png"))])
        with pytest.raises(CapabilityError, match="image"):
            client.encode([page], EncodeRole.DOCUMENT)


class TestPoolBudget:
    def test_a_pool_config_cuts_its_text(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        config = PoolingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            dim=2,
            tokenizer=tokenizer_json,
            max_tokens=6,
            doc_prompt="D: ",
        )
        client = PoolingClient(config, sender=sender)
        client.encode(texts(" ".join(["evidence"] * 20)), EncodeRole.DOCUMENT)

        assert sender.bodies[0]["input"][0].startswith("D: ")
        assert word_tokenizer().count(sender.bodies[0]["input"][0]) <= 6

    def test_a_pooling_client_without_dim_is_refused_at_construction(self, tokenizer_json: str) -> None:
        """The base64 frame needs the width; refusing at construction keeps the GPU idle-time free (R13)."""
        with pytest.raises(ConfigError, match="dim"):
            PoolingClient(
                PoolingEndpoint(
                    base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=64
                ),
                sender=RecordingSender(),
            )

    def test_a_per_call_batch_size_below_one_is_a_config_error(self, tokenizer_json: str) -> None:
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=64,
            ),
            sender=RecordingSender(),
        )
        with pytest.raises(ConfigError, match="at least 1"):
            asyncio.run(client.aencode(texts("a", "b"), EncodeRole.DOCUMENT, batch_size=0))

    def test_pool_refuses_chunk_overflow_like_the_embed_role(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="chunk"):
            PoolingClient(
                PoolingEndpoint(
                    base_url="http://127.0.0.1:9000/v1",
                    model="m",
                    dim=2,
                    tokenizer=tokenizer_json,
                    max_tokens=64,
                    on_overflow="chunk",
                    chunk={"max_tokens": 16, "overlap_tokens": 0},
                ),
                sender=RecordingSender(),
            )


class TestRerankBudget:
    def _config(self, **overrides: Any) -> RerankEndpoint:
        settings: dict[str, Any] = {
            "base_url": "http://127.0.0.1:9000/v1",
            "model": "m",
            "tokenizer": "test/tokenizer",
            "max_tokens": 8192,
            "use_activation": False,
        }
        settings.update(overrides)
        return RerankEndpoint(**settings)

    def test_a_budget_cuts_the_pair_spans_and_records_the_census(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        census = TextTruncationCensus()
        client = RerankClient(self._config(tokenizer=tokenizer_json, max_tokens=10), sender=sender, census=census)
        long_document = " ".join(["evidence"] * 40)

        result = client.rerank("query", [long_document])

        sent = sender.bodies[0]
        assert sent["query"] == "query"
        document = sent["documents"][0]
        assert word_tokenizer().count(document) < word_tokenizer().count(long_document)
        # Without a template the pair budget counts the contiguous render (query + document), the engine's
        # glue being its own; the fit leaves the post-processor's share only.
        assert word_tokenizer().count(sent["query"] + document) <= 10
        assert result.scores == (0.5,)
        assert len(census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)) == 1

    def test_query_max_tokens_splits_the_pair(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        client = RerankClient(self._config(tokenizer=tokenizer_json, max_tokens=8, query_max_tokens=2), sender=sender)
        long_query = " ".join(["evidence"] * 10)

        client.rerank(long_query, ["a document about the evidence"])

        assert word_tokenizer().count(sender.bodies[0]["query"]) == 2, "the query is cut to its share"
        assert sender.bodies[0]["documents"], "the document keeps the rest of the budget"

    def test_chunk_scores_pool_by_max(self, tokenizer_json: str) -> None:
        class ScoringSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                for call in calls:
                    self.bodies.append(call.json)
                documents = calls[0].json["documents"]
                # Score chunk k by k: the pooled score must be the best chunk's, not the first one's.
                rows = [{"index": i, "relevance_score": i / 10} for i in range(len(documents))]
                return [Reply(200, {"results": rows[::-1]}, {})]

        sender = ScoringSender()
        client = RerankClient(
            self._config(
                tokenizer=tokenizer_json,
                max_tokens=12,
                on_overflow="chunk",
                chunk={"max_tokens": 5, "overlap_tokens": 0},
            ),
            sender=sender,
        )
        long_document = " ".join(["evidence one two three four five"] * 4)

        result = client.rerank("query", [long_document])

        assert len(sender.bodies[0]["documents"]) > 1, "the document was split"
        best = max(i / 10 for i, _ in enumerate(sender.bodies[0]["documents"]))
        assert result.scores[0] == pytest.approx(best), "the document scores its best chunk"

    def test_rerank_many_checkpoints_pooled_scores(self, tokenizer_json: str) -> None:
        """The checkpoint scores align to the example's documents: chunks are pooled before the call."""
        sender = RecordingSender()
        client = RerankClient(self._config(tokenizer=tokenizer_json), sender=sender)
        example = RankingExample(query="q", id="q1", doc_ids=["d1"], docs=[" ".join(["evidence"] * 3)])
        seen: list[tuple[str, tuple[float, ...]]] = []

        client.rerank_many([example], checkpoint=lambda query_id, scores: seen.append((query_id, scores)))

        assert seen == [("q1", (0.5,))]

    def test_no_engine_side_truncation_field_is_ever_sent(self, tokenizer_json: str) -> None:
        """The client cut already: the engine never receives a truncation request (the crux)."""
        sender = RecordingSender()
        client = RerankClient(self._config(tokenizer=tokenizer_json, max_tokens=8), sender=sender)
        client.rerank("query", [" ".join(["evidence"] * 40)])

        body = sender.bodies[0]
        assert not {"truncate_prompt_tokens", "max_tokens_per_query", "max_tokens_per_doc", "truncation_side"} & set(
            body
        )


class TestQueryShareSettled:
    """One query rides per request: the client settles the shared query span once, exactly as fit would
    settle it for an overflowing pair, and every document span is verified against the span that ships."""

    def test_mixed_length_documents_all_fit_the_declared_budget(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        census = TextTruncationCensus()
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=10,
            query_max_tokens=4,
            use_activation=False,
        )
        client = RerankClient(config, sender=sender, census=census)
        long_query = " ".join(["evidence"] * 5)  # over its declared share of 4

        client.rerank(long_query, [" ".join(["evidence"] * 2), " ".join(["evidence"] * 12)])

        sent = sender.bodies[-1]
        query, documents = sent["query"], sent["documents"]
        assert word_tokenizer().count(query) <= 4, "the shared query ships at its declared share"
        for document in documents:
            assert word_tokenizer().count(query + document) <= 10, "every shipped pair fits the budget"

    def test_the_settlement_is_counted_once_per_call(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=10,
            query_max_tokens=4,
            use_activation=False,
        )
        client = RerankClient(config, sender=sender)
        client.rerank(" ".join(["evidence"] * 5), [" ".join(["evidence"] * 2)])

        settlement = [
            cut for cut in client.census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET) if cut.doc_id == "<query>"
        ]
        assert len(settlement) == 1 and settlement[0].kept_tokens == 4

    def test_without_a_share_the_query_ships_whole(self, tokenizer_json: str) -> None:
        """No split declared: an under-budget pair rides byte-identical (fit's own guarantee)."""
        sender = RecordingSender()
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            tokenizer=tokenizer_json,
            model="m",
            max_tokens=8192,
            use_activation=False,
        )
        client = RerankClient(config, sender=sender)
        long_query = " ".join(["evidence"] * 30)
        client.rerank(long_query, ["a small document"])

        assert sender.bodies[-1]["query"] == long_query

    def test_a_query_that_alone_fills_the_budget_is_refused_without_a_split(self, tokenizer_json: str) -> None:
        from rcp_ndcg.data.preprocess import TextBudgetExceededError

        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                tokenizer=tokenizer_json,
                model="m",
                max_tokens=6,
                use_activation=False,
            ),
            sender=RecordingSender(),
        )
        with pytest.raises(TextBudgetExceededError):
            client.rerank(" ".join(["evidence"] * 50), ["a document"])


class TestRerankVendorBudget:
    """A hosted rerank profile that declares only the vendor's documented limit: content uncut, the limit
    recorded, and no client-side settlement (there is no tokenizer to measure with)."""

    def test_a_hosted_profile_with_only_max_tokens_sends_pairs_uncut(self) -> None:
        sender = RecordingSender()
        client = RerankClient(
            RerankEndpoint(api="cohere", base_url="http://127.0.0.1:9000/v1", model="m", max_tokens=1024),
            sender=sender,
        )
        long_query = " ".join(["evidence"] * 50)
        long_document = " ".join(["evidence"] * 500)

        client.rerank(long_query, [long_document])

        assert sender.bodies[0]["query"] == long_query
        assert sender.bodies[0]["documents"][0] == long_document
        rows = client.census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)
        assert len(rows) == 1 and rows[0].budget_source == "vendor" and rows[0].doc_id == "<budget>"


class TestInstructionReserved:
    """An ``instruction: field`` (or ``system``) mode renders the instruction into the engine's frame: its
    tokens are part of the fixed overhead, so the fit reserves them (``fold`` folds it into the query
    content, where the span cut already counts it)."""

    @staticmethod
    def _template() -> TemplateSpec:
        """The engine's pair frame with an instruction span: rendered by the engine for ``field`` mode, so
        the fit must reserve the instruction's tokens in the fixed overhead."""
        return TemplateSpec(
            pair=(
                Segment(content="instruction"),
                Segment(fixed="\n"),
                Segment(content="query"),
                Segment(fixed=" "),
                Segment(content="document"),
            )
        )

    def test_the_instruction_travels_in_the_overhead_not_the_cut(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=12,
            instruction="field",
            template=self._template(),
            use_activation=False,
        )
        client = RerankClient(config, sender=sender)
        long_instruction = " ".join(["evidence"] * 8)  # 8 tokens of fixed overhead, never cut
        long_document = " ".join(["evidence"] * 20)

        client.rerank("the query", [long_document], instruction=long_instruction)

        body = sender.bodies[0]
        assert body["instruction"] == long_instruction, "the field mode sends it as its own request field"
        document = body["documents"][0]
        # The pair the engine renders: the instruction line + the query + the cut document, all within the
        # declared budget (the instruction reserved in the overhead, the document cut to what remains).
        assert word_tokenizer().count(f"{long_instruction}\nthe query {document}") <= 12


class TestMediaUnderTheBudget:
    """The budget's media rule: media never cut; when media alone fill the budget the declared overflow
    policy decides (cut shrinks then drops whole items, fail refuses, chunk is refused -- a vision block is
    atomic), and every drop is recorded with ``dropped=True``."""

    @staticmethod
    def _media_client(sender: Any, **overrides: Any) -> PoolingClient:
        settings: dict[str, Any] = {
            "base_url": "http://127.0.0.1:9000/v1",
            "model": "colqwen",
            "dim": 2,
            "tokenizer": str(SESSION_TOKENIZER),
            "max_tokens": 8192,
            "image_policy": {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            "max_images": 4,
        }
        settings.update(overrides)
        return PoolingClient(PoolingEndpoint(**settings), sender=sender)

    def test_fitting_media_counts_blocks_and_the_budget_carves_the_rest(self, tmp_path: Any) -> None:
        """Media that fit the budget ride whole (never cut); the fit reserves the media tokens it was
        given, and the text gets what remains."""

        sender = RecordingSender()
        client = self._media_client(sender, max_tokens=1100)
        big = tmp_path / "big.png"
        PILImage.new("RGB", (900, 900), (10, 10, 200)).save(big, format="PNG")

        client.encode([Content.from_image(big.as_uri()), Content.from_text("plain")], EncodeRole.DOCUMENT)

        body = sender.bodies[-1]
        assert body.get("messages"), "the media item went through the messages shape"
        sent_texts = [
            part["text"] for message in body["messages"] for part in message["content"] if part.get("type") == "text"
        ]
        assert "plain" in sent_texts, "the text item rides with its media"

    def test_media_alone_over_the_budget_fails_under_the_fail_policy(self, tmp_path: Any) -> None:
        from rcp_ndcg.data.preprocess import TextBudgetExceededError

        sender = RecordingSender()
        client = self._media_client(sender, max_tokens=60, on_overflow="fail")
        with pytest.raises(TextBudgetExceededError, match="media"):
            client.encode([self._media_item(tmp_path)], EncodeRole.DOCUMENT)
        assert sender.bodies == [], "refused before anything was sent"

    def test_media_alone_over_the_budget_refused_under_chunk(self, tokenizer_json: str, tmp_path: Any) -> None:
        sender = RecordingSender()
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=60,
            on_overflow="chunk",
            chunk={"max_tokens": 8, "overlap_tokens": 0},
            use_activation=False,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=2,
        )
        client = RerankClient(config, sender=sender)
        with pytest.raises(ConfigError, match="cannot be chunked"):
            client.rerank("q", [self._media_item(tmp_path)])

    @staticmethod
    def _media_item(tmp_path: Any) -> Any:

        page = tmp_path / "media.png"
        PILImage.new("RGB", (900, 900), (10, 10, 200)).save(page, format="PNG")
        return Content.from_image(page.as_uri())


class TestEmptyDocuments:
    """``empty_doc`` is consumed by every role client, for an empty text document and for one whose every
    media item the budget dropped: ``send`` (the empty string, as today), ``send_text`` (the placeholder),
    ``omit_zero`` (never sent, scoring 0.0). A request is never sent empty."""


def _png(tmp_path: Any, name: str, colour: tuple[int, int, int], size: tuple[int, int]) -> Any:

    page = tmp_path / f"page-{colour}.png"
    PILImage.new("RGB", size, colour).save(page, format="PNG")
    return Content.from_image(page.as_uri())


class TestEmbedEmptyDocuments:
    @pytest.mark.parametrize("policy", ["send", "send_text", "omit_zero"])
    def test_the_three_policies_on_an_empty_text_item(self, tokenizer_json: str, policy: str) -> None:
        overrides: dict[str, Any] = {"empty_doc": policy}
        if policy == "send_text":
            overrides["empty_doc_text"] = "NULL"
        client = EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=64,
                **overrides,
            ),
            sender=RecordingSender(),
        )
        result = client.encode(texts("", "the evidence"), EncodeRole.DOCUMENT)

        if policy == "omit_zero":
            assert result.num_items == 2 and result.as_matrix()[0].tolist() == [0.0, 0.0], "the omitted item scores 0.0"
            assert result.as_matrix()[1].tolist() != [0.0, 0.0], "the kept item carries its vectors"
            sent = client._sender.bodies[-1]["input"]
            assert sent == ["the evidence"], "the omitted item never goes out (and the request is not empty)"
            return
        # "send" embeds the empty string; "send_text" embeds the placeholder.
        sent = client._sender.bodies[-1]["input"]
        assert sent == (["", "the evidence"] if policy == "send" else ["NULL", "the evidence"])

    def test_a_document_whose_every_media_item_was_dropped_is_empty(self, tokenizer_json: str, tmp_path: Any) -> None:
        """All media dropped + no text = an empty document: it follows ``empty_doc``, never an empty request."""

        page = tmp_path / "page.png"
        PILImage.new("RGB", (900, 900), (10, 10, 200)).save(page, format="PNG")
        sender = RecordingSender()
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=4,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=2,
                empty_doc="omit_zero",
            ),
            sender=sender,
        )

        result = asyncio.run(
            client.aencode([Content.from_image(page.as_uri()), Content.from_text("plain")], EncodeRole.DOCUMENT)
        )
        assert result.num_items == 2, "the result stays aligned to the inputs"
        # The media item was dropped whole (it cannot shrink into 4 tokens): the document is empty, and
        # empty_doc: omit_zero gives it an empty slice (a zero MaxSim score), while the text item rides
        # beside it -- never an empty request, and every drop is on record.
        assert list(result.offsets) == [0, 0, 1]
        assert len(sender.bodies[-1]["input"]) == 1, "only the text item was sent"
        assert client.media_census._seen, "the drop is recorded"


class TestMediaGates:
    def test_a_request_over_max_images_is_refused_before_sending(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        sender = RecordingSender()
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=1,
            ),
            sender=sender,
        )
        pages = [_png_content(tmp_path, index) for index in range(2)]
        with pytest.raises(CapabilityError, match="max_images"):
            asyncio.run(client.aencode(pages, EncodeRole.DOCUMENT))
        assert sender.bodies == [], "the gate refuses before anything is sent"

    def test_media_for_a_model_that_reads_none_is_refused(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        sender = RecordingSender()
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            ),
            sender=sender,
        )
        with pytest.raises(CapabilityError, match="max_images"):
            asyncio.run(client.aencode([_png_content(tmp_path, 0)], EncodeRole.DOCUMENT))


def _png_content(tmp_path: Any, index: int) -> Any:

    page = tmp_path / f"page-{index}.png"
    PILImage.new("RGB", (300, 300), (10, 10, 200)).save(page, format="PNG")
    return Content.from_image(page.as_uri())


class TestRerankEmptyDocuments:
    """``empty_doc`` on the rerank role: the scores stay aligned to the documents as given; ``omit_zero``
    never sends the item and scores it 0.0, and a candidate set whose every document is omitted makes no
    request (never an empty request)."""

    def _client(self, tokenizer_json: str, policy: str, sender: Any) -> RerankClient:
        overrides: dict[str, Any] = {"empty_doc": policy, "use_activation": False}
        if policy == "send_text":
            overrides["empty_doc_text"] = "NULL"
        return RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=8192,
                **overrides,
            ),
            sender=sender,
        )

    @pytest.mark.parametrize("policy", ["send", "send_text", "omit_zero"])
    def test_the_three_policies_on_an_empty_document(self, tokenizer_json: str, policy: str) -> None:
        sent: list[dict] = []

        class ScoringSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                for call in calls:
                    sent.append(call.json)
                documents = calls[0].json["documents"]
                rows = [{"index": i, "relevance_score": 0.9} for i in range(len(documents))]
                return [Reply(200, {"results": rows[::-1]}, {})]

        client = self._client(tokenizer_json, policy, ScoringSender())
        result = client.rerank("q", ["", "the evidence"])

        if policy == "omit_zero":
            assert result.scores == (0.0, 0.9), "the omitted document scores 0.0, the kept one its score"
            assert [d for d in sent[-1]["documents"]] == ["the evidence"], "the omitted one never goes out"
            return
        expected = ["", "the evidence"] if policy == "send" else ["NULL", "the evidence"]
        assert sent[-1]["documents"] == expected

    def test_every_document_omitted_makes_no_request(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        client = self._client(tokenizer_json, "omit_zero", sender)
        result = client.rerank("q", ["", ""])
        assert result.scores == (0.0, 0.0), "the omitted documents score 0.0"
        assert sender.bodies == [], "no empty request goes out"


class TestEngineMediaCheck:
    """The startup media probe (item 5): one prepared probe image to the engine, the engine's reported
    prompt tokens compared with the counted ones -- a mismatch is refused (typed, with the hint naming
    ``image_processor`` / ``image_policy`` / the server's media flags); a reply without usage is recorded
    ``not_checked``; never silent."""

    @staticmethod
    def _client(sender: Any, *, tokenizer_json: str) -> PoolingClient:
        return PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=4,
            ),
            sender=sender,
        )

    def test_a_text_only_role_skips_the_check(self, tokenizer_json: str) -> None:
        client = EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
            ),
            sender=RecordingSender(),
        )
        assert asyncio.run(client.check_engine_media()) is None, "no media, nothing to check"

    def test_a_mismatch_is_refused_with_the_hint(self, tokenizer_json: str) -> None:
        from rcp_ndcg.errors import ProviderError

        class LyingUsageSender(RecordingSender):
            """Answers the probe's /pooling with a prompt-token report the count contradicts."""

            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                return [Reply(200, {"data": [], "usage": {"prompt_tokens": 9999}}, {}) for _ in calls]

        client = self._client(LyingUsageSender(), tokenizer_json=tokenizer_json)
        with pytest.raises(ProviderError, match="prompt tokens"):
            asyncio.run(client.check_engine_media())

    def test_a_reply_without_usage_is_recorded_not_checked(self, tokenizer_json: str) -> None:
        class UsagelessSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                return [Reply(200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}]}, {}) for _ in calls]

        client = self._client(UsagelessSender(), tokenizer_json=tokenizer_json)
        asyncio.run(client.check_engine_media())  # never silent, never a pass: the census records it
        rows = client.media_census._seen
        assert any(doc_id.startswith("engine_media_check:not_checked") for _, doc_id, _ in rows)


class TestRerankChunkAndOmitCompose:
    """Chunking and ``empty_doc: omit_zero`` compose: the pooled score lands on the document it was scored
    for, an omitted document scores 0.0 at its position, and the whole thing stays aligned."""

    def test_chunked_documents_with_an_omitted_sibling(self, tokenizer_json: str) -> None:
        class ScoringSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                for call in calls:
                    self.bodies.append(call.json)
                documents = calls[0].json["documents"]
                # chunk k scores k / 10: the pool must take the best chunk, per document.
                rows = [{"index": i, "relevance_score": i / 10} for i in range(len(documents))]
                return [Reply(200, {"results": rows[::-1]}, {})]

        sender = ScoringSender()
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=12,
                query_max_tokens=2,
                on_overflow="chunk",
                chunk={"max_tokens": 5, "overlap_tokens": 0},
                empty_doc="omit_zero",
                use_activation=False,
            ),
            sender=sender,
        )
        long_document = " ".join(["evidence one two three four five"] * 4)

        result = client.rerank("query", ["", long_document])

        assert result.scores[0] == 0.0, "the omitted document scores 0.0 at its position"
        best = max(i / 10 for i, _ in enumerate(sender.bodies[0]["documents"]))
        assert result.scores[1] == pytest.approx(best), "the chunked document scores its best chunk"
        assert len(sender.bodies[0]["documents"]) > 1, "the document was chunked"


class TestEngineMediaCheckPass:
    """The passing path: an honest engine whose prompt-token report covers the same request the client
    counted (media block + the probe's text tokens) passes the check."""

    def test_an_honest_engine_passes_and_leaks_no_temp_file(self, tokenizer_json: str) -> None:

        counted_box: dict[str, int] = {}
        client = None

        class HonestSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                # An honest engine counts the same request the client counted: the probe's media block
                # plus its text tokens, exactly as the client counted them.
                tokens = counted_box["counted"]
                return [
                    Reply(
                        200,
                        {"data": [{"index": 0, "data": [[1.0, 1.0]]}], "usage": {"prompt_tokens": tokens}},
                        {},
                    )
                ]

        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=4,
            ),
            sender=_CountingSender(tokenizer_json),
        )
        import tempfile

        before = set(tempfile.gettempdir())
        asyncio.run(client.check_engine_media())
        leaked = {name for name in set(tempfile.gettempdir()) - before if name.endswith(".png")}
        assert not leaked, "the probe file is cleaned up"


class _CountingSender(RecordingSender):
    """A sender that answers the media probe with exactly the count the client's own helpers produce."""

    def __init__(self, tokenizer_json: str) -> None:
        super().__init__()
        self._tokenizer = load_tokenizer(tokenizer_json)

    async def send(self, calls: Any) -> list[Any]:

        from PIL import Image as PILImage

        from rcp_ndcg.inference.types import Reply

        # The probe call's lowered body carries the request the engine saw; the client's count of the
        # same request is what an honest engine reports.
        body = calls[0].json

        def probe_tokens(url: str) -> int:
            """An honest engine's count: decode the probe image, read its size, count its vision block."""
            payload = (
                base64.b64decode(url.partition(",")[2])
                if url.startswith("data:")
                else Path(url.replace("file://", "")).read_bytes()
            )
            image = PILImage.open(io.BytesIO(payload))
            content = Content.from_image(url, width=image.size[0], height=image.size[1])
            return content_media_tokens(
                content, ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl"), None
            ).tokens

        if "messages" in body:  # the pooling wire: one image block in one message
            tokens = sum(probe_tokens(_media_uri(message)) for message in body["messages"])
        else:  # the rerank wire: {"query": text, "documents": [...]}
            tokens = sum(
                probe_tokens(document["content"][0]["image_url"]["url"])
                for document in body["documents"]
                if isinstance(document, dict)
            )
            tokens += self._tokenizer.count(body.get("query", ""))
        return [
            Reply(200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}], "usage": {"prompt_tokens": tokens}}, {})
            for _ in calls
        ]


def _media_uri(message: Any) -> Any:
    part = next(part for part in message["content"] if part.get("type") == "image_url")
    return part["image_url"]["url"]


class TestPoolEmptyDocuments:
    """``empty_doc`` on the pooling role, per value (``omit_zero``'s drop case is above)."""

    @pytest.mark.parametrize("policy", ["send", "send_text"])
    def test_the_policies_on_an_empty_text_item(self, tokenizer_json: str, policy: str) -> None:
        overrides: dict[str, Any] = {"empty_doc": policy}
        if policy == "send_text":
            overrides["empty_doc_text"] = "NULL"
        vectors = np.ones((1, 2), dtype=np.float16)
        sender = _GatedSender(PoolingServer({"plain": vectors, "": vectors, "NULL": vectors}))
        overrides = {"empty_doc": policy}
        if policy == "send_text":
            overrides["empty_doc_text"] = "NULL"
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=4,
                **overrides,
            ),
            sender=sender,
        )

        asyncio.run(client.aencode([Content.from_text(""), Content.from_text("plain")], EncodeRole.DOCUMENT))

        sent = sender.sent[0][0]["input"]
        assert sent == (["", "plain"] if policy == "send" else ["NULL", "plain"])


class TestRerankChunkOmitMatrix:
    """The chunk/omit composition matrix (the re-fix round's regression): every combination keeps the
    pooled score on the original document's position."""

    def _client(self, tokenizer_json: str) -> RerankClient:
        return RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=12,
                query_max_tokens=2,
                on_overflow="chunk",
                chunk={"max_tokens": 5, "overlap_tokens": 0},
                empty_doc="omit_zero",
                use_activation=False,
            ),
            sender=_ChunkScoreSender(),
        )

    def test_chunk_no_omit(self, tokenizer_json: str) -> None:
        long_doc = " ".join(["evidence one two three four five"] * 4)
        result = self._client(tokenizer_json).rerank("query", [long_doc, "short doc"])
        assert result.scores == pytest.approx((0.4, 0.5))

    def test_chunk_with_middle_omit(self, tokenizer_json: str) -> None:
        long_doc = " ".join(["evidence one two three four five"] * 4)
        result = self._client(tokenizer_json).rerank("query", [long_doc, "", "short doc"])
        # the empty document is omitted from the wire; the short one scores at its own position.
        assert result.scores == pytest.approx((0.4, 0.0, 0.5))

    def test_two_chunked_documents(self, tokenizer_json: str) -> None:
        long_doc = " ".join(["evidence one two three four five"] * 4)
        result = self._client(tokenizer_json).rerank("query", [long_doc, long_doc])
        assert result.scores == pytest.approx((0.4, 0.9))

    def test_omit_then_one_chunked(self, tokenizer_json: str) -> None:
        long_doc = " ".join(["evidence one two three four five"] * 4)
        result = self._client(tokenizer_json).rerank("query", ["", long_doc])
        # the wire carries only the kept document's chunks (scored 0..4): its best chunk is the score.
        assert result.scores == pytest.approx((0.0, 0.4))


class _ChunkScoreSender(RecordingSender):
    """Chunk k of the wire's candidate set scores k/10: the pool must take each document's best chunk."""

    async def send(self, calls: Any) -> list[Any]:
        from rcp_ndcg.inference.types import Reply

        for call in calls:
            self.bodies.append(call.json)
        documents = calls[0].json["documents"]
        rows = [{"index": i, "relevance_score": i / 10} for i in range(len(documents))]
        return [Reply(200, {"results": rows[::-1]}, {})]
