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
from rcp_ndcg_core.content import Content, ImagePart, TextPart

from rcp_ndcg.data.prepare import apply_media_fit
from rcp_ndcg.data.preprocess import TextBudgetExceededError, TextTruncationCensus
from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens
from rcp_ndcg.data.templates import Segment, TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import CapabilityError, ConfigError
from rcp_ndcg.inference import EmbeddingClient, PoolingClient, RerankClient
from rcp_ndcg.inference.clients.rerank import QUERY_DOC_ID
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
    def test_the_client_exposes_the_text_budget_it_fits_to(self, tokenizer_json: str) -> None:
        """Harnesses and case loaders read the budget the client built, never rebuild it from the config."""
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=6,
            query_max_tokens=4,
        )
        budget = EmbeddingClient(config, sender=RecordingSender()).text_budget
        assert budget is not None
        assert (budget.max_tokens, budget.query_max_tokens, budget.tokenizer) == (6, 4, tokenizer_json)
        unbudgeted = EmbeddingEndpoint(base_url="http://127.0.0.1:9000/v1", model="m", api="cohere")
        assert EmbeddingClient(unbudgeted, sender=RecordingSender()).text_budget is None

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

    def test_query_max_tokens_above_max_tokens_is_refused(self) -> None:
        """On the embedding roles ``max_tokens`` is the document shape's budget and the model's whole input
        budget; a query budget above it cannot fit the served context."""
        with pytest.raises(ConfigError, match="whole input budget") as caught:
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer="test/word-level",
                max_tokens=1024,
                query_max_tokens=1025,
            )
        assert "query_max_tokens" in (caught.value.hint or ""), "the refusal names the field to change"

    def test_a_prompt_prefix_beside_a_template_is_refused(self) -> None:
        """One home for a prompt prefix (rec-qwen3-embedding-0.6b): the client prepends query_prompt before
        the template renders, so declaring both doubles the prefix. Refused, naming the template segment to
        use instead; the fields stay for template-less configs (hosted profiles)."""
        template = TemplateSpec(query=(Segment(fixed="Instruct: task\\nQuery:"), Segment(content="query")))
        with pytest.raises(ConfigError, match="doubles the prefix") as caught:
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer="test/word-level",
                max_tokens=64,
                template=template,
                query_prompt="Instruct: task\\nQuery:",
            )
        assert "fixed segment" in caught.value.hint and "query" in caught.value.hint
        with pytest.raises(ConfigError, match="doubles the prefix") as caught:
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer="test/word-level",
                max_tokens=64,
                template=TemplateSpec(document=(Segment(fixed="D: "), Segment(content="document"))),
                doc_prompt="D: ",
            )
        assert "fixed segment" in caught.value.hint and "document" in caught.value.hint
        # Template-less configs (hosted profiles) keep the fields.
        EmbeddingEndpoint(
            api="cohere", base_url="http://127.0.0.1:9000/v1", model="m", max_tokens=64, query_prompt="Q: "
        )

    def test_the_query_shape_budget_caps_the_query_not_the_document(self, tokenizer_json: str) -> None:
        """The per-shape budget (the topk hand-off: query 1024, document 8192): ``query_max_tokens`` caps
        the query shape whole; ``max_tokens`` keeps capping the document shape; the census rows name the
        shape's budget."""
        sender = RecordingSender()
        census = TextTruncationCensus()
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8,
            query_max_tokens=3,
        )
        client = EmbeddingClient(config, sender=sender, census=census)
        long_text = " ".join(["evidence"] * 30)
        client.encode(texts(long_text), EncodeRole.QUERY)
        query_sent = sender.bodies[-1]["input"][0]
        assert word_tokenizer().count(query_sent) <= 3
        client.encode(texts(long_text), EncodeRole.DOCUMENT)
        document_sent = sender.bodies[-1]["input"][0]
        assert 3 < word_tokenizer().count(document_sent) <= 8
        rows = [cut.as_row() for cut in census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)]
        assert [(row["shape"], row["budget_tokens"]) for row in rows] == [("query", 3), ("document", 8)]

    def test_query_max_tokens_equal_to_max_tokens_is_a_legal_per_shape_budget(self, tokenizer_json: str) -> None:
        """The == boundary: both shapes capped the same is a legal per-shape budget -- the client constructs
        (the budget layer refuses only a share ABOVE the whole budget) and the query shape honours it."""
        sender = RecordingSender()
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=6,
            query_max_tokens=6,
        )
        client = EmbeddingClient(config, sender=sender)
        client.encode(texts(" ".join(["evidence"] * 30)), EncodeRole.QUERY)
        assert word_tokenizer().count(sender.bodies[-1]["input"][0]) <= 6


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

    def test_the_pool_config_refuses_a_declared_messages_shape(self, tokenizer_json: str) -> None:
        """The pooling wire lowers chat parts for its media items itself; a config-declared messages shape
        would leave every text batch on the rendered-string route while the identity declared the chat
        form -- refused, never silently inert."""
        with pytest.raises(ConfigError, match="messages"):
            PoolingClient(
                PoolingEndpoint(
                    base_url="http://127.0.0.1:9000/v1",
                    model="m",
                    dim=2,
                    tokenizer=tokenizer_json,
                    max_tokens=64,
                    request_shape="messages",
                ),
                sender=RecordingSender(),
            )

    def test_the_pooling_config_honours_the_per_shape_budget_too(self, tokenizer_json: str) -> None:
        """``query_max_tokens`` on a :class:`PoolingEndpoint` caps the query shape (its whole budget there)."""
        sender = RecordingSender()
        census = TextTruncationCensus()
        config = PoolingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=64,
            query_max_tokens=4,
            dim=2,
        )
        client = PoolingClient(config, sender=sender, census=census)
        client.encode(texts(" ".join(["evidence"] * 30)), EncodeRole.QUERY)
        assert word_tokenizer().count(sender.bodies[0]["input"][0]) <= 4
        client.encode(texts(" ".join(["evidence"] * 30)), EncodeRole.DOCUMENT)
        assert 4 < word_tokenizer().count(sender.bodies[-1]["input"][0]) <= 64


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

    def test_request_shape_other_than_text_is_refused(self, tokenizer_json: str) -> None:
        """The rerank wires implement text only; a declared messages/token_ids shape would be silently inert
        (the field is CONTENT, so two configs would hash differently and behave identically) -- refused,
        naming where the other routes do land."""
        with pytest.raises(ConfigError, match="request_shape"):
            self._config(tokenizer=tokenizer_json, request_shape="token_ids")
        with pytest.raises(ConfigError, match="rerank wires") as caught:
            self._config(tokenizer=tokenizer_json, request_shape="messages")
        assert "drop request_shape" in (caught.value.hint or ""), "the refusal names the field to change"

    def test_the_census_names_the_documents_original_positions(self, tokenizer_json: str) -> None:
        """With ``empty_doc: omit_zero``, a later document's census cut names ITS position -- never the kept
        position an earlier omission displaced."""
        census = TextTruncationCensus()
        client = RerankClient(
            self._config(tokenizer=tokenizer_json, max_tokens=10, empty_doc="omit_zero"),
            sender=RecordingSender(),
            census=census,
        )
        long_document = " ".join(["evidence"] * 40)

        client.rerank("query", ["", long_document])

        rows = [cut.doc_id for cut in census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)]
        assert rows, "the over-cap document records its cut"
        assert all(doc_id.isdigit() for doc_id in rows), rows
        assert 1 in [int(doc_id) for doc_id in rows], f"the over-cap document is named at its original position: {rows}"
        assert "0" not in rows, "the omitted document made no request and records no cut"

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

    def test_a_share_settlement_in_an_under_budget_pair_names_its_cause(self, tokenizer_json: str) -> None:
        """The shared query settles at its share although every pair fits the budget whole: the client
        changed what it sends, and the settlement row says so -- ``cause: query_share`` and the uncut probe
        request's size (the frame with the uncut query, an empty document), which is under the budget. No
        pair row is recorded: the documents ship whole."""
        from rcp_ndcg.data.preprocess import rendered_pair_tokens

        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=40,
            query_max_tokens=4,
            use_activation=False,
        )
        client = RerankClient(config, sender=RecordingSender())
        query = " ".join(["evidence"] * 6)
        client.rerank(query, ["a b c"])

        (settlement,) = client.census.cuts()
        assert settlement.doc_id == QUERY_DOC_ID
        assert settlement.cause == "query_share"
        assert client.text_budget is not None
        uncut = rendered_pair_tokens(client.text_budget, word_tokenizer(), query=query, document="")
        assert settlement.original_request_tokens == uncut < 40

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
        assert client.media_census.recorded(), "the drop is recorded"


class TestPerSideMedia:
    """Per-side media (2b, G3): ``media_sides`` names which sides of the retrieval pair may carry media
    (the topk reference rejects image queries); a client refuses media on a side that may not, with a
    typed error naming the field."""

    @staticmethod
    def _pool_client(tokenizer_json: str, **overrides: Any) -> PoolingClient:
        settings: dict[str, Any] = {
            "base_url": "http://127.0.0.1:9000/v1",
            "model": "colqwen",
            "dim": 2,
            "tokenizer": tokenizer_json,
            "max_tokens": 8192,
            "image_policy": {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            "max_images": 2,
        }
        settings.update(overrides)
        return PoolingClient(PoolingEndpoint(**settings), sender=RecordingSender())

    def test_media_on_a_forbidden_side_is_refused_naming_the_field(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        client = self._pool_client(tokenizer_json, media_sides=["document"])
        with pytest.raises(CapabilityError, match="media_sides"):
            asyncio.run(client.aencode([_png_content(tmp_path, 0)], EncodeRole.QUERY))
        # The allowed side goes out whole.
        result = asyncio.run(client.aencode([_png_content(tmp_path, 0)], EncodeRole.DOCUMENT))
        assert result.num_items == 1

    def test_media_on_the_rerank_query_is_refused_naming_the_field(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            use_activation=False,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=2,
            media_sides=["document"],
        )
        client = RerankClient(config, sender=RecordingSender())
        with pytest.raises(CapabilityError, match="media_sides"):
            client.rerank(_png_content(tmp_path, 0), ["the document"])
        # The same config, media on the document side: refused by nothing (the fake answers it).
        scores = client.rerank("the query", [_png_content(tmp_path, 1)])
        assert len(scores.scores) == 1

    def test_media_sides_default_to_both_sides(self, tokenizer_json: str) -> None:
        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            use_activation=False,
        )
        assert config.media_sides == ("query", "document")

    def test_an_explicitly_empty_media_sides_allows_no_side(self, tokenizer_json: str, tmp_path: Any) -> None:
        """The empty field is a real declaration (a config without media fields may carry it): media on ANY
        side is refused naming the field, not silently widened back to both sides."""
        from rcp_ndcg.errors import CapabilityError

        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            use_activation=False,
            media_sides=[],
        )
        client = RerankClient(config, sender=RecordingSender())
        assert client.media_sides() == frozenset()
        with pytest.raises(CapabilityError, match="media_sides"):
            client.rerank(_png_content(tmp_path, 0), ["the document"])
        with pytest.raises(CapabilityError, match="no side"):
            client.rerank("the query", [_png_content(tmp_path, 1)])

    def test_declaring_media_fields_with_no_allowed_side_is_refused(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="media_sides") as caught:
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=8192,
                use_activation=False,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                media_sides=[],
            )
        assert "media_sides" in (caught.value.hint or ""), "the refusal names the field to change"


class TestEmptyQuery:
    """``empty_query`` (2f, qwen3-vl-reranker): an empty QUERY has a policy, like ``empty_doc`` --
    ``refuse`` (the default) raises a typed error naming the query id, ``send`` sends the empty string."""

    @staticmethod
    def _client(tokenizer_json: str, *, empty_query: str) -> RerankClient:
        return RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=8192,
                use_activation=False,
                empty_query=empty_query,  # type: ignore[arg-type]
            ),
            sender=RecordingSender(),
        )

    def test_the_default_refuses_an_empty_query_naming_the_id(self, tokenizer_json: str) -> None:
        from rcp_ndcg.errors import DataError

        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=8192,
                use_activation=False,
            ),
            sender=RecordingSender(),
        )
        with pytest.raises(DataError, match="q42"):
            asyncio.run(client.arerank("", ["the document"], query_id="q42"))
        # The default is the field's value, and the refusal names the field.
        assert client.config.empty_query == "refuse"
        with pytest.raises(DataError, match="empty_query"):
            asyncio.run(client.arerank("", ["the document"]))
        # The synchronous path names the id too, through its own query_id argument.
        with pytest.raises(DataError, match="q42"):
            client.rerank("", ["the document"], query_id="q42")

    def test_send_keeps_todays_empty_string(self, tokenizer_json: str) -> None:
        client = self._client(tokenizer_json, empty_query="send")
        result = client.rerank("", ["the document"])
        assert len(result.scores) == 1

    def test_a_non_empty_query_is_never_refused(self, tokenizer_json: str) -> None:
        client = self._client(tokenizer_json, empty_query="refuse")
        result = client.rerank("the query", ["the document"])
        assert len(result.scores) == 1


class TestMediaGates:
    """``max_images``/``max_videos`` gate per wire request (as the judge's per-request gate): the pooling
    wire sends one media item per call, so per item; two single-image items with ``max_images: 1`` are
    two compliant calls. Media the model does not read is still refused, before anything is sent."""

    @staticmethod
    def _pool_client(tokenizer_json: str, *, max_images: int) -> PoolingClient:
        return PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=max_images,
            ),
            sender=RecordingSender(),
        )

    def test_per_item_gate_allows_one_image_per_wire_call(self, tokenizer_json: str, tmp_path: Any) -> None:
        """The pooling wire sends one media item per call: two single-image items are two compliant
        requests under ``max_images: 1``."""
        client = self._pool_client(tokenizer_json, max_images=1)
        result = asyncio.run(
            client.aencode([_png_content(tmp_path, 0), _png_content(tmp_path, 1)], EncodeRole.DOCUMENT)
        )
        assert result.num_items == 2

    def test_one_request_over_the_limit_is_refused(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        client = self._pool_client(tokenizer_json, max_images=1)
        page = tmp_path / "two.png"
        from PIL import Image as PILImage

        PILImage.new("RGB", (300, 300), (10, 10, 200)).save(page, format="PNG")
        content = Content.from_parts(
            [
                __import__("rcp_ndcg_core.content", fromlist=["ImagePart"]).ImagePart(
                    ref=__import__("rcp_ndcg_core.content", fromlist=["MediaRef"]).MediaRef(
                        uri=page.as_uri(), mime="image/png"
                    )
                ),
                __import__("rcp_ndcg_core.content", fromlist=["ImagePart"]).ImagePart(
                    ref=__import__("rcp_ndcg_core.content", fromlist=["MediaRef"]).MediaRef(
                        uri=page.as_uri(), mime="image/png"
                    )
                ),
            ]
        )
        with pytest.raises(CapabilityError, match="accepts 1") as caught:
            asyncio.run(client.aencode([content], EncodeRole.DOCUMENT))
        assert "max_images" in (caught.value.hint or ""), "the refusal's hint names the knob"

    def test_media_for_a_model_that_reads_none_is_refused(self, tokenizer_json: str, tmp_path: Any) -> None:
        from rcp_ndcg.errors import CapabilityError

        client = self._pool_client(tokenizer_json, max_images=0)
        with pytest.raises(CapabilityError, match="max_images") as caught:
            asyncio.run(client.aencode([_png_content(tmp_path, 0)], EncodeRole.DOCUMENT))
        assert "declare max_images" in (caught.value.hint or ""), "the refusal's hint names the knob"


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
        rows = client.media_census.recorded()
        assert any(doc_id.startswith("engine_media_check:not_checked") for _corpus, doc_id, _uri, _dropped in rows)


class TestRerankChunkAndOmitCompose:
    """Chunking and ``empty_doc: omit_zero`` compose: the pooled score lands on the document it was scored
    for, an omitted document scores 0.0 at its position, and the whole thing stays aligned."""

    def test_chunked_census_rows_name_their_original_document(self, tokenizer_json: str) -> None:
        """A chunked document's census rows carry its original position (``<position>#<chunk>``)."""
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
            sender=RecordingSender(),
            census=TextTruncationCensus(),
        )
        long_document = " ".join(["evidence one two three four five"] * 4)

        client.rerank("query", ["", long_document])

        rows = [cut.doc_id for cut in client.census.cuts(mechanism=TextTruncationCensus.TEXT_BUDGET)]
        assert any(doc_id.startswith("1#") for doc_id in rows), f"chunk rows name their original document: {rows}"
        assert all(doc_id.startswith("1#") for doc_id in rows if "#" in doc_id), rows

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
    """The passing path: an honest engine whose media DELTA (its report with the image minus its report
    without it) matches the counted media tokens. The counts are hand-verified from the request, not taken
    from the client's own helpers: a 224x224 image under the qwen2_vl policy is 8x8 = 64 patches plus the
    two vision wrapper markers = 66 tokens, whatever the client counts."""

    TEMPLATE_TOKENS = 137  # a served chat template's cost, the same on both probe requests

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

    @staticmethod
    def _honest_pool_sender(media_tokens: int = 66, template_tokens: int = 137) -> Any:
        """A served pooling engine: both probe requests ride the chat template (``template_tokens``), the
        with-image one adds the media block (``media_tokens``) -- hand-verified, not the client's arithmetic."""

        class HonestPoolingSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                replies = []
                for call in calls:
                    message = call.json.get("messages")
                    carries_image = any(
                        part.get("type") == "image_url"
                        for item in message or []
                        for part in item.get("content", [])
                        if isinstance(part, dict)
                    )
                    tokens = (media_tokens if carries_image else 0) + template_tokens
                    replies.append(
                        Reply(
                            200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}], "usage": {"prompt_tokens": tokens}}, {}
                        )
                    )
                return replies

        return HonestPoolingSender()

    def test_a_served_chat_template_does_not_fail_the_check(self, tokenizer_json: str) -> None:
        """An engine that renders a chat template around every probe request: its two reports differ by the
        media block alone, the template cancels in the delta, and the check passes -- a template beyond the
        client's counting never fails a correct engine."""
        client = self._client(self._honest_pool_sender(), tokenizer_json=tokenizer_json)
        asyncio.run(client.check_engine_media())  # the delta (66) matches the counted media tokens

    def test_an_honest_engine_passes_and_leaks_no_temp_file(self, tokenizer_json: str) -> None:
        import tempfile

        client = self._client(self._honest_pool_sender(), tokenizer_json=tokenizer_json)
        before = set(tempfile.gettempdir())
        asyncio.run(client.check_engine_media())
        leaked = {name for name in set(tempfile.gettempdir()) - before if name.endswith(".png")}
        assert not leaked, "the probe file is cleaned up"

    def test_the_rerank_probes_the_same_delta(self, tokenizer_json: str) -> None:
        """The rerank wire's baseline is its own body without the media: the honest engine's delta matches
        the counted media tokens and the check passes on a served rerank engine with a chat template."""

        class HonestRerankSender(RecordingSender):
            async def send(self, calls: Any) -> list[Any]:
                from rcp_ndcg.inference.types import Reply

                replies = []
                for call in calls:
                    documents = call.json["documents"]
                    carries_image = any(isinstance(document, dict) for document in documents)
                    tokens = (66 if carries_image else 0) + self.template
                    replies.append(
                        Reply(
                            200,
                            {"results": [{"index": 0, "relevance_score": 0.5}], "usage": {"prompt_tokens": tokens}},
                            {},
                        )
                    )
                return replies

        sender = HonestRerankSender()
        sender.template = self.TEMPLATE_TOKENS
        from rcp_ndcg.inference import RerankClient
        from rcp_ndcg.inference.config import RerankEndpoint

        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=8192,
                use_activation=False,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )
        asyncio.run(client.check_engine_media())


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


class TestVideoContainerFit:
    """A ``wire: video_url`` container is one prepared item: the media fit decides it whole (shrink does
    not apply to a container), and ``apply_media_fit`` consumes its decision -- a drop removes the part,
    a keep rides as prepared.

    The fit machinery is policy-independent, so these tests build the policies directly (a judge's
    ``preprocessing.video`` or a retrieval role's ``video_policy`` declares the same policy; the messages
    lowering sends the container as a ``video_url`` part)."""

    @staticmethod
    def _policies() -> tuple[ImagePolicy, Any]:
        from rcp_ndcg.data.resolution import VideoPolicy

        return ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl"), VideoPolicy(
            num_frames=4, wire="video_url", engine_video_pinning=True
        )

    @staticmethod
    def _container_content(tmp_path: Any) -> Any:
        from rcp_ndcg_core.content import VideoPart

        from rcp_ndcg.data.media import MediaRef

        container = tmp_path / "clip.mp4"
        container.write_bytes(b"mp4-bytes")
        return Content.from_parts(
            [
                VideoPart(
                    frames=[],
                    ref=MediaRef(
                        uri=container.as_uri(),
                        mime="video/mp4",
                        width=64,
                        height=64,
                        num_frames=64,
                    ),
                )
            ]
        )

    def test_a_container_in_the_media_fit_consumes_its_decision(self, tokenizer_json: str, tmp_path: Any) -> None:
        """``apply_media_fit`` consumes one decision per prepared item: a container is kept whole when the
        budget allows, dropped whole when it does not -- never a spurious DataError."""
        from PIL import Image as PILImage

        from rcp_ndcg.data.prepare import fit_media_to_budget, prepare_request

        image, video = self._policies()
        big = tmp_path / "page.png"
        PILImage.new("RGB", (900, 900), (10, 10, 200)).save(big, format="PNG")
        request = prepare_request([self._container_content(tmp_path), Content.from_image(big.as_uri())], image, video)
        fit = fit_media_to_budget(request.media, image=image, video=video, text_budget_tokens=10)
        out = apply_media_fit(request.contents, fit)
        assert len(out) == 2

    def test_a_container_that_cannot_fit_is_dropped_whole(self, tokenizer_json: str, tmp_path: Any) -> None:
        """A container cannot shrink: under ``cut`` the fit drops it whole, recorded under its doc_id."""
        from rcp_ndcg.data.prepare import fit_media_to_budget, prepare_request

        image, video = self._policies()
        request = prepare_request([self._container_content(tmp_path)], image, video)
        fit = fit_media_to_budget(request.media, image=image, video=video, text_budget_tokens=5)
        assert fit.dropped_positions == (0,) and fit.decisions == (None,)


class TestDropCensusDocIds:
    """Every drop census row records the ORIGINAL prepared item under ITS input's doc_id -- never the
    role name, never another document's id (the re-fix round's shifts)."""

    def test_rerank_pair_query_image_and_document_image_both_dropped(self, tokenizer_json: str, tmp_path: Any) -> None:

        from rcp_ndcg.data.prepare import MediaCensus

        census = MediaCensus()
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=4,
                use_activation=False,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=RecordingSender(),
            media_census=census,
        )
        query_image = _png_content(tmp_path, 10)
        document_image = _png_content(tmp_path, 20)
        client._fit_media_for_request([query_image, document_image], doc_ids=[QUERY_DOC_ID, "0"])
        rows = {doc_id for _corpus, doc_id, _uri, _dropped in census.recorded()}
        assert rows == {QUERY_DOC_ID, "0"}, "each drop under its own input's doc_id, never the role name"

    def test_a_document_with_two_images_both_dropped(self, tokenizer_json: str, tmp_path: Any) -> None:

        from rcp_ndcg.data.prepare import MediaCensus

        census = MediaCensus()
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=4,
                use_activation=False,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=4,
            ),
            sender=RecordingSender(),
            media_census=census,
        )
        page = _png_content(tmp_path, 30)
        document = Content.from_parts([page.parts[0], page.parts[0]])  # the same page twice

        client._fit_media_for_request([Content.from_text("query"), document], doc_ids=[QUERY_DOC_ID, "0"])

        rows = {doc_id for _corpus, doc_id, _uri, _dropped in census.recorded()}
        assert rows == {"0"}, "both drops under the document's id, never the role name"

    def test_pool_content_with_two_images_both_dropped(self, tokenizer_json: str, tmp_path: Any) -> None:

        from rcp_ndcg.data.prepare import MediaCensus

        census = MediaCensus()
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=4,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=4,
            ),
            sender=RecordingSender(),
            media_census=census,
        )
        page = _png_content(tmp_path, 40)
        document = Content.from_parts([page.parts[0], page.parts[0]])  # the same page twice

        asyncio.run(client.aencode([document], EncodeRole.DOCUMENT))

        rows = {doc_id for _corpus, doc_id, _uri, _dropped in census.recorded()}
        assert rows == {"0"}, "both drops under the item's id, never the role name"


def _words(count: int) -> str:
    """A text of exactly ``count`` tokens in the word tokenizer (one token per word)."""
    return " ".join(["alpha"] * count)


def _image_content(tmp_path: Any, index: int, size: int) -> Any:
    """One square PNG (``size`` a multiple of the processor's patch factor, so the token count is exact)."""
    from rcp_ndcg_core.content import Content

    page = tmp_path / f"page-{index}-{size}.png"
    PILImage.new("RGB", (size, size), (10, 10, 200)).save(page, format="PNG")
    return Content.from_image(page.as_uri())


def _media_tokens_of(document: Any, policy: ImagePolicy) -> int:
    """The vision blocks of one lowered rerank document, as an honest engine charges them."""
    tokens = 0
    for part in document.get("content", []) if isinstance(document, dict) else []:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        url = part["image_url"]["url"]
        payload = (
            base64.b64decode(url.partition(",")[2])
            if url.startswith("data:")
            else Path(url.replace("file://", "")).read_bytes()
        )
        image = PILImage.open(io.BytesIO(payload))
        content = Content.from_image(url, width=image.size[0], height=image.size[1])
        tokens += content_media_tokens(content, policy, None).tokens
    return tokens


class _HonestBudgetRerankServer(RecordingSender):
    """A rerank server that charges what an honest engine charges: for a pointwise reranker every call
    renders one prompt per (query, document) pair -- the query tokens plus that document's text and media
    blocks -- and the recorded number per call is the DEEPEST pair (the engine's longest prompt), so a test
    can assert no pair went out over the budget."""

    def __init__(self, tokenizer: Any, policy: ImagePolicy) -> None:
        self.tokenizer = tokenizer
        self.policy = policy
        self.prompt_tokens: list[int] = []
        self.bodies: list[dict[str, Any]] = []

    async def send(self, calls: Any) -> list[Any]:
        for call in calls:
            body = call.json
            self.bodies.append(body)
            query_tokens = self.tokenizer.count(body["query"])
            deepest = 0
            for document in body["documents"]:
                pair = query_tokens + self.tokenizer.count(document if isinstance(document, str) else "")
                pair += _media_tokens_of(document, self.policy)
                deepest = max(deepest, pair)
            self.prompt_tokens.append(deepest)
        documents = calls[0].json["documents"]
        rows = [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]
        return [Reply(200, {"results": rows[::-1]}, {})]


class TestRerankPairFitWithMedia:
    """The pair fit with media (the sweep's blocker): a candidate set with mixed media settles ONE query
    span for the whole batch -- no crash, and no pair shipped over the budget."""

    POLICY = {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"}

    @staticmethod
    def _client(tokenizer_json: str, sender: Any, *, max_tokens: int) -> RerankClient:
        return RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=max_tokens,
                query_max_tokens=70,
                use_activation=False,
                image_policy=dict(TestRerankPairFitWithMedia.POLICY),
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )

    def test_one_plain_and_one_media_document_settle_one_query_span(self, tokenizer_json: str, tmp_path: Any) -> None:
        """A plain and a media document in one candidate set: the old fit re-cut the query per pair (the
        probe reserved only the query's media) and died on its own consistency check; now one span settles
        across the batch and the scores stay aligned to the documents."""
        sender = _HonestBudgetRerankServer(load_tokenizer(tokenizer_json), ImagePolicy(**self.POLICY))
        client = self._client(tokenizer_json, sender, max_tokens=200)
        result = client.rerank(_words(40), ["some document text", _image_content(tmp_path, 1, 392)], instruction="find")

        assert len(result.scores) == 2, "the scores stay aligned to the documents"
        assert sender.prompt_tokens and max(sender.prompt_tokens) <= 200, "no pair ships over the budget"

    def test_a_media_pair_is_never_shipped_over_the_budget(self, tokenizer_json: str, tmp_path: Any) -> None:
        """The sweep's A2 (repro_pair_budget.py): one two-image document and a long query with a generous
        share. The old fit settled the query against a probe that reserved only the query's media and
        shipped that span with the pair's media -- 300 query tokens + 512 media = 812 over max_tokens 700,
        leaving the truncation to the engine. The wire now carries the span the pair fit verified."""
        page = _image_content(tmp_path, 0, 896)  # 900x900 class: one image ~1026 tokens, two over the budget
        document = Content.from_parts([*page.parts, *page.parts])
        sender = _HonestBudgetRerankServer(load_tokenizer(tokenizer_json), ImagePolicy(**self.POLICY))
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=700,
                query_max_tokens=600,
                use_activation=False,
                image_policy={"min_px": 200704, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )
        result = client.rerank(_words(300), [document], instruction="find")

        assert result.scores == (0.0,)
        assert sender.prompt_tokens, "the request went out"
        assert max(sender.prompt_tokens) <= 700, "the shipped pair is within the budget"

    def test_a_heavy_media_document_does_not_blame_the_query(self, tokenizer_json: str, tmp_path: Any) -> None:
        """A document whose media shrink the pair's cap below the settled query: the fit attributes the
        constraint to the media it reserved, and the request is served -- the query is never blamed for
        media that fit."""
        sender = _HonestBudgetRerankServer(load_tokenizer(tokenizer_json), ImagePolicy(**self.POLICY))
        client = self._client(tokenizer_json, sender, max_tokens=200)
        document = Content.from_parts([*_image_content(tmp_path, 2, 392).parts, TextPart(text="some text")])

        result = client.rerank(_words(40), [document], instruction="find")

        assert result.scores == (0.0,)
        assert sender.prompt_tokens and max(sender.prompt_tokens) <= 200, "no pair ships over the budget"


class TestOnePreparationScalesLinearly:
    """A corpus encode is ONE request through the client (the retrieval API hands it the whole corpus), so
    slicing the one preparation per item must cost each item once: a per-item slice that re-walks every
    content made a media corpus quadratic (10k page images, ~10^8 part walks before a request went out)."""

    def test_the_per_item_media_fit_walks_each_item_a_bounded_number_of_times(
        self, tokenizer_json: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        count = 60
        contents = [_image_content(tmp_path, index, 56) for index in range(count)]
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((2, 2), dtype=np.float16), media_vector=np.ones((2, 2), dtype=np.float16))
        )
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )
        walks = [0]
        media_refs = ImagePart.media_refs

        def counted(part: ImagePart) -> Any:
            walks[0] += 1
            return media_refs(part)

        monkeypatch.setattr(ImagePart, "media_refs", counted)

        result = asyncio.run(client.aencode(contents, EncodeRole.DOCUMENT))

        assert result.num_items == count
        assert walks[0] <= 8 * count, f"{walks[0]} part walks for {count} items: the slicing is not linear"


class TestPoolFramePerShape:
    """The pooling media allowance reserves the REQUEST SHAPE's frame, not the query's (the verifier's
    R17): a document batch leaves the ``document`` frame's tokens beside the media, or the text fit refuses
    the request blaming the media that fit."""

    POLICY = {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"}

    def _client(self, tokenizer_json: str, sender: Any, *, max_tokens: int, template: Any) -> PoolingClient:
        return PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=max_tokens,
                template=template,
                image_policy=dict(self.POLICY),
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )

    def test_a_document_batch_reserves_the_document_frame(self, tokenizer_json: str, tmp_path: Any) -> None:
        """Frame-heavy ``document`` shape (101 tokens), light ``query`` shape, max_tokens 300, one
        258-token media item: the old allowance reserved the query's ~0-token frame and the request died
        (``the fixed template overhead (101 tokens) plus the declared media (258) already fill the budget``);
        the fit must shrink/drop within the room the DOCUMENT frame leaves and serve the request."""
        template = TemplateSpec(
            query=(Segment(content="query"),),
            document=(Segment(fixed=" ".join(["frame"] * 101)), Segment(content="document")),
        )
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((2, 2), dtype=np.float16), media_vector=np.ones((2, 2), dtype=np.float16))
        )
        client = self._client(tokenizer_json, sender, max_tokens=300, template=template)

        result = asyncio.run(client.aencode([_image_content(tmp_path, 0, 448)], EncodeRole.DOCUMENT))

        assert result.num_items == 1, "the request is served (the media fit ran within the document frame)"

    def test_a_document_only_template_never_reserves_a_query_frame(self, tokenizer_json: str, tmp_path: Any) -> None:
        """A template with no ``query`` shape must not be blamed for the client's own lookup (the old code
        asked for the ``query`` frame on a document batch and died ``the template declares no 'query'
        shape``)."""
        template = TemplateSpec(document=(Segment(content="document"),))
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((2, 2), dtype=np.float16), media_vector=np.ones((2, 2), dtype=np.float16))
        )
        client = self._client(tokenizer_json, sender, max_tokens=300, template=template)

        result = asyncio.run(client.aencode([_image_content(tmp_path, 0, 448)], EncodeRole.DOCUMENT))

        assert result.num_items == 1, "the request is served against its own shape's frame"

    def test_a_query_batch_reserves_against_its_own_query_budget(self, tokenizer_json: str, tmp_path: Any) -> None:
        """With a per-shape ``query_max_tokens``, the query shape's budget is that share (fit measures a query
        against it), so the media allowance must be counted from it too: a 258-token query image under a
        200-token query budget is shrunk to fit and served -- an allowance from ``max_tokens`` keeps the
        image whole, and the text fit then refuses the request blaming media the fit had kept."""
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((2, 2), dtype=np.float16), media_vector=np.ones((2, 2), dtype=np.float16))
        )
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="colqwen",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                query_max_tokens=200,
                image_policy=dict(self.POLICY),
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )

        result = asyncio.run(client.aencode([_image_content(tmp_path, 0, 448)], EncodeRole.QUERY))

        assert result.num_items == 1, "the query is served within its own budget"


class TestEmbedMessagesMediaFit:
    """The embed role's ``messages`` route fits its media exactly as the pooling route does: one preparation
    of the request, sliced per item (never a second preparation of already-prepared contents), against the
    item shape's budget minus its fixed frame."""

    POLICY = {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"}

    def _client(self, tokenizer_json: str, *, max_tokens: int, template: Any = None, census: Any = None) -> Any:
        from rcp_ndcg.inference import EmbeddingClient
        from rcp_ndcg.inference.config import EmbeddingEndpoint

        return EmbeddingClient(
            EmbeddingEndpoint(
                base_url="fake://seed/7?dim=4",
                model="m",
                request_shape="messages",
                tokenizer=tokenizer_json,
                max_tokens=max_tokens,
                template=template,
                image_policy=dict(self.POLICY),
                image_processor="qwen2_vl",
                max_images=2,
            ),
            media_census=census,
        )

    def test_a_document_reserves_the_document_frame(self, tokenizer_json: str, tmp_path: Any) -> None:
        """Frame-heavy ``document`` shape (101 tokens), max_tokens 300, one 258-token image: an allowance of
        the bare max_tokens keeps the image whole and the text fit then refuses the request; the media fit
        must shrink within the room the frame leaves and serve it."""
        template = TemplateSpec(
            query=(Segment(content="query"),),
            document=(Segment(fixed=" ".join(["frame"] * 101)), Segment(content="document")),
        )
        client = self._client(tokenizer_json, max_tokens=300, template=template)

        result = client.encode([_image_content(tmp_path, 0, 448)], EncodeRole.DOCUMENT)
        client.close()

        assert result.num_items == 1, "the request is served (the media fit ran within the document frame)"

    def test_the_census_rows_name_the_source_never_a_data_uri(self, tokenizer_json: str, tmp_path: Any) -> None:
        """The kept and the dropped media are recorded under their SOURCE uri: a second preparation of the
        already-prepared contents would record the inlined ``data:`` bytes instead."""
        from rcp_ndcg.data.prepare import MediaCensus

        census = MediaCensus()
        client = self._client(tokenizer_json, max_tokens=4, census=census)
        page = _image_content(tmp_path, 0, 448)

        client.encode([Content.from_parts([*page.parts, *page.parts])], EncodeRole.DOCUMENT)
        client.close()

        rows = census.recorded()
        assert rows and any(dropped for *_rest, dropped in rows), "the drops are recorded"
        assert not [uri for _corpus, _doc, uri, _dropped in rows if uri.startswith("data:")], rows


def test_the_pair_fit_invariant_message_carries_its_numbers() -> None:
    """The bug-report DataError interpolates its numbers -- no literal braces in the message a user reports."""
    from rcp_ndcg.errors import DataError

    client = RerankClient(
        RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=str(SESSION_TOKENIZER),
            max_tokens=12,
            use_activation=False,
        ),
        sender=_HonestBudgetRerankServer(
            load_tokenizer(str(SESSION_TOKENIZER)), ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl")
        ),
    )
    from rcp_ndcg.data.preprocess import ContentParts

    contents: list[ContentParts] = [("q", " ".join(["word"] * 50))]
    with pytest.raises(DataError, match="over the budget of 12") as caught:
        client._assert_pairs_within_budget("q", contents, [0], None)  # type: ignore[arg-type]
    assert "{" not in str(caught.value), "the message interpolates its values"


class TestLongQueryDoesNotDropFittingMedia:
    """The pair media allowance reserves the SETTLED query's render, not the raw unsettled one (the
    verifier's R2): a query over its share ships at its share, so media that fit beside the shipped pair
    must go whole (``fit_media_to_budget`` rule 1) -- never dropped against a query length that never
    ships."""

    def test_a_shared_query_keeps_media_that_fit_the_shipped_pair(self, tokenizer_json: str, tmp_path: Any) -> None:
        policy = ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl")
        sender = _HonestBudgetRerankServer(load_tokenizer(tokenizer_json), policy)
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                tokenizer=tokenizer_json,
                max_tokens=1000,
                query_max_tokens=10,
                use_activation=False,
                image_policy=dict(TestRerankPairFitWithMedia.POLICY),
                image_processor="qwen2_vl",
                max_images=2,
            ),
            sender=sender,
        )
        document = Content.from_parts([*_image_content(tmp_path, 9, 392).parts, TextPart(text="a b c")])

        result = client.rerank(_words(1000), [document], instruction=None)

        assert result.scores == (0.0,)
        assert not [1 for _c, _d, _u, dropped in client.media_census.recorded() if dropped], (
            "no media drop: the image fits the shipped 10-token-share pair"
        )
        body = sender.bodies[0]["documents"][0]
        assert "image_url" in __import__("json").dumps(body), "the document's image rides the wire"
        assert max(sender.prompt_tokens) <= 1000, "within the budget"
