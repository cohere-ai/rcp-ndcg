"""The stage-order invariants of the one preparation pipeline.

The pipeline (:data:`rcp_ndcg.inference.clients._base.STAGES`) is one declared order shared by every
role client: content normalisation -> empty handling -> media preparation -> template render -> budget
fit -> wire lowering. These tests pin the order itself and the invariants that follow from it: the
empty policy fires before the template frames a document and again on the media fit's all-dropped
outputs; one frame rides every route; and the per-row record is emitted for every change and only for
a change.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content, ImagePart, TextPart

from rcp_ndcg.inference.clients._base import STAGES
from rcp_ndcg.inference.clients.embed import EmbeddingClient
from rcp_ndcg.inference.clients.pool import PoolingClient
from rcp_ndcg.inference.clients.rerank import RerankClient
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.types import EncodeRole, Reply
from tests.conftest import SESSION_TOKENIZER
from tests.inference._pooling import PoolingServer, server_sender


class _PoolingSender:
    """A sender over the offline pooling fake: one token matrix per input, so a prepare runs end to end."""

    def __init__(self, server: PoolingServer) -> None:
        self._inner = server_sender(server)
        self.sent: list[list[dict[str, Any]]] = []

    async def send(self, calls: Any) -> list[Any]:
        self.sent.append([call.json for call in calls])
        return await self._inner.send(calls)

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


class _RecordingClient(EmbeddingClient):
    """An embed client that records which pipeline stage ran, in order.

    Each stage's one home on the base is wrapped; the runners compose them, so the record is the order
    the pipeline actually ran -- what the invariant asserts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.stages_run: list[str] = []
        super().__init__(*args, **kwargs)

    def _record(self, stage: str) -> None:
        self.stages_run.append(stage)

    def _stage_normalise(self, contents, *, side, prompt, instruction=None):  # noqa: ANN001, ANN202
        self._record("normalise")
        return super()._stage_normalise(contents, side=side, prompt=prompt, instruction=instruction)

    def _apply_empty_documents(self, contents, *, changes=None, prefix=""):  # noqa: ANN001, ANN202
        self._record("empty")
        return super()._apply_empty_documents(contents, changes=changes, prefix=prefix)

    def _stage_media(self, contents, *, shape, positions, changes):  # noqa: ANN001, ANN202
        self._record("media")
        return super()._stage_media(contents, shape=shape, positions=positions, changes=changes)

    def _stage_budget(self, contents, *, shape, media_tokens, positions, instruction=None, changes):  # noqa: ANN001, ANN202
        self._record("render")
        self._record("budget")
        return super()._stage_budget(
            contents,
            shape=shape,
            media_tokens=media_tokens,
            positions=positions,
            instruction=instruction,
            changes=changes,
        )

    def _stage_lower(self, contents, *, result, shape):  # noqa: ANN001, ANN202
        self._record("lower")
        return super()._stage_lower(contents, result=result, shape=shape)


def _embed_client(sender: Any, **overrides: Any) -> _RecordingClient:
    settings: dict[str, Any] = {
        "base_url": "http://engine:8000/v1",
        "model": "m",
        "tokenizer": str(SESSION_TOKENIZER),
        "max_tokens": 64,
    }
    settings.update(overrides)
    return _RecordingClient(EmbeddingEndpoint(**settings), sender=sender)


class _NoopSender:
    """A sender that answers nothing: the invariants below run up to the send (the stages run first)."""

    async def send(self, calls: Any) -> list[Any]:
        raise AssertionError("no request is expected: the stage order is decided before the send")

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


class TestTheDeclaredOrder:
    """``normalise -> empty -> media -> render -> budget -> lower``, on the runner every role composes."""

    def test_the_stage_tuple_is_the_declared_pipeline(self) -> None:
        assert STAGES == ("normalise", "empty", "media", "render", "budget", "lower")

    def test_the_runner_runs_the_stages_in_the_declared_order(self) -> None:
        client = _embed_client(_NoopSender())
        client.stages_run.clear()  # the constructor probes nothing; the list starts empty
        prepared = client._prepare([Content.from_text("a"), Content.from_text("b")], EncodeRole.DOCUMENT)
        assert prepared.items and prepared.positions == (0, 1)
        assert client.stages_run == ["normalise", "empty", "media", "render", "budget", "lower"]

    def test_an_omitted_empty_item_never_reaches_the_media_stage(self) -> None:
        """An empty document under ``omit_zero`` is decided at the empty stage: nothing is fetched,
        sized or counted for it afterwards."""
        client = _embed_client(_NoopSender(), empty_doc="omit_zero")
        client._prepare([Content.from_text(""), Content.from_text("kept")], EncodeRole.DOCUMENT)
        # The media stage runs once (for the kept items only); the empty stage ran twice only if the
        # media fit dropped something -- with no media it runs exactly once.
        assert client.stages_run == ["normalise", "empty", "media", "render", "budget", "lower"]

    def test_the_media_stage_may_send_the_empty_policy_back(self, tmp_path: Any) -> None:
        """A document whose every media item the budget dropped is empty: the same empty policy decides
        again on the fitted content (the media stage's re-entry), so the policy always sees the content
        as it will be sent."""
        from PIL import Image

        page = tmp_path / "page.png"
        Image.new("RGB", (900, 900), (10, 10, 200)).save(page, format="PNG")
        client = _embed_client(
            _NoopSender(),
            request_shape="messages",
            max_tokens=4,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=2,
            empty_doc="omit_zero",
        )
        prepared = client._prepare([Content.from_image(page.as_uri())], EncodeRole.DOCUMENT)
        assert prepared.omitted == (0,), "the all-dropped document follows the empty policy"
        assert client.stages_run == ["normalise", "empty", "media", "empty", "render", "budget", "lower"]


class TestEmptyBeforeTheFrame:
    """``empty_doc`` is decided on the content, before the template frames it."""

    def test_an_empty_document_is_never_a_framed_non_empty_turn(self) -> None:
        """``send_text`` substitutes the placeholder on the CONTENT, before any frame: the sent text is
        the frame around the placeholder, never the frame around an empty turn with the placeholder
        somewhere outside it."""
        client = _embed_client(_NoopSender(), empty_doc="send_text", empty_doc_text="NULL")
        prepared = client._prepare([Content.from_text(""), Content.from_text("evidence")], EncodeRole.DOCUMENT)
        texts = [content.text for content in prepared.items]
        assert texts == ["NULL", "evidence"]
        # and the record names the substitution, once, for the row it changed
        records = [record for record in client.processing if record.changed]
        assert [record.input_id for record in records] == ["0"]
        assert records[0].mechanisms == ("empty_doc",)

    def test_no_record_for_a_row_sent_as_given(self) -> None:
        """The record is the pipeline's one output: emitted for every change and only for a change --
        an uncut row under budget makes no record and writes no census row."""
        client = _embed_client(_NoopSender(), max_tokens=64)
        client._prepare([Content.from_text("fine as given")], EncodeRole.DOCUMENT)
        assert client.processing == []
        assert len(client.census) == 0


class TestOneFramePerRoute:
    """One frame on every route: the text route sends the declared frame once; the ``messages`` route
    sends the content span and leaves the frame to the engine's chat template."""

    def test_the_text_route_sends_the_render_once(self) -> None:
        sender = _NoopSender()
        client = _embed_client(sender, template={"document": [{"fixed": "Passage: "}, {"content": "document"}]})
        prepared = client._prepare([Content.from_text("body")], EncodeRole.DOCUMENT)
        assert prepared.items[0].text == "Passage: body"

    def test_the_messages_route_sends_the_content_span_not_the_render(self) -> None:
        sender = _NoopSender()
        client = _embed_client(
            sender,
            request_shape="messages",
            template={"document": [{"fixed": "Passage: "}, {"content": "document"}]},
        )
        prepared = client._prepare([Content.from_text("body")], EncodeRole.DOCUMENT)
        # The engine's chat template frames it; framing here too would frame twice.
        assert prepared.items[0].text == "body"

    def test_the_messages_route_frames_media_once(self, tmp_path: Any) -> None:
        """A media item's conversation carries the caption once -- the declared frame is the chat
        template's business, never the client's twice."""
        from rcp_ndcg_core.content import MediaRef

        page = tmp_path / "page.png"
        page.write_bytes(self._png())
        caption = Content.from_parts([TextPart(text="a caption"), ImagePart(ref=MediaRef(uri=page.as_uri()))])
        sender = _NoopSender()
        client = _embed_client(
            sender,
            request_shape="messages",
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=2,
            template={"document": [{"fixed": "Passage: "}, {"content": "document"}]},
        )
        prepared = client._prepare([caption], EncodeRole.DOCUMENT)
        assert prepared.items[0].text == "a caption"

    @staticmethod
    def _png() -> bytes:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (120, 120, 120)).save(buffer, format="PNG")
        return buffer.getvalue()


class TestPoolAndRerankComposeTheSameOrder:
    """The pool role runs the same runner; the rerank's pair preparation composes the same stages."""

    def test_the_pool_client_runs_the_declared_order(self) -> None:
        recorded: list[str] = []

        class RecordingPool(PoolingClient):
            def _stage_normalise(self, contents, *, side, prompt, instruction=None):  # noqa: ANN001, ANN202
                recorded.append("normalise")
                return super()._stage_normalise(contents, side=side, prompt=prompt, instruction=instruction)

            def _apply_empty_documents(self, contents, *, changes=None, prefix=""):  # noqa: ANN001, ANN202
                recorded.append("empty")
                return super()._apply_empty_documents(contents, changes=changes, prefix=prefix)

            def _stage_media(self, contents, *, shape, positions, changes):  # noqa: ANN001, ANN202
                recorded.append("media")
                return super()._stage_media(contents, shape=shape, positions=positions, changes=changes)

            def _stage_budget(self, contents, *, shape, media_tokens, positions, instruction=None, changes):  # noqa: ANN001, ANN202
                recorded.append("render")
                recorded.append("budget")
                return super()._stage_budget(
                    contents,
                    shape=shape,
                    media_tokens=media_tokens,
                    positions=positions,
                    instruction=instruction,
                    changes=changes,
                )

            def _stage_lower(self, contents, *, result, shape):  # noqa: ANN001, ANN202
                recorded.append("lower")
                return super()._stage_lower(contents, result=result, shape=shape)

        client = RecordingPool(
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="colqwen",
                dim=2,
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=64,
            ),
            sender=_PoolingSender(PoolingServer({}, default=np.ones((2, 2), dtype=np.float16))),
        )
        client._prepare([Content.from_text("a"), Content.from_text("b")], EncodeRole.DOCUMENT)
        assert recorded == ["normalise", "empty", "media", "render", "budget", "lower"]

    def test_the_rerank_pair_preparation_runs_the_stages_in_order(self) -> None:
        recorded: list[str] = []

        class RecordingRerank(RerankClient):
            def _stage_normalise(self, contents, *, side, prompt, instruction=None):  # noqa: ANN001, ANN202
                recorded.append("normalise")
                return super()._stage_normalise(contents, side=side, prompt=prompt, instruction=instruction)

            def _apply_empty_documents(self, contents, *, changes=None, prefix=""):  # noqa: ANN001, ANN202
                recorded.append("empty")
                return super()._apply_empty_documents(contents, changes=changes, prefix=prefix)

            def _prepare_request(self, contents, *, doc_ids=None):  # noqa: ANN001, ANN202
                recorded.append("media")
                return super()._prepare_request(contents, doc_ids=doc_ids)

            def _fit(self, inputs, shape, **kwargs: Any):  # noqa: ANN001, ANN202
                recorded.append("render")
                recorded.append("budget")
                return super()._fit(inputs, shape, **kwargs)

        client = RecordingRerank(
            RerankEndpoint(
                base_url="http://engine:8000/v1",
                model="rr",
                tokenizer=str(SESSION_TOKENIZER),
                max_tokens=64,
                use_activation=False,
            ),
            sender=_RecordingRerankSender(),
        )
        client.rerank("what", ["doc one", "doc two"])
        # The pair pipeline: the query's and the documents' normalise, the media stage (one preparation),
        # the empty stage on the media-fitted documents, then the pair fit (render + budget) -- the shared
        # fit's settlement probe first, then the batch's pairs.
        assert recorded == [
            "normalise",
            "normalise",
            "media",
            "empty",
            "render",
            "budget",
            "render",
            "budget",
        ], recorded


class _RecordingRerankSender:
    """A rerank sender: one score per document, no engine (the reply shape the rerank wires answer)."""

    def __init__(self) -> None:
        self.bodies: list[Any] = []

    async def send(self, calls: Any) -> list[Any]:
        replies = []
        for call in calls:
            self.bodies.append(call.json)
            documents = call.json.get("documents", [])
            rows = [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]
            replies.append(Reply(200, {"results": rows}, {}))
        return replies

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)
