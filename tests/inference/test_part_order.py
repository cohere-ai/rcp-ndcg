"""A5: an item's text parts stay in their own places around its media parts.

``[text A, image, text B]`` is one input the model reads in order: the caption before the page and the
caption after it are different inputs, and the fit's cut applies to each text part where it stands -- a
later text part is never hoisted into the first slot. The three served roles that carry media (the embed
and pool roles' ``messages`` route and the rerank role's document body) each pin it here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage
from rcp_ndcg_core.content import Content, TextPart

from rcp_ndcg.data.prepare import prepare_request
from rcp_ndcg.data.resolution import ImagePolicy
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.inference import EmbeddingClient, PoolingClient, RerankClient
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.types import EncodeRole, Reply
from tests.conftest import SESSION_TOKENIZER

IMAGE_POLICY = {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"}


class RecordingSender:
    """Records every request body and answers each role's shape plausibly."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def send(self, calls: Any) -> list[Any]:
        replies: list[Any] = []
        for call in calls:
            body = call.json
            self.bodies.append(body)
            if call.path == "/pooling":
                items = [{"index": i, "data": [[1.0, 1.0]]} for i in range(1)]
                replies.append(Reply(200, {"data": items, "usage": {"prompt_tokens": 1}}, {}))
            elif call.path == "/rerank":
                rows = [{"index": i, "relevance_score": 0.5} for i in range(len(body["documents"]))]
                replies.append(Reply(200, {"results": rows[::-1]}, {}))
            else:
                replies.append(Reply(200, {"data": [{"index": 0, "embedding": [1.0, 1.0]}]}, {}))
        return replies

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


def _page(tmp_path: Path) -> Content:
    path = tmp_path / "page.png"
    PILImage.new("RGB", (300, 300), (10, 10, 200)).save(path, format="PNG")
    return Content.from_image(path.as_uri())


def _interleaved(tmp_path: Path) -> Content:
    """``[text A, image, text B]``: text on both sides of the page."""
    return Content.from_parts([TextPart(text="alpha beta"), *_page(tmp_path).parts, TextPart(text="gamma delta")])


def _embed(tmp_path: Path, sender: RecordingSender, **overrides: Any) -> EmbeddingClient:
    settings: dict[str, Any] = {
        "base_url": "http://engine:8000/v1",
        "model": "m",
        "tokenizer": str(SESSION_TOKENIZER),
        "max_tokens": 8192,
        "request_shape": "messages",
        "image_policy": IMAGE_POLICY,
        "max_images": 1,
    }
    settings.update(overrides)
    return EmbeddingClient(EmbeddingEndpoint(**settings), sender=sender)


def _pool(sender: RecordingSender, **overrides: Any) -> PoolingClient:
    settings: dict[str, Any] = {
        "base_url": "http://engine:8000/v1",
        "model": "m",
        "tokenizer": str(SESSION_TOKENIZER),
        "max_tokens": 8192,
        "dim": 2,
        "image_policy": IMAGE_POLICY,
        "max_images": 1,
    }
    settings.update(overrides)
    return PoolingClient(PoolingEndpoint(**settings), sender=sender)


def _rerank(sender: RecordingSender, **overrides: Any) -> RerankClient:
    settings: dict[str, Any] = {
        "base_url": "http://engine:8000/v1",
        "model": "m",
        "tokenizer": str(SESSION_TOKENIZER),
        "max_tokens": 8192,
        "use_activation": False,
        "image_policy": IMAGE_POLICY,
        "max_images": 1,
    }
    settings.update(overrides)
    return RerankClient(RerankEndpoint(**settings), sender=sender)


def _part_kinds(parts: list[dict[str, Any]]) -> list[str]:
    return [part["type"] for part in parts]


def _texts(parts: list[dict[str, Any]]) -> list[str]:
    return [str(part["text"]) for part in parts if part["type"] == "text"]


def test_embed_keeps_each_text_part_in_place(tmp_path: Path) -> None:
    sender = RecordingSender()
    client = _embed(tmp_path, sender)
    client.encode([_interleaved(tmp_path)], EncodeRole.DOCUMENT)
    parts = sender.bodies[-1]["messages"][0]["content"]
    assert _part_kinds(parts) == ["text", "image_url", "text"]
    assert _texts(parts) == ["alpha beta", "gamma delta"], "text after the media is not hoisted"


def test_pool_keeps_each_text_part_in_place(tmp_path: Path) -> None:
    sender = RecordingSender()
    client = _pool(sender)
    client.encode([_interleaved(tmp_path)], EncodeRole.DOCUMENT)
    parts = sender.bodies[-1]["messages"][0]["content"]
    assert _part_kinds(parts) == ["text", "image_url", "text"]
    assert _texts(parts) == ["alpha beta", "gamma delta"], "text after the media is not hoisted"


def test_rerank_keeps_each_text_part_in_place(tmp_path: Path) -> None:
    sender = RecordingSender()
    client = _rerank(sender)
    client.rerank("the query", [_interleaved(tmp_path)])
    document = sender.bodies[-1]["documents"][0]
    assert _part_kinds(document["content"]) == ["text", "image_url", "text"]
    assert _texts(document["content"]) == ["alpha beta", "gamma delta"], "text after the media is not hoisted"


def _media_tokens(tmp_path: Path) -> int:
    page = _page(tmp_path)
    policy = ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl")
    return prepare_request([page], policy, None).tokens.tokens


def test_embed_applies_the_fit_cut_to_each_part_in_place(tmp_path: Path) -> None:
    """The fit's cut is distributed over the parts where they stand: with room for ``alpha beta`` and
    ``gamma``, the first part is kept whole, the page stays in the middle, and the last part is cut to
    its own prefix -- and the cut is recorded once per part."""
    sender = RecordingSender()
    budget = _media_tokens(tmp_path) + 3
    client = _embed(tmp_path, sender, max_tokens=budget)
    client.encode([_interleaved(tmp_path)], EncodeRole.DOCUMENT)
    parts = sender.bodies[-1]["messages"][0]["content"]
    assert _part_kinds(parts) == ["text", "image_url", "text"]
    assert _texts(parts) == ["alpha beta", "gamma"], "each text part carries its own slice of the kept span"
    (cut,) = client.census.cuts()
    assert (cut.doc_id, cut.kept_chars, cut.original_chars) == ("0", len("gamma"), len("gamma delta")), (
        "only the part the cut shortened is recorded, under the input's id"
    )
    assert cut.cause == "budget_cut"


@pytest.mark.parametrize("client_kind", ["embed", "pool", "rerank"])
def test_the_parts_survive_the_prepare_itself(tmp_path: Path, client_kind: str) -> None:
    """The prepared item's own parts (before the wire lowering) keep the given order too."""
    sender = RecordingSender()
    content = _interleaved(tmp_path)
    if client_kind == "embed":
        prepared = _embed(tmp_path, sender)._prepare([content], EncodeRole.DOCUMENT)
    elif client_kind == "pool":
        prepared = _pool(sender)._prepare([content], EncodeRole.DOCUMENT)
    else:
        client = _rerank(sender)
        _, documents, _, _, _ = client._fit_pair(Content.from_text("the query"), [content], instruction=None)
        assert _part_kinds_of(documents[0]) == ["text", "image", "text"]
        return
    assert _part_kinds_of(prepared.items[0]) == ["text", "image", "text"]
    assert [part.text for part in prepared.items[0].parts if isinstance(part, TextPart)] == [
        "alpha beta",
        "gamma delta",
    ]


def _part_kinds_of(content: Content) -> list[str]:
    return [part.type for part in content.parts]


def test_a_normalising_template_refuses_a_multipart_media_content(tmp_path: Path) -> None:
    """A declared normalisation is applied to the joined text, so its cut span is not a prefix of the raw
    parts and cannot be distributed over them: the combination is refused before anything is sent, never
    silently hoisted into the first slot."""
    sender = RecordingSender()
    client = _embed(
        tmp_path,
        sender,
        template={"document": [{"content": "document"}], "normalize": ["strip"]},
    )
    with pytest.raises(ConfigError, match="normalisation") as caught:
        client.encode([_interleaved(tmp_path)], EncodeRole.DOCUMENT)
    assert "hoisted" in str(caught.value)
    assert sender.bodies == [], "nothing was sent"


def test_a_normalising_pair_template_refuses_a_multipart_media_document(tmp_path: Path) -> None:
    """The rerank pair path refuses the same combination, before any fit records a cut."""
    sender = RecordingSender()
    client = _rerank(
        sender,
        template={"pair": [{"content": "query"}, {"content": "document"}], "normalize": ["strip"]},
    )
    with pytest.raises(ConfigError, match="normalisation"):
        client.rerank("the query", [_interleaved(tmp_path)])
    assert sender.bodies == []


def test_a_pair_fit_refuses_a_non_string_part(tmp_path: Path) -> None:
    """The pair shape's ``parts`` validation is as strict as the single shapes': a non-string part is a
    typed refusal, never coerced."""
    from rcp_ndcg.data.preprocess import TextBudget, fit

    with pytest.raises(DataError, match="text-part lists"):
        fit(
            [("the query", "the document")],
            shape="pair",
            budget=TextBudget(tokenizer=str(SESSION_TOKENIZER), max_tokens=64),
            parts=[(("the query",), (1,))],  # type: ignore[list-item]
        )
