"""The model layer over media and chat-shaped records (review A1): a corpus with media builds, a media
request replays only for its own content identity and declared processing, and a record the model layer
cannot model is **skipped and named** -- never a bare ``ValueError``, never a whole-corpus failure.

The corpus here is a small in-test one (the shape the wave's media recording writes): one ``messages``
request carrying an inline image, one pointwise ``/rerank`` request whose document is a content-parts
object, and records that carry something the strategy does not model.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from rcp_ndcg_test.engines import (
    ChatPrompts,
    EngineFacts,
    Exchange,
    MediaIdentity,
    MediaPrompt,
    PairPrompts,
    RequestPrompts,
    StringsPrompts,
    VllmEmulator,
    request_context,
    verification_record,
)
from tests._engines import observation_corpus

from tests._tokenizers import word_tokenizer

FACTS = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", 512)
SOURCE = "x-rcp-ndcg-emulator-source"


def image_part(payload: bytes, *, mime: str = "image/png") -> dict[str, Any]:
    """One inline image part, as the media lowering writes it."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")},
    }


def identity_of(part: dict[str, Any], *, processing: str = "p0", tokens: int = 7) -> tuple[MediaIdentity, int]:
    """A media model for the tests: the sent bytes' SHA-256 under the given processing, ``tokens`` for it."""
    url = str((part.get("image_url") or {}).get("url") or (part.get("video_url") or {}).get("url"))
    digest = hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
    kind = "image" if part.get("type") == "image_url" else "video"
    return MediaIdentity(kind=kind, sha256=digest, processing=processing), tokens


def chat_render(conversation: list[Any], add_generation_prompt: bool) -> str:
    """A stand-in engine chat render: the conversation's text parts joined (the real one is the served
    chat template, which the wiring reads; this test pins the emulator's own behaviour)."""
    texts = [
        str(part.get("text", ""))
        for message in conversation
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "chat " + " ".join(texts)


def chat_strategy(media: Any = identity_of, *, slot: str = "vector") -> ChatPrompts:
    return ChatPrompts(render=chat_render, slot=slot, media=media)


def chat_body(part: dict[str, Any] | None, text: str = "the page") -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if part is not None:
        content.append(part)
    return {"model": "tiny", "messages": [[{"role": "user", "content": content}]], "add_special_tokens": True}


def embedding_reply(vector: list[float], usage: int) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
        "usage": {"prompt_tokens": usage, "total_tokens": usage},
    }


def pair_body(document: Any, query: str = "which page") -> dict[str, Any]:
    return {"model": "tiny", "query": query, "documents": [document]}


def rerank_reply(score: float, usage: int) -> dict[str, Any]:
    return {
        "id": "score-1",
        "model": "tiny",
        "results": [{"index": 0, "relevance_score": score}],
        "usage": {"prompt_tokens": usage, "total_tokens": usage},
    }


def pair_template() -> Any:
    """The product's own pair template (R30): a fixed head, the two content spans, a fixed tail."""

    class _Template:
        @staticmethod
        def render(shape: str, tokenizer: Any, *, query: str, document: str) -> str:
            return f"pair {query} <> {document}"

        @staticmethod
        def adds_special_tokens(shape: str) -> bool:
            return False

    return _Template()


# ---------------------------------------------------------------------------
# A media/chat record replays; only its own content identity and processing do
# ---------------------------------------------------------------------------


def test_a_messages_record_with_media_replays_for_its_content_identity(tmp_path: Path) -> None:
    """One recorded chat request carrying an image replays: the same bytes under the same declared
    processing answer the recorded vector, and the usage count is the engine's (the render plus the media
    tokens)."""
    tokenizer = word_tokenizer()
    part = image_part(b"a page of pixels")
    body = chat_body(part)
    render = chat_render(body["messages"][0], False)
    usage = tokenizer.count(render, add_special_tokens=True) + 7
    corpus = observation_corpus(
        tmp_path / "corpus",
        [
            Exchange(
                0,
                "POST",
                "/v1/embeddings",
                body,
                200,
                {"content-type": "application/json"},
                embedding_reply([1.0, 2.0], usage),
            )
        ],
    )
    emulator = VllmEmulator.from_corpus(corpus, chat_strategy(), tokenizer, FACTS, dim=2)
    assert emulator.unmodelled_records == ()
    answer = emulator.answer("/v1/embeddings", "POST", body)
    assert answer.status_code == 200, answer.json()
    assert answer.headers[SOURCE] == "replayed"
    assert answer.json()["data"][0]["embedding"] == [1.0, 2.0]
    assert answer.json()["usage"]["prompt_tokens"] == usage


def test_another_image_is_a_different_question(tmp_path: Path) -> None:
    """The same conversation text with different image bytes is a different replay key: the surrogate
    answers it, marked -- never another image's recorded vector."""
    tokenizer = word_tokenizer()
    recorded = chat_body(image_part(b"the recorded page"))
    corpus = observation_corpus(
        tmp_path / "corpus",
        [Exchange(0, "POST", "/v1/embeddings", recorded, 200, {}, embedding_reply([1.0, 0.0], 9))],
    )
    emulator = VllmEmulator.from_corpus(corpus, chat_strategy(), tokenizer, FACTS, dim=2)
    other = chat_body(image_part(b"another page"))
    answer = emulator.answer("/v1/embeddings", "POST", other)
    assert answer.status_code == 200 and answer.headers[SOURCE] == "surrogate"
    assert answer.json()["data"][0]["embedding"] != [1.0, 0.0]


def test_the_declared_processing_is_part_of_the_media_key(tmp_path: Path) -> None:
    """The same bytes under another declared processing (another resize, another pin) are a different
    replay key: the media identity carries the processing beside the content hash, so a strategy whose
    declared processing moved never answers from the other processing's observation."""
    part = image_part(b"a page")
    under_p0 = chat_strategy(lambda part: identity_of(part, processing="p0")).prompts(chat_body(part))
    under_p1 = chat_strategy(lambda part: identity_of(part, processing="p1")).prompts(chat_body(part))
    assert under_p0.item_key(0) != under_p1.item_key(0)
    again = chat_strategy(lambda part: identity_of(part, processing="p0")).prompts(chat_body(part))
    assert under_p0.item_key(0) == again.item_key(0)
    assert "p0" in under_p0.item_key(0) and "p1" not in under_p0.item_key(0)


def test_a_chat_batch_keys_every_conversation(tmp_path: Path) -> None:
    """A batch (a list of conversations) yields one prompt per conversation, in order: the first replays,
    the second (another image) answers the surrogate -- and the reply is mixed."""
    tokenizer = word_tokenizer()
    first = {"role": "user", "content": [{"type": "text", "text": "one"}, image_part(b"page one")]}
    second = {"role": "user", "content": [{"type": "text", "text": "two"}, image_part(b"page two")]}
    body = {"model": "tiny", "messages": [[first], [second]], "add_special_tokens": True}
    reply = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
        ],
    }
    corpus = observation_corpus(tmp_path / "corpus", [Exchange(0, "POST", "/v1/embeddings", body, 200, {}, reply)])
    emulator = VllmEmulator.from_corpus(corpus, chat_strategy(), tokenizer, FACTS, dim=2)
    assert emulator.unmodelled_records == ()
    batch = {
        "model": "tiny",
        "messages": [
            [first],
            [{"role": "user", "content": [{"type": "text", "text": "two"}, image_part(b"another page")]}],
        ],
        "add_special_tokens": True,
    }
    answer = emulator.answer("/v1/embeddings", "POST", batch)
    assert answer.status_code == 200 and answer.headers[SOURCE] == "mixed"


def test_a_rerank_media_side_replays_with_the_pair_strategy(tmp_path: Path) -> None:
    """A pointwise rerank whose document is a content-parts object (the media wire) replays: the pair's
    text render plus the document's media identity and the engine's tokens for it."""
    tokenizer = word_tokenizer()
    part = image_part(b"a document page")
    body = pair_body({"content": [part, {"type": "text", "text": "a caption"}]})
    corpus = observation_corpus(
        tmp_path / "corpus", [Exchange(0, "POST", "/rerank", body, 200, {}, rerank_reply(0.75, 12))]
    )
    strategy = PairPrompts(template=pair_template(), tokenizer=tokenizer, media=identity_of)
    emulator = VllmEmulator.from_corpus(corpus, strategy, tokenizer, FACTS, dim=2)
    assert emulator.unmodelled_records == ()
    answer = emulator.answer("/rerank", "POST", body)
    assert answer.status_code == 200 and answer.headers[SOURCE] == "replayed"
    assert answer.json()["results"][0]["relevance_score"] == 0.75


# ---------------------------------------------------------------------------
# An unmodelled record is skipped and named
# ---------------------------------------------------------------------------


def test_an_unmodelled_field_is_skipped_and_named_not_fatal(tmp_path: Path) -> None:
    """One record carrying a field the emulator does not model no longer fails the whole corpus: the corpus
    builds, the record is named in ``unmodelled_records``, the good record still replays, and the
    unmodelled one answers the marked 400."""
    tokenizer = word_tokenizer()
    good = {"model": "tiny", "input": ["the a"]}
    bad = {**good, "truncate_prompt_tokens": 4}
    corpus = observation_corpus(
        tmp_path / "corpus",
        [
            Exchange(0, "POST", "/v1/embeddings", good, 200, {}, embedding_reply([1.0, 0.0], 2)),
            Exchange(1, "POST", "/v1/embeddings", bad, 200, {}, embedding_reply([0.0, 1.0], 2)),
        ],
    )
    emulator = VllmEmulator.from_corpus(corpus, StringsPrompts(), tokenizer, FACTS, dim=2)
    assert len(emulator.unmodelled_records) == 1
    assert "#1" in emulator.unmodelled_records[0] and "truncate_prompt_tokens" in emulator.unmodelled_records[0]
    assert emulator.answer("/v1/embeddings", "POST", good).headers[SOURCE] == "replayed"
    refused = emulator.answer("/v1/embeddings", "POST", bad)
    assert refused.status_code == 400 and refused.headers[SOURCE] == "refused-unmodelled"


def test_a_chat_record_without_a_chat_strategy_is_skipped_and_named(tmp_path: Path) -> None:
    """A ``messages`` record read by a strategy with no chat render is skipped and named -- it is not a
    whole-corpus failure, and a request for it is refused marked (never answered from a text observation)."""
    tokenizer = word_tokenizer()
    good = {"model": "tiny", "input": ["the a"]}
    chat = chat_body(None)
    corpus = observation_corpus(
        tmp_path / "corpus",
        [
            Exchange(0, "POST", "/v1/embeddings", good, 200, {}, embedding_reply([1.0, 0.0], 2)),
            Exchange(1, "POST", "/v1/embeddings", chat, 200, {}, embedding_reply([0.0, 1.0], 4)),
        ],
    )
    emulator = VllmEmulator.from_corpus(corpus, StringsPrompts(), tokenizer, FACTS, dim=2)
    assert len(emulator.unmodelled_records) == 1 and "#1" in emulator.unmodelled_records[0]
    assert emulator.answer("/v1/embeddings", "POST", good).headers[SOURCE] == "replayed"
    refused = emulator.answer("/v1/embeddings", "POST", chat)
    assert refused.status_code == 400 and refused.headers[SOURCE] == "refused-unmodelled"
    assert "chat" in refused.json()["error"]["message"]


def test_a_media_record_without_a_media_model_is_skipped_and_named(tmp_path: Path) -> None:
    """A media part a strategy cannot key is skipped and named (a typed error, not a bare ``ValueError``):
    the rest of the corpus still builds."""
    tokenizer = word_tokenizer()
    body = pair_body({"content": [image_part(b"a page")]})
    corpus = observation_corpus(
        tmp_path / "corpus", [Exchange(0, "POST", "/rerank", body, 200, {}, rerank_reply(0.5, 12))]
    )
    emulator = VllmEmulator.from_corpus(
        corpus, PairPrompts(template=pair_template(), tokenizer=tokenizer), tokenizer, FACTS, dim=2
    )
    assert len(emulator.unmodelled_records) == 1 and "media" in emulator.unmodelled_records[0]
    assert emulator.observations == {}


def test_the_verification_record_names_the_unmodelled_records(tmp_path: Path) -> None:
    """The emulator's manifest is the verification record: it carries ``unmodelled_records`` (empty when
    the model layer covers the whole corpus), so a skipped record is never silent."""
    tokenizer = word_tokenizer()
    body = chat_body(None)
    corpus = observation_corpus(
        tmp_path / "corpus", [Exchange(0, "POST", "/v1/embeddings", body, 200, {}, embedding_reply([1.0], 4))]
    )
    emulator = VllmEmulator.from_corpus(corpus, StringsPrompts(), tokenizer, FACTS, dim=2)
    record = verification_record(corpus, [], verified_at="2026-10-09", emulator=emulator)
    assert record["unmodelled_records"] == list(emulator.unmodelled_records)
    assert record["unmodelled_records"], "the record must name the skipped chat record"


def test_a_request_prompts_dispatch_reads_both_shapes(tmp_path: Path) -> None:
    """A text-shaped recipe that sends its media items as ``messages`` needs both derivations: the dispatch
    routes an ``input`` body to the text strategy and a ``messages`` body to the chat one."""
    strategy = RequestPrompts(StringsPrompts(), chat_strategy())
    text = strategy.prompts({"input": ["the a"]})
    assert text.prompts == ("the a",)
    chat = strategy.prompts(chat_body(None))
    assert isinstance(chat.prompts[0], MediaPrompt)
    assert chat.count(0, word_tokenizer()) == word_tokenizer().count("chat the page", add_special_tokens=False)


def test_the_chat_route_defaults_add_special_tokens_to_false() -> None:
    """vLLM's chat routes default ``add_special_tokens`` to false: a chat body that omits the field and one
    that sends false key the same context (the completion-shaped default would key them apart)."""
    without = {key: value for key, value in chat_body(None).items() if key != "add_special_tokens"}
    omitted, _ = request_context("embeddings", without)
    explicit, _ = request_context("embeddings", {**without, "add_special_tokens": False})
    completion, _ = request_context("embeddings", {"input": ["the a"]})
    assert omitted == explicit == {"add_special_tokens": False}
    assert completion["add_special_tokens"] is True


def test_the_part_order_is_part_of_the_media_key() -> None:
    """The engine places each vision block where its part stands, so ``[text, image]`` and ``[image, text]``
    are different prompts: they must not share a replay key (the product treats the part order as
    behaviour-shaping), even though the text-only render is the same."""
    part = image_part(b"a page")
    text_first = chat_strategy().prompts(
        {"model": "tiny", "messages": [[{"role": "user", "content": [{"type": "text", "text": "A"}, part]}]]}
    )
    image_first = chat_strategy().prompts(
        {"model": "tiny", "messages": [[{"role": "user", "content": [part, {"type": "text", "text": "A"}]}]]}
    )
    assert text_first.item_key(0) != image_first.item_key(0)
    assert "image" in text_first.item_key(0) and "text" in text_first.item_key(0)
    assert text_first.prompts[0].placement == ("text", "image")
    assert image_first.prompts[0].placement == ("image", "text")


def test_a_media_part_the_wiring_cannot_read_is_skipped_and_named(tmp_path: Path) -> None:
    """A 2xx record whose media part the *wiring's* media model cannot read (undecodable bytes, a header that
    states no geometry) is unmodelled, not a raised PIL/assert error: `from_corpus` skips and names it, the
    corpus still builds, and a request for it answers the marked 400 (the contract the lane's tests pin for
    the emulator's own refusals)."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from tests._engines import load_recipe, media_model

    from tests.conftest import RECIPES

    recipe = load_recipe("fixture-vl-embed", root=RECIPES)
    tokenizer = tokenizer_of(recipe)
    corrupt = base64.b64encode(b"not a png at all").decode("ascii")
    body = chat_body({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{corrupt}"}})
    corpus = observation_corpus(
        tmp_path / "corpus", [Exchange(0, "POST", "/v1/embeddings", body, 200, {}, embedding_reply([1.0], 4))]
    )
    emulator = VllmEmulator.from_corpus(
        corpus, chat_strategy(media=media_model(recipe, tokenizer)), tokenizer, FACTS, dim=2
    )
    assert len(emulator.unmodelled_records) == 1 and "#0" in emulator.unmodelled_records[0]
    refused = emulator.answer("/v1/embeddings", "POST", body)
    assert refused.status_code == 400 and refused.headers[SOURCE] == "refused-unmodelled"


def test_the_wiring_builds_a_media_model_for_a_media_recipe() -> None:
    """The test wiring's strategy for a media recipe (the fixture vision embedder) keys a sent image by its
    bytes' SHA-256 under the recipe's declared processing and counts what the engine adds for it -- the
    product's own media count, not a re-derivation."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from rcp_ndcg_test.observe.media_set import image_entry
    from tests._engines import load_recipe, prompt_strategy

    from tests.conftest import RECIPES

    recipe = load_recipe("fixture-vl-embed", root=RECIPES)
    tokenizer = tokenizer_of(recipe)
    entry = image_entry("icon", 64, 64)
    body = {
        "model": recipe.id,
        "messages": [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": entry["uri"]}}]}]],
        "add_special_tokens": True,
        "add_generation_prompt": True,
    }
    prompt_set = prompt_strategy(recipe, tokenizer).prompts(body)
    prompt = prompt_set.prompts[0]
    assert isinstance(prompt, object) and getattr(prompt, "media", ())
    identity = prompt.media[0]
    assert identity.kind == "image" and identity.sha256 == entry["sha256"]
    assert "qwen2_vl" in identity.processing and "max_px" in identity.processing
    assert prompt.media_tokens > 0
    assert prompt_set.count(0, tokenizer) == tokenizer.count(prompt.text, add_special_tokens=True) + prompt.media_tokens
