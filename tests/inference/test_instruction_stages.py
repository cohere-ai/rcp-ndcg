"""The task instruction at the formatting stage (decision 33): the recipe's mode places it, once.

``Dataset.task_instruction`` is one instruction for the whole task. Where the recipe's template declares
an ``instruction`` span, the span carries it (the engine renders it); otherwise the generic default
prefixes the query side (``Task: <instruction>\\nQuery: <text>``); ``instruction: none`` sends none. The
per-query instruction (``Query.instruction``, the data's own) is appended by the *data* layer -- the
caller hands the client the query text mteb's dataloader would read -- so the client never folds it twice.

The endpoint must DECLARE the policy: ``instruction: None`` means undeclared, and a task instruction
arriving at an endpoint that never chose one is refused (naming ``fold``/``none``), never silently applied
or dropped -- a recipe that declares nothing must not have its text changed by a dataset it never met.
"""

from __future__ import annotations

from typing import Any

import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.templates import Segment, TemplateSpec
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.clients.embed import EmbeddingClient
from rcp_ndcg.inference.config import EmbeddingEndpoint
from rcp_ndcg.inference.types import Call, EncodeRole, Reply
from tests.inference._embed import FakeSender

INSTRUCTION = "Given a claim, find documents that refute the claim"


def _handler(call: Call) -> Reply:
    batch = [text for text in call.json["input"]]
    return Reply(200, {"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(len(batch))]}, {})


def _client(sender: Any, **overrides: Any) -> EmbeddingClient:
    settings: dict[str, Any] = {"base_url": "http://engine:8000/v1", "model": "m", "instruction": "fold"}
    settings.update(overrides)
    return EmbeddingClient(EmbeddingEndpoint(**settings), sender=sender)


def _sent(sender: FakeSender) -> list[str]:
    return [text for call in sender.calls for text in call.json["input"]]


class TestTheTaskInstruction:
    def test_the_generic_default_prefixes_the_query_side(self, tokenizer_json: str) -> None:
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64)

        client.encode([Content.from_text("find docs")], EncodeRole.QUERY, instruction=INSTRUCTION)

        assert _sent(sender) == [f"Task: {INSTRUCTION}\nQuery: find docs"]

    def test_the_document_side_is_not_prefixed(self, tokenizer_json: str) -> None:
        """The generic default is the query's frame: a document side is sent as it is."""
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64)

        client.encode([Content.from_text("a document")], EncodeRole.DOCUMENT)

        assert _sent(sender) == ["a document"]

    def test_a_document_side_instruction_without_a_span_is_refused(self, tokenizer_json: str) -> None:
        """The generic default frames the query side only: a document-side task instruction with nowhere to
        go is refused, never silently dropped."""
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64)

        with pytest.raises(ConfigError, match="document side carries a task instruction"):
            client.encode([Content.from_text("a document")], EncodeRole.DOCUMENT, instruction=INSTRUCTION)
        assert sender.calls == []

    def test_instruction_none_sends_no_instruction(self, tokenizer_json: str) -> None:
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64, instruction="none")

        client.encode([Content.from_text("find docs")], EncodeRole.QUERY, instruction=INSTRUCTION)

        assert _sent(sender) == ["find docs"]

    def test_an_undeclared_instruction_policy_is_refused(self, tokenizer_json: str) -> None:
        """``None`` means UNDECLARED, never ``fold``: a task instruction arriving at an endpoint that never
        chose a policy is refused with the two choices named, instead of silently changing the text of a
        recipe that declares nothing (AGENTS.md: nothing is defaulted silently)."""
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64, instruction=None)

        with pytest.raises(ConfigError, match="declares no instruction policy"):
            client.encode([Content.from_text("find docs")], EncodeRole.QUERY, instruction=INSTRUCTION)
        assert sender.calls == []

    def test_a_dataset_without_an_instruction_needs_no_declaration(self, tokenizer_json: str) -> None:
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64, instruction=None)

        client.encode([Content.from_text("find docs")], EncodeRole.QUERY)

        assert _sent(sender) == ["find docs"]

    def test_the_recipe_template_places_the_instruction_in_its_span(self, tokenizer_json: str) -> None:
        """A template with an ``instruction`` span takes the instruction there (the fit renders it), and the
        client does NOT also prefix it: one placement, never two."""
        sender = FakeSender(_handler)
        client = _client(
            sender,
            tokenizer=tokenizer_json,
            max_tokens=128,
            template=TemplateSpec(
                query=(
                    Segment(fixed="Instruct: "),
                    Segment(content="instruction"),
                    Segment(fixed="\nQuery: "),
                    Segment(content="query"),
                )
            ),
        )

        client.encode([Content.from_text("find docs")], EncodeRole.QUERY, instruction=INSTRUCTION)

        assert _sent(sender) == [f"Instruct: {INSTRUCTION}\nQuery: find docs"]

    def test_the_per_query_instruction_the_caller_appended_is_not_doubled(self, tokenizer_json: str) -> None:
        """The caller hands the client the query text the data layer formatted (mteb's append); the client
        adds only the task frame."""
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64)

        client.encode([Content.from_text("find docs about turtles")], EncodeRole.QUERY, instruction=INSTRUCTION)

        assert _sent(sender) == [f"Task: {INSTRUCTION}\nQuery: find docs about turtles"]

    def test_the_side_prompt_stays_attached_to_the_content(self, tokenizer_json: str) -> None:
        """The model's own prefix stays immediately before the text it prefixes; the generic task frame wraps
        the whole (prompt + text)."""
        sender = FakeSender(_handler)
        client = _client(sender, tokenizer=tokenizer_json, max_tokens=64, query_prompt="query: ")

        client.encode([Content.from_text("find docs")], EncodeRole.QUERY, instruction=INSTRUCTION)

        assert _sent(sender) == [f"Task: {INSTRUCTION}\nQuery: query: find docs"]


class TestTheConfigDeclaration:
    def test_the_instruction_field_is_content_and_optional(self, tokenizer_json: str) -> None:
        """The field enters the run identity when declared and is absent (``None``) otherwise, so a config
        that never declares it keeps the identity it had."""
        from rcp_ndcg.support.identity import identity_payload

        undeclared = EmbeddingEndpoint(model="m", tokenizer=tokenizer_json, max_tokens=64)
        assert "instruction" not in identity_payload(undeclared)
        declared = EmbeddingEndpoint(model="m", tokenizer=tokenizer_json, max_tokens=64, instruction="none")
        assert identity_payload(declared)["instruction"] == "none"
