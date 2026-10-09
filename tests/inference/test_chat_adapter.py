"""The ``openai_chat`` wire: the request body it builds, the answer it reads, the refusals it maps, the usage.

The judge's request is the OpenAI chat-completions shape exactly as the OpenAI SDK used to build it, with
``extra_body`` merged at the top level; the answer's text, reasoning channel, ``finish_reason`` and usage
come back as a :class:`~rcp_ndcg.inference.types.Completion`; and the endpoint's refusals map onto typed
errors -- media counts and the answer schema are capabilities, everything else one request's refusal.
"""

from __future__ import annotations

from typing import Any

import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg.errors import CapabilityError, RequestRejectedError
from rcp_ndcg.inference.adapters.chat import (
    REASONING_KEYS,
    REASONING_WATCH,
    OpenAIChat,
    build_messages,
    media_counts,
)
from rcp_ndcg.inference.types import Reply, TokenCount
from rcp_ndcg.judging import JudgeConfig
from rcp_ndcg.judging.client import CompletionInput

SCHEMA = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}
PAGE = Content.from_parts([ImagePart(ref=MediaRef(uri="data:image/png;base64,AAAA", mime="image/png"))])

CONFIG = JudgeConfig(base_url="http://judge.test/v1", model="m")


def _reply(body: Any, status: int = 200) -> Reply:
    return Reply(status=status, body=body, headers={}, url="http://judge.test/v1")


def _answer(content: str = "ok", finish_reason: str | None = "stop", **fields: Any) -> dict:
    """One chat-completion body; ``finish_reason`` lands on the choice (``None`` removes it), the rest on the
    top level."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    choice: dict[str, Any] = {"index": 0, "message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [choice],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100},
    }
    body.update(fields)
    return body


class TestTheRequest:
    def test_a_free_answer_sends_the_prompt_and_no_sampling_settings(self) -> None:
        (call,) = OpenAIChat(CONFIG).calls(CompletionInput(user_prompt="judge this"), model="m")
        assert (call.method, call.path) == ("POST", "/chat/completions")
        assert call.json == {"model": "m", "messages": [{"role": "user", "content": "judge this"}]}

    def test_the_sampling_settings_are_sent_and_extra_body_is_merged_at_the_top_level(self) -> None:
        """``max_tokens`` is refused by the OpenAI API for reasoning models; vLLM and SGLang read either."""
        config = JudgeConfig(
            base_url="http://judge.test/v1",
            model="m",
            temperature=0.3,
            max_output_tokens=64,
            extra_body={"reasoning_effort": "low", "top_k": 1},
        )
        (call,) = OpenAIChat(config).calls(CompletionInput(user_prompt="judge this"), model="m")
        assert call.json == {
            "model": "m",
            "messages": [{"role": "user", "content": "judge this"}],
            "temperature": 0.3,
            "max_completion_tokens": 64,
            "reasoning_effort": "low",
            "top_k": 1,
        }

    def test_a_json_schema_answer_carries_the_response_format(self) -> None:
        (call,) = OpenAIChat(CONFIG).calls(CompletionInput(user_prompt="judge this", response_format=SCHEMA), model="m")
        assert call.json["response_format"] == SCHEMA
        (free,) = OpenAIChat(CONFIG).calls(CompletionInput(user_prompt="judge this"), model="m")
        assert "response_format" not in free.json


class TestTheMediaGate:
    def test_images_to_a_text_judge_are_refused_before_the_request_is_built(self) -> None:
        with pytest.raises(CapabilityError, match="images"):
            OpenAIChat(CONFIG).calls(CompletionInput(user_prompt="j", user_content=PAGE), model="m")

    def test_a_window_over_the_declared_limits_is_refused_before_sending(self) -> None:
        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_images=1)
        with pytest.raises(CapabilityError, match="max_images"):
            OpenAIChat(config).calls(
                CompletionInput(user_prompt="j", user_content=Content.from_parts([*PAGE.parts, *PAGE.parts])),
                model="m",
            )

    def test_the_media_counts_count_frames_as_images_and_containers_as_videos(self) -> None:
        assert media_counts(None) == (0, 0)
        assert media_counts(PAGE) == (1, 0)


class TestTheAnswer:
    @pytest.mark.parametrize("key", REASONING_KEYS)
    def test_the_reasoning_channel_is_read_from_either_key(self, key: str) -> None:
        body = _answer()
        body["choices"][0]["message"][key] = "I compare the documents."
        completion = OpenAIChat(CONFIG).interpret(CompletionInput(user_prompt="j"), [_reply(body)])
        assert completion.reasoning == "I compare the documents." and completion.response == "ok"

    def test_finish_reason_is_passed_through_as_it_comes(self) -> None:
        aborted = OpenAIChat(CONFIG).interpret(
            CompletionInput(user_prompt="j"), [_reply(_answer("ok", finish_reason="abort"))]
        )
        assert aborted.finish_reason == "abort"
        missing = OpenAIChat(CONFIG).interpret(
            CompletionInput(user_prompt="j"), [_reply(_answer("ok", finish_reason=None))]
        )
        assert missing.finish_reason is None

    def test_the_tokens_cross_as_the_endpoint_reported_them(self) -> None:
        completion = OpenAIChat(CONFIG).interpret(CompletionInput(user_prompt="j"), [_reply(_answer())])
        assert (completion.input_tokens, completion.output_tokens) == (1_000, 100)
        body = _answer()
        del body["usage"]
        quiet = OpenAIChat(CONFIG).interpret(CompletionInput(user_prompt="j"), [_reply(body)])
        assert quiet.input_tokens is None and quiet.output_tokens is None

    def test_usage_reads_the_tokens_of_one_reply(self) -> None:
        adapter = OpenAIChat(CONFIG)
        assert adapter.usage(_reply(_answer())) == TokenCount(input_tokens=1_000, output_tokens=100)
        body = _answer()
        del body["usage"]
        assert adapter.usage(_reply(body)) is None

    def test_the_system_fingerprint_of_one_answer_is_read(self) -> None:
        adapter = OpenAIChat(CONFIG)
        assert adapter.fingerprint(_reply(_answer(system_fingerprint="vllm-0.30.0"))) == "vllm-0.30.0"
        assert adapter.fingerprint(_reply(_answer())) is None
        assert adapter.fingerprint(_reply(_answer(system_fingerprint=""))) is None


class TestTheRefusals:
    @pytest.mark.parametrize(
        ("message", "kind"),
        [
            ("At most 4 image(s) may be provided in one prompt.", "image"),
            ("Image count 12 exceeds limit 10 per request.", "image"),
            ("Too many videos in the request", "video"),
        ],
    )
    def test_a_refused_media_count_is_a_capability_error_naming_the_servers_limit(
        self, message: str, kind: str
    ) -> None:
        with pytest.raises(CapabilityError, match="refused the number of") as caught:
            OpenAIChat(CONFIG).interpret(
                CompletionInput(user_prompt="j"), [_reply({"error": {"message": message}}, 400)]
            )
        assert "per-request media limit" in (caught.value.hint or "") and "judges.md" in caught.value.hint
        assert caught.value.details == {"kind": kind, "status": 400}

    def test_a_pixel_refusal_is_not_a_media_count_refusal(self) -> None:
        body = {"error": {"message": "Image dimensions 9000x9000 exceed the limit"}}
        with pytest.raises(RequestRejectedError, match="HTTP 400"):
            OpenAIChat(CONFIG).interpret(CompletionInput(user_prompt="j"), [_reply(body, 400)])

    def test_a_refused_answer_schema_is_a_capability_error_only_when_one_was_sent(self) -> None:
        adapter = OpenAIChat(CONFIG)
        body = {"error": {"message": "response_format json_schema is not supported"}}
        request = CompletionInput(user_prompt="j", response_format=SCHEMA)
        with pytest.raises(CapabilityError, match="refused the answer schema") as caught:
            adapter.interpret(request, [_reply(body, 400)])
        assert "decoding: free" in (caught.value.hint or "")
        with pytest.raises(RequestRejectedError):  # the same words without a schema sent: an ordinary refusal
            adapter.interpret(CompletionInput(user_prompt="j"), [_reply(body, 400)])

    @pytest.mark.parametrize("status", [400, 402, 409, 413, 418, 422])
    def test_any_other_refusal_is_one_request_error_naming_the_status(self, status: int) -> None:
        with pytest.raises(RequestRejectedError, match=f"HTTP {status}"):
            OpenAIChat(CONFIG).interpret(
                CompletionInput(user_prompt="j"), [_reply({"error": {"message": "no"}}, status)]
            )

    def test_an_answer_with_no_choices_is_a_refused_request(self) -> None:
        adapter = OpenAIChat(CONFIG)
        with pytest.raises(RequestRejectedError, match="no choices"):
            adapter.interpret(CompletionInput(user_prompt="j"), [_reply({"id": "x", "choices": []})])
        with pytest.raises(RequestRejectedError, match="no choices"):
            adapter.interpret(CompletionInput(user_prompt="j"), [_reply({"id": "x"})])

    def test_a_body_that_is_not_a_chat_completion_is_a_refused_request(self) -> None:
        adapter = OpenAIChat(CONFIG)
        with pytest.raises(RequestRejectedError, match="not a chat completion"):
            adapter.interpret(CompletionInput(user_prompt="j"), [_reply(b"plain bytes")])

    def test_the_error_text_is_found_wherever_the_endpoint_nested_it(self) -> None:
        body = {"detail": [{"msg": "Too many videos in the request"}]}
        with pytest.raises(CapabilityError, match="refused the number of videos"):
            OpenAIChat(CONFIG).interpret(CompletionInput(user_prompt="j"), [_reply(body, 422)])


class TestTheReasoningWatch:
    SCHEMA_REQUEST = CompletionInput(user_prompt="judge", response_format=SCHEMA)

    @pytest.mark.parametrize(("reasoning", "warned"), [(None, 1), ("", 1), ("I compare the documents.", 0)])
    def test_answers_without_reasoning_under_a_schema_warn_once(
        self, reasoning: str | None, warned: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = OpenAIChat(CONFIG)
        body = _answer('{"ranking": [1]}')
        if reasoning is not None:
            body["choices"][0]["message"]["reasoning_content"] = reasoning
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            for _ in range(3 * REASONING_WATCH):
                adapter.interpret(self.SCHEMA_REQUEST, [_reply(body)])
        assert sum("reasoning parser" in record.getMessage() for record in caplog.records) == warned

    def test_free_decoding_is_not_watched_for_reasoning(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = OpenAIChat(CONFIG)
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            for _ in range(2 * REASONING_WATCH):
                adapter.interpret(CompletionInput(user_prompt="judge"), [_reply(_answer())])
        assert not any("reasoning parser" in record.getMessage() for record in caplog.records)


def test_messages_lower_a_text_prompt_to_a_string_and_refuse_nothing() -> None:
    assert build_messages(CompletionInput(user_prompt="hi")) == [{"role": "user", "content": "hi"}]
