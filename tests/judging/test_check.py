"""``judge check``: the fixed-window conformance probe (structured output, parse, reasoning channel)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from rcp_ndcg.judging import JudgeConfig, check_judge
from rcp_ndcg.judging.check import _probe_stage
from rcp_ndcg.judging.client import JudgeClient

# A tournament answer in the shape the parser reads (a ranking of the window's two documents).
_TOURNAMENT = json.dumps({"ranking": [1, 2], "scores": {"1": 1.0, "2": 0.0}, "reasoning": "probe"})
# A rubric answer for the two probe documents (all criteria zero).
_RUBRIC = json.dumps(
    {
        "documents": [
            {"rank": 1, "criteria": {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}, "reasoning": "probe"},
            {"rank": 2, "criteria": {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}, "reasoning": "probe"},
        ]
    }
)


def _answer(content: str, *, reasoning: str | None = None, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": {"message": content}})
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return httpx.Response(
        status,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


class Endpoint:
    """An OpenAI-compatible endpoint answering the probe from a script of responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0) if self.responses else _answer(_TOURNAMENT)


def _probe(endpoint: Endpoint, **config: object):
    settings = {"base_url": "http://judge.test/v1", "model": "m", "max_retries": 0, **config}
    client = JudgeClient(JudgeConfig(**settings), httpx_transport=httpx.MockTransport(endpoint))
    return asyncio.run(_probe_stage(client, "tournament")), client


def test_the_offline_judge_passes_the_probe() -> None:
    report = check_judge(JudgeConfig.fake(0))
    assert report.ok
    assert [check.stage for check in report.checks] == ["tournament", "rubric"]
    assert all(check.accepted and check.parsed for check in report.checks)
    assert report.usage.requests == 2
    # the fake judge declares no decoding: no schema is sent, and no reasoning channel comes back
    assert not any(check.schema_sent for check in report.checks)
    assert all(check.reasoning_channel == "absent" for check in report.checks)


def test_the_schema_is_sent_when_the_judge_decodes_json() -> None:
    config = JudgeConfig.fake(0).model_copy(update={"decoding": "json_schema"})
    report = check_judge(config)
    assert report.ok
    assert all(check.schema_sent for check in report.checks)


def test_a_reasoning_channel_beside_the_answer_is_reported_separated() -> None:
    check, client = _probe(Endpoint(_answer(_TOURNAMENT, reasoning="thinking about it")))
    assert check.ok
    assert check.reasoning_channel == "separated"
    assert check.detail == ""
    assert client.usage.output_tokens == 5


def test_a_missing_reasoning_channel_carries_the_advisory() -> None:
    check, _ = _probe(Endpoint(_answer(_TOURNAMENT)), decoding="json_schema")
    assert check.ok
    assert check.schema_sent
    assert check.reasoning_channel == "absent"
    assert "reasoning parser" in check.detail


def test_a_refused_schema_is_reported_not_raised() -> None:
    refusal = _answer("response_format json_schema is not supported", status=400)
    check, _ = _probe(Endpoint(refusal), decoding="json_schema")
    assert not check.ok
    assert not check.accepted and not check.parsed
    assert "CapabilityError" in check.detail


def test_an_answer_that_does_not_parse_is_reported_not_raised() -> None:
    check, _ = _probe(Endpoint(_answer("not json at all")))
    assert not check.ok
    assert check.accepted and not check.parsed
    assert check.detail


def test_the_report_says_which_recipe_the_config_came_from() -> None:
    config = JudgeConfig.fake(0).model_copy(update={"recipe": "some-recipe"})
    report = check_judge(config)
    assert report.recipe == "some-recipe"
    assert report.model == "fake"


def test_a_config_without_a_url_is_refused_by_the_client() -> None:
    from rcp_ndcg.errors import ConfigError

    with pytest.raises(ConfigError, match="no base_url"):
        check_judge(JudgeConfig(model="m"))
