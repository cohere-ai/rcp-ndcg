"""The offline judge: deterministic, answering the real prompts in the JSON the real parsers read."""

from __future__ import annotations

import asyncio
import json

from rcp_ndcg_core.content import Content

from rcp_ndcg.llm import load_prompt
from rcp_ndcg.llm._parsing.listwise import parse_calibrated_listwise
from rcp_ndcg.llm._parsing.rubric import parse_rubric_criteria
from rcp_ndcg.llm.client import Completion, CompletionInput
from rcp_ndcg.testing import FakeJudge

ABILITY = {"low": -2.0, "mid": 0.0, "high": 2.0}


def _ask(judge: FakeJudge, stage: str, texts: list[str]) -> Completion:
    template = load_prompt(stage).template(with_num_documents=stage == "tournament")
    prompt = template.resolve(query="q", documents=[Content.from_text(text) for text in texts])
    return asyncio.run(judge.complete(CompletionInput(user_prompt=prompt)))


def _judge(**kwargs) -> FakeJudge:
    return FakeJudge(ABILITY, **kwargs)


def test_the_same_window_gets_the_same_answer_and_the_seed_changes_it() -> None:
    assert _ask(_judge(), "rubric", ["mid", "high", "low"]) == _ask(_judge(), "rubric", ["mid", "high", "low"])
    windows = [[a, b] for a in ABILITY for b in ABILITY if a != b]
    assert any(_ask(_judge(seed=1), "rubric", w) != _ask(_judge(seed=2), "rubric", w) for w in windows)


def test_tournament_answers_parse_and_rank_by_ability() -> None:
    completion = _ask(_judge(), "tournament", ["low", "high", "mid"])
    ranking, _ = parse_calibrated_listwise("q", completion, ["low", "high", "mid"])
    assert ranking == ["high", "mid", "low"]


def test_rubric_answers_parse_and_pass_rates_follow_ability_and_severity() -> None:
    def pass_rate(judge: FakeJudge, text: str) -> float:
        windows = [[text, a, b] for a in ABILITY for b in ABILITY]
        verdicts = [
            parse_rubric_criteria("q", _ask(judge, "rubric", w), [f"d{i}" for i in range(3)], 5)["d0"] for w in windows
        ]
        return sum(sum(v.values()) for v in verdicts) / (5 * len(verdicts))

    strict, lenient = _judge(), _judge(severity=-1.0)
    assert pass_rate(strict, "low") < pass_rate(strict, "mid") < pass_rate(strict, "high")
    assert pass_rate(lenient, "mid") > pass_rate(strict, "mid")


def test_the_answer_is_json_and_the_usage_is_counted() -> None:
    judge = _judge()
    json.loads(_ask(judge, "tournament", ["low", "mid"]).response)
    assert judge.usage.requests == 1 and judge.usage.input_tokens > 0


def test_the_offline_judge_of_a_config_is_a_real_client_over_the_fake_route() -> None:
    from rcp_ndcg.llm import JudgeClient, JudgeConfig

    client = JudgeClient.from_config(JudgeConfig.fake(seed=3))
    assert type(client) is JudgeClient and client.model == "fake"
    prompt = '<documents>\n<doc id="doc_1">\ntiny document a\n</doc>\n</documents>'
    completion = asyncio.run(client.complete(CompletionInput(user_prompt=prompt)))
    assert json.loads(completion.response)["scores"] == {"1": -1.1676}
    assert completion.finish_reason == "stop" and client.usage.requests == 1


def test_the_fake_judge_of_a_config_keeps_every_field_of_it() -> None:
    from rcp_ndcg.llm import JudgeConfig

    config = JudgeConfig.fake(3).model_copy(
        update={
            "model": "fake-b",
            "tokenizer": "/x/tokenizer.json",
            "context_tokens": 4096,
            "max_images": 4,
            "decoding": "json_schema",
            "concurrency": 2,
        }  # fmt: skip
    )
    client = FakeJudge.from_config(config)
    assert client.seed == 3 and client.model == "fake-b"
    assert client.config == config
