"""The estimate of a judging pass before it runs: calls, tokens and wall time."""

from __future__ import annotations

from pathlib import Path

import pytest
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.judging import JudgeConfig, RubricSchedule, TournamentSchedule, estimate, judge
from rcp_ndcg.judging.client import Completion, CompletionInput
from rcp_ndcg.judging.judging import CHAT_TEMPLATE_TOKENS
from rcp_ndcg.judging.tokens import APPROX_TOKENS_PER_IMAGE, APPROXIMATION_NOTE, IMAGE_APPROXIMATION_NOTE
from rcp_ndcg.testing import FakeJudge, tiny_rows


def test_the_estimate_counts_the_schedules_calls() -> None:
    rows, _ = tiny_rows()
    judge = JudgeConfig(base_url="http://h/v1", model="m", concurrency=4)
    result = estimate(rows, None, judge)
    pools = [len(row.doc_ids) for row in rows]
    assert result.stages["tournament"].calls == sum(TournamentSchedule().calls_per_query(n) for n in pools)
    assert result.stages["rubric"].calls == sum(RubricSchedule().calls_per_query(n) for n in pools)
    assert result.calls == result.stages["tournament"].calls + result.stages["rubric"].calls
    assert result.input_tokens > 0 and result.wall_s > 0


def test_an_estimate_reports_calls_and_tokens_and_no_usd() -> None:
    """Judges run on the user's own engines: an estimate counts calls, tokens and time, never dollars."""
    rows, _ = tiny_rows()
    projected = estimate(rows, None, JudgeConfig(base_url="http://h/v1", model="m"), stages=["rubric"])
    payload = projected.model_dump(mode="json", by_alias=True)
    assert set(payload) == {
        "schema", "calls", "input_tokens", "output_tokens", "wall_s", "stages", "input_token_count", "assumptions",
    }  # fmt: skip
    assert set(payload["stages"]["rubric"]) == {"calls", "input_tokens", "output_tokens"}
    assert projected.calls > 0 and projected.input_tokens > 0 and projected.output_tokens > 0
    assert not any("usd" in note.lower() or "price" in note.lower() for note in projected.assumptions)
    assert "price" not in JudgeConfig.model_fields


def test_a_context_window_caps_the_text_counted(word_tokenizer_file: Path) -> None:
    rows, _ = tiny_rows()
    long_rows = [row.model_copy(update={"docs": [(text + " ") * 500 for text in row.docs]}) for row in rows]
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m", tokenizer=str(word_tokenizer_file))
    whole = estimate(long_rows, None, judge_cfg, stages=["rubric"])
    capped = estimate(long_rows, None, judge_cfg.model_copy(update={"context_tokens": 4_096}), stages=["rubric"])
    assert capped.input_tokens < whole.input_tokens
    assert capped.input_tokens <= capped.calls * 4_096


def test_without_a_tokenizer_the_context_cuts_nothing_and_the_count_is_labelled_approximate() -> None:
    rows, _ = tiny_rows()
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m")
    whole = estimate(rows, None, judge_cfg, stages=["rubric"])
    with_context = estimate(rows, None, judge_cfg.model_copy(update={"context_tokens": 4_096}), stages=["rubric"])
    assert with_context.input_tokens == whole.input_tokens
    assert whole.input_token_count == "approximate" and APPROXIMATION_NOTE in whole.assumptions[0]


class _Recording(FakeJudge):
    """The fake judge, keeping every request it answers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[CompletionInput] = []

    async def complete(self, request: CompletionInput) -> Completion:
        self.requests.append(request)
        return await super().complete(request)


def test_with_a_tokenizer_the_estimate_counts_the_prompts_exactly(tmp_path: Path, word_tokenizer_file: Path) -> None:
    """Every window shows all three documents, so each call's prompt is the estimate's per-call count exactly."""
    docs = {"a": "the evidence & the answer, " * 20, "b": "one two three " * 30, "c": "a query of the page"}
    rows = [RankingExample(query_id="q", query="the query", doc_ids=list(docs), docs=list(docs.values()))]
    schedule = RubricSchedule(window=3, placements_per_doc=4.0)
    fake = _Recording(lambda text: 0.0)
    fake.config = fake.config.model_copy(update={"tokenizer": str(word_tokenizer_file)})

    projected = estimate(rows, None, fake.config, stages=["rubric"], schedules={"rubric": schedule})
    judge(rows, None, fake, stage="rubric", out=tmp_path, schedule=schedule)

    words = load_tokenizer(str(word_tokenizer_file))
    assert projected.calls == len(fake.requests)
    assert projected.input_tokens == sum(words.count(r.user_prompt) + CHAT_TEMPLATE_TOKENS for r in fake.requests)
    assert projected.input_token_count == "exact" and str(word_tokenizer_file) in projected.assumptions[0]


def _pages(images_per_page: int) -> list[RankingExample]:
    def page(index: int) -> Content:
        refs = [ImagePart(ref=MediaRef(uri=f"p{index}-{k}.png")) for k in range(images_per_page)]
        return Content.from_parts([TextPart(text=f"page {index}"), *refs])

    return [RankingExample(query_id="q", query="q", doc_ids=["p0", "p1", "p2"], contents=[page(i) for i in range(3)])]


def test_a_hosted_judges_images_are_approximated_and_labelled() -> None:
    """No image_processor: the engine sizes the images, so their tokens are an assumption, stated as one."""
    from rcp_ndcg.errors import RcpNdcgWarning

    hosted = JudgeConfig(base_url="http://h/v1", model="m", max_images=6, context_tokens=272_000)
    with pytest.warns(RcpNdcgWarning, match="image tokens are approximate") as warned:
        one, two = (estimate(_pages(n), None, hosted, stages=["rubric"]) for n in (1, 2))
    window = min(RubricSchedule.for_modality("image").window, 3)
    assert two.input_tokens - one.input_tokens == two.calls * window * APPROX_TOKENS_PER_IMAGE
    assert f"image tokens {IMAGE_APPROXIMATION_NOTE}" in two.assumptions
    assert {w.message.code for w in warned} == {"APPROXIMATE_IMAGE_TOKENS"}  # type: ignore[union-attr]


def test_the_assumptions_name_only_the_stages_estimated() -> None:
    rows, _ = tiny_rows()
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m")
    rubric = " ".join(estimate(rows, None, judge_cfg, stages=["rubric"]).assumptions)
    both = " ".join(estimate(rows, None, judge_cfg).assumptions)
    assert "tournament" not in rubric and "rubric window" in rubric
    assert "tournament window" in both and "adaptive tournament phase" in both


def test_a_text_budget_over_uncountable_images_is_refused_as_the_pass_refuses_it(word_tokenizer_file: Path) -> None:
    budgeted = JudgeConfig(
        base_url="http://h/v1", model="m", max_images=6, context_tokens=272_000, tokenizer=str(word_tokenizer_file)
    )
    with pytest.raises(ConfigError, match="native-size image policy"):
        estimate(_pages(1), None, budgeted, stages=["rubric"])


@pytest.mark.parametrize("stage", ["tournament", "rubric"])
def test_the_estimate_of_a_subset_counts_the_calls_the_pass_makes(stage: str, tmp_path) -> None:
    from rcp_ndcg.judging import judge
    from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, FakeJudge

    rows, ability = tiny_rows()
    fake = FakeJudge(lambda text: ability[text.split()[-1]])
    pools = {row.id: list(row.doc_ids) for row in rows if row.id == "q2"}
    docs = {"q2": ["q2-d10", "q2-d11", "q2-d03"]}
    schedule = TINY_TOURNAMENT if stage == "tournament" else TINY_RUBRIC
    projected = estimate(rows, pools, fake.config, stages=(stage,), docs=docs, schedules={stage: schedule})
    judge(rows, pools, fake, stage=stage, out=tmp_path, docs=docs, schedule=schedule)
    assert projected.calls == fake.usage.requests


def test_the_character_approximation_is_labelled_a_heuristic_not_a_bound() -> None:
    assert "heuristic" in APPROXIMATION_NOTE and "bound" not in APPROXIMATION_NOTE


def test_the_output_cap_holds_at_max_output_tokens() -> None:
    """The estimate's per-call output is at most ``max_output_tokens`` when the window's per-document share
    would exceed it (the cap is not a floor): the estimate must not drift under a small declared budget."""
    rows, _ = tiny_rows()
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m")
    whole = estimate(rows, None, judge_cfg, stages=["rubric"])
    capped = estimate(rows, None, judge_cfg.model_copy(update={"max_output_tokens": 8}), stages=["rubric"])
    assert capped.output_tokens == capped.calls * 8, "every call is capped at the declared budget"
    assert whole.output_tokens > capped.output_tokens


def test_the_estimate_passes_the_judges_tokenizer_to_the_media_count(
    monkeypatch: pytest.MonkeyPatch, word_tokenizer_file: Path
) -> None:
    """An fps container's timestamp lines are tokenizer-dependent: the estimate must pass the judge's
    tokenizer into the media count (dropping the pass silently reverts to the family's bound)."""
    from rcp_ndcg.judging import judging as judging_module

    seen: dict[str, object] = {}
    real = judging_module._media_tokens

    def recorder(contents, preprocessing, **kwargs):
        seen.update(kwargs)
        return real(contents, preprocessing, **kwargs)

    monkeypatch.setattr(judging_module, "_media_tokens", recorder)
    rows, _ = tiny_rows()
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m", tokenizer=str(word_tokenizer_file), context_tokens=4096)

    estimate(rows, None, judge_cfg, stages=["rubric"])

    assert seen.get("tokenizer") is not None, "the estimate's media count must use the judge's tokenizer"
