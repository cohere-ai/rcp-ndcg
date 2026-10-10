"""judge(): one judging path over an append-only store, with the fake judge standing in for a model."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import httpx
import pytest
from rcp_ndcg_core.records import RankingExample
from rcp_ndcg_core.schemas import JudgementSet

from rcp_ndcg.data.preprocess import ChunkPolicy, Preprocessing, TextPolicy, chunk_ranking_example
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError, IdentityError
from rcp_ndcg.judging import JudgeClient, JudgeConfig, JudgementStore, RubricSchedule, judge
from rcp_ndcg.judging._fake import _DOC_BLOCK, _answer_text
from rcp_ndcg.judging._parsing.schema import response_format
from rcp_ndcg.judging.client import BackendUnavailableError, Completion, CompletionInput
from rcp_ndcg.judging.judging import CHAT_TEMPLATE_TOKENS, MAX_ATTEMPTS, prompt_overhead_tokens, window_tokens
from rcp_ndcg.judging.prompts import load_prompt
from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, FakeJudge, tiny_rows

ROWS, ABILITY = tiny_rows()


def _fake(**kwargs) -> FakeJudge:
    return FakeJudge(lambda text: ABILITY[text.split()[-1]], **kwargs)


def _tournament(out: Path, judge_client: FakeJudge | None = None, **kwargs):
    return judge(ROWS, None, judge_client or _fake(), stage="tournament", out=out, schedule=TINY_TOURNAMENT, **kwargs)


def _rubric(out: Path, judge_client: FakeJudge | None = None, **kwargs):
    kwargs.setdefault("schedule", TINY_RUBRIC)
    return judge(ROWS, None, judge_client or _fake(), stage="rubric", out=out, **kwargs)


def _tokenized(fake: FakeJudge, tokenizer: Path | str, **update) -> FakeJudge:
    """``fake`` with a tokenizer (and any other config fields) on its judge config."""
    fake.config = fake.config.model_copy(update={"tokenizer": str(tokenizer), **update})
    return fake


class TestAPass:
    def test_every_window_is_stored_valid_under_one_family(self, tmp_path: Path) -> None:
        result = _rubric(tmp_path)
        (family,) = result.families.values()
        assert (family.judge_model, family.criteria) == ("fake", ("C1", "C2", "C3", "C4", "C5"))
        assert result.query_ids() == [("dataset", "q1"), ("dataset", "q2")]
        assert all(j.valid and j.family_key == family.key for j in result.judgements)
        assert all(set(p.criteria) == set(family.criteria) for j in result.judgements for p in j.placements)
        assert len(JudgementStore(tmp_path).records("rubric")) == len(result.judgements)
        identity = json.loads((tmp_path / "identity.json").read_text())
        assert identity["stages"]["rubric"]["family_key"] == family.key

    def test_the_tournament_scores_follow_ability(self, tmp_path: Path) -> None:
        result = _tournament(tmp_path)
        assert all(j.valid and j.ranking for j in result.judgements)
        wins = sum(
            (a.score > b.score) == (ABILITY[a.doc_id] > ABILITY[b.doc_id])
            for j in result.judgements
            for a in j.placements
            for b in j.placements
            if a.doc_id < b.doc_id
        )
        pairs = sum(len(j.placements) * (len(j.placements) - 1) // 2 for j in result.judgements)
        assert wins / pairs > 0.8

    def test_the_same_judge_writes_the_same_records(self, tmp_path: Path) -> None:
        def records(out: Path) -> list[tuple]:
            result = _tournament(out)
            return [(j.record_id, tuple(p.score for p in j.placements)) for j in result.judgements]

        assert records(tmp_path / "a") == records(tmp_path / "b")

    def test_the_schedule_seed_decides_the_window_draws(self, tmp_path: Path) -> None:
        def windows(seed: int, out: Path) -> list[tuple[str, ...]]:
            schedule = TINY_TOURNAMENT.model_copy(update={"seed": seed})
            result = judge(ROWS, None, _fake(), stage="tournament", out=out, schedule=schedule)
            return [tuple(p.doc_id for p in j.placements) for j in result.judgements]

        assert windows(42, tmp_path / "a") == windows(42, tmp_path / "b")
        assert windows(42, tmp_path / "a") != windows(7, tmp_path / "c")


class TestResume:
    def test_a_rerun_asks_only_for_missing_windows(self, tmp_path: Path) -> None:
        first = _fake()
        full = _tournament(tmp_path, first)
        again = _fake()
        assert _tournament(tmp_path, again).judgements == full.judgements
        assert again.usage.requests == 0

        path = JudgementStore(tmp_path).path("tournament")
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-3]) + lines[-3][:40])  # three windows lost, the last one torn mid-write
        resumed = _fake()
        result = _tournament(tmp_path, resumed)
        assert resumed.usage.requests == 3
        assert {j.record_id for j in result.judgements} == {j.record_id for j in full.judgements}

    def test_another_identity_is_refused_by_name_unless_forced(self, tmp_path: Path) -> None:
        _rubric(tmp_path)
        other = RubricSchedule(window=4, placements_per_doc=2.0)
        with pytest.raises(IdentityError, match="schedule.window") as caught:
            _rubric(tmp_path, schedule=other)
        assert any("schedule.window" in line for line in caught.value.details["differences"])
        result = _rubric(tmp_path, schedule=other, force=True)
        assert all(len(j.placements) <= 4 for j in result.judgements)
        assert list((tmp_path / ".superseded").glob("*/rubric.jsonl"))

    def test_an_identity_change_of_shape_is_named_not_crashed_on(self, tmp_path: Path) -> None:
        store = JudgementStore(tmp_path)
        family = _rubric(tmp_path / "x").families.popitem()[1]
        store.claim("rubric", {"judge": {"chat_template_kwargs": None}}, family)
        with pytest.raises(IdentityError, match="chat_template_kwargs"):
            store.claim("rubric", {"judge": {"chat_template_kwargs": {"enable_thinking": False}}}, family)

    def test_a_resumed_refusal_supersedes_the_first_generation_of_windows(self, tmp_path: Path) -> None:
        """A pass that re-asks a refused window refits under the new answer, so the later-phase windows of
        the first fit are superseded: the store holds one generation, one window per sequence, and the refit
        reads a clean pass's window set."""
        from rcp_ndcg.judging.schedule import _balanced_groups, query_rng

        docs = [f"d{index:02d}" for index in range(20)]
        rows = [RankingExample(query_id="q", query="a query", doc_ids=docs, docs=[f"document {doc}" for doc in docs])]
        schedule = RubricSchedule(window=5, placements_per_doc=2.0)
        n_random, _ = schedule.windows_for(len(docs))
        target = {
            docs[index]
            for index in _balanced_groups(len(docs), 5, n_random, query_rng(schedule.seed, "dataset", "q"))[0]
        }

        def _ability(text: str) -> float:
            return float(text.split()[-1][1:])

        class _RefusesOne(FakeJudge):
            """Refuses the random window that shows exactly ``target``; answers every other window."""

            def _answer(self, request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                prompt = body["messages"][-1]["content"]
                if isinstance(prompt, str) and all(doc in prompt for doc in target):
                    return httpx.Response(400, json={"error": {"message": "prompt too long"}})
                return super()._answer(request)

        first = judge(rows, None, _RefusesOne(_ability), stage="rubric", out=tmp_path, schedule=schedule)
        assert len(first.judgements) == schedule.calls_per_query(len(docs)) == 8
        (refused,) = [j for j in first.judgements if not j.valid]
        assert refused.phase == "random" and refused.response is None

        judge(rows, None, FakeJudge(_ability), stage="rubric", out=tmp_path, schedule=schedule)
        records = list(JudgementStore(tmp_path).records("rubric").values())
        assert len(records) == 8 and len({j.window_seq for j in records}) == 8
        assert all(j.valid for j in records)

        clean = judge(rows, None, FakeJudge(_ability), stage="rubric", out=tmp_path / "clean", schedule=schedule)

        def window_set(judgements):
            return {(j.window_seq, tuple(p.doc_id for p in j.placements)) for j in judgements}

        assert window_set(records) == window_set(clean.judgements)


class TestSubsets:
    def test_a_tournament_subset_gets_the_windows_its_size_gives(self, tmp_path: Path) -> None:
        """Two documents re-judged on the paper's tournament: their placements' windows, not a whole query's."""
        from rcp_ndcg.judging import TournamentSchedule, estimate

        schedule = TournamentSchedule()
        subset = {"q1": ["q1-d02", "q1-d07"]}
        client = _fake()
        result = judge(ROWS, None, client, stage="tournament", out=tmp_path, schedule=schedule, docs=subset)
        projected = estimate(ROWS, None, client.config, stages=["tournament"], schedules={"tournament": schedule},
                             docs=subset)  # fmt: skip
        # Windows of the pair at the placements per document: 4 random and 2 stratified windows, each mirrored, and
        # one adaptive window (one batch: every batch would ask the same pair again); not a whole query's 216.
        assert schedule.windows_for(2) == (4, 2, 1)
        assert client.usage.requests == len(result.judgements) == projected.calls == 2 * (4 + 2) + 1

    @pytest.mark.parametrize("size", [1, 2, 4, 10, 11])
    @pytest.mark.parametrize("stage", ["tournament", "rubric"])
    def test_the_estimate_is_the_windows_asked_for_any_pool(self, tmp_path: Path, stage: str, size: int) -> None:
        """The shipped schedules on pools below, at and above a window: the estimate counts every call made."""
        from rcp_ndcg.judging import estimate
        from rcp_ndcg.judging.schedule import schedule_for

        schedule = schedule_for(stage, "text")
        subset = {"q1": ROWS[0].doc_ids[:size]}
        client = _fake()
        if stage == "tournament" and size < 2:
            # A pool of one has no pair to compare; the pass refuses it rather than judging nothing silently.
            with pytest.raises(DataError, match="at least two"):
                judge(ROWS, None, client, stage=stage, out=tmp_path, schedule=schedule, docs=subset)
            assert client.usage.requests == 0
            return
        result = judge(ROWS, None, client, stage=stage, out=tmp_path, schedule=schedule, docs=subset)
        projected = estimate(ROWS, None, client.config, stages=[stage], schedules={stage: schedule}, docs=subset)
        assert client.usage.requests == len(result.judgements) == projected.calls == schedule.calls_per_query(size)
        assert projected.calls > 0

    def test_a_tiny_pool_asks_one_adaptive_window_not_one_per_batch(self, tmp_path: Path) -> None:
        """A pool no larger than the adaptive window: every batch would ask the whole pool again, whose
        comparisons the first window's answers already cover -- so the schedule asks it once."""
        from rcp_ndcg.judging import TournamentSchedule

        schedule = TournamentSchedule()
        subset = {"q1": ROWS[0].doc_ids[:4]}
        client = _fake()
        result = judge(ROWS, None, client, stage="tournament", out=tmp_path, schedule=schedule, docs=subset)
        adaptive = [tuple(p.doc_id for p in j.placements) for j in result.judgements if j.phase == "adaptive"]
        assert len(adaptive) == 1
        assert client.usage.requests == schedule.calls_per_query(4)

    def test_docs_judges_only_those_documents(self, tmp_path: Path) -> None:
        pools = {row.id: row.doc_ids[:8] for row in ROWS}
        subset = {"q2": ["q2-d10", "q2-d11"]}
        result = judge(ROWS, None, _fake(), stage="rubric", out=tmp_path, schedule=TINY_RUBRIC, docs=subset)
        assert result.query_ids() == [("dataset", "q2")]
        assert {p.doc_id for j in result.judgements for p in j.placements} == set(subset["q2"])
        with pytest.raises(DataError, match="not among its candidates"):
            judge(ROWS, pools, _fake(), stage="rubric", out=tmp_path / "y", schedule=TINY_RUBRIC, docs=subset)

    def test_candidates_restrict_and_order_the_pool(self, tmp_path: Path) -> None:
        pools = {"q1": ["q1-d09", "q1-d01", "q1-d05"]}
        result = judge(ROWS, pools, _fake(), stage="rubric", out=tmp_path, schedule=TINY_RUBRIC)
        assert result.query_ids() == [("dataset", "q1")]
        assert {p.doc_id for j in result.judgements for p in j.placements} == set(pools["q1"])


class _Garbled(FakeJudge):
    """An endpoint that answers, but in prose the parser refuses."""

    def _answer(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "I think doc_1 is great."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )


class _Refuses(FakeJudge):
    """An endpoint that refuses every request with HTTP 400."""

    def _answer(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "prompt too long"}})


class _Broken(FakeJudge):
    """A judge client with a defect in it."""

    def _answer(self, request: httpx.Request) -> httpx.Response:
        raise TypeError("a defect, not an answer")


class _GoesDown(FakeJudge):
    """A judge whose endpoint starts refusing connections after ``after`` answers."""

    def __init__(self, after: int, **kwargs) -> None:
        super().__init__(lambda text: ABILITY[text.split()[-1]], **kwargs)
        self.after = after

    def _answer(self, request: httpx.Request) -> httpx.Response:
        if self.usage.requests >= self.after:
            raise httpx.ConnectError("connection refused")
        return super()._answer(request)


class TestFailures:
    def test_an_unparseable_answer_is_asked_again_then_recorded_invalid(self, tmp_path: Path) -> None:
        garbled = _Garbled()
        result = judge(ROWS[:1], {"q1": ["q1-d00", "q1-d01"]}, garbled, stage="rubric", out=tmp_path)
        assert result.judgements and not any(j.valid for j in result.judgements)
        assert all(j.invalid_category == "no_json" for j in result.judgements)
        assert all(j.response == "I think doc_1 is great." and j.finish_reason == "stop" for j in result.judgements)
        assert garbled.usage.requests == MAX_ATTEMPTS * len(result.judgements)

    def test_an_unparseable_answer_is_kept_when_the_pass_resumes(self, tmp_path: Path) -> None:
        pools = {"q1": ["q1-d00", "q1-d01"]}
        first = judge(ROWS[:1], pools, _Garbled(), stage="rubric", out=tmp_path)
        resumed = _fake()
        again = judge(ROWS[:1], pools, resumed, stage="rubric", out=tmp_path)
        assert resumed.usage.requests == 0
        assert again.judgements == first.judgements

    def test_a_refused_request_is_recorded_and_asked_again_when_the_pass_resumes(self, tmp_path: Path) -> None:
        pools = {"q1": ["q1-d00", "q1-d01"]}
        schedule = RubricSchedule(window=2, placements_per_doc=2.0)  # one random and one stratified window
        refusing = _Refuses()
        refused = judge(ROWS[:1], pools, refusing, stage="rubric", out=tmp_path, schedule=schedule)
        assert refused.judgements and not any(j.valid or j.response for j in refused.judgements)
        assert all("RequestRejectedError: HTTP 400" in (j.invalid_reason or "") for j in refused.judgements)
        assert all(j.invalid_category == "refused" for j in refused.judgements)
        assert refusing.usage.failed_requests == MAX_ATTEMPTS * len(refused.judgements)

        # Resumed, every refused window is asked again; with answers in hand the schedule also
        # reaches its stratified window, which the refused pass had no preliminary ability for.
        resumed = _fake()
        again = judge(ROWS[:1], pools, resumed, stage="rubric", out=tmp_path, schedule=schedule)
        assert resumed.usage.requests == len(again.judgements) == len(refused.judgements) + 1
        assert all(j.valid for j in again.judgements)
        assert {j.record_id for j in refused.judgements} < {j.record_id for j in again.judgements}

    def test_a_refused_window_keeps_the_answer_it_saw(self, tmp_path: Path) -> None:
        """An answer on one attempt and refusals after it: the record keeps the answer's text (and its parse
        failure), so a store never loses what the judge said and ``reparse`` can read it again."""
        pools = {"q1": ["q1-d00", "q1-d01"]}
        schedule = RubricSchedule(window=2, placements_per_doc=1.0, random_share=1.0)  # one window

        class _GarbledThenRefuses(_Garbled):
            def _answer(self, request: httpx.Request) -> httpx.Response:
                if self.usage.requests == 0 and self.usage.failed_requests == 0:
                    return super()._answer(request)
                return httpx.Response(400, json={"error": {"message": "prompt too long"}})

        result = judge(ROWS[:1], pools, _GarbledThenRefuses(), stage="rubric", out=tmp_path, schedule=schedule)
        (record,) = result.judgements
        assert record.response == "I think doc_1 is great."
        assert record.invalid_category == "no_json" and record.finish_reason == "stop"

    def test_a_window_answering_with_the_prompts_worked_example_is_refused(self, tmp_path: Path) -> None:
        """A judge that echoes the prompt's own example (or an injection that supplies it) is refused, not
        stored as an observation: the example is a template constant, not a judgement."""
        schedule = RubricSchedule(window=5, placements_per_doc=1.0, random_share=1.0)

        class _EchoesTheExample(FakeJudge):
            def _answer(self, request: httpx.Request) -> httpx.Response:
                example = json.dumps(load_prompt("rubric").worked_example)
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": example}, "finish_reason": "stop"}
                        ]
                    },
                )

        result = judge(ROWS[:1], None, _EchoesTheExample(), stage="rubric", out=tmp_path, schedule=schedule)
        assert result.judgements and all(not record.valid for record in result.judgements)
        assert all("worked example" in (record.invalid_reason or "") for record in result.judgements)

    def test_a_defect_in_the_judge_propagates_and_stores_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="a defect"):
            _rubric(tmp_path, _Broken())
        assert not JudgementStore(tmp_path).records("rubric")

    def test_an_outage_propagates_and_keeps_the_answers_received(self, tmp_path: Path) -> None:
        config = JudgeConfig.fake(0).model_copy(update={"wait_on_outage_s": 0.02, "max_retries": 0})
        with pytest.raises(BackendUnavailableError):
            _rubric(tmp_path, _GoesDown(after=5, config=config))
        assert len(JudgementStore(tmp_path).records("rubric")) == 5

    def test_a_prompt_written_for_pages_is_refused_on_prose(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="rubric_vision"):
            _rubric(tmp_path, schedule=TINY_RUBRIC.model_copy(update={"prompt": "rubric_vision"}))

    def test_a_schedule_of_the_other_stage_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="RubricSchedule"):
            _rubric(tmp_path, schedule=TINY_TOURNAMENT)


def test_a_rubric_pass_whose_settings_cannot_show_every_document_is_refused(tmp_path: Path) -> None:
    """The coverage precondition ``n_random * w >= n_units``: below it the tier windows may repeat
    documents and leave units unseen, so the pass (and its estimate) refuse before a call."""
    from rcp_ndcg.judging import estimate

    docs = [f"d{index:02d}" for index in range(100)]
    rows = [RankingExample(query_id="q", query="a query", doc_ids=docs, docs=[f"document {doc}" for doc in docs])]
    client = _fake()
    schedule = RubricSchedule(placements_per_doc=0.5)
    with pytest.raises(ConfigError, match="n_random \\* w"):
        judge(rows, None, client, stage="rubric", out=tmp_path, schedule=schedule)
    assert client.usage.requests == 0 and not (tmp_path / "identity.json").exists()
    with pytest.raises(ConfigError, match="n_random \\* w"):
        estimate(rows, None, client.config, stages=["rubric"], schedules={"rubric": schedule})
    # The shipped placements and the per-modality windows hold the precondition.
    assert RubricSchedule().uncovered_units(100) == 0
    assert RubricSchedule.for_modality("image").uncovered_units(100) == 0


def test_a_dropped_stratified_phase_is_warned_and_recorded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When the random phase produced no valid answer the rubric's stratified phase cannot be selected: the
    drop is warned and recorded in the store's census, never silent."""
    schedule = RubricSchedule(window=2, placements_per_doc=2.0)
    pools = {"q1": ["q1-d00", "q1-d01"]}
    with caplog.at_level(logging.WARNING, logger="rcp_ndcg"):
        result = judge(ROWS[:1], pools, _Refuses(), stage="rubric", out=tmp_path, schedule=schedule)
    assert len(result.judgements) == 1  # the random window; the stratified one was dropped
    assert any("stratified phase" in message for message in caplog.messages)
    rows = [json.loads(line) for line in (tmp_path / "preprocessing.jsonl").read_text().splitlines()]
    (dropped,) = [row for row in rows if row["mechanism"] == "phase_dropped"]
    assert dropped["query_id"] == "q1" and dropped["windows"] == 1 and dropped["phase"] == "stratified"


def test_chunked_documents_are_judged_by_chunk_and_recorded_under_their_document(
    tmp_path: Path, word_tokenizer_file: Path
) -> None:
    long = RankingExample(
        query_id="q",
        query="tiny query",
        doc_ids=["a", "b"],
        docs=["intro " * 30 + "tiny document q1-d09", "tiny document q1-d00"],
    )
    policy = Preprocessing(chunk=ChunkPolicy(max_tokens=20, overlap_tokens=3))
    fake = _tokenized(FakeJudge(lambda text: 2.0 if text.endswith("q1-d09") else -2.0), word_tokenizer_file)
    result = judge([long], None, fake, stage="rubric", out=tmp_path, preprocessing=policy, schedule=TINY_RUBRIC)
    assert all(j.valid for j in result.judgements)
    placements = [p for j in result.judgements for p in j.placements]
    assert {p.doc_id for p in placements} == {"a", "b"}
    assert {p.chunk_id for p in placements if p.doc_id == "a"} >= {"a#0", "a#1"}
    assert {p.chunk_id for p in placements if p.doc_id == "b"} == {None}
    (family,) = result.families.values()
    assert family.preprocessing == Preprocessing(chunk=policy.chunk).key


def test_the_rubric_asks_enough_windows_to_show_every_chunk(tmp_path: Path, word_tokenizer_file: Path) -> None:
    """The window floor counts what the windows show: two documents of three chunks each take three windows of
    two, and the random phase alone covers every chunk (the coverage precondition)."""
    rows = [
        RankingExample(
            query_id="q",
            query="tiny query",
            doc_ids=["a", "b"],
            docs=["x " * 240 + "tiny document q1-d09", "y " * 240 + "tiny document q1-d00"],
        )
    ]
    policy = Preprocessing(chunk=ChunkPolicy(max_tokens=100, overlap_tokens=10))
    schedule = RubricSchedule(window=2, placements_per_doc=1.0, random_share=1.0)
    fake = _tokenized(FakeJudge(lambda text: 1.0), word_tokenizer_file)
    result = judge(rows, None, fake, stage="rubric", out=tmp_path, preprocessing=policy, schedule=schedule)
    tokenizer = load_tokenizer(str(word_tokenizer_file))
    chunks = chunk_ranking_example(rows[0], policy.chunk, tokenizer)
    assert len(chunks.doc_ids) == 6
    assert len(result.judgements) == 3  # counting documents, the placements would give round(1.0 * 2 / 2) = 1 window
    assert {p.chunk_id for j in result.judgements for p in j.placements} == set(chunks.doc_ids)


class _Recording(FakeJudge):
    """The fake judge, keeping every request it answers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[CompletionInput] = []

    async def complete(self, request: CompletionInput) -> Completion:
        self.requests.append(request)
        return await super().complete(request)


class TestTheWindowTextBudget:
    """A judge that declares its context and tokenizer gets each document cut to its window's share of tokens.

    The documents are full of ``&``, which the prompt escapes to ``&amp;`` (three word-level tokens for one), so a
    budget counted on the raw text instead of the text as the prompt carries it would overflow the context.
    """

    CONTEXT, OUTPUT = 2_000, 200
    DOCS = {doc: f"{doc} " + "the & " * 600 for doc in ("a", "b", "c", "d")}
    ROWS = [RankingExample(query_id="q", query="a query", doc_ids=list(DOCS), docs=list(DOCS.values()))]
    SCHEDULE = RubricSchedule(window=2, placements_per_doc=2.0)

    def _judge(self, tmp_path: Path, tokenizer: Path) -> tuple[_Recording, int, list]:
        fake = _Recording(lambda text: {"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0}[text.split()[0]])
        _tokenized(fake, tokenizer, context_tokens=self.CONTEXT, max_output_tokens=self.OUTPUT)
        result = judge(self.ROWS, None, fake, stage="rubric", out=tmp_path, schedule=self.SCHEDULE)
        words = load_tokenizer(str(tokenizer))
        overhead = prompt_overhead_tokens(load_prompt("rubric"), "rubric", "a query", 2, words)
        budget = window_tokens(fake.config, 2, overhead_tokens=overhead)
        assert budget is not None and 0 < budget < words.count(html.escape(self.DOCS["a"]))
        return fake, budget, list(result.judgements)

    def test_every_request_fits_the_context_counted_with_the_tokenizer(
        self, tmp_path: Path, word_tokenizer_file: Path
    ) -> None:
        fake, _, _ = self._judge(tmp_path, word_tokenizer_file)
        words = load_tokenizer(str(word_tokenizer_file))
        room = self.CONTEXT - self.OUTPUT - CHAT_TEMPLATE_TOKENS
        counts = [words.count(request.user_prompt) for request in fake.requests]
        assert counts and all(room - 8 <= count <= room for count in counts)  # within the context, and tight

    def test_each_document_is_a_verbatim_prefix_within_the_budget(
        self, tmp_path: Path, word_tokenizer_file: Path
    ) -> None:
        fake, budget, judgements = self._judge(tmp_path, word_tokenizer_file)
        words = load_tokenizer(str(word_tokenizer_file))
        bodies = [body for request in fake.requests for _, body in _DOC_BLOCK.findall(request.user_prompt)]
        assert len(bodies) == sum(len(j.placements) for j in judgements) > 0
        for body in bodies:
            assert budget - 2 <= words.count(body) <= budget  # counted as sent, escaped
            assert self.DOCS[body.split()[0]].startswith(html.unescape(body))

    def test_each_cut_is_a_row_of_the_preprocessing_record(self, tmp_path: Path, word_tokenizer_file: Path) -> None:
        _, _, judgements = self._judge(tmp_path, word_tokenizer_file)
        rows = [json.loads(line) for line in (tmp_path / "preprocessing.jsonl").read_text().splitlines()]
        cuts = [row for row in rows if row["mechanism"] == "window_budget"]
        shown = sorted(p.doc_id for j in judgements for p in j.placements)
        assert sorted(row["doc_id"] for row in cuts) == shown
        assert all(
            (row["query_id"], row["original_tokens"], row["original_chars"]) == ("q", 1_201, len(self.DOCS["a"]))
            and row["kept_tokens"] < row["original_tokens"]
            for row in cuts
        )

    def test_a_planned_window_is_rendered_at_its_own_size_budget(
        self, tmp_path: Path, word_tokenizer_file: Path
    ) -> None:
        """A plan's windows are cut for their own size, so the same window asked in any plan shows the same
        text and reuses its record (before, every window of a plan took the longest window's budget)."""
        pool = list(self.DOCS)
        plan = [pool[:4], pool[:2]]

        def _pass(out: Path, windows: dict) -> _Recording:
            client = _Recording(lambda text: {"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0}[text.split()[0]])
            _tokenized(client, word_tokenizer_file, context_tokens=self.CONTEXT, max_output_tokens=self.OUTPUT)
            judge(self.ROWS, None, client, stage="tournament", out=out, windows=windows)
            return client

        mixed = _pass(tmp_path / "plan", {"q": plan})
        short = {r.user_prompt for r in mixed.requests if r.user_prompt.count('<doc id="doc_') == 2}
        assert short and len(mixed.requests) == 4  # two windows, each mirrored
        alone = _pass(tmp_path / "alone", {"q": [plan[1]]})
        assert {r.user_prompt for r in alone.requests} == short, "the same window renders the same text"
        again = _pass(tmp_path / "plan", {"q": [plan[1]]})
        assert again.usage.requests == 0  # the same window is one record, whatever plan asks it

    def test_without_a_tokenizer_documents_are_sent_whole_and_the_judge_is_warned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _Recording(lambda text: {"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0}[text.split()[0]])
        fake.config = fake.config.model_copy(update={"context_tokens": self.CONTEXT})
        with caplog.at_level(logging.WARNING):
            judge(self.ROWS, None, fake, stage="rubric", out=tmp_path, schedule=self.SCHEDULE)
        bodies = [body for request in fake.requests for _, body in _DOC_BLOCK.findall(request.user_prompt)]
        assert bodies and all(html.unescape(body) == self.DOCS[body.split()[0]].strip() for body in bodies)
        assert "no tokenizer" in caplog.text


class TestTheTokenizer:
    def test_a_policy_that_cuts_without_a_tokenizer_is_refused_before_a_call(self, tmp_path: Path) -> None:
        for policy in (
            Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=10)),
            Preprocessing(chunk=ChunkPolicy(max_tokens=10, overlap_tokens=2)),
        ):
            fake = _fake()
            with pytest.raises(ConfigError, match="tokenizer"):
                _rubric(tmp_path, fake, preprocessing=policy)
            assert fake.usage.requests == 0

    def test_the_family_key_changes_with_the_tokenizer_and_the_rubric_key_does_not(
        self, tmp_path: Path, word_tokenizer_file: Path
    ) -> None:
        from tests._tokenizers import byte_bpe_tokenizer, save

        (tmp_path / "bpe").mkdir()
        bpe_file = save(byte_bpe_tokenizer(), tmp_path / "bpe")
        (plain,) = _rubric(tmp_path / "none").families.values()
        (words,) = _rubric(tmp_path / "words", _tokenized(_fake(), word_tokenizer_file)).families.values()
        (bpe,) = _rubric(tmp_path / "bpe_store", _tokenized(_fake(), bpe_file)).families.values()
        assert plain.tokenizer is None
        assert words.tokenizer == load_tokenizer(str(word_tokenizer_file)).sha256
        assert len({plain.key, words.key, bpe.key}) == 3
        assert plain.rubric_key == words.rubric_key == bpe.rubric_key  # the tokenizer is the judge's

    def test_the_identity_holds_the_tokenizers_content_and_the_store_its_name(
        self, tmp_path: Path, word_tokenizer_file: Path
    ) -> None:
        store = tmp_path / "store"
        _rubric(store, _tokenized(_fake(), word_tokenizer_file))
        (entry,) = JudgementStore(store).identities().values()
        assert entry["identity"]["preprocessing"]["tokenizer"] == {
            "sha256": load_tokenizer(str(word_tokenizer_file)).sha256
        }
        assert "tokenizer" not in entry["identity"]["judge"]
        assert entry["sources"]["tokenizer"] == str(word_tokenizer_file)
        # The same tokenizer.json at another path, or named by its directory, resumes the store: nothing is asked.
        moved = tmp_path / "elsewhere"
        moved.mkdir()
        (moved / "tokenizer.json").write_bytes(word_tokenizer_file.read_bytes())
        for spelling in (moved / "tokenizer.json", moved):
            again = _tokenized(_fake(), spelling)
            _rubric(store, again)
            assert again.usage.requests == 0, spelling

    def test_the_same_prompt_and_dataset_under_other_spellings_resume_the_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.data import load_dataset

        (tmp_path / "world").mkdir()
        (tmp_path / "world" / "dataset.jsonl").write_text(
            "".join(json.dumps({"query_id": r.id, "query": r.text, "doc_ids": r.doc_ids, "docs": r.docs}) + "\n"
                    for r in ROWS),
            encoding="utf-8",
        )  # fmt: skip
        prompts = [tmp_path / "a" / "my_rubric.txt", tmp_path / "b" / "copy.txt"]
        for path in prompts:
            path.parent.mkdir()
            path.write_text(load_prompt("rubric").text, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        uris = ["jsonl:world/dataset.jsonl", "jsonl:./world/../world/dataset.jsonl"]
        store = tmp_path / "store"
        for index, (uri, prompt) in enumerate(zip(uris, prompts, strict=True)):
            client = _fake()
            schedule = TINY_RUBRIC.model_copy(update={"prompt": str(prompt)})
            judge(load_dataset(uri), None, client, stage="rubric", out=store, schedule=schedule)
            assert (client.usage.requests > 0) == (index == 0), uri
        (entry,) = JudgementStore(store).identities().values()
        assert entry["identity"]["dataset"]["uri"] == f"jsonl:{(tmp_path / 'world' / 'dataset.jsonl').resolve()}"
        assert JudgementStore(store).schedule("rubric").prompt == str(prompts[0])  # as the first pass named it


def test_the_load_time_cuts_are_recorded_once_per_store(tmp_path: Path, word_tokenizer_file: Path) -> None:
    policy = Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=2))
    store = tmp_path / "store"
    record = store / "preprocessing.jsonl"

    def cuts() -> list[tuple[str, str]]:
        rows = [json.loads(line) for line in record.read_text().splitlines()] if record.is_file() else []
        return sorted((row["mechanism"], row["doc_id"]) for row in rows)

    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)
    first = cuts()
    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)

    shown = sorted({doc for row in ROWS for doc in row.doc_ids})
    assert first == [("doc_policy", doc) for doc in shown]
    assert cuts() == first

    # Another stage under another policy cuts the same documents differently: those cuts are recorded too.
    longer = Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=3))
    _tournament(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=longer)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    kept = {(row["doc_id"], row["kept_tokens"]) for row in rows if row["mechanism"] == "doc_policy"}
    assert kept == {(doc, 2) for doc in shown} | {(doc, 3) for doc in shown}


def test_a_torn_last_census_line_does_not_poison_the_store(
    tmp_path: Path, word_tokenizer_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A pass killed mid-append leaves a torn last line of ``preprocessing.jsonl``; the documented recovery
    story (re-run over the same store) must survive it: the torn line is skipped with a warning, and a row
    that is not merely torn but malformed is a typed error naming the line."""
    policy = Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=2))
    store = tmp_path / "store"
    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)
    record = store / "preprocessing.jsonl"
    record.write_text(record.read_text(encoding="utf-8") + '{"mechanism": "doc_pol', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="rcp_ndcg"):
        _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)
    assert any("torn" in message for message in caplog.messages)

    record.write_text(
        "".join(record.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]) + "not json at all\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="preprocessing.jsonl"):
        _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)


def test_the_window_text_budget_follows_the_context() -> None:
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m", context_tokens=10_000, max_output_tokens=1_000)
    assert window_tokens(judge_cfg.model_copy(update={"context_tokens": None}), 10, overhead_tokens=500) is None
    assert window_tokens(judge_cfg, 10, overhead_tokens=500) == (10_000 - 500 - 1_000) // 10
    with pytest.raises(CapabilityError, match="images 10,000"):
        window_tokens(judge_cfg, 10, overhead_tokens=500, media_tokens_per_doc=1_000)


class _SchemaEndpoint:
    """An in-process OpenAI-compatible endpoint: records every request payload and answers as the fake judge."""

    def __init__(self, version: str = "1.0") -> None:
        self.fake = _fake()
        self.version = version
        self.requests: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            model = {"id": "m", "object": "model", "owned_by": "engine", "max_model_len": 32768}
            return httpx.Response(
                200,
                json={"object": "list", "data": [model]},
                headers={"server": "uvicorn", "x-engine-version": self.version},
            )
        payload = json.loads(request.content)
        self.requests.append(payload)
        answer = _answer_text(self.fake.seed, self.fake.ability, self.fake.severity, payload["messages"][-1]["content"])
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "system_fingerprint": f"engine-{self.version}",
            },
        )

    def client(self, **fields) -> JudgeClient:
        config = JudgeConfig(base_url="http://judge.test/v1", model="m", **fields)
        return JudgeClient(config, httpx_transport=httpx.MockTransport(self))


def test_the_store_records_what_the_endpoint_serves_beside_the_identity(tmp_path: Path) -> None:
    first = judge(ROWS, None, _SchemaEndpoint("1.0").client(), stage="rubric", out=tmp_path, schedule=TINY_RUBRIC)
    (entry,) = JudgementStore(tmp_path).identities().values()
    assert entry["engines"] == [
        {
            "url": "http://judge.test/v1",
            "model": "m",
            "owned_by": "engine",
            "max_model_len": 32768,
            "headers": {"server": "uvicorn", "x-engine-version": "1.0"},
            "system_fingerprint": "engine-1.0",
        }
    ]
    # The same engine again, with nothing left to ask: its report (no fingerprint this time) is already recorded.
    judge(ROWS, None, _SchemaEndpoint("1.0").client(), stage="rubric", out=tmp_path, schedule=TINY_RUBRIC)
    assert len(JudgementStore(tmp_path).identities()["rubric"]["engines"]) == 1
    # Another engine version is runtime information: the store is resumed, not refused, and both are recorded.
    endpoint = _SchemaEndpoint("2.0")
    second = judge(ROWS, None, endpoint.client(), stage="rubric", out=tmp_path, schedule=TINY_RUBRIC)
    assert first.families == second.families and endpoint.requests == []
    (entry,) = JudgementStore(tmp_path).identities().values()
    assert [engine["headers"]["x-engine-version"] for engine in entry["engines"]] == ["1.0", "2.0"]


class TestStructuredOutput:
    """A judge declared to constrain output gets each window's answer schema; the family records the decoding."""

    @pytest.mark.parametrize("stage", ["tournament", "rubric"])
    def test_a_declared_endpoint_is_sent_the_stages_schema_for_each_window(self, tmp_path: Path, stage) -> None:
        endpoint = _SchemaEndpoint()
        schedule = TINY_TOURNAMENT if stage == "tournament" else TINY_RUBRIC
        result = judge(
            ROWS, None, endpoint.client(decoding="json_schema"), stage=stage, out=tmp_path, schedule=schedule
        )
        (family,) = result.families.values()
        assert family.decoding == "json_schema"
        assert all(j.valid for j in result.judgements)
        assert len(endpoint.requests) == len(result.judgements)
        for payload in endpoint.requests:
            window = payload["messages"][-1]["content"].count('<doc id="doc_')
            assert payload["response_format"] == response_format(stage, window, family.criteria)
            assert payload["response_format"]["type"] == "json_schema"

    def test_an_undeclared_endpoint_decodes_freely_under_another_family(self, tmp_path: Path) -> None:
        free, constrained = _SchemaEndpoint(), _SchemaEndpoint()
        loose = judge(ROWS, None, free.client(), stage="rubric", out=tmp_path / "a", schedule=TINY_RUBRIC)
        strict = judge(
            ROWS,
            None,
            constrained.client(decoding="json_schema"),
            stage="rubric",
            out=tmp_path / "b",
            schedule=TINY_RUBRIC,
        )
        assert free.requests and not any("response_format" in payload for payload in free.requests)
        (loose_family,), (strict_family,) = loose.families.values(), strict.families.values()
        assert loose_family.decoding == "free"
        assert loose_family.key != strict_family.key


def test_a_remote_store_is_refused_before_the_judge_is_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.judging import judge as judge_pass
    from rcp_ndcg.testing import FakeJudge

    monkeypatch.chdir(tmp_path)  # a regression must not write a store into the checkout
    with pytest.raises(ConfigError, match="local") as refused:
        judge_pass(None, {}, FakeJudge(), stage="rubric", out="gs://bucket/store")
    assert "--mirror" in (refused.value.hint or "")


class TestThePrompt:
    def test_a_prompt_without_the_passages_placeholder_is_refused_before_any_call(self, tmp_path: Path) -> None:
        from rcp_ndcg.judging.prompts import load_prompt

        prompt = tmp_path / "rubric.txt"
        prompt.write_text(load_prompt("rubric").text.replace("{passages_placeholder}", ""), encoding="utf-8")
        judge_client = _fake()
        with pytest.raises(ConfigError, match="passages_placeholder"):
            _rubric(tmp_path / "store", judge_client, schedule=TINY_RUBRIC.model_copy(update={"prompt": str(prompt)}))
        assert judge_client.usage.requests == 0
        assert not (tmp_path / "store").exists()

    def test_the_store_keeps_each_prompts_text_by_its_hash(self, tmp_path: Path) -> None:
        from rcp_ndcg.judging.prompts import load_prompt

        tournament, rubric = _tournament(tmp_path), _rubric(tmp_path)
        for judged, name in ((tournament, "tournament"), (rubric, "rubric")):
            (family,) = judged.families.values()
            kept = tmp_path / "prompts" / f"{family.prompt_hash}.txt"
            assert kept.read_text(encoding="utf-8") == load_prompt(name).text


class TestPlannedWindows:
    def test_exactly_the_given_windows_are_asked_each_with_its_mirror(self, tmp_path: Path) -> None:
        judge_client = _fake()
        q1 = ROWS[0].doc_ids
        windows = {ROWS[0].id: [[q1[0], q1[1], q1[2]], [q1[0], q1[3]]]}
        judged = _tournament(tmp_path, judge_client, windows=windows)

        assert judge_client.usage.requests == 4  # two windows, each also asked reversed (the schedule mirrors)
        shown = sorted(tuple(p.doc_id for p in j.placements) for j in judged.judgements)
        assert shown == sorted([(q1[0], q1[1], q1[2]), (q1[2], q1[1], q1[0]), (q1[0], q1[3]), (q1[3], q1[0])])
        assert {j.phase for j in judged.judgements} == {None}
        again = _fake()
        _tournament(tmp_path, again, windows=windows)
        assert again.usage.requests == 0  # asked once: a rerun reuses the stored windows

    def test_an_empty_planned_window_list_is_refused_before_anything_is_asked(self, tmp_path: Path) -> None:
        """A query whose window list is empty has nothing to ask; a crash after other queries stored their
        answers (the bare ``max()`` ValueError) would break the recovery workflow mid-flight."""
        q1 = ROWS[0].doc_ids
        with pytest.raises(ConfigError, match="no windows"):
            _tournament(tmp_path, windows={ROWS[0].id: [[q1[0], q1[1]]], ROWS[1].id: []})
        assert not JudgementStore(tmp_path).identities()  # refused before the store was claimed

    def test_a_planned_window_is_one_record_whatever_plan_or_grouping_asks_it(self, tmp_path: Path) -> None:
        first_plan = [["q1-d00", "q1-d05", "q1-d09"]]
        second_plan = [["q1-new", "q1-d02"], ["q1-new", "q1-d07"]]
        asked = []
        for windows in (first_plan, second_plan + first_plan, second_plan, [first_plan[0], first_plan[0]]):
            client = _fake()
            _tournament(tmp_path, client, windows={"q1": windows})
            asked.append(client.usage.requests)
        assert asked == [2, 4, 0, 0]  # each window mirrored; a window already judged is never asked again
        records = list(JudgementStore(tmp_path).records("tournament").values())
        assert len(records) == 6 == len({tuple(p.doc_id for p in r.placements) for r in records})
        assert all(r.window_seq is None and r.phase is None for r in records)
        assert len(JudgementStore(tmp_path).path("tournament").read_text().splitlines()) == 6

    def test_windows_and_docs_are_exclusive_and_ids_must_be_candidates(self, tmp_path: Path) -> None:
        q1 = ROWS[0].doc_ids
        with pytest.raises(ConfigError, match="docs"):
            _tournament(tmp_path, windows={ROWS[0].id: [[q1[0], q1[1]]]}, docs={ROWS[0].id: [q1[0]]})
        with pytest.raises(DataError, match="nope"):
            _tournament(tmp_path, windows={ROWS[0].id: [[q1[0], "nope"]]})

    def test_an_empty_docs_entry_is_refused_before_anything_is_asked(self, tmp_path: Path) -> None:
        """A subset map that names no document has nothing to judge; the pass used to claim a store and
        report success over zero windows."""
        client = _fake()
        with pytest.raises(ConfigError, match="no documents"):
            _rubric(tmp_path, client, docs={ROWS[0].id: []})
        assert client.usage.requests == 0 and not (tmp_path / "identity.json").exists()

    def test_a_tournament_query_with_fewer_than_two_candidates_is_refused(self, tmp_path: Path) -> None:
        """A pool of one has no pair to compare; the pass was silent (no record, no warning)."""
        client = _fake()
        with pytest.raises(DataError, match="at least two"):
            judge(ROWS, {"q1": [ROWS[0].doc_ids[0]]}, client, stage="tournament", out=tmp_path)
        assert client.usage.requests == 0 and not (tmp_path / "identity.json").exists()

    def test_duplicate_query_ids_are_refused(self, tmp_path: Path) -> None:
        """Two rows with one query id fuse: the second row's documents are never judged (or the first's
        answers are served for both)."""
        client = _fake()
        with pytest.raises(DataError, match="duplicate query id"):
            judge((ROWS[0], ROWS[0].model_copy(update={"docs": ["other " + d for d in ROWS[0].docs]})), None, client,
                  stage="rubric", out=tmp_path, schedule=TINY_RUBRIC)  # fmt: skip
        assert client.usage.requests == 0 and not (tmp_path / "identity.json").exists()


class TestIdentities:
    """What names a corpus and an instrument: the store identity, the family key and the record ids."""

    def test_two_row_corpora_never_share_a_store_or_a_record_id(self, tmp_path: Path) -> None:
        """Two row-sequence passes of different corpora that share query and document ids: the store gate
        refuses the second (its rows digest differs), and across stores the record ids differ, so a merge
        keeps both corpora's windows (it used to fuse them, pass 2 silently reusing pass 1's answers)."""
        other = tuple(
            row.model_copy(
                update={"text": f"a different corpus: {row.text}", "docs": [f"other {d}" for d in row.doc_ids]}
            )  # fmt: skip
            for row in ROWS
        )
        first_store, second_store = tmp_path / "a", tmp_path / "b"
        first = _tournament(first_store)
        # The same rows again resume the store as before (the same rows digest).
        again = _fake()
        _tournament(first_store, again)
        assert again.usage.requests == 0
        # Another corpus (same query and document ids) is refused into the same store.
        with pytest.raises(IdentityError, match="dataset"):
            judge(other, None, _fake(), stage="tournament", out=first_store, schedule=TINY_TOURNAMENT)
        # Into a new store it gets its own record ids, and a merge keeps both corpora's windows.
        second = judge(other, None, _fake(), stage="tournament", out=second_store, schedule=TINY_TOURNAMENT)
        assert {j.record_id for j in first.judgements} & {j.record_id for j in second.judgements} == set()
        merged = JudgementSet.merge([first, second])
        assert len(merged.judgements) == len(first.judgements) + len(second.judgements)

    def test_the_family_gains_a_declared_judge_setting_and_the_store_refuses_a_change(self, tmp_path: Path) -> None:
        """temperature, the output and context budgets, extra_body and the wire adapter are CONTENT: a family
        judged under one never pools with one judged under another (cross-store there is no gate)."""
        (family,) = _rubric(tmp_path).families.values()
        assert family.temperature is None and family.max_output_tokens is None  # defaults stay out of the key
        hot = _fake(config=JudgeConfig.fake(seed=0).model_copy(update={"temperature": 0.7}))
        (hot_family,) = _rubric(tmp_path / "hot", hot).families.values()
        assert hot_family.temperature == 0.7
        assert hot_family.key != family.key
        assert hot_family.rubric_key == family.rubric_key  # the instrument, not the judge
        # A store judged under the default refuses a pass of the changed instrument (and vice versa).
        changed = _fake()
        changed.config = changed.config.model_copy(update={"temperature": 0.7})
        with pytest.raises(IdentityError, match="temperature"):
            _rubric(tmp_path, changed)

    def test_a_row_input_records_its_rows_digest_in_the_store_identity(self, tmp_path: Path) -> None:
        _tournament(tmp_path)
        (entry,) = JudgementStore(tmp_path).identities().values()
        assert entry["identity"]["dataset"]["name"] == "dataset"
        assert len(entry["identity"]["dataset"]["rows_sha256"]) == 64

    def test_the_text_formatting_version_enters_the_family_and_the_record_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The text-formatting rule is code: a bump of ``TEXT_FORMATTING_VERSION`` re-keys the family and
        every record id, so judgements built from the old strings never pool with the new ones."""
        from rcp_ndcg_core.records import TEXT_FORMATTING_VERSION

        from rcp_ndcg.judging import judging as judging_module

        first = _rubric(tmp_path / "a")
        (first_family,) = first.families.values()
        assert first_family.text_formatting == TEXT_FORMATTING_VERSION
        monkeypatch.setattr(judging_module, "TEXT_FORMATTING_VERSION", "rcp-text/999")
        second = _rubric(tmp_path / "b")
        (second_family,) = second.families.values()
        assert second_family.text_formatting == "rcp-text/999"
        assert second_family.key != first_family.key
        assert not {j.record_id for j in first.judgements} & {j.record_id for j in second.judgements}

    def test_another_fake_seed_is_another_instrument(self, tmp_path: Path) -> None:
        """The offline judge's seed decides every draw, so it is part of the instrument: two seeds never
        share a store (and the config's own identity carries it)."""
        first = _rubric(tmp_path)
        (first_family,) = first.families.values()
        assert first_family.fake_seed == 0
        assert JudgeConfig.fake(0).identity() != JudgeConfig.fake(7).identity()
        other = _fake(seed=7)
        with pytest.raises(IdentityError, match="fake_seed"):
            _rubric(tmp_path, other)
        second = _rubric(tmp_path / "other", other)
        (second_family,) = second.families.values()
        assert second_family.fake_seed == 7
        assert second_family.key != first_family.key


class _GarbledThenWell(_Garbled):
    """An endpoint that garbles its first two answers, then answers well."""

    def __init__(self) -> None:
        super().__init__()
        self.well_after = 2

    def _answer(self, request: httpx.Request) -> httpx.Response:
        if self.usage.requests < self.well_after:
            return super()._answer(request)
        return FakeJudge._answer(self, request)


def test_a_garbled_answer_is_retried_up_to_three_attempts_then_well(tmp_path: Path) -> None:
    """The retry loop's contract, pinned: up to MAX_ATTEMPTS (the docs' 'three attempts in total') requests
    per window, and an answer that turns well on a later attempt is a valid record, not a retry-count
    fiction: one window spends two garbled attempts, the rest one each."""
    assert MAX_ATTEMPTS == 3
    flaky = _GarbledThenWell()
    result = judge(ROWS[:1], {"q1": ["q1-d00", "q1-d01"]}, flaky, stage="rubric", out=tmp_path)
    assert result.judgements and all(j.valid for j in result.judgements)
    assert flaky.usage.requests == len(result.judgements) + 2


def test_the_completion_reserve_is_capped_at_half_of_what_is_left() -> None:
    """``max_output_tokens`` reserves at most half of the usable context (a huge declared completion budget
    must not starve the window's text to nothing); a small one reserves itself exactly."""
    judge_cfg = JudgeConfig(base_url="http://h/v1", model="m", context_tokens=10_000)
    small = judge_cfg.model_copy(update={"max_output_tokens": 1_000})
    assert window_tokens(small, 10, overhead_tokens=500) == (10_000 - 500 - 1_000) // 10
    huge = judge_cfg.model_copy(update={"max_output_tokens": 9_000})
    # usable = 9500; the reserve is min(9000, 9500/2) = 4750: the text keeps half of what is left.
    assert window_tokens(huge, 10, overhead_tokens=500) == (9_500 - 9_500 // 2) // 10


def test_a_long_invalid_reason_is_cut_with_a_marker_not_silently(tmp_path: Path) -> None:
    """The declared cap of the schema's ``invalid_reason`` (2000): a longer diagnostic carries the cut marker,
    so a truncation is declared, recorded policy."""
    pool = {"q1": ["q1-d00", "q1-d01"]}

    class _ProlixRefusal(_Refuses):
        """Refuses with a very long message."""

        def _answer(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "x" * 5000}})

    result = judge(ROWS[:1], pool, _ProlixRefusal(), stage="rubric", out=tmp_path)
    assert result.judgements and not any(j.valid for j in result.judgements)
    (reason,) = {j.invalid_reason for j in result.judgements}
    assert len(reason) == 2000 and reason.endswith("(cut)")


def test_a_census_append_cuts_the_torn_tail_first(tmp_path: Path, word_tokenizer_file: Path) -> None:
    """A pass killed mid-append leaves a torn last census row; the NEXT pass's first append must cut it, not
    merge into it -- a merged line is refused by every later read (the fix's own promise)."""
    policy = Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=2))
    store = tmp_path / "store"
    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=policy)
    record = store / "preprocessing.jsonl"
    good = record.read_text(encoding="utf-8")
    record.write_text(good + '{"mechanism": "doc_pol', encoding="utf-8")  # the killed pass's torn row

    # The resumed pass cuts differently (so it appends: its first append must cut the torn tail) and a third
    # pass reads the file clean.
    longer = Preprocessing(text=TextPolicy(on_overflow="truncate", max_tokens=3))
    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=longer, force=True)
    _rubric(store, _tokenized(FakeJudge(lambda text: 0.0), word_tokenizer_file), preprocessing=longer)
    assert record.read_text(encoding="utf-8").endswith("\n"), "the append started on a fresh line"
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert all(isinstance(row.get("mechanism"), str) for row in rows), "no merged fragment survived"


def test_naming_the_default_wire_leaves_the_family_key_alone(tmp_path: Path) -> None:
    """`api: openai_chat` names the default wire: the same instrument, so the family key is the unset case's
    (the digest carries the adapter only when it differs from the default, the operator's rule)."""
    (default,) = _rubric(tmp_path).families.values()
    named = _fake(config=JudgeConfig.fake(seed=0).model_copy(update={"api": "openai_chat"}))
    (named_default,) = _rubric(tmp_path / "named", named).families.values()
    assert named_default.key == default.key
    # And the store resumes across the spelling: it is the same instrument.
    again = _fake()
    again.config = again.config.model_copy(update={"api": "openai_chat"})
    _rubric(tmp_path, again)  # no IdentityError
