"""judge(): one judging path over an append-only store, with the fake judge standing in for a model."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import httpx
import pytest
from rcp_ndcg_core._records import RankingExample

from rcp_ndcg.data.preprocess import ChunkPolicy, Preprocessing, TextPolicy, chunk_ranking_example
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError, IdentityError
from rcp_ndcg.llm import JudgeClient, JudgeConfig, JudgementStore, RubricSchedule, judge
from rcp_ndcg.llm._fake import _DOC_BLOCK, _answer_text
from rcp_ndcg.llm._parsing.schema import response_format
from rcp_ndcg.llm.client import BackendUnavailableError, Completion, CompletionInput
from rcp_ndcg.llm.judging import CHAT_TEMPLATE_TOKENS, MAX_ATTEMPTS, prompt_overhead_tokens, window_tokens
from rcp_ndcg.llm.prompts import load_prompt
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


class TestSubsets:
    def test_a_tournament_subset_gets_the_windows_its_size_gives(self, tmp_path: Path) -> None:
        """Two documents re-judged on the paper's tournament: their placements' windows, not a whole query's."""
        from rcp_ndcg.llm import TournamentSchedule, estimate

        schedule = TournamentSchedule()
        subset = {"q1": ["q1-d02", "q1-d07"]}
        client = _fake()
        result = judge(ROWS, None, client, stage="tournament", out=tmp_path, schedule=schedule, docs=subset)
        projected = estimate(ROWS, None, client.config, stages=["tournament"], schedules={"tournament": schedule},
                             docs=subset)  # fmt: skip
        # Windows of the pair at the placements per document: 4 random and 2 stratified windows, each mirrored, and
        # one adaptive window in each of 7 batches; not a whole query's 216.
        assert schedule.windows_for(2) == (4, 2, 1)
        assert client.usage.requests == len(result.judgements) == projected.calls == 2 * (4 + 2) + 7

    @pytest.mark.parametrize("size", [1, 2, 4, 10, 11])
    @pytest.mark.parametrize("stage", ["tournament", "rubric"])
    def test_the_estimate_is_the_windows_asked_for_any_pool(self, tmp_path: Path, stage: str, size: int) -> None:
        """The shipped schedules on pools below, at and above a window: the estimate counts every call made."""
        from rcp_ndcg.llm import estimate
        from rcp_ndcg.llm.schedule import schedule_for

        schedule = schedule_for(stage, "text")
        subset = {"q1": ROWS[0].doc_ids[:size]}
        client = _fake()
        result = judge(ROWS, None, client, stage=stage, out=tmp_path, schedule=schedule, docs=subset)
        projected = estimate(ROWS, None, client.config, stages=[stage], schedules={stage: schedule}, docs=subset)
        assert client.usage.requests == len(result.judgements) == projected.calls == schedule.calls_per_query(size)
        assert (size < 2 and stage == "tournament") or projected.calls > 0

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
    """The window floor counts what the windows show: two documents of three chunks each take three windows of two."""
    rows = [
        RankingExample(
            query_id="q",
            query="tiny query",
            doc_ids=["a", "b"],
            docs=["x " * 240 + "tiny document q1-d09", "y " * 240 + "tiny document q1-d00"],
        )
    ]
    policy = Preprocessing(chunk=ChunkPolicy(max_tokens=100, overlap_tokens=10))
    schedule = RubricSchedule(window=2, placements_per_doc=1.0)
    fake = _tokenized(FakeJudge(lambda text: 1.0), word_tokenizer_file)
    result = judge(rows, None, fake, stage="rubric", out=tmp_path, preprocessing=policy, schedule=schedule)
    tokenizer = load_tokenizer(str(word_tokenizer_file))
    assert len(chunk_ranking_example(rows[0], policy.chunk, tokenizer).doc_ids) == 6
    assert len(result.judgements) == 3  # counting documents, the placements would give round(1.0 * 2 / 2) = 1 window


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
            "".join(json.dumps({"query_id": r.id, "query": r.query, "doc_ids": r.doc_ids, "docs": r.docs}) + "\n"
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
    from rcp_ndcg.llm import judge as judge_pass
    from rcp_ndcg.testing import FakeJudge

    monkeypatch.chdir(tmp_path)  # a regression must not write a store into the checkout
    with pytest.raises(ConfigError, match="local") as refused:
        judge_pass(None, {}, FakeJudge(), stage="rubric", out="gs://bucket/store")
    assert "--mirror" in (refused.value.hint or "")


class TestThePrompt:
    def test_a_prompt_without_the_passages_placeholder_is_refused_before_any_call(self, tmp_path: Path) -> None:
        from rcp_ndcg.llm.prompts import load_prompt

        prompt = tmp_path / "rubric.txt"
        prompt.write_text(load_prompt("rubric").text.replace("{passages_placeholder}", ""), encoding="utf-8")
        judge_client = _fake()
        with pytest.raises(ConfigError, match="passages_placeholder"):
            _rubric(tmp_path / "store", judge_client, schedule=TINY_RUBRIC.model_copy(update={"prompt": str(prompt)}))
        assert judge_client.usage.requests == 0
        assert not (tmp_path / "store").exists()

    def test_the_store_keeps_each_prompts_text_by_its_hash(self, tmp_path: Path) -> None:
        from rcp_ndcg.llm.prompts import load_prompt

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
