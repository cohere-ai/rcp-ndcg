"""``reparse`` and ``rcp-ndcg judge reparse``: stored answers read again by the current parser, into a new store.

The source store is written by a stand-in for an older parser that decoded the whole answer strictly, as the
paper-era extraction did, so a valid object followed by a stray ``}`` was recorded invalid. The fake judge
appends that brace to some answers and answers others in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.judge import judge_group
from rcp_ndcg.errors import IdentityError
from rcp_ndcg.judging import JudgementStore, judge, reparse
from rcp_ndcg.judging._fake import _uniform
from rcp_ndcg.judging._parsing import listwise
from rcp_ndcg.judging._parsing.common import PARSE_VERSION, UnparseableAnswer, strip_code_fence, strip_reasoning
from rcp_ndcg.judging.client import Completion, CompletionInput
from rcp_ndcg.testing import TINY_TOURNAMENT, FakeJudge, tiny_rows

ROWS, ABILITY = tiny_rows()


class _Sloppy(FakeJudge):
    """By a draw per window (so every attempt at a window gets the same answer): about a third of the answers get
    a stray trailing brace, and about one in six is prose that names a ranking."""

    def __init__(self) -> None:
        super().__init__(lambda text: ABILITY[text.split()[-1]])

    async def complete(self, request: CompletionInput) -> Completion:
        answer = await super().complete(request)
        draw = _uniform("sloppy", request.user_prompt)
        if draw < 0.17:
            return answer.model_copy(update={"response": "Ranking: 2, 1, 3, 5, 4"})
        if draw < 0.5:
            return answer.model_copy(update={"response": answer.response + "}"})
        return answer


def _strict_decode(text: str | None) -> dict:
    """The older extraction: the whole stripped answer must be one JSON object."""
    try:
        return json.loads(strip_code_fence(strip_reasoning(text or "")))
    except json.JSONDecodeError as exc:
        raise UnparseableAnswer(f"the JSON object does not decode: {exc}", "invalid_json") from exc


@pytest.fixture
def old_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "v1"
    with monkeypatch.context() as patch:
        patch.setattr(listwise, "decode_answer", _strict_decode)
        patch.setattr("rcp_ndcg.judging.judging.PARSE_VERSION", 1)
        judge(ROWS, None, _Sloppy(), stage="tournament", out=store, schedule=TINY_TOURNAMENT)
    return store


def _scores(response: str) -> dict[str, float]:
    """The scores of the answer's JSON object, by prompt slot."""
    stated, _ = json.JSONDecoder().raw_decode(response)
    return {f"doc_{int(key)}": value for key, value in stated["scores"].items()}


def test_reparse_recovers_the_answers_the_older_parser_refused(old_store: Path, tmp_path: Path) -> None:
    before = JudgementStore(old_store).read()
    braced = [j for j in before.judgements if j.response and j.response.endswith("}}}")]
    prose = [j for j in before.judgements if j.response and j.response.startswith("Ranking")]
    assert braced and prose and not any(j.valid for j in braced + prose)

    after = reparse(old_store, tmp_path / "v2")

    (old_family,), (new_family,) = before.families.values(), after.families.values()
    assert (old_family.parse_version, new_family.parse_version) == (1, PARSE_VERSION)
    assert new_family.model_copy(update={"parse_version": 1}) == old_family
    assert len(after) == len(before)
    assert not {j.record_id for j in after.judgements} & {j.record_id for j in before.judgements}
    by_window = {(j.query_id, j.window_seq): j for j in after.judgements}
    for old in braced:
        new = by_window[(old.query_id, old.window_seq)]
        assert new.valid and new.phase == old.phase and new.response == old.response
        # Every score is the judge's own number from its JSON object.
        stated = _scores(old.response or "")
        assert {p.unit_id: p.score for p in new.placements} == {
            p.unit_id: stated[f"doc_{p.position}"] for p in new.placements
        }
    for old in prose:
        new = by_window[(old.query_id, old.window_seq)]
        assert not new.valid and new.invalid_category == "no_json" and new.ranking is None
    for old in before.judgements:
        if old.valid:
            assert by_window[(old.query_id, old.window_seq)].placements == old.placements
    assert JudgementStore(tmp_path / "v2").read().families == after.families


def test_the_command_reports_recovered_still_invalid_and_unchanged(old_store: Path, tmp_path: Path) -> None:
    before = JudgementStore(old_store).read()
    out = tmp_path / "v2"
    result = CliRunner().invoke(judge_group, ["reparse", "--judgements", str(old_store), "--out", str(out), "--json"])

    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    assert data["schema"] == "rcp-ndcg.reparse-report.v1" and data["parse_version"] == PARSE_VERSION
    (row,) = data["stages"]
    invalid = [j for j in before.judgements if not j.valid]
    prose = sum(1 for j in invalid if (j.response or "").startswith("Ranking"))
    assert row["records"] == len(before)
    assert (row["recovered"], row["still_invalid"]) == (len(invalid) - prose, prose)
    assert row["unchanged"] == len(before) - len(invalid) and row["changed"] == 0
    assert row["invalid_categories"] == {"no_json": prose}


def test_reparse_never_writes_into_its_source_or_over_a_store(old_store: Path, tmp_path: Path) -> None:
    lines = JudgementStore(old_store).path("tournament").read_text()
    with pytest.raises(IdentityError, match="never into its source"):
        reparse(old_store, old_store)
    reparse(old_store, tmp_path / "v2")
    with pytest.raises(IdentityError, match="already holds a judgement store"):
        reparse(old_store, tmp_path / "v2")
    assert JudgementStore(old_store).path("tournament").read_text() == lines
    code = CliRunner().invoke(judge_group, ["reparse", "--judgements", str(old_store), "--out", str(old_store)])
    assert code.exit_code == 11


def test_a_remote_store_is_refused_as_the_target(
    old_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rcp_ndcg.errors import ConfigError

    monkeypatch.chdir(tmp_path)  # a regression must not write a store into the checkout
    with pytest.raises(ConfigError, match="local") as refused:
        reparse(old_store, "s3://bucket/reparsed")
    assert "--mirror" in (refused.value.hint or "")
