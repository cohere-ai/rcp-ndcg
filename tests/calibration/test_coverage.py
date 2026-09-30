"""The calibration's window coverage: invalid windows per query, the warnings they raise, and the strict refusal.

A query is flagged when more than 5% of its windows of a stage the fit reads are invalid, or when any of its
adaptive tournament windows is. The tiny world's store has 13 tournament and 8 rubric windows per query; the
tests invalidate some of them.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from rcp_ndcg_core.schemas import Family, Judgement, JudgementSet, Placement

from rcp_ndcg.calibration import Calibration, calibrate
from rcp_ndcg.calibration.coverage import coverage_flags, window_coverage
from rcp_ndcg.cli.calibration import calibration_group
from rcp_ndcg.errors import DataError, RcpNdcgWarning
from rcp_ndcg.eval import evaluate
from rcp_ndcg.llm import JudgementStore
from rcp_ndcg.testing import TinyWorld


def _invalid(judgement: Judgement, category: str = "truncated") -> Judgement:
    return Judgement.model_validate(
        {**judgement.model_dump(), "valid": False, "invalid_reason": "cut", "invalid_category": category}
    )


def _fit_only(judgements: JudgementSet, world: TinyWorld) -> JudgementSet:
    return judgements.select(lambda j: not any(p.doc_id in world.rejudged_docs for p in j.placements))


def _with_invalid(judgements: JudgementSet, broken: set[tuple[str, str, int]], category: str = "truncated"):
    """``judgements`` with the windows ``(stage, query_id, window_seq)`` in ``broken`` made invalid."""
    records = tuple(
        _invalid(j, category) if (j.stage, j.query_id, j.window_seq) in broken else j for j in judgements.judgements
    )
    return JudgementSet(judgements=records, families=judgements.families)


def _first(judgements: JudgementSet, stage: str, query_id: str, phase: str) -> int:
    return min(
        j.window_seq for j in judgements.judgements if (j.stage, j.query_id, j.phase) == (stage, query_id, phase)
    )


@pytest.fixture(scope="module")
def flagged(world: TinyWorld, judgements: JudgementSet) -> JudgementSet:
    """q1 loses one adaptive tournament window; q2 loses two of its rubric windows (25%)."""
    base = _fit_only(judgements, world)
    adaptive = _first(base, "tournament", "q1", "adaptive")
    return _with_invalid(base, {("tournament", "q1", adaptive), ("rubric", "q2", 0), ("rubric", "q2", 1)})


def test_clean_judgements_fit_without_a_warning(world: TinyWorld, judgements: JudgementSet) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RcpNdcgWarning)
        clean = calibrate(_fit_only(judgements, world))
    assert clean.warnings == [] and clean.coverage.flagged_queries == []
    assert clean.coverage.invalid_windows == {}
    assert {stage: count.model_dump() for stage, count in clean.coverage.windows.items()} == {
        "rubric": {"windows": 16, "invalid": 0},
        "tournament": {"windows": 26, "invalid": 0},
    }


def test_invalid_windows_are_counted_per_query_by_stage_phase_and_category(flagged: JudgementSet) -> None:
    with pytest.warns(RcpNdcgWarning) as caught:
        fitted = calibrate(flagged)

    (warning,) = [w.message for w in caught if isinstance(w.message, RcpNdcgWarning)]
    assert warning.code == "INVALID_WINDOWS"
    assert "2 with more than 5% invalid windows" in warning.message
    assert "1 with an invalid adaptive window" in warning.message
    invalid = {
        q: {st: e.model_dump() for st, e in stages.items()} for q, stages in fitted.coverage.invalid_windows.items()
    }
    assert invalid["dataset||q1"] == {
        "tournament": {
            "windows": 13,
            "invalid": 1,
            "share": 1 / 13,
            "invalid_by_phase": {"adaptive": 1},
            "invalid_by_category": {"truncated": 1},
        }
    }
    assert invalid["dataset||q2"]["rubric"]["invalid_by_phase"] == {"random": 2}
    flags = {(flag.query, flag.stage): flag.reasons for flag in fitted.coverage.flagged_queries}
    assert flags == {("dataset||q1", "tournament"): ["share", "adaptive"], ("dataset||q2", "rubric"): ["share"]}
    assert fitted.warnings == [warning.to_dict()] == [w.model_dump() for w in fitted.diagnostics.warnings]


def test_a_strict_fit_refuses_a_flagged_query(flagged: JudgementSet) -> None:
    with pytest.raises(DataError, match="refusing a strict calibration") as caught:
        calibrate(flagged, strict=True)
    assert {flag["query"] for flag in caught.value.details["flagged"]} == {"dataset||q1", "dataset||q2"}


def test_only_the_stages_the_fit_reads_are_checked(world: TinyWorld, judgements: JudgementSet) -> None:
    base = _fit_only(judgements, world)
    broken = _with_invalid(base, {("tournament", "q1", _first(base, "tournament", "q1", "adaptive"))})
    with pytest.raises(DataError):
        calibrate(broken, strict=True)
    calibrate(broken, mode="rubric_only", strict=True)


def _windows(stage: str, phases: list[str], invalid: set[int]) -> list[Judgement]:
    family = Family(stage=stage, judge_model="m", prompt_hash="0" * 64, parse_version=2)  # type: ignore[arg-type]
    recorded_at = datetime.now(UTC)
    records = []
    for seq, phase in enumerate(phases):
        record = Judgement(
            record_id=f"r{seq}",
            dataset="d",
            query_id="q",
            stage=stage,  # type: ignore[arg-type]
            family_key=family.key,
            window_seq=seq,
            phase=phase,  # type: ignore[arg-type]
            placements=(Placement(position=1, doc_id="a", score=0.0), Placement(position=2, doc_id="b", score=1.0)),
            recorded_at=recorded_at,
        )
        records.append(_invalid(record, "no_json") if seq in invalid else record)
    return records


@pytest.mark.parametrize(
    ("invalid", "reasons"),
    [
        ({0}, []),  # 1 of 40 random windows: 2.5%
        ({0, 1}, []),  # exactly 5% is not more than 5%
        ({0, 1, 2}, ["share"]),
        ({39}, ["adaptive"]),
    ],
    ids=["one-random", "five-percent", "over-five-percent", "one-adaptive"],
)
def test_the_flag_is_more_than_five_percent_or_any_adaptive_window(invalid: set[int], reasons: list[str]) -> None:
    records = _windows("tournament", ["random"] * 30 + ["stratified"] * 8 + ["adaptive"] * 2, invalid)
    flags = coverage_flags(window_coverage(records, ("tournament",)))
    assert [flag.reasons for flag in flags] == ([reasons] if reasons else [])


def test_the_warning_reaches_the_saved_calibration_its_show_and_the_eval_report(
    flagged: JudgementSet, world: TinyWorld, tmp_path: Path
) -> None:
    with pytest.warns(RcpNdcgWarning):
        fitted = calibrate(flagged)
    fitted.save(tmp_path / "cal")
    loaded = Calibration.load(tmp_path / "cal")
    assert loaded.warnings == fitted.warnings and loaded.coverage == fitted.coverage

    shown = CliRunner().invoke(calibration_group, ["show", "--calibration", str(tmp_path / "cal"), "--json"])
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["data"]["warnings"][0]["code"] == "INVALID_WINDOWS"
    text = CliRunner().invoke(calibration_group, ["show", "--calibration", str(tmp_path / "cal")]).stdout
    assert "warning INVALID_WINDOWS" in text

    report = evaluate(world.rankings, dataset=world.dataset, gains=loaded, metrics=["rcp_ndcg"], bootstrap=0)
    assert [w.code for w in report.warnings if w.code == "INVALID_WINDOWS"] == ["INVALID_WINDOWS"]


def test_fit_strict_through_the_command(world: TinyWorld, flagged: JudgementSet, tmp_path: Path) -> None:
    store = JudgementStore(tmp_path / "store")
    for stage, entry in JudgementStore(world.judgements).identities().items():
        family = Family.model_validate(entry["family"])
        store.claim(stage, entry["identity"], family)  # type: ignore[arg-type]
    for judgement in flagged.judgements:
        store.append(judgement)
    args = ["fit", "--judgements", str(tmp_path / "store"), "--out", str(tmp_path / "cal")]

    strict = CliRunner().invoke(calibration_group, [*args, "--strict", "--json"])
    assert strict.exit_code == 12, strict.stdout
    assert not (tmp_path / "cal").exists()

    warned = CliRunner().invoke(calibration_group, [*args, "--json"])
    assert warned.exit_code == 0, warned.stdout
    envelope = json.loads(warned.stdout)
    assert [w["code"] for w in envelope["warnings"]] == ["INVALID_WINDOWS"]
