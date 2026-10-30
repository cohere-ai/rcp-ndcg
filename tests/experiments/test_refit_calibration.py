"""Offline checks for the public-data 2PL refit experiment's joins and holdout."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments import refit_calibration as experiment


def _files(tmp_path: Path, *, missing_document: bool = False) -> dict[str, Path]:
    run_file = tmp_path / "runs.parquet"
    rubric_file = tmp_path / "rubric.parquet"
    runs = pd.DataFrame(
        [
            (experiment.TOURNAMENT_MODEL, True, "q1", "d1", -1.0),
            (experiment.TOURNAMENT_MODEL, True, "q1", "d2", 1.0),
            ("Qwen3.5-397B-A17B-NVFP4 (criteria, judge)", True, "q1", "d1", 88.0),
        ],
        columns=["model", "is_judge", "query-id", "corpus-id", "score"],
    )
    rubric = pd.DataFrame(
        [
            ("q1", "d1", 1, 0, 0, 0, 0),
            ("q1", "d1", 0, 0, 0, 0, 0),
            ("q1", "d3" if missing_document else "d2", 1, 1, 0, 0, 0),
        ],
        columns=["query-id", "corpus-id", *experiment.CRITERIA],
    )
    runs.to_parquet(run_file)
    rubric.to_parquet(rubric_file)
    return {
        "provenance/runs/NanoSciFactRetrieval.parquet": run_file,
        "provenance/judge_criteria/NanoSciFactRetrieval.parquet": rubric_file,
    }


def test_load_subset_uses_only_tournament_scores_and_keeps_placements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _files(tmp_path)
    monkeypatch.setattr(experiment, "fetch", lambda *args: None)
    monkeypatch.setattr(experiment, "local_path", lambda _repo, path: paths[path])

    observations, scores = experiment.load_subset("NanoSciFactRetrieval")

    query = "NanoSciFactRetrieval||q1"
    assert scores == {query: {"d1": -1.0, "d2": 1.0}}
    assert len(observations[query]) == 3
    assert [doc for doc, _ in observations[query]].count("d1") == 2


def test_load_subset_rejects_unscored_rubric_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _files(tmp_path, missing_document=True)
    monkeypatch.setattr(experiment, "fetch", lambda *args: None)
    monkeypatch.setattr(experiment, "local_path", lambda _repo, path: paths[path])

    with pytest.raises(ValueError, match="rubric documents without tournament scores"):
        experiment.load_subset("NanoSciFactRetrieval")


def test_holdout_keeps_all_placements_for_a_document_together() -> None:
    rows = {"q1": [(f"d{i}", {"C1": i % 2}) for i in range(10) for _ in range(3)]}

    train, heldout = experiment.split_documents(rows, fraction=0.2)

    train_docs = {doc for doc, _ in train["q1"]}
    heldout_docs = {doc for doc, _ in heldout["q1"]}
    assert not train_docs & heldout_docs
    assert len(heldout_docs) == 2
    assert len(heldout["q1"]) == 6
    assert experiment.split_documents(rows, fraction=0.2) == (train, heldout)


def test_holdout_requires_two_documents_per_query() -> None:
    with pytest.raises(ValueError, match="two or more judged documents"):
        experiment.split_documents({"q1": [("only", {"C1": 1})]})


@pytest.mark.skipif(sys.platform == "win32", reason="the package's existing fcntl import needs POSIX")
def test_constant_baseline_uses_training_rates_and_reports_all_verdicts() -> None:
    zeros = {criterion: 0 for criterion in experiment.CRITERIA}
    first = dict(zeros, C1=1)
    train = {"q1": [("d1", first), ("d2", zeros)]}
    heldout = {"q1": [("d3", first)]}

    report = experiment._constant_baseline(train, heldout)

    assert report["per_criterion"]["C1"]["train_pass_rate"] == 0.5
    assert report["per_criterion"]["C1"]["brier"] == pytest.approx(0.25)
    assert report["overall"]["count"] == 5
    assert report["overall"]["brier"] == pytest.approx(0.05)


@pytest.mark.skipif(sys.platform == "win32", reason="the package's existing fcntl import needs POSIX")
def test_raw_bt_logistic_baseline_uses_scores_and_reports_all_verdicts() -> None:
    scores = {f"d{i}": (i - 15) / 5 for i in range(30)}

    def verdict(i: int) -> dict[str, int]:
        label = int((i >= 10) != (i in (4, 16)))
        return {criterion: label for criterion in experiment.CRITERIA}

    train = {"q1": [(f"d{i}", verdict(i)) for i in range(20)]}
    heldout = {"q1": [(f"d{i}", verdict(i)) for i in range(20, 30)]}
    logistic = experiment._logistic_baseline(train, heldout, {"q1": scores})
    constant = experiment._constant_baseline(train, heldout)

    assert logistic["overall"]["count"] == 50
    assert logistic["overall"]["brier"] < constant["overall"]["brier"]
    for criterion in experiment.CRITERIA:
        row = logistic["per_criterion"][criterion]
        assert row["count"] == 10
        assert row["raw_bt_slope"] > 0
        assert math.isfinite(row["brier"])
