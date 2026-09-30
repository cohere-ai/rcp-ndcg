"""The paper's judging schedules, run through judge() on one query of 150 candidates with the fake judge.

``tests/llm/test_schedule.py`` pins the schedule's numbers; these tests pin what judge() does with them:
which windows it asks in which phase, and when it refits.

* Tournament: 53 balanced random windows and 27 windows over tiers of the preliminary Bradley-Terry
  ability, each also asked reversed, then 7 batches of at most 8 adaptive windows over the top 150 with a
  refit after each batch: 2 * (53 + 27) + 56 = 216 calls.
* Rubric: 100 windows, the first 50 balanced random, the other 50 over tiers of the preliminary Rasch ability.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.irt import Priors
from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator
from rcp_ndcg_core.irt._rasch import RaschEstimator
from rcp_ndcg_core.schemas import Judgement

from rcp_ndcg.calibration._projection import bradley_terry, namespace
from rcp_ndcg.llm import RubricSchedule, TournamentSchedule, judge
from rcp_ndcg.testing import FakeJudge

N_DOCS = 150
DOC_IDS = [f"d{index:03d}" for index in range(N_DOCS)]
ABILITY = {doc: -3.0 + 6.0 * index / (N_DOCS - 1) for index, doc in enumerate(DOC_IDS)}
ROWS = [RankingExample(query_id="q", query="a query", doc_ids=DOC_IDS, docs=[f"document {doc}" for doc in DOC_IDS])]


def _judge_recording_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, estimator: type):
    """Judge the query; return the result, its judgements in window order and ``(calls so far, abilities)`` per fit."""
    fake = FakeJudge(lambda text: ABILITY[text.split()[-1]])
    fits: list[tuple[int, dict[str, float]]] = []
    fit = estimator.fit_lbfgs

    def recording_fit(self, *args, **kwargs):
        result = fit(self, *args, **kwargs)
        fits.append((fake.usage.requests, dict(self.get_scores() or {})))
        return result

    monkeypatch.setattr(estimator, "fit_lbfgs", recording_fit)
    schedule = TournamentSchedule() if stage == "tournament" else RubricSchedule()
    result = judge(ROWS, None, fake, stage=stage, out=tmp_path, schedule=schedule)
    windows = sorted(result.judgements, key=lambda judgement: judgement.window_seq)
    assert [j.window_seq for j in windows] == list(range(len(windows)))
    assert fake.usage.requests == len(windows)
    return result, windows, fits


def _ids(judgement: Judgement) -> list[str]:
    return [placement.doc_id for placement in judgement.placements]


def _is_tier(window: list[str], ability: dict[str, float]) -> bool:
    """Whether ``window`` is a run of consecutive documents in the order of ``ability`` (highest first)."""
    order = sorted(DOC_IDS, key=lambda doc: ability.get(doc, 0.0), reverse=True)
    positions = sorted(order.index(doc) for doc in window)
    return positions == list(range(positions[0], positions[0] + len(window)))


def _assert_balanced(windows: list[list[str]]) -> None:
    """Every document is drawn equally often, give or take one."""
    counts = Counter(doc for window in windows for doc in window)
    assert set(counts) == set(DOC_IDS)
    assert max(counts.values()) - min(counts.values()) <= 1


def _assert_mirrored(windows: list[Judgement]) -> None:
    assert len(windows) % 2 == 0
    for first, second in zip(windows[::2], windows[1::2], strict=True):
        assert _ids(second) == _ids(first)[::-1]


def test_the_tournament_asks_the_papers_216_windows_in_three_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, windows, fits = _judge_recording_fits(tmp_path, monkeypatch, "tournament", BradleyTerryEstimator)
    assert len(windows) == 216
    assert all(len(_ids(window)) == 10 for window in windows)

    # A preliminary fit after the random windows, one after the stratified windows, one after each adaptive batch.
    assert [calls for calls, _ in fits] == [106, 160, 168, 176, 184, 192, 200, 208, 216]

    random, stratified, adaptive = windows[:106], windows[106:160], windows[160:]
    assert [window.phase for window in windows] == ["random"] * 106 + ["stratified"] * 54 + ["adaptive"] * 56
    _assert_mirrored(random)
    _assert_balanced([_ids(window) for window in random[::2]])
    assert not any(_is_tier(_ids(window), fits[0][1]) for window in random[::2][:5]), "random windows are not tiers"

    _assert_mirrored(stratified)
    assert all(_is_tier(_ids(window), fits[0][1]) for window in stratified)

    # Adaptive: 7 batches of 8, not mirrored; each window is ten consecutive documents of the fit before its
    # batch, and the windows range over the whole top 150, not only its head.
    starts = []
    for batch in range(7):
        ability = fits[1 + batch][1]
        order = sorted(ability, key=lambda doc: ability[doc], reverse=True)
        for window in adaptive[8 * batch : 8 * (batch + 1)]:
            starts.append(order.index(_ids(window)[0]))
            assert _ids(window) == order[starts[-1] : starts[-1] + 10]
    assert min(starts) == 0 and max(starts) >= 100

    # The calibration refits the stored windows through the grammar the live tournament fitted them with. The
    # refit starts cold while the live fit was warm-started batch by batch; both stop within the optimiser's
    # tolerance of the same optimum (2e-3 apart here, as for the paper's released scores).
    live = fits[-1][1]
    thetas, _ = bradley_terry(result, l2=Priors().bt_l2)
    assert thetas[namespace("dataset", "q")] == pytest.approx(live, abs=5e-3)


def test_the_rubric_asks_the_papers_50_random_and_50_stratified_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, windows, fits = _judge_recording_fits(tmp_path, monkeypatch, "rubric", RaschEstimator)
    assert len(windows) == 100
    assert all(len(_ids(window)) == 10 for window in windows)
    assert [calls for calls, _ in fits] == [50]

    random, stratified = windows[:50], windows[50:]
    assert [window.phase for window in windows] == ["random"] * 50 + ["stratified"] * 50
    _assert_balanced([_ids(window) for window in random])
    assert all(_is_tier(_ids(window), fits[0][1]) for window in stratified)
    assert not all(_is_tier(_ids(window), fits[0][1]) for window in random)
