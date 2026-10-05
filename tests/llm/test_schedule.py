"""The judging schedules: the paper's numbers, stated once, and the window numerics they drive."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import pytest
from pydantic import ValidationError

from rcp_ndcg.llm import schedule as sched
from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule


def test_the_default_tournament_is_the_papers_216_calls_with_56_adaptive_at_a_pool_of_150() -> None:
    schedule = TournamentSchedule()
    assert (schedule.window, schedule.mirror, schedule.adaptive_window, schedule.adaptive_batches) == (10, True, 10, 7)
    # Exactly the paper's counts at its pool, whatever the float rounding of the placements.
    assert schedule.windows_for(150) == (53, 27, 8)
    assert schedule.phase_calls(150) == (2 * (53 + 27), 56)
    assert schedule.calls_per_query(150) == 216


def test_the_default_rubric_is_the_papers_100_windows_half_random_at_a_pool_of_150() -> None:
    schedule = RubricSchedule()
    assert (schedule.window, schedule.random_share) == (10, 0.5)
    assert schedule.windows_for(150) == (50, 50)
    assert schedule.calls_per_query(150) == 100


def test_the_placements_are_the_papers_window_counts_as_exact_fractions() -> None:
    tournament, rubric = TournamentSchedule(), RubricSchedule()
    assert tournament.random_placements == pytest.approx(53 * 10 / 150, rel=1e-15)
    assert tournament.stratified_placements == pytest.approx(1.8, rel=1e-15)
    assert tournament.adaptive_placements == pytest.approx(7 * 8 * 10 / 150, rel=1e-15)
    assert rubric.placements_per_doc == pytest.approx(100 * 10 / 150, rel=1e-15)


def test_window_counts_scale_with_the_pool() -> None:
    tournament, rubric = TournamentSchedule(), RubricSchedule()
    assert tournament.windows_for(300) == (106, 54, 16)
    assert tournament.windows_for(75) == (26, 14, 4)  # 26.5 rounds half to even
    assert rubric.windows_for(300) == (100, 100)
    assert rubric.windows_for(30) == (10, 10)


def test_media_schedules_change_only_the_window() -> None:
    assert TournamentSchedule.for_modality("text") == TournamentSchedule()
    image = TournamentSchedule.for_modality("image")
    assert image == TournamentSchedule(window=5, adaptive_window=5)
    # The text schedule's placements in windows of 5: the paper's 106 random and 54 stratified windows at 150; the
    # adaptive phase asks 7 x 16 windows (the placements' exact count).
    assert image.windows_for(150) == (106, 54, 16)
    assert RubricSchedule.for_modality("image").windows_for(150) == (62, 63)
    assert RubricSchedule.for_modality("video").windows_for(150) == (100, 100)
    assert RubricSchedule.for_modality("video").placements_per_doc == RubricSchedule().placements_per_doc


def test_small_pools_and_subsets() -> None:
    assert TournamentSchedule().calls_per_query(1) == 0
    # A pool smaller than a window: windows of the whole pool, so the placements per document hold.
    assert TournamentSchedule().windows_for(4) == (4, 2, 1)  # round(3.53), round(1.8), round(0.53)
    assert TournamentSchedule().calls_per_query(4) == 2 * (4 + 2) + 1
    assert TournamentSchedule().windows_for(8) == (4, 2, 1)
    # A pool no larger than the adaptive window: one adaptive window in total, whatever the placements -- every
    # batch would ask the whole pool in its current order, and a repeated window answers the same thing again.
    assert TournamentSchedule(stratified_placements=5, adaptive_placements=20).windows_for(8) == (4, 5, 1)
    assert TournamentSchedule(stratified_placements=5, adaptive_placements=20).calls_per_query(8) == 2 * (4 + 5) + 1
    assert TournamentSchedule(adaptive_batches=0).windows_for(8) == (4, 2, 0)
    rubric = RubricSchedule()
    assert rubric.windows_for(0) == (0, 0)
    assert rubric.windows_for(4) == (4, 3)  # round(6.67) windows of the whole pool; 3.5 rounds half to even
    assert rubric.calls_per_query(4) == 7
    # A re-judged subset of 15 documents gets the windows its size gives, as any pool of 15.
    assert rubric.windows_for(15) == (5, 5)
    # Every document is seen at least once, and every chunk when documents are judged in chunks.
    assert sum(RubricSchedule(placements_per_doc=0.2).windows_for(95)) == 10
    assert sum(RubricSchedule(placements_per_doc=0.2).windows_for(15, n_units=95)) == 10
    # At least one random window, however small the share.
    assert RubricSchedule(random_share=0.01).windows_for(20) == (1, 12)


def test_a_pool_no_larger_than_the_adaptive_window_runs_one_adaptive_batch() -> None:
    """Every adaptive window of such a pool is the whole pool in its current order, so another batch repeats it."""
    schedule = TournamentSchedule()
    assert schedule.adaptive_batches_for(2) == 1
    assert schedule.adaptive_batches_for(10) == 1
    assert schedule.adaptive_batches_for(11) == 7  # above the adaptive window, distinct windows are possible again
    assert schedule.adaptive_batches_for(150) == 7
    assert TournamentSchedule(adaptive_batches=0).adaptive_batches_for(8) == 0
    assert TournamentSchedule(adaptive_batches=3).adaptive_batches_for(8) == 1
    # The estimate and the pass read the same rule.
    assert schedule.phase_calls(8) == (2 * (4 + 2), 1)
    assert schedule.phase_calls(150) == (2 * (53 + 27), 56)


@pytest.mark.parametrize("n", [2, 4, 7, 10, 12, 23, 150])
def test_placements_per_document_hold_for_any_pool(n: int) -> None:
    """Each phase shows a document its placements, to rounding, however small the pool."""
    tournament, rubric = TournamentSchedule(), RubricSchedule()
    random_windows, stratified, _ = tournament.windows_for(n)
    window = min(tournament.window, n)
    assert abs(random_windows * window / n - tournament.random_placements) <= window / (2 * n)
    assert abs(stratified * window / n - tournament.stratified_placements) <= window / (2 * n)
    window = min(rubric.window, n)
    assert abs(sum(rubric.windows_for(n)) * window / n - rubric.placements_per_doc) <= window / (2 * n)


def test_a_window_of_the_whole_pool_is_drawn_in_its_own_order_each_time() -> None:
    groups = sched._balanced_groups(4, 10, 4, random.Random(0))
    assert all(sorted(group) == [0, 1, 2, 3] for group in groups)
    assert len({tuple(group) for group in groups}) > 1
    ids = ["a", "b", "c", "d"]
    theta = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    stratified = sched._stratified_groups(ids, theta, 4, 3, random.Random(0))
    assert len(stratified) == 3 and all(sorted(group) == ids for group in stratified)


def test_placements_are_positive_and_the_random_share_a_share() -> None:
    with pytest.raises(ValidationError):
        RubricSchedule(random_share=1.5)
    with pytest.raises(ValidationError):
        RubricSchedule(placements_per_doc=0)
    with pytest.raises(ValidationError):
        TournamentSchedule(random_placements=-1)


def test_each_query_draws_from_its_own_stream() -> None:
    first = [sched.query_rng(42, "ds", "q1").random() for _ in range(2)]
    assert first == [sched.query_rng(42, "ds", "q1").random() for _ in range(2)]
    assert first != [sched.query_rng(42, "ds", "q2").random() for _ in range(2)]


def _numerics_digest() -> str:
    """Every window the scheduler draws on a grid of pools and seeds, as one digest."""
    out = []
    for seed in range(6):
        for n in (7, 23, 150):
            rng = random.Random(seed)
            ids = [f"d{i}" for i in range(n)]
            groups = sched._balanced_groups(n, 10, 53, rng)
            theta = {d: rng.gauss(0, 1.5) for d in ids}
            stratified = sched._stratified_groups(ids, theta, min(10, n), 27, rng)
            observed: Counter[tuple[str, str]] = Counter()
            for group in groups:
                for i, a in enumerate(group):
                    for b in group[i + 1 :]:
                        observed[sched._canonical_pair(ids[a], ids[b])] += 1
            boundaries = sched._compute_boundary_values(theta, observed, top_k=150)
            adaptive = sched._greedy_select_windows(theta, boundaries, min(10, n), num_windows=8, overlap_discount=0.3)
            out.append((groups, stratified, boundaries, adaptive))
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()


def test_the_window_numerics_are_the_reference_implementations() -> None:
    """Pinned against the scheduler the paper's runs used: any change to a draw changes the digest.

    The draws at pools larger than a window (23, 150) are the paper's; a pool smaller than a window (7) draws each
    window of the whole pool in its own order.
    """
    assert _numerics_digest() == "11707e05029e6730160d7022573929be5e9c188c7f9d1b2b829acc6c00161389"
