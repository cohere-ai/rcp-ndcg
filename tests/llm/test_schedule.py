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


def test_the_default_tournament_is_the_papers_216_calls_with_56_adaptive() -> None:
    schedule = TournamentSchedule()
    assert (schedule.window, schedule.random_windows, schedule.stratified_windows, schedule.mirror) == (
        10,
        53,
        27,
        True,
    )
    assert (schedule.adaptive_batches, schedule.adaptive_windows_per_batch) == (7, 8)
    assert schedule.adaptive_calls == 56
    assert schedule.calls_per_query(150) == 2 * (53 + 27) + 56 == 216


def test_the_default_rubric_is_the_papers_100_windows_half_random() -> None:
    schedule = RubricSchedule()
    assert (schedule.window, schedule.windows_per_query, schedule.random_windows) == (10, 100, 50)
    assert schedule.windows_for(150) == (50, 50)
    assert schedule.calls_per_query(150) == 100


def test_media_schedules_use_smaller_windows() -> None:
    assert TournamentSchedule.for_modality("text") == TournamentSchedule()
    image = TournamentSchedule.for_modality("image")
    assert (image.window, image.random_windows, image.stratified_windows, image.adaptive_calls) == (5, 106, 54, 119)
    assert RubricSchedule.for_modality("image").window == 8
    assert RubricSchedule.for_modality("video").windows_per_query == 200


def test_small_pools_and_subsets() -> None:
    assert TournamentSchedule().calls_per_query(1) == 0
    # A pool no larger than a window: one stratified window, one adaptive window per batch.
    assert TournamentSchedule().calls_per_query(8) == 2 * (53 + 1) + 7
    rubric = RubricSchedule()
    assert rubric.windows_for(0) == (0, 0)
    # A re-judged subset of 15 of 150 documents gets its share of the windows.
    assert rubric.windows_for(15, pool_size=150) == (5, 5)
    # Every document is seen at least once, and every chunk when documents are judged in chunks.
    assert sum(RubricSchedule(windows_per_query=2, random_windows=1).windows_for(95)) == 10
    assert sum(RubricSchedule(windows_per_query=2, random_windows=1).windows_for(15, n_units=95)) == 10


def test_random_windows_cannot_exceed_the_total() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        RubricSchedule(windows_per_query=10, random_windows=11)


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
    """Pinned against the scheduler the paper's runs used: any change to a draw changes the digest."""
    assert _numerics_digest() == "d10f77ef01ee07b9db4630e8804bac1e70f55d8775f701ff18e0f4f770e80a9c"
