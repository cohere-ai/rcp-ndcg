"""Unit tests for :mod:`rcp_ndcg.retrieval.accel`.

These exercise the *single-process* branch of every helper (the
``num_processes == 1`` no-op fall-back).  Multi-process behaviour can
only be exercised under an ``accelerate launch`` invocation and is left
to integration testing.
"""

from __future__ import annotations

import pytest

# This module exercises the ``[local]`` extra.  Skipped rather than failed on a
# base install: the zero-GPU spine is meant to install without it, and a red
# suite there would say the package is broken when it is merely lighter.
pytest.importorskip("accelerate", reason="needs `uv sync --extra local`")

from unittest import mock

import pytest


def test_single_process_state_metadata() -> None:
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    assert state.process_index == 0
    assert state.num_processes == 1
    assert state.is_main_process is True


def test_shard_identity_when_single_rank() -> None:
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    items = list(range(7))
    local, idx = state.shard(items)
    assert local == items
    assert idx == list(range(7))


def test_shard_with_synthetic_world_size() -> None:
    """Re-run shard math by patching num_processes / process_index."""
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    with (
        mock.patch.object(type(state), "num_processes", new_callable=mock.PropertyMock, return_value=3),
        mock.patch.object(type(state), "process_index", new_callable=mock.PropertyMock, return_value=1),
    ):
        local, idx = state.shard(list(range(7)))
        # Round-robin stride: rank 1 of 3 covers positions 1, 4 (1 + 3k < 7).
        assert idx == [1, 4]
        assert local == [1, 4]


def test_shard_round_robin_covers_all_positions_disjointly() -> None:
    """Every position is owned by exactly one rank (a valid partition)."""
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    n, world = 7, 3
    covered: list[int] = []
    for rank in range(world):
        with (
            mock.patch.object(type(state), "num_processes", new_callable=mock.PropertyMock, return_value=world),
            mock.patch.object(type(state), "process_index", new_callable=mock.PropertyMock, return_value=rank),
        ):
            _, idx = state.shard(list(range(n)))
        covered.extend(idx)
    assert sorted(covered) == list(range(n))
    # Counts balanced to within one item.
    counts = [len(list(range(r, n, world))) for r in range(world)]
    assert max(counts) - min(counts) <= 1


def test_reorder_global_reassembles_disjoint_shards() -> None:
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    per_rank_items = [["a", "b"], ["c"], ["d", "e"]]
    per_rank_indices = [[0, 4], [2], [1, 3]]
    out = state.reorder_global(per_rank_items, per_rank_indices, total_size=5)
    assert out == ["a", "d", "c", "e", "b"]


def test_reorder_global_detects_missing_positions() -> None:
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    with pytest.raises(RuntimeError, match="missing 1 positions"):
        state.reorder_global([["x"]], [[0]], total_size=2)


def test_gather_object_single_process_is_identity() -> None:
    from rcp_ndcg.retrieval.accel import AccelState

    state = AccelState()
    payload = {"foo": [1, 2, 3]}
    assert state.gather_object(payload) == [payload]
