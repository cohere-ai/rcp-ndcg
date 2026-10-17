"""``Rankings.save`` writes what ``load_rankings`` reads back, in every writable format."""

from __future__ import annotations

from pathlib import Path

import pytest

from rcp_ndcg.data import Rankings, load_rankings


@pytest.mark.parametrize("suffix", ["parquet", "jsonl", "csv"])
def test_a_saved_rankings_file_loads_back_equal(tmp_path: Path, suffix: str) -> None:
    rankings = Rankings.concat(
        [
            Rankings.from_scores({"q1": {"d1": 2.0, "d2": 1.5}}, system="a"),
            Rankings.from_scores({"q1": {"d2": 0.5}}, system="b"),
        ]
    )

    written = rankings.save(tmp_path / f"run.{suffix}")

    assert load_rankings(written) == rankings
