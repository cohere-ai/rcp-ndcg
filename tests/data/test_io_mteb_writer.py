"""The MTEB writer's layout, round-tripped through the product's own Hub reader.

``tests/eval/mteb/test_export.py`` runs the same writer against mteb's own loader, but that file needs the
``[mteb]`` extra; these round trips run everywhere (``hub_stubs`` serves the written directory as a Hub
repository at a pinned commit).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.io import hub as hub_module
from rcp_ndcg.data.io.mteb import MtebWriter
from rcp_ndcg.errors import ConfigError
from tests.data.hub_stubs import SHA, _fake_hub_file, _fake_listing


def a_dataset(**overrides) -> Dataset:
    records = dict(
        name="NanoArguAnaRetrieval",
        queries=[{"query_id": "q1", "text": "what is a tortoise"}, {"query_id": "q2", "text": "how long"}],
        corpus=[{"doc_id": "d1", "text": "a tortoise"}, {"doc_id": "d2", "text": "other"}],
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2},
            {"query_id": "q2", "doc_id": "d2", "grade": 1},
        ],
    )
    records.update(overrides)
    return Dataset.from_records(**records)


def write_and_reload(dataset: Dataset, tmp_path: Path, **kwargs) -> Dataset:
    """Write the layout and read it back through the Hub reader (the fake serves the written directory)."""
    out = tmp_path / "layout"
    MtebWriter().write_dataset(dataset, str(out), **kwargs)
    with (
        mock.patch.object(hub_module, "_hub_file", _fake_hub_file(out)),
        mock.patch.object(hub_module, "_hub_listing", _fake_listing(out)),
    ):
        reloaded = load_dataset(f"hf://o/r/default@{SHA}")
        # The tables are lazy: read them while the fake Hub serves the written directory.
        _ = reloaded.queries, reloaded.corpus
        return reloaded


def test_a_task_instruction_round_trips_through_the_written_layout(tmp_path: Path) -> None:
    """The writer wrote no task instruction: a lifted BRIGHT instruction vanished on export, and the Hub
    reader reloaded task_instruction None, so every role formatted its queries without the task's frame."""
    dataset = a_dataset(task_instruction="Given a claim, find documents that refute the claim")

    reloaded = write_and_reload(dataset, tmp_path)

    assert reloaded.task_instruction == "Given a claim, find documents that refute the claim"
    assert all(query.instruction is None for query in reloaded.queries.values()), "lifted back, not per-query"


def test_a_per_side_task_instruction_needs_a_card(tmp_path: Path) -> None:
    """One instruction column cannot hold mteb's {query, document} prompt: refused rather than dropped."""
    dataset = a_dataset(task_instruction={"query": "q side", "document": "d side"})

    with pytest.raises(ConfigError, match="per-side"):
        MtebWriter().write_dataset(dataset, str(tmp_path / "layout"))


def test_a_documents_title_is_written_verbatim(tmp_path: Path) -> None:
    """A title's own whitespace is part of the model's text: mteb joins ``(title + ' ' + body).strip()``, so
    stripping the title at write time changed the joined text between the source and a republished layout."""
    dataset = a_dataset(corpus=[{"doc_id": "d1", "title": "Foo ", "text": "body"}, {"doc_id": "d2", "text": "other"}])
    out = tmp_path / "layout"

    MtebWriter().write_dataset(dataset, str(out))

    rows = pq.read_table(out / "corpus" / "test-00000-of-00001.parquet").to_pylist()
    assert [row["title"] for row in rows] == ["Foo ", ""]


def test_gains_without_thetas_round_trip(tmp_path: Path) -> None:
    """The writer emits the gain column whenever gains is given and the theta column only when thetas is
    given; the reader used to read a gain only when the same row's theta was non-null, so a gains-only
    dataset reloaded ``gains=None`` with no warning."""
    dataset = a_dataset(
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2, "gain": 0.8},
            {"query_id": "q1", "doc_id": "d2", "grade": 0, "gain": 0.1},
        ],
    )

    reloaded = write_and_reload(dataset, tmp_path)

    assert reloaded.gains == dataset.gains
    assert reloaded.thetas is None
