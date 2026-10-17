"""Fixtures shared by the retrieval tests: the BEIR toy dataset (three documents, two queries)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rcp_ndcg.data import load_dataset

DOCS = {
    "d1": "tortoises move slowly across the sand",
    "d2": "hares run fast in open fields",
    "d3": "the sand dunes of the desert",
}


@pytest.fixture
def dataset(tmp_path: Path) -> Any:
    """A three-document, two-query BEIR directory (every document relevant to exactly one query)."""
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "text": t}) + "\n" for d, t in DOCS.items()))
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "slow tortoises"})
        + "\n"
        + json.dumps({"_id": "q2", "text": "fast hares"})
        + "\n"
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n")
    return load_dataset(f"beir:{root}")
