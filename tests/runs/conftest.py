"""A tiny dataset and one finished offline run, shared by the run tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp_ndcg.runs import Pipeline, RunConfig
from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, tiny_rows

STEPS = ["tournament", "rubric", "calibrate", "evaluate"]


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two queries with their pools, bodies and labels, as a rankings file."""
    path = tmp_path_factory.mktemp("data") / "rows.jsonl"
    rows, _ = tiny_rows()
    lines = [
        json.dumps({"query_id": r.id, "query": r.query, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
        for r in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tiny_config(data: Path, **fields) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": "tiny",
            "dataset": f"jsonl:{data}",
            "judge": "fake",
            "steps": STEPS,
            "tournament": TINY_TOURNAMENT.model_dump(),
            "rubric": TINY_RUBRIC.model_dump(),
            **fields,
        }
    )


@pytest.fixture(scope="session")
def finished(data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One completed run; tests copy it before changing anything."""
    pipeline = Pipeline(tiny_config(data), runs_dir=str(tmp_path_factory.mktemp("runs")))
    pipeline.run()
    return Path(pipeline.layout.root)
