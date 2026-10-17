"""One finished offline run shared by the results tests (the full pipeline on the tiny world)."""

from __future__ import annotations

import json
from pathlib import Path

from rcp_ndcg.data import Rankings
from rcp_ndcg.runs import Pipeline
from rcp_ndcg.testing import tiny_rows
from tests.runs.conftest import tiny_config


def build_run(root: Path, *, systems: dict[str, str] | None = None) -> Path:
    """Run the tiny pipeline offline under ``root`` and return the run directory.

    The run scores one system of its own (``mine``) besides the reference systems ``candidates`` and ``judge``.
    """
    root.mkdir(parents=True, exist_ok=True)
    rows, _ = tiny_rows()
    data = root / "rows.jsonl"
    data.write_text(
        "".join(
            json.dumps({"query_id": r.id, "query": r.text, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    if systems is None:
        mine = root / "mine.parquet"
        Rankings.from_orders({r.id: list(r.doc_ids) for r in rows}, system="mine").save(mine)
        systems = {"mine": str(mine)}
    pipeline = Pipeline(tiny_config(data, evaluation={"k": 5, "systems": systems}), runs_dir=str(root / "runs"))
    pipeline.run()
    return Path(pipeline.layout.root)
