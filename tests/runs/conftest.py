"""The run tests' helpers: the step list and the config a tiny offline run builds from.

The two fixtures the run tests request by name (``data``, the tiny rankings file, and ``finished``, one
completed run) live in ``tests/conftest.py``: pytest binds a conftest's fixtures to the directory collector
object it first saw, and a file argument makes pytest collect that directory's parent package again, which
leaves a subdirectory's own fixtures invisible to the files under the new collector. The root collector is
never re-created, so the fixtures there reach every run test whatever the argument set. The helpers below
are plain functions, imported by name, so their home is this file.
"""

from __future__ import annotations

from pathlib import Path

from rcp.runs import RunConfig
from rcp.testing import TINY_RUBRIC, TINY_TOURNAMENT

STEPS = ["tournament", "rubric", "calibrate", "evaluate"]


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
