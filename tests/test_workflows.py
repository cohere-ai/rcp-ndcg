"""The GitHub workflows parse: a YAML error disables a whole workflow on GitHub, not only the broken step."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda path: path.name)
def test_the_workflow_parses_and_names_its_jobs(workflow: Path) -> None:
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and document.get("jobs"), f"{workflow.name} declares no jobs"
    for name, job in document["jobs"].items():
        steps = job.get("steps", [])
        assert all(isinstance(step, dict) for step in steps), f"{workflow.name}: a step of {name} is not a mapping"
