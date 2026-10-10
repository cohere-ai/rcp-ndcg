"""The runner contract kit: what a plugin runner's own tests call before it ships."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rcp_ndcg.runners import JobSpec, JobStatus, LocalRunner
from rcp_ndcg.testing import runner_conformance


class GoodRunner:
    """A well-behaved plugin: the four methods, a renderer, the optional members declared."""

    name = "good"
    renders_phases = True
    run_root = "/scratch/runs"

    def __init__(self, **options: object) -> None:
        self.options = options

    def render(self, jobs: list[JobSpec]) -> dict[str, str]:
        return {job.name: "#!/usr/bin/env bash\n" for job in jobs}

    def submit(self, jobs: list[JobSpec]) -> list[str]:
        return [f"h{index}" for index, _ in enumerate(jobs, 1)]

    def status(self, handle: str) -> JobStatus:
        return JobStatus.COMPLETED

    def logs(self, handle: str, *, tail: int | None = None) -> str:
        return ""

    def cancel(self, handle: str) -> None:
        pass


def test_a_conforming_plugin_passes() -> None:
    runner_conformance(GoodRunner(queue="gpu"), job=JobSpec(name="j", argv=("true",)))
    # The shape-only check needs no job (a scheduler a unit test cannot reach).
    runner_conformance(GoodRunner())


def test_the_built_in_local_runner_passes(tmp_path: Path) -> None:
    runner = LocalRunner(log_dir=str(tmp_path / "logs"))
    runner_conformance(runner, job=JobSpec(name="contract", argv=(sys.executable, "-c", "")))


def test_a_broken_plugin_fails_with_every_problem_named() -> None:
    class Broken:
        name = ""
        renders_phases = "yes"
        run_root = ""
        render = "not callable"

        def submit(self, jobs: list[JobSpec]) -> list[str]:
            return ["h"]

    with pytest.raises(AssertionError) as failed:
        runner_conformance(Broken(), job=JobSpec(name="j", argv=("true",)))
    text = str(failed.value)
    assert "name is not a non-empty string" in text
    assert "status is missing or not callable" in text
    assert "logs is missing or not callable" in text
    assert "cancel is missing or not callable" in text
    assert "renders_phases is not a bool" in text
    assert "run_root is not a non-empty string" in text
    assert "render is not callable" in text


def test_the_answers_shapes_are_checked() -> None:
    class Wrong(GoodRunner):
        def submit(self, jobs: list[JobSpec]) -> list[str]:
            return ["h1", "h2"]  # one handle per job

        def status(self, handle: str) -> str:
            return "ok"  # not a JobStatus

    with pytest.raises(AssertionError) as failed:
        runner_conformance(Wrong(), job=JobSpec(name="j", argv=("true",)))
    assert "not one handle per job" in str(failed.value)

    class OneHandle(GoodRunner):
        def status(self, handle: str) -> str:
            return "ok"

    with pytest.raises(AssertionError, match="not a JobStatus"):
        runner_conformance(OneHandle(), job=JobSpec(name="j", argv=("true",)))
