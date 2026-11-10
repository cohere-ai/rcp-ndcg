"""A third-party runner, installed as a test distribution, loads, runs a step and fails typed.

The runner seam is the ``rcp.runners`` entry-point group: a plugin publishes a name and a class, the registry
resolves it with :func:`~rcp.runners.registry.get_runner`, and the run layer calls the four methods.  This test
is the plugin's own suite: it writes a real distribution (a module and its ``.dist-info`` with
``entry_points.txt``) into ``tmp_path``, puts it on ``sys.path``, and drives it exactly as ``rcp run start
--runner`` does.  The distribution exists only for the test, so a checkout without it is green.

The contract itself is :func:`rcp.testing.runner_conformance` (the four methods and the answers' shapes); this
test adds the two facts the helper cannot know: the entry point's name is the runner's ``name``, and a job that
exits 4 surfaces as :class:`~rcp.errors.MissingInputError`, the same typed error the built-in local runner
raises -- never a bare :class:`~rcp.runners.base.RunnerError`.

The broken runners are deliberately installed too: the conformance check must reject them, or the check would
pass a plugin that breaks the run layer.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from rcp.errors import ConfigError, MissingInputError
from rcp.runners import JobSpec, JobStatus, LocalRunner, get_runner
from rcp.testing import runner_conformance

DIST_NAME = "rcp-runner-conformance"
"""The test distribution's metadata name (never installed outside the test)."""

ENTRY_POINTS = {
    "conformance": "MinimalRunner",
    "broken-conformance": "BrokenRunner",
    "wrong-failure": "WrongFailureRunner",
}
"""The names the test distribution publishes, and the class each one names."""

#: The distribution's one module.  ``MinimalRunner`` delegates to the local supervisor (``LocalRunner``): the
#: point of the seam is the four methods, not a scheduler.  ``BrokenRunner`` answers a string where the run
#: layer reads a ``JobStatus``; ``WrongFailureRunner`` turns a job's own typed failure into ``RunnerError``.
RUNNER_MODULE = textwrap.dedent(
    """
    from rcp.errors import MissingInputError
    from rcp.runners import LocalRunner
    from rcp.runners.base import RunnerError


    class MinimalRunner(LocalRunner):
        \"\"\"A minimal plugin: the local supervisor, with one option of its own.\"\"\"

        name = "conformance"

        def __init__(self, *, marker: str = "minimal", **options):
            self.marker = marker
            super().__init__(**options)


    class BrokenRunner(MinimalRunner):
        \"\"\"A plugin that breaks the answer shape: ``status`` must be a JobStatus.\"\"\"

        name = "broken-conformance"

        def status(self, handle):
            return "completed"


    class WrongFailureRunner(MinimalRunner):
        \"\"\"A plugin that loses a job's typed failure: exit 4 is MissingInputError, not RunnerError.\"\"\"

        name = "wrong-failure"

        def submit(self, jobs):
            try:
                return super().submit(jobs)
            except MissingInputError as exc:
                raise RunnerError("the job failed") from exc
    """
).strip()


@contextmanager
def _test_distribution(tmp_path: Path) -> Iterator[None]:
    """Write and import the test distribution, then remove it (and its entry points) on exit."""
    site = tmp_path / "site"
    site.mkdir()
    module = site / "rcp_runner_conformance.py"
    module.write_text(RUNNER_MODULE, encoding="utf-8")
    dist_info = site / f"{DIST_NAME.replace('-', '_')}-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: 0.1\n", encoding="utf-8")
    lines = ["[rcp.runners]"]
    lines.extend(f"{name} = rcp_runner_conformance:{target}" for name, target in ENTRY_POINTS.items())
    (dist_info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(site))
        importlib.invalidate_caches()


def assert_plugin_conforms(runner: Any, *, name: str, tmp_path: Path) -> None:
    """The plugin-level conformance check: the public shape, a real step, and the typed failure.

    A third-party runner's own suite can copy this test, or call
    :func:`~rcp.testing.runner_conformance` and assert the failure mapping itself.

    Raises:
        AssertionError: the runner's ``name`` is not the entry point's, the public shape check fails, the step
            does not run, or a job exiting 4 does not surface as :class:`~rcp.errors.MissingInputError`.
    """
    assert runner.name == name, f"the runner's name is {runner.name!r}, not its entry point {name!r}"
    runner_conformance(runner, job=JobSpec(name="shape", argv=(sys.executable, "-c", "")))

    marker = tmp_path / "one-step.txt"
    step = JobSpec(
        name="one-step",
        argv=(
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran'); print('one step')",
        ),
    )
    handles = runner.submit([step])
    assert isinstance(handles, list) and len(handles) == 1, f"submit returned {handles!r}, not one handle"
    handle = handles[0]
    assert runner.status(handle) is JobStatus.COMPLETED, f"the step ended {runner.status(handle)!r}"
    assert marker.read_text(encoding="utf-8") == "ran", "the runner did not run the step"
    assert "one step" in runner.logs(handle), f"the step's output is not in its log: {runner.logs(handle)!r}"
    runner.cancel(handle)  # a finished job is left as it is
    assert runner.status(handle) is JobStatus.COMPLETED

    try:
        runner.submit([JobSpec(name="missing", argv=(sys.executable, "-c", "raise SystemExit(4)"))])
    except MissingInputError as exc:
        assert type(exc) is MissingInputError, f"the job's failure is a {type(exc).__name__}, not MissingInputError"
        assert exc.details == {"job": "missing", "exit_code": 4}, exc.details
        assert exc.retryable is False
    except Exception as exc:  # noqa: BLE001 - the point of the check is the class a plugin raises
        raise AssertionError(f"a job exiting 4 surfaced as {type(exc).__name__}, not MissingInputError: {exc}") from exc
    else:
        raise AssertionError("a job exiting 4 did not raise MissingInputError")


def test_a_minimal_runner_installed_as_a_distribution_loads_runs_and_fails_typed(tmp_path: Path) -> None:
    """The gate: the plugin is a real distribution, resolved through ``rcp.runners``, not a monkeypatched entry."""
    with _test_distribution(tmp_path):
        runner = get_runner("conformance", log_dir=str(tmp_path / "logs"), marker="hello")

        assert type(runner).__name__ == "MinimalRunner"
        assert type(runner).__module__ == "rcp_runner_conformance", "the runner did not come from the test distribution"
        assert getattr(runner, "marker", None) == "hello", "the runner's own option did not reach its constructor"
        assert_plugin_conforms(runner, name="conformance", tmp_path=tmp_path)

        with pytest.raises(ConfigError, match="invalid options for the 'conformance' runner"):
            get_runner("conformance", nonsense=True)


def test_a_runner_that_breaks_the_answer_shape_fails_the_conformance_check(tmp_path: Path) -> None:
    """``status`` returns a string: the run layer would compare it with JobStatus members, so the check fails."""
    with _test_distribution(tmp_path):
        runner = get_runner("broken-conformance", log_dir=str(tmp_path / "logs"))

        with pytest.raises(AssertionError, match="not a JobStatus"):
            assert_plugin_conforms(runner, name="broken-conformance", tmp_path=tmp_path)


def test_a_runner_that_misreports_a_failure_fails_the_conformance_check(tmp_path: Path) -> None:
    """A job's own error class is the caller's signal (resume, fix the input): a plugin must not flatten it."""
    with _test_distribution(tmp_path):
        runner = get_runner("wrong-failure", log_dir=str(tmp_path / "logs"))

        with pytest.raises(AssertionError, match="not MissingInputError"):
            assert_plugin_conforms(runner, name="wrong-failure", tmp_path=tmp_path)


def test_removing_the_test_distribution_leaves_the_suite_green(tmp_path: Path) -> None:
    """The distribution is the test's, not the checkout's: without it the group has no such name."""
    with _test_distribution(tmp_path):
        assert get_runner("conformance").name == "conformance"

    with pytest.raises(ConfigError, match="unknown runner 'conformance'"):
        get_runner("conformance")


def test_the_built_in_local_runner_is_the_failure_reference(tmp_path: Path) -> None:
    """The typed error the plugin must match is the one the local supervisor itself raises."""
    with pytest.raises(MissingInputError) as built_in:
        LocalRunner(log_dir=str(tmp_path / "logs")).submit(
            [JobSpec(name="missing", argv=(sys.executable, "-c", "raise SystemExit(4)"))]
        )

    assert type(built_in.value) is MissingInputError
    assert built_in.value.details == {"job": "missing", "exit_code": 4}
