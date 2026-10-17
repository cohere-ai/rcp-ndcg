"""The runner seam's contract, as one check a third-party runner's own tests call.

:func:`runner_conformance` checks the de-facto contract of the ``rcp_ndcg.runners`` entry-point group: the four
:class:`~rcp_ndcg.runners.base.JobRunner` methods and the answers a caller depends on, plus the optional members
the run layer reads (``render``, ``renders_phases``, ``run_root``). It is the runner counterpart of
:func:`~rcp_ndcg.testing.io_conformance` and :func:`~rcp_ndcg.testing.results_conformance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcp_ndcg.runners.base import JobSpec, JobStatus


def runner_conformance(runner: Any, *, job: JobSpec | None = None) -> None:
    """The runner seam's contract, as one check a plugin runner's own tests call.

    Checks, collected as a list rather than a first failure:

    * ``name`` is a non-empty string and ``submit``, ``status``, ``logs`` and ``cancel`` are callable;
    * ``render``, when declared, is callable and returns one script per job name (``{job.name: str}``);
    * ``renders_phases``, when declared, is a bool, and ``run_root``, when declared, is a non-empty string;
    * with ``job``: ``submit([job])`` returns one non-empty handle per job, ``status(handle)`` answers a
      :class:`~rcp_ndcg.runners.base.JobStatus`, ``logs(handle)`` answers a string, and ``cancel(handle)`` does
      not raise (a job that already ended is left as it is).

    Args:
        runner: A constructed runner (``get_runner(name, **options)``), not a class: construction is the
            plugin's own business (its options are its own).
        job: A job the runner can actually run, for the behaviour checks; ``None`` checks the shape only. Pass a
            phased job whose phases start engines to exercise a runner that declares ``renders_phases``.

    Raises:
        AssertionError: naming every failed check, so a third party's test suite fails with the list -- never
            somewhere inside the run layer.
    """
    failures: list[str] = []
    cls = type(runner).__name__
    name = getattr(runner, "name", None)
    if not isinstance(name, str) or not name:
        failures.append(f"{cls}.name is not a non-empty string: {name!r}")
    for method in ("submit", "status", "logs", "cancel"):
        if not callable(getattr(runner, method, None)):
            failures.append(f"{cls}.{method} is missing or not callable")
    render = getattr(runner, "render", None)
    if render is not None and not callable(render):
        failures.append(f"{cls}.render is not callable")
    phases = getattr(runner, "renders_phases", None)
    if phases is not None and not isinstance(phases, bool):
        failures.append(f"{cls}.renders_phases is not a bool: {phases!r}")
    root = getattr(runner, "run_root", None)
    if root is not None and (not isinstance(root, str) or not root):
        failures.append(f"{cls}.run_root is not a non-empty string: {root!r}")
    if job is not None and callable(render):
        try:
            rendered = render([job])
        except Exception as exc:  # noqa: BLE001 - any rendering failure is a contract failure
            failures.append(f"{cls}.render([job]) raised {type(exc).__name__}: {exc}")
        else:
            if not isinstance(rendered, dict) or set(rendered) != {job.name}:
                failures.append(f"{cls}.render([job]) returned {rendered!r}, not {{job.name: script}}")
            elif not all(isinstance(text, str) for text in rendered.values()):
                failures.append(f"{cls}.render([job]) returned a non-string script")
    if job is None or failures:
        if failures:
            raise AssertionError("the runner contract failed:\n  - " + "\n  - ".join(failures))
        return
    try:
        handles = runner.submit([job])
    except Exception as exc:  # noqa: BLE001 - any submission failure is a contract failure
        failures.append(f"{cls}.submit([job]) raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(handles, Sequence) or isinstance(handles, str) or len(handles) != 1:
            failures.append(f"{cls}.submit([job]) returned {handles!r}, not one handle per job")
        else:
            handle = handles[0]
            if not isinstance(handle, str) or not handle:
                failures.append(f"{cls}.submit([job]) returned a handle that is not a non-empty string: {handle!r}")
            else:
                failures.extend(_handle_problems(runner, cls, handle))
    if failures:
        raise AssertionError("the runner contract failed:\n  - " + "\n  - ".join(failures))


def _handle_problems(runner: Any, cls: str, handle: str) -> list[str]:
    """The ``status``/``logs``/``cancel`` checks of one handle."""
    failures: list[str] = []
    try:
        state = runner.status(handle)
    except Exception as exc:  # noqa: BLE001 - any status failure is a contract failure
        failures.append(f"{cls}.status(handle) raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(state, JobStatus):
            failures.append(f"{cls}.status(handle) returned {state!r}, not a JobStatus")
    try:
        text = runner.logs(handle)
    except Exception as exc:  # noqa: BLE001 - any log failure is a contract failure
        failures.append(f"{cls}.logs(handle) raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(text, str):
            failures.append(f"{cls}.logs(handle) returned {type(text).__name__}, not str")
    try:
        runner.cancel(handle)
    except Exception as exc:  # noqa: BLE001 - any cancellation failure is a contract failure
        failures.append(f"{cls}.cancel(handle) raised {type(exc).__name__}: {exc}")
    return failures


__all__ = ["runner_conformance"]
