"""Run a scheduler's command-line tool and turn failures into :class:`RunnerError`."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from rcp_ndcg.runners.base import RunnerError


def run_cli(argv: Sequence[str], *, input_text: str | None = None) -> str:
    """Run ``argv`` and return its stdout.

    Raises:
        RunnerError: the tool is not on ``PATH`` (not retryable) or exits non-zero (the message
            carries its stderr).
    """
    try:
        proc = subprocess.run(list(argv), input=input_text, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RunnerError(
            f"`{argv[0]}` not found on PATH; it is needed to talk to the scheduler",
            hint=f"run this on a host where `{argv[0]}` is installed (a login node of the cluster, a host with "
            "kubectl configured)",
            retryable=False,
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RunnerError(f"`{' '.join(argv)}` exited {proc.returncode}: {detail}")
    return proc.stdout


__all__ = ["run_cli"]
