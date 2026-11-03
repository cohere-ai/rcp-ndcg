"""Run a scheduler's command-line tool and turn failures into :class:`RunnerError`."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from rcp_ndcg.runners.base import RunnerError


def run_cli(argv: Sequence[str], *, input_text: str | None = None) -> str:
    """Run ``argv`` and return its stdout.

    The tool's output is decoded as UTF-8 whatever the host locale, an undecodable byte replaced with U+FFFD
    (:meth:`~rcp_ndcg.runners.slurm.SlurmRunner.logs` reads its output file the same way), so one odd byte in a
    scheduler's log does not fail the run.

    Raises:
        RunnerError: the tool is not on ``PATH`` (not retryable) or exits non-zero (the message
            carries its stderr).
    """
    try:
        # The scheduler's own bytes, decoded the same way whatever the host locale: `text=True` alone used the
        # locale's encoding, so a tool emitting a non-UTF-8 byte crashed the run with UnicodeDecodeError instead of
        # a rendered log (`slurm.logs` reads the output file the same way).
        proc = subprocess.run(
            list(argv),
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
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
