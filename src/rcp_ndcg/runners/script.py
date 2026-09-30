"""Render the script a task runs, from a :class:`JobSpec`.

Pure functions, no I/O. The script is the same for every runner; only how it
reaches the task differs (``bash -c`` locally, the body of an ``sbatch``
script, a Kubernetes container command)::

    #!/usr/bin/env bash
    set -euo pipefail
    cd <workdir>
    export KEY=value ...
    exec <argv>

``exec`` hands the process to the command, so a scheduler's SIGTERM reaches it
directly. On a **host** the command runs in the environment the script starts in
(an activated venv, a SLURM node with the package installed). In a **container**
nothing is built or maintained by this package: the job runs a stock image with
uv and Python 3.12 (:data:`COORDINATOR_IMAGE`), and an ``rcp-ndcg`` command is
run through ``uvx``, which installs this release from PyPI when the job starts,
held to the release's locked versions by its constraints file and to CPU-only
PyTorch (:func:`install_argv`). An image without uv (an engine's image, on Kubernetes) gets it from ``pip`` first.

A job that starts the judge's engine (``JobSpec.serve``) runs one supervision script, the same on SLURM and in a
Kubernetes pod (:func:`supervise`): it starts the engine once in the background, waits until a replica answers
its readiness path (at most ``startup_timeout_s``, and never after the engine has exited), runs the coordinator in
the background, and ends with whichever ends first. An engine that exits ends the job non-zero; a coordinator that
exits stops the engine, and its status is the job's. Resuming is cheap, so the job fails rather than hold its
allocation; retrying on a healthy node is the scheduler's business.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence

from rcp_ndcg.runners.base import JobSpec
from rcp_ndcg.support.serve import ServeConfig

#: A stock image with uv and Python 3.12: the coordinator installs the package into it when the job starts.
COORDINATOR_IMAGE = "ghcr.io/astral-sh/uv:python3.12-trixie-slim"
#: PyTorch's index of CPU-only wheels: the coordinator calibrates on CPU and needs no CUDA libraries.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
#: The constraints file of a release (``uv export`` of its lock), attached to the GitHub release of its tag.
CONSTRAINTS_URL = "https://github.com/cohere-ai/rcp-ndcg/releases/download/v{version}/requirements-constraints.txt"
#: The extras a coordinator installs: the calibration fit, Hub datasets, and the S3 and Azure mirrors (gs:// is core).
COORDINATOR_EXTRAS = ("calibrate", "hf", "s3", "azure")


def install_argv(argv: Sequence[str], version: str | None = None) -> tuple[str, ...]:
    """``argv`` as a stock image runs it: an ``rcp-ndcg`` command through ``uvx`` at this package's version.

    Args:
        argv: The command; one that is not ``rcp-ndcg ...`` is returned unchanged.
        version: The release to install; default the installed package's, so a job runs what submitted it.

    Returns:
        ``uvx --from rcp-ndcg[<extras>]==<version> --constraints <release constraints> --index <CPU torch>
        --index-strategy unsafe-best-match rcp-ndcg ...``. The CPU index is searched with PyPI for every
        package (``unsafe-best-match``), so torch resolves to its ``+cpu`` build; the constraints file pins
        every version to the release's lock.
    """
    if not argv or argv[0] != "rcp-ndcg":
        return tuple(argv)
    if version is None:
        from rcp_ndcg import __version__ as version
    return (
        "uvx",
        "--from",
        f"rcp-ndcg[{','.join(COORDINATOR_EXTRAS)}]=={version}",
        "--constraints",
        CONSTRAINTS_URL.format(version=version),
        "--index",
        TORCH_CPU_INDEX,
        "--index-strategy",
        "unsafe-best-match",
        *argv,
    )


def quote_argv(argv: Sequence[str]) -> str:
    """Shell-quote ``argv`` into one POSIX-safe string (``["a b", "c"]`` -> ``"'a b' c"``)."""
    return shlex.join(argv)


def export_lines(env: Mapping[str, str]) -> list[str]:
    """``export KEY=value`` lines with quoted values."""
    return [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]


def worker_script(
    spec: JobSpec,
    *,
    install: bool,
    workdir: str | None,
    env: Mapping[str, str] | None = None,
    prologue: Sequence[str] = (),
) -> str:
    """The bash script one task of ``spec`` runs.

    Args:
        spec: The job.
        install: Run an ``rcp-ndcg`` command through ``uvx`` (:func:`install_argv`), for a stock image.
        workdir: Directory to ``cd`` into first; ``None`` keeps the start directory.
        env: Environment the runner adds before the job's own (e.g. the replica URLs).
        prologue: Shell lines run before the command (e.g. :func:`wait_for_replicas`).
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if workdir:
        lines.append(f"cd {shlex.quote(workdir)}")
    lines += export_lines({**(env or {}), **spec.env})
    lines += prologue
    argv = install_argv(spec.argv) if install else spec.argv
    if argv[0] == "uvx" and argv != spec.argv:
        lines += bootstrap_uv()
    lines.append(f"exec {quote_argv(argv)}")
    return "\n".join(lines) + "\n"


#: Where an image without uv gets it (``pip install --target``, which PEP 668's externally-managed marker allows).
UV_BOOTSTRAP_DIR = "${TMPDIR:-/tmp}/rcp-ndcg-uv"


def bootstrap_uv() -> list[str]:
    """Bash lines that install uv with the image's ``python3 -m pip`` when ``uvx`` is not on ``PATH``."""
    return [
        "if ! command -v uvx >/dev/null; then",
        f'  python3 -m pip install --quiet --target "{UV_BOOTSTRAP_DIR}" uv',
        f'  export PATH="{UV_BOOTSTRAP_DIR}/bin:$PATH"',
        "fi",
    ]


def heredoc(var: str, script: str) -> list[str]:
    """Bash lines that read ``script`` into the variable ``var``, verbatim (a quoted heredoc expands nothing)."""
    tag = f"RCP_NDCG_{var}"
    # `read -d ''` returns 1 at the end of its input.
    return [f"read -r -d '' {var} <<'{tag}' || true", script.rstrip("\n"), tag]


def engine_script(serve: ServeConfig) -> str:
    """The script one engine replica runs: its environment, then its command, exec'd so a stop signal reaches it."""
    return "\n".join([*export_lines(serve.env), f"exec {quote_argv(serve.command)}"]) + "\n"


#: Seconds between two readiness probes of the engine replicas.
PROBE_INTERVAL_S: float = 10
#: Seconds a process that is stopped gets between ``SIGTERM`` and ``SIGKILL``.
STOP_GRACE_S = 20
#: The exit status of a job whose engine exited or never answered.
ENGINE_FAILED = 1

#: One GET of a URL that fails unless it answers 2xx within 5 seconds (Python's standard library only).
_GET = "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)"


def wait_for_replicas(serve: ServeConfig, hosts: str) -> list[str]:
    """Bash lines that wait until one engine replica answers ``GET <readiness_path>``, or exit non-zero.

    The judge client sets aside replicas that are not up yet, so one is enough to start; waiting for it keeps the
    first requests, and the record of what the endpoint serves, from meeting an engine that is still loading. The
    wait ends with exit status :data:`ENGINE_FAILED` after ``serve.startup_timeout_s`` seconds, or at once when the
    engine this script started (``$RCP_NDCG_ENGINE_PID``, if set) has exited.

    Args:
        serve: The replicas.
        hosts: Shell words that expand to the replicas' hosts (e.g. ``"${HOSTS[@]}"``).
    """
    url = f"http://$host:{serve.port}{serve.readiness_path}"
    path, timeout = serve.readiness_path, serve.startup_timeout_s
    return [
        "rcp_ndcg_any_ready() {",
        '  for host in "$@"; do',
        f'    python3 -c {shlex.quote(_GET)} "{url}" >/dev/null 2>&1 && return 0',
        "  done",
        "  return 1",
        "}",
        "rcp_ndcg_wait_ready() {",
        f"  local deadline=$((SECONDS + {timeout})) status",
        f'  echo "rcp-ndcg: waiting up to {timeout} s for an engine replica to answer {path}" >&2',
        '  until rcp_ndcg_any_ready "$@"; do',
        '    if [[ -n "${RCP_NDCG_ENGINE_PID:-}" ]] && ! kill -0 "$RCP_NDCG_ENGINE_PID" 2>/dev/null; then',
        "      status=0",
        '      wait "$RCP_NDCG_ENGINE_PID" || status=$?',
        "      RCP_NDCG_ENGINE_PID=",
        f'      echo "rcp-ndcg: the engine exited with status $status before it answered {path}; its output is '
        'above" >&2',
        f"      exit {ENGINE_FAILED}",
        "    fi",
        "    if ((SECONDS >= deadline)); then",
        f'      echo "rcp-ndcg: no engine replica answered {path} within {timeout} s (serve.startup_timeout_s); '
        'stopping the job" >&2',
        f"      exit {ENGINE_FAILED}",
        "    fi",
        f"    sleep {PROBE_INTERVAL_S:g}",
        "  done",
        "}",
        f"rcp_ndcg_wait_ready {hosts}",
    ]


#: The exit status of a job whose shell or image lacks a tool the script needs (:func:`require_tools`).
TOOL_MISSING = 1


def require_tools(*, uv: bool) -> list[str]:
    """Bash lines that stop the job at once, naming what is missing, unless the tools the script needs are there.

    The supervision script needs bash 4.3 or later (``wait -n``) and ``python3`` on ``PATH`` (the readiness probe
    uses its standard library). With ``uv``, the coordinator runs through ``uvx`` in the same image, so ``uvx``
    or, to install it (:func:`bootstrap_uv`), ``python3 -m pip``. Checked before the engine starts, so a job on an
    image that lacks one fails in seconds instead of after the engine has loaded, or after
    ``serve.startup_timeout_s`` of probes that could never answer.
    """
    lines = [
        "if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then",
        '  echo "rcp-ndcg: this job needs bash 4.3 or later (for wait -n), and its bash is $BASH_VERSION; use an '
        'image or node with a newer bash" >&2',
        f"  exit {TOOL_MISSING}",
        "fi",
        "if ! command -v python3 >/dev/null; then",
        '  echo "rcp-ndcg: this job needs python3 on PATH (the engine readiness probe uses its standard library), '
        'and there is none; use an image or node that has python3" >&2',
        f"  exit {TOOL_MISSING}",
        "fi",
    ]
    if uv:
        lines += [
            "if ! command -v uvx >/dev/null && ! python3 -m pip --version >/dev/null 2>&1; then",
            '  echo "rcp-ndcg: this job runs the coordinator with uvx, which is not on PATH, and python3 has no pip to '
            'install uv with; use an engine image that has uv, or python3 with pip" >&2',
            f"  exit {TOOL_MISSING}",
            "fi",
        ]
    return lines


def supervise(serve: ServeConfig, *, engine: str, coordinator: str, hosts: str, uv: bool = False) -> list[str]:
    """Bash lines that run the engine and the coordinator together, and end the job when either ends.

    First the tools the script needs are checked (:func:`require_tools`). The engine starts once, in the background
    (no restart); the coordinator starts in the background once a replica answers (:func:`wait_for_replicas`);
    then the script waits for the first of the two to end:

    * the engine: the coordinator is stopped, and the job exits with :data:`ENGINE_FAILED` and a message naming
      the engine;
    * the coordinator: the engine is stopped, and the job exits with the coordinator's status.

    On ``SIGTERM``, ``SIGINT`` and at any exit both are stopped: ``SIGTERM``, then ``SIGKILL`` after
    :data:`STOP_GRACE_S` seconds. Needs bash 4.3 or later (``wait -n``).

    Args:
        serve: The replicas.
        engine: The command line that starts the engine (e.g. ``srun ... bash -c "$ENGINE"``), without ``&``.
        coordinator: The command line that runs the coordinator (e.g. ``bash -c "$WORKER"``), without ``&``.
        hosts: Shell words that expand to the replicas' hosts.
        uv: The coordinator runs through ``uvx`` beside the engine, in its image (a Kubernetes pod of one replica).
    """
    return [
        *require_tools(uv=uv),
        "rcp_ndcg_stop() {  # SIGTERM the processes, SIGKILL what is left after the grace period",
        "  local pid waited=0",
        "  local -a left",
        '  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done',
        "  while true; do",
        "    left=()",
        '    for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then left+=("$pid"); fi; done',
        "    if ((${#left[@]} == 0)); then return 0; fi",
        f"    if ((waited >= {STOP_GRACE_S * 10})); then  # tenths of a second",
        '      kill -KILL "${left[@]}" 2>/dev/null || true',
        "      return 0",
        "    fi",
        "    sleep 0.1",
        "    waited=$((waited + 1))",
        "  done",
        "}",
        "rcp_ndcg_cleanup() {",
        "  trap '' TERM INT",
        "  local -a pids=()",
        '  if [[ -n "${RCP_NDCG_COORDINATOR_PID:-}" ]]; then pids+=("$RCP_NDCG_COORDINATOR_PID"); fi',
        '  if [[ -n "${RCP_NDCG_ENGINE_PID:-}" ]]; then pids+=("$RCP_NDCG_ENGINE_PID"); fi',
        '  if ((${#pids[@]})); then rcp_ndcg_stop "${pids[@]}"; fi',
        "}",
        "trap rcp_ndcg_cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        f"{engine} &",
        "RCP_NDCG_ENGINE_PID=$!",
        *wait_for_replicas(serve, hosts),
        f"{coordinator} &",
        "RCP_NDCG_COORDINATOR_PID=$!",
        "status=0",
        "wait -n || status=$?",
        'if kill -0 "$RCP_NDCG_COORDINATOR_PID" 2>/dev/null; then',
        '  echo "rcp-ndcg: the engine exited with status $status while the run was going; stopping the run (resume '
        'it with rcp-ndcg run resume, or let the scheduler retry the job)" >&2',
        f"  exit {ENGINE_FAILED}",
        "fi",
        'if ! kill -0 "$RCP_NDCG_ENGINE_PID" 2>/dev/null; then  # both have ended: the status is the coordinator\'s',
        "  status=0",
        '  wait "$RCP_NDCG_COORDINATOR_PID" || status=$?',
        "fi",
        'exit "$status"',
    ]


__all__ = [
    "CONSTRAINTS_URL",
    "COORDINATOR_EXTRAS",
    "COORDINATOR_IMAGE",
    "ENGINE_FAILED",
    "PROBE_INTERVAL_S",
    "STOP_GRACE_S",
    "TOOL_MISSING",
    "TORCH_CPU_INDEX",
    "UV_BOOTSTRAP_DIR",
    "bootstrap_uv",
    "engine_script",
    "export_lines",
    "heredoc",
    "install_argv",
    "quote_argv",
    "require_tools",
    "supervise",
    "wait_for_replicas",
    "worker_script",
]
