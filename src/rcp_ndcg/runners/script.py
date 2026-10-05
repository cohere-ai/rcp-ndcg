"""Render the scripts a task runs, from a :class:`JobSpec`.

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

A job with phases (``JobSpec.phases``) runs them in order in one allocation: each phase that starts engines runs one
supervision script (:func:`supervise`) — the same on SLURM and in a Kubernetes container — which starts the phase's
engines once in the background (no restart), waits until every role has a replica answering its readiness path (at
most ``startup_timeout_s``, and never after an engine has exited), exports the engines' URLs to the coordinator in
``RCP_NDCG_ENGINES`` (:data:`~rcp_ndcg.support.serve.ENGINES_ENV`), runs the phase's coordinator in the background,
and ends the phase with whichever ends first. An engine that exits ends the job non-zero; a coordinator that exits
stops the engines, and its status is the phase's — the next phase starts only after the previous coordinator exited
0 and its engines were stopped. A phase without engines runs its command directly. Resuming is cheap, so the job
fails rather than hold its allocation; ``rcp-ndcg run resume --run <dir> --runner <name>`` submits the run again.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rcp_ndcg.runners.base import JobSpec
from rcp_ndcg.support.serve import ENGINES_ENV, ServeConfig

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
        spec: The job, or one of its phases (the same job with the phase's ``argv``).
        install: Run an ``rcp-ndcg`` command through ``uvx`` (:func:`install_argv`), for a stock image.
        workdir: Directory to ``cd`` into first; ``None`` keeps the start directory.
        env: Environment the runner adds before the job's own (e.g. an empty ``RCP_NDCG_ENGINES`` for a phase
            without engines).
        prologue: Shell lines run before the command.
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


def engines_env_value[R: str](engines: Mapping[R, ServeConfig], urls: Mapping[R, Sequence[str]]) -> str:
    """The ``RCP_NDCG_ENGINES`` value of one phase: JSON of ``role -> {urls, wait_on_outage_s}``, roles sorted.

    Args:
        engines: The phase's engines by the role whose config each serves.
        urls: Per role, the replica base URLs, one per replica (known at render time).

    Returns:
        JSON such as ``{"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}``. The
        coordinator applies it as a runtime overlay, so it is never written into a config and never reaches an
        identity. Shell-quote the return value for the script; when the engine hosts are the nodes of a scheduler
        allocation, known only when the job starts, build the JSON at run time instead with
        :func:`engines_env_spec` and :func:`engines_env_command`.
    """
    return json.dumps(
        {
            role: {"urls": list(urls[role]), "wait_on_outage_s": engines[role].outage_timeout_s}
            for role in sorted(engines)
        }
    )


#: The python program (for ``python3 -c``) that builds ``RCP_NDCG_ENGINES`` at run time, from one argument per
#: role, ``role:port:wait_on_outage_s:<comma-joined hosts>`` (no field holds a colon). Used when the engine hosts
#: are the nodes of a scheduler allocation, known only when the job starts.
_ENGINES_SPEC = """import json, sys
out = {}
for spec in sys.argv[1:]:
    role, port, wait, hosts = spec.split(":", 3)
    out[role] = {"urls": [f"http://{h}:{port}/v1" for h in hosts.split(",")], "wait_on_outage_s": int(wait)}
print(json.dumps(out))
"""


def engines_env_spec() -> list[str]:
    """Bash lines that read the run-time ``RCP_NDCG_ENGINES`` builder (:data:`_ENGINES_SPEC`) into its variable,
    for phases whose engine hosts the script learns only when the job starts (see :func:`engines_env_command`).
    """
    tag = "RCP_NDCG_ENGINES_SPEC"
    return [f"read -r -d '' {tag} <<'{tag}' || true", *_ENGINES_SPEC.rstrip("\n").split("\n"), tag]


def engines_env_command[R: str](engines: Mapping[R, ServeConfig], host_vars: Mapping[R, str]) -> str:
    """Shell words that compute a phase's ``RCP_NDCG_ENGINES`` when the job runs, for hosts known only then.

    Args:
        engines: The phase's engines by role (roles sorted, per :func:`engines_env_value`).
        host_vars: Per role, shell words that expand to the role's replica hosts, comma-joined (e.g.
            ``IFS=,; echo "${HOSTS_JUDGE[*]}"``), which the script has read from the scheduler's node list.

    Returns:
        Shell words for :func:`supervise`'s ``engines_env``: a command substitution that runs the stdlib-only
        program :data:`_ENGINES_SPEC` (read by :func:`engines_env_spec` into ``RCP_NDCG_ENGINES_SPEC``) with one
        argument per role, ``role:port:wait_on_outage_s:<comma-joined hosts>``.
    """
    specs = " ".join(
        f'"{role}:{engines[role].port}:{engines[role].outage_timeout_s}:$( {host_vars[role]} )"'
        for role in sorted(engines)
    )
    return '"$(python3 -c "$RCP_NDCG_ENGINES_SPEC" ' + specs + ')"'


#: Seconds between two readiness probes of the engine replicas.
PROBE_INTERVAL_S: float = 10
#: Seconds a process that is stopped gets between ``SIGTERM`` and ``SIGKILL``.
STOP_GRACE_S = 20
#: The exit status of a job whose engine exited or never answered.
ENGINE_FAILED = 1

#: One GET of a URL that fails unless it answers 2xx within 5 seconds (Python's standard library only).
_GET = "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)"

#: The bash variable that holds a readiness wait's engine pid when the script does not start the engine itself
#: (its replicas are already running, e.g. a Kubernetes StatefulSet's): never assigned, so the wait skips the
#: engine-exited check.
REMOTE_ENGINE_PID = "RCP_NDCG_ENGINE_PID_REMOTE"


def readiness_functions() -> list[str]:
    """Bash definitions of the readiness probe: ``rcp_ndcg_any_ready PORT PATH HOST...`` answers once one host
    serves, and ``rcp_ndcg_wait_ready PID_VAR TIMEOUT PORT PATH HOST...`` loops until then, exits
    :data:`ENGINE_FAILED` after ``TIMEOUT`` seconds, and at once when the engine whose pid is in the variable named
    ``PID_VAR`` has exited (a variable that is never set, e.g. :data:`REMOTE_ENGINE_PID`, skips that check).
    """
    return [
        "rcp_ndcg_any_ready() {  # PORT PATH HOST...",
        '  local port="$1" path="$2"; shift 2',
        '  for host in "$@"; do',
        f'    python3 -c {shlex.quote(_GET)} "http://$host:$port$path" >/dev/null 2>&1 && return 0',
        "  done",
        "  return 1",
        "}",
        "rcp_ndcg_wait_ready() {  # PID_VAR TIMEOUT PORT PATH HOST...",
        '  local pid_var="$1" timeout="$2" port="$3" path="$4"; shift 4',
        "  local deadline=$((SECONDS + timeout)) status",
        '  echo "rcp-ndcg: waiting up to $timeout s for an engine replica to answer $path" >&2',
        '  until rcp_ndcg_any_ready "$port" "$path" "$@"; do',
        '    if [[ -n "${!pid_var:-}" ]] && ! kill -0 "${!pid_var}" 2>/dev/null; then',
        "      status=0",
        '      wait "${!pid_var}" || status=$?',
        "      printf -v \"$pid_var\" ''",
        '      echo "rcp-ndcg: the engine exited with status $status before it answered $path; its output is '
        'above" >&2',
        f"      exit {ENGINE_FAILED}",
        "    fi",
        "    if ((SECONDS >= deadline)); then",
        '      echo "rcp-ndcg: no engine replica answered $path within $timeout s (serve.startup_timeout_s); '
        'stopping the job" >&2',
        f"      exit {ENGINE_FAILED}",
        "    fi",
        f"    sleep {PROBE_INTERVAL_S:g}",
        "  done",
        "}",
    ]


def wait_for_replicas(serve: ServeConfig, hosts: str, pid_var: str) -> list[str]:
    """Bash lines that wait until one of ``serve``'s replicas answers ``GET <readiness_path>``, or exit non-zero.

    The judge client sets aside replicas that are not up yet, so one is enough to start; waiting for it keeps the
    first requests, and the record of what the endpoint serves, from meeting an engine that is still loading. The
    wait ends with exit status :data:`ENGINE_FAILED` after ``serve.startup_timeout_s`` seconds, or at once when the
    engine the script started (the variable named ``pid_var``, if set) has exited.

    Args:
        serve: The replicas.
        hosts: Shell words that expand to the replicas' hosts (e.g. ``"${HOSTS[@]}"``).
        pid_var: Name of the bash variable that holds the engine's pid, watched while it loads; a name that is
            never set (e.g. :data:`REMOTE_ENGINE_PID`) skips the engine-exited check.
    """
    return [
        *readiness_functions(),
        f"rcp_ndcg_wait_ready {pid_var} {serve.startup_timeout_s} {serve.port} "
        f"{shlex.quote(serve.readiness_path)} {hosts}",
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


@dataclass(frozen=True)
class EngineStep:
    """One role's engine replicas within a phase, as the supervision script starts, waits for and stops them.

    Attributes:
        serve: The replicas' configuration (port, readiness path, startup timeout, outage timeout).
        role: The role whose config the engines serve (``judge``, ``encoder`` or ``reranker``); it names the
            failure message.
        start: The command line that starts the replicas, one per host, without ``&``; ``None`` when they are
            already running elsewhere (a Kubernetes StatefulSet, run-scoped), so the script only waits for them.
        hosts: Shell words that expand to the replicas' hosts (e.g. ``"${HOSTS_JUDGE[@]}"`` or ``127.0.0.1``).
    """

    serve: ServeConfig
    role: str
    start: str | None
    hosts: str


def supervise(engines: Sequence[EngineStep], *, coordinator: str, engines_env: str, uv: bool = False) -> list[str]:
    """Bash lines that run one phase's engines and its coordinator together, and end the phase when either ends.

    First the tools the script needs are checked (:func:`require_tools`). Every engine with a ``start`` line starts
    once, in the background (no restart), all of them in parallel; then the script waits until each engine has a
    replica answering (:func:`wait_for_replicas`, remote engines included), exports the engines' URLs to the
    coordinator in ``RCP_NDCG_ENGINES``, and starts the coordinator in the background; then it waits for the first
    of them to end:

    * an engine: the coordinator is stopped, and the phase exits with :data:`ENGINE_FAILED` and a message naming
      the engine (only engines the script started are watched; replicas that live elsewhere are restarted by
      whatever runs them, and the judge bounds the outage instead);
    * the coordinator: what the script started is stopped and reaped, and the phase exits with the coordinator's
      status -- unless a started engine ended non-zero before the coordinator's exit is observed (``wait -n`` does
      not report such a child when the script runs as ``bash -c``, as a Kubernetes container command does), which
      fails the phase with :data:`ENGINE_FAILED`.

    A phase whose coordinator exited 0 stops its engines and the script continues with the next phase (the
    renderers run one call per engine phase, in order); a non-zero exit of either ends the job. On ``SIGTERM``,
    ``SIGINT`` and at any exit the started engines and the coordinator are stopped, and reaped, so the next
    phase's ``wait -n`` cannot see a stale status. Needs bash 4.3 or later (``wait -n``).

    Args:
        engines: The phase's engines, in the order they start (roles sorted, per the renderers); at least one with
            a ``start`` line.
        coordinator: The command line that runs the phase's coordinator (e.g. ``bash -c "$WORKER_1"``), without
            ``&``.
        engines_env: Shell words to assign to ``RCP_NDCG_ENGINES`` once every role answers: the quoted JSON of
            :func:`engines_env_value`, or a command substitution that builds it from the hosts
            (:func:`engines_env_command`).
        uv: The coordinator runs through ``uvx`` beside the engines, in their image (a Kubernetes container).
    """
    pid_vars = [f"RCP_NDCG_ENGINE_PID{'_' + str(i) if i else ''}" for i in range(sum(1 for e in engines if e.start))]
    pids = iter(pid_vars)
    steps = [(step, next(pids) if step.start else REMOTE_ENGINE_PID) for step in engines]
    started = [line for step, pid in steps if step.start for line in (f"{step.start} &", f"{pid}=$!")]
    waits = [
        f"rcp_ndcg_wait_ready {pid} {step.serve.startup_timeout_s} {step.serve.port} "
        f"{shlex.quote(step.serve.readiness_path)} {step.hosts}"
        for step, pid in steps
    ]
    return [
        *require_tools(uv=uv),
        "rcp_ndcg_stop() {  # SIGTERM the processes, SIGKILL what is left after the grace period, and reap them",
        "  local pid waited=0",
        "  local -a left",
        '  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done',
        "  while true; do",
        "    left=()",
        '    for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then left+=("$pid"); fi; done',
        "    if ((${#left[@]} == 0)); then break; fi",
        f"    if ((waited >= {STOP_GRACE_S * 10})); then  # tenths of a second",
        '      kill -KILL "${left[@]}" 2>/dev/null || true',
        "      break",
        "    fi",
        "    sleep 0.1",
        "    waited=$((waited + 1))",
        "  done",
        '  for pid in "$@"; do  # reap what this script started, so the next phase\'s wait -n cannot see it',
        '    wait "$pid" 2>/dev/null || true',
        "  done",
        "}",
        "rcp_ndcg_cleanup() {",
        "  trap '' TERM INT",
        "  local -a pids=()",
        '  if [[ -n "${RCP_NDCG_COORDINATOR_PID:-}" ]]; then pids+=("$RCP_NDCG_COORDINATOR_PID"); fi',
        *(f'  if [[ -n "${{{pid}:-}}" ]]; then pids+=("${{{pid}}}"); fi' for pid in pid_vars),
        '  if ((${#pids[@]})); then rcp_ndcg_stop "${pids[@]}"; fi',
        "}",
        "trap rcp_ndcg_cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        *started,
        *readiness_functions(),
        *waits,
        # A command substitution is declared and assigned separately (shellcheck SC2155); a quoted literal is
        # exported in one line (assigning it separately trips SC2089/SC2090).
        *(
            [f"export {ENGINES_ENV}", f"{ENGINES_ENV}={engines_env}"]
            if engines_env.startswith('"$(')
            else [f"export {ENGINES_ENV}={engines_env}"]
        ),
        f"{coordinator} &",
        "RCP_NDCG_COORDINATOR_PID=$!",
        "status=0",
        "wait -n || status=$?",
        'if kill -0 "$RCP_NDCG_COORDINATOR_PID" 2>/dev/null; then',
        '  echo "rcp-ndcg: '
        + (
            f"the {next(step.role for step, pid in steps if step.start)} engine exited"
            if len(pid_vars) == 1
            else "an engine exited"
        )
        + " with status $status while the run was going; stopping the run (submit it again with rcp-ndcg run resume "
        '--run <run dir> --runner <this runner>)" >&2',
        f"  exit {ENGINE_FAILED}",
        "fi",
        # The coordinator has ended. A started engine that also ended non-zero failed while the run was going
        # (invisible to wait -n when the script runs as bash -c, as a Kubernetes container command does): reap
        # what this script started and fail fast. 127 is a status wait -n already consumed.
        *(
            [
                "failed=0",
                "for pid in " + " ".join(f"${{{pid}}}" for pid in pid_vars) + "; do",
                '  if ! kill -0 "$pid" 2>/dev/null; then',
                "    es=0",
                '    wait "$pid" 2>/dev/null || es=$?',
                '    if [ "$es" -ne 0 ] && [ "$es" -ne 127 ]; then failed=$es; fi',
                "  fi",
                "done",
                'if [ "$failed" -ne 0 ]; then',
                '  echo "rcp-ndcg: '
                + (
                    f"the {next(step.role for step, pid in steps if step.start)} engine exited"
                    if len(pid_vars) == 1
                    else "an engine exited"
                )
                + " with status $failed while the run was going; stopping the run (submit it again with "
                'rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2',
                f"  exit {ENGINE_FAILED}",
                "fi",
            ]
            if pid_vars
            else []
        ),
        'if [ "$status" -ne 0 ]; then',
        '  exit "$status"',
        "fi",
        # The coordinator exited 0: stop the engines that outlived it before the next phase starts.
        *([f"rcp_ndcg_stop {' '.join(f'"${pid}"' for pid in pid_vars)}"] if pid_vars else []),
    ]


__all__ = [
    "CONSTRAINTS_URL",
    "COORDINATOR_EXTRAS",
    "COORDINATOR_IMAGE",
    "ENGINE_FAILED",
    "EngineStep",
    "PROBE_INTERVAL_S",
    "REMOTE_ENGINE_PID",
    "STOP_GRACE_S",
    "TOOL_MISSING",
    "TORCH_CPU_INDEX",
    "UV_BOOTSTRAP_DIR",
    "bootstrap_uv",
    "engine_script",
    "engines_env_command",
    "engines_env_spec",
    "engines_env_value",
    "export_lines",
    "heredoc",
    "install_argv",
    "quote_argv",
    "readiness_functions",
    "require_tools",
    "supervise",
    "wait_for_replicas",
    "worker_script",
]
