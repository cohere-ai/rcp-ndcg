# Extension points

The package's extension seams are entry-point groups. A downstream product installs a distribution that
publishes one, and the package loads it only when a config names it. The job-runner seam's contract is below.

## Job runners (`rcp.runners`)

A runner decides *where* a job runs. The `local`, `slurm` and `kubernetes` runners are entries of this group; a
third-party runner is a distribution that publishes one:

```toml
[project.entry-points."rcp.runners"]
my-runner = "my_package.runner:MyRunner"
```

`rcp.runners.get_runner("my-runner", **options)` loads the entry point only when the name is selected and calls
`MyRunner(**options)` with the run config's `runner.options`. A name provided by more than one installed
distribution is refused, so a plugin cannot shadow a built-in. An unknown name lists the installed runners. The
runner's `name` attribute is the entry point's name.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The entry point's name: a non-empty string. |
| `synchronous: bool = False` | Whether `submit` runs the job in this process: then the handle is the job's name, known before it runs, and the job's failure is raised from `submit`. A scheduler runner leaves it false. |
| `__init__(**options)` | Receive `runner.options` as keyword arguments. An invalid option raises `TypeError` or `ValueError`; the registry reports it as `ConfigError` naming the runner. |
| `submit(jobs) -> list[str]` | Submit the `JobSpec`s in order and return one non-empty handle per job. A runner that runs jobs in this process (see `synchronous`) returns each job's name and raises the job's failure here. |
| `status(handle) -> JobStatus` | The job's current state, non-blocking. An unknown handle answers `JobStatus.UNKNOWN`, not an exception. |
| `logs(handle, *, tail=None) -> str` | The job's output so far; `tail` keeps the last lines. |
| `cancel(handle) -> None` | Stop a pending or running job; a finished job is left as it is. |

`JobSpec` is one command line (`argv`) with its image, resources and environment. It can be the phases to run
in order (`phases`) instead. Exactly one of the two is set. The run layer reads four optional members when a runner declares them:

- `render(jobs) -> dict[str, str]` returns the script or manifest that each job would submit, without
  submitting anything. `run start --dry-run` prints it. It is called before the run directory is written, so a
  runner refuses a job it cannot run there.
- `renders_phases = True` declares that `render` and `submit` take a phased job, whose phases' engines are the
  runner's to start. A runner without it is handed the whole-run command as the job's `argv`. A job whose
  phases start engines is refused.
- `run_root` is the directory that a run is restored into inside the job, from the run's mirror. A runner whose
  jobs do not see the submitting host's files sets it. Runners without it share the files.
- `note(handle)` explains why a pending job is still pending (a scheduler's unschedulable reason).

`local`, `slurm` and `kubernetes` are the reference implementations; `LocalRunner` is the smallest and the one
to read first. A runner that adds no scheduling can delegate to it.

### Errors a runner may raise

- `ConfigError` (exit 3) for an unknown or ambiguous runner name, or options that are not the runner's options.
  The registry raises it around the constructor; a plugin does not need to import it for its own options.
- `RunnerError` (exit 6) for a scheduler command that failed, or a job the runner cannot submit as specified.
- A job that exits non-zero is the job's failure, not the runner's. A `synchronous` runner raises the error
  class of the job's exit code from `submit`. That class is `rcp.errors.error_class`, for example
  `MissingInputError` for 4. A code that no error class uses is a non-retryable `RunnerError`. A scheduler
  runner returns a handle and reports `JobStatus.FAILED`, and its output is in `logs`.

A caller branches on the typed error. `MissingInputError` names an input to produce, and `ProviderError` names a
failure to retry. Flattening a job's failure into a generic `RunnerError` loses that decision.

### A minimal runner

This runner adds one option and delegates the four methods to the local supervisor. A scheduler runner replaces
the bodies of the four methods; the signatures and the answers are the contract.

```python
from rcp.runners import JobSpec, JobStatus, LocalRunner


class MyRunner:
    name = "my-runner"
    synchronous = True

    def __init__(self, *, log_dir: str) -> None:
        self._local = LocalRunner(log_dir=log_dir)

    def submit(self, jobs: list[JobSpec]) -> list[str]:
        return self._local.submit(jobs)

    def status(self, handle: str) -> JobStatus:
        return self._local.status(handle)

    def logs(self, handle: str, *, tail: int | None = None) -> str:
        return self._local.logs(handle, tail=tail)

    def cancel(self, handle: str) -> None:
        self._local.cancel(handle)
```

A plugin's own tests call `rcp.testing.runner_conformance` with a constructed runner and a job that it can
actually run. The helper checks the four methods and the answers' shapes. It also checks `render`,
`renders_phases` and `run_root` when the runner declares them. A failure is one assertion, and it lists every
broken check.

<!-- snippet: skip (the plugin is installed by the package under test) -->
```python
from rcp.runners import JobSpec, get_runner
from rcp.testing import runner_conformance

runner = get_runner("my-runner", log_dir="logs")
runner_conformance(runner, job=JobSpec(name="smoke", argv=("true",)))
```

The plugin's test should also run one real step and one job that exits 4. The step proves the handle, and it
proves that `status` and `logs` describe the same job. The failing job proves that the failure is a typed error,
not a generic runner error. The built-in `LocalRunner` raises that same typed error. The repository's `tests/contract/test_runner_plugin.py`
installs a minimal runner as a test distribution, resolves it through this group, and makes both checks.
