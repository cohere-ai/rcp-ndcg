"""The worker script: one command, quoted once, exec'd so the scheduler's signals reach it; in a stock image, the
package installs itself through uvx."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from rcp_ndcg import __version__
from rcp_ndcg.runners import JobPhase, JobSpec
from rcp_ndcg.runners.kubernetes import KubernetesRunner
from rcp_ndcg.runners.local import LocalRunner
from rcp_ndcg.runners.script import CONSTRAINTS_URL, TORCH_CPU_INDEX, install_argv, merge_phase_env, worker_script
from rcp_ndcg.runners.slurm import SlurmOptions, SlurmRunner


def test_host_script_golden() -> None:
    spec = JobSpec(name="j", argv=("rcp-ndcg", "run", "start", "a dir"), env={"HF_HOME": "/cache"})
    assert worker_script(spec, install=False, workdir="/work") == (
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /work\nexport HF_HOME=/cache\nexec rcp-ndcg run start 'a dir'\n"
    )


def test_in_a_stock_image_the_release_installs_itself_with_uvx() -> None:
    spec = JobSpec(name="j", argv=("rcp-ndcg", "--help"))
    last = worker_script(spec, install=True, workdir=None).splitlines()[-1]
    assert last == (
        f"exec uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]=={__version__}' --constraints "
        f"{CONSTRAINTS_URL.format(version=__version__)} --index {TORCH_CPU_INDEX} --index-strategy unsafe-best-match "
        "rcp-ndcg --help"
    )
    assert CONSTRAINTS_URL.startswith("https://github.com/cohere-ai/rcp-ndcg/releases/download/v")
    assert install_argv(("python3", "-V")) == ("python3", "-V")  # only the package's own command is installed


def test_hostile_argv_survives_bash_unchanged(tmp_path) -> None:
    """Quotes, pipes, braces, dollars and JSON reach the program byte for byte."""
    hostile = ["a'b", 'c"d', "$(touch pwned)", "x|y;z", '{"k": [1, 2]}', "*", ""]
    argv = ("python3", "-c", "import json, sys; print(json.dumps(sys.argv[1:]))", *hostile)
    script = worker_script(JobSpec(name="j", argv=argv), install=False, workdir=str(tmp_path))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    import json

    assert json.loads(out.stdout) == hostile
    assert not (tmp_path / "pwned").exists()


def test_the_phase_overlay_owns_the_engines_variable() -> None:
    """A job env entry named RCP_NDCG_ENGINES once defeated every phase's own value: the worker re-exported the
    job's env inside the child, after `supervise` had exported the phase's JSON in the parent."""
    from rcp_ndcg.support.serve import ENGINES_ENV

    job = '{"judge": {"urls": ["http://job:1/v1"]}}'
    phase = '{"judge": {"urls": ["http://phase:2/v1"]}}'
    assert merge_phase_env({ENGINES_ENV: job, "HF_HOME": "/job"}, {ENGINES_ENV: phase}) == {
        ENGINES_ENV: phase,
        "HF_HOME": "/job",
    }
    # a job value the runner does not own still wins, and the runner's other additions stay under it
    assert merge_phase_env({"HF_HOME": "/job"}, {"UV_CACHE_DIR": "/scratch", "HF_HOME": "/runner"}) == {
        "HF_HOME": "/job",
        "UV_CACHE_DIR": "/scratch",
    }
    assert merge_phase_env({ENGINES_ENV: job}, None) == {ENGINES_ENV: job}


def test_a_job_needs_a_command() -> None:
    with pytest.raises(ValueError, match="argv must not be empty"):
        JobSpec(name="j", argv=())
    with pytest.raises(ValueError):
        JobSpec(name="Not_A_Name", argv=("true",))


def test_an_image_without_uv_installs_it_with_pip_before_the_coordinator_starts(tmp_path) -> None:
    """An engine's image (the pod of one replica on Kubernetes) has python3 and pip, but maybe no uv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # pip --target puts uv's executables in <target>/bin (PEP 668 does not apply to --target).
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" > {tmp_path}/pip-args\n'
        'target="${@: -2:1}"\n'
        'mkdir -p "$target/bin"\n'
        'printf \'#!/usr/bin/env bash\\necho uvx "$*"\\n\' > "$target/bin/uvx"\n'
        'chmod +x "$target/bin/uvx"\n'
    )
    (bin_dir / "python3").chmod(0o755)
    for tool in ("bash", "mkdir", "chmod"):  # and nothing else: this host may have uv
        (bin_dir / tool).symlink_to(shutil.which(tool))
    script = worker_script(JobSpec(name="j", argv=("rcp-ndcg", "--help")), install=True, workdir=None)
    env = {"PATH": str(bin_dir), "TMPDIR": str(tmp_path)}
    done = subprocess.run([str(bin_dir / "bash"), "-c", script], env=env, capture_output=True, text=True, check=True)
    assert (tmp_path / "pip-args").read_text() == f"-m pip install --quiet --target {tmp_path}/rcp-ndcg-uv uv\n"
    assert done.stdout.startswith("uvx --from") and done.stdout.rstrip().endswith("rcp-ndcg --help")
    # An image with uv, and a host job, install nothing.
    assert "pip install" not in worker_script(
        JobSpec(name="j", argv=("rcp-ndcg", "--help")), install=False, workdir=None
    )
    assert "pip install" not in worker_script(JobSpec(name="j", argv=("python3", "-V")), install=True, workdir=None)


def test_an_install_source_replaces_the_release_urls_in_the_uvx_command() -> None:
    """A wheelhouse (staged wheels: a pre-release or an air-gapped node) is passed as ``--find-links`` with
    ``--no-index`` -- nothing comes from PyPI or the torch index -- and a constraints file replaces the
    release URL."""
    argv = install_argv(("rcp-ndcg", "--help"), wheelhouse="/shared/wheelhouse", constraints="/shared/c.txt")
    assert argv == (
        "uvx",
        "--from",
        f"rcp-ndcg[calibrate,hf,s3,azure]=={__version__}",
        "--constraints",
        "/shared/c.txt",
        "--find-links",
        "/shared/wheelhouse",
        "--no-index",
        "rcp-ndcg",
        "--help",
    )
    assert TORCH_CPU_INDEX not in argv  # the wheelhouse stages the CPU torch wheels too
    # a constraints file alone replaces only the release URL; a URL wheelhouse is rendered verbatim
    assert "--constraints https://storage.example/c.txt" in " ".join(
        install_argv(("rcp-ndcg", "--help"), constraints="https://storage.example/c.txt")
    )
    https = " ".join(install_argv(("rcp-ndcg", "--help"), wheelhouse="https://storage.example/wheels"))
    assert "--find-links https://storage.example/wheels --no-index" in https
    assert "--constraints https://github.com/cohere-ai/rcp-ndcg/releases/download/v" in https  # the release URL stays


def test_a_wheelhouse_reaches_the_rendered_scripts_of_the_runners_that_install(tmp_path) -> None:
    """The coordinator's install source renders wherever an install happens: a container on SLURM and the
    Kubernetes pod; shellcheck-clean, like the rest of the script."""
    from tests.runners.shell import assert_shellcheck_clean

    spec = JobSpec(name="j", argv=("rcp-ndcg", "--help"))
    slurm = SlurmRunner(
        container_runtime="pyxis", wheelhouse="/shared/wheels", constraints="/shared/wheels/c.txt"
    ).render([spec])["j"]
    assert "--find-links /shared/wheels --no-index" in slurm and "--constraints /shared/wheels/c.txt" in slurm
    assert_shellcheck_clean(slurm)
    pod = KubernetesRunner(
        wheelhouse="https://storage.example/wheels", constraints="https://storage.example/wheels/c.txt"
    ).render([spec])["j"]
    assert "--find-links https://storage.example/wheels --no-index" in pod


def test_an_install_source_is_refused_where_nothing_installs() -> None:
    """The local runner runs the coordinator in this host's environment, and SLURM without a container runtime
    runs it on the node: neither installs the release, so a wheelhouse or constraints file there is refused."""
    from rcp_ndcg.errors import ConfigError

    with pytest.raises(ConfigError, match="installs nothing"):
        LocalRunner(wheelhouse="/shared/wheels")
    with pytest.raises(ConfigError, match="container_runtime"):
        SlurmRunner(wheelhouse="/shared/wheels")
    with pytest.raises(ConfigError, match="container_runtime"):
        SlurmRunner(constraints="/shared/c.txt")
    assert SlurmRunner(container_runtime="apptainer", constraints="/shared/c.txt").options.constraints


def test_with_argv_refuses_an_empty_command() -> None:
    """The per-phase builder validates what model_copy would skip: an empty phase command is refused, not
    rendered into a script that crashes, and so is a string (which would be char-split into words)."""
    job = JobSpec(name="j", phases=(JobPhase(argv=("echo", "hi")),))
    with pytest.raises(ValueError, match="argv must not be empty"):
        job.with_argv(())
    with pytest.raises(ValueError, match="a sequence of words, not a string"):
        job.with_argv("rcp-ndcg run resume")


def test_a_wheelhouse_scheme_uv_cannot_read_is_refused_at_config_time() -> None:
    """uv's --find-links and --constraints read local directories and http(s) URLs: a bucket scheme would fail
    at job start, so the config refuses it and names the fix."""
    from rcp_ndcg.errors import ConfigError

    for value in ("gs://bucket/wheels", "s3://bucket/wheels", "gcs://bucket/wheels", "gs:/bucket/wheels", "  ", " /x "):
        with pytest.raises(ConfigError, match="uv cannot read|empty or has surrounding whitespace"):
            SlurmRunner(wheelhouse=value, container_runtime="pyxis")
        with pytest.raises(ConfigError, match="uv cannot read|empty or has surrounding whitespace"):
            SlurmRunner(constraints=value, container_runtime="pyxis")
    with pytest.raises(ConfigError, match="names no host"):
        SlurmRunner(wheelhouse="http://", container_runtime="pyxis")  # uv resolves no host from this
    # a local path, a file:// URL and an http(s):// URL are readable by uv
    assert SlurmRunner(wheelhouse="/shared/wheels", container_runtime="pyxis").options.wheelhouse
    assert SlurmRunner(wheelhouse="file:///shared/wheels", container_runtime="pyxis").options.wheelhouse
    assert SlurmRunner(wheelhouse="https://storage.example/wheels", container_runtime="pyxis").options.wheelhouse


def test_a_local_wheelhouse_path_is_recorded_absolute_and_a_url_is_left_alone() -> None:
    """The job record re-creates the runner from any directory: a local wheelhouse or constraints path is
    absolute like the other PATHS; a URL already names its location."""
    options = SlurmOptions(wheelhouse="wheels", constraints="wheels/c.txt", container_runtime="pyxis")
    resolved = options.resolved()
    assert resolved["wheelhouse"].endswith("/wheels") and resolved["wheelhouse"].startswith("/")
    assert resolved["constraints"].endswith("/wheels/c.txt")
    remote = SlurmRunner(wheelhouse="https://storage.example/wheels", container_runtime="pyxis").options.resolved()
    assert remote["wheelhouse"] == "https://storage.example/wheels"


@pytest.mark.parametrize("name", ["HF HOME", "A;echo INJECTED;B", "1X", "", "X-Y"])
def test_an_environment_name_that_is_no_shell_identifier_is_refused_everywhere(name: str) -> None:
    """`export HF HOME=/x` once set HOME silently, and `export A;echo INJECTED;B=1` ran echo."""
    from pydantic import ValidationError

    from rcp_ndcg.runners import JobOptions, ServeConfig

    with pytest.raises(ValidationError, match="not an environment variable name"):
        JobSpec(name="j", argv=("true",), env={name: "1"})
    with pytest.raises(ValidationError, match="not an environment variable name"):
        JobOptions(env={name: "1"})
    with pytest.raises(ValidationError, match="not an environment variable name"):
        ServeConfig(image="i:1", command="serve", env={name: "1"})
    assert JobSpec(name="j", argv=("true",), env={"_HF_HOME2": "1"}).env == {"_HF_HOME2": "1"}


def test_the_uv_bootstrap_installs_from_the_wheelhouse_when_one_is_given(tmp_path) -> None:
    """An air-gapped node (the wheelhouse option's whole point) has no PyPI: the bootstrap installs uv from
    the wheelhouse too, --no-index, like the release install it stands next to (the uv wheel is staged there)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" > {tmp_path}/pip-args\n'
        'target="${@: -2:1}"\n'
        'mkdir -p "$target/bin"\n'
        'printf \'#!/usr/bin/env bash\\necho uvx "$*"\\n\' > "$target/bin/uvx"\n'
        'chmod +x "$target/bin/uvx"\n'
    )
    (bin_dir / "python3").chmod(0o755)
    for tool in ("bash", "mkdir", "chmod"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    script = worker_script(
        JobSpec(name="j", argv=("rcp-ndcg", "--help")),
        install=True,
        workdir=None,
        wheelhouse="/shared/wheels",
    )
    env = {"PATH": str(bin_dir), "TMPDIR": str(tmp_path)}
    subprocess.run([str(bin_dir / "bash"), "-c", script], env=env, capture_output=True, text=True, check=True)
    pip = (tmp_path / "pip-args").read_text()
    assert "--find-links /shared/wheels" in pip and "--no-index" in pip, pip
    assert "pypi" not in pip.lower()
