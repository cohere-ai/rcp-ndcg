"""The worker script: one command, quoted once, exec'd so the scheduler's signals reach it; in a stock image, the
package installs itself through uvx."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from rcp_ndcg import __version__
from rcp_ndcg.runners import JobSpec
from rcp_ndcg.runners.kubernetes import KubernetesRunner
from rcp_ndcg.runners.local import LocalRunner
from rcp_ndcg.runners.script import CONSTRAINTS_URL, TORCH_CPU_INDEX, install_argv, worker_script
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
    gs = " ".join(install_argv(("rcp-ndcg", "--help"), wheelhouse="gs://bucket/wheelhouse"))
    assert "--find-links gs://bucket/wheelhouse --no-index" in gs
    assert "--constraints https://github.com/cohere-ai/rcp-ndcg/releases/download/v" in gs  # the release URL stays


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
    pod = KubernetesRunner(wheelhouse="gs://bucket/wheels", constraints="gs://bucket/wheels/c.txt").render([spec])["j"]
    assert "--find-links gs://bucket/wheels --no-index" in pod


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


def test_a_local_wheelhouse_path_is_recorded_absolute_and_a_url_is_left_alone() -> None:
    """The job record re-creates the runner from any directory: a local wheelhouse or constraints path is
    absolute like the other PATHS; a URL already names its location."""
    options = SlurmOptions(wheelhouse="wheels", constraints="wheels/c.txt", container_runtime="pyxis")
    resolved = options.resolved()
    assert resolved["wheelhouse"].endswith("/wheels") and resolved["wheelhouse"].startswith("/")
    assert resolved["constraints"].endswith("/wheels/c.txt")
    remote = SlurmRunner(wheelhouse="gs://b/w", container_runtime="pyxis").options.resolved()
    assert remote["wheelhouse"] == "gs://b/w"


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
        ServeConfig(image="i", command="serve", env={name: "1"})
    assert JobSpec(name="j", argv=("true",), env={"_HF_HOME2": "1"}).env == {"_HF_HOME2": "1"}
