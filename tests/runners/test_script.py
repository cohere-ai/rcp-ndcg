"""The worker script: one command, quoted once, exec'd so the scheduler's signals reach it; in a stock image, the
package installs itself through uvx."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from rcp_ndcg import __version__
from rcp_ndcg.runners import JobSpec
from rcp_ndcg.runners.script import CONSTRAINTS_URL, TORCH_CPU_INDEX, install_argv, worker_script


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
