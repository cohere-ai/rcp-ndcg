"""The job scripts: the freeze-diff guard, submit.sh's argv under KJOBS=echo, and the token-leak grep.

The scripts run on the node (bootstrap.sh, wave0.sh) or wherever the operator runs them (rc_build.sh,
submit.sh); what can be exercised on CPU is their plan, their guards and their refusals - never a node.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import sandbox_path

JOBS = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs"
BOOTSTRAP = JOBS / "bootstrap.sh"
SUBMIT = JOBS / "submit.sh"
RC_BUILD = JOBS / "rc_build.sh"
REPORT_PY = JOBS / "report.py"
REFERENCE_DEPS = JOBS / "reference_deps.py"
WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "wave0.sh"
E2E_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "e2e.sh"
WAVE0_HOST = JOBS / "wave0_host.py"

SCRIPTS = (BOOTSTRAP, SUBMIT, RC_BUILD, WAVE0_SH, E2E_SH)

needs_shellcheck = pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")


@needs_shellcheck
def test_every_script_passes_shellcheck() -> None:
    for script in SCRIPTS:
        completed = subprocess.run(["shellcheck", str(script)], capture_output=True, text=True)
        assert completed.returncode == 0, f"{script.name}: {completed.stdout}{completed.stderr}"


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_script_parses(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_bootstrap_without_arguments_prints_usage_and_fails() -> None:
    """No arguments: no mode. (Without the mounted helpers it refuses earlier, equally loudly.)"""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path(__file__).parent), "RCP_GCS_AUTH_FILE": "/nonexistent"}
    completed = subprocess.run(["bash", str(BOOTSTRAP)], capture_output=True, text=True, env=env)
    assert completed.returncode == 1
    assert "GCS helpers" in completed.stderr


# --- the freeze-diff guard (the engine environment may gain exactly the declared plugins) --------------


@pytest.fixture()
def bootstrap_functions() -> str:
    """The guard functions, by sourcing bootstrap.sh (its main runs nothing when sourced)."""
    return str(BOOTSTRAP)


def test_bootstrap_freeze_guard_passes_an_unchanged_environment(bootstrap_functions: str, tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    shutil.copy(before, after)
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_bootstrap_freeze_guard_fires_beyond_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    """A fake pip freeze that installs a rogue package fails the guard, with the package named."""
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    after.write_text("pkg-a==1.0\npkg-b==2.0\nrogue==9.9\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "changed beyond the declared plugins" in completed.stderr
    assert "rogue" in completed.stderr


def test_bootstrap_freeze_guard_allows_exactly_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    """The plugin's two freeze shapes pass: name==version and the local-wheel direct URL."""
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    for installed in (
        "My_Plugin @ file:///opt/wheels/My_Plugin-1.2.3-py3-none-any.whl\n",
        "my_plugin==1.2.3\n",
    ):
        after.write_text("pkg-a==1.0\npkg-b==2.0\n" + installed, encoding="utf-8")
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
                "bash",
                str(before),
                str(after),
                str(allowed),
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (installed, completed.stderr)


def test_bootstrap_freeze_guard_fires_on_an_upgrade_beyond_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\n", encoding="utf-8")
    after.write_text("pkg-a==1.1\nmy_plugin==1.2.3\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1


def test_freeze_name_of_parses_wheel_names_and_specs(bootstrap_functions: str) -> None:
    """A wheel filename, a plain name, a versioned spec and a staged path all give the canonical name."""
    cases = [
        ("My_Plugin-1.2.3-py3-none-any.whl", "my-plugin"),  # a wheel filename: the first dash splits
        ("/opt/staged/fixture-plug/plugin_wheel-1.0.0-py3-none-any.whl", "plugin-wheel"),
        ("some-plugin", "some-plugin"),  # a pip spec keeps its dashes
        ("my-plugin==1.2.3", "my-plugin"),
        ("/staged/org/pkg-2.0", "pkg-2-0"),  # consistent with how its freeze line canonicalises
    ]
    for argument, expected in cases:
        completed = subprocess.run(
            ["bash", "-c", f'source "{bootstrap_functions}" && freeze_name_of "$1"', "bash", argument],
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == expected, argument


# --- the recipe plugin wheels (the staged tree and the staged wheelhouse, never an index) ------------


def _fake_engine_python(tmp_path: Path, *, fail_spec: str) -> Path:
    """A fake ENGINE_PYTHON: logs every invocation's argv, fails the install of ``fail_spec``."""
    log = tmp_path / "engine-python.log"
    failure = ""
    if fail_spec:
        failure = (
            f'if [[ "$*" == *"{fail_spec}"* ]]; then '
            f'echo "ERROR: No matching distribution for {fail_spec}" >&2; exit 1; fi\n'
        )
    script = tmp_path / "fake-engine-python"
    script.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {log!s}\n{failure}exit 0\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _install_plugin_wheels(
    tmp_path: Path, specs: str, *, fail_spec: str = "", extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Run bootstrap's install_plugin_wheels (its functions, by sourcing) with a fake engine python.

    ``extra`` names EXTRA_DIRS entries staged under ``<stage>/extra/<name>/``; a name ending in ``/wheelhouse``
    stages that entry with a wheelhouse directory, any other name without one."""
    stage = tmp_path / "stage"
    (stage / "wheelhouse").mkdir(parents=True)
    for entry in extra:
        (stage / "extra" / entry).mkdir(parents=True)
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    specs_file = tmp_path / "specs.txt"
    specs_file.write_text(specs, encoding="utf-8")
    allowed = tmp_path / "allowed.txt"
    failed = tmp_path / "failed.txt"
    fake = _fake_engine_python(tmp_path, fail_spec=fail_spec)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{BOOTSTRAP}" && ENGINE_PYTHON="{fake}" STAGE_DIR="{stage}" RECIPES_ROOT="{recipes}" '
            f'install_plugin_wheels "{specs_file}" "{allowed}" "{failed}"',
        ],
        capture_output=True,
        text=True,
    )
    return completed


def test_bootstrap_installs_a_named_plugin_from_the_staged_wheelhouse_only(tmp_path: Path) -> None:
    """A plugin named by a recipe and not staged as a file installs from the staged wheelhouse only
    (--no-index --find-links <stage>/wheelhouse) - never an index (an index search once found no matching
    distribution although the wheel sat in the staged wheelhouse)."""
    completed = _install_plugin_wheels(tmp_path, "my-plugin==1.2.3\n")
    assert completed.returncode == 0, completed.stderr
    log = (tmp_path / "engine-python.log").read_text(encoding="utf-8")
    assert "my-plugin==1.2.3" in log
    assert "--no-index" in log and "--find-links" in log
    argv = log.split()
    links = [argv[index + 1] for index, word in enumerate(argv) if word == "--find-links"]
    # Exactly the stage's wheelhouse: with no extra/ entry the extra-wheelhouse glob matches nothing, and its
    # unexpanded pattern is never passed on as a link.
    assert links == [str(tmp_path / "stage" / "wheelhouse")]
    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8").strip() == "my-plugin"
    assert not (tmp_path / "failed.txt").exists() or not (tmp_path / "failed.txt").read_text(encoding="utf-8").strip()


def test_bootstrap_finds_a_named_plugin_in_every_staged_extra_wheelhouse(tmp_path: Path) -> None:
    """A plugin wheel staged through EXTRA_DIRS lands under <stage>/extra/<name>/wheelhouse: the named install
    also finds links there -- each existing extra wheelhouse, still never an index."""
    completed = _install_plugin_wheels(tmp_path, "my-plugin\n", extra=("one/wheelhouse", "two", "three/wheelhouse"))
    assert completed.returncode == 0, completed.stderr
    argv = (tmp_path / "engine-python.log").read_text(encoding="utf-8").split()
    links = [argv[index + 1] for index, word in enumerate(argv) if word == "--find-links"]
    stage = tmp_path / "stage"
    assert links == [
        str(stage / "wheelhouse"),
        str(stage / "extra" / "one" / "wheelhouse"),
        str(stage / "extra" / "three" / "wheelhouse"),
    ]
    assert "--no-index" in argv and argv[-1] == "my-plugin"


def test_bootstrap_a_plugin_found_nowhere_is_recorded_with_its_exact_name(tmp_path: Path) -> None:
    """A plugin found nowhere never fails the job: it is recorded with its exact name (the wave then
    fails exactly the recipes that name it) and the rest of the plugins still install."""
    exact = "Private-Plugin.Name==1.2.3"
    completed = _install_plugin_wheels(tmp_path, f"{exact}\nother-plugin\n", fail_spec=exact)
    assert completed.returncode == 0, completed.stderr  # the loop carries on; the job never dies here
    assert exact in completed.stderr  # reported, with the exact name (not a canonicalised one)
    assert (tmp_path / "failed.txt").read_text(encoding="utf-8").splitlines() == [exact]
    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8").splitlines() == ["other-plugin"]
    assert "other-plugin" in (tmp_path / "engine-python.log").read_text(encoding="utf-8")


# --- the reference venv: the image's torch stack, kept by constraint and checked after install ------


def _bash_bootstrap_function(body: str) -> subprocess.CompletedProcess[str]:
    """Run bootstrap.sh's sourced shell functions (one per concept under test)."""
    return subprocess.run(["bash", "-c", f'source "{BOOTSTRAP}" && {body}'], capture_output=True, text=True)


def test_image_constraints_are_the_images_full_freeze(tmp_path: Path) -> None:
    """The reference install's constraint file is the image's FULL pip freeze (the
    torch/torchvision/torchaudio/triton stack included): pip then resolves nothing of the image stack
    (--no-deps) and nothing of it can be replaced (a resolved install fails on the image torch's
    unregistered dependency tree)."""
    freeze = tmp_path / "freeze.txt"
    freeze.write_text(
        "nvidia-nccl-cu13==2.29.7\npip==25.2\ntorch==2.13.0\ntorchvision==0.28.0\n"
        "torchaudio==2.13.0\ntriton==3.5.0\nvllm==0.31.0\n",
        encoding="utf-8",
    )
    constraints = tmp_path / "constraints.txt"
    completed = _bash_bootstrap_function(f'image_constraints "{freeze}" "{constraints}"')
    assert completed.returncode == 0, completed.stderr
    assert constraints.read_text(encoding="utf-8") == freeze.read_text(encoding="utf-8")
    assert "torch==2.13.0" in constraints.read_text(encoding="utf-8")  # the item-4 stack, pinned too


def test_image_constraints_refuses_to_leave_the_install_unconstrained(tmp_path: Path) -> None:
    """A freeze with no torch== pin: an unconstrained install could silently swap the image's CUDA
    torch for the wheelhouse's CPU torch -- an error with a hint, not a default."""
    freeze = tmp_path / "freeze.txt"
    freeze.write_text("pip==25.2\nvllm==0.31.0\n", encoding="utf-8")
    constraints = tmp_path / "constraints.txt"
    completed = _bash_bootstrap_function(f'image_constraints "{freeze}" "{constraints}"')
    assert completed.returncode != 0
    assert "torch==" in completed.stderr and "REFERENCE_REQUIREMENTS" in completed.stderr


def _fake_reference_python(tmp_path: Path, *, fail: bool) -> Path:
    """A fake reference interpreter: logs its argv (optionally failing every install)."""
    log = tmp_path / "reference-python.log"
    failure = "exit 1\n" if fail else "exit 0\n"
    script = tmp_path / "fake-reference-python"
    script.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {log!s}\n{failure}',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_reference_install_uses_the_staged_wheelhouse_under_the_image_pins(tmp_path: Path) -> None:
    """The reference install: from the staged wheelhouse only, --no-deps (the image's stack is never
    resolved), held to the image's full freeze as constraints (so nothing of the image can be
    replaced)."""
    fake = _fake_reference_python(tmp_path, fail=False)
    pins, requirements = tmp_path / "freeze.txt", tmp_path / "req.txt"
    pins.write_text("torch==2.13.0\nnvidia-nccl-cu13==2.29.7\n", encoding="utf-8")
    requirements.write_text("transformers==4.57.0\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    completed = _bash_bootstrap_function(f'reference_install "{fake}" "{pins}" "{requirements}" "{wheelhouse}"')
    assert completed.returncode == 0, completed.stderr
    log = (tmp_path / "reference-python.log").read_text(encoding="utf-8")
    assert "pip install" in log
    assert "--no-deps" in log  # the image's stack is never resolved (the nvidia-nccl failure)
    assert "--no-index" in log and f"--find-links {wheelhouse}" in log
    assert f"-c {pins}" in log and f"-r {requirements}" in log


def test_reference_complete_installs_the_venvs_own_missing_deps(tmp_path: Path) -> None:
    """What --no-deps cannot pull is completed from the staged wheelhouse (jobs/reference_deps.py),
    and the bootstrap fails loudly when the wheelhouse cannot satisfy it."""
    fake = _fake_reference_python(tmp_path, fail=True)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    helper = Path(__file__).resolve().parent.parent / "jobs" / "reference_deps.py"
    completed = _bash_bootstrap_function(f'reference_complete "{helper}" "{fake}" "{wheelhouse}"')
    assert completed.returncode != 0
    assert "requirements-reference.txt" in completed.stderr and "wheelhouse" in completed.stderr


@pytest.mark.parametrize(("sdk_dirs", "expected"), [("", "python"), ("planted", "gcloud")])
def test_bootstrap_searches_only_the_declared_sdk_dirs(tmp_path: Path, sdk_dirs: str, expected: str) -> None:
    """bootstrap.sh's own Cloud SDK search (after the auth script) reads RCP_GCLOUD_SDK_DIRS: an SDK under
    $HOME (the default list's first entry) stays off PATH when the list is empty, and is used when named."""
    sdk_bin = tmp_path / "google-cloud-sdk" / "bin"
    sdk_bin.mkdir(parents=True)
    (sdk_bin / "gcloud").write_text("#!/usr/bin/env true\n", encoding="utf-8")
    (sdk_bin / "gcloud").chmod(0o755)
    (tmp_path / "fakes").mkdir()
    auth = tmp_path / "gcs_auth.sh"
    auth.write_text("# placeholder auth: not the real script\n", encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "envs", str(tmp_path / "stage"), "--state", str(tmp_path / "state")],
        capture_output=True,
        text=True,
        env={
            "PATH": sandbox_path(tmp_path / "fakes", "bash", "mkdir", "mktemp", "rm", "date"),
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
            "RCP_GCLOUD_SDK_DIRS": str(sdk_bin) if sdk_dirs == "planted" else sdk_dirs,
            "RCP_GCS_AUTH_FILE": str(auth),
            "RCP_GCS_HELPER_SH": str(JOBS / "gcs.sh"),
            "RCP_GCS_HELPER_PY": str(JOBS / "gcs.py"),
            "RCP_REPORT_PY": str(REPORT_PY),
            "RCP_REFERENCE_DEPS_PY": str(REFERENCE_DEPS),
        },
    )
    assert f"bootstrap: GCS transfer path: {expected}\n" in completed.stderr, completed.stderr


def test_bootstrap_envs_end_to_end_reaches_the_report(tmp_path: Path) -> None:
    """Drive bootstrap main (envs mode, what wave 0 calls) end to end with stubbed externals - guards
    the whole run, not just sourced functions: every mounted helper resolves through its RCP_*
    override and the run completes with bootstrap.json (engine, client, reference with
    torch_is_image_build)."""
    work = tmp_path / "work"
    (work / "bin").mkdir(parents=True)
    (work / "gcs").mkdir()
    (work / "stage" / "wheelhouse").mkdir(parents=True)
    state = work / "state"
    real_python = Path("/usr/bin/python3")
    if not real_python.is_file():
        pytest.skip("no /usr/bin/python3 (the venv's python has no ensurepip)")
    # the stub GCS helper (sourced) + an empty auth script (executed, never printed)
    (work / "gcs" / "gcs.sh").write_text(
        'gcs_sdk_on_path() { :; }\ngcs_transfer_detect() { echo "stub"; }\ngcs_cp() { :; }\n', encoding="utf-8"
    )
    (work / "gcs" / "gcs.py").write_text("", encoding="utf-8")
    auth = work / "gcs_auth.sh"
    auth.write_text("", encoding="utf-8")
    # the stub engine python (the image's python3): vllm present, fla absent, a two-line freeze
    (work / "bin" / "python3").write_text(
        f"#!/usr/bin/env bash\nREAL={real_python}\n"
        'case "$*" in\n'
        '  *"import vllm"*)   echo "0.31.0"; exit 0 ;;\n'
        '  *"import fla"*)    exit 1 ;;\n'
        "  \"-m pip freeze\"*) printf 'torch==2.13.0\\nvllm==0.31.0\\n'; exit 0 ;;\n"
        '  "-m pip install"*) exit 0 ;;\n'
        "esac\n"
        'exec "$REAL" "$@"\n',
        encoding="utf-8",
    )
    (work / "bin" / "python3").chmod(0o755)
    # the stub client mechanism (uvx) and the stub uv (venv from the system python: it ships ensurepip)
    (work / "bin" / "uvx").write_text(
        "#!/usr/bin/env bash\n"
        'echo \'{"rcp-ndcg": "0.0.1", "rcp-ndcg-core": "0.0.1", "rcp-ndcg-vllm": "0.0.1", "inert_present": {}}\'\n',
        encoding="utf-8",
    )
    (work / "bin" / "uvx").chmod(0o755)
    (work / "bin" / "uv").write_text(
        f'#!/usr/bin/env bash\nif [[ "$1" == "venv" ]]; then exec {real_python} -m venv "${{@: -1}}"; fi\nexit 1\n',
        encoding="utf-8",
    )
    (work / "bin" / "uv").chmod(0o755)
    # the staged RC: the reference needs only pip (already satisfied in any pip-venv)
    stage = work / "stage"
    (stage / "requirements-reference.txt").write_text("pip\n", encoding="utf-8")
    (stage / "requirements-constraints.txt").write_text("torch==2.13.0\n", encoding="utf-8")
    files = [
        {
            "path": rel,
            "sha256": hashlib.sha256((stage / rel).read_bytes()).hexdigest(),
        }
        for rel in ("requirements-reference.txt", "requirements-constraints.txt")
    ]
    (stage / "manifest.json").write_text(
        json.dumps({"version": "0.0.1", "commit": "scratch", "files": files, "cpu_inert_wheels": []}),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{work / 'bin'}:{os.environ['PATH']}",
        "RCP_GCS_AUTH_FILE": str(auth),
        "RCP_GCS_HELPER_SH": str(work / "gcs" / "gcs.sh"),
        "RCP_GCS_HELPER_PY": str(work / "gcs" / "gcs.py"),
        "RCP_REPORT_PY": str(REPORT_PY),
        "RCP_REFERENCE_DEPS_PY": str(REFERENCE_DEPS),
        "UV_CACHE_DIR": str(work / "uv-cache"),
    }
    env.pop("REFERENCE_DEPS_PY", None)
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "envs", str(stage), "--state", str(state)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "unbound variable" not in completed.stderr
    report = json.loads((state / "bootstrap.json").read_text(encoding="utf-8"))
    assert set(report) >= {"engine", "client", "reference"}
    assert "torch_is_image_build" in report["reference"]  # item 4: torch recorded, with its build


def test_reference_install_conflict_fails_loudly(tmp_path: Path) -> None:
    """A requirement that would replace the image's torch stack: the install fails loudly (the pip
    conflict surfaces, with the way out in the bootstrap's message), never a silent swap."""
    fake = _fake_reference_python(tmp_path, fail=True)
    pins, requirements = tmp_path / "pins.txt", tmp_path / "req.txt"
    pins.write_text("torch==2.13.0\n", encoding="utf-8")
    requirements.write_text("torch==2.14.0\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    completed = _bash_bootstrap_function(f'reference_install "{fake}" "{pins}" "{requirements}" "{wheelhouse}"')
    assert completed.returncode != 0
    assert "replace the image's torch" in completed.stderr and "REFERENCE_REQUIREMENTS" in completed.stderr


def _torch_json(tmp_path: Path, name: str, version: str | None, cuda: str | None) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"version": version, "cuda": cuda}) + "\n", encoding="utf-8")
    return path


def test_check_reference_torch_accepts_the_image_build(tmp_path: Path) -> None:
    """The reference sees exactly the image's CUDA torch: torch_is_image_build is true, no failure."""
    image = _torch_json(tmp_path, "image.json", "2.13.0+cu128", "12.8")
    reference = _torch_json(tmp_path, "reference.json", "2.13.0+cu128", "12.8")
    completed = _bash_bootstrap_function(f'check_reference_torch "{reference}" "{image}"')
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "true"


def test_check_reference_torch_rejects_a_cpu_torch_on_a_gpu_node(tmp_path: Path) -> None:
    """A CPU torch where the image ships CUDA torch is the blocker this guards: a failed bootstrap."""
    image = _torch_json(tmp_path, "image.json", "2.13.0+cu128", "12.8")
    reference = _torch_json(tmp_path, "reference.json", "2.14.0+cpu", None)
    completed = _bash_bootstrap_function(f'check_reference_torch "{reference}" "{image}"')
    assert completed.returncode != 0
    assert completed.stdout.strip() == "false"
    assert "a CPU torch on a GPU node is a failed bootstrap" in completed.stderr


def test_check_reference_torch_keeps_the_images_own_cpu_build(tmp_path: Path) -> None:
    """An identical CPU torch pair IS the image's just not-CUDA build: recorded (torch_is_image_build
    false), no failure - the failed-bootstrap rule is a CPU torch where the CUDA image's build is."""
    image = _torch_json(tmp_path, "image.json", "2.13.0", None)
    reference = _torch_json(tmp_path, "reference.json", "2.13.0", None)
    completed = _bash_bootstrap_function(f'check_reference_torch "{reference}" "{image}"')
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "false"


def test_check_reference_torch_rejects_a_replaced_torch(tmp_path: Path) -> None:
    """A requirement that replaced the image's torch with another CUDA build is still a failure: the
    reference must keep the image's own torch stack."""
    image = _torch_json(tmp_path, "image.json", "2.13.0+cu128", "12.8")
    reference = _torch_json(tmp_path, "reference.json", "2.14.0+cu129", "12.9")
    completed = _bash_bootstrap_function(f'check_reference_torch "{reference}" "{image}"')
    assert completed.returncode != 0
    assert completed.stdout.strip() == "false"
    assert "2.13.0+cu128" in completed.stderr and "2.14.0+cu129" in completed.stderr


def test_torch_probe_reports_the_seen_torch_build() -> None:
    """The probe both sides run: the interpreter's torch version and whether it is a CUDA build.  The
    comparison happens in subprocesses - this harness process must stay free of torch (test_no_torch)."""
    completed = _bash_bootstrap_function(f'torch_probe "{sys.executable}"')
    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout)
    assert set(probe) == {"version", "cuda"}
    if probe["version"] is None:
        pytest.skip("torch is not importable in this environment")
    expected = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, check=True
    )
    assert probe["version"] == expected.stdout.strip()


# --- submit.sh: the operator's submission, KJOBS=echo prints the plan ----------------------------------


FAKE_TOKEN = "hf_fake_0123456789abcdef"


def _scratch_auth(tmp_path: Path) -> Path:
    """A placeholder auth file in tmp_path; submit.sh only names its path, never reads it."""
    path = tmp_path / "gcs_auth.sh"
    path.write_text("# placeholder: not the real script\n", encoding="utf-8")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    """The submit environment: config, auth and token paths from tmp_path, no machine paths."""
    config = tmp_path / "config.yaml"
    config.write_text("worker: {cpu: 4}\n", encoding="utf-8")
    token = tmp_path / "hf-token"
    token.write_text(f"{FAKE_TOKEN}\n", encoding="utf-8")
    return {
        "PATH": "/usr/bin:/bin",
        "KJOBS": "echo",
        "RCP_KJOBS_CONFIG": str(config),
        "RCP_GCS_AUTH_FILE": str(_scratch_auth(tmp_path)),
        "RCP_HF_TOKEN_FILE": str(token),
    }


def _submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    monkeypatch.chdir(JOBS.parent.parent.parent)
    env = _env(tmp_path)
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(SUBMIT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_submit_prints_the_expected_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KJOBS=echo prints one kjobs-go submit per wave, with the priority class and shm overrides."""
    completed = _submit(
        tmp_path, monkeypatch,
        "--priority", "dev-high", "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a",
    )  # fmt: skip
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("kjobs-go") or line.startswith("echo ")]
    words = shlex.split(lines[0])
    assert "submit" in words
    assert "app=rcp-wave-a" in words
    assert "priority_class=dev-high" in words
    assert "worker.shared_memory=128Gi" in words
    command = next(word for word in words if word.startswith("worker.command="))
    # The recipe wave: bootstrap.sh's wave mode, with the wave's list resolved on the node.
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh"
        " wave gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/wave-a --wave wave-a"
    )
    assert f"files.bootstrap.from_file={JOBS / 'bootstrap.sh'}" in words
    assert f"files.report.from_file={REPORT_PY}" in words
    assert f"files.refdeps.from_file={JOBS / 'reference_deps.py'}" in words  # the reference completion helper
    assert f"files.gcsauth.from_file={tmp_path / 'gcs_auth.sh'}" in words
    config_flag = words[words.index("-f") + 1]
    assert config_flag == str(tmp_path / "config.yaml")  # RCP_KJOBS_CONFIG, not a default path


def test_submit_chains_waves_beyond_max_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At most --max-jobs jobs in flight: wave i for i >= N depends on wave i-N (the plan shows it)."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--max-jobs",
        "2",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "w1",
        "w2",
        "w3",
        "w4",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    submissions = [line for line in completed.stdout.splitlines() if "submit" in line and "app=rcp-" in line]
    assert len(submissions) == 4
    deps = [shlex.split(line) for line in submissions]
    depends = [next((word for word in words if word.startswith("depends_on=")), None) for words in deps]
    assert depends == [None, None, "depends_on=rcp-w1", "depends_on=rcp-w2"]


def test_submit_never_prints_the_token_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The token file's value reaches the argv only in a real run; echo mode prints the substitution."""
    completed = _submit(tmp_path, monkeypatch, "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a")
    assert completed.returncode == 0
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"
    assert "secret.HF_TOKEN=" in completed.stdout  # the placeholder is part of the plan


def test_submit_wave0_mounts_the_wave0_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "wave0",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "wave0",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    assert any(word.startswith("files.wave0.from_file=") and word.endswith("wave0.sh") for word in words)
    assert any(word.startswith("files.wave0host.from_file=") for word in words)
    command = next(word for word in words if word.startswith("worker.command="))
    assert "/etc/rcp/files/wave0/wave0.sh" in command


def test_submit_creates_the_submit_dir_when_it_does_not_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RCP_SUBMIT_DIR is created when missing (nested paths included); the job CLI's log lands there."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kjobs = fake_bin / "kjobs-go"
    kjobs.write_text('#!/usr/bin/env bash\necho "submitted; follow with: kjobs logs rcp-wave-a"\n')
    kjobs.chmod(0o755)
    out_dir = tmp_path / "nested" / "submit-out"
    completed = _submit(
        tmp_path, monkeypatch,
        "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a",
        env_overrides={
            "KJOBS": str(kjobs),
            "RCP_SUBMIT_DIR": str(out_dir),
            "RCP_IMAGE_DIGEST": "sha256:" + "0" * 64,
        },
    )  # fmt: skip
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (out_dir / "kjobs-wave-a.log").is_file()


def test_submit_fails_with_a_usage_message_without_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No defaults: the script refuses to run without RCP_KJOBS_CONFIG, RCP_GCS_AUTH_FILE, RCP_HF_TOKEN_FILE."""
    for missing in ("RCP_KJOBS_CONFIG", "RCP_GCS_AUTH_FILE", "RCP_HF_TOKEN_FILE"):
        env = _env(tmp_path)
        env.pop(missing)
        monkeypatch.chdir(JOBS.parent.parent.parent)
        completed = subprocess.run(
            ["bash", str(SUBMIT), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode != 0, (missing, completed.stdout)
        assert missing in completed.stderr


def test_submit_refuses_an_unknown_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _submit(
        tmp_path, monkeypatch, "--script", "deploy", "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a"
    )
    assert completed.returncode == 2
    assert "bootstrap, wave0 or e2e" in completed.stderr


def test_submit_e2e_mounts_the_e2e_script_and_names_the_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The T4 submission (a scenario wave): e2e.sh mounted beside bootstrap.sh, the wave's scenario list
    resolved on the node, and no token value in the plan."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "e2e",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "e2e",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    command = next(word for word in words if word.startswith("worker.command="))
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/e2e/e2e.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/e2e --wave e2e"
    )
    assert any(word.startswith("files.e2e.from_file=") and word.endswith("e2e.sh") for word in words)
    assert any(word.startswith("files.bootstrap.from_file=") for word in words)
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"


def test_the_e2e_dry_plan_names_the_scenarios_and_the_client_mechanism() -> None:
    """``E2E_DRY=1`` prints what the node would run (the operator's plan), and runs nothing."""
    completed = subprocess.run(
        ["bash", str(E2E_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/e2e", "--wave", "e2e"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "E2E_DRY": "1", "RCP_GCS_AUTH_FILE": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "text-four-phases | outage | identity | vidore" in completed.stdout
    assert "python -m rcp_ndcg_test.e2e" in completed.stdout
    assert "bootstrap.sh envs" in completed.stdout


def test_the_e2e_script_needs_a_wave_name() -> None:
    completed = subprocess.run(
        ["bash", str(E2E_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/e2e"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "RCP_GCS_AUTH_FILE": "/nonexistent"},
    )
    assert completed.returncode == 2
    assert "--wave" in completed.stderr


# --- rc_build.sh: the pairs staging (one home: rcp-ndcg-vllm/pairs/) -------------------------


def _stage_pairs(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    """Source rc_build.sh and run its ``stage_pairs`` over a scratch checkout and stage."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{RC_BUILD}" && stage_pairs "{checkout}" "{stage}"',
            "bash",
        ],
        capture_output=True,
        text=True,
    )


def test_rc_build_stages_pairs_from_the_packages_home(tmp_path: Path) -> None:
    """The pairs files live at rcp-ndcg-vllm/pairs/: rc_build.sh stages exactly that directory."""
    checkout = tmp_path / "checkout"
    pairs = checkout / "rcp-ndcg-vllm" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "fixture-embed.jsonl").write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    completed = _stage_pairs(tmp_path, checkout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    staged = tmp_path / "stage" / "pairs" / "fixture-embed.jsonl"
    assert staged.is_file(), f"the pairs file was not staged: {sorted((tmp_path / 'stage').rglob('*'))}"


def test_rc_build_never_stages_a_root_pairs_directory(tmp_path: Path) -> None:
    """One home: a stray <checkout-root>/pairs/ is refused (never silently staged), said on stderr."""
    checkout = tmp_path / "checkout"
    (checkout / "pairs").mkdir(parents=True)
    (checkout / "pairs" / "stray.jsonl").write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    completed = _stage_pairs(tmp_path, checkout)
    assert completed.returncode != 0
    assert "rcp-ndcg-vllm/pairs" in completed.stderr
    assert not (tmp_path / "stage" / "pairs").exists()


def test_rc_build_without_any_pairs_stages_nothing(tmp_path: Path) -> None:
    """A checkout without pairs stages no pairs dir (the wave runner then reports its missing pairs)."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    completed = _stage_pairs(tmp_path, checkout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "stage" / "pairs").exists()
