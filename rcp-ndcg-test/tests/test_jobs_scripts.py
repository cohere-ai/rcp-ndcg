"""The job scripts: the freeze-diff guard, submit.sh's argv under KJOBS=echo, and the token-leak grep.

The scripts run on the node (bootstrap.sh, wave0.sh) or wherever the operator runs them (rc_build.sh,
submit.sh); what can be exercised on CPU is their plan, their guards and their refusals - never a node.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.conftest import sandbox_path

JOBS = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs"
BOOTSTRAP = JOBS / "bootstrap.sh"
SUBMIT = JOBS / "submit.sh"
RC_BUILD = JOBS / "rc_build.sh"
REPORT_PY = JOBS / "report.py"
REFERENCE_DEPS = JOBS / "reference_deps.py"
WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "wave0.sh"
E2E_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "e2e.sh"
CASEFILL_SH = JOBS / "casefill.sh"
JUDGECHECK_SH = JOBS / "judgecheck.sh"
JUDGECHECK_PROBE = JOBS / "judgecheck_probe.py"
TOPK_PROBE_SH = JOBS / "topk_probe.sh"
TOPK_PROBE_PY = JOBS / "topk_probe.py"
TOPK_PROBE_COSINES = JOBS / "wave-cosines-topk-small.json"
WAVE0_HOST = JOBS / "wave0_host.py"

SCRIPTS = (BOOTSTRAP, SUBMIT, RC_BUILD, WAVE0_SH, E2E_SH, CASEFILL_SH, JUDGECHECK_SH, TOPK_PROBE_SH)

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
    """A fake ENGINE_PYTHON: logs every invocation's argv, answers the installed-plugin probe from
    ``FAKE_PLUGIN_VERSION`` (default 0.0.1), and fails the install of ``fail_spec``."""
    log = tmp_path / "engine-python.log"
    failure = ""
    if fail_spec:
        failure = (
            f'if [[ "$*" == *"{fail_spec}"* ]]; then '
            f'echo "ERROR: No matching distribution for {fail_spec}" >&2; exit 1; fi\n'
        )
    script = tmp_path / "fake-engine-python"
    script.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {log!s}\n'
        'if [[ "$*" == *"import importlib.metadata"* ]]; then echo "${FAKE_PLUGIN_VERSION:-0.0.1}"; exit 0; fi\n'
        f"{failure}exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _install_plugin_wheels(
    tmp_path: Path,
    specs: str,
    *,
    fail_spec: str = "",
    extra: tuple[str, ...] = (),
    wheels: tuple[str, ...] = (),
    installed: str = "0.0.1",
) -> subprocess.CompletedProcess[str]:
    """Run bootstrap's install_plugin_wheels (its functions, by sourcing) with a fake engine python.

    ``extra`` names EXTRA_DIRS entries staged under ``<stage>/extra/<name>/``; a name ending in ``/wheelhouse``
    stages that entry with a wheelhouse directory, any other name without one.  ``wheels`` names stub wheel
    files to write under the stage (relative paths), so a test can stage several versions.  ``installed`` is
    the version the fake engine environment reports for ``rcp-ndcg-vllm`` after the install."""
    stage = tmp_path / "stage"
    (stage / "wheelhouse").mkdir(parents=True)
    for entry in extra:
        (stage / "extra" / entry).mkdir(parents=True)
    for relative in wheels:
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub wheel")
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
        env={**os.environ, "FAKE_PLUGIN_VERSION": installed},
    )
    return completed


def test_bootstrap_pins_the_shipped_plugin_install_to_the_staged_wheel(tmp_path: Path) -> None:
    """Item 9/F2: the engine installs the exact staged wheel the wave cross-checks against the behaviour
    fingerprint.  A bare name would let pip pick the highest version across every extra wheelhouse, so
    the engine could run a plugin build the recording's key does not cover."""
    completed = _install_plugin_wheels(
        tmp_path,
        "rcp-ndcg-vllm\n",
        extra=("private/wheelhouse",),
        wheels=(
            "wheelhouse/rcp_ndcg_vllm-0.0.1-py3-none-any.whl",
            "extra/private/wheelhouse/rcp_ndcg_vllm-0.0.2-py3-none-any.whl",
        ),
    )
    assert completed.returncode == 0, completed.stderr
    log = (tmp_path / "engine-python.log").read_text(encoding="utf-8")
    assert "wheelhouse/rcp_ndcg_vllm-0.0.1-py3-none-any.whl" in log, log
    assert "0.0.2" not in log, log


def test_bootstrap_refuses_a_versioned_shipped_plugin_spec_the_staged_wheel_cannot_satisfy(
    tmp_path: Path,
) -> None:
    """Item 9: a versioned shipped-plugin spec installs as named, and when the version the engine ended up
    with differs from the staged wheel the wave hashes, the plugin is refused with its exact name -- the
    recipe fails instead of recording a corpus that claims code the engine did not run."""
    completed = _install_plugin_wheels(
        tmp_path,
        "rcp-ndcg-vllm==0.0.2\n",
        wheels=("wheelhouse/rcp_ndcg_vllm-0.0.1-py3-none-any.whl",),
        installed="0.0.2",
    )
    assert completed.returncode == 0, completed.stderr
    log = (tmp_path / "engine-python.log").read_text(encoding="utf-8")
    assert "rcp-ndcg-vllm==0.0.2" in log, log  # installed as named, never silently overridden
    assert (tmp_path / "failed.txt").read_text(encoding="utf-8").strip() == "rcp-ndcg-vllm==0.0.2"
    assert "staged" in completed.stderr and "0.0.1" in completed.stderr


def test_bootstrap_accepts_a_versioned_spec_the_staged_wheel_satisfies(tmp_path: Path) -> None:
    """A versioned spec whose installed version equals the staged wheel's passes the item-9 check."""
    completed = _install_plugin_wheels(
        tmp_path,
        "rcp-ndcg-vllm==0.0.1\n",
        wheels=("wheelhouse/rcp_ndcg_vllm-0.0.1-py3-none-any.whl",),
        installed="0.0.1",
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8").strip() == "rcp-ndcg-vllm"
    failed = tmp_path / "failed.txt"
    assert not failed.exists() or not failed.read_text(encoding="utf-8").strip()


def test_bootstrap_refuses_a_hostile_manifest_version_end_to_end(tmp_path: Path) -> None:
    """Security F2: the downloaded manifest's version is validated before any use, so the payload never
    runs even though the wrapper quoting is the second line of defence."""
    work = tmp_path / "work"
    (work / "gcs").mkdir(parents=True)
    (work / "gcs" / "gcs.sh").write_text(
        'gcs_sdk_on_path() { :; }\ngcs_transfer_detect() { echo "stub"; }\ngcs_cp() { :; }\n', encoding="utf-8"
    )
    (work / "gcs" / "gcs.py").write_text("", encoding="utf-8")
    auth = work / "gcs_auth.sh"
    auth.write_text("", encoding="utf-8")
    pwned = tmp_path / "pwned"
    stage = work / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text(
        json.dumps(
            {
                "version": f"0.0.1$(touch {pwned})",
                "commit": "scratch",
                "files": [],
                "cpu_inert_wheels": [],
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "RCP_GCS_AUTH_FILE": str(auth),
        "RCP_GCS_HELPER_SH": str(work / "gcs" / "gcs.sh"),
        "RCP_GCS_HELPER_PY": str(work / "gcs" / "gcs.py"),
        "UV_CACHE_DIR": str(work / "uv-cache"),
    }
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "envs", str(stage), "--state", str(work / "state")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode != 0
    assert "not a plain version string" in completed.stderr
    assert not pwned.exists()


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


def test_reference_install_installs_the_locks_pins_from_the_wheelhouse(tmp_path: Path) -> None:
    """The family reference install: the lock's pins from the staged wheelhouse only, --no-deps (the
    image's stack is never resolved and the family's own pins take precedence over the image's copies)."""
    fake = _fake_reference_python(tmp_path, fail=False)
    lock = tmp_path / "reference.lock"
    lock.write_text("# rcp-reference-lock: rcp-reference-lock/1\ntransformers==4.57.6\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    completed = _bash_bootstrap_function(f'reference_install "{fake}" "{lock}" "{wheelhouse}"')
    assert completed.returncode == 0, completed.stderr
    log = (tmp_path / "reference-python.log").read_text(encoding="utf-8")
    assert "pip install" in log
    assert "--no-deps" in log
    assert "--no-index" in log and f"--find-links {wheelhouse}" in log
    assert f"-r {lock}" in log
    assert "-c " not in log  # no image-freeze constraints: the family's own pins win (decision 35)


def test_reference_complete_installs_the_venvs_own_missing_deps(tmp_path: Path) -> None:
    """What --no-deps cannot pull is completed from the staged wheelhouse (jobs/reference_deps.py),
    and the bootstrap fails loudly when the wheelhouse cannot satisfy it."""
    fake = _fake_reference_python(tmp_path, fail=True)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    helper = Path(__file__).resolve().parent.parent / "jobs" / "reference_deps.py"
    completed = _bash_bootstrap_function(f'reference_complete "{helper}" "{fake}" "{wheelhouse}"')
    assert completed.returncode != 0
    assert "reference.lock" in completed.stderr and "wheelhouse" in completed.stderr


#: The system tools ``bootstrap.sh`` itself calls on the path this test exercises (its own
#: ``command -v python3``, the shell tools of the envs path) -- every one the script reaches is on the
#: sandbox PATH, so a missing tool cannot make the run pass for a reason the test did not intend.
_BOOTSTRAP_ENV_TOOLS = (
    "bash",
    "cat",
    "date",
    "dirname",
    "mkdir",
    "mktemp",
    "python3",
    "realpath",
    "rm",
    "sed",
    "tr",
)


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
            "PATH": sandbox_path(tmp_path / "fakes", *_BOOTSTRAP_ENV_TOOLS),
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


def test_bootstrap_writes_a_quoted_client_wrapper_and_refuses_a_hostile_version(
    bootstrap_functions: str, tmp_path: Path
) -> None:
    """Security F2: the wrapper's argv comes from a DOWNLOADED manifest field, so every word is
    shell-quoted into the file and the version itself must be a plain version string.  A `$(...)`
    payload in the version never reaches the wrapper as shell syntax."""
    state = tmp_path / "state"
    state.mkdir()
    pwned = tmp_path / "pwned"
    # A hostile version string: quoted, it is inert; unquoted, bash would run the substitution when the
    # wrapper runs.  Single quotes in the test body keep the payload a literal in the sourcing shell.
    body = (
        "CLIENT_ARGS=(uvx --from 'rcp-ndcg-vllm[test]==0.0.1$(touch " + str(pwned) + ")' "
        "--with rcp-ndcg-test==0.0.1 --find-links '" + str(tmp_path / "harness") + "' --no-index)\n"
        f'UV_CACHE_DIR="{tmp_path}/cache"\n'
        f'write_client_wrapper "{state}" "{tmp_path}/uv"\n'
    )
    completed = _bash_bootstrap_function(body)
    assert completed.returncode == 0, completed.stderr
    wrapper = (state / "client").read_text(encoding="utf-8")
    assert "$(" not in wrapper, wrapper
    # Running the wrapper must not run the payload either: the exec line is a word, not shell syntax.
    subprocess.run(["bash", str(state / "client"), "--version"], capture_output=True, text=True)
    assert not pwned.exists()
    # The wrapper still names the harness and the harness wheelhouse directory, just quoted.
    assert "rcp-ndcg-test==0.0.1" in wrapper and "/harness" in wrapper
    # And the version guard refuses the same payload outright (PEP 440 shape only).
    refused = _bash_bootstrap_function("validate_version '0.0.1$(touch " + str(pwned) + ")'")
    assert refused.returncode != 0 and "version" in refused.stderr
    assert not pwned.exists()
    accepted = _bash_bootstrap_function('validate_version "0.0.1rc1"')
    assert accepted.returncode == 0, accepted.stderr


def test_bootstrap_finds_the_staged_plugin_wheel(bootstrap_functions: str, tmp_path: Path) -> None:
    """Item 9: the wave gets the staged rcp_ndcg_vllm wheel so it can hash its modules against the
    behaviour fingerprint's plugin inputs; a stage without the wheel passes no --plugin-wheel."""
    stage = tmp_path / "stage"
    (stage / "wheelhouse").mkdir(parents=True)
    wheel = stage / "wheelhouse" / "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"stub wheel")
    completed = _bash_bootstrap_function(f'STAGE_DIR="{stage}"; staged_plugin_wheel rcp-ndcg-vllm')
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(wheel)
    empty = tmp_path / "empty"
    (empty / "wheelhouse").mkdir(parents=True)
    completed = _bash_bootstrap_function(f'STAGE_DIR="{empty}"; staged_plugin_wheel rcp-ndcg-vllm')
    assert completed.returncode != 0 and completed.stdout.strip() == ""


# --- the wave-mode command: RCP_RECORD_CORPUS is the recording pass's switch ---------------------------


def test_bootstrap_wave_command_adds_record_corpus_only_when_asked(tmp_path: Path) -> None:
    """The recording pass's switch (plan v2 C1): ``RCP_RECORD_CORPUS=1`` adds ``--record-corpus`` to
    the wave-mode ``run_wave`` argv, and unset (or 0) leaves the argv byte-identical.

    Wave mode's argv was untested before this: the block execs ``run_wave`` directly and no test
    constructed the command.  The pairs directory, the staged plugin wheel and the reference store are
    resolved by the same helper the exec runs, so this asserts the command the pod really runs.
    """
    stage = tmp_path / "stage"
    (stage / "pairs").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    wave_list = stage / "wave-lists" / "wave-a.txt"

    def wave_command(record: str | None) -> list[str]:
        env = {"PATH": "/usr/bin:/bin"}
        if record is not None:
            env["RCP_RECORD_CORPUS"] = record
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{BOOTSTRAP}" && STATE="{state}" OUT_URI="gs://YOUR-BUCKET/waves/wave-a" '
                f'STAGE_DIR="{stage}" wave_command "{wave_list}" "{stage}/recipes" 8',
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.splitlines()

    base = [
        "--recipes",
        f"@{wave_list}",
        "--recipes-root",
        f"{stage}/recipes",
        "--gpus",
        "8",
        "--out",
        f"{state}/wave",
        "--upload",
        "gs://YOUR-BUCKET/waves/wave-a",
        "--record",
        "--failed-plugins",
        f"{state}/plugin-failures.txt",
        "--reference-root",
        f"{state}/reference",
        "--pairs-dir",
        f"{stage}/pairs",
    ]
    assert wave_command(None) == base
    assert wave_command("0") == base  # only 1 turns the recording on
    assert wave_command("1") == base[:15] + ["--record-corpus"] + base[15:]


def test_bootstrap_envs_end_to_end_reaches_the_report(tmp_path: Path) -> None:
    """Drive bootstrap main (envs mode, what wave 0 calls) end to end with stubbed externals - guards
    the whole run, not just sourced functions: every mounted helper resolves through its RCP_*
    override and the run completes with bootstrap.json (engine, client, and one reference environment
    per family with its lock hash and torch record; owner decision 35)."""
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
    # the staged RC: one family with a lock (no pins: the family needs nothing beyond the image)
    stage = work / "stage"
    lock = stage / "recipes" / "demo" / "reference.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        "# rcp-reference-lock: rcp-reference-lock/1\n# family: demo\n"
        "# image: vllm/vllm-openai:v0.31.0\n# own-torch: false\n",
        encoding="utf-8",
    )
    # the stub client mechanism (uvx): node-shaped -- it refuses a closure without the harness wheel,
    # logs its argv, then runs the probe/wrapper command the bootstrap asked for (so the probe's own
    # imports, rcp_ndcg_test included, actually execute).
    uvx_log_path = work / "uvx.log"
    (work / "bin" / "uvx").write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'%s\\n\' "$*" >> "{uvx_log_path}"\n'
        'case "$*" in *"--with rcp-ndcg-test==0.0.1"*) ;; *) '
        'echo "uvx: the harness spec is missing" >&2; exit 1 ;; esac\n'
        'case "$*" in *"/harness"*) ;; *) echo "uvx: the harness find-links is missing" >&2; exit 1 ;; esac\n'
        'if [[ "$*" == *"reference_env check"* ]]; then\n'
        f'  sha="$(sha256sum "{lock}" | cut -d" " -f1)"\n'
        '  echo "{\\"family\\": \\"demo\\", \\"lock_sha256\\": \\"$sha\\", \\"facts\\": {\\"torch\\": null}}"\n'
        "  exit 0\n"
        "fi\n"
        'args=("$@")\n'
        "i=0\n"
        "while (( i < ${#args[@]} )); do\n"
        '  case "${args[i]}" in\n'
        "    --from|--with|--constraints|--find-links) ((i+=2)) ;;\n"
        "    --no-index) ((i+=1)) ;;\n"
        "    *) break ;;\n"
        "  esac\n"
        "done\n"
        'exec "${args[@]:i}"\n',
        encoding="utf-8",
    )
    (work / "bin" / "uvx").chmod(0o755)
    # the stub uv (venv from the system python: it ships ensurepip)
    (work / "bin" / "uv").write_text(
        f'#!/usr/bin/env bash\nif [[ "$1" == "venv" ]]; then exec {real_python} -m venv "${{@: -1}}"; fi\nexit 1\n',
        encoding="utf-8",
    )
    (work / "bin" / "uv").chmod(0o755)
    (stage / "requirements-constraints.txt").write_text("torch==2.13.0\n", encoding="utf-8")
    (stage / "harness").mkdir()
    (stage / "harness" / "rcp_ndcg_test-0.0.1-py3-none-any.whl").write_bytes(b"stub harness wheel")
    files = [
        {
            "path": rel,
            "sha256": hashlib.sha256((stage / rel).read_bytes()).hexdigest(),
        }
        for rel in (
            "requirements-constraints.txt",
            "harness/rcp_ndcg_test-0.0.1-py3-none-any.whl",
            "recipes/demo/reference.lock",
        )
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
    reference = report["reference"]
    assert reference["n_families"] == 1
    assert "demo" in reference["families"]
    assert reference["families"]["demo"]["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert "torch_is_image_build" in reference["families"]["demo"]
    # A2: the client closure carries the unpublished harness wheel from its own staged directory, and the
    # probe imports rcp_ndcg_test in the client environment (the wave runner's first command needs it).
    uvx_log = (work / "uvx.log").read_text(encoding="utf-8")
    assert "--with rcp-ndcg-test==0.0.1" in uvx_log, uvx_log
    assert f"--find-links {stage}/harness" in uvx_log, uvx_log
    versions = json.loads((state / "client-versions.json").read_text(encoding="utf-8"))
    assert versions["rcp-ndcg-test"] == "0.0.1", versions
    wrapper = (state / "client").read_text(encoding="utf-8")
    assert "rcp-ndcg-test==0.0.1" in wrapper and "/harness" in wrapper, wrapper
    # A recipes root without any family lock: the families file is still created (empty) and the
    # bootstrap completes with no reference environment (the round-3 finding's shape).
    lock.unlink()
    (stage / "manifest.json").write_text(
        json.dumps(
            {
                "version": "0.0.1",
                "commit": "scratch",
                "files": [entry for entry in files if entry["path"] != "recipes/demo/reference.lock"],
                "cpu_inert_wheels": [],
            }
        ),
        encoding="utf-8",
    )
    second_state = work / "state-empty"
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "envs", str(stage), "--state", str(second_state)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    empty = json.loads((second_state / "bootstrap.json").read_text(encoding="utf-8"))
    assert empty["reference"]["n_families"] == 0


def test_reference_install_failure_is_loud(tmp_path: Path) -> None:
    """A pin the staged wheelhouse cannot satisfy fails the install loudly, with the way out."""
    fake = _fake_reference_python(tmp_path, fail=True)
    lock = tmp_path / "reference.lock"
    lock.write_text("# rcp-reference-lock: rcp-reference-lock/1\ntransformers==4.57.6\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    completed = _bash_bootstrap_function(f'reference_install "{fake}" "{lock}" "{wheelhouse}"')
    assert completed.returncode != 0
    assert "reference.lock" in completed.stderr and "stage its wheels" in completed.stderr


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
        "RCP_SUBMIT_DIR": str(tmp_path / "submit"),
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
    # The recipe wave: bootstrap.sh's wave mode, with the wave's list resolved on the node, behind the token
    # wrapper that reads the mounted token file (the value never reaches an argv).
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash "
        "/etc/rcp/files/bootstrap/bootstrap.sh wave gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/wave-a --wave wave-a"
    )
    assert f"files.bootstrap.from_file={JOBS / 'bootstrap.sh'}" in words
    assert f"files.report.from_file={REPORT_PY}" in words
    assert f"files.refdeps.from_file={JOBS / 'reference_deps.py'}" in words  # the reference completion helper
    assert f"files.gcsauth.from_file={tmp_path / 'gcs_auth.sh'}" in words
    config_flag = words[words.index("-f") + 1]
    assert config_flag == str(tmp_path / "config.yaml")  # RCP_KJOBS_CONFIG, not a default path


def test_the_submit_output_dir_is_the_one_the_environment_gives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test environment pins ``RCP_SUBMIT_DIR`` inside tmp_path: a submit must never litter the system
    temp directory (the class of write outside ``tmp_path`` the runner tests fixed twice)."""
    completed = _submit(tmp_path, monkeypatch, "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "submit").is_dir()


def test_the_default_submit_output_dir_follows_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``RCP_SUBMIT_DIR``, the scratch output dir lives under ``${TMPDIR}``, never a hardcoded /tmp."""
    (tmp_path / "tmp").mkdir()
    completed = _submit(
        tmp_path,
        monkeypatch,
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "wave-a",
        env_overrides={"RCP_SUBMIT_DIR": "", "TMPDIR": str(tmp_path / "tmp")},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list((tmp_path / "tmp").glob("rcp-submit.*")), "the scratch output dir must live under TMPDIR"


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
    """The token file's value never reaches the plan: the plan mounts the file and the worker reads it there."""
    completed = _submit(tmp_path, monkeypatch, "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a")
    assert completed.returncode == 0
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"
    assert "secret.HF_TOKEN" not in completed.stdout
    assert f"files.hftoken.from_file={tmp_path / 'hf-token'}" in completed.stdout
    assert "/etc/rcp/hf_token" in completed.stdout


def test_the_token_value_never_reaches_the_job_clis_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run mounts the token file and reads it inside the job: the value is in no process argv (readable
    through ``/proc/<pid>/cmdline`` while the job CLI runs), only in the file the operator named."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    dump = tmp_path / "argv.txt"
    kjobs = fake_bin / "kjobs-go"
    kjobs.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > {shlex.quote(str(dump))}\n'
        'echo "submitted; follow with: kjobs logs rcp-wave-a"\n'
    )
    kjobs.chmod(0o755)
    out_dir = tmp_path / "submit-out"
    mount = tmp_path / "mounted-token"
    completed = _submit(
        tmp_path,
        monkeypatch,
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "wave-a",
        env_overrides={
            "KJOBS": str(kjobs),
            "RCP_SUBMIT_DIR": str(out_dir),
            "RCP_IMAGE_DIGEST": "sha256:" + "0" * 64,
            "RCP_HF_TOKEN_MOUNT": str(mount),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    argv = dump.read_text(encoding="utf-8")
    assert FAKE_TOKEN not in argv
    assert f"files.hftoken.from_file={tmp_path / 'hf-token'}" in argv
    assert f"files.hftoken.mount_path={mount}" in argv
    command = next(word for word in argv.splitlines() if word.startswith("worker.command="))
    assert "/etc/rcp/files/hftoken/hf_token_env.sh" in command
    wrapper = out_dir / "hf_token_env.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert wrapper.is_file() and FAKE_TOKEN not in text and "HF_TOKEN" in text
    # The wrapper runs: it exports the mounted file's value (the trailing newline stripped by $()) and execs
    # the worker with HF_TOKEN in its environment.
    mount.write_text(f"{FAKE_TOKEN}\n", encoding="utf-8")
    ran = subprocess.run(
        ["bash", str(wrapper), "bash", "-c", 'printf %s "$HF_TOKEN"'], capture_output=True, text=True, check=True
    )
    assert ran.stdout == FAKE_TOKEN


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


def test_submit_groups_a_wave_by_engine_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decisions 38 and 35: a wave list mixing engine images becomes one job per image, each with
    the recipe's container image and its own filtered list mounted (the brief's two-image wave)."""
    fixture_recipes = JOBS.parents[2] / "tests" / "fixtures" / "recipes"
    stage = tmp_path / "stage"
    (stage / "wave-lists").mkdir(parents=True)
    for family, image in (("family-a", "registry.example.com/a:1"), ("family-b", "registry.example.com/b:2")):
        shutil.copytree(fixture_recipes / "fixture-embed", stage / "recipes" / family)
        yaml = stage / "recipes" / family / "family.yaml"
        text = yaml.read_text(encoding="utf-8")
        text = text.replace("id: fixture-embed", f"id: {family}")
        text = text.replace('image: "vllm/vllm-openai:v0.31.0"', f'image: "{image}"')
        yaml.write_text(text, encoding="utf-8")
    (stage / "wave-lists" / "wave-a.txt").write_text("family-a\nfamily-b\n", encoding="utf-8")
    uv = shutil.which("uv")
    assert uv is not None, "submit.sh groups a wave through `uv run`; uv must be on PATH"
    completed = _submit(
        tmp_path, monkeypatch,
        str(stage), "gs://YOUR-BUCKET/waves", "wave-a",
        env_overrides={"RCP_IMAGE_DIGEST": "sha256:" + "0" * 64, "PATH": f"{Path(uv).parent}:/usr/bin:/bin"},
    )  # fmt: skip
    assert completed.returncode == 0, completed.stdout + completed.stderr
    submissions = [
        shlex.split(line)
        for line in completed.stdout.splitlines()
        if line.startswith("echo submit") or line.startswith("kjobs-go")
    ]
    assert len(submissions) == 2, completed.stdout
    images = {next(word for word in words if word.startswith("env.RCP_IMAGE=")) for words in submissions}
    assert images == {"env.RCP_IMAGE=registry.example.com/a:1", "env.RCP_IMAGE=registry.example.com/b:2"}
    apps = [next(word for word in words if word.startswith("app=")) for words in submissions]
    assert all(len(app) <= 26 for app in apps), apps  # a long app leaves the CLI's hostname slice empty
    assert len(set(apps)) == 2, apps  # one app per image
    commands = [next(word for word in words if word.startswith("worker.command=")) for words in submissions]
    assert all("--wave-list /etc/rcp/files/wavelist/wave-a." in command for command in commands)
    assert all(any(word.startswith("files.wavelist.from_file=") for word in words) for words in submissions)


def test_submit_groups_a_wave_by_engine_image_over_the_gcloud_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gs:// branch: a stubbed gcloud proves the directory fetch is recursive, creates the local
    destination first and copies the source's contents (the round-2 finding's shape)."""
    import os

    fixture_recipes = JOBS.parents[2] / "tests" / "fixtures" / "recipes"
    recipes = tmp_path / "recipes"
    for family, image in (("family-a", "registry.example.com/a:1"), ("family-b", "registry.example.com/b:2")):
        shutil.copytree(fixture_recipes / "fixture-embed", recipes / family)
        yaml = recipes / family / "family.yaml"
        text = yaml.read_text(encoding="utf-8")
        text = text.replace("id: fixture-embed", f"id: {family}")
        text = text.replace('image: "vllm/vllm-openai:v0.31.0"', f'image: "{image}"')
        yaml.write_text(text, encoding="utf-8")
    wave_list = tmp_path / "wave-a.txt"
    wave_list.write_text("family-a\nfamily-b\n", encoding="utf-8")
    log = tmp_path / "gcloud.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'if [[ "$*" == *"storage cp"* && "$*" == *"--recursive"* ]]; then\n'
        '  dest="${@: -1}"\n'
        '  src="${@: -2:1}"\n'
        '  [[ -d "$dest" ]] || { echo "strict-gcloud: destination $dest does not exist" >&2; exit 1; }\n'
        '  [[ "$src" == */recipes/* ]] || { echo "strict-gcloud: unexpected source $src" >&2; exit 1; }\n'
        '  cp -r "$RCP_TEST_RECIPES"/. "$dest"/\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"storage cp"* ]]; then\n'
        '  dest="${@: -1}"\n'
        '  src="${@: -2:1}"\n'
        '  [[ -d "$(dirname "$dest")" ]] || exit 1\n'
        '  [[ "$src" == */wave-a.txt ]] || exit 1\n'
        '  cp "$RCP_TEST_WAVE_LIST" "$dest"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "gcloud").chmod(0o755)
    completed = _submit(
        tmp_path, monkeypatch,
        "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a",
        env_overrides={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RCP_TEST_RECIPES": str(recipes),
            "RCP_TEST_WAVE_LIST": str(wave_list),
            "RCP_IMAGE_DIGEST": "sha256:" + "0" * 64,
        },
    )  # fmt: skip
    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = log.read_text(encoding="utf-8")
    assert "--recursive" in calls
    submissions = [
        shlex.split(line)
        for line in completed.stdout.splitlines()
        if line.startswith("echo submit") or line.startswith("kjobs-go")
    ]
    assert len(submissions) == 2, completed.stdout
    images = {next(word for word in words if word.startswith("env.RCP_IMAGE=")) for words in submissions}
    assert images == {"env.RCP_IMAGE=registry.example.com/a:1", "env.RCP_IMAGE=registry.example.com/b:2"}


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
    assert "bootstrap, wave0, e2e, casefill, judgecheck or topk-probe" in completed.stderr


def test_submit_forwards_record_corpus_only_when_the_operator_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recording pass's switch reaches the pod: ``RCP_RECORD_CORPUS`` set in the operator's
    environment becomes ``env.RCP_RECORD_CORPUS`` in the job spec (the way RCP_IMAGE is forwarded);
    unset, the spec carries no such line."""
    plain = _submit(tmp_path, monkeypatch, "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a")
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert "env.RCP_RECORD_CORPUS" not in plain.stdout

    recorded = _submit(
        tmp_path,
        monkeypatch,
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "wave-a",
        env_overrides={"RCP_RECORD_CORPUS": "1"},
    )
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    words = shlex.split(next(line for line in recorded.stdout.splitlines() if line.startswith("echo ")))
    assert "env.RCP_RECORD_CORPUS=1" in words


def test_submit_casefill_mounts_the_casefill_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cuda case-fill job: casefill.sh mounted beside bootstrap.sh (which it runs in envs mode) and
    run with the stage and the wave's output prefix; no token value in the plan."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "casefill",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "casefill",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    command = next(word for word in words if word.startswith("worker.command="))
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash "
        "/etc/rcp/files/casefill/casefill.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/casefill"
    )
    assert any(word.startswith("files.casefill.from_file=") and word.endswith("casefill.sh") for word in words)
    assert any(word.startswith("files.bootstrap.from_file=") for word in words)
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"


def test_casefill_end_to_end_fills_the_four_and_uploads_what_it_filled(tmp_path: Path) -> None:
    """Drive casefill.sh's node path with stubs: the bootstrap envs call carries the four-recipe wave
    list (the reference venvs), each fill runs through the client with ``--device cuda`` over the
    staged cases copy, one recipe's failure does not stop the next, the edited YAMLs upload under
    ``<OUT>/cases/<recipe>/``, and the job exits 1 with the failure recorded in the report."""
    stage = tmp_path / "stage"
    for recipe in ("topk-embed-v1-small", "qwen3-embedding-0.6b", "qwen3-vl-embedding-2b", "qwen3-vl-reranker-2b"):
        (stage / "cases" / recipe).mkdir(parents=True)
        (stage / "cases" / recipe / "gen-short.yaml").write_text(f"id: {recipe}/gen-short\n", encoding="utf-8")
        (stage / "cases" / recipe / "gen-video.yaml").write_text(f"id: {recipe}/gen-video\n", encoding="utf-8")
    (stage / "manifest.json").write_text('{"version": "0.0.1"}\n', encoding="utf-8")
    auth = tmp_path / "gcs_auth.sh"
    auth.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    helper = tmp_path / "gcs.py"
    helper.write_text("", encoding="utf-8")
    gcs_log = tmp_path / "gcs.log"
    gcs_sh = tmp_path / "gcs.sh"
    gcs_sh.write_text(
        'gcs_sdk_on_path() { :; }\ngcs_cp() { printf "%s\\n" "cp $*" >> "$STUB_GCS_LOG"; }\n',
        encoding="utf-8",
    )
    bootstrap_log = tmp_path / "bootstrap.log"
    client_log = tmp_path / "client.log"
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_BOOTSTRAP_LOG"\n'
        'state=""; wave_list=""\n'
        'while (($#)); do case "$1" in --state) state="$2"; shift 2;;'
        ' --wave-list) wave_list="$2"; shift 2;; *) shift;; esac; done\n'
        "while read -r recipe; do\n"
        '  case "$recipe" in\n'
        "    topk-embed-v1-small) family=topk-embed-v1 ;;\n"
        "    qwen3-embedding-0.6b) family=qwen3-embedding ;;\n"
        "    qwen3-vl-embedding-2b) family=qwen3-vl-embedding ;;\n"
        "    qwen3-vl-reranker-2b) family=qwen3-vl-reranker ;;\n"
        '    *) echo "stub bootstrap: unknown recipe $recipe" >&2; exit 1 ;;\n'
        "  esac\n"
        '  mkdir -p "$state/reference/$family/bin"\n'
        '  printf "#!/usr/bin/env bash\\nexit 0\\n" > "$state/reference/$family/bin/python"\n'
        '  chmod +x "$state/reference/$family/bin/python"\n'
        'done < "$wave_list"\n'
        'echo \'{"schema": "stub-bootstrap"}\' > "$state/bootstrap.json"\n'
        "cat > \"$state/client\" <<'CLIENT'\n"
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_CLIENT_LOG"\n'
        'recipe=""; cases=""; prev=""\n'
        'for arg in "$@"; do [[ "$prev" == "--recipe" ]] && recipe="$arg";'
        ' [[ "$prev" == "--cases-root" ]] && cases="$arg"; prev="$arg"; done\n'
        'if [[ "$recipe" == "qwen3-vl-embedding-2b" ]]; then'
        ' echo "casefill: the stub reference failed" >&2; exit 2; fi\n'
        'printf \'{"recipe": "%s", "mode": "score", "device": "cuda", "pending_before": 2,'
        ' "filled": [{"case_id": "%s/gen-short", "path": "%s/%s/gen-short.yaml",'
        ' "notes_updated": true}], "skipped": [],'
        ' "exceptions": [{"case_id": "%s/gen-video", "path": "%s/%s/gen-video.yaml",'
        ' "reason": "the reference refuses this container", "notes_updated": true}]}\\n\''
        ' "$recipe" "$recipe" "$cases" "$recipe" "$recipe" "$cases" "$recipe"\n'
        "CLIENT\n"
        'chmod +x "$state/client"\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(CASEFILL_SH), str(stage), "gs://YOUR-BUCKET/waves/casefill"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RCP_GCS_AUTH_FILE": str(auth),
            "RCP_GCS_HELPER_SH": str(gcs_sh),
            "RCP_GCS_HELPER_PY": str(helper),
            "RCP_BOOTSTRAP_SH": str(bootstrap),
            "RCP_REPORT_PY": str(REPORT_PY),
            "RCP_CASEFILL_DATE": "2026-10-12",
            "STUB_GCS_LOG": str(gcs_log),
            "STUB_BOOTSTRAP_LOG": str(bootstrap_log),
            "STUB_CLIENT_LOG": str(client_log),
        },
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr  # the failed recipe fails the job
    assert "casefill: 1 recipe(s) failed: qwen3-vl-embedding-2b" in completed.stderr
    state_line = next(line for line in completed.stdout.splitlines() if "casefill: state " in line)
    state = Path(state_line.split("casefill: state ", 1)[1])
    # the wave list is the four recipes (the reference environments the fills need)
    assert (state / "case-fill-wave.txt").read_text(encoding="utf-8").splitlines() == [
        "topk-embed-v1-small",
        "qwen3-embedding-0.6b",
        "qwen3-vl-embedding-2b",
        "qwen3-vl-reranker-2b",
    ]
    bootstrap_argv = bootstrap_log.read_text(encoding="utf-8")
    assert f"envs {stage} --state {state} --recipes {stage}/recipes" in bootstrap_argv, bootstrap_argv
    assert f"--wave-list {state}/case-fill-wave.txt" in bootstrap_argv, bootstrap_argv
    # every fill ran through the client, on cuda, over the cases copy; the failure did not stop the next
    fills = client_log.read_text(encoding="utf-8").splitlines()
    assert len(fills) == 4, fills
    for recipe, family in (
        ("topk-embed-v1-small", "topk-embed-v1"),
        ("qwen3-embedding-0.6b", "qwen3-embedding"),
        ("qwen3-vl-embedding-2b", "qwen3-vl-embedding"),
        ("qwen3-vl-reranker-2b", "qwen3-vl-reranker"),
    ):
        line = next(entry for entry in fills if f"--recipe {recipe} " in entry)
        assert f"--reference-python {state}/reference/{family}/bin/python" in line, line
        assert "--device cuda" in line and "--filled-at 2026-10-12" in line and "--budget-s 7200" in line, line
        assert f"--cases-root {state}/cases" in line, line
    # the edited YAMLs of the three successful recipes upload -- the filled cases and the refused
    # case's recorded exception -- and the failure did not stop the recipe after it; the failed one's
    # do not
    for recipe in ("topk-embed-v1-small", "qwen3-embedding-0.6b", "qwen3-vl-reranker-2b"):
        assert (state / "filled-cases" / recipe / "gen-short.yaml").is_file(), recipe
        assert (state / "filled-cases" / recipe / "gen-video.yaml").is_file(), recipe
    assert not (state / "filled-cases" / "qwen3-vl-embedding-2b").exists()
    calls = gcs_log.read_text(encoding="utf-8")
    assert f"cp {state}/filled-cases gs://YOUR-BUCKET/waves/casefill/cases dir" in calls, calls
    assert f"cp {state}/casefill-report.json gs://YOUR-BUCKET/waves/casefill/casefill-report.json file" in calls, calls
    report = json.loads((state / "casefill-report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["failed_step"] == "casefill"
    fills = report["fills"]
    assert fills["qwen3-vl-embedding-2b"]["state"] == "failed" and "exit 2" in fills["qwen3-vl-embedding-2b"]["error"]
    assert fills["qwen3-embedding-0.6b"]["recipe"] == "qwen3-embedding-0.6b"
    assert set(fills) == {
        "topk-embed-v1-small",
        "qwen3-embedding-0.6b",
        "qwen3-vl-embedding-2b",
        "qwen3-vl-reranker-2b",
    }
    assert report["bootstrap"]["schema"] == "stub-bootstrap"


def test_the_casefill_dry_plan_names_the_four_recipes_and_the_reference_pythons() -> None:
    """``CASEFILL_DRY=1`` prints what the node would run (the operator's plan), and runs nothing: the
    four recipes' case_fill commands with ``--device cuda`` and the staged cases root, the wave list
    the reference environments are built from, and the upload prefix."""
    completed = subprocess.run(
        ["bash", str(CASEFILL_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/casefill"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "CASEFILL_DRY": "1", "RCP_GCS_AUTH_FILE": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = completed.stdout
    for recipe, family in (
        ("topk-embed-v1-small", "topk-embed-v1"),
        ("qwen3-embedding-0.6b", "qwen3-embedding"),
        ("qwen3-vl-embedding-2b", "qwen3-vl-embedding"),
        ("qwen3-vl-reranker-2b", "qwen3-vl-reranker"),
    ):
        assert f"--recipe {recipe}" in plan, plan
        assert f"--reference-python <state>/reference/{family}/bin/python" in plan, plan
    assert plan.count("--device cuda") == 4, plan
    assert plan.count("--cases-root <stage>/cases") == 4, plan
    assert "bootstrap.sh envs" in plan and "case-fill-wave.txt" in plan
    assert "gs://YOUR-BUCKET/waves/casefill/cases/" in plan
    for recipe in ("topk-embed-v1-small", "qwen3-embedding-0.6b", "qwen3-vl-embedding-2b", "qwen3-vl-reranker-2b"):
        assert recipe in plan


def test_casefill_pins_the_four_recipes_to_the_catalog_and_the_cases() -> None:
    """The script's four recipe id -> family pairs are the real catalog's (a renamed family directory
    fails here first), and the four recipes' generated case corpus is the one the job fills: 41 cases,
    of which topk's ``query-cap-1024`` is ``kind: none`` -- the fill's one recorded skip, it carries no
    values by design -- and 14 are media-bearing (fillable since the media translation landed).  A new
    case, a changed kind or a media file moving the counts must revisit the job's scope; the model-card
    cases are not counted (the card printed their values, so no fill applies)."""
    from rcp_ndcg_vllm.recipe import load_recipe

    script = CASEFILL_SH.read_text(encoding="utf-8")
    cases_root = Path(__file__).resolve().parents[1] / "cases"
    expected = [
        ("topk-embed-v1-small", "topk-embed-v1"),
        ("qwen3-embedding-0.6b", "qwen3-embedding"),
        ("qwen3-vl-embedding-2b", "qwen3-vl-embedding"),
        ("qwen3-vl-reranker-2b", "qwen3-vl-reranker"),
    ]
    for recipe_id, family in expected:
        assert f"{recipe_id} {family}" in script, (recipe_id, family)
        recipe = load_recipe(recipe_id)
        assert recipe._dir is not None and recipe._dir.name == family, (recipe_id, recipe._dir)
    generated = 0
    kind_none = 0
    media = 0
    for recipe_id, _family in expected:
        for path in (cases_root / recipe_id).glob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            if "source:\n  kind: generated" not in text:
                continue
            generated += 1
            kind_none += int("kind: none" in text)
            media += int("image:" in text or "video:" in text)
    assert (generated, kind_none, media) == (41, 1, 14)


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
        "worker.command=/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash "
        "/etc/rcp/files/e2e/e2e.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/e2e --wave e2e"
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


# --- rc_build.sh: the pairs staging (one home: rcp-ndcg-test/pairs/) --------------------------


def test_rc_build_stages_the_recipes_from_the_package_data_path() -> None:
    """The staged recipes come from the package-data path the layout move created, through the built
    wheel's extraction: the script stages the wheel's ``rcp_ndcg_vllm/recipes`` package data, and the
    source tree that wheel is built from exists in the checkout with its family directories.

    The regression: the script copied ``rcp-ndcg-vllm/recipes`` -- a path that has not existed since the
    layout move (the recipes are package data under ``src/rcp_ndcg_vllm/``), so under ``set -euo
    pipefail`` the whole RC build aborted before staging anything.
    """
    script = RC_BUILD.read_text(encoding="utf-8")
    assert "stage_recipes" in script and "python3 -m zipfile" in script, "the recipes come from the wheel"
    repo = Path(__file__).resolve().parents[2]
    source = repo / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"
    assert source.is_dir(), f"{source} does not exist in the checkout"
    assert (source / "qwen3-reranker" / "family.yaml").is_file()


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


# --- rc_build.sh: the published distributions, by name (GPU-E1) ------------------------------


def test_rc_build_builds_exactly_the_published_distributions(tmp_path: Path) -> None:
    """GPU-E1: ``uv build --all-packages`` picked up the unpublished workspace member rcp-ndcg-test and
    the build then refused ``dist/ holds other versions``.  The build step builds exactly the three
    published distributions, by name, one ``--package`` per invocation as release.yml does -- and the
    wheelhouse it stages lists no unpublished wheel."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "uv.log"
    (fake_bin / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "uv $*" >> {log}\n'
        'if [[ "$1" == "--version" ]]; then echo "uv 1.2.3"; exit 0; fi\n'
        'if [[ "$1" != "build" ]]; then exit 0; fi\n'
        'out=""; i=0; packages=()\n'
        'for arg in "$@"; do\n'
        '  if [[ "$prev" == "--out-dir" ]]; then out="$arg"; fi\n'
        '  if [[ "$prev" == "--package" ]]; then packages+=("$arg"); fi\n'
        '  prev="$arg"\n'
        "done\n"
        'mkdir -p "${out:-none}"\n'
        'for package in "${packages[@]}"; do touch "$out/${package//-/_}-0.0.1-py3-none-any.whl"; done\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "uv").chmod(0o755)
    dist = tmp_path / "dist"
    completed = subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && build_published "$1"', "bash", str(dist)],
        capture_output=True,
        text=True,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
    builds = [shlex.split(line) for line in log.read_text(encoding="utf-8").splitlines()]
    packages = [[words[words.index("--package") + 1] for words in builds if "--package" in words]][0]
    assert packages == ["rcp-ndcg-core", "rcp-ndcg", "rcp-ndcg-vllm"], builds  # release order, by name
    assert all("--all-packages" not in words for words in builds)
    # The wheelhouse the build stages: exactly the published wheels, never the unpublished member's.
    wheelhouse = tmp_path / "wheelhouse"
    shutil.copytree(dist, wheelhouse)
    listed = sorted(path.name for path in wheelhouse.iterdir())
    assert listed == [
        "rcp_ndcg-0.0.1-py3-none-any.whl",
        "rcp_ndcg_core-0.0.1-py3-none-any.whl",
        "rcp_ndcg_vllm-0.0.1-py3-none-any.whl",
    ], listed
    assert not any(name.startswith("rcp_ndcg_test") for name in listed)
    assert "uv build --all-packages" not in RC_BUILD.read_text(encoding="utf-8")  # the trap is gone from the script
    # A2/A3: the unpublished harness distribution is built too, by name, into ITS OWN directory -- never
    # dist/ (whose six published files the check above pins) -- and the node's client environment installs
    # it from <stage>/harness/.
    harness = tmp_path / "harness"
    completed = subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && build_harness "$1"', "bash", str(harness)],
        capture_output=True,
        text=True,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
    builds = [shlex.split(line) for line in log.read_text(encoding="utf-8").splitlines()]
    packages = [[words[words.index("--package") + 1] for words in builds if "--package" in words]][0]
    assert packages == ["rcp-ndcg-core", "rcp-ndcg", "rcp-ndcg-vllm", "rcp-ndcg-test"], builds
    assert sorted(path.name for path in harness.iterdir()) == ["rcp_ndcg_test-0.0.1-py3-none-any.whl"]
    # The staged tree the real build copies it into: the script names the harness directory beside dist/.
    assert 'stage/"$RC_NAME"/harness' in RC_BUILD.read_text(encoding="utf-8")


def test_rc_build_stages_pairs_from_the_packages_home(tmp_path: Path) -> None:
    """The pairs files live at rcp-ndcg-test/pairs/ after the layout move: rc_build.sh stages exactly that
    directory (the node consumes <pairs-dir>/<recipe>.jsonl)."""
    checkout = tmp_path / "checkout"
    pairs = checkout / "rcp-ndcg-test" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "fixture-embed.jsonl").write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    completed = _stage_pairs(tmp_path, checkout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    staged = tmp_path / "stage" / "pairs" / "fixture-embed.jsonl"
    assert staged.is_file(), f"the pairs file was not staged: {sorted((tmp_path / 'stage').rglob('*'))}"


def test_rc_build_refuses_a_stray_pre_layout_pairs_home(tmp_path: Path) -> None:
    """One home: a stray <checkout-root>/pairs/ or a pre-layout <checkout>/rcp-ndcg-vllm/pairs/ is
    refused (never silently staged), said on stderr with the one home named."""
    for stray in ("pairs", "rcp-ndcg-vllm/pairs"):
        checkout = tmp_path / "checkout"
        (checkout / stray).mkdir(parents=True)
        (checkout / stray / "stray.jsonl").write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
        completed = _stage_pairs(tmp_path, checkout)
        assert completed.returncode != 0, stray
        assert "rcp-ndcg-test/pairs" in completed.stderr, (stray, completed.stderr)
        assert not (tmp_path / "stage" / "pairs").exists()


def test_rc_build_without_any_pairs_stages_nothing(tmp_path: Path) -> None:
    """A checkout without pairs stages no pairs dir (the wave runner then reports its missing pairs)."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    completed = _stage_pairs(tmp_path, checkout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "stage" / "pairs").exists()


# --- rc_build.sh: the cases staging (one home: rcp-ndcg-test/cases/) --------------------------


def _stage_cases(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    """Source rc_build.sh and run its ``stage_cases`` over a scratch checkout and stage."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && stage_cases "{checkout}" "{stage}"', "bash"],
        capture_output=True,
        text=True,
    )


def test_rc_build_stages_the_cases_from_the_packages_home(tmp_path: Path) -> None:
    """The reference cases live at rcp-ndcg-test/cases/ (the cases lanes' home): rc_build.sh stages
    exactly that directory as <stage>/cases/ -- what the casefill job reads and edits on the node
    (the wheel drops cases/, so the stage is the node's only copy)."""
    checkout = tmp_path / "checkout"
    cases = checkout / "rcp-ndcg-test" / "cases" / "fixture-embed"
    cases.mkdir(parents=True)
    (cases / "gen-short.yaml").write_text("id: fixture-embed/gen-short\n", encoding="utf-8")
    completed = _stage_cases(tmp_path, checkout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "stage" / "cases" / "fixture-embed" / "gen-short.yaml").is_file()


def test_rc_build_refuses_a_stray_root_cases_directory(tmp_path: Path) -> None:
    """One home: a stray <checkout-root>/cases/ is refused (never silently staged), said on stderr
    with the one home named."""
    checkout = tmp_path / "checkout"
    (checkout / "cases" / "fixture-embed").mkdir(parents=True)
    (checkout / "cases" / "fixture-embed" / "gen-short.yaml").write_text(
        "id: fixture-embed/gen-short\n", encoding="utf-8"
    )
    completed = _stage_cases(tmp_path, checkout)
    assert completed.returncode != 0
    assert "rcp-ndcg-test/cases" in completed.stderr
    assert not (tmp_path / "stage" / "cases").exists()


def test_rc_build_without_any_cases_stages_nothing(tmp_path: Path) -> None:
    """A checkout without the cases stages none, said on stderr (the casefill job then reports its
    missing cases root); it is a warning, not a build failure."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    completed = _stage_cases(tmp_path, checkout)
    assert completed.returncode == 0
    assert "staging no cases" in completed.stderr
    assert not (tmp_path / "stage" / "cases").exists()


# --- rc_build.sh: the recipes come from the built wheel's package data ------------------------


def _fake_wheel(checkout: Path, *, with_recipes: bool = True) -> Path:
    """A minimal rcp_ndcg_vllm wheel in <checkout>/dist, with (or without) the recipes package data."""
    import zipfile

    dist = checkout / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    wheel = dist / "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("rcp_ndcg_vllm/__init__.py", "")
        if with_recipes:
            archive.writestr("rcp_ndcg_vllm/recipes/fixture-embed/recipe.yaml", "id: fixture-embed\n")
            archive.writestr("rcp_ndcg_vllm/recipes/README.md", "# recipes\n")
    return wheel


def _stage_recipes(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    """Source rc_build.sh and run its ``stage_recipes`` over a scratch checkout and stage."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && stage_recipes "$1" "$2"', "bash", str(checkout), str(stage)],
        capture_output=True,
        text=True,
    )


def test_rc_build_stages_the_recipes_from_the_built_wheel(tmp_path: Path) -> None:
    """Layout-move item 3: the recipes are the rcp-ndcg-vllm wheel's package data.  rc_build.sh stages the
    whole ``rcp_ndcg_vllm/recipes/`` directory out of the BUILT wheel -- never a per-recipe file list and
    never the pre-move ``rcp-ndcg-vllm/recipes`` source path -- so the node runs exactly the shipped
    recipes (a recipe missing from the wheel fails the build, not the wave)."""
    checkout = tmp_path / "checkout"
    _fake_wheel(checkout)
    completed = _stage_recipes(tmp_path, checkout)
    assert completed.returncode == 0, completed.stderr
    stage = tmp_path / "stage"
    assert (stage / "recipes" / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8") == "id: fixture-embed\n"
    assert (stage / "recipes" / "README.md").is_file()
    assert not (stage / ".wheel").exists()  # the extraction scratch is removed


def test_rc_build_fails_when_the_wheel_carries_no_recipes(tmp_path: Path) -> None:
    """A wheel whose package data is missing the recipes fails the build loudly (the node would otherwise
    run an older or empty recipe set); a checkout without a wheel fails loudly too."""
    checkout = tmp_path / "checkout"
    _fake_wheel(checkout, with_recipes=False)
    completed = _stage_recipes(tmp_path, checkout)
    assert completed.returncode != 0
    assert "recipes" in completed.stderr and "package data" in completed.stderr
    assert not (tmp_path / "stage" / "recipes").exists()
    empty = tmp_path / "empty-checkout"
    empty.mkdir()
    completed = _stage_recipes(tmp_path, empty)
    assert completed.returncode != 0
    assert "rcp_ndcg_vllm wheel" in completed.stderr


# --- rc_build.sh: the wave lists and the whole staged tree ------------------------------------


def _stage_wave_lists(tmp_path: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    """Source rc_build.sh and run its ``stage_wave_lists`` over a scratch checkout and stage."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && stage_wave_lists "$1" "$2"', "bash", str(checkout), str(stage)],
        capture_output=True,
        text=True,
    )


def test_rc_build_stages_wave_lists_from_the_tooling_home(tmp_path: Path) -> None:
    """The wave lists live at rcp-ndcg-test/wave-lists/ (owner decision, 2026-10-09): rc_build.sh stages
    exactly that directory as <stage>/wave-lists/ (what bootstrap.sh resolves a wave name against)."""
    checkout = tmp_path / "checkout"
    lists = checkout / "rcp-ndcg-test" / "wave-lists"
    lists.mkdir(parents=True)
    (lists / "all-retrieval.txt").write_text("fixture-embed\n", encoding="utf-8")
    completed = _stage_wave_lists(tmp_path, checkout)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "stage" / "wave-lists" / "all-retrieval.txt").read_text(encoding="utf-8") == "fixture-embed\n"


def test_rc_build_refuses_a_stray_root_wave_lists_directory(tmp_path: Path) -> None:
    """One home: a stray <checkout-root>/wave-lists/ is refused, naming rcp-ndcg-test/wave-lists."""
    checkout = tmp_path / "checkout"
    (checkout / "wave-lists").mkdir(parents=True)
    (checkout / "wave-lists" / "stray.txt").write_text("fixture-embed\n", encoding="utf-8")
    completed = _stage_wave_lists(tmp_path, checkout)
    assert completed.returncode != 0
    assert "rcp-ndcg-test/wave-lists" in completed.stderr
    assert not (tmp_path / "stage" / "wave-lists").exists()


def test_rc_build_without_wave_lists_stages_none(tmp_path: Path) -> None:
    """A checkout without the committed lists stages none, said on stderr (EXTRA_DIRS can carry private
    lists); it is a warning, not a build failure."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    completed = _stage_wave_lists(tmp_path, checkout)
    assert completed.returncode == 0
    assert "staging no wave lists" in completed.stderr


def test_rc_build_stages_the_recipe_tree_before_it_reads_the_family_locks() -> None:
    """The wheelhouse's reference wheels come from the families' ``reference.lock`` files under the staged
    recipes, so the recipe tree must be staged before the lock glob runs: with the old order the glob saw an
    empty stage, downloaded no reference wheels, and the node's reference install failed on the first lock
    pin (the rc0 wave0 bootstrap failure: transformers==4.57.6 was not in the staged wheelhouse)."""
    script = RC_BUILD.read_text(encoding="utf-8")
    stage_tree_at = script.index('stage_tree "$SRC" "stage/$RC_NAME"')
    lock_glob_at = script.index('for lock in stage/"$RC_NAME"/recipes/*/reference.lock')
    assert stage_tree_at < lock_glob_at


def test_rc_build_downloads_each_family_lock_separately() -> None:
    """The families pin different transformers versions (4.57.6 and 5.19.0), so one pip resolution over
    all locks is unsatisfiable: each lock is downloaded in its own resolution into the shared wheelhouse
    (the rc0 rebuild failed with ResolutionImpossible over the batch), the direct pins hash-verified and
    their closure unhashed (what reference_deps.py completes the family's venv from on the node)."""
    script = RC_BUILD.read_text(encoding="utf-8")
    assert script.count('pip download --quiet --no-deps -r "$WORK/direct.txt"') == 2  # the exact pins
    assert script.count('pip download --quiet "$pin"') == 2  # each pin's closure, in its own resolution
    assert script.count("hashed the local rcp-ndcg pin") == 1  # the staged lock's local pin gets a hash
    assert 'lock_args+=(-r "$lock")' not in script  # the batch resolution that failed


def test_rc_build_stages_the_real_checkout(tmp_path: Path) -> None:
    """The guard that cannot rot: ``stage_tree`` on the ACTUAL checkout stages the recipes (from the built
    wheel's package data), every pairs file and the committed wave lists -- a moved path fails here first.
    The wheel is packed from the real package data, so the test also proves that path exists."""
    import zipfile

    repo = Path(__file__).resolve().parents[2]
    recipes_src = repo / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"
    assert recipes_src.is_dir(), f"the recipe package data moved: {recipes_src}"
    wheel = tmp_path / "dist" / "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in sorted(recipes_src.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path("rcp_ndcg_vllm/recipes") / path.relative_to(recipes_src)))
    stage = tmp_path / "stage"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{RC_BUILD}" && stage_tree "$1" "$2" "$3"',
            "bash",
            str(repo),
            str(stage),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    staged_recipes = sorted(path.name for path in (stage / "recipes").iterdir() if path.is_dir())
    assert staged_recipes == sorted(path.name for path in recipes_src.iterdir() if path.is_dir())
    assert (stage / "recipes" / "README.md").is_file()
    pairs_src = repo / "rcp-ndcg-test" / "pairs"
    assert sorted(path.name for path in (stage / "pairs").iterdir()) == sorted(
        path.name for path in pairs_src.iterdir()
    )
    lists_src = repo / "rcp-ndcg-test" / "wave-lists"
    assert (stage / "wave-lists" / "all-retrieval.txt").read_bytes() == (lists_src / "all-retrieval.txt").read_bytes()
    cases_src = repo / "rcp-ndcg-test" / "cases"
    assert sorted(path.name for path in (stage / "cases").iterdir()) == sorted(
        path.name for path in cases_src.iterdir()
    )
    assert (stage / "cases" / "topk-embed-v1-small" / "text-short-single.yaml").is_file()


# --- rc_build.sh: the pins (decision 18: vllm names no sibling) --------------------------------


def _check_pins(
    tmp_path: Path, *, core_pin: str = '  "rcp-ndcg-core==0.0.1",\n', vllm_deps: str = '  "pydantic>=2.0",\n'
) -> subprocess.CompletedProcess[str]:
    """Source rc_build.sh and run ``check_pins`` on a scratch checkout with the given manifests."""
    checkout = tmp_path / "checkout"
    (checkout / "rcp-ndcg").mkdir(parents=True)
    (checkout / "rcp-ndcg-vllm").mkdir(parents=True)
    (checkout / "rcp-ndcg" / "pyproject.toml").write_text(
        '[project]\nname = "rcp-ndcg"\nversion = "0.0.1"\ndependencies = [\n' + core_pin + "]\n", encoding="utf-8"
    )
    (checkout / "rcp-ndcg-vllm" / "pyproject.toml").write_text(
        '[project]\nname = "rcp-ndcg-vllm"\nversion = "0.0.1"\ndependencies = [\n' + vllm_deps + "]\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", "-c", f'source "{RC_BUILD}" && check_pins "$1" 0.0.1', "bash", str(checkout)],
        capture_output=True,
        text=True,
    )


def test_rc_build_pin_check_accepts_the_real_manifests(tmp_path: Path) -> None:
    """rcp-ndcg pins rcp-ndcg-core==VERSION; the lean rcp-ndcg-vllm names NO sibling (decision 18: no
    lockstep version pin between the two packages -- the manifest does not depend on rcp-ndcg)."""
    completed = _check_pins(tmp_path)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("sibling", ["rcp-ndcg==0.0.1", "rcp-ndcg-core==0.0.1", "rcp-ndcg-test==0.0.1"])
def test_rc_build_pin_check_refuses_a_sibling_in_the_vllm_manifest(tmp_path: Path, sibling: str) -> None:
    """The pre-fix check asserted the opposite and aborted every build: ANY workspace sibling named in
    the vllm manifest is refused (not only rcp-ndcg), naming decision 18."""
    completed = _check_pins(tmp_path, vllm_deps=f'  "{sibling}",\n')
    assert completed.returncode != 0
    assert "no sibling" in completed.stderr
    assert sibling in completed.stderr


def test_rc_build_pin_check_accepts_the_package_naming_itself(tmp_path: Path) -> None:
    """The manifest's own distribution (rcp-ndcg-vllm) is not a sibling: an entry naming it is not the
    workspace-sibling refusal (nothing sane names itself, but the check must not confuse the names)."""
    completed = _check_pins(tmp_path, vllm_deps='  "rcp-ndcg-vllm==0.0.1",\n')
    assert completed.returncode == 0, completed.stderr


def test_rc_build_pin_check_refuses_a_missing_core_pin(tmp_path: Path) -> None:
    completed = _check_pins(tmp_path, core_pin="")
    assert completed.returncode != 0
    assert "rcp-ndcg-core==0.0.1" in completed.stderr


# --- judgecheck.sh: the release checklist's judge item (both thinking modes, the media probe) --------


#: The ten shipped judges, in the script's own order: the six the release checklist names first (a
#: time-boxed run covers them before the four Gemma 4 judges).
JUDGE_PRIORITY = (
    "qwen3.5-397b-a17b-nvfp4",
    "gpt-oss-120b",
    "qwen3.6-27b-fp8",
    "qwen3.8-27b-fp8",
    "qwen3.8-flash-next-fp8",
    "qwen3.8-flash-next-nvfp4",
)
JUDGE_EXTRA = ("gemma-4-12b-it", "gemma-4-26b-a4b-it", "gemma-4-26b-a4b-nvfp4", "gemma-4-31b-it-nvfp4")
JUDGES = JUDGE_PRIORITY + JUDGE_EXTRA


def _judgecheck_recipes() -> list[str]:
    """The recipe ids the script's RECIPES array names, in order."""
    import re

    text = JUDGECHECK_SH.read_text(encoding="utf-8")
    block = text.split("RECIPES=(", 1)[1].split(")", 1)[0]
    return re.findall(r'^\s*"([^"]+)"\s*$', block, flags=re.MULTILINE)


def test_judgecheck_pins_the_ten_judges_to_the_catalog_and_the_checklists_six_first() -> None:
    """The script's ten ids are exactly the catalog's judge variants, the checklist's six first: a renamed
    recipe or a new judge fails here first (the casefill recipe/family pin's shape)."""
    from rcp_ndcg_vllm.recipe import load_recipe

    recipes = _judgecheck_recipes()
    assert recipes == list(JUDGES), recipes
    for recipe_id in recipes:
        recipe = load_recipe(recipe_id)
        assert recipe.role == "judge", recipe_id
        assert recipe.client.get("tokenizer"), recipe_id  # a served judge declares its tokenizer
    import re

    from rcp_ndcg_vllm.recipe import default_recipes_root

    # The shipped package data, never a checkout path: a dirty worktree's stale directories are not recipes.
    judge_ids = set()
    for family in sorted(default_recipes_root().iterdir()):
        yaml_path = family / "family.yaml"
        if not yaml_path.is_file():
            continue
        text = yaml_path.read_text(encoding="utf-8")
        if "role: judge" not in text:
            continue
        judge_ids.update(re.findall(r"^\s+- id: (\S+)\s*$", text, flags=re.MULTILINE))
    assert set(recipes) == judge_ids, sorted(judge_ids)


def test_judgecheck_runs_each_familys_documented_thinking_modes() -> None:
    """Only the Gemma 4 recipe documents a thinking-mode toggle: its judges run both modes (off, the shipped
    default, and on, the declared `chat_template_kwargs` choice), and the `on` body carries the recipe's own
    `extra_body` merged in (`--set judge.extra_body` replaces the whole block).  Every other recipe
    documents none: its single `default` mode passes no toggle at all, never a guessed flag."""
    from rcp_ndcg_test.jobs.judgecheck_probe import judge_modes
    from rcp_ndcg_vllm.recipe import load_recipe

    for recipe_id in JUDGES:
        recipe = load_recipe(recipe_id)
        shipped = recipe.client.get("extra_body") or {}
        modes = judge_modes(recipe_id)
        if recipe.client.get("image_processor") == "gemma4":
            assert [row["name"] for row in modes] == ["off", "on"], modes
            assert modes[0]["extra_body"] == {}, modes  # the shipped block is the documented default
            assert modes[1]["extra_body"] == {**shipped, "chat_template_kwargs": {"enable_thinking": True}}, modes
        else:
            assert [row["name"] for row in modes] == ["default"], modes
            assert modes[0]["extra_body"] == {}, modes  # no toggle: the recipe's own block stands


def test_judgecheck_vision_toggle_is_the_gemma_recipe_s_documented_choice() -> None:
    """The Gemma 4 recipe's own note documents the thinking-on toggle verbatim (`--set judge.extra_body`
    with `chat_template_kwargs.enable_thinking`); the helper's vision-family switch is that choice, so a
    changed recipe note fails here before the node runs a different instrument."""
    from rcp_ndcg_vllm.recipe import default_recipes_root

    notes = (default_recipes_root() / "gemma-4-12b" / "family.yaml").read_text(encoding="utf-8")
    assert '"chat_template_kwargs": {"enable_thinking": true}' in notes
    assert "the parser reads the template kwarg, not reasoning_effort" in notes


def test_judgecheck_media_probe_uses_the_engines_own_stock_budget() -> None:
    """The media probe's declared policy is the processor family's own stock budget (the engine keeps the
    prepared size, so the engine's delta and the client's count describe the same bytes); a text-only
    config declares none."""
    from rcp_ndcg_test.jobs.judgecheck_probe import probe_policy

    from rcp_ndcg.data.resolution import PROCESSORS

    qwen = probe_policy("qwen3_vl")
    assert (qwen.min_px, qwen.max_px) == (PROCESSORS["qwen3_vl"].min_pixels, PROCESSORS["qwen3_vl"].max_pixels)
    gemma = probe_policy("gemma4")
    assert gemma.max_soft_tokens == PROCESSORS["gemma4"].soft_tokens
    assert probe_policy(None) is None


def test_judgecheck_media_report_is_the_products_verdict() -> None:
    """The media report runs the product's own probe (`RoleClient.check_engine_media`) under the resolved
    policy and reports its outcome; a text-only judge is recorded `checked: false`, never silently passed."""

    from rcp_ndcg_test.jobs.judgecheck_probe import media_report

    from rcp_ndcg.data.prepare import MediaCensus
    from rcp_ndcg.judging.client import JudgeConfig

    class StubClient:
        """The product's client surface the report reads: the policy it was given, the census, the probe."""

        def __init__(self) -> None:
            self.image_policy = None
            self.media_census = MediaCensus()

        async def check_engine_media(self) -> None:
            assert self.image_policy is not None, "the report must hand the probe its policy"
            self.media_census.record(corpus="judge", doc_id="engine_media_check:ok", media=[], dropped=False)

    vision = JudgeConfig.model_validate({"model": "qwen3.5-397b-a17b-nvfp4", "image_processor": "qwen3_vl"})
    report = media_report(vision, client=StubClient())
    assert report["checked"] is True and report["ok"] is True and report["outcome"] == "ok", report
    assert report["processor"] == "qwen3_vl" and report["policy"]["min_px"] == 65536, report

    text_only = JudgeConfig.model_validate({"model": "gpt-oss-120b"})
    report = media_report(text_only, client=StubClient())
    assert report["checked"] is False and report["reason"] == "text-only", report


def test_judgecheck_media_report_checks_an_engine_over_the_wire() -> None:
    """The media report against a stub chat engine: an engine whose media delta equals the client's counted
    tokens passes and both numbers are recorded; an engine whose delta differs is the product's typed
    mismatch (never a silent pass).  The probe's prepared 224x224 image resizes to the qwen3_vl stock floor
    (256x256 = 8x8 patches + the two vision markers = 66 media tokens)."""
    import http.server
    import threading

    from rcp_ndcg_test.jobs.judgecheck_probe import media_report

    from rcp_ndcg.judging.client import JudgeConfig

    state = {"delta": 66}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            self._send({"object": "list", "data": [{"id": "qwen3.6-27b-fp8"}]})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            payload = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
            carries = any(
                part.get("type") == "image_url"
                for message in payload.get("messages", [])
                for part in (message.get("content") if isinstance(message.get("content"), list) else [])
                if isinstance(part, dict)
            )
            usage = {"prompt_tokens": 137 + (state["delta"] if carries else 0), "completion_tokens": 4}
            answer = {
                "id": "stub",
                "model": payload.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
                "usage": usage,
            }
            self._send(answer)

        def _send(self, document: dict) -> None:
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = JudgeConfig.model_validate(
            {
                "model": "qwen3.6-27b-fp8",
                "image_processor": "qwen3_vl",
                "max_images": 10,
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            }
        )
        honest = media_report(config)
        assert honest["outcome"] == "ok" and honest["ok"] is True, honest
        assert (honest["counted"], honest["reported"]) == (66, 66), honest
        state["delta"] = 67
        lying = media_report(config)
        assert lying["outcome"] == "mismatch" and lying["ok"] is False, lying
        assert "the client counted 66" in lying["detail"], lying
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_submit_judgecheck_mounts_the_judgecheck_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge-check job: judgecheck.sh mounted beside bootstrap.sh (whose envs mode builds the client
    environment) and its probe helper beside it, run with the stage and the wave's output prefix; no token
    value in the plan."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "judgecheck",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "judgecheck",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    command = next(word for word in words if word.startswith("worker.command="))
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash "
        "/etc/rcp/files/judgecheck/judgecheck.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/judgecheck"
    )
    assert any(word.startswith("files.judgecheck.from_file=") and word.endswith("judgecheck.sh") for word in words)
    assert any(
        word.startswith("files.judgecheckprobe.from_file=") and word.endswith("judgecheck_probe.py") for word in words
    )
    assert any(word.startswith("files.bootstrap.from_file=") for word in words)
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"


def test_submit_topk_probe_mounts_the_probe_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The topk probe job: topk_probe.sh mounted beside its module and the wave-cosines fixture, with
    bootstrap.sh (whose envs mode builds the client environment and the family's reference venv), run with
    the stage and the probe's own output prefix; no token value in the plan and no knob line when the
    operator set none (the wrapper's own defaults apply)."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "topk-probe",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "topk-probe",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    command = next(word for word in words if word.startswith("worker.command="))
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash "
        "/etc/rcp/files/topkprobe/topk_probe.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/topk-probe"
    )
    assert any(word.startswith("files.topkprobe.from_file=") and word.endswith("topk_probe.sh") for word in words)
    assert any(word.startswith("files.topkprobepy.from_file=") and word.endswith("topk_probe.py") for word in words)
    assert any(
        word.startswith("files.topkprobecosines.from_file=") and word.endswith("wave-cosines-topk-small.json")
        for word in words
    )
    assert any(word.startswith("files.bootstrap.from_file=") for word in words)
    assert not any(word.startswith("env.RCP_TOPK_") for word in words)
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"


def test_submit_topk_probe_forwards_the_operators_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``RCP_TOPK_*`` knob the wrapper documents reaches the pod as ``env.RCP_TOPK_*``; the mounted
    helper paths are the node's own and are never forwarded from the operator's environment."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "topk-probe",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "topk-probe",
        env_overrides={
            "RCP_TOPK_VARIANT": "small",
            "RCP_TOPK_ROWS": "26",
            "RCP_TOPK_DEVICE": "cuda:1",
            "RCP_TOPK_REFERENCE_DEVICE": "cuda:2",
            "RCP_TOPK_GPU_FRACTION": "0.3",
            "RCP_TOPK_DTYPE": "float32",
            "RCP_TOPK_REFERENCE_DTYPE": "float32",
            "RCP_TOPK_CONTROL_VECTORS": "/etc/rcp/files/topkprobe/control-row26.json",
            "RCP_TOPK_MODEL": "topk-io/topk-embed-v1-small",
            "RCP_TOPK_REVISION": "0" * 40,
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    for knob, value in (
        ("RCP_TOPK_VARIANT", "small"),
        ("RCP_TOPK_ROWS", "26"),
        ("RCP_TOPK_DEVICE", "cuda:1"),
        ("RCP_TOPK_REFERENCE_DEVICE", "cuda:2"),
        ("RCP_TOPK_GPU_FRACTION", "0.3"),
        ("RCP_TOPK_DTYPE", "float32"),
        ("RCP_TOPK_REFERENCE_DTYPE", "float32"),
        ("RCP_TOPK_CONTROL_VECTORS", "/etc/rcp/files/topkprobe/control-row26.json"),
        ("RCP_TOPK_MODEL", "topk-io/topk-embed-v1-small"),
        ("RCP_TOPK_REVISION", "0" * 40),
    ):
        assert f"env.{knob}={value}" in words, (knob, words)
    # the mounted helpers' own node paths are the wrapper's defaults, never the operator's environment
    assert not any(word.startswith("env.RCP_TOPK_PROBE_PY") for word in words)
    assert not any(word.startswith("env.RCP_TOPK_WAVE_COSINES") for word in words)


def test_the_topk_probe_dry_plan_names_the_probe_and_the_upload_prefix() -> None:
    """``TOPK_DRY=1`` prints what the node would run (the operator's plan) and runs nothing: the
    environments step, the engine-API check, the probe argv with the mounted module and the wave-cosines
    fixture, the selected rows, and the upload prefix."""
    completed = subprocess.run(
        ["bash", str(TOPK_PROBE_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/topk-probe"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "TOPK_DRY": "1",
            "RCP_GCS_AUTH_FILE": "/nonexistent",
            "RCP_TOPK_ROWS": "26",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = completed.stdout
    assert "bootstrap.sh envs" in plan
    assert "/etc/rcp/files/topkprobe/topk_probe.py --check-engine" in plan
    assert "/etc/rcp/files/topkprobe/topk_probe.py" in plan
    assert "/etc/rcp/files/topkprobe/wave-cosines-topk-small.json" in plan
    assert "--dtype bfloat16 --reference-dtype bfloat16" in plan
    assert "--rows 26" in plan
    assert "gs://YOUR-BUCKET/waves/topk-probe/" in plan


def test_the_topk_probe_wrapper_forwards_the_dtypes(tmp_path: Path) -> None:
    """The dtype knobs reach the probe's argv: the engine's LLM dtype and the checkpoint's own load
    dtype on the reference side; unset means bfloat16 on both sides (the family's shipped dtype)."""
    stage = tmp_path / "stage"
    (stage / "recipes" / "topk-embed-v1").mkdir(parents=True)
    (stage / "recipes" / "topk-embed-v1" / "family.yaml").write_text("id: topk-embed-v1\n", encoding="utf-8")
    (stage / "pairs").mkdir()
    (stage / "pairs" / "topk-embed-v1-small.jsonl").write_text("", encoding="utf-8")
    calls = tmp_path / "calls.log"
    probe_py = tmp_path / "topk_probe.py"
    probe_py.write_text(
        "import pathlib, sys\n"
        f"open({str(calls)!r}, 'a', encoding='utf-8').write(' '.join(sys.argv[1:]) + chr(10))\n"
        "if '--check-engine' not in sys.argv:\n"
        "    out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'probe-report.md').write_text('stub', encoding='utf-8')\n",
        encoding="utf-8",
    )
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'state=""\n'
        'while (($#)); do [[ "$1" == "--state" ]] && { state="$2"; shift 2; continue; }; shift; done\n'
        'mkdir -p "$state/reference/topk-embed-v1/bin"\n'
        "printf '#!/bin/sh\\n' > \"$state/reference/topk-embed-v1/bin/python\"\n"
        'chmod +x "$state/reference/topk-embed-v1/bin/python"\n',
        encoding="utf-8",
    )
    gcs = tmp_path / "gcs.sh"
    gcs.write_text("upload_tree() { :; }\n", encoding="utf-8")
    # A gcloud that succeeds shadows the real one: the transfer is not this test's subject.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud = bin_dir / "gcloud"
    gcloud.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    gcloud.chmod(0o755)
    base_env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "RCP_GCS_AUTH_FILE": "/nonexistent",
        "RCP_BOOTSTRAP_SH": str(bootstrap),
        "RCP_TOPK_PROBE_PY": str(probe_py),
        "RCP_TOPK_WAVE_COSINES": str(TOPK_PROBE_COSINES),
        "RCP_TOPK_MODEL": "topk-io/topk-embed-v1-small",
        "RCP_TOPK_REVISION": "0" * 40,
        "RCP_GCS_HELPER_SH": str(gcs),
    }
    for extra, expected in (
        ({}, "--dtype bfloat16 --reference-dtype bfloat16"),
        (
            {"RCP_TOPK_DTYPE": "float32", "RCP_TOPK_REFERENCE_DTYPE": "float32"},
            "--dtype float32 --reference-dtype float32",
        ),
    ):
        calls.unlink(missing_ok=True)
        completed = subprocess.run(
            ["bash", str(TOPK_PROBE_SH), str(stage), str(tmp_path / "out")],
            capture_output=True,
            text=True,
            env=base_env | extra,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        lines = calls.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "--check-engine"
        assert expected in lines[1], lines[1]
        # the wrapper builds the probe's argv by hand: the module's own parser must accept every flag
        parsed = _probe_module().build_parser().parse_args(shlex.split(lines[1]))
        assert (parsed.dtype, parsed.reference_dtype) == (
            extra.get("RCP_TOPK_DTYPE", "bfloat16"),
            extra.get("RCP_TOPK_REFERENCE_DTYPE", "bfloat16"),
        )


# --- topk_probe.py: the engine-side accessor and the image's API contract ---------------------------------


def _probe_module():
    """The probe module (stdlib-only at import time: the vLLM and torch imports are inside functions)."""
    from rcp_ndcg_test.jobs import topk_probe

    return topk_probe


class _DecoratedLikeVllm(torch.nn.Module):
    """A module shaped like vLLM's ``@support_torch_compile`` output: ``__call__`` calls ``forward``."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.forward(*args, **kwargs)


def test_the_probe_forces_the_single_process_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model must live in this process for the forward hooks and the tower calls to reach it, and
    vLLM reads the switch when it builds the engine: ``force_single_process_engine`` sets it."""
    probe = _probe_module()
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    probe.force_single_process_engine()
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_the_probe_walks_the_engines_documented_model_paths() -> None:
    """v0.31.0's in-process route (``llm_engine.model_executor.driver_worker.model_runner.model``, the
    path ``LLMEngine._get_driver_model_for_cleanup`` itself reads) and the worker-wrapped shape both
    resolve; the probe returns the model object itself."""
    probe = _probe_module()
    model = object()
    driver_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
    llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(model_executor=types.SimpleNamespace(driver_worker=driver_worker))
    )
    assert probe.engine_model(llm) is model
    wrapped = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            model_executor=types.SimpleNamespace(
                driver_worker=types.SimpleNamespace(
                    worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
                )
            )
        )
    )
    assert probe.engine_model(wrapped) is model


def test_the_probe_names_every_model_path_it_tried_when_the_api_moves() -> None:
    """A multiprocess engine (no ``model_executor``) fails loudly, with every path tried and the switch
    that puts the EngineCore in this process -- never an ``AttributeError`` from a guess."""
    probe = _probe_module()
    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace())
    with pytest.raises(RuntimeError) as raised:
        probe.engine_model(llm)
    message = str(raised.value)
    for path in probe.ENGINE_MODEL_PATHS:
        assert ".".join(path) in message
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in message


def test_the_probe_engine_check_names_the_api_the_probe_depends_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check-engine`` is the in-image contract check: it passes on a vLLM whose switch is honoured,
    whose ``LLMEngine`` still assigns ``model_executor`` and whose ``LLM`` carries the offline pooling
    entry points -- and fails, naming the fact, on one that moved (the failure the first paid run hit)."""
    probe = _probe_module()

    class _Engine:
        def __init__(self) -> None:
            self.engine_core = types.SimpleNamespace(engine_core=types.SimpleNamespace(model_executor=object()))
            self.model_executor = self.engine_core.engine_core.model_executor

    class _LLM:
        def encode(self) -> None: ...

        def apply_model(self) -> None: ...

    vllm = types.SimpleNamespace(
        __version__="0.31.0",
        envs=types.SimpleNamespace(VLLM_ENABLE_V1_MULTIPROCESSING=False),
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.llm_engine", types.SimpleNamespace(LLMEngine=_Engine))
    vllm.LLM = _LLM
    assert probe.check_engine() == 0
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in capsys.readouterr().out

    # The engine in a separate process: no model_executor assignment in the class the image ships.
    class _MultiprocessEngine:
        def __init__(self) -> None:
            self.engine_core = types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "vllm.v1.engine.llm_engine", types.SimpleNamespace(LLMEngine=_MultiprocessEngine))
    assert probe.check_engine() == 1
    assert "no longer assigns model_executor" in capsys.readouterr().err


def test_the_topk_probe_wrapper_checks_the_engine_api_before_loading_weights(tmp_path: Path) -> None:
    """The wrapper runs ``--check-engine`` before the probe and stops there when it fails: a moved engine
    API costs seconds, never a weights load (the first paid run's failure class)."""
    stage = tmp_path / "stage"
    (stage / "recipes" / "topk-embed-v1").mkdir(parents=True)
    (stage / "recipes" / "topk-embed-v1" / "family.yaml").write_text("id: topk-embed-v1\n", encoding="utf-8")
    (stage / "pairs").mkdir()
    (stage / "pairs" / "topk-embed-v1-small.jsonl").write_text("", encoding="utf-8")
    calls = tmp_path / "calls.log"
    probe_py = tmp_path / "topk_probe.py"
    probe_py.write_text(
        "import sys\n"
        f"open({str(calls)!r}, 'a', encoding='utf-8').write(' '.join(sys.argv[1:]) + chr(10))\n"
        "sys.exit(1 if '--check-engine' in sys.argv else 0)\n",
        encoding="utf-8",
    )
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'state=""\n'
        'while (($#)); do [[ "$1" == "--state" ]] && { state="$2"; shift 2; continue; }; shift; done\n'
        'mkdir -p "$state/reference/topk-embed-v1/bin"\n'
        "printf '#!/bin/sh\\n' > \"$state/reference/topk-embed-v1/bin/python\"\n"
        'chmod +x "$state/reference/topk-embed-v1/bin/python"\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "bash",
            str(TOPK_PROBE_SH),
            str(stage),
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RCP_GCS_AUTH_FILE": "/nonexistent",
            "RCP_BOOTSTRAP_SH": str(bootstrap),
            "RCP_TOPK_PROBE_PY": str(probe_py),
            "RCP_TOPK_WAVE_COSINES": str(TOPK_PROBE_COSINES),
            "RCP_TOPK_MODEL": "topk-io/topk-embed-v1-small",
            "RCP_TOPK_REVISION": "0" * 40,
        },
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["--check-engine"]
    assert "not loading the weights" in completed.stderr


def test_the_probe_records_a_forward_that_a_decorated_call_bypasses() -> None:
    """vLLM's ``@support_torch_compile`` replaces a decorated class's ``__call__`` with one that calls
    ``self.forward(...)`` directly (``Qwen3_VisionPatchEmbed`` and ``Qwen3_VisionBlock`` at v0.31.0), so
    ``register_forward_hook`` sees nothing while the probe's recorder -- which wraps the bound
    ``forward`` -- sees the call.  The first node run's E2/E3 tables held only the undecorated
    ``merger``, which read as "the merger is the first divergent layer"."""
    probe = _probe_module()
    module = _DecoratedLikeVllm()
    decorated_hook: dict[str, object] = {}
    inner_hook: dict[str, object] = {}
    handles = [
        module.register_forward_hook(lambda m, i, o: decorated_hook.setdefault("decorated", o)),
        module.inner.register_forward_hook(lambda m, i, o: inner_hook.setdefault("inner", o)),
    ]
    try:
        output, sink = probe._run_recorded(functools.partial(module, torch.ones(1, 2)), module, {"inner": "inner"})
    finally:
        for handle in handles:
            handle.remove()
    assert decorated_hook == {}, "nn.Module's forward hooks never see a decorated module's own call"
    assert "inner" in inner_hook, "an undecorated child's hook fires (the merger did in the first run)"
    assert list(sink) == ["inner"]
    assert torch.equal(sink["inner"], output)
    # the wrap is restored, so a second call is recorded again and the module still works
    _, again = probe._run_recorded(functools.partial(module, torch.ones(1, 2)), module, {"inner": "inner"})
    assert list(again) == ["inner"]
    assert module.forward.__self__ is module, "the module's own bound forward is back"


def test_the_probe_compares_tensors_that_differ_only_by_a_singleton_dimension() -> None:
    """vLLM's vision tower keeps a leading 1 on every block's output where the checkpoint's does not;
    the comparison squeezes it and says so -- refusing to compare reads as "clean" (the trap the
    missing tensors were).  A genuinely different shape still gets no elementwise comparison."""
    probe = _probe_module()
    squeezed = probe.stats(torch.ones(2, 1, 3), torch.ones(2, 3))
    assert squeezed["max_abs_diff"] == 0.0 and squeezed["cosine"] == pytest.approx(1.0)
    assert "singleton dimensions squeezed" in squeezed["note"]
    assert squeezed["equal"] is False, "the raw tensors are not equal: only their data is"
    differing = probe.stats(torch.ones(2, 3), torch.ones(2, 4))
    assert differing.get("max_abs_diff") is None
    assert differing["note"] == "shapes differ: no elementwise comparison"


def test_the_probe_names_the_tensors_a_recording_missed() -> None:
    """A recording that missed modules says so (an absent tensor reads as "clean" in the tables)."""
    probe = _probe_module()
    names = {"patch_embed": "patch_embed", "merger": "merger", "blocks.0": "block_0"}
    assert probe._missing_recorded(names, {"merger": object()}) == ["block_0", "patch_embed"]
    assert probe._missing_recorded(names, {key: object() for key in names.values()}) == []


def test_the_judgecheck_dry_plan_names_the_ten_judges_and_the_documented_modes() -> None:
    """``JUDGECHECK_DRY=1`` prints what the node would run (the operator's plan), and runs nothing: the ten
    judges in order, each served on its own port, both thinking modes' checks with the recipe's client block
    and the family's toggle, the media probe, and the upload prefix."""
    completed = subprocess.run(
        ["bash", str(JUDGECHECK_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/judgecheck"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "JUDGECHECK_DRY": "1", "RCP_GCS_AUTH_FILE": "/nonexistent"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = completed.stdout
    for index, recipe in enumerate(JUDGES):
        port = 8100 + index
        assert f"rcp-ndcg-vllm serve {recipe} --port {port}" in plan, plan
        assert f"--judge {recipe} --judge-model {recipe}" in plan, plan
        assert f"--set judge.base_url=http://127.0.0.1:{port}/v1" in plan, plan
        assert f"modes --judge {recipe}" in plan, plan
        assert f"media --judge {recipe}" in plan, plan
    assert plan.count("rcp-ndcg judge check") == 10, plan  # one command shape per judge, per documented mode
    assert "the Gemma 4 judges' two (off, on), the others'" in plan, plan
    assert "bootstrap.sh envs" in plan and "judgecheck-wave.txt" in plan
    assert "gs://YOUR-BUCKET/waves/judgecheck/" in plan
    assert "continuing past a failure" in plan


def test_judgecheck_end_to_end_serves_checks_media_and_uploads(tmp_path: Path) -> None:
    """Drive judgecheck.sh's node path with stubs: bootstrap envs builds the client wrapper, every judge is
    served and waited for on its own port, both thinking modes' checks run with the recipe's client block and
    the family's toggle, the media probe runs for every judge, one judge whose engine dies does not stop the
    next, every engine's process group is stopped, the JSONs and the report upload, and the job exits 1 with
    the failure recorded."""
    auth = tmp_path / "gcs_auth.sh"
    auth.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    gcs_log = tmp_path / "gcs.log"
    gcs_sh = tmp_path / "gcs.sh"
    gcs_sh.write_text(
        'gcs_sdk_on_path() { :; }\ngcs_cp() { printf "%s\\n" "cp $*" >> "$STUB_GCS_LOG"; }\n', encoding="utf-8"
    )
    helper = tmp_path / "gcs.py"
    helper.write_text("", encoding="utf-8")
    # The stub engine: a /v1/models server (the job's readiness wait reads it).
    stub_engine = tmp_path / "stub_engine.py"
    stub_engine.write_text(
        "import http.server, json, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        body = json.dumps({'object': 'list', 'data': [{'id': sys.argv[2]}]}).encode()\n"
        "        self.send_response(200)\n"
        "        self.send_header('content-type', 'application/json')\n"
        "        self.send_header('content-length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n",
        encoding="utf-8",
    )
    bootstrap_log = tmp_path / "bootstrap.log"
    client_log = tmp_path / "client.log"
    serve_pids = tmp_path / "serve-pids"
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_BOOTSTRAP_LOG"\n'
        'state=""; wave_list=""\n'
        "while (($#)); do\n"
        '  case "$1" in --state) state="$2"; shift 2;; --wave-list) wave_list="$2"; shift 2;; *) shift;; esac\n'
        "done\n"
        'printf "%s\\n" "wave-list: $wave_list" >> "$STUB_BOOTSTRAP_LOG"\n'
        'echo \'{"schema": "stub-bootstrap"}\' > "$state/bootstrap.json"\n'
        "cat > \"$state/client\" <<'CLIENT'\n"
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_CLIENT_LOG"\n'
        'case "$1" in\n'
        "  rcp-ndcg-vllm)\n"
        '    port=""; prev=""\n'
        '    for arg in "$@"; do [[ "$prev" == "--port" ]] && port="$arg"; prev="$arg"; done\n'
        '    echo "$$" >> "$STUB_SERVE_PIDS"\n'
        '    if [[ "${3:-}" == "gpt-oss-120b" ]]; then echo "stub engine: CUDA out of memory" >&2; exit 3; fi\n'
        '    exec python3 "$STUB_ENGINE_PY" "$port" "${3:-}"\n'
        "    ;;\n"
        "  rcp-ndcg)\n"
        '    echo \'{"schema": "rcp-ndcg.cli.v1", "command": "judge check", "ok": true}\'\n'
        "    ;;\n"
        "  python)\n"
        '    if [[ "${2:-}" == "-m" && "${3:-}" == "rcp_ndcg_test.jobs.judgecheck_probe" ]]; then\n'
        '      recipe=""; prev=""\n'
        '      for arg in "$@"; do [[ "$prev" == "--judge" ]] && recipe="$arg"; prev="$arg"; done\n'
        '      case "${4:-}" in\n'
        "        modes)\n"
        '          if [[ "$recipe" == gemma-4-* ]]; then\n'
        '            printf \'off\\non\\t{"mode": "on"}\\n\'\n'
        "          else\n"
        "            printf 'default\\n'\n"
        "          fi\n"
        "          ;;\n"
        "        media)\n"
        '          out=""; prev=""\n'
        '          for arg in "$@"; do [[ "$prev" == "--out" ]] && out="$arg"; prev="$arg"; done\n'
        '          printf \'{"checked": true, "outcome": "ok"}\\n\' > "$out"\n'
        "          ;;\n"
        "      esac\n"
        "    fi\n"
        "    ;;\n"
        "esac\n"
        "CLIENT\n"
        'chmod +x "$state/client"\n',
        encoding="utf-8",
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text('{"version": "0.0.1"}\n', encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(JUDGECHECK_SH), str(stage), "gs://YOUR-BUCKET/waves/judgecheck"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RCP_GCS_AUTH_FILE": str(auth),
            "RCP_GCS_HELPER_SH": str(gcs_sh),
            "RCP_GCS_HELPER_PY": str(helper),
            "RCP_BOOTSTRAP_SH": str(bootstrap),
            "RCP_REPORT_PY": str(REPORT_PY),
            "RCP_JUDGECHECK_PROBE_PY": str(JUDGECHECK_PROBE),
            "STUB_BOOTSTRAP_LOG": str(bootstrap_log),
            "STUB_CLIENT_LOG": str(client_log),
            "STUB_SERVE_PIDS": str(serve_pids),
            "STUB_ENGINE_PY": str(stub_engine),
            "STUB_GCS_LOG": str(gcs_log),
            "RCP_JUDGECHECK_STARTUP_S": "20",
        },
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr  # the dead engine fails the job
    assert "judgecheck: 1 recipe(s) failed: gpt-oss-120b" in completed.stderr
    state_line = next(line for line in completed.stdout.splitlines() if "judgecheck: state " in line)
    state = Path(state_line.split("judgecheck: state ", 1)[1])
    # the bootstrap envs call carries the ten-recipe wave list (the client environment)
    wave_list = (state / "judgecheck-wave.txt").read_text(encoding="utf-8").splitlines()
    assert wave_list == list(JUDGES), wave_list
    bootstrap_argv = bootstrap_log.read_text(encoding="utf-8")
    assert f"envs {stage} --state {state}" in bootstrap_argv, bootstrap_argv
    assert f"--wave-list {state}/judgecheck-wave.txt" in bootstrap_argv, bootstrap_argv
    # every judge served on its own port and waited for; the dead one recorded, the next still ran
    log = client_log.read_text(encoding="utf-8")
    for index, recipe in enumerate(JUDGES):
        assert f"rcp-ndcg-vllm serve {recipe} --port {8100 + index}" in log, log
    # the documented thinking modes: the Gemma 4 judges' two (off without a toggle, on with the declared
    # body), every other judge's single `default` mode with no toggle passed; the dead judge ran neither
    for index, recipe in enumerate(JUDGES):
        lines = [entry for entry in log.splitlines() if f"judge check --judge {recipe} " in entry]
        expected = 0 if recipe == "gpt-oss-120b" else (2 if recipe.startswith("gemma-4-") else 1)
        assert len(lines) == expected, (recipe, lines)
        for line in lines:
            assert f"--judge-model {recipe}" in line, line
            assert f"--set judge.base_url=http://127.0.0.1:{8100 + index}/v1" in line, line
            assert "--json" in line, line
        if recipe.startswith("gemma-4-"):
            assert any('--set judge.extra_body={"mode": "on"}' in line for line in lines), lines
            assert any("judge.extra_body" not in line for line in lines), lines
        else:
            assert all("judge.extra_body" not in line for line in lines), lines
    assert log.count("judge check") == 13, log  # five single-mode judges + four two-mode ones
    # the media probe ran for every judge (the helper records text-only itself); gpt-oss's engine never booted
    assert log.count("judgecheck_probe media") == 9, log
    assert not (state / "media-gpt-oss-120b.json").exists()
    # every engine's process group was stopped
    for pid in (int(line) for line in serve_pids.read_text(encoding="utf-8").splitlines()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    # the checks' JSONs upload per judge and mode; the failed judge's engine log does not become a check
    calls = gcs_log.read_text(encoding="utf-8")
    single_upload = (
        f"cp {state}/check-qwen3.5-397b-a17b-nvfp4-default.json "
        "gs://YOUR-BUCKET/waves/judgecheck/qwen3.5-397b-a17b-nvfp4/default.json file"
    )
    gemma_upload = (
        f"cp {state}/check-gemma-4-31b-it-nvfp4-on.json "
        "gs://YOUR-BUCKET/waves/judgecheck/gemma-4-31b-it-nvfp4/on.json file"
    )
    assert single_upload in calls and gemma_upload in calls, calls
    assert "check-gpt-oss-120b-default.json gs://YOUR-BUCKET/waves/judgecheck/gpt-oss-120b/default.json" not in calls
    assert f"cp {state}/judgecheck-report.json gs://YOUR-BUCKET/waves/judgecheck/judgecheck-report.json file" in calls
    report = json.loads((state / "judgecheck-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "rcp-ndcg.judgecheck-report.v1"
    assert report["passed"] is False and report["failed_step"] == "judgecheck"
    assert set(report["recipes"]) == set(JUDGES)
    assert report["recipes"]["gpt-oss-120b"]["state"] == "failed"
    assert "CUDA out of memory" in report["recipes"]["gpt-oss-120b"]["error"]
    assert report["recipes"]["qwen3.5-397b-a17b-nvfp4"]["state"] == "verified"
    single = report["recipes"]["qwen3.5-397b-a17b-nvfp4"]
    assert set(single["modes"]) == {"default"} and single["modes"]["default"]["ok"] is True
    assert single["modes_note"] == "single documented mode: the recipe documents no thinking-mode toggle"
    assert single["media"]["outcome"] == "ok"
    gemma = report["recipes"]["gemma-4-12b-it"]
    assert set(gemma["modes"]) == {"off", "on"} and all(row["ok"] for row in gemma["modes"].values())
    assert "modes_note" not in gemma
    assert isinstance(report["engine"]["vllm"], str)
