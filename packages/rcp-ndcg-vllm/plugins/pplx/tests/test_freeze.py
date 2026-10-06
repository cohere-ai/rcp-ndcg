"""The simulated ``--no-deps`` freeze check.

The binding rule (GPU-VALIDATION.md item 1, plugin lanes): installing the plugin wheel
with ``--no-deps`` into an environment that has only vLLM must change ``pip freeze`` by
exactly this one distribution. The GPU wave runs that check for real
(``scripts/check_no_deps_freeze.sh`` against the engine environment); here it is
simulated against whatever interpreter runs the tests: the wheel is built with
``uv build --offline`` into ``tmp_path``, installed with ``--no-deps`` into a fresh
``--system-site-packages`` venv, and the freeze diff must be exactly the one
distribution — which also proves the wheel carries no ``Requires-Dist`` at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
WHEEL_NAME = "rcp_ndcg_vllm_pplx-0.0.1-py3-none-any.whl"


def _build_wheel(out_dir: Path) -> Path:
    result = subprocess.run(
        ["uv", "build", "--offline", "-o", str(out_dir), str(PLUGIN_DIR)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(f"uv build failed in this environment: {result.stderr[-300:]}")
    wheel = out_dir / WHEEL_NAME
    if not wheel.exists():
        pytest.skip(f"uv build produced no {WHEEL_NAME}: {result.stdout[-300:]}")
    return wheel


def test_wheel_is_pure_python_with_no_requirements(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "wheel")
    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        wheel_meta = archive.read("rcp_ndcg_vllm_pplx-0.0.1.dist-info/WHEEL").decode()
        metadata = archive.read("rcp_ndcg_vllm_pplx-0.0.1.dist-info/METADATA").decode()
        entry_points = archive.read("rcp_ndcg_vllm_pplx-0.0.1.dist-info/entry_points.txt").decode()
        names = archive.namelist()
    assert "Wheel-Version: 1" in wheel_meta
    assert "Root-Is-Purelib: true" in wheel_meta, "the wheel must stay pure python (py3-none-any)"
    assert "Requires-Dist" not in metadata, (
        "--no-deps install must be able to resolve nothing: no dependency may be declared"
    )
    assert "[vllm.general_plugins]" in entry_points
    assert "rcp_vllm_pplx = rcp_vllm_pplx:register" in entry_points
    assert any(name.endswith("rcp_vllm_pplx/model.py") for name in names)


def _python_with_venv_support(tmp_path: Path) -> str | None:
    """A python whose ``venv`` can bootstrap pip (uv-managed pythons ship no ensurepip)."""
    probe = tmp_path / "probe"
    for candidate in (sys.executable, "/usr/bin/python3"):
        result = subprocess.run([candidate, "-m", "venv", str(probe)], capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and (probe / "bin" / "pip").exists():
            return candidate
    return None


@pytest.mark.skipif(
    sys.prefix == getattr(sys, "base_prefix", sys.prefix),
    reason="no active virtualenv to simulate the freeze check in (a bare interpreter)",
)
def test_no_deps_install_changes_freeze_by_exactly_one_distribution(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "wheel")
    bootstrap = _python_with_venv_support(tmp_path)
    if bootstrap is None:
        pytest.skip(
            "no python in this environment can create a venv with pip (ensurepip is "
            "absent from uv-managed interpreters); the GPU wave runs the real check "
            "with the engine image's pip via scripts/check_no_deps_freeze.sh"
        )
    env_dir = tmp_path / "venv"
    subprocess.run(
        [bootstrap, "-m", "venv", "--system-site-packages", str(env_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    pip = env_dir / "bin" / "pip"

    def freeze() -> str:
        result = subprocess.run(
            [str(pip), "freeze", "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        return "\n".join(sorted(result.stdout.splitlines()))

    before = freeze()
    install = subprocess.run(
        [str(pip), "install", "--no-deps", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert install.returncode == 0, install.stderr
    after = freeze()

    added = sorted(set(after.splitlines()) - set(before.splitlines()))
    removed = sorted(set(before.splitlines()) - set(after.splitlines()))
    assert len(added) == 1 and added[0].startswith("rcp-ndcg-vllm-pplx"), (
        f"pip freeze must gain exactly the one distribution; gained {added}"
    )
    assert removed == [], f"pip freeze lost {removed}: the install disturbed the environment"
