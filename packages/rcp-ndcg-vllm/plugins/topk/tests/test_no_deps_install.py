"""The simulated ``--no-deps`` freeze check, run against the built wheel.

Builds the wheel the way the release does (``uv build``; skipped with a clear
reason where no builder exists), then runs
``tests/check_no_deps_install.py`` — the same script the GPU wave wraps
around the real engine-environment install.  The rejection tests forge
broken wheels (declared dependency, compiled artifact, platform tag) into
tmp_path and require the check to fail loudly: the freeze delta alone cannot
catch a declared dependency, because ``--no-deps`` never installs one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from conftest import PLUGIN_ROOT

CHECK_SCRIPT = Path(__file__).resolve().parent / "check_no_deps_install.py"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build the pure wheel with uv (as the release does) and yield its path.

    Skipped when uv is unavailable: the freeze check needs a wheel, and the
    package tree alone is not one.  The GPU wave runs the same check against
    the staged wheel instead (README).
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH; no wheel builder available on this machine")
    out_dir = tmp_path_factory.mktemp("wheelhouse")
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir), str(PLUGIN_ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(PLUGIN_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(f"uv build failed:\n{result.stderr}")
    # The wheel build (setuptools backend) leaves intermediate artifacts in
    # the package tree; the checkout must stay clean (tests write only to
    # tmp_path), so remove what the build created.
    for stray in (PLUGIN_ROOT / "build", PLUGIN_ROOT / "src" / "rcp_ndcg_vllm_topk.egg-info"):
        shutil.rmtree(stray, ignore_errors=True)
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    yield wheels[0]


def load_check_module():
    """Import the standalone check script (stdlib only) as a module."""
    spec = spec_from_file_location("check_no_deps_install", CHECK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wheel_is_pure_python(built_wheel: Path) -> None:
    """The built wheel is py3-none-any with no compiled artifacts."""
    check = load_check_module()
    tag = check.assert_pure_python(built_wheel)
    assert tag == "py3-none-any"
    assert "rcp_ndcg_vllm_topk" in built_wheel.name


def test_no_deps_install_changes_freeze_by_exactly_the_wheel(built_wheel: Path, tmp_path: Path) -> None:
    """The binding rule: ``pip install --no-deps`` into a venv that only has
    pip adds exactly this one distribution (nothing pulled, nothing
    upgraded); pip renders the direct-path install as ``name @ file://...``,
    so the comparison is on the canonical distribution name."""
    check = load_check_module()
    requirement = check.check_install(built_wheel, tmp_path / "venv")
    assert check._canonical_name(requirement) == "rcp-ndcg-vllm-topk"


def test_check_script_standalone_exit_zero(built_wheel: Path) -> None:
    """The GPU-wave script runs standalone and exits 0 with the verdict line."""
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), str(built_wheel)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK: --no-deps install changed pip freeze by")


def _forge_wheel(source: Path, target: Path, edits: dict[bytes, bytes], extra: tuple[str, bytes] | None = None) -> Path:
    """Copy ``source`` to ``target`` with per-file content replacements plus
    one appended entry (the metadata regression probes below)."""
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            for old, new in edits.items():
                if old in data:
                    data = data.replace(old, new)
            dst.writestr(info, data)
        if extra is not None:
            dst.writestr(extra[0], extra[1])
    return target


def test_check_rejects_declared_dependencies(built_wheel: Path, tmp_path: Path) -> None:
    """A wheel that declares a dependency must fail the check: --no-deps would
    leave it uninstalled in the engine image (the freeze delta cannot catch
    this, because --no-deps never installs dependencies)."""
    check = load_check_module()
    forged = _forge_wheel(
        built_wheel,
        tmp_path / "with-deps.whl",
        {b"Requires-Python: >=3.12\n": b"Requires-Python: >=3.12\nRequires-Dist: torch>=2.0\n"},
    )
    with pytest.raises(RuntimeError, match="declares dependencies"):
        check.assert_pure_python(forged)


def test_check_rejects_compiled_artifacts(built_wheel: Path, tmp_path: Path) -> None:
    """A .so inside the wheel fails the purity check loudly."""
    check = load_check_module()
    forged = _forge_wheel(
        built_wheel,
        tmp_path / "compiled.whl",
        {},
        extra=("rcp_ndcg_vllm_topk/_evil.so", b"\x7fELF"),
    )
    with pytest.raises(RuntimeError, match="compiled artifacts"):
        check.assert_pure_python(forged)


def test_check_rejects_platform_tagged_wheels(built_wheel: Path, tmp_path: Path) -> None:
    """A non-py3-none-any wheel tag fails the purity check."""
    check = load_check_module()
    forged = _forge_wheel(
        built_wheel,
        tmp_path / "platform.whl",
        {b"Tag: py3-none-any\n": b"Tag: cp312-cp312-linux_x86_64\n"},
    )
    with pytest.raises(RuntimeError, match="wheel tag"):
        check.assert_pure_python(forged)
