"""The version guard: refuse vLLM outside the tested range with a clear message."""

from __future__ import annotations

import pytest
import rcp_ndcg_vllm_topk.guard as guard  # noqa: I001 - conftest prepends src/ to sys.path


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.31.0", (0, 31)),  # the vllm/vllm-openai:v0.31.0 image
        ("0.31.1", (0, 31)),
        ("0.31.0rc2", (0, 31)),  # pre-releases of the tested line count in
        ("0.30.2", (0, 30)),
        ("0.32.0", (0, 32)),
        ("0.32.0.dev0", (0, 32)),
    ],
)
def test_parse_vllm_minor_version(version: str, expected: tuple[int, int]) -> None:
    """The (major, minor) prefix is read; suffixes after the minor are ignored."""
    assert guard.parse_vllm_minor_version(version) == expected


@pytest.mark.parametrize("version", ["garbage", ""])
def test_parse_vllm_minor_version_rejects_underspecified(
    version: str,
) -> None:
    """A string without a major and minor number cannot be range-checked; the
    caller refuses it (a two-component release like 0.31 is fine and counts
    as the tested line)."""
    assert guard.parse_vllm_minor_version(version) is None


def test_ensure_vllm_version_accepts_the_tested_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0.31.x passes and is returned as (major, minor)."""
    monkeypatch.setattr(guard, "installed_vllm_version", lambda: "0.31.0")
    assert guard.ensure_vllm_version() == (0, 31)


@pytest.mark.parametrize("version", ["0.30.3", "0.32.0", "0.32.0.dev0"])
def test_ensure_vllm_version_refuses_other_lines(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """Outside [0.31, 0.32) the guard raises with the tested range named."""
    monkeypatch.setattr(guard, "installed_vllm_version", lambda: version)
    with pytest.raises(RuntimeError, match=r">=0\.31,<0\.32") as excinfo:
        guard.ensure_vllm_version()
    # The message must carry the refused version so the operator sees what ran.
    assert version in str(excinfo.value)


def test_installed_vllm_version_without_vllm_names_the_distribution() -> None:
    """Where vLLM is absent (CPU dev environment) the guard refuses with a
    clear message instead of a bare PackageNotFoundError."""
    try:
        version = guard.installed_vllm_version()
    except RuntimeError as error:
        assert "vLLM" in str(error)
        pytest.skip(f"vLLM is not installed on this CPU environment: {error}")
    assert isinstance(version, str) and version
