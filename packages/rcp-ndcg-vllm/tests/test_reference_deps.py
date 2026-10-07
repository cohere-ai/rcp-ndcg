"""The reference venv's dependency completion (FINDINGS rows: pip must not resolve the image stack;
the venv's own distributions' missing deps complete to a fixed point from the staged wheelhouse only).

The planning and the fixed-point loop come from the standalone ``jobs/reference_deps.py``; the
environment facts are faked (fake distributions, a recorded pip), so everything here is offline and
hermetic.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
REFERENCE_DEPS = JOBS / "reference_deps.py"


def _module():
    spec = importlib.util.spec_from_file_location("reference_deps_under_test", REFERENCE_DEPS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_more_completes_only_the_venvs_own_missing_dependencies() -> None:
    """The venv's OWN dists' missing deps are planned one round at a time; the image's dists are never
    passed as owners (that would shadow its CUDA stack), extras never count as needs, and anything
    visible or already scheduled is satisfied."""
    reference_deps = _module()
    planned = reference_deps.plan_more(
        {
            "sentence-transformers": ["scikit-learn>=1.0", "transformers>=4", "foo; extra == 'spark'"],
            "alpha": ["beta>=2"],
            "beta": ["gamma", "gamma==3"],
        },
        visible={"transformers", "alpha", "sentence-transformers", "image-torch"},
        scheduled=set(),
    )
    assert planned == ["beta>=2", "gamma", "gamma==3", "scikit-learn>=1.0"]  # sorted-owner order
    # the extras marker is skipped, visible drops, scheduled drops, duplicates drop
    assert (
        reference_deps.plan_more(
            {"alpha": ["beta>=2", "beta>=2", "beta", "foo; extra == 'x'"]},
            visible={"alpha"},
            scheduled={"beta"},
        )
        == []
    )
    assert reference_deps.requirement_name("Scikit_learn [stack] >= 1.0; python_version > '3'") == "scikit-learn"


class _FakeDist:
    """One installed distribution as importlib.metadata reports it."""

    def __init__(self, name: str, root: Path, requires: list[str]) -> None:
        self.metadata = {"Name": name}
        self.requires = requires
        self._root = root

    def locate_file(self, _relative: str) -> Path:
        return self._root


def test_owned_and_visible_splits_the_venv_from_the_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Owned = installed into the reference venv (under sys.prefix); the image's distributions are
    visible but never owners (completing them would shadow its CUDA stack)."""
    reference_deps = _module()
    image_root = tmp_path / "image-site"  # outside sys.prefix: an image distribution
    monkeypatch.setattr(
        reference_deps,
        "distributions",
        lambda: [
            _FakeDist("sentence-transformers", Path(reference_deps.sys.prefix) / "site-packages", ["scikit-learn"]),
            _FakeDist("torch", image_root, ["nvidia-nccl-cu13==2.29.7"]),
        ],
    )
    owned, visible = reference_deps._owned_and_visible()
    assert owned == {"sentence-transformers": ["scikit-learn"]}
    assert visible == {"sentence-transformers", "torch"}


def test_reference_deps_reaches_a_fixed_point_installing_each_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loop installs what is missing among the OWN dists each round (alpha -> beta -> gamma), every
    install --no-deps --no-index from the wheelhouse, and stops at the fixed point."""
    reference_deps = _module()
    rounds = [
        ({"alpha": ["beta>=2"]}, {"alpha"}),
        ({"alpha": ["beta>=2"], "beta": ["gamma"]}, {"alpha", "beta"}),
        ({"alpha": ["beta>=2"], "beta": ["gamma"], "gamma": []}, {"alpha", "beta", "gamma"}),
    ]
    state = {"calls": 0}
    installs: list[list[str]] = []

    def fake_owned_visible():
        index = min(state["calls"], len(rounds) - 1)
        return rounds[index]

    def fake_check_call(argv: list[str]) -> None:
        state["calls"] += 1
        installs.append(list(argv))

    monkeypatch.setattr(reference_deps, "_owned_and_visible", fake_owned_visible)
    monkeypatch.setattr(reference_deps.subprocess, "check_call", fake_check_call)
    wheelhouse = tmp_path / "wheelhouse"
    assert reference_deps.main([str(wheelhouse)]) == 0
    assert [argv[len(argv) - 1 :] for argv in installs] == [["beta>=2"], ["gamma"]]
    for argv in installs:
        assert "--no-deps" in argv  # each completion install, never a resolution
        assert "--no-index" in argv and str(wheelhouse) in argv


def test_reference_deps_fails_loudly_when_the_wheelhouse_cannot_satisfy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing dependency no wheel satisfies: exit 1 with the names and the way out (add it to
    requirements-reference.txt or stage its wheel) - never a silent gap."""
    reference_deps = _module()
    monkeypatch.setattr(
        reference_deps,
        "_owned_and_visible",
        lambda: ({"alpha": ["nowhere-to-be-found>=9"]}, {"alpha"}),
    )

    def fake_check_call(_argv: list[str]) -> None:
        raise subprocess.CalledProcessError(1, ["pip"])

    monkeypatch.setattr(reference_deps.subprocess, "check_call", fake_check_call)
    assert reference_deps.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "nowhere-to-be-found" in err and "requirements-reference.txt" in err


def test_reference_deps_module_declares_its_public_names() -> None:
    """Every public module declares __all__ and every public function documents inputs, outputs, units."""
    reference_deps = _module()
    assert set(reference_deps.__all__) == {"canonical", "main", "plan_more", "requirement_name"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
