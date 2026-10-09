"""The reference venv's dependency completion (the install must not resolve the image stack; the
venv's own distributions' missing deps complete to a fixed point from the staged wheelhouse only).

The planning and the fixed-point loop come from the standalone ``jobs/reference_deps.py``; the
environment facts are faked (fake distributions, a recorded pip), so everything here is offline and
hermetic.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs"
REFERENCE_DEPS = JOBS / "reference_deps.py"


def _module():
    spec = importlib.util.spec_from_file_location("reference_deps_under_test", REFERENCE_DEPS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_more_completes_only_the_venvs_own_missing_dependencies() -> None:
    """The venv's OWN dists' missing deps are planned one round at a time; the image's dists are never
    completed (that would shadow its CUDA stack), extras never count as needs, and anything satisfied
    by a visible version or already scheduled drops."""
    reference_deps = _module()
    planned = reference_deps.plan_more(
        {
            "sentence-transformers": ["scikit-learn>=1.0", "transformers>=4", "foo; extra == 'spark'"],
            "alpha": ["beta>=2"],
            "beta": ["gamma", "gamma==3"],  # both satisfied by the visible 3.0
        },
        visible={
            "alpha": ["1.0"],
            "beta": ["2.0"],
            "gamma": ["3.0"],
            "sentence-transformers": ["2.2"],
            "transformers": ["4.57.0"],
            "image-torch": ["2.13.0+cu128"],
        },
    )
    assert planned == ["scikit-learn>=1.0"]  # sorted-owner order; satisfied and extras drop
    assert (
        reference_deps.plan_more(
            {"alpha": ["beta>=2", "beta>=2", "beta"]},
            visible={"alpha": ["1"], "beta": ["7.0"]},
            scheduled={"beta"},
        )
        == []
    )
    assert reference_deps.requirement_name("Scikit_learn [stack] >= 1.0; python_version > '3'") == "scikit-learn"
    # an `extra !=` marker means the need exists (nothing asks for extras), unlike `extra ==`
    assert reference_deps.plan_more({"a": ["b; extra == 'x'", "c; extra != 'x'"]}, visible={"a": ["1"]}) == ["c"]


def test_plan_more_refuses_to_leave_an_image_dependency_stale() -> None:
    """A visible IMAGE dist whose version cannot satisfy an owned dist's need is a hard error with the
    way out (installing over it would shadow the image's CUDA stack) - never silently unmet."""
    reference_deps = _module()
    with pytest.raises(reference_deps.UnsatisfiableImageRequirement, match="REFERENCE_REQUIREMENTS"):
        reference_deps.plan_more({"a": ["b>=2"]}, visible={"a": ["1"], "b": ["1.0"]})
    # a venv-OWN dist at a stale version is upgraded inside the venv (planned again)
    assert reference_deps.plan_more({"b": [], "a": ["b>=2"]}, visible={"a": ["1"], "b": ["1.0"]}) == ["b>=2"]


class _FakeDist:
    """One installed distribution as importlib.metadata reports it."""

    def __init__(self, name: str, root: Path, requires: list[str], version: str = "") -> None:
        self.metadata = {"Name": name, "Version": version}
        self.requires = requires
        self._root = root

    def locate_file(self, _relative: str) -> Path:
        return self._root


def test_owned_and_visible_splits_the_venv_from_the_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Owned = installed into the reference venv (under sys.prefix); the image's distributions are
    visible with their versions but never owners (completing them would shadow its CUDA stack)."""
    reference_deps = _module()
    image_root = tmp_path / "image-site"  # outside sys.prefix: an image distribution
    monkeypatch.setattr(
        reference_deps,
        "distributions",
        lambda: [
            _FakeDist(
                "sentence-transformers", Path(reference_deps.sys.prefix) / "site-packages", ["scikit-learn"], "2.2"
            ),
            _FakeDist("torch", image_root, ["nvidia-nccl-cu13==2.29.7"], "2.13.0+cu128"),
        ],
    )
    owned, visible = reference_deps._owned_and_visible()
    assert owned == {"sentence-transformers": ["scikit-learn"]}
    assert visible == {"sentence-transformers": ["2.2"], "torch": ["2.13.0+cu128"]}


def test_reference_deps_reaches_a_fixed_point_installing_each_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loop installs what is missing among the OWN dists each round (alpha -> beta -> gamma), every
    install --no-deps --no-index from the wheelhouse, and stops at the fixed point."""
    reference_deps = _module()
    rounds = [
        ({"alpha": ["beta>=2"]}, {"alpha": ["1"]}),
        ({"alpha": ["beta>=2"], "beta": ["gamma"]}, {"alpha": ["1"], "beta": ["2"]}),
        ({"alpha": ["beta>=2"], "beta": ["gamma"], "gamma": []}, {"alpha": ["1"], "beta": ["2"], "gamma": ["3"]}),
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
        lambda: ({"alpha": ["nowhere-to-be-found>=9"]}, {"alpha": ["1"]}),
    )

    def fake_check_call(_argv: list[str]) -> None:
        raise subprocess.CalledProcessError(1, ["pip"])

    monkeypatch.setattr(reference_deps.subprocess, "check_call", fake_check_call)
    assert reference_deps.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "nowhere-to-be-found" in err
    assert "requirements-reference.txt" in err


def test_reference_deps_module_declares_its_public_names() -> None:
    """Every public module declares __all__ and every public function documents inputs, outputs, units."""
    reference_deps = _module()
    assert set(reference_deps.__all__) == {
        "UnsatisfiableImageRequirement",
        "canonical",
        "main",
        "plan_more",
        "requirement_name",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
