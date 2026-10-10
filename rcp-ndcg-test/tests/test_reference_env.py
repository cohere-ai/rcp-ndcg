"""The per-family reference environments (owner decision 35): lock parsing, the family selection and the
post-install import check."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.jobs import reference_env as renv
from rcp_ndcg_test.jobs import reference_lock as rl

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"
IMAGE_STACK = ROOT / "rcp-ndcg-vllm" / "reference-image-v0.31.0.txt"

FREEZE = "torch==2.13.0+cu130\nnvidia-cublas==13.1.1.3\ntransformers==5.17.0\n"


def _fake_compiler(requirements: list[str], **kwargs: object) -> str:
    return "".join(f"{rl.requirement_name(spec)}==9.9.9\n" for spec in requirements)


def _lock(tmp_path: Path, reference_in: str, *, family: str = "demo") -> Path:
    lock_text = rl.build_lock(
        reference_in,
        FREEZE,
        family=family,
        image="vllm/vllm-openai:v0.31.0",
        index_url=None,
        uv="uv",
    )
    path = tmp_path / f"{family}.lock"
    path.write_text(lock_text, encoding="utf-8")
    return path


def test_parse_lock_reads_the_header_and_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    path = _lock(tmp_path, "transformers>=4.51\ntorch>=2.0\n")
    info = renv.parse_lock(path)
    assert info.family == "demo"
    assert info.image == "vllm/vllm-openai:v0.31.0"
    assert info.own_torch is False
    assert info.pins == {"transformers": "9.9.9"}
    assert info.image_constraints == {"torch": "2.13.0+cu130"}
    assert len(info.sha256) == 64


def test_parse_lock_refuses_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "nope.lock"
    path.write_text("transformers==4.51\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="reference-lock"):
        renv.parse_lock(path)


def test_family_rows_groups_variants_and_reports_load_failures() -> None:
    rows, failures = renv.family_rows(RECIPES, ["qwen3-reranker-0.6b", "qwen3-reranker-4b", "no-such-recipe"])
    assert [row[0] for row in rows] == ["qwen3-reranker"]
    family, lock, own_torch = rows[0]
    assert lock == RECIPES / "qwen3-reranker" / "reference.lock"
    assert own_torch is True  # the flash-attn family declares its own stack
    assert "no-such-recipe" in failures


def test_reference_python_is_the_family_venv() -> None:
    assert renv.reference_python("/state/reference", "zerank") == Path("/state/reference/zerank/bin/python")


def test_import_problems_accepts_an_installed_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The check runs in the target python (here the dev environment): numpy is installed at its pin."""
    import importlib.metadata as metadata

    version = metadata.version("numpy")
    monkeypatch.setattr(rl, "compile_requirements", lambda requirements, **kwargs: f"numpy=={version}\n")
    lock = renv.parse_lock(_lock(tmp_path, "numpy\n"))
    problems, facts = renv.import_problems(sys.executable, lock)
    assert problems == []
    assert facts["numpy"] == version


def test_import_problems_names_the_family_and_the_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rl, "compile_requirements", lambda requirements, **kwargs: "definitely-not-installed==1.0\n")
    lock = renv.parse_lock(_lock(tmp_path, "definitely-not-installed\n"))
    problems, _facts = renv.import_problems(sys.executable, lock)
    assert problems and problems[0].startswith("family demo:")
    assert "definitely-not-installed==1.0 is not installed" in problems[0]


def test_check_cli_reports_ok_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    import importlib.metadata as metadata

    version = metadata.version("numpy")
    monkeypatch.setattr(rl, "compile_requirements", lambda requirements, **kwargs: f"numpy=={version}\n")
    lock = _lock(tmp_path, "numpy\n")
    assert renv.main(["check", "--python", sys.executable, "--lock", str(lock)]) == 0
    bad = tmp_path / "bad.lock"
    bad.write_text(lock.read_text(encoding="utf-8").replace(f"numpy=={version}", "numpy==0.0.1"), encoding="utf-8")
    assert renv.main(["check", "--python", sys.executable, "--lock", str(bad)]) == 1
