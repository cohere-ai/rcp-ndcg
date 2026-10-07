"""Every example runs, is short, and uses only the public surface.

The offline examples run in a fresh working directory. The network examples (the released data on the Hugging Face
Hub, MTEB) run only with ``RCP_NDCG_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.docs._markdown import ROOT

EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
NETWORK = {"01_score_released_suite.py", "07_mteb.py", "09_serve_recipe_score.py"}
PUBLIC = {
    "mteb",
    "rcp_ndcg",
    "rcp_ndcg.examples",
    "rcp_ndcg.eval.mteb",
    "rcp_ndcg.retrieval",
    "rcp_ndcg.testing",
    "rcp_ndcg_vllm",
    "rcp_ndcg_vllm.recipe",
}


def _run(example: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(example)], cwd=cwd, capture_output=True, text=True, timeout=600, check=False
    )


def test_the_example_set_is_complete() -> None:
    assert [p.name[:2] for p in EXAMPLES] == ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    from rcp_ndcg.examples import run_config_names, tiny

    assert run_config_names() == ["nano_nfcorpus_gpt5", "rejudge_nfcorpus", "tiny"]
    assert {path.name for path in tiny().iterdir()} >= {"rows.jsonl", "systems.jsonl", "released.json"}


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_examples_are_short_and_use_only_the_public_surface(example: Path) -> None:
    source = example.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 60
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    prefixes = {"rcp_ndcg", "rcp_ndcg_core", "rcp_ndcg_vllm", "mteb"}
    ours = {name for name in imported if name.split(".")[0] in prefixes}
    assert ours <= PUBLIC
    assert all(name.split(".")[0] in sys.stdlib_module_names for name in imported - ours)


@pytest.mark.parametrize("example", [e for e in EXAMPLES if e.name not in NETWORK], ids=lambda p: p.stem)
def test_offline_examples_run(example: Path, tmp_path: Path) -> None:
    result = _run(example, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip()


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RCP_NDCG_NETWORK_TESTS"), reason="set RCP_NDCG_NETWORK_TESTS=1 (HF Hub)")
@pytest.mark.parametrize("example", [e for e in EXAMPLES if e.name in NETWORK], ids=lambda p: p.stem)
def test_network_examples_run(example: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("huggingface_hub")
    if example.name == "07_mteb.py":
        pytest.importorskip("mteb")
    if example.name == "09_serve_recipe_score.py":
        pytest.importorskip("rcp_ndcg_vllm")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    result = _run(example, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
