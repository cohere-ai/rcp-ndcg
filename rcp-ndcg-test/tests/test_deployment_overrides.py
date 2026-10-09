"""The served provenance of the deployment overrides and of an unshipped (user) recipe.

The corpus manifest is what a later reader has to explain a difference between two recordings with, so the
two facts this lane adds to it are asserted here: the engine's recorded argv carries the serve-time
deployment overrides, and a recipe loaded from a path is marked unshipped with an ``unverified`` status.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_test.fingerprint import use_tokenizer_store
from rcp_ndcg_test.observe.provenance import engine_facts, recipe_facts
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe
from rcp_ndcg_vllm.serve import run_console

CORPORA_TOKENIZERS = Path(__file__).resolve().parents[1] / "corpora" / "vllm-0.31.0" / "_tokenizers"
use_tokenizer_store(CORPORA_TOKENIZERS)
"""The observation corpora vend their tokenizers here: the fingerprint resolves offline (as the golden guard)."""

SHIPPED = "qwen3-reranker-0.6b"


def _no_gpu(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """A runner stub for the node probes: the machine has no ``nvidia-smi``."""
    return subprocess.CompletedProcess(argv, 1, "", "not found")


def _user_family(tmp_path: Path) -> Path:
    """A user recipe file: a copy of a shipped family, re-idd (what an operator owns)."""
    target = tmp_path / "user-qwen3-reranker"
    shutil.copytree(default_recipes_root() / "qwen3-reranker", target)
    yaml_path = target / "family.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["id"] = target.name
    first = dict(data["variants"][0])
    first["id"] = "user-qwen3-reranker-0.6b"
    data["variants"] = [first]
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_the_provenance_records_the_overridden_argv(capsys: pytest.CaptureFixture[str]) -> None:
    """``engine.serve_argv`` is the record of how the engine was started: the deployment overrides are in
    it, exactly as the console printed (and execs) them."""
    recipe = load_recipe(SHIPPED)
    code = run_console(["serve", recipe.id, "--dry-run", "--set", "serve.max_num_seqs=64", "--set", "resources.gpus=2"])
    assert code == 0
    argv = shlex.split(capsys.readouterr().out.splitlines()[0])
    facts = engine_facts(
        image=recipe.engine.image,
        serve_argv=argv,
        engine_python=None,
        started=None,
        ready_wait_s=None,
        runner=_no_gpu,
    )
    recorded = facts["serve_argv"]
    assert recorded[recorded.index("--max-num-seqs") + 1] == "64"
    assert recorded[recorded.index("--tensor-parallel-size") + 1] == "2"


def test_an_unshipped_recipe_is_marked_in_the_provenance(tmp_path: Path) -> None:
    """The recipe block says ``shipped: false`` and ``unverified``: a file of the operator's own carries no
    verification, whatever its ``status`` block claims."""
    directory = _user_family(tmp_path)
    yaml_path = directory / "family.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["status"] = {"state": "verified", "image": "vllm/vllm-openai:v0.31.0", "date": "2026-01-01", "report": "x"}
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    facts = recipe_facts(load_recipe(directory))
    assert facts["shipped"] is False
    assert facts["status"]["state"] == "unverified"
    assert facts["id"] == "user-qwen3-reranker-0.6b"
    assert facts["behaviour_fingerprint"], "the fingerprint is still recorded for an unshipped recipe"

    shipped = recipe_facts(load_recipe(SHIPPED))
    assert shipped["shipped"] is True
