"""The wave runner against stub engines: two recipes on two slots, one failing beside one passing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rcp_ndcg_vllm.jobs.run_wave import run_wave

from tests.conftest import RECIPES, STUB, sample_pairs

VLLM_CMD = f"{sys.executable} {STUB}"


def test_wave_runs_two_recipes_on_two_slots(tmp_path: Path) -> None:
    """Two fixture recipes, two slots, both verified end to end with --record."""
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-rerank-pointwise", "fixture-embed"],
        RECIPES,
        gpus=2,
        out_dir=out,
        record=True,
        pairs_dir=RECIPES.parent.parent / "fixtures" / "pairs",  # missing on purpose: equivalence is skipped
        vllm_cmd=VLLM_CMD,
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    # Without pairs the equivalence step is skipped, so the recipes verify as "skipped equivalence" -> not verified.
    assert by_id["fixture-rerank-pointwise"]["state"] == "failed"
    assert by_id["fixture-rerank-pointwise"]["steps"]["equivalence"]["state"] == "skipped"
    assert (out / "fixture-rerank-pointwise" / "serve.log").is_file()
    assert (out / "fixture-rerank-pointwise" / "status.json").is_file()
    assert (out / "wave.json").is_file()
    assert "fixture-rerank-pointwise" in (out / "WAVE.md").read_text(encoding="utf-8")
    assert by_id["fixture-rerank-pointwise"]["steps"]["record"]["state"] == "passed"
    # Test mode gives EVERY engine --port 0: slot 1 must not get the privileged port 1 (CI binds no port < 1024).
    for row in document["recipes"]:
        argv = row["serve_argv"]
        assert argv[argv.index("--port") + 1] == "0", (row["recipe"], row["port"])


def test_wave_equivalence_passes_and_one_failure_never_stops_the_wave(tmp_path: Path) -> None:
    """With pairs, the clean recipe verifies; the noisy recipe fails its gates while the wave completes."""
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    for recipe_id in ("fixture-rerank-pointwise", "fixture-rerank-noisy"):
        (pairs_dir / f"{recipe_id}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sample_pairs()), encoding="utf-8"
        )
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-rerank-pointwise", "fixture-rerank-noisy"],
        RECIPES,
        gpus=2,
        out_dir=out,
        pairs_dir=pairs_dir,
        vllm_cmd=VLLM_CMD,
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-rerank-pointwise"]["state"] == "verified"
    assert by_id["fixture-rerank-noisy"]["state"] == "failed"
    assert document["passed"] is False
    equivalence = json.loads((out / "fixture-rerank-pointwise" / "equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["passed"] is True
    noisy_status = json.loads((out / "fixture-rerank-noisy" / "status.json").read_text(encoding="utf-8"))
    assert noisy_status["steps"]["equivalence"]["state"] == "failed"


def test_a_recipe_needing_too_many_gpus_fails_immediately(tmp_path: Path) -> None:
    """A recipe that needs more GPUs than the wave has fails without ever starting an engine."""
    out = tmp_path / "wave"
    document = run_wave(["fixture-embed"], RECIPES, gpus=0, out_dir=out)
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "needs 1 GPUs, the wave has 0" in row["error"]
    assert (out / "fixture-embed" / "status.json").is_file()


def test_recipe_ids_resolve_from_the_root_or_fail_clearly(tmp_path: Path) -> None:
    """An unknown id is a wave-request error (exit code 2 territory), not a recipe failure."""
    import pytest
    from rcp_ndcg_vllm.errors import HarnessError

    with pytest.raises(HarnessError, match="unknown recipe id"):
        run_wave(["no-such-recipe"], RECIPES, gpus=1, out_dir=tmp_path / "empty")
    with pytest.raises(HarnessError, match="no recipes under"):
        run_wave([], tmp_path, gpus=1, out_dir=tmp_path / "empty2")


def test_a_broken_reference_fails_only_its_recipe(tmp_path: Path) -> None:
    """A reference that raises at call time is one recipe's failure, never the wave's crash."""
    import shutil

    broken_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, broken_root)
    # The fixture references import their number module from the fixtures directory; a copied tree needs it.
    shutil.copy(RECIPES.parent / "deterministic.py", broken_root.parent / "deterministic.py")
    # A self-contained broken reference: stage 1 passes (ids and tokenizer hook), stage 2 raises.
    (broken_root / "fixture-rerank-pointwise" / "reference.py").write_text(
        "class _Tokenizer:\n"
        "    def encode(self, text):\n"
        "        return [len(word) for word in text.split()]\n\n"
        "    def id_to_token(self, token_id):\n"
        "        return f'tok:{token_id}'\n\n\n"
        "def load(device):\n"
        "    return None\n\n\n"
        "def tokenizer():\n"
        "    return _Tokenizer()\n\n\n"
        "def render(query, document, instruction):\n"
        "    return [1, 2, 3]\n\n\n"
        "def score(query, documents, instruction):\n"
        "    raise RuntimeError('the reference is broken')\n",
        encoding="utf-8",
    )
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    for recipe_id in ("fixture-rerank-pointwise", "fixture-embed"):
        (pairs_dir / f"{recipe_id}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sample_pairs()), encoding="utf-8"
        )
    document = run_wave(
        ["fixture-rerank-pointwise", "fixture-embed"],
        broken_root,
        gpus=2,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs_dir,
        vllm_cmd=VLLM_CMD,
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-rerank-pointwise"]["state"] == "failed"
    assert by_id["fixture-embed"]["state"] == "verified"
    equivalence = by_id["fixture-rerank-pointwise"]["steps"]["equivalence"]
    assert "the reference is broken" in (equivalence.get("error") or "")


def test_a_failed_smoke_step_fails_the_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe whose smoke request fails is not reported verified."""
    import rcp_ndcg_vllm.jobs.run_wave as module

    monkeypatch.setattr(module, "_smoke", lambda recipe, base_url: {"state": "failed", "error": "boom"})
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "fixture-rerank-pointwise.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_pairs()), encoding="utf-8"
    )
    document = run_wave(
        ["fixture-rerank-pointwise"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs_dir,
        vllm_cmd=VLLM_CMD,
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert row["steps"]["smoke"]["state"] == "failed"


import pytest  # noqa: E402  (kept at the end: only the last two tests need it)
