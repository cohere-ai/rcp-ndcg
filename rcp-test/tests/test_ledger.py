"""The post-wave ledger tool (``rcp_test.jobs.ledger``): the rows and the status blocks come from the
waves' own evidence, a wave whose RC commit's tree does not compute today's behaviour key is refused by
name, and a second ``--write`` changes nothing.

The evidence is a wave directory as :mod:`rcp_test.jobs.run_wave` writes it (``wave.json``,
``WAVE.md``, the recipe's ``status.json`` and ``equivalence.json``), and the "RC commit" is a real git
commit of the writable copy, so the key is derived exactly as the tool derives it in the campaign: from the
tree at that commit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rcp_test.jobs.ledger import LedgerEvidenceError, main, plan

ROOT = Path(__file__).resolve().parents[2]
WRITTEN = (
    "rcp-vllm/src/rcp_vllm/recipes",
    "rcp-test/tests/recipes/golden",
    "docs/reference/gpu-validation.md",
    "rcp-vllm/README.md",
)
TREE = (
    "rcp-core/src",
    "rcp/src",
    "rcp-vllm/src",
    "rcp-test/src",
    "rcp-test/tests/recipes/golden",
    "docs/reference/gpu-validation.md",
    "rcp-vllm/README.md",
)


def _checkout(tmp_path: Path) -> Path:
    """A writable copy of the sources, the goldens and the two rendered pages, as one git commit."""
    root = tmp_path / "checkout"
    for relative in TREE:
        source, target = ROOT / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, target)
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    return root


def _commit(root: Path) -> str:
    """Commit the working tree and return its commit id (the wave's tested RC)."""
    git = ["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "tree"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _wave(evidence: Path, name: str, recipe: str, commit: str, state: str = "verified") -> Path:
    """One wave's evidence directory: the wave summary (as data and as prose) and the recipe's documents."""
    wave = evidence / name / "vllm-vllm-openai-v0-33-0"
    (wave / recipe).mkdir(parents=True, exist_ok=True)
    passed = state == "verified"
    (wave / "wave.json").write_text(
        json.dumps(
            {
                "rc": {"name": "rc1", "commit": commit, "manifest_sha256": "0" * 64},
                "recipes": [{"recipe": recipe, "state": state, "steps": {}}],
                "fingerprints": {},
                "engine_versions": {recipe: "0.33.0"},
                "verdict": "passed" if passed else "failed",
                "passed": passed,
                "finished": "2026-11-10T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    verdict = "PASSED" if passed else "FAILED"
    (wave / "WAVE.md").write_text(
        f"# Wave summary\n\n| recipe | gpus | state | error |\n|---|---|---|---|\n| {recipe} | [0] | {state} |  |\n"
        f"\n- tested RC: rc1 at commit {commit} (manifest sha256 {'0' * 64})\n\nVerdict: **{verdict}**\n",
        encoding="utf-8",
    )
    (wave / recipe / "status.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "state": state,
                "engine_version": "0.33.0",
                "steps": {"equivalence": {"state": "passed" if passed else "failed"}},
            }
        ),
        encoding="utf-8",
    )
    if passed:
        (wave / recipe / "equivalence.json").write_text(
            json.dumps({"recipe": recipe, "image": "vllm/vllm-openai:v0.33.0", "passed": True}),
            encoding="utf-8",
        )
    return wave


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file the tool writes, by path, as bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for relative in WRITTEN
        for path in ([root / relative] if (root / relative).is_file() else sorted((root / relative).rglob("*")))
        if path.is_file()
    }


def test_the_ledger_refuses_evidence_that_does_not_compute_todays_key(tmp_path: Path) -> None:
    """A passing wave is evidence only when its RC commit's tree computes the key the tree computes today;
    a wave that claims a row without its own documents is refused too."""
    root, evidence = _checkout(tmp_path), tmp_path / "waves"
    commit = _commit(root)
    family = root / "rcp-vllm/src/rcp_vllm/recipes/octen-embedding/family.yaml"
    original = family.read_text(encoding="utf-8")
    _wave(evidence, "equivalence-octen-embedding", "octen-embedding-8b", commit)
    rows = {row.recipe: row for row in plan(evidence, root)}
    assert rows["octen-embedding-8b"].state == "verified"  # the RC tree computes today's key
    # A wave recorded on a tree that keys another behaviour (a serve field moved since) is refused, by name.
    family.write_text(original.replace("dtype: bfloat16", "dtype: float32"), encoding="utf-8")
    stale = _commit(root)
    family.write_text(original, encoding="utf-8")
    _wave(evidence, "equivalence-octen-embedding", "octen-embedding-8b", stale)
    with pytest.raises(LedgerEvidenceError, match="octen-embedding-8b"):
        plan(evidence, root)
    # The same wave at today's key, but without the documents its row rests on, is refused rather than written.
    _wave(evidence, "equivalence-octen-embedding", "octen-embedding-8b", commit)
    (evidence / "equivalence-octen-embedding/vllm-vllm-openai-v0-33-0/octen-embedding-8b/status.json").unlink()
    with pytest.raises(LedgerEvidenceError, match="octen-embedding-8b"):
        plan(evidence, root)


def test_the_ledger_writes_the_waves_rows_and_a_second_write_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run prints the table and writes nothing; ``--write`` writes the row, the status block, the
    golden and the catalog cell, and the same evidence writes the same bytes again."""
    root, evidence = _checkout(tmp_path), tmp_path / "waves"
    commit = _commit(root)
    _wave(evidence, "equivalence-octen-embedding", "octen-embedding-8b", commit)
    argv = ["--evidence", str(evidence), "--repo-root", str(root)]
    before = _snapshot(root)
    assert main(argv) == 0 and "octen-embedding-8b" in capsys.readouterr().out
    assert _snapshot(root) == before, "the default is a dry run"
    assert main([*argv, "--write"]) == 0
    written = _snapshot(root)
    family = (root / "rcp-vllm/src/rcp_vllm/recipes/octen-embedding/family.yaml").read_text()
    # the octen family's other two sizes have no wave, so the verdict is the verified variant's own block
    assert (
        "  - id: octen-embedding-8b\n    status:\n      state: verified\n"
        '      image: "vllm/vllm-openai:v0.33.0"\n      date: "2026-11-10"\n'
    ) in family
    assert "`octen-embedding-8b` | rerank |" not in written["docs/reference/gpu-validation.md"].decode()
    assert (
        written["rcp-test/tests/recipes/golden/octen-embedding-8b.json"]
        != before["rcp-test/tests/recipes/golden/octen-embedding-8b.json"]
    )
    assert main([*argv, "--write"]) == 0
    assert _snapshot(root) == written, "a second --write on the same evidence is a no-op"
