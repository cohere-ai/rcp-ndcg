"""The staged plugin wheel vs the behaviour fingerprint (rcp-fp/4).

The fingerprint hashes the plugin modules the harness resolves; the engine runs the wheel bootstrap
installed.  Nothing used to cross-check the two: a corpus could be recorded under a fingerprint whose
plugin hashes belonged to another build than the wheel the pod served.  The wave hashes the modules
inside the staged wheel the same way and refuses to record when they differ (item 9)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
from rcp_ndcg_test.fingerprint import plugin_module_hashes
from rcp_ndcg_test.jobs import run_wave as run_wave_module
from rcp_ndcg_test.jobs.run_wave import run_wave
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs

REFERENCE_PYTHON = sys.executable
VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"


def _plugin_recipe_root(tmp_path: Path) -> Path:
    """A copy of the fixture recipes with ``serve.plugin: rcp-ndcg-vllm`` (so the fingerprint keys the
    plugin modules) and the tokenizer where the copied recipe's relative path finds it."""
    root = tmp_path / "recipes"
    shutil.copytree(RECIPES, root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", root.parent / "deterministic.py")
    yaml = root / "fixture-embed" / "family.yaml"
    yaml.write_text(
        yaml.read_text(encoding="utf-8").replace(
            "  plugin: null\n",
            "  plugin: rcp-ndcg-vllm\n  plugin_architectures: [PplxContextualModel]\n",
        ),
        encoding="utf-8",
    )
    return root


def _wheel(tmp_path: Path, recipe: object, *, corrupt: str | None = None) -> Path:
    """A wheel carrying the installed plugin module sources (the same bytes ``find_spec`` resolves),
    optionally with one module's bytes changed: the fingerprint and the wheel then disagree."""
    package = Path(importlib.util.find_spec("rcp_ndcg_vllm").origin or "").parent
    wheel = tmp_path / "staged" / "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        for module in plugin_module_hashes(recipe):  # type: ignore[arg-type]
            origin = Path(importlib.util.find_spec(module).origin or "")
            data = origin.read_bytes()
            if corrupt is not None and module == corrupt:
                data += b"\n# tampered\n"
            # The real wheel's members carry the package prefix (``rcp_ndcg_vllm/...``).
            archive.writestr(str(origin.relative_to(package.parent)), data)
    return wheel


def _wave(tmp_path: Path, root: Path, wheel: Path) -> dict:
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    return run_wave(
        ["fixture-embed"],
        root,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs,
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
        record_corpus=True,
        plugin_wheel=str(wheel),
    )


def test_the_staged_plugin_wheel_is_hashed_and_recorded(tmp_path: Path) -> None:
    """The wheel bootstrap installed is recorded in the corpus's model block (its SHA-256) and its
    modules hash to the fingerprint's plugin inputs, so the recording is accepted."""
    root = _plugin_recipe_root(tmp_path)
    recipe = load_recipe(root / "fixture-embed")
    wheel = _wheel(tmp_path, recipe)
    document = _wave(tmp_path, root, wheel)
    row = document["recipes"][0]
    assert row["state"] == "verified", row["steps"]["observation_corpus"]
    corpus = Path(row["steps"]["observation_corpus"]["corpus_dir"])
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["plugin"]["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_a_staged_plugin_wheel_that_differs_from_the_fingerprint_is_refused(tmp_path: Path) -> None:
    """A wheel whose module bytes differ from the source the fingerprint hashed is refused before the
    corpus is written: the recording would otherwise claim a plugin build the engine did not run."""
    root = _plugin_recipe_root(tmp_path)
    recipe = load_recipe(root / "fixture-embed")
    module = next(iter(plugin_module_hashes(recipe)))
    wheel = _wheel(tmp_path, recipe, corrupt=module)
    document = _wave(tmp_path, root, wheel)
    row = document["recipes"][0]
    step = row["steps"]["observation_corpus"]
    assert step["state"] == "failed", step
    assert module in step["error"] and "wheel" in step["error"], step
    assert row["state"] == "failed"
    assert not list((tmp_path / "wave").glob("observations/**/manifest.json"))


def test_a_wave_without_a_plugin_wheel_records_the_fingerprint_hashes_unchanged(tmp_path: Path) -> None:
    """No staged wheel given (a development run): the check is skipped, the plugin block records the
    wheel as unavailable, and the fingerprint's plugin hashes still key the corpus."""
    root = _plugin_recipe_root(tmp_path)
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    document = run_wave(
        ["fixture-embed"],
        root,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs,
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
        record_corpus=True,
    )
    row = document["recipes"][0]
    assert row["state"] == "verified", {k: row.get(k) for k in ("state", "error", "steps")}
    corpus = Path(row["steps"]["observation_corpus"]["corpus_dir"])
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert "unavailable" in manifest["model"]["plugin"]["wheel_sha256"]
    assert manifest["model"]["plugin"]["name"] == "rcp-ndcg-vllm"
    assert plugin_module_hashes(load_recipe(root / "fixture-embed"))  # the fingerprint keys these modules
    # The skip is visible in the step document: a plugin corpus recorded without a wheel says so.
    assert "not cross-checked" in str(row["steps"]["observation_corpus"]["plugin_wheel"])


def test_a_corrupt_staged_plugin_wheel_is_a_named_step_failure(tmp_path: Path) -> None:
    """A wheel file that is not a readable zip fails the corpus step with its reason, never an unhandled
    traceback that leaves the step reading ``running``."""
    root = _plugin_recipe_root(tmp_path)
    wheel = tmp_path / "staged" / "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"this is not a zip")
    document = _wave(tmp_path, root, wheel)
    row = document["recipes"][0]
    step = row["steps"]["observation_corpus"]
    assert step["state"] == "failed", step
    assert "cannot be read" in step["error"] and "BadZipFile" in step["error"]
    assert row["state"] == "failed"


def test_an_unexpected_zip_layer_error_is_a_named_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any zip-layer failure (an unsupported compression method, an encrypted member, a truncated central
    directory) folds into the same named refusal: the step fails with the reason instead of escaping."""
    root = _plugin_recipe_root(tmp_path)
    recipe = load_recipe(root / "fixture-embed")
    wheel = _wheel(tmp_path, recipe)
    module = next(iter(plugin_module_hashes(recipe)))

    def broken_read(self: zipfile.ZipFile, name: str) -> bytes:
        raise RuntimeError(f"{name} is encrypted")

    monkeypatch.setattr(zipfile.ZipFile, "read", broken_read)
    with pytest.raises(Exception, match="cannot be read"):
        run_wave_module._staged_plugin_hashes(wheel, [module])


@pytest.mark.parametrize("module", ["rcp_ndcg_vllm.models", "rcp_ndcg_vllm.models.pplx.config"])
def test_plugin_module_hashes_are_public_and_cover_package_inits(module: str, tmp_path: Path) -> None:
    """``plugin_module_hashes`` is the one home of the module set the fingerprint keys (a package's
    ``__init__`` included), so the wave's wheel check hashes exactly the same modules."""
    recipe = load_recipe(_plugin_recipe_root(tmp_path) / "fixture-embed")
    hashes = plugin_module_hashes(recipe)
    assert module in hashes and hashes[module].startswith("sha256:")
