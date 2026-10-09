"""The wave runner's uploads: the wave summary lands, every upload is verified and retried, a failed
upload is recorded in the recipe's status row, and a wave with a failed upload exits non-zero (B1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rcp_ndcg_test.jobs import run_wave as run_wave_module
from rcp_ndcg_test.jobs.run_wave import main, run_wave

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs

REFERENCE_PYTHON = sys.executable
VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"


def _wave(tmp_path: Path, **overrides: object) -> dict:
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    kwargs: dict = {
        "recipe_ids": ["fixture-embed"],
        "recipes_root": RECIPES,
        "gpus": 1,
        "out_dir": tmp_path / "wave",
        "pairs_dir": pairs,
        "reference_python": REFERENCE_PYTHON,
        "vllm_cmd": f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        "port_base": 0,
    }
    kwargs.update(overrides)
    return run_wave(**kwargs)


def test_the_wave_summary_reaches_the_upload_uri(tmp_path: Path) -> None:
    """The wave summary is written BEFORE the last upload: ``wave.json`` and ``WAVE.md`` land at the
    destination beside the per-recipe directories, and the upload's outcome is recorded in the wave
    document and in each recipe's ``status.json`` row."""
    bucket = tmp_path / "bucket"
    document = _wave(tmp_path, upload=str(bucket))
    assert document["passed"] is True
    assert document["upload"]["ok"] is True and document["upload"]["attempts"] == 1
    summary = json.loads((bucket / "wave.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True and summary["recipes"]
    assert (bucket / "WAVE.md").is_file()
    assert (bucket / "fixture-embed" / "status.json").is_file()
    assert document["recipes"][0]["upload"]["ok"] is True
    local = json.loads((tmp_path / "wave" / "fixture-embed" / "status.json").read_text(encoding="utf-8"))
    assert local["upload"]["ok"] is True


def test_an_upload_that_does_not_land_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transfer that reports success but leaves nothing at the destination is caught by the
    verification: the wave document records it, the recipe's status row records it, and the wave
    does not pass (the old upload was fire-and-forget: a wave whose every upload failed still PASSed)."""
    monkeypatch.setattr(run_wave_module, "_upload_cli", lambda *args, **kwargs: True)  # claims success
    monkeypatch.setattr(run_wave_module, "_UPLOAD_BACKOFF_S", 0)
    bucket = tmp_path / "bucket"
    document = _wave(tmp_path, upload=str(bucket))
    assert document["passed"] is False
    assert document["upload"]["ok"] is False
    assert document["upload"]["attempts"] == run_wave_module._UPLOAD_ATTEMPTS
    assert "does not hold" in str(document["upload"]["error"])
    assert document["upload_failures"]["fixture-embed"]["ok"] is False
    local = json.loads((tmp_path / "wave" / "fixture-embed" / "status.json").read_text(encoding="utf-8"))
    assert local["upload"]["ok"] is False
    assert "UPLOAD FAILED" in (tmp_path / "wave" / "WAVE.md").read_text(encoding="utf-8")


def test_a_transient_upload_failure_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed attempt is retried (with backoff) and the retry's success is recorded: the upload is
    no longer a single fire-and-forget copy."""
    calls = {"n": 0}
    real = run_wave_module._upload_storage

    def flaky(source: Path, uri: str) -> str | None:
        calls["n"] += 1
        if calls["n"] == 1:
            return "transient 503 from the object store"
        return real(source, uri)

    monkeypatch.setattr(run_wave_module, "_upload_cli", lambda *args, **kwargs: False)
    monkeypatch.setattr(run_wave_module, "_upload_storage", flaky)
    monkeypatch.setattr(run_wave_module, "_UPLOAD_BACKOFF_S", 0)
    document = _wave(tmp_path, upload=str(tmp_path / "bucket"))
    assert document["passed"] is True, (document.get("upload"), document.get("upload_failures"))
    upload = document["recipes"][0]["upload"]
    assert upload["ok"] is True and upload["attempts"] == 2 and upload["error"] is None


def test_main_exits_non_zero_when_an_upload_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exit code carries the upload failure: a wave whose uploads all failed is not green."""
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    monkeypatch.setattr(run_wave_module, "_upload_cli", lambda *args, **kwargs: False)
    monkeypatch.setattr(run_wave_module, "_upload_storage", lambda *args, **kwargs: "the object store is down")
    monkeypatch.setattr(run_wave_module, "_UPLOAD_BACKOFF_S", 0)
    argv = [
        "--recipes",
        "fixture-embed",
        "--recipes-root",
        str(RECIPES),
        "--gpus",
        "1",
        "--out",
        str(tmp_path / "wave"),
        "--upload",
        str(tmp_path / "bucket"),
        "--pairs-dir",
        str(pairs),
        "--reference-python",
        REFERENCE_PYTHON,
        "--vllm-cmd",
        f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        "--port-base",
        "0",
    ]
    assert main(argv) == 1
