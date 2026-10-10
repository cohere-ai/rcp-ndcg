"""The wave runner's observation-corpus step and its re-record-changed-only mode (stub engine, CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rcp_ndcg_test.fingerprint import behaviour_fingerprint
from rcp_ndcg_test.jobs import run_wave as run_wave_module
from rcp_ndcg_test.jobs.run_wave import run_wave
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs

VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"
REFERENCE_PYTHON = sys.executable


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


def _checks(step: dict) -> dict[str, dict]:
    return {check["check"]: check for check in step["checks"]}


def test_the_wave_records_an_observation_corpus_at_its_keyed_path(tmp_path: Path) -> None:
    """``--record-corpus`` writes the corpus at ``observations/<engine>-<version>/<recipe>/<fingerprint>/<at>/``,
    keyed by the one behaviour fingerprint; the engine restarts between the in-process passes and the
    after-restart pass (a new run id); the corpus is checked against the equivalence stage's replies."""
    document = _wave(tmp_path, record_corpus=True)
    row = document["recipes"][0]
    step = row["steps"]["observation_corpus"]
    assert step["state"] == "passed", [check for check in step.get("checks", []) if not check["passed"]] or step
    fingerprint = behaviour_fingerprint(load_recipe(RECIPES / "fixture-embed"))
    assert row["behaviour_fingerprint"] == fingerprint == document["fingerprints"]["fixture-embed"]
    assert document["engine_versions"] == {"fixture-embed": "test-stub"}
    corpus = Path(step["corpus_dir"])
    assert corpus.parent == tmp_path / "wave" / "observations" / "vllm-test-stub" / "fixture-embed" / fingerprint
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    passes = manifest["collector"]["passes"]
    assert [entry["repetition"] for entry in passes] == ["same_process_1", "same_process_2", "after_restart"]
    assert passes[2]["server_run_id"] != passes[0]["server_run_id"]
    assert manifest["plan"]["strata"]["edge:while_loading"] == {"present": True}
    consistency = _checks(step)["equivalence_consistent"]
    assert consistency["passed"] is True and consistency["compared"] > 0, consistency


def test_changed_since_skips_the_unchanged_and_records_the_changed(tmp_path: Path) -> None:
    """--changed-since re-records only what moved (OBSERVATIONS-SPEC 7): an unchanged recipe is listed as
    skipped_unchanged and never served again; a changed fingerprint or a new engine version records again."""
    first = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave1")
    index = tmp_path / "wave1" / "wave.json"
    assert index.is_file() and first["fingerprints"] and first["engine_versions"]
    second = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave2", changed_since_index=index)
    assert second["skipped_unchanged"] == ["fixture-embed"]
    assert second["recipes"] == []
    document = json.loads(index.read_text(encoding="utf-8"))
    document["fingerprints"]["fixture-embed"] = "0" * 64
    index.write_text(json.dumps(document), encoding="utf-8")
    third = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave3", changed_since_index=index)
    assert third["skipped_unchanged"] == []
    assert third["changes"] == {"fixture-embed": ["behaviour_fingerprint"]}
    assert third["recipes"] and third["recipes"][0]["recipe"] == "fixture-embed"


def test_a_new_engine_version_records_every_recipe_again(tmp_path: Path) -> None:
    """The behaviour fingerprint does not include the engine: a recording under another engine version is a
    new corpus key, so an unchanged fingerprint under a new engine records again (and the protocol layer is
    due for that engine version)."""
    index = tmp_path / "old-wave.json"
    recipe = load_recipe(RECIPES / "fixture-embed")
    index.write_text(
        json.dumps(
            {
                "fingerprints": {"fixture-embed": behaviour_fingerprint(recipe)},
                "engine_versions": {"fixture-embed": "0.30.0"},
            }
        ),
        encoding="utf-8",
    )
    document = _wave(tmp_path, record_corpus=True, changed_since_index=index)
    assert document["skipped_unchanged"] == []
    assert document["changes"] == {"fixture-embed": ["engine_version"]}
    assert document["protocol_due"] == ["test-stub"]


def test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5: the corpus key is the version the RUNNING engine reports (its ``/version``), never the declared
    image string; with a measured version the status table is checked instead of skipped."""
    monkeypatch.setattr(run_wave_module, "_pod_engine_version", lambda run, vllm_cmd: "0.31.0")
    document = _wave(tmp_path, record_corpus=True)
    row = document["recipes"][0]
    assert document["engine_versions"] == {"fixture-embed": "0.31.0"}
    corpus = Path(row["steps"]["observation_corpus"]["corpus_dir"])
    assert corpus.parent.parent.parent.name == "vllm-0.31.0"
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"]["version"] == "0.31.0"
    # The measured table is APPLIED (the stub is deliberately stricter than real vLLM on unknown fields,
    # so it records mismatches rather than the vacuous pass an unmeasured version gets).
    statuses = _checks(row["steps"]["observation_corpus"])["statuses_as_expected"]
    assert statuses["mismatches"], statuses


def test_a_digest_pinned_recipe_records_the_pod_version_not_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipe declaring a digest-pinned image (the nightly case) still records the version the pod
    reports: the old key was the digest's 64 hex characters and its status expectations were vacuous."""
    import shutil

    root = tmp_path / "recipes"
    shutil.copytree(RECIPES, root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", root.parent / "tokenizer.json")
    yaml = root / "fixture-embed" / "family.yaml"
    yaml.write_text(
        yaml.read_text(encoding="utf-8").replace(
            'image: "vllm/vllm-openai:v0.31.0"',
            'image: "registry.example.com/engine:nightly-abc@sha256:' + "a" * 64 + '"',
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_wave_module, "_pod_engine_version", lambda run, vllm_cmd: "0.31.0")
    document = _wave(tmp_path, record_corpus=True, recipes_root=root)
    row = document["recipes"][0]
    assert document["engine_versions"] == {"fixture-embed": "0.31.0"}
    assert "vllm-0.31.0" in row["steps"]["observation_corpus"]["corpus_dir"]
    # The status expectations follow the POD's version (0.31.0's measured table), not the digest's
    # 64-hex "version" that made every expectation return None.
    assert _checks(row["steps"]["observation_corpus"])["statuses_as_expected"]["mismatches"]


def test_a_recipe_whose_engine_version_cannot_be_probed_is_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither the engine's ``/version`` nor the engine environment answers, the corpus cannot be
    keyed: the recipe fails loudly instead of silently keying by the declared image."""
    monkeypatch.setattr(run_wave_module, "_pod_engine_version", lambda run, vllm_cmd: None)
    document = _wave(tmp_path, record_corpus=True)
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "/version" in (row.get("error") or "")


def test_probe_engine_version_asks_the_running_engine() -> None:
    """``_probe_engine_version`` reads the version from the live engine's ``/version`` route (the pod),
    which is what the wave records -- not the recipe's declared image string."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's interface
            body = json.dumps({"version": "0.31.0"}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert run_wave_module._probe_engine_version(server.server_port) == "0.31.0"
    finally:
        server.shutdown()
        thread.join()


def test_pod_engine_version_composes_the_route_and_the_engine_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5's composition, pinned without monkeypatching it: the live ``/version`` route first, the engine
    environment's own ``vllm`` second, ``None`` when neither answers (so the recording refuses), and the
    test stub in test mode.  A mutation that reverts to the declared image must fail here."""
    import http.server
    import threading
    import types

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's interface
            body = json.dumps({"version": "9.9.9"}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert run_wave_module._pod_engine_version(types.SimpleNamespace(port=server.server_port), None) == "9.9.9"
    finally:
        server.shutdown()
        thread.join()
    # The engine environment's own vllm is the fallback when /version does not answer (a closed port).
    script = tmp_path / "engine-python"
    script.write_text("#!/usr/bin/env bash\necho 0.42.0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("RCP_ENGINE_PYTHON", str(script))
    assert run_wave_module._pod_engine_version(types.SimpleNamespace(port=1), None) == "0.42.0"
    monkeypatch.delenv("RCP_ENGINE_PYTHON", raising=False)
    assert run_wave_module._pod_engine_version(types.SimpleNamespace(port=1), None) is None
    assert run_wave_module._pod_engine_version(types.SimpleNamespace(port=1), "stub") == "test-stub"


def test_the_quality_step_refuses_a_missing_engine_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F4: when neither version probe answered, the T3 quality step fails instead of recording the
    declared image's tag as the engine version."""
    monkeypatch.setattr(run_wave_module, "_pod_engine_version", lambda run, vllm_cmd: None)
    document = _wave(tmp_path, quality=True)
    step = document["recipes"][0]["steps"]["quality"]
    assert step["state"] == "failed"
    assert "/version" in step["error"]


def test_an_all_skipped_changed_since_wave_reports_skipped(tmp_path: Path) -> None:
    """B5: an all-skipped ``--changed-since`` wave verified nothing, so it reports SKIPPED -- never PASS
    with zero evidence (the old document said ``passed: true`` for an empty recipe list)."""
    _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave1")
    second = _wave(
        tmp_path,
        record_corpus=True,
        out_dir=tmp_path / "wave2",
        changed_since_index=tmp_path / "wave1" / "wave.json",
    )
    assert second["recipes"] == [] and second["skipped_unchanged"] == ["fixture-embed"]
    assert second["verdict"] == "skipped" and second["passed"] is False
    markdown = run_wave_module._wave_markdown(second)
    assert "Verdict: **SKIPPED**" in markdown
    assert "fixture-embed" in markdown  # the skipped ids are named, not hidden


def test_the_corpus_step_carries_its_budget_and_log_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The corpus step is an ordinary harness step: it carries a declared wall-clock budget (from its
    request count: three passes over the rows plus the standing probes) and its start/end boundary lands
    on the pod log like every other step.  It used to bypass the step machinery (no budget, no watch, no
    log line), so a stuck request in this mode had no bound."""
    document = _wave(tmp_path, record_corpus=True)
    step = document["recipes"][0]["steps"]["observation_corpus"]
    assert step["state"] == "passed"
    assert step["budget_s"] >= 600  # the formula's base plus three passes over the rows and the probes
    assert step["secs"] >= 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run_wave: ")]
    assert "run_wave: fixture-embed observation_corpus start" in lines
    assert any(line.startswith("run_wave: fixture-embed observation_corpus passed ") for line in lines)
