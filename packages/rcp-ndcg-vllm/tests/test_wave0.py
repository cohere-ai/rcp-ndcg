"""Wave 0 on CPU: the dry-mode plan, the report tool and schema, and every probe that runs in the
client environment - against the stub engine and a fake HF cache. Nothing here needs a GPU."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm.jobs.wave0_report import Wave0Report, wave0_report_schema

from tests.conftest import STUB, TOKENIZER

PY = sys.executable
WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_vllm" / "jobs" / "wave0.sh"
REPORT_PY = Path(__file__).resolve().parents[1] / "jobs" / "report.py"
WAVE0_HOST = Path(__file__).resolve().parents[1] / "jobs" / "wave0_host.py"

PINNED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --- the dry mode ---------------------------------------------------------------------------------------


def test_wave0_dry_mode_prints_the_plan_and_runs_nothing() -> None:
    completed = subprocess.run(
        ["bash", str(WAVE0_SH), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves/wave0"],
        capture_output=True,
        text=True,
        env={**os.environ, "WAVE0_DRY": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    plan = completed.stdout
    for marker in ("preflight", "host", "bootstrap", "reach", "engines", "embed", "evict", "stop"):
        assert marker in plan, marker
    assert f"Qwen/Qwen3-Embedding-0.6B@{PINNED_REVISION}" in plan  # the pinned revision from the research
    assert "gs://YOUR-BUCKET/rc0" in plan and "gs://YOUR-BUCKET/waves/wave0" in plan
    assert "nothing ran" in plan


def test_wave0_without_arguments_prints_usage() -> None:
    completed = subprocess.run(["bash", str(WAVE0_SH)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "usage" in completed.stderr


# --- report.py: the stdlib assembler every step merges into ---------------------------------------------


def test_report_tool_init_merge_fail_emit(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    subprocess.run(
        [PY, str(REPORT_PY), "init", "--file", str(report), "--schema", "rcp-ndcg.wave0-report.v1"], check=True
    )
    fragment = tmp_path / "fragment.json"
    fragment.write_text(json.dumps({"ok": True}), encoding="utf-8")
    subprocess.run(
        [PY, str(REPORT_PY), "merge", "--file", str(report), "--key", "host", "--fragment", str(fragment)], check=True
    )
    subprocess.run(
        [PY, str(REPORT_PY), "merge", "--file", str(report), "--key", "reach.hub", "--literal", '{"status": 200}'],
        check=True,
    )
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["host"] == {"ok": True}
    assert document["reach"]["hub"] == {"status": 200}
    subprocess.run(
        [PY, str(REPORT_PY), "fail", "--file", str(report), "--step", "reach", "--reason", "the Hub is down"],
        check=True,
    )
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["passed"] is False and document["failed_step"] == "reach" and document["error"] == "the Hub is down"
    emitted = subprocess.run(
        [PY, str(REPORT_PY), "emit", "--file", str(report)], capture_output=True, text=True, check=True
    )
    assert json.loads(emitted.stdout) == document


# --- wave0_host.py: step (a) on the image python, and the Hub probe's refusal ---------------------------


def test_wave0_host_collects_the_node_facts(tmp_path: Path) -> None:
    fragment_path = tmp_path / "host.json"
    completed = subprocess.run(
        [PY, str(WAVE0_HOST), "host", "--report", str(fragment_path), "--workdir", str(tmp_path / "wd")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    for key in ("image", "image_digest", "driver", "gpus", "free_disk_bytes", "shm_bytes", "python", "passed"):
        assert key in fragment, key
    assert fragment["passed"] is True


def test_wave0_host_measures_before_the_workdir_exists_without_creating_it(tmp_path: Path) -> None:
    """The host probe's disk check (node-runtime item 8): a not-yet-created path is measured at its
    nearest existing parent, and the probe is a measurement, not a mkdir."""
    workdir = tmp_path / "not" / "created" / "engines"
    fragment_path = tmp_path / "host.json"
    completed = subprocess.run(
        [PY, str(WAVE0_HOST), "host", "--report", str(fragment_path), "--workdir", str(workdir)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    assert fragment["free_disk_bytes"] > 0
    assert not workdir.exists()  # measured at the parent, never created


def test_wave0_host_hub_probe_refuses_without_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    completed = subprocess.run(
        [PY, str(WAVE0_HOST), "hub", "--model", "org/model", "--report", str(tmp_path / "hub.json")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "HF_TOKEN" in completed.stderr


# --- the embed check (e) against the stub engine ---------------------------------------------------------


def test_embed_check_fits_embeds_and_tokenizes(tmp_path: Path) -> None:
    """20 texts (5 over the 128-token budget) through fit + the product's wire path; /tokenize per input."""
    port = _free_port()
    engine = subprocess.Popen(
        [PY, str(STUB), "--port", str(port), "--tokenizer", str(TOKENIZER)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        report = tmp_path / "embed.json"
        completed = subprocess.run(
            [
                PY,
                "-m",
                "rcp_ndcg_vllm.jobs.wave0_probe",
                "embed",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--served-model-name",
                "fixture",
                "--tokenizer",
                str(TOKENIZER),
                "--budget",
                "128",
                "--count",
                "20",
                "--over",
                "5",
                "--report",
                str(report),
            ],  # fmt: skip
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fragment = json.loads(report.read_text(encoding="utf-8"))
    finally:
        engine.terminate()
        engine.wait(timeout=10)
    assert fragment["passed"], fragment
    assert fragment["n_inputs"] == 20 and fragment["n_over_length"] == 5
    assert fragment["client"]["n_vectors"] == 20 and fragment["client"]["finite"]
    check = fragment["tokenize_check"]
    assert check["passed"] is True and check["checked"] == 20
    over = [row for row in check["rows"] if row["kind"] == "over_length"]
    assert len(over) == 5 and all(row["cut"] and row["fit_tokens"] <= 128 for row in over)


# --- the eviction probe (f) on a fake cache ---------------------------------------------------------------


def test_evict_probe_on_a_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "hub"
    model_dir = cache / "models--org--model"
    (model_dir / "snapshots" / "rev").mkdir(parents=True)
    (model_dir / "snapshots" / "rev" / "weights.safetensors").write_bytes(b"0" * (2 << 20))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    report = tmp_path / "evict.json"
    completed = subprocess.run(
        [
            PY,
            "-m",
            "rcp_ndcg_vllm.jobs.wave0_probe",
            "evict",
            "--model",
            "org/model",
            "--model",
            "org/absent",
            "--report",
            str(report),
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    fragment = json.loads(report.read_text(encoding="utf-8"))
    assert fragment["passed"], fragment
    entries = {entry["model"]: entry for entry in fragment["models"]}
    assert entries["org/model"]["removed"] is True
    assert entries["org/model"]["freed_bytes"] >= 2 << 20
    assert entries["org/model"]["free_disk_before_bytes"] > 0
    assert entries["org/absent"]["removed"] is False and entries["org/absent"]["error"] is None


# --- the engine supervision (d) and the stop assert (g), with the stub engine as the stand-in ------------


def _stub_spec(tmp_path: Path, count: int = 2) -> Path:
    """Two slots' spec with the stub engine as the engine: distinct ports, tmpdirs and log dirs."""
    slots = []
    for slot in range(count):
        port = _free_port()
        directory = tmp_path / f"slot-{slot}"
        (directory / "tmp").mkdir(parents=True)
        spec = {
            "slot": slot,
            "model": f"stub-{slot}",
            "served_model_name": f"stub-{slot}",
            "argv": [
                PY,
                str(STUB),
                "--port",
                str(port),
                "--served-model-name",
                f"stub-{slot}",
                "--tokenizer",
                str(TOKENIZER),
            ],  # fmt: skip
            "cuda_visible_devices": str(slot),
            "port": port,
            "vllm_port": _free_port(),
            "tmpdir": str(directory / "tmp"),
            "log_dir": str(directory / "logs"),
        }
        (directory / "logs").mkdir(exist_ok=True)
        slots.append(spec)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"startup_timeout_s": 120, "slots": slots}, indent=2), encoding="utf-8")
    return spec_path


def test_engines_start_and_stop_probe(tmp_path: Path) -> None:
    """Two stub engines on two slots come up together, each answering its own name; stopping leaves none."""
    spec = _stub_spec(tmp_path)
    state = tmp_path / "engines.json"
    start_report = tmp_path / "engines-fragment.json"
    start = subprocess.run(
        [
            PY,
            "-m",
            "rcp_ndcg_vllm.jobs.wave0_probe",
            "engines-start",
            "--spec",
            str(spec),
            "--state",
            str(state),
            "--report",
            str(start_report),
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    assert start.returncode == 0, start.stdout + start.stderr
    fragment = json.loads(start_report.read_text(encoding="utf-8"))
    assert fragment["concurrent"] is True and fragment["passed"], fragment
    slots = fragment["slots"]
    assert {slot["slot"] for slot in slots} == {0, 1}
    assert slots[0]["models"] == ["stub-0"] and slots[1]["models"] == ["stub-1"]

    stop_report = tmp_path / "stop-fragment.json"
    stop = subprocess.run(
        [
            PY,
            "-m",
            "rcp_ndcg_vllm.jobs.wave0_probe",
            "engines-stop",
            "--state",
            str(state),
            "--report",
            str(stop_report),
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    assert stop.returncode == 0, stop.stdout + stop.stderr
    stop_fragment = json.loads(stop_report.read_text(encoding="utf-8"))
    assert stop_fragment["all_stopped"] is True and stop_fragment["scan_found"] == []
    assert stop_fragment["passed"] is True


# --- the report schema -------------------------------------------------------------------------------------


def _full_report(tmp_path: Path) -> Path:
    """A complete wave-0 report assembled through report.py, from fragments in the shipped shapes."""
    report = tmp_path / "wave0-report.json"
    fragments = {
        "host": {
            "image": "vllm/vllm-openai:v0.31.0",
            "image_digest": "sha256:" + "0" * 64,
            "driver": "580.0",
            "gpus": [{"index": 0, "name": "B200", "memory_total_mib": 183000}],
            "free_disk_bytes": 1 << 40,
            "free_disk_path": "/tmp/work",
            "shm_bytes": 128 << 30,
            "python": {"engine": "3.12.9"},
            "nvidia_smi_present": True,
            "collected": "2026-10-06T07:00:00Z",
            "passed": True,
        },
        "bootstrap": {
            "engine": {"python": "3.12.9", "vllm": "0.31.0", "freeze_unchanged": True, "plugins": []},
            "client": {
                "rcp-ndcg": "0.0.1",
                "rcp-ndcg-core": "0.0.1",
                "rcp-ndcg-vllm": "0.0.1",
                "mechanism": "uvx",
                "install_s": 12,
            },
            "reference": {"python": "3.12.9", "torch": "2.14.0", "transformers": "5.1.0"},
            "install_s": 90,
            "passed": True,
        },
        "reach.hub": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": PINNED_REVISION,
            "status": 200,
            "hub_sha": PINNED_REVISION,
            "pinned_match": True,
            "passed": True,
        },
        "reach.gcs": {
            "uri": "gs://YOUR-BUCKET/waves/wave0/wave0/probe/x",
            "wrote_bytes": 32,
            "listed": True,
            "read_equal": True,
            "deleted": True,
            "passed": True,
        },
        "engines": {
            "slots": [
                {
                    "slot": 0,
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "pid": 11,
                    "port": 8100,
                    "ready_after_s": 90.0,
                    "models": ["qwen3-embedding-0-6b"],
                    "engine_version": "0.31.0",
                    "served_model_name_expected": "qwen3-embedding-0-6b",
                },
                {
                    "slot": 1,
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "pid": 12,
                    "port": 8101,
                    "ready_after_s": 20.0,
                    "models": ["all-minilm-l6-v2"],
                    "engine_version": "0.31.0",
                    "served_model_name_expected": "all-minilm-l6-v2",
                },
            ],
            "concurrent": True,
            "isolation": {
                "0": {
                    "cuda_visible_devices": "0",
                    "port": 8100,
                    "vllm_port": 9200,
                    "tmpdir": "/tmp/w/s0",
                    "log_dir": "/tmp/w/l0",
                },
                "1": {
                    "cuda_visible_devices": "1",
                    "port": 8101,
                    "vllm_port": 9201,
                    "tmpdir": "/tmp/w/s1",
                    "log_dir": "/tmp/w/l1",
                },
            },
            "passed": True,
        },
        "embed": {
            "budget_tokens": 512,
            "n_inputs": 20,
            "n_over_length": 5,
            "cuts_recorded": 5,
            "overhead_tokens": 1,
            "client_budget_wired": True,
            "client": {"n_vectors": 20, "dim": 1024, "finite": True},
            "tokenize_check": {
                "checked": 20,
                "passed": True,
                "rows": [
                    {
                        "id": "0",
                        "kind": "short",
                        "fit_tokens": 9,
                        "engine_tokens": 9,
                        "ids_equal": True,
                        "cut": False,
                        "text_head": "The quick brown fox...",
                    }
                ],
            },
            "passed": True,
        },
        "evict": {
            "models": [
                {
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "cache_bytes": 1 << 30,
                    "removed": True,
                    "freed_bytes": 1 << 30,
                    "free_disk_before_bytes": 1 << 40,
                    "free_disk_after_bytes": (1 << 40) + (1 << 30),
                    "error": None,
                }
            ],
            "note": None,
            "passed": True,
        },
        "stop": {
            "pids": [11, 12],
            "leaked": [],
            "scan_found": [],
            "scan_all": [],
            "all_stopped": True,
            "free_disk_after_stop_bytes": 1 << 40,
            "passed": True,
        },
        "finished": {"at": "2026-10-06T08:00:00Z"},
    }
    subprocess.run(
        [PY, str(REPORT_PY), "init", "--file", str(report), "--schema", "rcp-ndcg.wave0-report.v1"], check=True
    )
    for key, value in fragments.items():
        subprocess.run(
            [PY, str(REPORT_PY), "merge", "--file", str(report), "--key", key, "--literal", json.dumps(value)],
            check=True,
        )
    return report


def test_wave0_report_validates_against_the_schema(tmp_path: Path) -> None:
    """The report the shell assembles is what the schema module declares."""
    document = json.loads(_full_report(tmp_path).read_text(encoding="utf-8"))
    parsed = Wave0Report.model_validate(document)
    assert parsed.passed is True
    assert parsed.embed is not None and parsed.embed.tokenize_check["checked"] == 20


def test_wave0_report_of_a_failed_run_validates(tmp_path: Path) -> None:
    """A fail-fast death: the envelope carries the step and the reason; later sections are absent."""
    report = tmp_path / "failed.json"
    subprocess.run(
        [PY, str(REPORT_PY), "init", "--file", str(report), "--schema", "rcp-ndcg.wave0-report.v1"], check=True
    )
    subprocess.run([PY, str(REPORT_PY), "fail", "--file", str(report), "--step", "bootstrap",
                    "--reason", "bootstrap.sh envs failed"], check=True)  # fmt: skip
    document = json.loads(report.read_text(encoding="utf-8"))
    parsed = Wave0Report.model_validate(document)
    assert parsed.passed is False and parsed.failed_step == "bootstrap" and parsed.embed is None


def test_wave0_schema_export_is_current(tmp_path: Path) -> None:
    """The committed schema file is the model's export."""
    schema_file = Path(__file__).resolve().parents[1] / "schema" / "wave0-report.schema.json"
    assert json.loads(schema_file.read_text(encoding="utf-8")) == wave0_report_schema()
