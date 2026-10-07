"""The observation corpus: raw-first records, the hash-chained manifest and the acceptance checks.

Everything here runs on CPU against the stub engine or on synthetic records: the corpus format, its
non-determinism report and derived tolerances, the behaviour fingerprint, the repository subset and
the change handling -- and the two mandated refusals: a credential-shaped string can never reach a
corpus, and a tampered file fails the manifest check.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from rcp_ndcg_vllm.observe.corpus import (
    CREDENTIAL_PATTERNS,
    DERIVED_TOLERANCE_RULE,
    behaviour_fingerprint,
    build_record,
    changed_since,
    expected_status,
    fingerprint_changes,
    subset_for_repository,
    summarise_nondeterminism,
    verify_corpus,
    write_corpus,
)
from rcp_ndcg_vllm.recipe import load_recipe
from rcp_ndcg_vllm.record import _entry_from_exchange, record_corpus

from tests.conftest import RECIPES, start_stub

_TOKENIZER_SHA = "ab" * 32


def _rows() -> list[dict]:
    return [
        {
            "request_id": "row-0",
            "stratum": "source:nanobeir",
            "query": "what is the capital of france",
            "documents": ["paris is the capital of france", "berlin is the capital of germany"],
            "_source": {"suite": "nanobeir", "subset": "NanoNQRetrieval", "query_id": "q0"},
            "_strata": ["source:nanobeir"],
        },
        {
            "request_id": "row-1",
            "stratum": "content:cjk",
            "query": "a query about \u65e5\u672c\u8a9e text",
            "documents": ["a document with combining e\u0301 accents"],
            "_source": {"suite": "synthetic", "content_kind": "cjk"},
            "_strata": ["content:cjk"],
        },
    ]


def _record(tmp_path: Path) -> Path:
    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(RECIPES.parent / "tokenizer.json"))
    try:
        report = record_corpus(
            recipe,
            engine.base_url,
            _rows(),
            tmp_path / "corpus",
            server_run_id="run-1",
            engine_facts={"version": "stub-1", "image": "stub"},
            after_restart_base_url=engine.base_url,
            batch_sizes=(1, 2),
        )
    finally:
        engine.stop()
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    return tmp_path / "corpus"


def test_the_corpus_records_repetitions_and_the_nondeterminism_report(tmp_path: Path) -> None:
    """Every request twice in one process and once after an engine restart; the measured deltas
    (non-determinism) travel beside the records with the derived tolerances (OBSERVATIONS-SPEC 2)."""
    directory = _record(tmp_path)
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records, "the corpus must carry raw records"
    assert all(record["record_schema"] == 1 for record in records)
    reps = {record["repetition"] for record in records}
    assert reps == {"same_process", "after_restart"}
    assert {record["server_run_id"] for record in records} == {"run-1"}
    assert all("id" not in record["request"]["headers"] for record in records)
    non = json.loads((directory / "nondeterminism.json").read_text(encoding="utf-8"))
    assert non["derived"]["rule"] == DERIVED_TOLERANCE_RULE
    assert non["derived"]["abs_tolerance"] >= 1e-9 and non["derived"]["rel_tolerance"] >= 1e-6
    assert non["per_input"], "the measured non-determinism must be stored per input"


def test_the_manifest_is_hash_chained_and_tampering_fails_the_check(tmp_path: Path) -> None:
    """A corpus's files are hashed into its manifest; a tampered file fails the manifest check."""
    directory = _record(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["integrity"]["manifest_sha256"]
    assert "records.jsonl" in manifest["integrity"]["files"]
    report = verify_corpus(directory)
    assert {check["check"]: check["passed"] for check in report["checks"]}["manifest_hashes"] is True
    records = directory / "records.jsonl"
    lines = records.read_text(encoding="utf-8").splitlines()
    # A subtle tamper: one MORE valid record (parseable, hash no longer matches the manifest).
    records.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    report = verify_corpus(directory)
    rows = {check["check"]: check for check in report["checks"]}
    assert rows["manifest_hashes"]["passed"] is False
    assert report["passed"] is False
    assert any("records.jsonl" in entry for entry in rows["manifest_hashes"]["mismatches"])
    # An unparseable tamper is refused by the same check, reported (never raised past the checks).
    records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = verify_corpus(directory)
    assert report["passed"] is False
    assert report["checks"][0]["passed"] is False


def test_no_credential_shaped_string_can_reach_a_corpus(tmp_path: Path) -> None:
    """Every corpus byte is scanned (OBSERVATIONS-SPEC 6) and authorisation headers never ride."""
    directory = _record(tmp_path)
    for path in sorted(directory.iterdir()):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", errors="replace")
            for pattern in CREDENTIAL_PATTERNS:
                assert pattern.search(text) is None, f"{path.name}: {pattern.pattern}"
    # The capture seam strips the headers that matter -- an authorisation header never reaches a record.
    request, response = _entry_from_exchange(
        {
            "url": "http://engine/v1/embeddings",
            "method": "POST",
            "request_bytes": "",
            "request_body": {"input": ["x"]},
            "status": 200,
            "headers": {"content-type": "application/json", "server": "uvicorn", "authorization": "Bearer x"},
            "response_bytes": "e30=",
            "response_json": {},
        }
    )
    assert set(request["headers"]) <= {"content-type"}
    assert set(response["headers"]) <= {"content-type", "server"}


def test_the_status_table_is_the_shake1c_measurement() -> None:
    """vLLM v0.31.0 (shakedown recordings): over-length refused 400, an unknown field ignored 200."""
    assert expected_status("/v1/embeddings", "ok", "0.31.0") == 200
    assert expected_status("/rerank", "over_length", "0.31.0") == 400
    assert expected_status("/v1/embeddings", "unknown_field", "0.31.0") == 200
    assert expected_status("/rerank", "unknown_field", "0.31.0") == 200
    assert expected_status("/v1/embeddings", "malformed_json", "0.31.0") == 400
    assert expected_status("/v1/embeddings", "wrong_model", "0.31.0") is None
    assert expected_status("/v1/embeddings", "ok", "9.9.9") is None, "an unmeasured engine answers no expectation"


def test_one_finite_score_and_uniform_vector_width(tmp_path: Path) -> None:
    """Scores must be finite and vectors one width per kind: a NaN or ragged width fails the check."""
    corpus = tmp_path / "corpus"
    ok = build_record(
        sequence=0,
        repetition="same_process",
        batch_context={"size": 1, "request_ids": ["r"], "positions": [0]},
        server_run_id="run",
        request={"method": "POST", "path": "/v1/embeddings", "headers": {}, "body_raw": "{}", "body_parsed": {}},
        response={
            "status": 200,
            "headers": {},
            "body_raw": "{}",
            "body_parsed": {"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 3}},
            "latency_s": 0.01,
        },
        inputs={"request_id": "r", "probe": "ok", "stratum": "ok", "token_counts": {}},
    )
    write_corpus(corpus, {"engine": {"version": "stub"}}, [ok, ok])
    (corpus / "nondeterminism.json").write_text(json.dumps(summarise_nondeterminism([ok, ok])), encoding="utf-8")
    assert verify_corpus(corpus)["passed"] is True
    broken = json.loads(json.dumps(ok))
    broken["response"]["body_parsed"] = {"data": [{"embedding": [float("nan"), 0.2]}]}
    broken["exchange_id"] = "other"
    write_corpus(corpus, {"engine": {"version": "stub"}}, [ok, broken])
    (corpus / "nondeterminism.json").write_text(json.dumps(summarise_nondeterminism([ok, broken])), encoding="utf-8")
    report = verify_corpus(corpus)
    rows = {check["check"]: check for check in report["checks"]}
    assert rows["scores_finite_and_vector_widths"]["passed"] is False


def test_nondeterminism_derives_its_tolerance_by_the_stored_rule() -> None:
    """Three sendings of one exchange, differing by 0.01: tolerance 3x the measurement (never hand-picked)."""

    def record(value: float, rep: str) -> dict:
        body_raw = json.dumps({"results": [{"relevance_score": value}]})
        return build_record(
            sequence=0,
            repetition=rep,
            batch_context={"size": 1, "request_ids": ["r"], "positions": [0]},
            server_run_id="run",
            request={"method": "POST", "path": "/rerank", "headers": {}, "body_raw": "{}", "body_parsed": {}},
            response={
                "status": 200,
                "headers": {},
                "body_raw": body_raw,
                "body_parsed": {"results": [{"relevance_score": value}]},
                "latency_s": 0.01,
            },
            inputs={"request_id": "r", "probe": "ok", "stratum": "ok", "token_counts": {}},
        )

    summary = summarise_nondeterminism(
        [record(0.50, "same_process"), record(0.51, "same_process"), record(0.50, "after_restart")]
    )
    assert abs(summary["summary"]["max_abs_diff"] - 0.01) < 1e-9
    assert abs(summary["derived"]["abs_tolerance"] - 0.03) < 1e-9
    assert summary["derived"]["bodies_differ"] is True
    assert summary["derived"]["rule"] == DERIVED_TOLERANCE_RULE


def test_the_behaviour_fingerprint_is_deterministic_and_names_its_changes() -> None:
    """GPU-VALIDATION 8: the fingerprint covers what can change what the model returns -- and a
    staleness failure names which input moved."""
    recipe = load_recipe(RECIPES / "fixture-embed")
    first = behaviour_fingerprint(recipe, tokenizer_sha256=_TOKENIZER_SHA, template_sha256=None)
    again = behaviour_fingerprint(recipe, tokenizer_sha256=_TOKENIZER_SHA, template_sha256=None)
    assert first == again
    assert len(first["fingerprint"]) == 64
    mutate = recipe.model_copy(update={"client": recipe.client.model_copy(update={"max_tokens": 127})})
    changed = behaviour_fingerprint(mutate, tokenizer_sha256=_TOKENIZER_SHA, template_sha256=None)
    assert changed["fingerprint"] != first["fingerprint"]
    moves = fingerprint_changes(first, changed)
    assert moves, "the staleness failure must name the input that changed"
    assert any("max_tokens" in move for move in moves), moves


def test_the_repository_subset_stays_within_its_budget_and_records_the_sampling(tmp_path: Path) -> None:
    directory = _record(tmp_path)
    out = subset_for_repository(directory, tmp_path / "subset", gcs_path="gs://YOUR-BUCKET/observations/one")
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["full_corpus"]["path"] == "gs://YOUR-BUCKET/observations/one"
    assert index["full_corpus"]["manifest_sha256"]
    assert index["sampling"]["rows"] > 0 and index["sampling"]["strata_covered"]
    with gzip.open(out / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == index["sampling"]["rows"]
    payload_bytes = (out / "records.jsonl.gz").stat().st_size
    assert payload_bytes <= 2 * 1024 * 1024


def test_changed_since_selects_only_the_changed_recipes() -> None:
    recipe = load_recipe(RECIPES / "fixture-embed")
    current = behaviour_fingerprint(recipe, tokenizer_sha256=_TOKENIZER_SHA, template_sha256=None)["fingerprint"]
    index = {"fingerprints": {recipe.id: current, "other-recipe": "stale"}}
    assert changed_since([recipe], index, tokenizer_sha256_of=lambda r: _TOKENIZER_SHA) == {
        "changed": [],
        "unchanged": [recipe.id],
    }
    index["fingerprints"][recipe.id] = "0" * 64
    assert changed_since([recipe], index, tokenizer_sha256_of=lambda r: _TOKENIZER_SHA) == {
        "changed": [recipe.id],
        "unchanged": [],
    }
