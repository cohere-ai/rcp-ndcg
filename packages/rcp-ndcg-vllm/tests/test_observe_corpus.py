"""The observation corpus on the collector's side: raw-first records, provenance, the acceptance checks.

Everything runs on CPU against the stub engine or on synthetic records: the record format and its repetitions,
the measured non-determinism (true repetitions only, volatile fields normalised away), the provenance blocks,
the acceptance checks (hashes, completeness, statuses, decoded vector widths, credentials, consistency with
the equivalence stage), the repository subset and the change handling.  The format's reader is the product's
(``rcp_ndcg.testing.corpus``); these tests read every corpus back through it.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_vllm.observe.corpus import (
    build_record,
    expected_status,
    subset_for_repository,
    summarise_nondeterminism,
    verify_corpus,
    write_corpus,
)
from rcp_ndcg_vllm.recipe import load_recipe
from rcp_ndcg_vllm.record import record_corpus

from tests.conftest import RECIPES, TOKENIZER, start_stub


def _rows() -> list[dict[str, Any]]:
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
            "query": "a query about 日本語 text",
            "documents": ["a document with combining é accents"],
            "_source": {"suite": "synthetic", "content_kind": "cjk"},
            "_strata": ["content:cjk"],
        },
    ]


def _no_gpu_runner(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
    """A subprocess runner for the provenance probes on a machine without GPUs or an engine environment."""
    return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]}: not found")


def _engine_facts(image: str = "registry.example.com/stub-engine:test") -> dict[str, Any]:
    """The engine block of a stub run: no vLLM version is claimed, so the measured vLLM status table (which
    the stub does not emulate for every probe) sets no expectation."""
    from rcp_ndcg_vllm.observe.provenance import engine_facts

    return engine_facts(
        image=image,
        serve_argv=["stub_engine.py"],
        engine_python=None,
        environ={"VLLM_LOGGING_LEVEL": "INFO", "VLLM_API_KEY_SECRET": "never-recorded"},
        started="2026-10-07T00:00:00+00:00",
        ready_wait_s=1.5,
        runner=_no_gpu_runner,
    )


def _record(tmp_path: Path, recipe_id: str = "fixture-embed", **overrides: Any) -> tuple[Path, dict[str, Any]]:
    """One corpus recorded against two stub engines (the second stands in for the restarted engine)."""
    recipe = load_recipe(RECIPES / recipe_id)
    first = start_stub("--tokenizer", str(TOKENIZER))
    second = start_stub("--tokenizer", str(TOKENIZER))
    try:
        kwargs: dict[str, Any] = {
            "server_run_id": "run-1",
            "engine_facts": _engine_facts(),
            "after_restart": (second.base_url, "run-2"),
            "batch_sizes": (1, 2),
        }
        kwargs.update(overrides)
        report = record_corpus(recipe, first.base_url, _rows(), tmp_path / "corpus", **kwargs)
    finally:
        first.stop()
        second.stop()
    return tmp_path / "corpus", report


def _records(directory: Path) -> list[dict[str, Any]]:
    from rcp_ndcg.testing.corpus import load_corpus

    return load_corpus(directory).records


def _checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["check"]: check for check in report["checks"]}


def _exchange(path: str, request: dict[str, Any], response: dict[str, Any], *, rep: str = "same_process_1") -> dict:
    return build_record(
        sequence=0,
        repetition=rep,
        batch_context={"size": 1, "request_ids": ["r"], "positions": [0]},
        server_run_id="run",
        request={
            "method": "POST",
            "path": path,
            "headers": {"content-type": "application/json"},
            "body_raw": json.dumps(request),
            "body_parsed": request,
        },
        response={
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body_raw": json.dumps(response),
            "body_parsed": response,
            "latency_s": 0.01,
        },
        inputs={"request_id": "r", "probe": "ok", "stratum": "ok", "token_counts": {}},
    )


def _synthetic_corpus(directory: Path, records: list[dict[str, Any]], **manifest: Any) -> Path:
    """A corpus of hand-built records with the stub's provenance, written through the collector's writer."""
    from rcp_ndcg_vllm.observe.provenance import collector_facts, model_facts, recipe_facts

    recipe = load_recipe(RECIPES / "fixture-multi-vector")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nondeterminism.json").write_text(json.dumps(summarise_nondeterminism(records)), encoding="utf-8")
    blocks = {
        "engine": _engine_facts(),
        "model": model_facts(recipe),
        "recipe": recipe_facts(recipe),
        "collector": collector_facts(wave_id="w", job_id="j", started="t0", finished="t1"),
        "plan": {"request_ids": ["r"]},
    }
    blocks.update(manifest)
    write_corpus(directory, blocks, records)
    return directory


# --- section 2: repetitions and the measured non-determinism -----------------------------------------------


def test_the_three_sendings_are_labelled_and_the_restart_is_its_own_run(tmp_path: Path) -> None:
    """Twice in one engine process (``same_process_1``, ``same_process_2``) and once after the restart, whose
    records carry the restarted engine's own ``server_run_id``."""
    directory, report = _record(tmp_path)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    records = _records(directory)
    by_repetition: dict[str, set[str]] = {}
    for record in records:
        by_repetition.setdefault(record["repetition"], set()).add(record["server_run_id"])
    assert by_repetition == {"same_process_1": {"run-1"}, "same_process_2": {"run-1"}, "after_restart": {"run-2"}}


def test_volatile_fields_never_count_as_nondeterminism() -> None:
    """Two sendings that differ only in the reply's request id and ``created`` timestamp are identical."""
    request = {"model": "m", "input": ["x"], "encoding_format": "float"}

    def reply(rid: str, created: int) -> dict[str, Any]:
        return {"id": rid, "created": created, "object": "list", "data": [{"index": 0, "embedding": [0.6, 0.8]}]}

    first = _exchange("/v1/embeddings", request, reply("embd-1", 1700000000))
    second = _exchange("/v1/embeddings", request, reply("embd-2", 1700000999), rep="same_process_2")
    summary = summarise_nondeterminism([first, second])
    assert summary["summary"]["max_abs_diff"] == 0.0
    assert summary["derived"]["bodies_differ"] is False


def test_no_repeated_request_means_no_tolerance() -> None:
    """Without a single request sent twice nothing was measured: the tolerance is declared unmeasured, not
    invented from the floors."""
    one = _exchange("/rerank", {"query": "a", "documents": ["b"]}, {"results": [{"index": 0, "relevance_score": 0.5}]})
    other = _exchange(
        "/rerank", {"query": "c", "documents": ["d"]}, {"results": [{"index": 0, "relevance_score": 0.9}]}
    )
    derived = summarise_nondeterminism([one, other])["derived"]
    assert derived["measured"] is False
    assert derived["abs_tolerance"] is None and derived["rel_tolerance"] is None


def test_the_tolerance_is_three_times_the_measured_repetition_drift() -> None:
    request = {"query": "q", "documents": ["d"]}

    def record(value: float, rep: str) -> dict[str, Any]:
        return _exchange("/rerank", request, {"results": [{"index": 0, "relevance_score": value}]}, rep=rep)

    summary = summarise_nondeterminism(
        [record(0.50, "same_process_1"), record(0.51, "same_process_2"), record(0.50, "after_restart")]
    )
    assert abs(summary["summary"]["max_abs_diff"] - 0.01) < 1e-9
    assert abs(summary["derived"]["abs_tolerance"] - 0.03) < 1e-9
    assert summary["derived"]["measured"] is True and summary["derived"]["bodies_differ"] is True


# --- section 4: provenance ------------------------------------------------------------------------------------


def test_the_manifest_carries_the_provenance_and_the_one_behaviour_fingerprint(tmp_path: Path) -> None:
    """The recipe block is keyed by ``rcp_ndcg_vllm.fingerprint`` (the one behaviour fingerprint, with its named
    inputs); every section-4 key holds a value or an explicit unavailable reason; no secret env var rides."""
    from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint, fingerprint_inputs

    from rcp_ndcg.testing.corpus import load_corpus

    directory, report = _record(tmp_path)
    assert _checks(report)["provenance_complete"]["passed"] is True
    manifest = load_corpus(directory).manifest
    recipe = load_recipe(RECIPES / "fixture-embed")
    assert manifest["recipe"]["behaviour_fingerprint"] == behaviour_fingerprint(recipe)
    assert manifest["recipe"]["fingerprint_inputs"] == fingerprint_inputs(recipe)
    assert manifest["engine"]["env"] == {"VLLM_LOGGING_LEVEL": "INFO"}
    assert manifest["engine"]["gpus"] == {"unavailable": "nvidia-smi exited 127: nvidia-smi: not found"}
    assert len(manifest["collector"]["host_sha256"]) == 64
    assert manifest["model"]["tokenizer_sha256"] == manifest["recipe"]["fingerprint_inputs"]["tokenizer_sha256"]


def test_a_manifest_missing_a_provenance_fact_fails_the_check(tmp_path: Path) -> None:
    """A fact that is neither recorded nor explained (here: the engine's image digest) fails acceptance."""
    request = {"model": "m", "input": ["x"], "encoding_format": "float"}
    reply = {"object": "list", "data": [{"index": 0, "embedding": [0.6, 0.8]}]}
    engine = {key: value for key, value in _engine_facts().items() if key != "image_digest"}
    directory = _synthetic_corpus(tmp_path / "c", [_exchange("/v1/embeddings", request, reply)], engine=engine)
    check = _checks(verify_corpus(directory))["provenance_complete"]
    assert check["passed"] is False and "engine.image_digest" in check["missing"]


# --- section 6: the acceptance checks --------------------------------------------------------------------------


def test_the_collector_checks_completeness_against_its_plan(tmp_path: Path) -> None:
    """The collector writes its plan into the manifest and the check runs on it: a planned request the corpus
    never sent fails acceptance (it is never silently missing)."""
    _, report = _record(tmp_path / "ok")
    assert _checks(report)["completeness"]["passed"] is True
    _, short = _record(tmp_path / "short", plan_ids=["row-0", "row-1", "row-never-sent"])
    completeness = _checks(short)["completeness"]
    assert completeness["passed"] is False and completeness["missing"] == ["row-never-sent"]
    assert short["passed"] is False


def test_an_authorization_header_anywhere_fails_the_credential_check(tmp_path: Path) -> None:
    request = {"model": "m", "input": ["Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789"]}
    reply = {"object": "list", "data": [{"index": 0, "embedding": [0.6, 0.8]}]}
    directory = _synthetic_corpus(tmp_path / "c", [_exchange("/v1/embeddings", request, reply)])
    check = _checks(verify_corpus(directory))["no_credentials"]
    assert check["passed"] is False and any("authorization" in hit for hit in check["hits"])


def _pooling_record(width: int, *, dtype: str = "float16") -> dict[str, Any]:
    frame = np.arange(3 * width, dtype=np.float16 if dtype == "float16" else np.float32) / 10.0
    request = {
        "model": "m",
        "input": ["a b c"],
        "task": "token_embed",
        "encoding_format": "base64",
        "embed_dtype": dtype,
    }
    reply = {
        "object": "list",
        "data": [{"index": 0, "object": "pooling", "data": base64.b64encode(frame.tobytes()).decode("ascii")}],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    return _exchange("/pooling", request, reply)


def test_a_base64_pooling_reply_is_decoded_and_checked_at_the_declared_width(tmp_path: Path) -> None:
    """A base64 ``/pooling`` frame carries its numbers only once decoded: decoded by the product's adapter at
    the recipe's declared ``dim`` (8 for fixture-multi-vector), a frame of another width fails the check."""
    good = _synthetic_corpus(tmp_path / "good", [_pooling_record(8)])
    assert _checks(verify_corpus(good))["scores_finite_and_vector_widths"]["passed"] is True
    bad = _synthetic_corpus(tmp_path / "bad", [_pooling_record(6)])
    check = _checks(verify_corpus(bad))["scores_finite_and_vector_widths"]
    assert check["passed"] is False, check


def test_the_equivalence_stage_must_have_seen_the_same_replies(tmp_path: Path) -> None:
    """Section 6: the equivalence stage's results for the same wave are consistent with the corpus."""
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_vllm.equivalence.wire import role_client

    from rcp_ndcg.inference.types import EncodeRole

    directory, _ = _record(tmp_path)
    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    noisy = start_stub("--tokenizer", str(TOKENIZER), "--noise", "0.3")
    try:
        exchanges: list[list[dict[str, Any]]] = []
        for url in (engine.base_url, noisy.base_url):
            client, capture = role_client(recipe, url)
            client.encode([Content.from_text("paris is the capital of france")], EncodeRole.DOCUMENT)
            exchanges.append(list(capture.exchanges))
    finally:
        engine.stop()
        noisy.stop()
    assert _checks(verify_corpus(directory, equivalence_exchanges=exchanges[0]))["equivalence_consistent"]["passed"]
    check = _checks(verify_corpus(directory, equivalence_exchanges=exchanges[1]))["equivalence_consistent"]
    assert check["passed"] is False and check["compared"] == 1, check
    nothing = _checks(verify_corpus(directory, equivalence_exchanges=[]))["equivalence_consistent"]
    assert nothing["passed"] is False, "no common request is never a vacuous pass"


def test_a_tampered_corpus_file_fails_acceptance(tmp_path: Path) -> None:
    directory, _ = _record(tmp_path)
    records = directory / "records.jsonl"
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    report = verify_corpus(directory)
    assert _checks(report)["manifest_hashes"]["passed"] is False and report["passed"] is False


def test_a_status_off_the_measured_table_fails_acceptance(tmp_path: Path) -> None:
    """A corpus that claims vLLM v0.31.0 and records an unknown field refused with 400 contradicts the
    shakedown's measurement (ignored, 200): the statuses check fails it."""
    record = _exchange("/v1/embeddings", {"model": "m", "input": ["x"], "unknown_field": 1}, {"error": {}})
    record["response"]["status"] = 400
    record["inputs"]["probe"] = "unknown_field"
    engine = _engine_facts(image="vllm/vllm-openai:v0.31.0")
    directory = _synthetic_corpus(tmp_path / "c", [record], engine=engine)
    check = _checks(verify_corpus(directory))["statuses_as_expected"]
    assert check["passed"] is False and check["mismatches"][0]["expected"] == 200


def test_the_status_table_is_the_shakedown_measurement() -> None:
    """vLLM v0.31.0 (shakedown recordings): over-length refused 400, an unknown field ignored 200."""
    assert expected_status("/v1/embeddings", "ok", "0.31.0") == 200
    assert expected_status("/rerank", "over_length", "0.31.0") == 400
    assert expected_status("/v1/embeddings", "unknown_field", "0.31.0") == 200
    assert expected_status("/v1/embeddings", "malformed_json", "0.31.0") == 400
    assert expected_status("/v1/embeddings", "wrong_model", "0.31.0") is None
    assert expected_status("/v1/embeddings", "ok", "9.9.9") is None, "an unmeasured engine answers no expectation"


# --- section 5: the repository subset ---------------------------------------------------------------------------


def test_the_repository_subset_covers_every_stratum_and_reads_back(tmp_path: Path) -> None:
    """A stratified sample: the first record of EVERY stratum is in, however the strata are ordered, and the
    subset reads back and verifies through the product's reader (its index hashes its files)."""
    from rcp_ndcg.testing.corpus import load_corpus

    request = {"model": "m", "input": ["x"], "encoding_format": "float"}
    reply = {"object": "list", "data": [{"index": 0, "embedding": [0.6, 0.8]}]}
    records = []
    for index, stratum in enumerate(["source:a"] * 4 + ["source:b"]):
        record = _exchange("/v1/embeddings", {**request, "input": [f"x{index}"]}, reply)
        record["inputs"]["stratum"] = stratum
        records.append(record)
    directory = _synthetic_corpus(tmp_path / "full", records, plan={"request_ids": ["r"]})
    out = subset_for_repository(
        directory, tmp_path / "subset", gcs_path="gs://YOUR-BUCKET/observations/x", per_recipe_bytes=900
    )
    subset = load_corpus(out)
    assert {row["inputs"]["stratum"] for row in subset.records} == {"source:a", "source:b"}
    report = verify_corpus(out)
    assert _checks(report)["manifest_hashes"]["passed"] is True, report


# --- section 7: change handling ---------------------------------------------------------------------------------


def test_changed_since_keys_on_the_fingerprint_and_the_engine_version() -> None:
    """The corpus key is (engine version, behaviour fingerprint): either moving records the recipe again."""
    from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint
    from rcp_ndcg_vllm.observe.corpus import changed_since

    recipe = load_recipe(RECIPES / "fixture-embed")
    index = {"fingerprints": {recipe.id: behaviour_fingerprint(recipe)}, "engine_versions": {recipe.id: "0.31.0"}}
    same = changed_since([recipe], index, engine_version_of=lambda _: "0.31.0")
    assert same["changed"] == [] and same["unchanged"] == [recipe.id] and same["protocol_due"] == []
    new_engine = changed_since([recipe], index, engine_version_of=lambda _: "0.32.0")
    assert new_engine["changes"] == {recipe.id: ["engine_version"]} and new_engine["protocol_due"] == ["0.32.0"]
    index["fingerprints"][recipe.id] = "0" * 64
    moved = changed_since([recipe], index, engine_version_of=lambda _: "0.31.0")
    assert moved["changes"] == {recipe.id: ["behaviour_fingerprint"]}


# --- section 1: the request set the collector sends -----------------------------------------------------------


def test_the_engine_tokenizes_exactly_the_prompt_the_client_sent(tmp_path: Path) -> None:
    """``/tokenize`` gets the rendered prompt the client put on the wire (fixture-embed's ``doc: ... [END]``
    frame), never the raw row text."""
    directory, report = _record(tmp_path)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    prompts = [
        record["request"]["body_parsed"]["prompt"]
        for record in _records(directory)
        if record["inputs"].get("probe") == "tokenize"
    ]
    assert prompts and all(prompt.startswith("doc: ") and prompt.endswith(" [END]") for prompt in prompts), prompts[:3]


def test_every_planned_request_is_recorded_the_ladder_uncut_too(tmp_path: Path) -> None:
    """The plan's ladder, long content kinds, wire variants and edges are all in the corpus (completeness), the
    uncut ladder exactly as planned."""
    from rcp_ndcg.testing.corpus import load_corpus

    directory, report = _record(tmp_path)
    assert _checks(report)["completeness"]["passed"] is True
    corpus = load_corpus(directory)
    planned = set(corpus.manifest["plan"]["request_ids"])
    assert {"ladder:over_10x", "ladder:over_10x:uncut", "wire:encoding_format=base64"} <= planned
    uncut = [r for r in corpus.records if r["inputs"]["request_id"] == "ladder:over_10x:uncut"]
    assert len(uncut) == 3, "sent once per pass"
    strata = corpus.manifest["plan"]["strata"]
    assert strata["edge:while_loading"]["present"] is False and strata["edge:while_loading"]["reason"]


def test_the_operator_commands_verify_and_subset_a_corpus(tmp_path: Path, capsys: Any) -> None:
    """``python -m rcp_ndcg_vllm.observe.corpus verify|subset``: exit 0 on an accepted corpus, 1 on a refused one."""
    from rcp_ndcg_vllm.observe.corpus import main

    directory, _ = _record(tmp_path)
    assert main(["verify", str(directory)]) == 0
    assert main(["subset", str(directory), str(tmp_path / "subset"), "--gcs-path", "gs://YOUR-BUCKET/o/x"]) == 0
    assert (tmp_path / "subset" / "index.json").is_file()
    (directory / "records.jsonl").write_text("", encoding="utf-8")
    assert main(["verify", str(directory)]) == 1
    assert '"passed": false' in capsys.readouterr().out
