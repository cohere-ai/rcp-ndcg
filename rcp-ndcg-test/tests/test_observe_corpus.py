"""The observation corpus on the collector's side: raw-first records, provenance, the acceptance checks.

Everything runs on CPU against the stub engine or on synthetic records: the record format and its repetitions,
the measured non-determinism (true repetitions only, volatile fields normalised away), the provenance blocks,
the acceptance checks (hashes, completeness, statuses, decoded vector widths, credentials, consistency with
the equivalence stage), the repository subset and the change handling.  The format's reader is the product's
(``rcp_ndcg_test.corpus``); these tests read every corpus back through it.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_test.observe.corpus import (
    build_record,
    expected_status,
    subset_for_repository,
    summarise_nondeterminism,
    verify_corpus,
    write_corpus,
)
from rcp_ndcg_test.record import record_corpus
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER, start_stub, token_ids_recipe


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
    from rcp_ndcg_test.observe.provenance import engine_facts

    return engine_facts(
        image=image,
        serve_argv=["stub_engine.py"],
        engine_python=None,
        environ={"VLLM_LOGGING_LEVEL": "INFO", "VLLM_API_KEY_SECRET": "never-recorded"},
        started="2026-10-07T00:00:00+00:00",
        ready_wait_s=1.5,
        runner=_no_gpu_runner,
    )


def test_the_engine_facts_record_the_patch_opt_in() -> None:
    """The engine environment block keeps the patch opt-in beside the VLLM_* knobs: the value the engine
    actually ran with is visible beside the behaviour fingerprint that keys the declared patches."""
    from rcp_ndcg_test.observe.provenance import engine_facts
    from rcp_ndcg_vllm.patches import PATCHES_ENV

    facts = engine_facts(
        image="registry.example.com/stub-engine:test",
        serve_argv=["stub_engine.py"],
        engine_python=None,
        environ={PATCHES_ENV: "pooling-full-context", "VLLM_LOGGING_LEVEL": "INFO", "VLLM_API_KEY_SECRET": "x"},
        started=None,
        ready_wait_s=None,
        runner=_no_gpu_runner,
    )
    assert facts["env"][PATCHES_ENV] == "pooling-full-context"
    assert "VLLM_API_KEY_SECRET" not in facts["env"]  # the secret filter still applies


def test_engine_facts_records_the_version_the_pod_reported() -> None:
    """B5: the provenance block's ``version`` is the running pod's version when the wave probed it, never
    the declared image's tag (a digest-pinned image used to record 64 hex characters as the version)."""
    from rcp_ndcg_test.observe.provenance import engine_facts

    facts = engine_facts(
        image="registry.example.com/engine:nightly-abc@sha256:" + "a" * 64,
        serve_argv=["stub_engine.py"],
        engine_python=None,
        environ={},
        started=None,
        ready_wait_s=None,
        version="0.31.0",
        runner=_no_gpu_runner,
    )
    assert facts["version"] == "0.31.0"
    assert facts["image"].startswith("registry.example.com/engine:nightly-abc@")


def test_the_standing_probes_carry_the_recipes_own_render_ids(tmp_path: Path) -> None:
    """A ``request_shape: token_ids`` recipe's standing probes carry the document-side render ids of the
    row's first document, and a stub engine that takes vLLM's token-id prompts accepts them.  The
    recipe's client is the referent: the probes' ``input`` equals the ids the client put on the wire for
    the same text (R30).  A bare text is a request the recipe's client never sends, and such an engine's
    pooler refuses it by name, which kills the EngineCore (the r2 wave's pplx-embed-v2-context failure)."""
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_test.equivalence import fitting
    from rcp_ndcg_test.equivalence.wire import role_client
    from rcp_ndcg_test.observe.adversarial import special_token_spellings
    from rcp_ndcg_test.observe.requests import corpus_plan
    from rcp_ndcg_test.record import _bare_probes, _Collector

    from rcp_ndcg.inference.types import EncodeRole

    recipe = token_ids_recipe(tmp_path)
    tokenizer = fitting.tokenizer_of(recipe)
    document = special_token_spellings(tokenizer)
    rows = [{"request_id": "row-0", "query": "a query", "documents": [document]}]
    plan = corpus_plan(recipe, tokenizer, rows)
    collected = _Collector(recipe, tokenizer)
    stub = start_stub("--tokenizer", str(TOKENIZER))
    try:
        _bare_probes(recipe, stub.base_url, plan.rows, plan.bare, collected, timeout_s=60.0)
    finally:
        stub.stop()
    probes = {str(record["inputs"]["probe"]): record for record in collected.records}
    # The client's own request for the same text, through the product's client (the referent).
    client, capture = role_client(recipe, None)
    client.encode([Content.from_text(document)], EncodeRole.DOCUMENT)
    sent = capture.exchanges[-1]["request_body"]["input"]
    assert isinstance(sent, list) and isinstance(sent[0], list), "the client's own wire carries ids"
    for probe in ("unknown_field", "wrong_model"):
        body = probes[probe]["request"]["body_parsed"]
        assert body["input"] == sent, probe
    empty = probes["empty_input"]["request"]["body_parsed"]["input"]
    assert isinstance(empty, list) and isinstance(empty[0], list), "the empty probe carries the empty render's ids"
    # The plan's bare wire row rides the same construction for the same text.
    assert probes["wire:encoding_format=float,embed_dtype=float16"]["request"]["body_parsed"]["input"] == sent
    # The stub (vLLM's token-id prompts) accepts the ids: the model answers instead of refusing the body.
    assert probes["wrong_model"]["response"]["status"] == 200
    assert probes["empty_input"]["response"]["status"] == 200
    assert probes["wire:encoding_format=float,embed_dtype=float16"]["response"]["status"] == 200


def test_a_token_ids_wires_model_layer_has_no_text_to_tokenize(tmp_path: Path) -> None:
    """A ``request_shape: token_ids`` wire's model layer puts no text on the wire: the record carries the
    ids as sent, and the corpus step probes no ``/tokenize`` prompt for them -- the probe's contract is the
    engine's tokenization of exactly what the client sent (R29), and there is none here."""
    from rcp_ndcg_test.equivalence import fitting
    from rcp_ndcg_test.record import _Collector, _model_layer

    recipe = token_ids_recipe(tmp_path)
    tokenizer = fitting.tokenizer_of(recipe)
    rows = [{"request_id": "row-0", "query": "a query", "documents": ["a document"]}]
    collected = _Collector(recipe, tokenizer)
    stub = start_stub("--tokenizer", str(TOKENIZER))
    try:
        sent = _model_layer(recipe, stub.base_url, rows, collected, (1,))
    finally:
        stub.stop()
    assert sent == [], "a token-ids wire carries no text; the ids ride the record's body"
    assert any(
        isinstance(record["request"]["body_parsed"].get("input", [None])[0], list)
        for record in collected.records
        if record["request"]["path"] == "/pooling"
    ), "the model layer sent the client's own ids"


def test_an_absent_after_restart_pass_fails_the_corpus(tmp_path: Path) -> None:
    """OBSERVATIONS-SPEC section 2 requires the third sending after an engine restart.  A corpus without it
    is refused, so a recipe can never be marked verified from it (the old acceptance checks all passed
    while the pass list said ``absent``)."""
    _, report = _record(tmp_path, after_restart=None, restart=lambda: None)
    assert report["passed"] is False
    check = _checks(report)["repetitions"]
    assert check["passed"] is False
    assert "after_restart" in json.dumps(check)


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
    from rcp_ndcg_test.corpus import load_corpus

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


def _synthetic_corpus(
    directory: Path, records: list[dict[str, Any]], *, recipe_id: str = "fixture-multi-vector", **manifest: Any
) -> Path:
    """A corpus of hand-built records with the stub's provenance, written through the collector's writer."""
    from rcp_ndcg_test.observe.provenance import collector_facts, model_facts, recipe_facts

    recipe = load_recipe(RECIPES / recipe_id)
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
    """The recipe block is keyed by ``rcp_ndcg_test.fingerprint`` (the one behaviour fingerprint, with its named
    inputs); every section-4 key holds a value or an explicit unavailable reason; no secret env var rides."""
    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.fingerprint import behaviour_fingerprint, fingerprint_inputs

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


def test_an_mrl_probes_reply_is_accepted_at_the_requested_width(tmp_path: Path) -> None:
    """E2 round 9 (the r3 wave's ``vector width 32 differs from 1024 on /v1/embeddings``): the corpus
    plan's MRL stratum sends one bare ``dimensions=k`` probe per declared ``k`` (requests.py
    ``_wire_variants``), and the engine's own Matryoshka gate cuts the reply to ``k``.  The check
    expected the recipe's full declared width on every vector row and tracked one width per route, so
    every MRL recipe's corpus failed on its own declared probes (the quoted mismatch is the per-route
    one; a recipe declaring ``client.dim`` gets the declared-width mismatch beside it).  This records a
    real MRL fixture end to end -- the stub slices every declared ``k``, the plan's probes included --
    and requires acceptance; the synthetic test below pins that a genuinely wrong width is still
    refused."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from rcp_ndcg_test.observe.requests import plan_recipe
    from rcp_ndcg_vllm.recipe import serve_argv

    recipe = load_recipe(RECIPES / "fixture-embed-mrl")
    plan = plan_recipe(recipe, tokenizer_of(recipe), {})
    rows = [{**row.to_pairs_row(), "request_id": f"pairs:{index}"} for index, row in enumerate(plan.rows)]
    argv = serve_argv(recipe, port=0, served_model_name=recipe.id)
    flags = ["127.0.0.1" if value == "0.0.0.0" else value for value in argv[argv.index(recipe.model) + 1 :]]
    first = start_stub("--tokenizer", str(TOKENIZER), *flags)
    second = start_stub("--tokenizer", str(TOKENIZER), *flags)
    try:
        report = record_corpus(
            recipe,
            first.base_url,
            rows,
            tmp_path / "corpus",
            server_run_id="run-1",
            engine_facts=_engine_facts(),
            after_restart=(second.base_url, "run-2"),
            batch_sizes=(1,),
        )
    finally:
        first.stop()
        second.stop()
    check = _checks(report)["scores_finite_and_vector_widths"]
    assert check["passed"] is True, check["mismatches"]
    probes = {
        record["inputs"]["request_id"]: len(record["response"]["body_parsed"]["data"][0]["embedding"])
        for record in _records(tmp_path / "corpus")
        if str(record["inputs"].get("request_id", "")).startswith("mrl:dimensions=")
        and record["response"]["status"] == 200
    }
    assert probes, "the plan's MRL stratum must have been recorded"
    assert probes["mrl:dimensions=2"] == 2 and probes["mrl:dimensions=4"] == 4 and probes["mrl:dimensions=8"] == 8


def test_a_reply_at_neither_the_requested_nor_the_declared_width_still_fails(tmp_path: Path) -> None:
    """The requested width is the expectation for a ``dimensions=k`` probe -- not a licence: a reply
    that is neither the requested ``k`` nor the declared width still fails the check, and the mismatch
    names the requested width."""
    declared = _synthetic_corpus(
        tmp_path / "ok", [_embedding_record(2, dimensions=2)], recipe_id="fixture-multi-vector"
    )
    assert _checks(verify_corpus(declared))["scores_finite_and_vector_widths"]["passed"] is True
    wrong = _synthetic_corpus(
        tmp_path / "wrong", [_embedding_record(6, dimensions=2)], recipe_id="fixture-multi-vector"
    )
    check = _checks(verify_corpus(wrong))["scores_finite_and_vector_widths"]
    assert check["passed"] is False, check
    assert any("requested 2" in mismatch for mismatch in check["mismatches"]), check["mismatches"]
    uncut = _synthetic_corpus(tmp_path / "uncut", [_embedding_record(6)], recipe_id="fixture-multi-vector")
    check = _checks(verify_corpus(uncut))["scores_finite_and_vector_widths"]
    assert check["passed"] is False and any("declared 8" in mismatch for mismatch in check["mismatches"])


def _embedding_record(width: int, *, dimensions: int | None = None) -> dict[str, Any]:
    """One ``/v1/embeddings`` exchange: an optional ``dimensions`` probe and a reply of the given width."""
    request: dict[str, Any] = {"model": "m", "input": ["a b c"], "encoding_format": "float"}
    if dimensions is not None:
        request["dimensions"] = dimensions
    reply = {"object": "list", "data": [{"index": 0, "embedding": [0.5] * width}]}
    return _exchange("/v1/embeddings", request, reply)


def _kept_rule_record(*, vectors: int, skip: list[int], ids: list[int]) -> dict[str, Any]:
    """One engine-side-rule ``/pooling`` exchange: the reply carries only the kept vectors while
    ``usage.prompt_tokens`` reports the full prompt, exactly the real engine's answer."""
    width = 8
    request = {
        "model": "m",
        "input": ["a b c d e f"],
        "task": "token_embed",
        "encoding_format": "float",
        "embed_dtype": "float16",
    }
    reply = {
        "object": "list",
        "data": [{"index": 0, "data": [[0.5] * width for _ in range(vectors)]}],
        "usage": {"prompt_tokens": len(ids), "total_tokens": len(ids)},
    }
    record = _exchange("/pooling", request, reply)
    record["inputs"].update({"side": "document", "sent_token_ids": [list(ids)]})
    return record


def _equivalence_exchange(record: dict[str, Any]) -> dict[str, Any]:
    """One corpus record as the equivalence stage's captured exchange of the same request (the same bytes,
    so the check pairs them by ``exchange_id``)."""
    request, response = record["request"], record["response"]
    return {
        "url": "http://engine" + request["path"],
        "method": request["method"],
        "request_bytes": base64.b64encode(json.dumps(request["body_parsed"]).encode("utf-8")).decode("ascii"),
        "request_body": request["body_parsed"],
        "status": response["status"],
        "headers": dict(response["headers"]),
        "response_bytes": base64.b64encode(json.dumps(response["body_parsed"]).encode("utf-8")).decode("ascii"),
        "response_json": response["body_parsed"],
    }


def test_a_kept_rule_reply_is_checked_against_the_declared_kept_count(tmp_path: Path) -> None:
    """pplx-embed-v2-late's engine-side document skip rule (E2 round-10: ``the /pooling reply reports 165
    prompt token(s) but its vectors decode to 152``): the served plugin drops the rule's positions, so the
    reply carries only the kept vectors while ``usage.prompt_tokens`` counts the whole prompt, and the
    product's adapter refused the decode when the corpus check declared no kept count.  The check reads the
    recipe's declared cut (the manifest's ``document_skip_engine_side``/``document_skip_token_ids``) and the
    engine's sent ids (the record's ``inputs.sent_token_ids``, the ``/tokenize`` probe's own ids) and
    expects the product's ``kept_vector_count``: a reply at the kept count passes; one at the full count
    (the engine ignored the rule) and one at a count that is neither fails."""
    from rcp_ndcg_test.observe.provenance import recipe_facts

    skip = [7, 9]
    ids = [1, 2, 7, 3, 9, 4]  # 6 prompt tokens, 4 kept
    recipe = load_recipe(RECIPES / "fixture-multi-vector")
    block = {**recipe_facts(recipe), "document_skip_engine_side": True, "document_skip_token_ids": skip}
    kept = _synthetic_corpus(tmp_path / "kept", [_kept_rule_record(vectors=4, skip=skip, ids=ids)], recipe=block)
    assert _checks(verify_corpus(kept))["scores_finite_and_vector_widths"]["passed"] is True
    full = _synthetic_corpus(tmp_path / "full", [_kept_rule_record(vectors=len(ids), skip=skip, ids=ids)], recipe=block)
    check = _checks(verify_corpus(full))["scores_finite_and_vector_widths"]
    assert check["passed"] is False, check
    assert any("declared keep-rule leaves 4 kept vector(s)" in mismatch for mismatch in check["mismatches"]), check
    neither = _synthetic_corpus(tmp_path / "neither", [_kept_rule_record(vectors=5, skip=skip, ids=ids)], recipe=block)
    check = _checks(verify_corpus(neither))["scores_finite_and_vector_widths"]
    assert check["passed"] is False and any("carries 5 token vector(s)" in m for m in check["mismatches"]), check
    # The equivalence stage's own exchange of the same request stays consistent: its candidate record carries
    # no /tokenize ids of its own, so the stored record's declared counts must decode BOTH sides (a candidate
    # decoded without them would lose its vectors and read as a mismatch).
    record = _kept_rule_record(vectors=4, skip=skip, ids=ids)
    consistent = _synthetic_corpus(tmp_path / "consistent", [record], recipe=block)
    check = _checks(verify_corpus(consistent, equivalence_exchanges=[_equivalence_exchange(record)]))[
        "equivalence_consistent"
    ]
    assert check["passed"] is True, check


def _per_chunk_record(*, vectors: int, ids: list[int]) -> dict[str, Any]:
    """One per-chunk ``/pooling`` exchange: the reply carries one row per chunk while ``usage.prompt_tokens``
    reports the whole prompt, exactly the real engine's answer for the per-chunk head."""
    width = 8
    request = {
        "model": "m",
        "input": [list(ids)],
        "task": "token_embed",
        "encoding_format": "float",
        "embed_dtype": "float16",
    }
    reply = {
        "object": "list",
        "data": [{"index": 0, "data": [[0.5] * width for _ in range(vectors)]}],
        "usage": {"prompt_tokens": len(ids), "total_tokens": len(ids)},
    }
    return _exchange("/pooling", request, reply)


def test_a_per_chunk_reply_is_checked_against_the_declared_chunk_count(tmp_path: Path) -> None:
    """pplx-embed-v2-context's per-chunk head (the rc0-202610130615 recording failure: ``the /pooling
    reply reports 9 prompt token(s) but its vectors decode to 1``): the served plugin pools one vector per
    document chunk, so a one-chunk document legitimately answers one vector.  The check reads the recipe's
    declared ``outputs: per_chunk`` and expects the product's own per-chunk count for each item (the
    plugin's segmentation over the sent ids): a reply at the chunk count passes; one at the prompt-token
    count (the engine ignored the per-chunk head) and one at a count that is neither fail, naming the
    expected count."""
    from rcp_ndcg_test.observe.provenance import recipe_facts
    from rcp_ndcg_vllm.models.pplx.chunk_rule import BOUNDARY_TOKEN_ID, DOCUMENT_PREFIX_TOKEN_IDS

    recipe = load_recipe(RECIPES / "fixture-multi-vector")
    block = {**recipe_facts(recipe), "outputs": "per_chunk"}
    # A two-chunk document: the prefix, the first chunk, the boundary marker, the second chunk.
    ids = [*DOCUMENT_PREFIX_TOKEN_IDS, 11, BOUNDARY_TOKEN_ID, 12, 13]
    chunks = _synthetic_corpus(tmp_path / "chunks", [_per_chunk_record(vectors=2, ids=ids)], recipe=block)
    assert _checks(verify_corpus(chunks))["scores_finite_and_vector_widths"]["passed"] is True
    full = _synthetic_corpus(tmp_path / "full", [_per_chunk_record(vectors=len(ids), ids=ids)], recipe=block)
    check = _checks(verify_corpus(full))["scores_finite_and_vector_widths"]
    assert check["passed"] is False, check
    assert any("declared per-chunk count 2" in mismatch for mismatch in check["mismatches"]), check
    neither = _synthetic_corpus(tmp_path / "neither", [_per_chunk_record(vectors=3, ids=ids)], recipe=block)
    check = _checks(verify_corpus(neither))["scores_finite_and_vector_widths"]
    assert check["passed"] is False, check
    assert any("decodes to 3 vector(s)" in mismatch for mismatch in check["mismatches"]), check
    # A token-per-position recipe is unchanged: the same reply at the prompt-token count passes when the
    # recipe declares no per-chunk head.
    tokenwise = _synthetic_corpus(
        tmp_path / "tokenwise", [_per_chunk_record(vectors=len(ids), ids=ids)], recipe=recipe_facts(recipe)
    )
    assert _checks(verify_corpus(tokenwise))["scores_finite_and_vector_widths"]["passed"] is True


def test_the_recorder_records_the_engine_ids_of_each_sent_render(tmp_path: Path) -> None:
    """The corpus carries the engine's own token ids of every sent render on the model-layer record
    (``inputs.sent_token_ids``), taken from the pass's ``/tokenize`` probes: the kept-count check needs them
    (and a repository subset, a sample, would otherwise drop the pairing and could not decode an
    engine-side-rule reply)."""
    directory, report = _record(tmp_path)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    records = _records(directory)
    tokenized = {
        record["request"]["body_parsed"]["prompt"]: record["response"]["body_parsed"]["tokens"]
        for record in records
        if record["inputs"].get("probe") == "tokenize" and record["response"].get("status") == 200
    }
    model = [
        record
        for record in records
        if record["inputs"].get("probe") == "ok"
        and record["inputs"].get("layer") == "model"
        and isinstance(record["request"].get("body_parsed"), dict)
        and isinstance(record["request"]["body_parsed"].get("input"), list)
    ]
    assert model, "the model layer must be recorded"
    checked = 0
    for record in model:
        texts = record["request"]["body_parsed"]["input"]
        ids = record["inputs"].get("sent_token_ids")
        assert isinstance(ids, list) and len(ids) == len(texts), record["inputs"]
        for text, sent in zip(texts, ids, strict=True):
            if text not in tokenized:
                assert sent is None  # no /tokenize probe for it: nothing is invented
                continue
            assert sent == tokenized[text], (record["inputs"]["request_id"], text)
            checked += 1
    assert checked, "at least one model-layer item must carry the engine's ids"


def test_the_equivalence_stage_must_have_seen_the_same_replies(tmp_path: Path) -> None:
    """Section 6: the equivalence stage's results for the same wave are consistent with the corpus."""
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_test.equivalence.wire import role_client

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
    from rcp_ndcg_test.corpus import load_corpus

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
    from rcp_ndcg_test.fingerprint import behaviour_fingerprint
    from rcp_ndcg_test.observe.corpus import changed_since

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
    from rcp_ndcg_test.corpus import load_corpus

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
    """``python -m rcp_ndcg_test.observe.corpus verify|subset``: exit 0 on an accepted corpus, 1 on a refused one."""
    from rcp_ndcg_test.observe.corpus import main

    directory, _ = _record(tmp_path)
    assert main(["verify", str(directory)]) == 0
    assert main(["subset", str(directory), str(tmp_path / "subset"), "--gcs-path", "gs://YOUR-BUCKET/o/x"]) == 0
    assert (tmp_path / "subset" / "index.json").is_file()
    (directory / "records.jsonl").write_text("", encoding="utf-8")
    assert main(["verify", str(directory)]) == 1
    assert '"passed": false' in capsys.readouterr().out


def test_the_written_manifest_never_hashes_the_verification_record(tmp_path: Path) -> None:
    """``verification.jsonl`` is appended after the recording: a corpus (re)written beside one leaves it out of
    the manifest's hashes, so a later append never breaks the integrity check."""
    from rcp_ndcg_test.corpus import VERIFICATION_FILE, integrity_mismatches, load_corpus

    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / VERIFICATION_FILE).write_text('{"schema": "rcp-ndcg.verification/1"}\n', encoding="utf-8")
    (directory / "nondeterminism.json").write_text('{"derived": {"measured": false}}', encoding="utf-8")
    record = {"record_schema": 1, "exchange_id": "0" * 64, "sequence": 0, "request": {}, "response": {}}
    manifest = json.loads(write_corpus(directory, {"recipe": {"id": "fixture-embed"}}, [record]).read_text())
    assert VERIFICATION_FILE not in manifest["integrity"]["files"]
    assert "nondeterminism.json" in manifest["integrity"]["files"]
    with (directory / VERIFICATION_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "rcp-ndcg.verification/1"}\n')
    assert integrity_mismatches(load_corpus(directory)) == []


def test_the_media_request_set_is_recorded_with_its_media(tmp_path: Path) -> None:
    """A media recipe's corpus records each media row with its image on the wire (one item per request, the
    chat-shaped route) and the media edges bare -- more images than ``max_images`` and an undecodable image,
    both refused by the engine."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from rcp_ndcg_test.observe.requests import plan_recipe
    from rcp_ndcg_vllm.recipe import serve_argv

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    plan = plan_recipe(recipe, tokenizer_of(recipe), {})
    rows = [{**row.to_pairs_row(), "request_id": f"pairs:{index}"} for index, row in enumerate(plan.rows)]
    rows = [row for row in rows if row.get("media")][:2] + [row for row in rows if not row.get("media")][:1]
    argv = serve_argv(recipe, port=0, served_model_name=recipe.id)
    flags = ["127.0.0.1" if value == "0.0.0.0" else value for value in argv[argv.index(recipe.model) + 1 :]]
    first = start_stub("--tokenizer", str(TOKENIZER), *flags)
    second = start_stub("--tokenizer", str(TOKENIZER), *flags)
    try:
        report = record_corpus(
            recipe,
            first.base_url,
            rows,
            tmp_path / "corpus",
            server_run_id="run-1",
            engine_facts=_engine_facts(),
            after_restart=(second.base_url, "run-2"),
            batch_sizes=(1,),
        )
    finally:
        first.stop()
        second.stop()
    records = _records(tmp_path / "corpus")
    media = [
        r
        for r in records
        if "image_url" in str(r["request"]["body_parsed"]) and r["inputs"].get("request_id", "").startswith("pairs:")
    ]
    assert media and all(r["response"]["status"] == 200 for r in media)
    edges = {
        r["inputs"]["request_id"]: r["response"]["status"]
        for r in records
        if r["inputs"].get("request_id", "").startswith("edge:")
    }
    assert edges["edge:too_many_images"] == 400 and edges["edge:corrupt_image"] == 400
    assert report["passed"] is True, [check for check in report["checks"] if not check.get("passed")]


def test_a_row_the_empty_policy_omits_is_recorded_as_its_omission(tmp_path: Path) -> None:
    """jina-reranker-v3's paper rule (``empty_doc: omit_zero_blank``): a whitespace-only document is never
    sent -- the client omits it and scores 0.0 without a model call -- so the plan's model-layer row for
    the ``content:whitespace_only`` kind (whose pairs row the stage-1 render check pruned) produced no
    exchange, and the corpus failed ``completeness`` with ``content:whitespace_only`` missing.  The
    omission is recorded like a client refusal -- the request id accounted for, nothing sent, no status --
    so the stratum is recorded and completeness holds."""
    recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    omitting = recipe.model_copy(update={"client": {**recipe.client, "empty_doc": "omit_zero_blank"}})
    rows = [
        {
            "request_id": "pairs:0",
            "stratum": "source:nanobeir",
            "query": "a query about retrieval",
            "documents": ["a document about retrieval"],
            "_source": {"suite": "nanobeir", "subset": "NanoNQRetrieval", "query_id": "q0"},
            "_strata": ["source:nanobeir"],
        }
    ]
    first = start_stub("--tokenizer", str(TOKENIZER))
    second = start_stub("--tokenizer", str(TOKENIZER))
    try:
        report = record_corpus(
            omitting,
            first.base_url,
            rows,
            tmp_path / "corpus",
            server_run_id="run-1",
            engine_facts=_engine_facts(),
            after_restart=(second.base_url, "run-2"),
            batch_sizes=(1,),
        )
    finally:
        first.stop()
        second.stop()
    records = _records(tmp_path / "corpus")
    omitted = [record for record in records if record["inputs"].get("probe") == "client_omission"]
    assert len(omitted) == 3, "one omission per pass (two in-process and one after the restart)"
    assert {record["inputs"]["request_id"] for record in omitted} == {"content:whitespace_only"}
    assert all(record["response"]["status"] is None for record in omitted)
    assert all("omit_zero_blank" in json.dumps(record["response"]) for record in omitted)
    assert report["passed"] is True, [check for check in report["checks"] if not check["passed"]]
    completeness = _checks(report)["completeness"]
    assert completeness["passed"] is True and completeness["missing"] == []
    # The kept row's model-layer request is recorded as before: the omission accounting never hides one.
    assert any(record["inputs"].get("request_id") == "pairs:0" for record in records)


def test_a_media_side_the_client_refuses_is_recorded_as_its_refusal(tmp_path: Path) -> None:
    """A client that refuses a media side (topk-embed-v1-small's under its skip ids; here a vision embedder
    left on the text route) records the refusal -- the request id accounted for, nothing sent, no status --
    instead of raising out of the corpus step and losing every text row with it; the corpus is accepted."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from rcp_ndcg_test.observe.requests import plan_recipe

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    text_route = recipe.model_copy(
        update={"client": {**recipe.client, "request_shape": "text", "add_generation_prompt": None}}
    )
    plan = plan_recipe(text_route, tokenizer_of(text_route), {})
    rows = [{**row.to_pairs_row(), "request_id": f"pairs:{index}"} for index, row in enumerate(plan.rows)]
    rows = [row for row in rows if row.get("media")][:1] + [row for row in rows if not row.get("media")][:1]
    first = start_stub("--tokenizer", str(TOKENIZER))
    second = start_stub("--tokenizer", str(TOKENIZER))
    try:
        report = record_corpus(
            text_route,
            first.base_url,
            rows,
            tmp_path / "corpus",
            server_run_id="run-1",
            engine_facts=_engine_facts(),
            after_restart=(second.base_url, "run-2"),
            batch_sizes=(1,),
        )
    finally:
        first.stop()
        second.stop()
    records = _records(tmp_path / "corpus")
    refused = [r for r in records if r["inputs"].get("probe") == "client_refusal"]
    assert len(refused) == 3, "one refusal per pass (two in-process and one after the restart)"
    assert all(r["response"]["status"] is None and "CapabilityError" in json.dumps(r["response"]) for r in refused)
    assert {r["inputs"]["request_id"] for r in refused} == {rows[0]["request_id"]}
    assert any(r["inputs"].get("request_id") == rows[1]["request_id"] for r in records), "the text row recorded"
    assert report["passed"] is True, [check for check in report["checks"] if not check.get("passed")]


# --- the repository corpus index (what a re-record replaces) ------------------------------------------------


def test_the_corpus_index_regenerates_from_the_committed_corpora(tmp_path: Path) -> None:
    """The repository's corpus index is generated from the committed corpora, never hand-edited.

    A re-record replaces a corpus directory; the index must follow it.  The tool writes one entry per
    corpus directory (``<recipe>/<fingerprint>``) with the facts the conformance suite's integrity
    check reads: the behaviour fingerprint, the SHA-256 of the committed manifest and the model
    revision.
    """
    import hashlib

    from rcp_ndcg_test.corpus import load_corpus, write_corpus_index

    engine = tmp_path / "vllm-0.31.0"
    records = [_exchange("/v1/embeddings", {"model": "m", "input": ["x"]}, {"object": "list", "data": []})]
    corpora = {
        "fixture-multi-vector": _synthetic_corpus(engine / "fixture-multi-vector" / ("a" * 64), records),
        "fixture-embed": _synthetic_corpus(engine / "fixture-embed" / ("b" * 64), records, recipe_id="fixture-embed"),
    }
    path = write_corpus_index(engine)
    index = json.loads(path.read_text(encoding="utf-8"))
    assert set(index["corpora"]) == {
        f"{recipe}/{fingerprint}"
        for recipe, fingerprint in [
            ("fixture-multi-vector", "a" * 64),
            ("fixture-embed", "b" * 64),
        ]
    }
    for recipe_id, directory in corpora.items():
        manifest = load_corpus(directory).manifest
        entry = index["corpora"][f"{recipe_id}/{directory.name}"]
        assert entry["behaviour_fingerprint"] == manifest["recipe"]["behaviour_fingerprint"]
        assert entry["manifest_sha256"] == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        assert entry["recipe_revision"] == manifest["model"]["revision"]

    # the file the tool writes is exactly the shape the conformance suite's integrity check reads
    assert load_corpus(corpora["fixture-embed"]).manifest["recipe"]["id"] == "fixture-embed"
