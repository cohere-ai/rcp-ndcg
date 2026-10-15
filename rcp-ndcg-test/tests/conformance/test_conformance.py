"""The conformance suite: every recorded exchange replays against its emulator (GPU-VALIDATION.md
items 2-7), staleness fails with the changed inputs named, and the registry refuses what it was not
verified against.

Every corpus under ``rcp-ndcg-test/corpora/`` is loaded through the corpus seam, its emulator is built
through ``tests._engines`` (the recipe's real tokenizer and prompt derivation), and **every recorded
exchange** is replayed and compared within the recorded non-determinism -- the measured bounds every
committed corpus declares (``measured: true``), exactly when a corpus measured none, and never over zero
comparisons (a corpus with no answered exchange is a named problem). The suite runs in CI on every push
and needs no network.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from rcp_ndcg_test.corpus import credential_findings, integrity_mismatches, load_corpus, verification_records
from rcp_ndcg_test.engines import compare_exchange, corpus_tolerance, exchanges_of, find_corpora
from tests._engines import ROOT, corpus_of, current_corpora, emulator_for, load_recipe, stale_corpora

ENGINES_ROOT = ROOT / "corpora"
INDEX = ENGINES_ROOT / "vllm-0.31.0" / "index.json"
WAIVERS = Path(__file__).resolve().parent / "waivers.json"


def corpus_dirs() -> list[Path]:
    """Every committed corpus directory the replay runs on: found by scanning manifests, the ones declared
    stale for re-recording (``tests/conformance/stale.json``) left out -- those must fail the staleness gate
    by name instead (:func:`test_a_declared_stale_corpus_fails_the_staleness_gate_by_name`)."""
    return current_corpora()


def all_corpus_dirs() -> list[Path]:
    """Every committed corpus directory, stale or not (integrity and credential scans cover them all)."""
    return find_corpora(ENGINES_ROOT)


def replay_problems(directory: Path) -> tuple[object, list[str]]:
    """Replay every recorded exchange of one corpus against its emulator: the conformance differences
    (identical status and body within the recorded non-determinism), each naming the record. A record
    whose request got no answer (no recorded status: a connection error, a client refusal, a declared
    omission) is not an exchange: there is nothing to replay, so it is left to the plan's completeness
    check instead. A corpus with no answered exchange at all is a NAMED problem, never a pass: zero
    comparisons replay "identically" and the verification record compares equal (both sides compute
    ``exchanges: 0``), so a measurement that never happened would read as a verified corpus -- the same
    non-vacuity rule the listwise sibling carries."""
    corpus = load_corpus(directory)
    answered = [exchange for exchange in exchanges_of(corpus) if exchange.answered]
    if not answered:
        return corpus, ["the corpus holds no answered exchange: the replay would pass vacuously"]
    emulator = emulator_for(corpus.manifest["recipe"]["id"])
    problems = []
    for exchange in answered:
        answer = emulator.handle(as_request(exchange))
        found = compare_exchange(exchange, answer, corpus_tolerance(corpus))
        problems.extend(f"#{exchange.sequence}: {problem}" for problem in found)
    return corpus, problems


def as_request(exchange) -> httpx.Request:
    """The recorded request as the emulator's HTTP surface reads it: the exact bytes where the corpus
    recorded them, else the parsed body encoded as JSON."""
    if exchange.request_raw is not None:
        content = exchange.request_raw
    elif exchange.request_body is None:
        content = b""
    else:
        content = json.dumps(exchange.request_body).encode("utf-8")
    headers = {"content-type": "application/json"} if content else {}
    return httpx.Request(exchange.method, f"http://engine{exchange.path}", content=content, headers=headers)


def test_every_recorded_exchange_replays_identically() -> None:
    """The conformance core: identical status and body within the recorded non-determinism."""
    for directory in corpus_dirs():
        _, problems = replay_problems(directory)
        assert not problems, f"{directory.parent.name}: replay drifted:\n" + "\n".join(problems)


def test_a_corpus_with_no_answered_exchange_fails_the_replay_by_name(tmp_path: Path) -> None:
    """The replay is non-vacuous: a corpus whose records all carry no response status replayed
    "identically" with zero comparisons, and its verification record compared equal (both sides compute
    ``exchanges: 0``), so a measurement that never happened would read as a pass.  Such a corpus is
    writable by the real pipeline -- an empty pairs file gives an empty plan, and the completeness check
    passes on an empty plan -- so the vacuity is a named problem here, like the listwise sibling's
    non-vacuity guard, and reaches the wave's record and the conformance report."""
    import shutil

    frozen = Path(__file__).resolve().parents[1] / "fixtures" / "observation_corpus_v1"
    directory = tmp_path / "no-answer"
    shutil.copytree(frozen, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A recipe the emulator registry can build, so the only thing under test is the vacuity guard.
    manifest["recipe"]["id"] = "qwen3-reranker-0.6b"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records = directory / "records.jsonl"
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        row["response"]["status"] = None  # the collector recorded that nothing crossed the wire
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _, problems = replay_problems(directory)
    assert problems == ["the corpus holds no answered exchange: the replay would pass vacuously"], problems


def test_usage_counts_use_the_recipes_real_tokenizer() -> None:
    """The replayed ``usage.prompt_tokens`` is the engine's own count (the recipe's real tokenizer
    files): the recorded values reproduce exactly (measured: the sum of each engine prompt's tokens).
    ``/models``, ``/tokenize`` and ``/health`` are the routes whose replies carry no ``usage`` (a model
    listing, a count and a readiness answer), so the check covers the routes that do."""
    for directory in corpus_dirs():
        corpus = load_corpus(directory)
        emulator = emulator_for(corpus.manifest["recipe"]["id"])
        for exchange in exchanges_of(corpus):
            if exchange.status != 200 or exchange.path.endswith(("/models", "/tokenize", "/health")):
                continue
            recorded = (exchange.response_json or {}).get("usage", {}).get("prompt_tokens")
            answered = emulator.answer(exchange.path, exchange.method, exchange.request_body).json()
            assert answered["usage"]["prompt_tokens"] == recorded, f"{directory.name} #{exchange.sequence}"


def test_a_tampered_corpus_fails_the_hash_check(tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "corpus"
    shutil.copytree(corpus_dirs()[0], copy)
    assert integrity_mismatches(load_corpus(copy)) == []
    import gzip

    records = copy / load_corpus(copy).records_file
    text = gzip.decompress(records.read_bytes()).decode("utf-8")
    records.write_bytes(gzip.compress(text.replace('"status": 200', '"status": 201', 1).encode("utf-8")))
    assert integrity_mismatches(load_corpus(copy)) == [f"{records.name}: hash mismatch"]


def test_manifest_and_index_hashes_hold() -> None:
    """Integrity (OBSERVATIONS-SPEC section 4): every corpus file hashes to its subset index, the manifest is
    the full corpus's, and every manifest hashes to the repository's corpus index."""
    import hashlib

    corpora = json.loads(INDEX.read_text(encoding="utf-8"))["corpora"]
    assert len(corpora) == len(all_corpus_dirs())
    for directory in all_corpus_dirs():
        corpus = load_corpus(directory)
        assert integrity_mismatches(corpus) == [], directory
        key = f"{directory.parent.name}/{directory.name}"
        actual = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        assert actual == corpora[key]["manifest_sha256"], f"{key}: the corpus index declares another manifest"
        fingerprint = corpus.manifest["recipe"]["behaviour_fingerprint"]
        assert corpora[key]["behaviour_fingerprint"] == fingerprint == directory.name, key


def test_no_credential_shaped_string_is_in_any_corpus() -> None:
    """Acceptance (OBSERVATIONS-SPEC section 6): no credential-shaped string in any committed corpus byte
    (``rcp_ndcg_test.corpus.credential_findings``) -- and the scan finds a planted one."""
    import gzip

    scanned = 0
    for path in sorted(ENGINES_ROOT.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".gz", ".jsonl"):
            raw = path.read_bytes()
            text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8", errors="replace")
            assert credential_findings(text) == [], path
            scanned += 1
    assert scanned >= 5 * len(all_corpus_dirs())
    assert credential_findings('{"headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwx"}}')
    assert credential_findings('{"authorization":12345}') == []  # a tokenizer vocabulary entry


def test_staleness_passes_for_the_unmoved_recipes() -> None:
    """Every committed corpus not declared stale is keyed by a fingerprint the repository reproduces exactly
    -- or a dated, unexpired waiver covers exactly what moved (the release checklist requires the file
    empty). A corpus that goes stale without being declared fails here, naming the inputs that moved. A
    lane that changes every recipe's behaviour inputs leaves every corpus stale until the wave re-records:
    the guard then is that the current set plus the declared stale set covers every recorded corpus, so no
    corpus is *silently* skipped (each declaration is still checked exactly by
    ``test_a_declared_stale_corpus_fails_the_staleness_gate_by_name``, and the release rule -- the stale
    list is empty at release -- is enforced by the release checklist)."""
    import datetime

    from rcp_ndcg_test.changes import recipe_state, waiver_covers

    waivers = json.loads(WAIVERS.read_text(encoding="utf-8"))
    today = datetime.date.today()
    recorded = {load_corpus(directory).manifest["recipe"]["id"] for directory in all_corpus_dirs()}
    # The replay set computed independently of stale.json: a corpus replays when the recipe's recomputed
    # fingerprint still equals its recorded one. The union guard then catches a corpus that is neither
    # replayable nor declared stale (a silent skip), even when stale.json itself is the thing under test.
    replayable = {
        load_corpus(directory).manifest["recipe"]["id"]
        for directory in all_corpus_dirs()
        if recipe_state(load_recipe(load_corpus(directory).manifest["recipe"]["id"]), ENGINES_ROOT / "vllm-0.31.0")[
            "state"
        ]
        == "unchanged"
    }
    assert replayable | set(stale_corpora()) == recorded, (
        "a recorded corpus is neither replayable nor declared stale: "
        f"{sorted(recorded - replayable - set(stale_corpora()))}"
    )
    assert replayable or stale_corpora(), "no committed corpus at all"
    for directory in corpus_dirs():
        recipe_id = load_corpus(directory).manifest["recipe"]["id"]
        state = recipe_state(load_recipe(recipe_id), ENGINES_ROOT / "vllm-0.31.0")
        changed = list(state["changed_inputs"])
        if state["state"] != "unchanged" and any(
            waiver_covers(waiver, recipe_id, changed, today=today) for waiver in waivers
        ):
            continue  # a dated, unexpired, recipe-specific waiver is the only way past staleness
        assert state["state"] == "unchanged", (
            f"{recipe_id}: stale corpus, changed fingerprint inputs {changed} -- re-record the recipe "
            "(python -m rcp_ndcg_test.changes changed), declare it in tests/conformance/stale.json, or add a "
            "dated entry to tests/conformance/waivers.json"
        )


@pytest.mark.parametrize("recipe_id", sorted(stale_corpora()))
def test_a_declared_stale_corpus_fails_the_staleness_gate_by_name(recipe_id: str) -> None:
    """A corpus the families' recipe changes left stale fails the staleness gate, naming the recipe and every
    fingerprint input that moved -- exactly the declared ones (a further drift fails here, and a corpus
    re-recorded under the current fingerprint fails here until its declaration is removed)."""
    from rcp_ndcg_test.changes import StaleCorpusError, recipe_state, resolve_corpus

    declared = stale_corpora()[recipe_id]
    recipe = load_recipe(recipe_id)
    corpora_root = ENGINES_ROOT / "vllm-0.31.0"
    with pytest.raises(StaleCorpusError) as error:
        resolve_corpus(recipe, corpora_root)
    message = str(error.value)
    assert f"recipe {recipe_id}: stale corpus" in message
    for name in declared["changed_inputs"]:
        assert name in message, f"{recipe_id}: the staleness message does not name {name}"
    state = recipe_state(recipe, corpora_root)
    assert state["recorded_fingerprints"] == [declared["recorded_fingerprint"]], state["recorded_fingerprints"]
    assert state["changed_inputs"] == declared["changed_inputs"], state["changed_inputs"]
    assert declared["reason"].strip() and declared["decided"], declared


def test_the_stale_declarations_name_committed_corpora_only() -> None:
    """Every declaration names a recipe with a committed corpus (a retired corpus leaves the list too)."""
    recorded = {load_corpus(directory).manifest["recipe"]["id"] for directory in all_corpus_dirs()}
    assert set(stale_corpora()) <= recorded, sorted(set(stale_corpora()) - recorded)


def test_staleness_names_the_changed_inputs_and_the_waiver_file_must_be_empty_at_release() -> None:
    """The mechanism and the release rule (GPU-VALIDATION.md item 8): staleness is a failure that names
    what moved; the only way past is a dated waiver, and the waiver file ships empty."""
    from rcp_ndcg_test.fingerprint import fingerprint_changes

    recipe = load_recipe("zerank-2-reranker")
    corpus = corpus_of(recipe)
    recorded = dict(corpus.manifest["recipe"]["fingerprint_inputs"])
    mutated = dict(recorded)
    mutated["template_file"] = "sha256:" + "0" * 64
    mutated["client.max_tokens"] = "1"
    assert fingerprint_changes(recorded, mutated) == ["client.max_tokens", "template_file"]

    # the release rule: the shipped waiver file holds no waivers at all
    waivers = json.loads(WAIVERS.read_text(encoding="utf-8"))
    assert waivers == [], f"the waiver file must be empty at release, found {waivers}"


def test_a_dated_unexpired_waiver_is_the_only_way_past_staleness() -> None:
    import datetime

    from rcp_ndcg_test.changes import waiver_covers

    today = datetime.date(2026, 10, 7)
    waiver = {
        "recipe_id": "qwen3-vl-reranker-2b",
        "changed_inputs": ["template_file"],
        "reason": "the template was normalised without changing the render (operator review)",
        "date": "2026-10-07",
        "expires": "2026-10-14",
    }
    # a waiver covers staleness only for its recipe and input names, while it has not expired
    assert waiver_covers(waiver, "qwen3-vl-reranker-2b", ["template_file"], today=today)
    assert not waiver_covers(waiver, "qwen3-embedding-0.6b", ["template_file"], today=today)
    assert not waiver_covers(waiver, "qwen3-vl-reranker-2b", ["template_file", "tokenizer_sha256"], today=today)
    assert not waiver_covers({**waiver, "expires": ""}, "qwen3-vl-reranker-2b", ["template_file"], today=today)
    assert not waiver_covers({**waiver, "reason": " "}, "qwen3-vl-reranker-2b", ["template_file"], today=today)
    expired = datetime.date(2026, 10, 15)
    assert not waiver_covers(waiver, "qwen3-vl-reranker-2b", ["template_file"], today=expired)


def test_every_corpus_carries_an_append_only_verification_record() -> None:
    """The conformance verifier's record (OBSERVATIONS-SPEC section 4): which emulator verified the
    corpus, against which engine version and recipe revision, with which tolerances, and the result.

    The latest committed record must be exactly what this suite's replay computes now (its date
    aside), so a record can only come from this code. ``RCP_APPEND_VERIFICATION=1`` appends the
    computed record (never rewriting the earlier ones) -- the only writer; tests otherwise write only
    to ``tmp_path``."""
    import datetime
    import os

    from rcp_ndcg_test.corpus import append_verification
    from rcp_ndcg_test.engines import verification_record

    appending = bool(os.environ.get("RCP_APPEND_VERIFICATION"))
    today = datetime.date.today().isoformat()
    for directory in corpus_dirs():
        corpus, problems = replay_problems(directory)
        computed = verification_record(
            corpus, problems, verified_at=today, emulator=emulator_for(corpus.manifest["recipe"]["id"])
        )
        if appending:
            append_verification(directory, computed)
        records = verification_records(directory)
        assert records, f"{directory}: no verification record (RCP_APPEND_VERIFICATION=1 writes one)"
        latest = dict(records[-1])
        assert latest["result"] == "pass", latest
        latest.pop("verified_at")
        computed.pop("verified_at")
        assert latest == computed, f"{directory}: the latest record is not this suite's result: {latest}"


def test_the_registry_resolves_by_engine_version_and_fingerprint() -> None:
    import pytest
    from rcp_ndcg_test.engines import registry

    from rcp_ndcg.errors import ConfigError

    emulator = emulator_for("zerank-2-reranker")
    assert emulator.verified is not None
    fingerprint = emulator.verified.behaviour_fingerprint
    assert registry.resolve("vllm", "0.31.0", fingerprint, "zerank-2-reranker") is emulator
    with pytest.raises(ConfigError) as error:
        registry.resolve("vllm", "0.31.0", "f" * 64, "zerank-2-reranker")
    assert fingerprint[:12] in str(error.value)  # the message names the registered fingerprint


def test_an_emulator_refuses_another_engine_version_or_recipe_revision() -> None:
    import pytest

    from rcp_ndcg.errors import ConfigError

    emulator = emulator_for("qwen3-embedding-0.6b")
    assert emulator.verified is not None
    emulator.require_verified_for(
        emulator.verified.recipe_id,
        emulator.verified.revision,
        emulator.verified.behaviour_fingerprint,
        emulator.verified.engine_version,
    )
    with pytest.raises(ConfigError) as error:
        emulator.require_verified_for(
            "qwen3-embedding-0.6b", "f" * 40, emulator.verified.behaviour_fingerprint, "0.31.0"
        )
    assert "revision" in str(error.value)
    with pytest.raises(ConfigError) as error:
        emulator.require_verified_for(
            emulator.verified.recipe_id,
            emulator.verified.revision,
            emulator.verified.behaviour_fingerprint,
            "0.32.0",
        )
    assert "engine_version" in str(error.value)


def test_out_of_tree_emulators_register_through_the_entry_point_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extensibility seam (GPU-VALIDATION.md item 8): a private repository's emulator registers
    through the ``rcp_ndcg.emulators`` entry-point group."""
    import importlib.metadata as metadata

    from rcp_ndcg_test.engines import registry

    class _Entry:
        name = "example-out-of-tree"

        @staticmethod
        def load():
            def provide():
                return [emulator_for("zerank-2-reranker")]

            return provide

    original = metadata.entry_points

    def fake_entry_points(**kwargs):
        return [_Entry()] if kwargs.get("group") == "rcp_ndcg.emulators" else original(**kwargs)

    registry.clear()
    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    try:
        assert registry.load_entry_points() == ["example-out-of-tree"]
        verified = emulator_for("zerank-2-reranker").verified
        assert verified is not None
        assert registry.fingerprints("vllm", "0.31.0", "zerank-2-reranker") == [verified.behaviour_fingerprint]
    finally:
        registry.clear()


# -- M1: measured, never assumed, non-determinism ------------------------------------------------------


def _rerank_exchange(sequence: int, body: dict, score: float):
    from rcp_ndcg_test.engines import Exchange

    reply = {"id": f"score-{sequence}", "model": "m", "results": [{"index": 0, "relevance_score": score}]}
    return Exchange(sequence, "POST", "/rerank", body, 200, {"content-type": "application/json"}, reply)


def test_the_tolerance_is_the_corpus_measured_one_and_never_invented(tmp_path: Path) -> None:
    """The emulators verify with the tolerance the corpus's non-determinism report derived from same-request
    repetitions (``nondeterminism.json``); with none measured there is no tolerance and replays compare
    exactly; a corpus without the report is refused."""
    from tests._engines import observation_corpus

    from rcp_ndcg.errors import DataError

    exchange = _rerank_exchange(0, {"model": "m", "query": "q", "documents": ["d"]}, 0.5)
    unmeasured = observation_corpus(tmp_path / "none", [exchange])
    assert corpus_tolerance(unmeasured) is None
    measured = observation_corpus(tmp_path / "measured", [exchange], tolerance=(3e-7, 1e-6))
    assert corpus_tolerance(measured) == (3e-7, 1e-6)
    from dataclasses import replace

    with pytest.raises(DataError, match="non-determinism report"):
        corpus_tolerance(replace(unmeasured, nondeterminism=None))  # the reader found no nondeterminism.json


def test_a_record_with_no_response_status_is_a_no_answer_never_status_zero(tmp_path: Path) -> None:
    """A record whose response carries no status is a *no answer*: the collector recorded that nothing
    crossed the wire (a connection error before any response, a client-side refusal, a declared omission),
    so the record is not an exchange to replay. The record view reads the absence as ``None`` -- never
    coerced to 0 -- and ``compare_exchange`` names the absence instead of comparing an invented exchange."""
    from rcp_ndcg_test.engines import Exchange, compare_exchange, exchanges_of
    from tests._engines import observation_corpus

    connection_error = Exchange(
        0, "GET", "/v1/models", None, None, {}, None, source="edge:while_loading", response_raw=b"ConnectError"
    )
    (loaded,) = exchanges_of(observation_corpus(tmp_path / "no-answer", [connection_error]))
    assert loaded.status is None  # before the fix: int(None) raised TypeError
    assert not loaded.answered
    problems = compare_exchange(loaded, httpx.Response(200, json={"object": "list"}), None)
    assert problems and "no response" in problems[0], problems


def test_a_record_without_an_integer_sequence_is_refused_by_name() -> None:
    """The collector's sending order is part of the record, like the status: a record that carries no
    integer sequence is refused by name, never defaulted to 0."""
    from rcp_ndcg_test.engines import Exchange

    from rcp_ndcg.errors import DataError

    with pytest.raises(DataError, match="integer sequence"):
        Exchange.from_record({"sequence": None, "request": {}, "response": {}})


def test_every_value_is_bounded_by_the_joint_condition() -> None:
    """A replayed number passes only within BOTH measured bounds (absolute and relative): never the
    looser of the two, never their sum."""
    from rcp_ndcg_test.engines import compare_exchange

    body = {"model": "m", "query": "q", "documents": ["d"]}
    recorded = _rerank_exchange(0, body, 1.0)
    replay = httpx.Response(200, json=_rerank_exchange(1, body, 1.0005).response, headers=recorded.response_headers)
    same = httpx.Response(200, json=recorded.response, headers=recorded.response_headers)
    assert compare_exchange(recorded, replay, (1e-3, 1e-6)), "inside abs, outside rel"
    assert compare_exchange(recorded, replay, (1e-6, 1e-3)), "inside rel, outside abs"
    assert compare_exchange(recorded, replay, (1e-3, 1e-3)) == []
    assert compare_exchange(recorded, replay, None), "unmeasured is exact"
    assert compare_exchange(recorded, same, None) == []


def test_every_committed_corpus_declares_the_tolerance_its_measurement_derived() -> None:
    """The tolerance is the corpus's own measurement, never invented: the release recordings sent every
    request twice in-process and once after a restart, so every committed corpus measured non-determinism
    (measured over the 12 ``nondeterminism.json`` files: ``measured: true``, ``repeated_requests``
    111-212) and carries the bounds its report derived.  The report and the reader agree either way, and
    at least one corpus must measure, so the release recordings are exercised here, not assumed; the
    ``measured: false`` branch (a shakedown-era corpus that sent every request once) is pinned on the
    synthetic corpus in ``test_the_tolerance_is_the_corpus_measured_one_and_never_invented``."""
    measured = 0
    for directory in corpus_dirs():
        report = json.loads((directory / "nondeterminism.json").read_text(encoding="utf-8"))
        derived = report["derived"]
        assert derived["measured"] == (derived["repeated_requests"] > 0), directory
        tolerance = corpus_tolerance(load_corpus(directory))
        if derived["measured"]:
            measured += 1
            assert tolerance == (derived["abs_tolerance"], derived["rel_tolerance"]), directory
        else:
            assert tolerance is None, directory
    assert measured, "no committed corpus measured a repetition: the release recordings are missing"


# -- the release plan's protocol strata (CORPUS_PLAN_VERSION 3) ----------------------------------------


def test_a_health_probe_answers_the_engines_readiness_reply() -> None:
    """The engine's readiness answer is an empty 200 with the server header and no content type (the
    recorded probe): a byte-asserting test must not see a JSON ``{}`` the engine never sent."""
    answer = emulator_for("qwen3-reranker-0.6b").answer("/health", "GET", None)
    assert answer.status_code == 200 and answer.content == b""
    assert answer.headers["server"] == "uvicorn" and "content-type" not in answer.headers


def test_a_malformed_body_is_refused_with_the_engines_validation_error() -> None:
    """A body that is not JSON is FastAPI's ``json_invalid`` error as vLLM formats it (the recorded
    ``{not json`` probe): the decoder's offset and message, the empty-dict input, the ``Bad Request``
    type and the ``body.<offset>`` param -- verbatim, never a paraphrase."""
    request = httpx.Request(
        "POST", "http://engine/rerank", content=b"{not json", headers={"content-type": "application/json"}
    )
    answer = emulator_for("jina-reranker-v3").handle(request)
    assert answer.status_code == 400
    assert answer.json() == {
        "error": {
            "message": (
                "1 validation error:\n  {'type': 'json_invalid', 'loc': 'body.1', 'msg': 'JSON decode error', "
                "'input': '<dict of 0 items>', 'ctx': {'error': 'Expecting property name enclosed in double quotes'}}"
            ),
            "type": "Bad Request",
            "param": "body.1",
            "code": 400,
        }
    }


def test_an_empty_rerank_is_refused_with_the_engines_message() -> None:
    """A ``/rerank`` with no documents is refused with the engine's own message, for the listwise builder
    (whose empty set still renders one prompt) and the pointwise template alike."""
    for recipe_id in ("jina-reranker-v3", "qwen3-reranker-0.6b"):
        answer = emulator_for(recipe_id).answer("/rerank", "POST", {"model": recipe_id, "query": "", "documents": []})
        assert answer.status_code == 400, recipe_id
        assert answer.json()["error"] == {
            "message": "At least one text_pair element must be given",
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        }, recipe_id


def test_the_media_caps_and_a_corrupt_part_are_the_engines_own_refusals() -> None:
    """The engine enforces ``limit_mm_per_prompt`` per prompt and refuses a media part its processor cannot
    decode: both 400 bodies are the engine's own (the recorded edge probes), never the emulator's marked
    refusal."""
    import base64

    emulator = emulator_for("qwen3-vl-embedding-2b")
    image = "data:image/png;base64," + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": image}}
    two = {
        "model": "qwen3-vl-embedding-2b",
        "messages": [{"role": "user", "content": [part, part]}],
    }
    answer = emulator.answer("/v1/embeddings", "POST", two)
    assert answer.status_code == 400
    assert answer.json()["error"]["message"] == "At most 1 image(s) may be provided in one prompt. (parameter=image)"
    corrupt = {
        "model": "qwen3-vl-embedding-2b",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}],
            }
        ],
    }
    answer = emulator.answer("/v1/embeddings", "POST", corrupt)
    assert answer.status_code == 400
    assert answer.json()["error"]["message"] == "Failed to load image: cannot identify image file <_io.BytesIO object>"


def test_a_rerank_document_is_echoed_as_the_engine_echoes_it() -> None:
    """The engine echoes a rerank document as sent: a part-list document as its whole ``content`` under
    ``multi_modal`` with ``text`` null, a string document as its text with ``multi_modal`` null (the
    recorded media rows)."""
    import base64

    emulator = emulator_for("qwen3-vl-reranker-2b")
    image = "data:image/png;base64," + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii")
    parts = [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": "a caption"}]
    body = {
        "model": "qwen3-vl-reranker-2b",
        "query": "q",
        "documents": ["a text document", {"content": parts}],
    }
    results = emulator.answer("/rerank", "POST", body).json()["results"]
    by_index = {entry["index"]: entry["document"] for entry in results}
    assert by_index[0] == {"text": "a text document", "multi_modal": None}
    assert by_index[1] == {"text": None, "multi_modal": parts}


def test_a_rerank_reply_is_compared_by_document_identity() -> None:
    """A rerank reply is compared by document identity: a near-tie rank flip within the measured tolerance
    is not a body drift (the recorded qwen3-reranker-8b case, whose re-derived corpus tolerance is abs
    0.1640625 / rel 0.6847826086956521), while a missing document or a score outside the tolerance still
    refuses."""
    from rcp_ndcg_test.engines import Exchange

    def reply(scores: list[tuple[int, float]]) -> dict:
        return {
            "id": "score-1",
            "model": "m",
            "results": [
                {"index": index, "document": {"text": f"d{index}", "multi_modal": None}, "relevance_score": score}
                for index, score in scores
            ],
        }

    tolerance = (0.1640625, 0.6847826086956521)  # qwen3-reranker-8b's re-derived corpus tolerance
    recorded = Exchange(0, "POST", "/rerank", {}, 200, {}, reply([(1, 0.542969), (2, 0.542969)]))
    flipped = reply([(2, 0.550781), (1, 0.535156)])
    assert compare_exchange(recorded, httpx.Response(200, json=flipped), tolerance) == []
    missing = reply([(1, 0.542969)])
    assert any(
        "document indices" in problem
        for problem in compare_exchange(recorded, httpx.Response(200, json=missing), tolerance)
    )
    moved = reply([(1, 5.0), (2, 0.542969)])
    assert any(
        ".results[1]" in problem for problem in compare_exchange(recorded, httpx.Response(200, json=moved), tolerance)
    )


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
"""A 1x1 PNG: the smallest image the product's own media probe reads, for the media-refusal pins."""


@pytest.mark.skipif(
    __import__("os").environ.get("RCP_NDCG_RELEASE") != "1",
    reason="the release rule: set RCP_NDCG_RELEASE=1 (the release checklist runs the suite with it)",
)
def test_the_stale_declarations_are_empty_at_release() -> None:
    """The release rule for re-recording (mirrors the waiver file's): every corpus declared stale has been
    re-recorded before the tag, so ``tests/conformance/stale.json`` ships empty."""
    stale = json.loads((Path(__file__).resolve().parent / "stale.json").read_text(encoding="utf-8"))
    assert stale == [], "stale corpora must be re-recorded before release: " + ", ".join(
        str(entry.get("recipe_id")) for entry in stale
    )


def _assert_rekeyed_entries(manifest: dict[str, Any], where: object) -> int:
    """One manifest's ``rekeyed`` entries: each keeps the old fingerprint AND the old manifest's digest and
    file SHA-256, which link the subset to the full corpus it was cut from without git history."""
    found = 0
    for entry in manifest["recipe"].get("rekeyed") or []:
        found += 1
        name = manifest["recipe"].get("id", where)
        for key in ("from_behaviour_fingerprint", "from_manifest_sha256", "from_manifest_file_sha256"):
            value = entry.get(key)
            assert isinstance(value, str) and len(value) == 64, f"{name}: rekeyed entry lacks {key}"
    return found


def test_a_rekeyed_corpus_names_the_manifest_it_was_rekeyed_from() -> None:
    """A re-key rewrites the manifest (its fingerprint and inputs), so the subset's manifest is no longer the
    full corpus's byte for byte: each ``rekeyed`` entry keeps the old fingerprint AND the old manifest's
    digest and file SHA-256, which link the subset to the full corpus it was cut from without git history.
    Every committed corpus that carries such an entry is checked, and the rule is pinned on a synthetic
    manifest so it cannot pass vacuously (the release recordings replaced the two re-keyed zerank corpora,
    so no committed corpus is re-keyed today)."""
    for directory in all_corpus_dirs():
        _assert_rekeyed_entries(load_corpus(directory).manifest, directory)
    digest = "0" * 64
    complete = {
        "from_behaviour_fingerprint": digest,
        "from_manifest_sha256": digest,
        "from_manifest_file_sha256": digest,
    }
    assert _assert_rekeyed_entries({"recipe": {"id": "synthetic", "rekeyed": [complete]}}, "synthetic") == 1
    with pytest.raises(AssertionError):
        _assert_rekeyed_entries(
            {"recipe": {"id": "synthetic", "rekeyed": [{"from_behaviour_fingerprint": digest}]}}, "synthetic"
        )


def _assert_redacted_entries(manifest: dict[str, Any], where: object) -> int:
    """One manifest's ``engine.redacted`` entries: each names the field it redacted, why and when, and keeps
    the original manifest's digest and file SHA-256 -- the link to the store's recorded original, which the
    tracked file is now a redacted view of."""
    found = 0
    for entry in (manifest.get("engine") or {}).get("redacted") or []:
        found += 1
        name = (manifest.get("recipe") or {}).get("id", where)
        for key in ("field_path", "reason", "date"):
            value = entry.get(key)
            assert isinstance(value, str) and value.strip(), f"{name}: redacted entry lacks {key}"
        for key in ("from_manifest_sha256", "from_manifest_file_sha256"):
            value = entry.get(key)
            assert isinstance(value, str) and len(value) == 64, f"{name}: redacted entry lacks {key}"
    return found


def test_a_redacted_corpus_names_the_manifest_it_was_redacted_from() -> None:
    """A redaction rewrites the manifest (its engine environment and argv), so the subset's manifest is no
    longer the store's original byte for byte: each ``engine.redacted`` entry keeps the original manifest's
    digest and file SHA-256, which link the tracked view to the recording it was cut from without git
    history, and names the field, the reason and the date -- nothing silent. Every committed corpus that
    carries such an entry is checked, and the rule is pinned on a synthetic manifest so it cannot pass
    vacuously."""
    for directory in all_corpus_dirs():
        _assert_redacted_entries(load_corpus(directory).manifest, directory)
    digest = "0" * 64
    complete = {
        "field_path": "engine.env",
        "reason": "the pod's Service links ride under the VLLM_ prefix the engine's own knobs use",
        "date": "2026-10-14",
        "from_manifest_sha256": digest,
        "from_manifest_file_sha256": digest,
    }
    assert _assert_redacted_entries({"recipe": {"id": "synthetic"}, "engine": {"redacted": [complete]}}, "s") == 1
    with pytest.raises(AssertionError):
        _assert_redacted_entries(
            {"engine": {"redacted": [{"field_path": "engine.env", "reason": "r", "date": "2026-10-14"}]}}, "s"
        )


def test_the_declared_stale_listwise_corpus_still_replays_its_role_requests() -> None:
    """Listwise replay coverage while jina-reranker-v3's corpus is declared stale.

    The staleness gate keeps the corpus out of the standard replay set (:func:`current_corpora`), and
    with it the package's only listwise prompt path (``EnginePrompts`` over the recipe's own N-passage
    builder).  The corpus's fingerprint inputs moved, but its recorded exchanges -- the ``/rerank``
    probes among them -- still replay identically against the current recipe's emulator, so the listwise
    path stays exercised on CPU until the re-recording lands; once the declaration is gone the standard
    replay covers it and this test skips.  The corpus stays non-gating release evidence: its status is
    the stale declaration's, and this test never relaxes that.
    """
    from rcp_ndcg_test.engines import compare_exchange, corpus_tolerance, exchanges_of
    from tests._engines import emulator_over

    if "jina-reranker-v3" not in stale_corpora():
        pytest.skip("the corpus is no longer declared stale: the standard replay set covers the listwise path")
    directory = next(
        directory
        for directory in all_corpus_dirs()
        if load_corpus(directory).manifest["recipe"]["id"] == "jina-reranker-v3"
    )
    corpus = load_corpus(directory)
    emulator = emulator_over("jina-reranker-v3", corpus)
    role_requests = [
        exchange for exchange in exchanges_of(corpus) if exchange.path.endswith("/rerank") and exchange.status == 200
    ]
    assert role_requests, "the corpus records no successful role request: the replay would pass vacuously"
    for exchange in exchanges_of(corpus):
        if not exchange.answered:
            continue  # no recorded response: not an exchange, nothing to replay (see replay_problems)
        problems = compare_exchange(exchange, emulator.handle(as_request(exchange)), corpus_tolerance(corpus))
        assert not problems, f"{directory.name} #{exchange.sequence}: {problems}"
