"""The conformance suite: every recorded exchange replays against its emulator (GPU-VALIDATION.md
items 2-7), staleness fails with the changed inputs named, and the registry refuses what it was not
verified against.

Every corpus under ``tests/contract/engines/`` is loaded through the corpus seam, its emulator is built
through ``tests._engines`` (the recipe's real tokenizer and prompt derivation), and **every recorded
exchange** is replayed and compared within the recorded non-determinism (exact for the shakedown
corpus). The suite runs in CI on every push and needs no network.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    (identical status and body within the recorded non-determinism), each naming the record."""
    corpus = load_corpus(directory)
    recipe_id = corpus.manifest["recipe"]["id"]
    emulator = emulator_for(recipe_id)
    problems = []
    for exchange in exchanges_of(corpus):
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


def test_usage_counts_use_the_recipes_real_tokenizer() -> None:
    """The replayed ``usage.prompt_tokens`` is the engine's own count (the recipe's real tokenizer
    files): the recorded values reproduce exactly (measured: the sum of each engine prompt's tokens)."""
    for directory in corpus_dirs():
        corpus = load_corpus(directory)
        emulator = emulator_for(corpus.manifest["recipe"]["id"])
        for exchange in exchanges_of(corpus):
            if exchange.status != 200 or exchange.path.endswith("/models"):
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
    empty). A corpus that goes stale without being declared fails here, naming the inputs that moved."""
    import datetime

    from rcp_ndcg_test.changes import recipe_state, waiver_covers

    waivers = json.loads(WAIVERS.read_text(encoding="utf-8"))
    today = datetime.date.today()
    assert corpus_dirs(), "no current corpus left: every conformance replay would be vacuous"
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

    recipe = load_recipe("qwen3-vl-reranker-2b")
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

    emulator = emulator_for("qwen3-reranker-8b")
    assert emulator.verified is not None
    fingerprint = emulator.verified.behaviour_fingerprint
    assert registry.resolve("vllm", "0.31.0", fingerprint, "qwen3-reranker-8b") is emulator
    with pytest.raises(ConfigError) as error:
        registry.resolve("vllm", "0.31.0", "f" * 64, "qwen3-reranker-8b")
    assert fingerprint[:12] in str(error.value)  # the message names the registered fingerprint


def test_an_emulator_refuses_another_engine_version_or_recipe_revision() -> None:
    import pytest

    from rcp_ndcg.errors import ConfigError

    emulator = emulator_for("zembed-1-embedding")
    assert emulator.verified is not None
    emulator.require_verified_for(
        emulator.verified.recipe_id,
        emulator.verified.revision,
        emulator.verified.behaviour_fingerprint,
        emulator.verified.engine_version,
    )
    with pytest.raises(ConfigError) as error:
        emulator.require_verified_for("zembed-1-embedding", "f" * 40, emulator.verified.behaviour_fingerprint, "0.31.0")
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
                return [emulator_for("qwen3-reranker-8b")]

            return provide

    original = metadata.entry_points

    def fake_entry_points(**kwargs):
        return [_Entry()] if kwargs.get("group") == "rcp_ndcg.emulators" else original(**kwargs)

    registry.clear()
    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    try:
        assert registry.load_entry_points() == ["example-out-of-tree"]
        verified = emulator_for("qwen3-reranker-8b").verified
        assert verified is not None
        assert registry.fingerprints("vllm", "0.31.0", "qwen3-reranker-8b") == [verified.behaviour_fingerprint]
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


def test_every_committed_corpus_declares_its_tolerance_unmeasured() -> None:
    """The shakedown sent every request once: no corpus measured a repetition, so none carries a tolerance."""
    for directory in corpus_dirs():
        report = json.loads((directory / "nondeterminism.json").read_text(encoding="utf-8"))
        assert report["derived"]["measured"] is False and report["derived"]["repeated_requests"] == 0, directory
        assert corpus_tolerance(load_corpus(directory)) is None


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


def test_a_rekeyed_corpus_names_the_manifest_it_was_rekeyed_from() -> None:
    """A re-key rewrites the manifest (its fingerprint and inputs), so the subset's manifest is no longer the
    full corpus's byte for byte: each ``rekeyed`` entry keeps the old fingerprint AND the old manifest's
    digest and file SHA-256, which link the subset to the full corpus it was cut from without git history."""
    found = 0
    for directory in all_corpus_dirs():
        manifest = load_corpus(directory).manifest
        for entry in manifest["recipe"].get("rekeyed") or []:
            found += 1
            name = manifest["recipe"]["id"]
            for key in ("from_behaviour_fingerprint", "from_manifest_sha256", "from_manifest_file_sha256"):
                value = entry.get(key)
                assert isinstance(value, str) and len(value) == 64, f"{name}: rekeyed entry lacks {key}"
    assert found, "no re-keyed corpus: the check would pass vacuously"
