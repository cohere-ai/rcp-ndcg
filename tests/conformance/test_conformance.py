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

import pytest

from rcp_ndcg.testing.engines import (
    compare_exchange,
    credential_findings,
    find_corpora,
    find_credential_patterns,
    load_corpus,
    verification_records,
    verify_corpus_hashes,
)
from tests._engines import ROOT, corpus_of, emulator_for, load_recipe

ENGINES_ROOT = ROOT / "tests" / "contract" / "engines"
INDEX = ENGINES_ROOT / "vllm-0.31.0" / "index.json"
WAIVERS = Path(__file__).resolve().parent / "waivers.json"


def corpus_dirs() -> list[Path]:
    """Every committed corpus directory, found by scanning manifests."""
    return find_corpora(ENGINES_ROOT)


def replay_problems(directory: Path) -> tuple[object, list[str]]:
    """Replay every recorded exchange of one corpus against its emulator: the conformance differences
    (identical status and body within the recorded non-determinism), each naming the record."""
    corpus = load_corpus(directory)
    recipe_id = corpus.manifest["recipe"]["id"]
    emulator = emulator_for(recipe_id)
    problems = []
    for exchange in corpus.exchanges:
        answer = emulator.answer(exchange.path, exchange.method, exchange.request_body)
        found = compare_exchange(
            exchange, exchange.method, exchange.path, answer.status_code, answer.json(), corpus.tolerance
        )
        problems.extend(f"#{exchange.sequence}: {problem}" for problem in found)
    return corpus, problems


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
        for exchange in corpus.exchanges:
            if exchange.status != 200 or exchange.path.endswith("/models"):
                continue
            recorded = (exchange.response_json or {}).get("usage", {}).get("prompt_tokens")
            answered = emulator.answer(exchange.path, exchange.method, exchange.request_body).json()
            assert answered["usage"]["prompt_tokens"] == recorded, f"{directory.name} #{exchange.sequence}"


def test_manifest_and_index_hashes_hold() -> None:
    """Integrity (OBSERVATIONS-SPEC section 4): every corpus file hashes to its manifest and every
    manifest hashes to the repository's corpus index. A tampered corpus fails here."""
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    corpora = index["corpora"]
    for directory in corpus_dirs():
        corpus = load_corpus(directory)
        assert verify_corpus_hashes(corpus) == [], directory
        key = f"{directory.parent.name}/{directory.name}"
        manifest_sha = corpora[key]["manifest_sha256"]
        import hashlib

        actual = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        assert actual == manifest_sha, f"{key}: manifest hashes to {actual}, the index declares {manifest_sha}"
        assert corpora[key]["behaviour_fingerprint"] == corpus.behaviour_fingerprint, key


def test_no_credential_shaped_string_is_in_any_corpus() -> None:
    """Acceptance (OBSERVATIONS-SPEC section 6): no credential-shaped string anywhere."""
    for path in sorted(ENGINES_ROOT.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".gz", ".jsonl"):
            findings = find_credential_patterns(path)
            assert findings == [], f"{path}: {findings}"
    assert credential_findings("") == []


def test_staleness_passes_for_the_unmoved_recipes() -> None:
    """Every committed corpus is keyed by a fingerprint the repository reproduces exactly -- or a dated,
    unexpired waiver covers exactly what moved (the release checklist requires the file empty)."""
    import datetime

    from rcp_ndcg_vllm.changes import recipe_state, waiver_covers

    waivers = json.loads(WAIVERS.read_text(encoding="utf-8"))
    today = datetime.date.today()
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
            "(python -m rcp_ndcg_vllm.changes changed) or add a dated entry to tests/conformance/waivers.json"
        )


def test_staleness_names_the_changed_inputs_and_the_waiver_file_must_be_empty_at_release() -> None:
    """The mechanism and the release rule (GPU-VALIDATION.md item 8): staleness is a failure that names
    what moved; the only way past is a dated waiver, and the waiver file ships empty."""
    from rcp_ndcg_vllm.fingerprint import fingerprint_changes

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

    from rcp_ndcg_vllm.changes import waiver_covers

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

    from rcp_ndcg.testing.engines import append_verification, verification_record

    appending = bool(os.environ.get("RCP_APPEND_VERIFICATION"))
    today = datetime.date.today().isoformat()
    for directory in corpus_dirs():
        corpus, problems = replay_problems(directory)
        computed = verification_record(corpus, problems, verified_at=today)
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

    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.testing.engines import registry

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


def test_out_of_tree_emulators_register_through_the_entry_point_group() -> None:
    """The extensibility seam (GPU-VALIDATION.md item 8): a private repository's emulator registers
    through the ``rcp_ndcg.emulators`` entry-point group."""
    import importlib.metadata as metadata

    from rcp_ndcg.testing.engines import registry

    class _Entry:
        name = "example-out-of-tree"

        @staticmethod
        def load():
            def provide():
                return [emulator_for("zerank-2-reranker")]

            return provide

    registry.clear()
    registry._entry_points_loaded = False
    original = metadata.entry_points

    def fake_entry_points(**kwargs):
        return [_Entry()] if kwargs.get("group") == "rcp_ndcg.emulators" else original(**kwargs)

    metadata.entry_points = fake_entry_points  # type: ignore[assignment]
    try:
        loaded = registry.load_entry_points()
    finally:
        metadata.entry_points = original  # type: ignore[assignment]
    assert loaded == ["example-out-of-tree"]
    assert emulator_for("zerank-2-reranker").verified is not None


def test_the_corpus_schema_migrations_never_invalidate_an_old_corpus(tmp_path: Path) -> None:
    """Schema evolution (OBSERVATIONS-SPEC section 7): readers support every document schema ever
    written; a frozen sample of each old schema is migrated at load."""
    from rcp_ndcg.testing.engines import load_corpus, register_line_migration

    corpus_dir = corpus_dirs()[0]
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    line = json.loads(__import__("gzip").decompress((corpus_dir / manifest["documents"]).read_bytes()).splitlines()[0])

    def from_zero(doc: dict) -> dict:
        return {"line_schema": 1, "sequence": doc["seq"], **{k: v for k, v in doc.items() if k != "seq"}}

    register_line_migration(0, from_zero)
    old = {
        "line_schema": 0,
        "seq": line["sequence"],
        **{k: v for k, v in line.items() if k not in ("line_schema", "sequence")},
    }
    manifest_copy = {**manifest, "documents": "exchanges.jsonl", "files": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_copy), encoding="utf-8")
    (tmp_path / "exchanges.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    migrated = load_corpus(tmp_path)
    assert migrated.exchanges[0].sequence == line["sequence"]
    assert migrated.exchanges[0].path == line["path"]


# -- M1: measured, never assumed, non-determinism ------------------------------------------------------


def _rerank_exchange(sequence: int, body: dict, score: float):
    from rcp_ndcg.testing.engines import Exchange

    reply = {"id": f"score-{sequence}", "model": "m", "results": [{"index": 0, "relevance_score": score}]}
    return Exchange(sequence, "POST", "/rerank", body, 200, {"content-type": "application/json"}, reply)


def test_non_determinism_is_measured_only_across_same_request_repetitions(tmp_path: Path) -> None:
    """Two requests that differ in any byte (here ``use_activation``) are different questions, never a
    repetition: with no true repetition the tolerance is declared unmeasured, not invented. A true
    repetition (an identical request digest) is measured."""
    from rcp_ndcg.testing.engines import Corpus, measure_non_determinism

    body = {"model": "m", "query": "q", "documents": ["d"], "use_activation": True}
    other = {key: value for key, value in body.items() if key != "use_activation"}
    different = Corpus(tmp_path, {}, (_rerank_exchange(0, body, 0.9246), _rerank_exchange(1, other, 0.9238)))
    block = measure_non_determinism(different)
    assert block["status"] == "unmeasured" and block["same_request_repetitions"] == 0, block
    assert block["tolerance_abs"] is None and block["tolerance_rel"] is None

    repeated = Corpus(tmp_path, {}, (_rerank_exchange(0, body, 0.9246), _rerank_exchange(1, dict(body), 0.9244)))
    block = measure_non_determinism(repeated)
    assert block["status"] == "measured" and block["same_request_repetitions"] == 1, block
    assert block["measured_max_abs"] == pytest.approx(0.0002)
    assert block["tolerance_abs"] == block["measured_max_abs"]
    assert block["tolerance_rel"] == block["measured_max_rel"]


def test_every_value_is_bounded_by_the_joint_condition() -> None:
    """A replayed number passes only within BOTH measured bounds (absolute and relative): never the
    looser of the two, never their sum."""
    from rcp_ndcg.testing.engines import compare_exchange

    body = {"model": "m", "query": "q", "documents": ["d"]}
    recorded = _rerank_exchange(0, body, 1.0)
    replay = _rerank_exchange(1, body, 1.0005).response
    assert compare_exchange(recorded, "POST", "/rerank", 200, replay, (1e-3, 1e-6)), "inside abs, outside rel"
    assert compare_exchange(recorded, "POST", "/rerank", 200, replay, (1e-6, 1e-3)), "inside rel, outside abs"
    assert compare_exchange(recorded, "POST", "/rerank", 200, replay, (1e-3, 1e-3)) == []
    assert compare_exchange(recorded, "POST", "/rerank", 200, replay, None), "unmeasured is exact"
    assert compare_exchange(recorded, "POST", "/rerank", 200, recorded.response, None) == []


def test_every_manifest_states_the_non_determinism_its_raw_records_measure() -> None:
    """The stored block is exactly what the versioned code recomputes from the raw records; the
    shakedown sent every request once, so its tolerance is declared unmeasured."""
    from rcp_ndcg.testing.engines import measure_non_determinism

    for directory in corpus_dirs():
        corpus = load_corpus(directory)
        assert corpus.manifest["non_determinism"] == measure_non_determinism(corpus), directory
        assert corpus.manifest["non_determinism"]["status"] == "unmeasured", directory
        assert corpus.tolerance is None
