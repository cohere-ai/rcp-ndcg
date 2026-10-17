"""The observation-corpus format's one reader (``rcp_ndcg_test.corpus``): schemas, integrity, normalisation.

The GPU waves write corpora (``rcp-ndcg-vllm``'s collector); the verified fake engines read them on CPU through
this module, so the reader is the single seam both sides share. The frozen sample under
``tests/fixtures/observation_corpus_v1/`` is a corpus written under ``RECORD_SCHEMA = 1``: every later reader must keep
loading it (OBSERVATIONS-SPEC section 7: a schema bump never invalidates an old corpus).
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
from rcp_ndcg_test import corpus as observation
from rcp_ndcg_test.corpus import (
    CORPUS_SCHEMA,
    RECORD_SCHEMA,
    credential_findings,
    integrity_mismatches,
    load_corpus,
    manifest_digest,
    normalise_body,
    register_record_migration,
)

from rcp_ndcg.errors import DataError

FROZEN_V1 = Path(__file__).resolve().parent / "fixtures" / "observation_corpus_v1"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "corpus"
    shutil.copytree(FROZEN_V1, target)
    return target


def test_the_frozen_schema_1_sample_loads_and_verifies() -> None:
    """A corpus written under RECORD_SCHEMA 1 loads with every record and passes its integrity check."""
    corpus = load_corpus(FROZEN_V1)
    assert corpus.manifest["schema"] == CORPUS_SCHEMA
    assert len(corpus.records) == corpus.manifest["integrity"]["records_count"] == 4
    assert {record["record_schema"] for record in corpus.records} == {RECORD_SCHEMA}
    assert corpus.records[0]["response"]["status"] == 200
    assert integrity_mismatches(corpus) == []
    assert corpus.subset_index is None


def test_a_tampered_file_fails_the_integrity_check(tmp_path: Path) -> None:
    """One more valid record: parseable, but no longer the bytes the manifest hashed."""
    directory = _copy(tmp_path)
    records = directory / "records.jsonl"
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    mismatches = integrity_mismatches(load_corpus(directory))
    assert any("records.jsonl" in entry for entry in mismatches), mismatches


def test_an_edited_manifest_fails_its_own_digest(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe"]["id"] = "another-recipe"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "manifest.json: manifest_sha256 mismatch" in integrity_mismatches(load_corpus(directory))


def test_a_repository_subset_reads_through_the_same_reader(tmp_path: Path) -> None:
    """``records.jsonl.gz`` and ``index.json`` (the repository subset) load like the full corpus, and its
    integrity is the index's: the copied manifest is the full corpus's, byte for byte."""
    source = load_corpus(FROZEN_V1)
    subset = tmp_path / "subset"
    subset.mkdir()
    shutil.copy(FROZEN_V1 / "manifest.json", subset / "manifest.json")
    shutil.copy(FROZEN_V1 / "nondeterminism.json", subset / "nondeterminism.json")
    with gzip.open(subset / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in source.records[:2]))
    observation.write_subset_index(subset, full_corpus_path="gs://YOUR-BUCKET/observations/x", sampling={"rows": 2})
    corpus = load_corpus(subset)
    assert len(corpus.records) == 2 and corpus.subset_index is not None
    assert integrity_mismatches(corpus) == []
    (subset / "records.jsonl.gz").write_bytes(gzip.compress((json.dumps(source.records[3]) + "\n").encode()))
    assert any("records.jsonl.gz" in entry for entry in integrity_mismatches(load_corpus(subset)))


def test_a_record_from_a_newer_collector_is_refused_loudly(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    records = directory / "records.jsonl"
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0]["record_schema"] = RECORD_SCHEMA + 1
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(DataError, match=r"record_schema"):
        load_corpus(directory)


def test_an_older_record_schema_is_migrated_on_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Readers support every schema ever written: an older record passes through its registered migration."""
    directory = _copy(tmp_path)
    records = directory / "records.jsonl"
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0]["record_schema"] = 0
    rows[0]["legacy_field"] = rows[0].pop("server_run_id")
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def zero_to_one(record: dict) -> dict:
        record = dict(record)
        record["server_run_id"] = record.pop("legacy_field")
        record["record_schema"] = 1
        return record

    monkeypatch.setattr(observation, "_MIGRATIONS", dict(observation._MIGRATIONS))
    with pytest.raises(DataError, match=r"no migration"):
        load_corpus(directory)
    register_record_migration(0, zero_to_one)
    loaded = load_corpus(directory)
    assert loaded.records[0]["record_schema"] == 1 and "legacy_field" not in loaded.records[0]


def test_a_manifest_of_another_format_is_refused(tmp_path: Path) -> None:
    directory = _copy(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "some-other-format/9"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match="some-other-format/9"):
        load_corpus(directory)


def test_normalisation_strips_only_the_volatile_fields() -> None:
    """``NORMALISATION_VERSION`` 1 strips request ids and creation timestamps, nothing else."""
    body = {
        "id": "embd-123",
        "created": 1700000000,
        "object": "list",
        "data": [{"index": 0, "embedding": [0.5], "created": 3}],
        "usage": {"prompt_tokens": 4},
    }
    assert normalise_body(body) == {
        "object": "list",
        "data": [{"index": 0, "embedding": [0.5]}],
        "usage": {"prompt_tokens": 4},
    }
    models = {"object": "list", "data": [{"id": "m", "created": 5, "permission": [{"id": "p", "created": 6}]}]}
    assert normalise_body(models) == {"object": "list", "data": [{"id": "m", "permission": [{}]}]}
    assert body["id"] == "embd-123", "the raw body is never mutated"
    assert normalise_body([1, 2]) == [1, 2] and normalise_body(None) is None


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
        '{"authorization": "Basic dXNlcjpwYXNzd29yZHBhc3N3b3Jk"}',
        "token hf_" + "A" * 30,
        "-----BEGIN RSA PRIVATE KEY-----",
        "key AIza" + "B" * 30,
        "sk-" + "c" * 32,
        '{"X-Api-Key": "abcdefghijklmnopqrstuvwx0123"}',
        "x-api-key: abcdefghijklmnopqrstuvwx0123",
        '{"cookie": "session=abcdef0123456789abcdef"}',
        "Set-Cookie: sid=abcdef0123456789abcdef; Path=/",
        '{"api_key": "abcdefghijklmnop"}',
        '{"client_secret": "abcdefghijklmnop"}',
        '{"password": "correct-horse-battery"}',
        '{"access_token": "abcdefghijklmnopqrst"}',
    ],
)
def test_every_credential_shape_is_found(text: str) -> None:
    assert credential_findings(text), text


def test_ordinary_corpus_text_carries_no_credential_finding() -> None:
    clean = json.dumps({"query": "how do I set an authorization header", "documents": ["Bearer of good news"]})
    assert credential_findings(clean) == []
    # tokenizer vocabularies spell the field names with integer ids, and prose names them in passing
    vocabulary = '{"password": 12345, "cookie": 67, "api_key": 8, "x-api-key": 9, "secret": 10}'
    assert credential_findings(vocabulary) == []
    assert credential_findings('{"query": "what is a cookie in a browser", "documents": ["the password field"]}') == []
    assert credential_findings((FROZEN_V1 / "records.jsonl").read_text(encoding="utf-8")) == []


def test_the_manifest_digest_ignores_only_its_own_field() -> None:
    manifest = load_corpus(FROZEN_V1).manifest
    assert manifest_digest(manifest) == manifest["integrity"]["manifest_sha256"]
    changed = json.loads(json.dumps(manifest))
    changed["integrity"]["manifest_sha256"] = "0" * 64
    assert manifest_digest(changed) == manifest["integrity"]["manifest_sha256"]
    changed["collector"] = {"other": True}
    assert manifest_digest(changed) != manifest["integrity"]["manifest_sha256"]


def test_the_corpus_exposes_its_nondeterminism_report(tmp_path: Path) -> None:
    """``nondeterminism.json`` is part of the format: the reader exposes it parsed (``None`` when absent), and
    an unreadable one is refused naming the file."""
    corpus = load_corpus(FROZEN_V1)
    on_disk = json.loads((FROZEN_V1 / "nondeterminism.json").read_text(encoding="utf-8"))
    assert corpus.nondeterminism == on_disk
    directory = _copy(tmp_path)
    (directory / "nondeterminism.json").unlink()
    assert load_corpus(directory).nondeterminism is None
    (directory / "nondeterminism.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="nondeterminism.json"):
        load_corpus(directory)


def test_the_verification_record_lives_beside_the_corpus_outside_its_hashes(tmp_path: Path) -> None:
    """``verification.jsonl`` is append-only and not part of the recorded corpus: a subset index written while
    one exists does not hash it, appending never breaks the integrity check, and the records read back in
    order (each line one record; earlier lines never rewritten)."""
    source = load_corpus(FROZEN_V1)
    subset = tmp_path / "subset"
    subset.mkdir()
    shutil.copy(FROZEN_V1 / "manifest.json", subset / "manifest.json")
    shutil.copy(FROZEN_V1 / "nondeterminism.json", subset / "nondeterminism.json")
    with gzip.open(subset / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in source.records))
    first = {"schema": observation.VERIFICATION_SCHEMA, "result": "pass", "verified_at": "2026-10-07"}
    observation.append_verification(subset, first)
    index = json.loads(
        observation.write_subset_index(subset, full_corpus_path="gs://YOUR-BUCKET/x", sampling={}).read_text()
    )
    assert observation.VERIFICATION_FILE not in index["files"]
    second = {**first, "verified_at": "2026-10-08"}
    observation.append_verification(subset, second)
    assert integrity_mismatches(load_corpus(subset)) == []
    assert observation.verification_records(subset) == [first, second]
    assert observation.verification_records(tmp_path / "absent") == []
    with pytest.raises(DataError, match="schema"):
        observation.append_verification(subset, {"result": "pass"})


def test_one_raw_body_normaliser_masks_what_the_parsed_one_strips() -> None:
    """The byte-level normaliser masks exactly the volatile values :func:`normalise_body` strips, so two raw
    replies that differ only in ids and stamps compare equal, and any other byte still differs."""
    first = b'{"id":"embd-1a2b","object":"list","created":1700000000,"data":[{"embedding":[0.5]}]}'
    second = b'{"id": "embd-9z8y","object":"list","created": 1800000000,"data":[{"embedding":[0.5]}]}'
    other = b'{"id":"embd-1a2b","object":"list","created":1700000000,"data":[{"embedding":[0.25]}]}'
    assert observation.normalise_raw(first) == observation.normalise_raw(second)
    assert observation.normalise_raw(first) != observation.normalise_raw(other)
    assert normalise_body(json.loads(first)) == normalise_body(json.loads(second))


def test_missing_provenance_is_public_and_names_every_gap() -> None:
    """The section-4 provenance check is the format's: a manifest key with neither a value nor an
    ``unavailable`` reason is named ``block.key``; a missing block is named whole."""
    manifest = {block: {key: "x" for key in keys} for block, keys in observation.PROVENANCE_KEYS.items()}
    assert observation.missing_provenance(manifest) == []
    manifest["engine"]["image_digest"] = None
    manifest["model"]["weights"] = {"unavailable": ""}
    manifest["model"]["plugin"] = {"unavailable": "no plugin: an in-tree architecture"}
    del manifest["collector"]
    assert observation.missing_provenance(manifest) == ["engine.image_digest", "model.weights", "collector"]
