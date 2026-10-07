"""The observation corpus on the collector's side (OBSERVATIONS-SPEC sections 2-7): records, the hash-chained
manifest, the acceptance checks, the repository subset and the change handling.

The format and its one reader live in the product, :mod:`rcp_ndcg.testing.corpus` (the verified fake engines
read corpora there, and the product never imports this package): its schema identifiers, ``load_corpus`` --
full corpora and repository subsets alike, older record schemas migrated --, the integrity check, the
normalisation comparisons apply and the credential scan.  This module writes corpora in that format and decides
whether one is accepted.

A corpus is keyed by engine + version and the recipe's **behaviour fingerprint**
(:func:`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`, GPU-VALIDATION.md item 8): the full corpus lives at
:func:`corpus_path` (``observations/<engine>-<version>/<recipe>/<fingerprint>/<recorded-at>/``, never
overwritten) and its repository subset at ``tests/contract/engines/<engine>-<version>/<recipe>/<fingerprint>/``.

Nothing derived is stored: vectors (:func:`derived_vectors`, decoded by the product's own adapters), normalised
bodies and tolerances are recomputed from the raw records by versioned code.

Public surface:

- :data:`DERIVED_TOLERANCE_RULE`, :data:`REPETITIONS`.
- :func:`corpus_key`, :func:`corpus_path`, :func:`build_record`, :func:`write_corpus`.
- :func:`derived_vectors`, :func:`summarise_nondeterminism`, :func:`expected_status`.
- :func:`verify_corpus` (the acceptance checks), :func:`subset_for_repository`.
- :func:`changed_since` (section 7: re-record changed only).
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rcp_ndcg.errors import RcpNdcgError
from rcp_ndcg.support.identity import hash_payload
from rcp_ndcg.testing.corpus import (
    CORPUS_SCHEMA,
    MANIFEST_FILE,
    NONDETERMINISM_FILE,
    NORMALISATION_VERSION,
    RECORD_SCHEMA,
    RECORDS_FILE,
    RECORDS_FILE_GZ,
    ObservationCorpus,
    credential_findings,
    integrity_mismatches,
    load_corpus,
    manifest_digest,
    normalise_body,
    write_subset_index,
)

from ..errors import HarnessError
from ..fingerprint import behaviour_fingerprint
from ..recipe import Recipe

__all__ = [
    "DERIVED_TOLERANCE_RULE",
    "REPETITIONS",
    "build_record",
    "changed_since",
    "corpus_key",
    "corpus_path",
    "derived_vectors",
    "expected_status",
    "subset_for_repository",
    "summarise_nondeterminism",
    "verify_corpus",
    "write_corpus",
]

REPETITIONS: tuple[str, ...] = ("same_process_1", "same_process_2", "after_restart")
"""The sendings of one request set (OBSERVATIONS-SPEC section 2): twice in one engine process, then once
after an engine restart (its own ``server_run_id``)."""

DERIVED_TOLERANCE_RULE = (
    "verification tolerance = 3x the maximum difference measured between repeated sendings of the SAME request "
    "(exchange_id: twice in one engine process and once after a restart), with absolute floor 1e-9 and relative "
    "floor 1e-6; no repeated request measured means no tolerance (declared unmeasured, never invented)"
)
"""How the measured non-determinism derives an emulator's verification tolerance (section 2)."""

_EXPECTED_STATUS_VLLM_0_31_0: dict[str, dict[str, int]] = {
    "/v1/embeddings": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/pooling": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/rerank": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/v1/models": {"ok": 200},
    "/tokenize": {"ok": 200},
}
"""The vLLM v0.31.0 status table the shakedown recordings measured (12 recipes): an unknown request field is
IGNORED with 200 on the role routes, an over-length prompt is refused with 400 and the OpenAI-style
``{"error": ...}`` body, malformed JSON is a 400.  Probes without a measured row (``wrong_model``,
``empty_input``, the wire variants, ...) are recorded with no expectation."""

PROVENANCE_KEYS: dict[str, tuple[str, ...]] = {
    "engine": (
        "image",
        "image_digest",
        "vllm_version",
        "vllm_commit",
        "torch",
        "cuda",
        "driver",
        "gpus",
        "serve_argv",
        "env",
        "started",
        "ready_wait_s",
    ),
    "model": (
        "id",
        "revision",
        "weights",
        "tokenizer_sha256",
        "template_sha256",
        "plugin",
        "hf_overrides",
        "pooler_config",
        "mm_processor_kwargs",
        "dtype",
    ),
    "recipe": ("id", "file_sha256", "behaviour_fingerprint", "fingerprint_inputs", "status"),
    "collector": (
        "package",
        "version",
        "commit",
        "generator_version",
        "corpus_plan_version",
        "record_schema",
        "seed",
        "dataset_commits",
        "wave_id",
        "job_id",
        "started",
        "finished",
        "host_sha256",
    ),
}
"""OBSERVATIONS-SPEC section 4's provenance, block by block: every key carries its value or an explicit
``{"unavailable": "<reason>"}`` (:func:`rcp_ndcg_vllm.observe.provenance.unavailable`), never nothing."""


def corpus_key(engine_version: str, recipe_id: str, fingerprint: str) -> str:
    """The corpus key: ``vllm-<version>/<recipe>/<fingerprint>`` (GPU-VALIDATION.md item 8)."""
    return f"vllm-{engine_version}/{recipe_id}/{fingerprint}"


def corpus_path(root: str | Path, engine_version: str, recipe_id: str, fingerprint: str, recorded_at: str) -> Path:
    """Where one recording's full corpus lives under ``root`` (OBSERVATIONS-SPEC section 5).

    ``<root>/observations/vllm-<version>/<recipe>/<fingerprint>/<recorded-at>/``: immutable (a new recording is
    a new path), so uploading ``root`` to the results prefix lands it at the spec's storage path.
    """
    stamp = recorded_at.replace(":", "").replace("+", "Z")
    return Path(root) / "observations" / corpus_key(engine_version, recipe_id, fingerprint) / stamp


def expected_status(path: str, probe: str, engine_version: str = "0.31.0") -> int | None:
    """The status the engine version is measured to answer for one probe kind, or ``None`` (unmeasured).

    Probe kinds with a measured row: ``ok``, ``over_length``, ``unknown_field``, ``malformed_json``.
    """
    if engine_version != "0.31.0":
        return None
    table = _EXPECTED_STATUS_VLLM_0_31_0.get(path)
    return table.get(probe) if table else None


def build_record(
    *,
    sequence: int,
    repetition: str,
    batch_context: dict[str, Any],
    server_run_id: str,
    request: dict[str, Any],
    response: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """One ``RECORD_SCHEMA`` exchange record (OBSERVATIONS-SPEC section 3).

    Inputs: the order sent (``sequence``), the repetition (:data:`REPETITIONS`), the batch context (companion
    request ids and positions), the engine run's id, the request and the response (method/path, the headers
    that matter -- never authorisation or gateway headers -- and the body **as sent/received**, UTF-8 text or
    ``{"base64": ...}``, with its parsed JSON form when it is JSON) and what the request set meant (``inputs``:
    request id, stratum labels, layer, token counts, media).  Output: the record; ``exchange_id`` is the
    SHA-256 of the canonical request (method, path, body), a content address.  Raises :class:`HarnessError`
    for a repetition outside :data:`REPETITIONS`.
    """
    if repetition not in REPETITIONS:
        raise HarnessError(f"repetition {repetition!r} is not one of {REPETITIONS}")
    canonical = {"method": request.get("method"), "path": request.get("path"), "body": request.get("body_raw")}
    return {
        "record_schema": RECORD_SCHEMA,
        "exchange_id": hash_payload(canonical),
        "sequence": int(sequence),
        "repetition": repetition,
        "batch_context": batch_context,
        "server_run_id": server_run_id,
        "request": request,
        "response": response,
        "inputs": inputs,
    }


def write_corpus(out_dir: str | Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    """Write one corpus: ``records.jsonl``, then ``manifest.json`` hash-chained over every other file.

    Inputs: the output directory (``nondeterminism.json`` written there first is hashed too), the manifest's
    provenance blocks and the records.  The manifest records the SHA-256 and size of every corpus file, the
    record count, and its own digest (``integrity.manifest_sha256``,
    :func:`rcp_ndcg.testing.corpus.manifest_digest`): a tampered file fails :func:`verify_corpus` by hash.
    Returns the manifest's path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORDS_FILE).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != MANIFEST_FILE
    }
    document: dict[str, Any] = {
        "schema": CORPUS_SCHEMA,
        "record_schema": RECORD_SCHEMA,
        "normalisation_version": NORMALISATION_VERSION,
        **manifest,
        "integrity": {"files": files, "records_count": len(records)},
    }
    document["integrity"]["manifest_sha256"] = manifest_digest(document)
    path = out / MANIFEST_FILE
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# derived views (recomputed, never stored)
# ---------------------------------------------------------------------------


def _raw_bytes(body_raw: Any) -> bytes:
    if isinstance(body_raw, dict) and "base64" in body_raw:
        import base64

        return base64.b64decode(body_raw["base64"])
    return str(body_raw or "").encode("utf-8")


def derived_vectors(record: dict[str, Any], *, dim: int | None = None) -> list[np.ndarray] | None:
    """The vectors one recorded reply carries, decoded by the product's own adapters (never re-implemented).

    A ``/v1/embeddings`` reply decodes through :class:`~rcp_ndcg.inference.adapters.embeddings.OpenAIEmbeddings`
    (float lists or base64 float32), a ``/pooling`` reply through
    :class:`~rcp_ndcg.inference.adapters.pooling.VllmPooling` -- float frames, base64 frames in the request's
    ``embed_dtype`` reshaped by ``dim``, and ``bytes`` frames split by the recorded ``metadata`` header.

    Inputs: one record and the recipe's declared token-vector width (``client.dim``; a base64 ``/pooling``
    frame needs it).  Output: one array per input item (``(dim,)`` dense, ``(tokens, dim)`` per token), or
    ``None`` for a reply that carries no vectors (another route, a refusal).  Raises :class:`HarnessError`
    naming the record when a vector reply does not decode (an undecodable body is never skipped silently).
    """
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.inference.adapters.embeddings import OpenAIEmbeddings
    from rcp_ndcg.inference.adapters.pooling import VllmPooling
    from rcp_ndcg.inference.types import EmbedRequest, EncodeRole, PoolRequest, Reply

    request, response = record.get("request") or {}, record.get("response") or {}
    path = str(request.get("path", ""))
    if response.get("status") != 200 or path not in ("/v1/embeddings", "/pooling"):
        return None
    sent = request.get("body_parsed") or {}
    inputs = sent.get("input")
    if not isinstance(inputs, list) or not inputs:
        return None
    contents = tuple(Content.from_text("x") for _ in inputs)
    parsed = response.get("body_parsed")
    body: Any = parsed if parsed is not None else _raw_bytes(response.get("body_raw"))
    reply = Reply(status=200, body=body, headers=dict(response.get("headers") or {}))
    try:
        if path == "/v1/embeddings":
            embeddings = OpenAIEmbeddings().interpret(
                EmbedRequest(contents=contents, role=EncodeRole.DOCUMENT), [reply]
            )
            return [np.asarray(row, dtype=np.float64) for row in np.asarray(embeddings.vectors)]
        dtype = str(sent.get("embed_dtype") or "float16")
        pool = PoolRequest(
            contents=contents,
            role=EncodeRole.DOCUMENT,
            embed_dtype="float32" if dtype == "float32" else "float16",
            dim=dim,
        )
        embeddings = VllmPooling().interpret(pool, [reply])
    except (RcpNdcgError, NotImplementedError, ValueError) as error:
        raise HarnessError(
            f"exchange {record.get('exchange_id')}: the vector reply does not decode: {error}"
        ) from error
    matrix = np.asarray(embeddings.vectors, dtype=np.float64)
    offsets = embeddings.offsets
    if offsets is None:
        return [row for row in matrix]
    return [matrix[int(offsets[index]) : int(offsets[index + 1])] for index in range(len(offsets) - 1)]


def _numeric_leaves(node: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf of a parsed body with its dotted path (booleans excluded)."""
    out: list[tuple[str, float]] = []
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        out.append((prefix or ".", float(node)))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            out.extend(_numeric_leaves(item, f"{prefix}[{index}]"))
    elif isinstance(node, dict):
        for key, value in node.items():
            out.extend(_numeric_leaves(value, f"{prefix}.{key}"))
    return out


def _comparable(record: dict[str, Any], dim: int | None) -> dict[str, float]:
    """One reply as ``{leaf path: value}``: the normalised body's numeric leaves, plus the decoded vectors'
    components (a base64 or bytes frame carries its numbers only once decoded)."""
    response = record.get("response") or {}
    leaves = dict(_numeric_leaves(normalise_body(response.get("body_parsed"))))
    try:
        vectors = derived_vectors(record, dim=dim)
    except HarnessError:
        vectors = None
    for item, vector in enumerate(vectors or []):
        for position, value in enumerate(np.asarray(vector).reshape(-1)):
            leaves[f"vector[{item}][{position}]"] = float(value)
    return leaves


def summarise_nondeterminism(records: list[dict[str, Any]], *, dim: int | None = None) -> dict[str, Any]:
    """The measured non-determinism over repeated sendings of the same request (OBSERVATIONS-SPEC section 2).

    Inputs: the corpus's records and the recipe's declared token-vector width (for base64 ``/pooling``
    frames).  Records are grouped by ``exchange_id`` -- the same request bytes, so the same input in the same
    batch context -- and only a group with at least two sendings is a measurement.  Per group: the
    repetitions and server runs it spans, the maximum absolute and relative differences of every numeric leaf
    of the normalised bodies and of every decoded vector component, the minimum cosine per vector, and
    whether statuses or normalised bodies differ at all.  Output: ``{"per_input", "summary", "derived"}``;
    ``derived`` holds the verification tolerances under :data:`DERIVED_TOLERANCE_RULE`, or
    ``measured: false`` and no tolerance when no request was sent twice.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("exchange_id")), []).append(record)
    per_input: dict[str, Any] = {}
    worst_abs = worst_rel = 0.0
    min_cosine: float | None = None
    measured = 0
    bodies_differ = statuses_differ = False
    for exchange_id, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        measured += 1
        reference = _comparable(group[0], dim)
        max_abs = max_rel = 0.0
        for other in group[1:]:
            leaves = _comparable(other, dim)
            for key in set(reference) | set(leaves):
                a, b = reference.get(key), leaves.get(key)
                if a is None or b is None:
                    continue
                delta = abs(a - b)
                max_abs = max(max_abs, delta)
                max_rel = max(max_rel, delta / max(abs(a), abs(b), 1e-12))
        cosines = _cosines(group, dim)
        statuses = {row["response"].get("status") for row in group}
        normalised = {
            json.dumps(normalise_body(row["response"].get("body_parsed")), sort_keys=True)
            if row["response"].get("body_parsed") is not None
            else json.dumps(row["response"].get("body_raw"), sort_keys=True)
            for row in group
        }
        entry = {
            "repetitions": [row.get("repetition") for row in group],
            "server_run_ids": sorted({str(row.get("server_run_id")) for row in group}),
            "batch_context": group[0].get("batch_context"),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "min_cosine": min(cosines) if cosines else None,
            "statuses_differ": len(statuses) > 1,
            "bodies_differ": len(normalised) > 1,
        }
        per_input[exchange_id] = entry
        worst_abs, worst_rel = max(worst_abs, max_abs), max(worst_rel, max_rel)
        if cosines:
            min_cosine = min(cosines) if min_cosine is None else min(min_cosine, *cosines)
        bodies_differ = bodies_differ or entry["bodies_differ"]
        statuses_differ = statuses_differ or entry["statuses_differ"]
    derived: dict[str, Any] = {
        "rule": DERIVED_TOLERANCE_RULE,
        "measured": measured > 0,
        "repeated_requests": measured,
        "statuses_differ": statuses_differ,
        "bodies_differ": bodies_differ,
        "abs_tolerance": max(3.0 * worst_abs, 1e-9) if measured else None,
        "rel_tolerance": max(3.0 * worst_rel, 1e-6) if measured else None,
    }
    return {
        "normalisation_version": NORMALISATION_VERSION,
        "per_input": per_input,
        "summary": {"max_abs_diff": worst_abs, "max_rel_diff": worst_rel, "min_cosine": min_cosine},
        "derived": derived,
    }


def _cosines(group: list[dict[str, Any]], dim: int | None) -> list[float]:
    """The cosine of every decoded vector of the first sending against the same vector in each other one."""
    try:
        first = derived_vectors(group[0], dim=dim)
    except HarnessError:
        return []
    out: list[float] = []
    for other in group[1:]:
        try:
            vectors = derived_vectors(other, dim=dim)
        except HarnessError:
            continue
        for a, b in zip(first or [], vectors or [], strict=False):
            a2, b2 = (
                np.asarray(a).reshape(-1, np.asarray(a).shape[-1]),
                np.asarray(b).reshape(-1, np.asarray(a).shape[-1]),
            )
            for row_a, row_b in zip(a2, b2, strict=False):
                norm = float(np.linalg.norm(row_a) * np.linalg.norm(row_b))
                out.append(float(np.dot(row_a, row_b) / norm) if norm else 1.0)
    return out


# ---------------------------------------------------------------------------
# the acceptance checks (OBSERVATIONS-SPEC section 6)
# ---------------------------------------------------------------------------


def _missing_provenance(manifest: dict[str, Any]) -> list[str]:
    """Every provenance key of :data:`PROVENANCE_KEYS` with neither a value nor an ``unavailable`` reason."""
    missing: list[str] = []
    for block, keys in PROVENANCE_KEYS.items():
        section = manifest.get(block)
        if not isinstance(section, dict):
            missing.append(block)
            continue
        for key in keys:
            value = section.get(key)
            if value is None or (isinstance(value, dict) and "unavailable" in value and not value["unavailable"]):
                missing.append(f"{block}.{key}")
    return missing


def verify_corpus(
    corpus_dir: str | Path,
    *,
    plan: list[str] | None = None,
    equivalence_exchanges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The acceptance checks (OBSERVATIONS-SPEC section 6); a corpus failing any of them is not used.

    Inputs: the corpus directory (a full corpus or a repository subset), optionally the request plan's ids (by
    default the manifest's ``plan.request_ids``) and the equivalence stage's captured exchanges of the same
    wave.  Output: ``{"passed": bool, "checks": [...]}``; every check row names its referent.  Checks, in
    order: ``manifest_hashes`` (a tampered file fails here), ``completeness`` (every planned request id;
    a subset is a sample and defers to its full corpus), ``statuses_as_expected``,
    ``scores_finite_and_vector_widths`` (every number finite, every vector decoded by the product's adapter
    at the declared width and dtype, one width per route), ``nondeterminism_report``,
    ``provenance_complete``, ``no_credentials``, and -- given the equivalence exchanges --
    ``equivalence_consistent`` (the same served replies, within the measured tolerance).
    """
    directory = Path(corpus_dir)
    try:
        corpus = load_corpus(directory)
    except RcpNdcgError as error:
        return {
            "passed": False,
            "checks": [
                {
                    "check": "manifest_hashes",
                    "passed": False,
                    "referent": "the corpus's files as recorded in its manifest (or a subset's index)",
                    "mismatches": [str(error)],
                }
            ],
        }
    checks: list[dict[str, Any]] = []

    def record_check(name: str, passed: bool, referent: str, **detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "referent": referent, **detail})

    mismatches = integrity_mismatches(corpus)
    record_check(
        "manifest_hashes",
        not mismatches,
        "the SHA-256 of every corpus file and of the manifest itself (a subset: its index's hashes)",
        mismatches=mismatches,
    )
    _completeness(corpus, plan, record_check)
    _statuses(corpus, record_check)
    _numbers(corpus, record_check)
    nondeterminism = _read_json(directory / NONDETERMINISM_FILE)
    record_check(
        "nondeterminism_report",
        isinstance(nondeterminism, dict) and isinstance(nondeterminism.get("derived"), dict),
        "nondeterminism.json with the measured deltas and the derived tolerances (OBSERVATIONS-SPEC section 2)",
    )
    missing = _missing_provenance(corpus.manifest)
    record_check(
        "provenance_complete",
        not missing,
        "every OBSERVATIONS-SPEC section 4 provenance key carries its value or an explicit unavailable reason",
        missing=missing,
    )
    hits: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            text = (gzip.decompress(raw) if path.name.endswith(".gz") else raw).decode("utf-8", errors="replace")
            hits.extend(f"{path.name}: {name}" for name in credential_findings(text))
    record_check("no_credentials", not hits, "every corpus byte scanned for credential shapes", hits=hits)
    if equivalence_exchanges is not None:
        _consistency(corpus, equivalence_exchanges, nondeterminism, record_check)
    return {"passed": bool(all(check["passed"] for check in checks)), "checks": checks}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _completeness(corpus: ObservationCorpus, plan: list[str] | None, record_check: Any) -> None:
    """Every planned request id appears in the corpus (a subset defers to its full corpus)."""
    if corpus.subset_index is not None:
        record_check(
            "completeness",
            True,
            "a repository subset is a stratified sample; completeness is checked on its full corpus",
            deferred_to=(corpus.subset_index.get("full_corpus") or {}).get("path"),
        )
        return
    planned = plan if plan is not None else (corpus.manifest.get("plan") or {}).get("request_ids")
    if planned is None:
        record_check("completeness", False, "the manifest records no request plan to check against", missing=[])
        return
    seen = {str(row.get("inputs", {}).get("request_id", "")) for row in corpus.records}
    seen |= {rid for row in corpus.records for rid in row.get("batch_context", {}).get("request_ids", [])}
    missing = sorted(set(planned) - seen)
    record_check("completeness", not missing, "every request id of the generator's plan appears", missing=missing)


def _statuses(corpus: ObservationCorpus, record_check: Any) -> None:
    engine_version = str((corpus.manifest.get("engine") or {}).get("version", "")).removeprefix("v")
    bad: list[dict[str, Any]] = []
    for row in corpus.records:
        path = str(row.get("request", {}).get("path", ""))
        probe = str(row.get("inputs", {}).get("probe", "ok"))
        expected = expected_status(path, probe, engine_version)
        status = row.get("response", {}).get("status")
        if expected is not None and status != expected:
            bad.append(
                {
                    "exchange_id": row.get("exchange_id"),
                    "path": path,
                    "probe": probe,
                    "status": status,
                    "expected": expected,
                }
            )
    record_check(
        "statuses_as_expected",
        not bad,
        "the measured vLLM status table: over-length 400, unknown field ignored with 200, malformed JSON 400",
        mismatches=bad,
    )


def _numbers(corpus: ObservationCorpus, record_check: Any) -> None:
    """Every numeric leaf finite; every vector reply decoded at the declared width; one width per route."""
    recipe_block = corpus.manifest.get("recipe") or {}
    dim = recipe_block.get("declared_dim") if isinstance(recipe_block.get("declared_dim"), int) else None
    problems: list[str] = []
    widths: dict[str, int] = {}
    for row in corpus.records:
        exchange = str(row.get("exchange_id"))
        for leaf, value in _numeric_leaves(row.get("response", {}).get("body_parsed")):
            if not math.isfinite(value):
                problems.append(f"{exchange}:{leaf} is not finite")
        try:
            vectors = derived_vectors(row, dim=dim)
        except HarnessError as error:
            problems.append(str(error))
            continue
        path = str(row.get("request", {}).get("path", ""))
        for vector in vectors or []:
            array = np.asarray(vector)
            if not np.all(np.isfinite(array)):
                problems.append(f"{exchange}: a decoded vector is not finite")
            width = int(array.shape[-1]) if array.ndim else 0
            if dim is not None and width != dim:
                problems.append(f"{exchange}: vector width {width} is not the declared {dim}")
            widths.setdefault(path, width)
            if widths[path] != width:
                problems.append(f"{exchange}: vector width {width} differs from {widths[path]} on {path}")
    record_check(
        "scores_finite_and_vector_widths",
        not problems,
        "every numeric leaf finite; every vector reply decoded by the product's adapter at the declared width "
        "and dtype; one width per route",
        mismatches=problems[:50],
    )


def _consistency(
    corpus: ObservationCorpus,
    exchanges: list[dict[str, Any]],
    nondeterminism: Any,
    record_check: Any,
) -> None:
    """The equivalence stage's served replies agree with the corpus's replies to the same requests."""
    from ..record import entry_from_exchange

    derived = (nondeterminism or {}).get("derived") or {}
    tolerance = derived.get("abs_tolerance") if derived.get("measured") else 0.0
    dim = (corpus.manifest.get("recipe") or {}).get("declared_dim")
    by_id: dict[str, dict[str, Any]] = {}
    for row in corpus.records:
        by_id.setdefault(str(row.get("exchange_id")), row)
    compared = 0
    mismatches: list[str] = []
    for exchange in exchanges:
        request, response = entry_from_exchange(exchange)
        candidate = build_record(
            sequence=0,
            repetition="same_process_1",
            batch_context={},
            server_run_id="equivalence",
            request=request,
            response=response,
            inputs={},
        )
        stored = by_id.get(candidate["exchange_id"])
        if stored is None:
            continue
        compared += 1
        if stored["response"].get("status") != response.get("status"):
            mismatches.append(
                f"{candidate['exchange_id']}: status {response.get('status')} vs {stored['response'].get('status')}"
            )
            continue
        ours, theirs = _comparable(stored, dim), _comparable(candidate, dim)
        worst = max((abs(ours[key] - theirs[key]) for key in set(ours) & set(theirs)), default=0.0)
        if set(ours) != set(theirs) or worst > float(tolerance or 0.0):
            mismatches.append(f"{candidate['exchange_id']}: replies differ by {worst} (tolerance {tolerance})")
    record_check(
        "equivalence_consistent",
        compared > 0 and not mismatches,
        "the equivalence stage's replies equal the corpus's replies to the same requests (within the measured "
        "tolerance); no common request is a failure, never a vacuous pass",
        compared=compared,
        mismatches=mismatches[:50],
    )


# ---------------------------------------------------------------------------
# the repository subset (OBSERVATIONS-SPEC section 5)
# ---------------------------------------------------------------------------


def subset_for_repository(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    gcs_path: str,
    per_recipe_bytes: int = 2 * 1024 * 1024,
) -> Path:
    """The repository subset of a full corpus: a stratified, deterministic sample within the byte budget.

    Inputs: the full corpus, the output directory and the full corpus's immutable storage path.  The sample
    takes the first record of every stratum (``inputs.stratum``) first -- every stratum is covered or the
    call fails --, then fills in corpus order until the budget (bytes of ``records.jsonl.gz``), and keeps the
    chosen records in corpus order.  Output: the subset directory: ``records.jsonl.gz``, the full corpus's
    ``manifest.json`` and ``nondeterminism.json`` byte for byte, and ``index.json``
    (:func:`rcp_ndcg.testing.corpus.write_subset_index`) with the recorded sampling.  Raises
    :class:`HarnessError` when the strata alone do not fit the budget.
    """
    source = Path(corpus_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(source)
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in corpus.records]
    first_of: dict[str, int] = {}
    for index, row in enumerate(corpus.records):
        first_of.setdefault(str(row.get("inputs", {}).get("stratum", "")), index)
    chosen = set(first_of.values())

    def compressed(indexes: set[int]) -> bytes:
        return gzip.compress("".join(lines[index] for index in sorted(indexes)).encode("utf-8"), mtime=0)

    if len(compressed(chosen)) > per_recipe_bytes:
        raise HarnessError(
            f"the first record of each of {len(chosen)} strata already exceeds the {per_recipe_bytes}-byte "
            "subset budget: the repository subset cannot cover every stratum"
        )
    for index in range(len(lines)):
        if index in chosen:
            continue
        trial = chosen | {index}
        if len(compressed(trial)) > per_recipe_bytes:
            break
        chosen = trial
    (out / RECORDS_FILE_GZ).write_bytes(compressed(chosen))
    shutil.copyfile(source / MANIFEST_FILE, out / MANIFEST_FILE)
    if (source / NONDETERMINISM_FILE).is_file():
        shutil.copyfile(source / NONDETERMINISM_FILE, out / NONDETERMINISM_FILE)
    write_subset_index(
        out,
        full_corpus_path=gcs_path,
        sampling={
            "rule": "the first record of every stratum, then corpus order until the budget; corpus order kept",
            "budget_bytes": per_recipe_bytes,
            "strata_covered": sorted(first_of),
            "rows": len(chosen),
            "of_rows": len(lines),
        },
    )
    return out


# ---------------------------------------------------------------------------
# change handling (OBSERVATIONS-SPEC section 7)
# ---------------------------------------------------------------------------


def changed_since(recipes: list[Recipe], corpus_index: dict[str, Any], *, engine_version_of: Any) -> dict[str, Any]:
    """Section 7's re-record-changed-only: the recipes whose corpus key moved since the previous index.

    Inputs: the recipes, the previous index (a ``wave.json``: ``fingerprints`` and ``engine_versions``, each
    ``{recipe_id: value}``) and a callable giving each recipe's engine version.  A recipe is recorded again
    when its behaviour fingerprint (:func:`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`) or its engine
    version differs from the index's -- the corpus key is both.  Output: ``{"changed", "unchanged",
    "changes", "protocol_due"}``: ``changes`` names why per changed recipe, ``protocol_due`` the engine
    versions the index has no recording of (their protocol layer is recorded once).
    """
    fingerprints = corpus_index.get("fingerprints") or {}
    versions = corpus_index.get("engine_versions") or {}
    changed: list[str] = []
    unchanged: list[str] = []
    changes: dict[str, list[str]] = {}
    for recipe in recipes:
        reasons: list[str] = []
        if fingerprints.get(recipe.id) != behaviour_fingerprint(recipe):
            reasons.append("behaviour_fingerprint")
        if versions.get(recipe.id) != engine_version_of(recipe):
            reasons.append("engine_version")
        if reasons:
            changed.append(recipe.id)
            changes[recipe.id] = reasons
        else:
            unchanged.append(recipe.id)
    known = {str(version) for version in versions.values()}
    due = sorted({str(engine_version_of(recipe)) for recipe in recipes} - known)
    return {"changed": changed, "unchanged": unchanged, "changes": changes, "protocol_due": due}


def main(argv: list[str] | None = None) -> int:
    """``python -m rcp_ndcg_vllm.observe.corpus verify DIR`` (the acceptance checks, exit 1 on a refusal) and
    ``... subset DIR OUT --gcs-path URI`` (the repository subset of a full corpus, checked before it is kept)."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_vllm.observe.corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="run the acceptance checks on one corpus directory")
    verify.add_argument("corpus")
    subset = commands.add_parser("subset", help="write the repository subset of a full corpus")
    subset.add_argument("corpus")
    subset.add_argument("out")
    subset.add_argument("--gcs-path", required=True, help="the full corpus's immutable storage path")
    subset.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args(argv)
    if args.command == "subset":
        out = subset_for_repository(args.corpus, args.out, gcs_path=args.gcs_path, per_recipe_bytes=args.max_bytes)
        report = verify_corpus(out)
    else:
        report = verify_corpus(args.corpus)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
