"""The observation-corpus format (OBSERVATIONS-SPEC sections 2-7): raw records, the hash-chained
manifest, the acceptance checks and the change handling.

The corpus of one recording is a directory (OBSERVATIONS-SPEC section 5)::

    records.jsonl          -- one :data:`RECORD_SCHEMA` object per exchange (raw first)
    nondeterminism.json    -- the measured repeated-sending deltas and the derived tolerances
    manifest.json          -- full provenance, the file hashes, and the recipe's behaviour fingerprint
    index.json             -- (repository subset) the GCS path and hash of the full corpus it samples

Keyed by engine + version and the recipe's **behaviour fingerprint** (GPU-VALIDATION.md item 8): the
SHA-256 over the model id and revision, the ``serve`` block, the template file's hash, the tokenizer's
SHA-256 and the client fields that shape the request -- :func:`behaviour_fingerprint` reports the
changed inputs when two fingerprints differ (``rcp_ndcg.support.identity.identity_differences``).

Nothing derived is stored: parsed bodies ride inside a record because the raw bytes ride beside them;
vectors, normalised bodies and tolerances are recomputed by versioned code
(:data:`NORMALISATION_VERSION` names the volatile fields it strips).

Public surface:

- :data:`RECORD_SCHEMA`, :data:`NORMALISATION_VERSION`, :data:`CORPUS_SCHEMA`.
- :func:`behaviour_fingerprint`, :func:`build_record`, :func:`expected_status`.
- :func:`write_corpus`, :func:`load_corpus`, :func:`verify_corpus`.
- :func:`summarise_nondeterminism` (and :data:`DERIVED_TOLERANCE_RULE`), :func:`subset_for_repository`.
- :func:`changed_since` (OBSERVATIONS-SPEC section 7: re-record changed only).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from rcp_ndcg.support.identity import hash_payload, identity_differences, identity_payload

from ..errors import HarnessError
from ..recipe import Recipe

__all__ = [
    "CORPUS_SCHEMA",
    "DERIVED_TOLERANCE_RULE",
    "NORMALISATION_VERSION",
    "RECORD_SCHEMA",
    "behaviour_fingerprint",
    "build_record",
    "changed_since",
    "corpus_key",
    "expected_status",
    "load_corpus",
    "subset_for_repository",
    "summarise_nondeterminism",
    "verify_corpus",
    "write_corpus",
]

RECORD_SCHEMA = 1
"""The record version: one JSONL object per exchange (OBSERVATIONS-SPEC section 3).  A bump never
invalidates an old corpus -- readers support every schema they ever wrote."""

NORMALISATION_VERSION = 1
"""The version of the derived-view code; it names the volatile response fields it strips (request ids
and ``created`` timestamps, and the ``/v1/models`` permission rows' ids and timestamps)."""

CORPUS_SCHEMA = "rcp-ndcg-vllm.observation-corpus.v1"
"""The corpus manifest's schema identifier (the artifact's own version)."""

DERIVED_TOLERANCE_RULE = (
    "verification tolerance = 3x the measured maximum difference over the repeated sendings "
    "(same process twice and once after an engine restart), with absolute floor 1e-9 and relative "
    "floor 1e-6; stored beside the measured numbers, never chosen by hand"
)
"""How the measured non-determinism derives an emulator's verification tolerance (section 2)."""

VOLATILE_FIELDS = ("id", "created")
"""Response fields the normalisation strips across body kinds (request ids and creation timestamps)."""

CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)authorization[\"'\\s:=]+(bearer|basic)\\s"),
)
"""Credential shapes the acceptance check scans every corpus byte for (OBSERVATIONS-SPEC section 6:
no credential-shaped string anywhere)."""

_ROLE_ROUTES = {"embed": "/v1/embeddings", "multi_vector": "/pooling", "rerank": "/rerank"}

_EXPECTED_STATUS_VLLM_0_31_0: dict[str, dict[str, int]] = {
    "/v1/embeddings": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/pooling": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/rerank": {"ok": 200, "over_length": 400, "unknown_field": 200, "malformed_json": 400},
    "/v1/models": {"ok": 200},
    "/tokenize": {"ok": 200},
}
"""The vLLM v0.31.0 status table the waves measured (shake1c recordings, 12 recipes: an unknown
request field is IGNORED with 200 on the role routes, an over-length prompt is refused with 400 and
the OpenAI-style ``{"error": ...}`` body, malformed JSON is a 400).  A corpus records what its engine
answered; this table is what the acceptance check expects from the engine version it names."""


def corpus_key(engine_version: str, recipe_id: str, fingerprint: str) -> str:
    """The corpus key: ``<engine>-<version>/<recipe>/<fingerprint>`` (GPU-VALIDATION.md item 8)."""
    return f"vllm-{engine_version}/{recipe_id}/{fingerprint}"


def behaviour_fingerprint(
    recipe: Recipe, *, tokenizer_sha256: str | None, template_sha256: str | None
) -> dict[str, Any]:
    """The recipe's behaviour fingerprint and its full input list (GPU-VALIDATION.md item 8).

    Inputs: the loaded recipe, the SHA-256 of its tokenizer's ``tokenizer.json`` and of its template
    file (``None`` where the recipe has none).  Output: ``{"fingerprint": <sha256 hex>, "inputs":
    {...}}`` -- the fingerprint over everything that can change what the model returns (model id and
    revision, the ``serve`` block canonically, the template and tokenizer hashes, and the client's
    content-identity fields), and the input list with its values so a staleness failure can name what
    changed.  rcp-ndcg's own internals are deliberately not part of it.
    """
    from ..recipe import serve_argv  # noqa: F401 - documented: the serve block replaces the argv

    serve = json.loads(recipe.serve.model_dump_json(exclude={"plugin", "io_processor_plugin"}))
    plugin = {"plugin": recipe.serve.plugin, "io_processor_plugin": recipe.serve.io_processor_plugin}
    try:
        client = identity_payload(recipe.client)
    except TypeError as error:
        raise HarnessError(f"recipe {recipe.id}: the client block declares no identity roles ({error})") from error
    inputs = {
        "model": recipe.model,
        "revision": recipe.revision,
        "serve": serve,
        "plugin": plugin,
        "template_sha256": template_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "client": client,
    }
    return {"fingerprint": hash_payload(inputs), "inputs": inputs}


def fingerprint_changes(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """What changed between two fingerprint input lists (the staleness failure's named inputs)."""
    return identity_differences(recorded.get("inputs", recorded), current.get("inputs", current))


def expected_status(path: str, probe: str, engine_version: str = "0.31.0") -> int | None:
    """The status the engine version is measured to answer for one probe kind, or ``None`` (unmeasured).

    Probe kinds: ``ok``, ``over_length``, ``unknown_field``, ``malformed_json``.
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
    """One :data:`RECORD_SCHEMA` exchange record (OBSERVATIONS-SPEC section 3).

    Inputs: the order sent (``sequence``), the repetition label (``same_process`` / ``after_restart``),
    the batch context (companion request ids and positions), the engine run id, the request and the
    response (method/path, the headers that matter -- never authorisation or gateway headers -- and
    the body **as sent/received**, UTF-8 text or base64 bytes, with its parsed JSON form when it is
    JSON), and what the generator meant (``inputs``: source ids, stratum labels, token counts, media).
    Output: the record, with ``exchange_id`` the SHA-256 of the canonical request (a content address).
    ``derived`` is **not** stored: recomputed by versioned code on load.
    """
    canonical = {
        "method": request.get("method"),
        "path": request.get("path"),
        "body": request.get("body_raw"),
    }
    record = {
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
    return record


def _file_hashes(directory: Path, *, excluding: tuple[str, ...] = ("manifest.json",)) -> dict[str, str]:
    """``{filename: sha256}`` for every corpus file except the manifest itself."""
    out: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in excluding:
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def write_corpus(out_dir: str | Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    """Write one corpus: ``records.jsonl`` then ``manifest.json`` hash-chained over the files.

    Inputs: the output directory, the manifest fields (the provenance builders of the collector fill
    engine/model/recipe/collector blocks) and the records.  The manifest records ``records_count``,
    the SHA-256 and byte size of every corpus file (``integrity.files``) and then its own SHA-256 in
    ``integrity.manifest_sha256`` -- a tampered file fails :func:`verify_corpus` by hash.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    integrity = {
        "files": {
            name: {"sha256": digest, "bytes": (out / name).stat().st_size} for name, digest in _file_hashes(out).items()
        },
        "records_count": len(records),
    }
    document = {
        "schema": CORPUS_SCHEMA,
        "schema_version": 1,
        "record_schema": RECORD_SCHEMA,
        "normalisation_version": NORMALISATION_VERSION,
        **manifest,
        "integrity": integrity,
    }
    document["integrity"]["manifest_sha256"] = hash_payload(document)
    (out / "manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out / "manifest.json"


def load_corpus(corpus_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a corpus: ``(manifest, records)``.  Raises :class:`HarnessError` on an unreadable artifact."""
    directory = Path(corpus_dir)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"the corpus at {directory} is unreadable: {error}") from error
    return manifest, records


def summarise_nondeterminism(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The measured non-determinism over the repeated sendings (OBSERVATIONS-SPEC section 2).

    Inputs: the corpus's records (every exchange sent twice in the same server process and once after
    an engine restart).  Output: per input (``exchange_id``) and per batch context, the maximum
    absolute and relative differences of the numeric leaves (scores and vector components), the
    cosine per vector, and whether statuses or bodies differ at all -- plus the derived verification
    tolerances under :data:`DERIVED_TOLERANCE_RULE` (stored with the measured numbers).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("exchange_id")), []).append(record)
    per_input: dict[str, Any] = {}
    worst_abs = 0.0
    worst_rel = 0.0
    bodies_differ = False
    statuses_differ = False
    for exchange_id, group in sorted(groups.items()):
        pairs = _numeric_leaves([row["response"].get("body_parsed") or {} for row in group])
        max_abs = max((abs(a - b) for a, b in pairs), default=0.0)
        max_rel = max((abs(a - b) / max(abs(a), abs(b), 1e-12) for a, b in pairs), default=0.0)
        statuses = {row["response"].get("status") for row in group}
        raws = {json.dumps(row["response"].get("body_raw"), sort_keys=True) for row in group}
        entry = {
            "repetitions": [row.get("repetition") for row in group],
            "batch_contexts": [row.get("batch_context") for row in group],
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "statuses_differ": len(statuses) > 1,
            "bodies_differ": len(raws) > 1,
        }
        per_input[exchange_id] = entry
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        bodies_differ = bodies_differ or entry["bodies_differ"]
        statuses_differ = statuses_differ or entry["statuses_differ"]
    derived = {
        "rule": DERIVED_TOLERANCE_RULE,
        "abs_tolerance": max(3.0 * worst_abs, 1e-9),
        "rel_tolerance": max(3.0 * worst_rel, 1e-6),
        "statuses_differ": statuses_differ,
        "bodies_differ": bodies_differ,
    }
    return {
        "per_input": per_input,
        "summary": {"max_abs_diff": worst_abs, "max_rel_diff": worst_rel},
        "derived": derived,
    }


def _numeric_leaves(bodies: list[Any]) -> list[tuple[float, float]]:
    """Every ``(first, rest)``-pair of aligned numeric leaves across the bodies (scores and vectors)."""
    if len(bodies) < 2:
        return []
    first, rest = bodies[0], bodies[1:]
    pairs: list[tuple[float, float]] = []

    def walk(node: Any, others: list[Any]) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            for other in others:
                if isinstance(other, (int, float)) and not isinstance(other, bool):
                    pairs.append((float(node), float(other)))
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                if all(isinstance(other, list) and index < len(other) for other in others):
                    walk(item, [other[index] for other in others])
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if all(isinstance(other, dict) and key in other for other in others):
                    walk(value, [other[key] for other in others])

    walk(first, rest)
    return pairs


def verify_corpus(corpus_dir: str | Path, *, plan: list[str] | None = None) -> dict[str, Any]:
    """The acceptance checks (OBSERVATIONS-SPEC section 6); a corpus failing any of them is not used.

    Inputs: the corpus directory and (optionally) the generator plan's request ids for the
    completeness check.  Output: a report ``{"passed": bool, "checks": [...]}``; every check row names
    its referent.  Checks, in order: manifest hashes valid (a tampered file fails here), completeness
    against the plan, every status as the engine version is measured to answer, every score finite,
    every vector's width consistent, the non-determinism report present, and no credential-shaped
    string anywhere.
    """
    directory = Path(corpus_dir)
    try:
        manifest, records = load_corpus(directory)
    except HarnessError as error:
        # A corpus that cannot even be read is refused by the checks below, never raised past them.
        return {
            "passed": False,
            "checks": [
                {
                    "check": "manifest_hashes",
                    "passed": False,
                    "referent": "the corpus's files as recorded in manifest.json",
                    "mismatches": [str(error)],
                }
            ],
        }
    checks: list[dict[str, Any]] = []

    def record_check(name: str, passed: bool, referent: str, **detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "referent": referent, **detail})

    # 1. integrity: the manifest's hashes over the corpus files (a tampered file fails here)
    mismatches: list[str] = []
    integrity = manifest.get("integrity", {})
    for name, entry in integrity.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            mismatches.append(f"{name}: hash mismatch")
    if integrity.get("manifest_sha256"):
        # The manifest's own hash covers the document as written without the field itself.
        recomputed = hash_payload(
            {**manifest, "integrity": {k: v for k, v in integrity.items() if k != "manifest_sha256"}}
        )
        if recomputed != integrity.get("manifest_sha256"):
            mismatches.append("manifest.json: manifest_sha256 mismatch")
    record_check(
        "manifest_hashes",
        not mismatches,
        "the SHA-256 of every corpus file and of the manifest itself, as recorded in manifest.json",
        mismatches=mismatches,
    )

    # 2. completeness against the generator's plan
    if plan is not None:
        seen = {part for row in records for part in str(row.get("inputs", {}).get("request_id", "")).split("+")}
        missing = sorted(set(plan) - seen)
        record_check(
            "completeness",
            not missing,
            "every request id of the generator's plan must appear in the corpus",
            missing=missing,
        )

    # 3. statuses as measured for the engine version
    engine_version = str(manifest.get("engine", {}).get("version", "")).removeprefix("v")
    bad_statuses: list[dict[str, Any]] = []
    for row in records:
        path = str(row.get("request", {}).get("path", ""))
        probe = str(row.get("inputs", {}).get("probe", "ok"))
        expected = expected_status(path, probe, engine_version)
        status = row.get("response", {}).get("status")
        if expected is not None and status != expected:
            bad_statuses.append(
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
        not bad_statuses,
        "the measured vLLM status table (shake1c recordings): over-length 400, unknown field ignored with 200",
        mismatches=bad_statuses,
    )

    # 4. scores finite + 5. vector widths consistent
    nonfinite: list[str] = []
    widths: dict[str, int] = {}
    for row in records:
        body = row.get("response", {}).get("body_parsed")
        for leaf_path, value in _walk_numbers(body):
            if not math.isfinite(value):
                nonfinite.append(f"{row.get('exchange_id')}:{leaf_path}")
        for shape in _vector_shapes(body):
            widths.setdefault(shape["kind"], shape["width"])
            if widths[shape["kind"]] != shape["width"]:
                nonfinite.append(f"{row.get('exchange_id')}:vector-width {shape['width']} != {widths[shape['kind']]}")
    record_check(
        "scores_finite_and_vector_widths",
        not nonfinite,
        "every numeric leaf finite; one width per vector kind",
        mismatches=nonfinite,
    )

    # 6. the non-determinism report present
    nondeterminism_path = directory / "nondeterminism.json"
    nondeterminism_ok = nondeterminism_path.is_file() and "derived" in json.loads(
        nondeterminism_path.read_text(encoding="utf-8") or "{}"
    )
    record_check(
        "nondeterminism_report",
        nondeterminism_ok,
        "nondeterminism.json with the measured deltas and the derived tolerances (OBSERVATIONS-SPEC section 2)",
    )

    # 7. no credential-shaped string anywhere
    hits: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", errors="replace")
            for pattern in CREDENTIAL_PATTERNS:
                if pattern.search(text):
                    hits.append(f"{path.name}: {pattern.pattern}")
    record_check("no_credentials", not hits, "every corpus byte scanned for credential shapes", hits=hits)

    return {"passed": bool(all(check["passed"] for check in checks)), "checks": checks}


def _walk_numbers(body: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf of a parsed body with its dotted path."""
    out: list[tuple[str, float]] = []
    if isinstance(body, bool):
        return out
    if isinstance(body, (int, float)):
        out.append((prefix or ".", float(body)))
    elif isinstance(body, list):
        for index, item in enumerate(body):
            out.extend(_walk_numbers(item, f"{prefix}[{index}]"))
    elif isinstance(body, dict):
        for key, value in body.items():
            out.extend(_walk_numbers(value, f"{prefix}.{key}"))
    return out


def _vector_shapes(body: Any) -> list[dict[str, Any]]:
    """The vector matrices of a parsed body (``embedding``, ``embeddings``, ``data[].embedding``) with
    their width, for the width-consistency check."""
    shapes: list[dict[str, Any]] = []

    def consider(kind: str, node: Any) -> None:
        if isinstance(node, list) and node:
            if all(isinstance(item, (int, float)) for item in node):
                shapes.append({"kind": kind, "width": len(node)})
            elif all(isinstance(item, list) for item in node):
                for item in node:
                    consider(kind, item)

    if isinstance(body, dict):
        for key in ("embedding", "embeddings", "vectors"):
            if key in body:
                consider(key, body[key])
        data = body.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "embedding" in item:
                    consider("embedding", item["embedding"])
    return shapes


def subset_for_repository(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    gcs_path: str,
    per_recipe_bytes: int = 2 * 1024 * 1024,
) -> Path:
    """The repository subset of a corpus (OBSERVATIONS-SPEC section 5): compressed JSONL, stratified.

    Inputs: the full corpus directory, where to write the subset and the GCS path of the full corpus.
    Output: the subset directory (``records.jsonl.gz``, the manifest, and ``index.json`` naming the
    GCS path and hash of the full corpus plus the recorded sampling).  The selection is deterministic
    (every stratum's first rows, then sequential) and stops at the byte budget; the sampling record
    is written, never silent.
    """
    source = Path(corpus_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, records = load_corpus(source)
    chosen: list[dict[str, Any]] = []
    seen_strata: set[str] = set()
    size = 0
    sampling = {"budget_bytes": per_recipe_bytes, "strata_covered": [], "rows": 0}
    for row in records:
        stratum = str(row.get("inputs", {}).get("stratum", ""))
        first_of_stratum = stratum not in seen_strata
        if not first_of_stratum and len(chosen) >= _min_coverage(records):
            break
        seen_strata.add(stratum)
        if first_of_stratum:
            sampling["strata_covered"].append(stratum)
        payload = json.dumps(row, ensure_ascii=False) + "\n"
        size += len(payload.encode("utf-8"))
        if size > per_recipe_bytes:
            break
        chosen.append(row)
    sampling["rows"] = len(chosen)

    with gzip.open(out / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen))
    full_manifest_hash = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index = {
        "schema": "rcp-ndcg-vllm.corpus-index.v1",
        "full_corpus": {"path": gcs_path, "manifest_sha256": full_manifest_hash},
        "sampling": sampling,
        "files": {
            name: {"sha256": digest, "bytes": (out / name).stat().st_size}
            for name, digest in _file_hashes(out, excluding=("index.json",)).items()
        },
        "integrity": {"manifest_sha256": hash_payload({**manifest})},
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _min_coverage(records: list[dict[str, Any]]) -> int:
    """The number of distinct strata in the corpus (the subset keeps at least every stratum's first row)."""
    return len({str(row.get("inputs", {}).get("stratum", "")) for row in records})


def changed_since(
    recipes: list[Recipe], corpus_index: dict[str, Any], *, tokenizer_sha256_of: Any
) -> dict[str, list[str]]:
    """Section 7's re-record-changed-only: the recipes whose behaviour fingerprint changed.

    Inputs: the recipes, the previous corpus ``index.json`` (``{recipe_id: fingerprint}`` under
    ``fingerprints``) and a callable returning each recipe's tokenizer SHA-256.  Output
    ``{"changed": [...], "unchanged": [...]}`` by recipe id -- only the changed are recorded again.
    """
    recorded = corpus_index.get("fingerprints", {})
    changed: list[str] = []
    unchanged: list[str] = []
    for recipe in recipes:
        current = behaviour_fingerprint(recipe, tokenizer_sha256=tokenizer_sha256_of(recipe), template_sha256=None)
        if recorded.get(recipe.id) == current["fingerprint"]:
            unchanged.append(recipe.id)
        else:
            changed.append(recipe.id)
    return {"changed": changed, "unchanged": unchanged}
