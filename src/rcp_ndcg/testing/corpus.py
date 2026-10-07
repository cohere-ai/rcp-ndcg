"""Observation corpora: the on-disk format and its one reader (OBSERVATIONS-SPEC sections 3-5 and 7).

A GPU wave records, per recipe, the raw exchanges with a served engine (``rcp-ndcg-vllm``'s collector writes
them); the verified fake engines and their conformance suite read them back on CPU. This module is the
format's one home, shared by the writer and every reader: the schema identifiers, :func:`load_corpus` (which
reads a full corpus's ``records.jsonl`` and a repository subset's ``records.jsonl.gz`` alike, and migrates
records of an older ``RECORD_SCHEMA``), the integrity check, the normalisation comparisons apply
(:func:`normalise_body`) and the credential scan every corpus byte must pass.

A corpus is one directory:

- ``records.jsonl`` -- one record per exchange (``RECORD_SCHEMA``): the request and the response as they
  crossed the wire (``body_raw``: UTF-8 text, or ``{"base64": ...}`` for bytes that are not), the headers
  that matter, ``exchange_id`` (the SHA-256 of the canonical request), ``sequence``, ``repetition``,
  ``batch_context``, ``server_run_id`` and ``inputs`` (what the request set meant by it);
- ``nondeterminism.json`` -- the measured differences between repeated sendings and the tolerances derived
  from them;
- ``manifest.json`` -- the provenance and ``integrity``: the SHA-256 and size of every other file, the record
  count, and the manifest's own digest (:func:`manifest_digest`).

A **repository subset** (``tests/contract/engines/<engine>-<version>/<recipe>/<fingerprint>/``) carries
``records.jsonl.gz`` (a stratified sample), the full corpus's ``manifest.json`` byte for byte, its
``nondeterminism.json`` and an ``index.json`` (:func:`write_subset_index`) naming where the full corpus lives
and hashing the subset's files; its integrity is the index's.

Nothing derived is stored: normalised bodies, decoded vectors and tolerances are recomputed by versioned code
(:data:`NORMALISATION_VERSION` names what :func:`normalise_body` strips).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rcp_ndcg.errors import DataError
from rcp_ndcg.support.identity import hash_payload

__all__ = [
    "CORPUS_SCHEMA",
    "INDEX_FILE",
    "MANIFEST_FILE",
    "NONDETERMINISM_FILE",
    "NORMALISATION_VERSION",
    "RECORDS_FILE",
    "RECORDS_FILE_GZ",
    "RECORD_SCHEMA",
    "SUBSET_INDEX_SCHEMA",
    "ObservationCorpus",
    "credential_findings",
    "integrity_mismatches",
    "load_corpus",
    "manifest_digest",
    "normalise_body",
    "register_record_migration",
    "write_subset_index",
]

CORPUS_SCHEMA = "rcp-ndcg.observation-corpus/1"
"""The corpus manifest's ``schema``: the format of the directory as a whole."""

SUBSET_INDEX_SCHEMA = "rcp-ndcg.observation-subset/1"
"""The repository subset's ``index.json`` ``schema``."""

RECORD_SCHEMA = 1
"""The version of one exchange record. A bump never invalidates an old corpus: :func:`load_corpus` migrates
every older record through the migrations registered with :func:`register_record_migration`."""

NORMALISATION_VERSION = 1
"""The version of :func:`normalise_body`'s rule. Version 1 strips the volatile leaves of a reply: the top-level
``id`` and ``created`` (request ids and timestamps), ``data[*].created``, and the ``/v1/models`` permission
rows' ``id`` and ``created``. A changed rule is a new version, never a silent edit of this one."""

RECORDS_FILE = "records.jsonl"
RECORDS_FILE_GZ = "records.jsonl.gz"
MANIFEST_FILE = "manifest.json"
NONDETERMINISM_FILE = "nondeterminism.json"
INDEX_FILE = "index.json"

_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""``{schema: migrate}``: each migration takes one record of ``schema`` to ``schema + 1``."""

_CREDENTIALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "authorization header",
        re.compile(r"(?i)authorization[\"']?\s*[:=]\s*[\"']?(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}"),
    ),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}")),
    ("google oauth token", re.compile(r"\bya29\.[0-9A-Za-z_-]{10,}")),
    ("secret key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
"""The credential shapes no corpus byte may carry (OBSERVATIONS-SPEC section 6), by name."""


@dataclass(frozen=True)
class ObservationCorpus:
    """One corpus as read from disk.

    Attributes:
        directory: The corpus directory.
        manifest: ``manifest.json`` as parsed (for a subset: the full corpus's manifest, copied byte for byte).
        records: The exchange records, migrated to :data:`RECORD_SCHEMA`, in file order.
        records_file: The file the records came from (``records.jsonl`` or ``records.jsonl.gz``).
        subset_index: The subset's ``index.json`` when this is a repository subset, else ``None``.
    """

    directory: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    records_file: str
    subset_index: dict[str, Any] | None = None


def register_record_migration(schema: int, migrate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Register the migration of one record from ``schema`` to ``schema + 1`` (applied by :func:`load_corpus`).

    Args:
        schema: The record schema the migration reads.
        migrate: A function from one record of ``schema`` to the same record at ``schema + 1``.
    """
    _MIGRATIONS[schema] = migrate


def _migrated(record: dict[str, Any], where: str) -> dict[str, Any]:
    """One record at :data:`RECORD_SCHEMA`, migrated forward step by step; refused when that is impossible."""
    schema = record.get("record_schema")
    if not isinstance(schema, int):
        raise DataError(f"{where}: the record carries no integer record_schema")
    if schema > RECORD_SCHEMA:
        raise DataError(
            f"{where}: record_schema {schema} was written by a newer collector; this reader knows {RECORD_SCHEMA}",
            hint="upgrade rcp-ndcg to read this corpus",
        )
    while schema < RECORD_SCHEMA:
        migrate = _MIGRATIONS.get(schema)
        if migrate is None:
            raise DataError(f"{where}: no migration from record_schema {schema} to {schema + 1} is registered")
        record = migrate(record)
        schema += 1
    return record


def load_corpus(directory: str | Path) -> ObservationCorpus:
    """Read one corpus directory (a full corpus or a repository subset).

    Args:
        directory: The corpus directory.

    Returns:
        The :class:`ObservationCorpus`; its records are migrated to :data:`RECORD_SCHEMA`.

    Raises:
        DataError: the manifest or the records are missing or do not parse, the manifest names another format,
            or a record's schema is newer than this reader or has no migration path.
    """
    root = Path(directory)
    try:
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataError(f"{root}: the corpus manifest is unreadable: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != CORPUS_SCHEMA:
        found = manifest.get("schema") if isinstance(manifest, dict) else type(manifest).__name__
        raise DataError(f"{root}: manifest schema {found!r} is not {CORPUS_SCHEMA!r}")
    if (root / RECORDS_FILE).is_file():
        name = RECORDS_FILE
        opener: Callable[[], str] = lambda: (root / RECORDS_FILE).read_text(encoding="utf-8")  # noqa: E731
    elif (root / RECORDS_FILE_GZ).is_file():
        name = RECORDS_FILE_GZ
        opener = lambda: gzip.decompress((root / RECORDS_FILE_GZ).read_bytes()).decode("utf-8")  # noqa: E731
    else:
        raise DataError(f"{root}: neither {RECORDS_FILE} nor {RECORDS_FILE_GZ} exists")
    records: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(opener().splitlines(), start=1):
            if not line.strip():
                raise DataError(f"{root / name}:{number}: an empty line (a corpus file is one record per line)")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DataError(f"{root / name}:{number}: a record is a JSON object")
            records.append(_migrated(row, f"{root / name}:{number}"))
    except (OSError, ValueError, EOFError) as error:
        raise DataError(f"{root / name}: unreadable: {error}") from error
    subset_index = None
    if name == RECORDS_FILE_GZ and (root / INDEX_FILE).is_file():
        try:
            subset_index = json.loads((root / INDEX_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DataError(f"{root / INDEX_FILE}: unreadable: {error}") from error
    return ObservationCorpus(
        directory=root, manifest=manifest, records=records, records_file=name, subset_index=subset_index
    )


def manifest_digest(manifest: dict[str, Any]) -> str:
    """The manifest's own digest: the SHA-256 of its canonical JSON without ``integrity.manifest_sha256``.

    Args:
        manifest: A parsed corpus manifest.

    Returns:
        The hex digest the writer stores in ``integrity.manifest_sha256``.
    """
    integrity = {key: value for key, value in (manifest.get("integrity") or {}).items() if key != "manifest_sha256"}
    return hash_payload({**manifest, "integrity": integrity})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrity_mismatches(corpus: ObservationCorpus) -> list[str]:
    """What fails the corpus's integrity check (``[]`` when every hash holds).

    A full corpus checks every file its manifest hashes and the manifest's own digest. A repository subset
    checks every file its ``index.json`` hashes, and that its ``manifest.json`` is the full corpus's, byte for
    byte (the index records the full manifest's SHA-256).

    Args:
        corpus: A corpus from :func:`load_corpus`.

    Returns:
        One line per mismatch, naming the file.
    """
    root = corpus.directory
    mismatches: list[str] = []
    if corpus.subset_index is not None:
        files = corpus.subset_index.get("files") or {}
        if RECORDS_FILE_GZ not in files:
            mismatches.append(f"{INDEX_FILE}: {RECORDS_FILE_GZ} is not hashed")
        expected_manifest = (corpus.subset_index.get("full_corpus") or {}).get("manifest_file_sha256")
        if expected_manifest != _sha256(root / MANIFEST_FILE):
            mismatches.append(f"{MANIFEST_FILE}: not the full corpus's manifest (its SHA-256 differs from the index)")
    else:
        integrity = corpus.manifest.get("integrity") or {}
        files = integrity.get("files") or {}
        if RECORDS_FILE not in files:
            mismatches.append(f"{MANIFEST_FILE}: {RECORDS_FILE} is not hashed")
        if integrity.get("manifest_sha256") != manifest_digest(corpus.manifest):
            mismatches.append(f"{MANIFEST_FILE}: manifest_sha256 mismatch")
        if integrity.get("records_count") != len(corpus.records):
            mismatches.append(
                f"{RECORDS_FILE}: {len(corpus.records)} records, the manifest counts {integrity.get('records_count')}"
            )
    for name, entry in sorted(files.items()):
        path = root / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
        elif _sha256(path) != (entry or {}).get("sha256"):
            mismatches.append(f"{name}: hash mismatch")
    return mismatches


def write_subset_index(directory: str | Path, *, full_corpus_path: str, sampling: dict[str, Any]) -> Path:
    """Write a repository subset's ``index.json``: where the full corpus lives, the sampling, the file hashes.

    Call it after the subset's other files are written (``manifest.json`` -- the full corpus's, copied byte
    for byte --, ``records.jsonl.gz`` and ``nondeterminism.json``).

    Args:
        directory: The subset directory.
        full_corpus_path: The full corpus's immutable storage path (e.g. ``gs://YOUR-BUCKET/observations/...``).
        sampling: The recorded sampling (rule, budget, strata covered, rows).

    Returns:
        The written ``index.json``.
    """
    root = Path(directory)
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in (INDEX_FILE, MANIFEST_FILE)
    }
    manifest_bytes = (root / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_bytes)
    index = {
        "schema": SUBSET_INDEX_SCHEMA,
        "full_corpus": {
            "path": full_corpus_path,
            "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_sha256": (manifest.get("integrity") or {}).get("manifest_sha256"),
        },
        "sampling": sampling,
        "files": files,
    }
    path = root / INDEX_FILE
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def normalise_body(body: Any) -> Any:
    """A reply body without its volatile leaves (:data:`NORMALISATION_VERSION` 1), for comparisons.

    Args:
        body: A parsed reply body (any JSON value).

    Returns:
        A copy without the top-level ``id`` and ``created``, ``data[*].created`` and
        ``data[*].permission[*].id`` / ``created``; any other value unchanged. The input is never mutated.
    """
    if not isinstance(body, dict):
        return copy.deepcopy(body)
    out = {key: copy.deepcopy(value) for key, value in body.items() if key not in ("id", "created")}
    data = out.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item.pop("created", None)
                permissions = item.get("permission")
                if isinstance(permissions, list):
                    for permission in permissions:
                        if isinstance(permission, dict):
                            permission.pop("id", None)
                            permission.pop("created", None)
    return out


def credential_findings(text: str) -> list[str]:
    """The credential shapes found in ``text``, by name (never the matched values).

    Args:
        text: Any corpus content (a whole file, one record).

    Returns:
        The sorted names of the shapes that matched; ``[]`` is clean.
    """
    return sorted({name for name, pattern in _CREDENTIALS if pattern.search(text)})
