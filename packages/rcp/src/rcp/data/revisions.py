"""What names a dataset source in an identity: the commit a Hub revision resolves to, and a local source's content.

A dataset revision such as ``main`` (or none at all) is a moving pointer. Identities that must say *which* corpus
was judged record the commit it resolved to instead (a judge's ``revision`` is recorded as configured):

* :func:`resolve_revision` asks the Hub (``HfApi.resolve_revision`` on huggingface-hub >= 1.x,
  ``HfApi.dataset_info`` on the 0.34 floor) once per process and falls back to the local cache
  (``<hub cache>/datasets--org--name/refs/<revision>``) when offline (``HF_HUB_OFFLINE=1``) or when the Hub cannot
  be reached. A revision that is already a full 40-character commit resolves to itself without any lookup.
* An online resolution records the ref in the cache (``refs/<ref>``), so the offline run without ``--revision``
  resolves the same commit from it; a download pinned to a commit cannot write that ref itself, which is why a
  cache an online run filled otherwise serves nothing offline.
* When neither answers, the commit is ``None`` and the result is not verified. Nothing is invented, and a typed
  warning (``UNPINNED_REVISION``, one per repository and revision argument per process) names the repository and
  the fix;
  the CLI collects it into its ``--json`` envelope's ``warnings`` and prints it on stderr otherwise.

The result is cached per ``(repo_id, revision)`` for the life of the process, so an identity that is computed many
times (a pipeline's resume checks) costs one metadata call per repository, not one per check.

A **local** source (``jsonl:rows.jsonl``, a ``beir:`` directory, a third party reader's ``acme:rows.jsonl``) has
no revision to pin it. It is named by its content instead: :func:`dataset_content_digest` digests the files the
reader reads -- a file's bytes, a directory's sorted listing -- so editing the file in place makes every step
that read it stale and refuses the judgement store that holds the old bytes. :func:`dataset_local_locations`
lists the locations a source's reader reads without reading them. A **remote** non-Hub source
(``jsonl:gs://bucket/rows.jsonl``, ``beir:s3://bucket/corpus``, ``images:https://...``) has no commit either, so
the same function fingerprints the objects its reader reads from the backend's own metadata: an object's size
and change stamp (its etag/generation, else its mtime), or a streamed SHA-256 when the backend reports neither;
a prefix is its sorted listing. A remote location the backend cannot answer for is refused -- an identity keyed
on the URI alone would reuse work over bytes nobody checked. A source read at a commit (``hf://``, ``mteb:``, a
shipped suite) carries no content digest. A **suite** is data: its identity is the definition its file states,
together with the commit its data is read at (:func:`_suite_definition_digest`), so an edited downstream
``suite.yaml`` is another dataset whether or not it pins a commit.
"""

from __future__ import annotations

import functools
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rcp.errors import DataError, RcpNdcgWarning
from rcp.storage.artifacts import artifact_ref
from rcp.storage.uri import is_remote
from rcp.support.identity import hash_payload, hash_strings
from rcp.support.logging import get_logger
from rcp.support.urls import safe_url

logger = get_logger(__name__)


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TRUE = {"1", "true", "yes", "on"}

#: Reader schemes whose location is remote or carries a commit: ``hf://``, ``suite:`` and ``mteb:`` name remote
#: data (a Hub source is pinned by its resolved commit). Every other scheme names a location on this host -- a
#: reader the digest does not know is not a source it may leave out of an identity. A ``suite:`` URI's own file
#: is a location on this host when nothing pins it (:func:`_unpinned_suite_file`), which is handled before this
#: list is read.
_COMMITTED_SCHEMES = ("hf", "suite", "mteb")

#: The reader-option suffix that names a location the reader *writes* (``media_out_uri``: where a Hub reader
#: writes the media it decodes). An output is neither content nor an input: it never enters an identity, and a
#: mirror never lists it. Every other option whose name ends in ``_uri`` names a location the reader reads.
_OUTPUT_OPTION_SUFFIX = "_out_uri"

#: The per-process digest cache for local dataset files: ``(resolved path, size, mtime_ns) -> digest``.
#: The files of a local dataset do not change under a run, and a resume check computes the identity many times;
#: a full hash per check would cost a pass over a multi-GB corpus each time.  A changed size or mtime is a new
#: key, so an edited file is re-hashed (a size+mtime pre-check, as the identity rules allow).
_LOCAL_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _local_source_digest(location: str) -> str | None:
    """A content digest of one local dataset location, or ``None`` when it is not a readable path.

    A file hashes as its bytes, streamed. A directory hashes as its sorted listing -- each file's relative
    name, size and ``mtime_ns`` -- which is the cheap size+mtime pre-check for a corpus whose readers walk a
    tree (``beir:``, ``images:``, ``videos:``, ``frames:``): an added, removed, renamed or re-encoded file
    moves the digest, and a full content hash of every image is never paid.
    """
    path = Path(location)
    try:
        if path.is_file():
            stat = path.stat()
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            cached = _LOCAL_DIGEST_CACHE.get(key)
            if cached is not None:
                return cached
            digest = artifact_ref(path).sha256
            if len(_LOCAL_DIGEST_CACHE) >= 64:
                _LOCAL_DIGEST_CACHE.clear()
            _LOCAL_DIGEST_CACHE[key] = digest
            return digest
        if path.is_dir():
            entries = sorted(entry for entry in path.rglob("*") if entry.is_file())
            return hash_strings(
                [f"{entry.relative_to(path)}\0{entry.stat().st_size}\0{entry.stat().st_mtime_ns}" for entry in entries]
            )
    except OSError:
        return None  # an unreadable source is the reader's error to report, not the identity's
    return None


@dataclass(frozen=True)
class ResolvedRevision:
    """A HuggingFace dataset revision and the commit it resolved to.

    Attributes:
        repo_id: ``org/name`` on the Hub.
        commit: The 40-character commit, or ``None`` when it could not be resolved.
    """

    repo_id: str
    commit: str | None

    @property
    def verified(self) -> bool:
        """Whether the commit is known."""
        return self.commit is not None

    def identity(self) -> dict[str, Any]:
        """The identity part: the commit and whether it is known."""
        return {"commit": self.commit, "verified": self.verified}


def is_commit(revision: str | None) -> bool:
    """Whether *revision* is already a full 40-character lowercase hex commit (the shape the Hub caches by).

    The public form of the pattern ``revisions.py`` resolves with: a caller that must tell a commit from a branch
    or tag (a snapshot listing is per commit, an offline hint is per resolved revision) reads it from here.

    Args:
        revision: The revision as given (a branch, tag, commit, or ``None``).

    Returns:
        ``True`` when *revision* is exactly 40 hex characters; ``False`` otherwise, and for ``None``.
    """
    return revision is not None and bool(_COMMIT.fullmatch(revision))


def _hub_offline() -> bool:
    """``HF_HUB_OFFLINE``, read now (``huggingface_hub`` freezes it at import)."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE


def _hub_cache_dir() -> Path:
    """The local hub cache: ``HF_HUB_CACHE``, else ``HF_HOME/hub``, else the library default."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:  # the ``hf`` extra is optional; this is the library's own default
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        return cache / "huggingface" / "hub"
    return Path(HF_HUB_CACHE)


@functools.cache
def resolve_revision(repo_id: str, revision: str | None = None) -> ResolvedRevision:
    """Resolve ``revision`` of the Hub dataset ``repo_id`` to a commit (see the module docstring).

    Args:
        repo_id: ``org/name`` on the Hub.
        revision: A branch, tag or commit; ``None`` means ``main``.

    Returns:
        The :class:`ResolvedRevision`; its ``commit`` is ``None`` when neither the Hub nor the local cache could
        answer.
    """
    ref = revision or "main"
    if is_commit(ref):
        return ResolvedRevision(repo_id, ref)
    _remove_corrupt_ref(repo_id, ref)
    commit = None if _hub_offline() else _hub_commit(repo_id, ref)
    if commit is None:
        commit = _cached_commit(repo_id, ref)
    if commit is None:
        warnings.warn(
            RcpNdcgWarning(
                "UNPINNED_REVISION",
                f"Dataset {repo_id}@{ref} resolved to no commit, so the identity records it as unverified: pin the "
                "exact revision with --revision <full sha> to make the run reproducible.",
            ),
            stacklevel=2,
        )
    return ResolvedRevision(repo_id, commit)


def _remove_corrupt_ref(repo_id: str, ref: str) -> None:
    """Remove a ``refs/<ref>`` the readers cannot use (bytes that are not valid UTF-8), best effort.

    huggingface_hub reads the ref unguarded, both in ``HfApi.resolve_revision`` and in ``hf_hub_download``, so a
    corrupt file would crash them; removing it costs the cached commit, which the next online resolution rewrites.
    """
    path = _hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
    try:
        if not path.is_file():
            return
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            path.unlink()
        except OSError:
            return
        logger.warning(f"Removed the corrupt hub cache ref {path}; the next online resolution rewrites it.")
    except OSError:
        pass  # unreadable for another reason (permissions): leave it to the caller's own error handling


def _hub_commit(repo_id: str, ref: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        resolve = getattr(api, "resolve_revision", None)  # huggingface-hub >= 1.x
        if resolve is not None:
            # One call resolves the ref and records refs/<ref> in the cache, best effort, so an offline run
            # without --revision later resolves the same commit from the cache.
            sha = getattr(resolve(repo_id, repo_type="dataset", revision=ref), "resolved", None)
        else:
            sha = getattr(api.dataset_info(repo_id, revision=ref), "sha", None)
            if isinstance(sha, str) and is_commit(sha):
                _record_ref(repo_id, ref, sha)
    except Exception as exc:  # noqa: BLE001 - any Hub failure falls back to the cache, and is logged
        logger.info(f"Hub lookup of dataset {repo_id}@{ref} failed ({type(exc).__name__}: {exc}); trying the cache")
        return None
    return sha if isinstance(sha, str) and is_commit(sha) else None


def _record_ref(repo_id: str, ref: str, commit: str) -> None:
    """Record ``refs/<ref> -> <commit>`` in the local hub cache, so an offline run resolves *ref* without the Hub.

    Best effort and atomic (temp file + rename): a cache that refuses the write costs one debug line, never a
    failure. huggingface-hub >= 1.x records the ref itself in ``HfApi.resolve_revision``; this fallback keeps the
    ``>=0.34`` floor working.
    """
    path = _hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
    try:
        if path.is_file() and path.read_text(encoding="utf-8").strip() == commit:
            return
    except (OSError, UnicodeDecodeError):
        pass  # unreadable or corrupt ref: write over it
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid():x}{uuid4().hex[:8]}.tmp")
        tmp.write_text(commit, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug(
            f"Could not record dataset {repo_id}@{ref} -> {commit} in the local hub cache ({exc}); an offline run "
            "without --revision will not resolve it"
        )


def _cached_commit(repo_id: str, ref: str) -> str | None:
    path = _hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
    try:
        sha = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):  # absent, unreadable or corrupt: nothing usable is recorded there
        return None
    return sha if isinstance(sha, str) and is_commit(sha) else None


def dataset_uri_revision(uri: str | None, revision: str | None = None) -> dict[str, Any] | None:
    """The resolved commit of a dataset URI (``hf://org/repo[/subset][@rev]``, ``suite:<name|path>``), else ``None``.

    Part of every identity that reads the dataset. The Hub loaders read the data at the commit resolved here, so
    the data and its identity agree; a moved upstream (a new commit behind ``main``) is read, and recorded, by the
    next process. A ``suite:`` URI takes its repository and its revision from the suite definition it names: the
    shipped suite, or the ``suite.yaml`` at the path (:func:`rcp.data.io.suite.resolve_suite_spec`). A downstream
    suite is named like a shipped one. An ``mteb:`` task records the revision mteb's own task metadata pins (the
    extra is required: the data could not have been loaded without it).

    Args:
        uri: The dataset URI a :class:`~rcp.data.Dataset` was loaded from.
        revision: The revision it was loaded at (overrides one in the URI, and the one a suite file declares).

    Returns:
        ``{"repo", "commit", "verified"}`` for a Hub dataset, a ``suite:`` suite or an ``mteb:`` task (its
        metadata's pinned dataset revision); ``None`` for any other source, and for a ``suite:`` name that is no
        suite (the loader refuses it with its own error).
    """
    if not uri:
        return None
    if uri.startswith("mteb:"):
        from rcp.data.io.mteb_task import MtebTaskReader

        return MtebTaskReader(uri.removeprefix("mteb:")).revision_payload()
    if uri.startswith("suite:"):
        from rcp.data.io.suite import resolve_suite_spec

        spec = resolve_suite_spec(uri.removeprefix("suite:"))
        if spec is None:
            return None
        repo = spec.repo
        revision = revision or spec.revision
    elif uri.startswith("hf://"):
        path, _, uri_revision = uri.removeprefix("hf://").partition("@")
        repo = "/".join(path.strip("/").split("/")[:2])
        revision = revision or uri_revision or None
    else:
        return None
    return {"repo": repo, **resolve_revision(repo, revision).identity()}


def _suite_definition_digest(name_or_path: str) -> str | None:
    """The digest of the definition a downstream suite file states, else ``None``.

    A suite is data: its identity is the definition its file states, together with the commit its data is read
    at. The definition digests as the validated :class:`~rcp.data.io.suite.SuiteSpec`, its canonical JSON -- never
    as the file's bytes, so a re-serialised or re-commented file states the same suite and keeps the identity,
    while an edit to ``subsets``, ``protocol``, ``name``, ``repo``, ``revision`` or ``gains`` is another dataset.

    The four shipped suites are package data, released with the code and frozen by tests; their identity is their
    name, so a recorded run's ``suite:<name>`` URI keeps it and they carry no definition digest. The distinction
    is the resolution order of :func:`~rcp.data.io.suite.resolve_suite_spec`: a *name* in the package's
    ``suites/`` directory is package data, and anything else is a file the caller points at -- a path to the
    package's own file included, which is then a suite the caller owns.

    Args:
        name_or_path: The ``suite:`` URI's remainder (a shipped suite name, or a path).

    Returns:
        The SHA-256 of the definition for a downstream suite; ``None`` for a shipped suite and for a path that
        is no suite at all (the loader refuses that one with its own error).
    """
    from rcp.data.io.suite import resolve_suite_spec, shipped_suite_names

    if name_or_path in shipped_suite_names():
        return None
    spec = resolve_suite_spec(name_or_path)
    return None if spec is None else hash_payload(spec.model_dump(mode="json"))


def _unpinned_suite_file(name_or_path: str) -> Path | None:
    """The suite file a ``suite:`` URI names when nothing pins it, else ``None``.

    The four shipped suites are files of this package, so nothing outside the package names them. A suite file
    outside the package that declares no full commit is a file on this host, and a mirror lists it; one that
    declares a commit names no local location (the commit names the data, and the definition is in the identity
    either way: :func:`_suite_definition_digest`).

    Args:
        name_or_path: The ``suite:`` URI's remainder (a shipped suite name, or a path).

    Returns:
        The ``suite.yaml`` path of an unpinned suite outside the package, else ``None``.
    """
    from rcp.data.io.suite import resolve_suite_spec, shipped_suite_names, suite_file

    if name_or_path in shipped_suite_names():
        return None
    spec = resolve_suite_spec(name_or_path)
    if spec is None or is_commit(spec.revision):
        return None
    return suite_file(name_or_path)


def _reader_inputs(options: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``(option, location)`` of every reader option that names a location the reader *reads*.

    An option whose name ends in ``_uri`` names a location (``qrels_uri``, ``queries_uri``); one that ends in
    :data:`_OUTPUT_OPTION_SUFFIX` names a location the reader writes (``media_out_uri``), which is not content.
    """
    return [
        (key, value)
        for key, value in (options or {}).items()
        if key.endswith("_uri") and not key.endswith(_OUTPUT_OPTION_SUFFIX) and isinstance(value, str)
    ]


def dataset_local_locations(uri: str, options: Mapping[str, Any] | None = None) -> dict[str, str]:
    """``{name: location}`` of the local locations a dataset URI's reader reads, without reading them.

    The dataset location of a reader scheme whose location is not remote (``jsonl:``, ``beir:``, ``images:``,
    ``videos:``, ``frames:``, ``pdf:`` and a third party's own scheme alike) is ``dataset``; every reader option
    whose name ends in ``_uri`` and whose value is a local path is ``dataset.options.<key>``. A committed source
    (``hf://``, ``mteb:``, a shipped suite) and a remote option value are absent: nothing local is named there.
    A ``suite:`` URI is ``dataset`` when its suite file is outside the package and declares no full commit
    (:func:`_unpinned_suite_file`). That location is the file the suite reader reads. An option that names a
    location the reader *writes* (``media_out_uri``) is no input: it is absent too.

    Args:
        uri: The dataset URI a :class:`~rcp.data.Dataset` was loaded from.
        options: The reader options it was loaded with (``qrels_uri``, ``queries_uri``, ...).
    """
    locations: dict[str, str] = {}
    scheme, sep, rest = uri.partition(":")
    if sep and rest:
        if scheme == "suite":
            file = _unpinned_suite_file(rest)
            if file is not None:
                locations["dataset"] = str(file)
        elif scheme not in _COMMITTED_SCHEMES and not is_remote(rest):
            locations["dataset"] = rest
    for key, value in _reader_inputs(options):
        if not is_remote(value):
            locations[f"dataset.options.{key}"] = value
    return locations


def dataset_content_digest(uri: str, options: Mapping[str, Any] | None = None) -> dict[str, str]:
    """``{name: digest}`` of the content a dataset URI's source states (empty for a source read at a commit).

    The one home of a dataset source's content identity, consumed by the run steps' identity
    (:meth:`rcp.runs.config.DatasetSource.identity`) and by the judgement store's dataset identity alike.
    A local location digests as its bytes (a file) or its sorted listing (a directory); a remote non-Hub
    location is fingerprinted from the backend's metadata -- an object's size and change stamp, or a streamed
    SHA-256 when the backend reports neither, and a prefix as its sorted listing -- so a replaced object at the
    same URI moves the digest like an edited file. Those names are :func:`dataset_local_locations`'s, so an
    identity can say which location moved.

    A **suite** source is named by the definition its file states, under ``definition``
    (:func:`_suite_definition_digest`): an edited downstream ``suite.yaml`` is another dataset whether or not it
    pins a commit, and a shipped suite carries none. A source read at a commit (``hf://``, ``mteb:``, a shipped
    suite) carries no digest at all: its bytes are the repository's, and nothing on this host names them. A
    reader option that names an output location (``media_out_uri``) is not content either.

    Args:
        uri: The dataset URI a :class:`~rcp.data.Dataset` was loaded from.
        options: The reader options it was loaded with (``qrels_uri``, ``queries_uri``, ...).

    Returns:
        ``{name: sha256}`` for every location and for a downstream suite's definition; empty when the source is
        read at a commit.

    Raises:
        DataError: a remote location the backend cannot answer for (missing, unreachable, or no backend
            installed): nothing there can name the bytes, and keying on the URI alone would reuse work over
            bytes nobody checked.
    """
    scheme, sep, rest = uri.partition(":")
    if sep and rest:
        if scheme == "suite":
            definition = _suite_definition_digest(rest)
            return {"definition": definition} if definition is not None else {}
        if scheme in _COMMITTED_SCHEMES:
            return {}
    return {
        name: digest
        for name, location in _dataset_locations(uri, options).items()
        if (digest := _source_digest(location)) is not None
    }


def _dataset_locations(uri: str, options: Mapping[str, Any] | None) -> dict[str, str]:
    """``{name: location}`` of every location a dataset URI's reader reads: local paths and remote objects alike."""
    locations = dataset_local_locations(uri, options)
    scheme, sep, rest = uri.partition(":")
    if sep and rest and scheme not in _COMMITTED_SCHEMES and is_remote(rest):
        locations["dataset"] = rest
    for key, value in _reader_inputs(options):
        if is_remote(value):
            locations[f"dataset.options.{key}"] = value
    return locations


def _source_digest(location: str) -> str | None:
    """The digest of one dataset location: its bytes (local) or its backend fingerprint (remote).

    ``None`` for a local path that cannot be read (the reader reports that); a remote location raises instead
    (:func:`_remote_source_digest`).
    """
    if is_remote(location):
        return _remote_source_digest(location)
    return _local_source_digest(location)


def _remote_source_digest(location: str) -> str:
    """The content fingerprint of one remote dataset location, from the backend's own metadata.

    A single object is its size and change stamp (the backend's etag/generation, else its mtime), or a streamed
    SHA-256 when the backend reports neither. A prefix is its sorted listing of relative names, sizes and
    change stamps -- the remote form of the local directory rule. The lookup asks the backend once per location
    per identity computation; a run computes the identity of a source a few times, not once per document.

    Raises:
        DataError: the backend cannot be reached, has no implementation installed, or answers for neither an
            object nor a prefix there.
    """
    from rcp import storage

    info: dict[str, Any] | None
    try:
        info = storage.info(location)
    except Exception:  # noqa: BLE001 - a prefix (or a missing object) is answered by the listing below
        info = None
    if info is not None and info.get("type") != "directory":
        return _remote_object_digest(location, info)
    try:
        entries = [entry for entry in storage.ls(location, recursive=True) if entry.rstrip("/") != location.rstrip("/")]
    except Exception as exc:  # noqa: BLE001 - one typed refusal for every backend failure
        raise _unfingerprinted(location, exc) from exc
    if entries:
        rows = []
        for entry in entries:
            rows.append(f"{storage.relative(entry, location)}\0{_remote_object_digest(entry, storage.info(entry))}")
        return hash_strings(sorted(rows))
    if info is not None:  # a directory the backend reports with no files under it
        return hash_strings([])
    raise _unfingerprinted(location, None)


def _remote_object_digest(location: str, info: dict[str, Any]) -> str:
    """One remote object's fingerprint: its size and change stamp, or a streamed SHA-256 when it has neither."""
    from rcp.data.media import _change_stamp

    stamp = _change_stamp(info)
    if stamp is None:
        return artifact_ref(location).sha256
    return f"{info.get('size')}\0{stamp}"


def _unfingerprinted(location: str, exc: Exception | None) -> DataError:
    """The typed refusal of a remote dataset location whose content cannot be named."""
    detail = "the backend reported neither an object nor a prefix there"
    if exc is not None:
        detail = f"{type(exc).__name__}: {exc}"
    return DataError(
        f"the remote dataset location {safe_url(location)!r} cannot be fingerprinted: {detail}",
        hint="check the URI and the credentials for its backend; an identity cannot name bytes it cannot read, "
        "so a resume or a judgement store would otherwise reuse work over an unchecked object",
    )


__all__ = [
    "ResolvedRevision",
    "dataset_content_digest",
    "dataset_local_locations",
    "dataset_uri_revision",
    "is_commit",
    "resolve_revision",
]
