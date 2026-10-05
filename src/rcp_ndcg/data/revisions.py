"""Resolve a HuggingFace dataset revision to the exact commit it names.

A dataset revision such as ``main`` (or none at all) is a moving pointer. Identities that must say *which* corpus
was judged record the commit it resolved to instead (a judge's ``revision`` is recorded as configured):

* :func:`resolve_revision` asks the Hub (``HfApi.dataset_info``) once per process and falls back to the local cache
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
"""

from __future__ import annotations

import functools
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rcp_ndcg.errors import RcpNdcgWarning
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TRUE = {"1", "true", "yes", "on"}


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
    return revision is not None and bool(_COMMIT.match(revision))


def hub_offline() -> bool:
    """``HF_HUB_OFFLINE``, read now (``huggingface_hub`` freezes it at import)."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE


def hub_cache_dir() -> Path:
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
    if _COMMIT.match(ref):
        return ResolvedRevision(repo_id, ref)
    _remove_corrupt_ref(repo_id, ref)
    commit = None if hub_offline() else _hub_commit(repo_id, ref)
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
    path = hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
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
            if isinstance(sha, str) and _COMMIT.match(sha):
                _record_ref(repo_id, ref, sha)
    except Exception as exc:  # noqa: BLE001 - any Hub failure falls back to the cache, and is logged
        logger.info(f"Hub lookup of dataset {repo_id}@{ref} failed ({type(exc).__name__}: {exc}); trying the cache")
        return None
    return sha if isinstance(sha, str) and _COMMIT.match(sha) else None


def _record_ref(repo_id: str, ref: str, commit: str) -> None:
    """Record ``refs/<ref> -> <commit>`` in the local hub cache, so an offline run resolves *ref* without the Hub.

    Best effort and atomic (temp file + rename): a cache that refuses the write costs one debug line, never a
    failure. huggingface-hub >= 1.x records the ref itself in ``HfApi.resolve_revision``; this fallback keeps the
    ``>=0.34`` floor working.
    """
    path = hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
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
    path = hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
    try:
        sha = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):  # absent, unreadable or corrupt: nothing usable is recorded there
        return None
    return sha if _COMMIT.match(sha) else None


def dataset_uri_revision(uri: str | None, revision: str | None = None) -> dict[str, Any] | None:
    """The resolved commit of a Hub dataset URI (``hf://org/repo[/subset][@rev]``, ``suite:<name>``), else ``None``.

    Part of every identity that reads the dataset. The Hub loaders read the data at the commit resolved here, so
    the data and its identity agree; a moved upstream (a new commit behind ``main``) is read, and recorded, by the
    next process.

    Args:
        uri: The dataset URI a :class:`~rcp_ndcg.data.Dataset` was loaded from.
        revision: The revision it was loaded at (overrides one in the URI).

    Returns:
        ``{"repo", "commit", "verified"}`` for a Hub dataset; ``None`` for any other source.
    """
    if not uri:
        return None
    if uri.startswith("suite:"):
        from rcp_ndcg.data.dataset import SUITES

        suite = SUITES.get(uri.removeprefix("suite:"))
        if suite is None:
            return None
        repo = suite.repo
    elif uri.startswith("hf://"):
        path, _, uri_revision = uri.removeprefix("hf://").partition("@")
        repo = "/".join(path.strip("/").split("/")[:2])
        revision = revision or uri_revision or None
    else:
        return None
    return {"repo": repo, **resolve_revision(repo, revision).identity()}


__all__ = [
    "ResolvedRevision",
    "dataset_uri_revision",
    "hub_cache_dir",
    "hub_offline",
    "is_commit",
    "resolve_revision",
]
