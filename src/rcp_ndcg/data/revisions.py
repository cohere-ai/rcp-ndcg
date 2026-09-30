"""Resolve a HuggingFace dataset revision to the exact commit it names.

A dataset revision such as ``main`` (or none at all) is a moving pointer. Identities that must say *which* corpus
was judged record the commit it resolved to instead (a judge's ``revision`` is recorded as configured):

* :func:`resolve_revision` asks the Hub (``HfApi.dataset_info``) once per process and falls back to the local cache
  (``<hub cache>/datasets--org--name/refs/<revision>``) when offline (``HF_HUB_OFFLINE=1``) or when the Hub cannot
  be reached. A revision that is already a full 40-character commit resolves to itself without any lookup.
* When neither answers, the commit is ``None`` and the result is not verified. Nothing is invented, and a warning
  names the repository once per process.

The result is cached per ``(repo_id, revision)`` for the life of the process, so an identity that is computed many
times (a pipeline's resume checks) costs one metadata call per repository, not one per check.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    commit = None if hub_offline() else _hub_commit(repo_id, ref)
    if commit is None:
        commit = _cached_commit(repo_id, ref)
    if commit is None:
        logger.warning(
            f"Could not resolve dataset {repo_id}@{ref} to a commit (Hub unreachable or offline, and not in the "
            f"local cache at {hub_cache_dir()}). The identity records it as unverified; pin a full commit as its "
            "revision to make it exact."
        )
    return ResolvedRevision(repo_id, commit)


def _hub_commit(repo_id: str, ref: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, revision=ref)
    except Exception as exc:  # noqa: BLE001 - any Hub failure falls back to the cache, and is logged
        logger.info(f"Hub lookup of dataset {repo_id}@{ref} failed ({type(exc).__name__}: {exc}); trying the cache")
        return None
    sha = getattr(info, "sha", None)
    return sha if isinstance(sha, str) and _COMMIT.match(sha) else None


def _cached_commit(repo_id: str, ref: str) -> str | None:
    path = hub_cache_dir() / f"datasets--{repo_id.replace('/', '--')}" / "refs" / ref
    try:
        sha = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return sha if _COMMIT.match(sha) else None


def dataset_uri_revision(uri: str | None, revision: str | None = None) -> dict[str, Any] | None:
    """The resolved commit of a Hub dataset URI (``hf://org/repo[/subset][@rev]``, ``suite:<name>``), else ``None``.

    Part of every identity that reads the dataset, so a moved upstream (a new commit behind ``main``) is noticed.

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
    "resolve_revision",
]
