"""HF weight caches on the node: measure, decide, evict (docs/how-to/release-candidates.md).

The pod has no persistent volume: engine weights land on the container filesystem, and a wave that
serves several models would fill it. The wave runner measures the free disk before each recipe and
evicts a model's cache after its engine stops (unless a later recipe reuses it); wave 0's eviction
check uses the same helpers, so there is one eviction implementation.

Every function is offline-safe: a Hub that does not answer means "size unknown" (the caller decides
whether that is fatal), never an exception through the runner. Sizes are bytes; counts are counts.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Eviction",
    "cache_dir_of",
    "disk_free_bytes",
    "evict",
    "hf_cache_root",
    "model_disk_bytes",
    "model_weight_files",
    "snapshot_bytes",
]

_HEADROOM_BYTES = 1 << 30
"""Free disk the engine needs beside the weights (compile caches, temporary files)."""

_WEIGHT_PATTERNS: tuple[str, ...] = ("*.safetensors", "*.bin", "*.pt")
"""The weight formats vLLM's loader tries in order, and the only formats it downloads at the repo root;
its default load config ignores ``original/**`` (the other copies a multi-format repo ships)."""

_MODEL_DIR = re.compile(r"^models--(?P<org>.+)--(?P<name>.+)$")


def hf_cache_root() -> Path:
    """The HF weights cache the engines download into (``$HF_HUB_CACHE``, else ``$HF_HOME/hub``,
    else ``~/.cache/huggingface/hub``)."""
    import os

    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cache_dir_of(model: str, *, cache_root: Path | None = None) -> Path:
    """The cache directory of ``model`` (``org/name`` -> ``models--org--name``) under the cache root."""
    return (cache_root or hf_cache_root()) / f"models--{model.replace('/', '--')}"


def disk_free_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding ``path`` (the container filesystem on the node); a path that
    does not exist yet (a fresh pod's empty cache) is measured at its nearest existing parent."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def snapshot_bytes(model: str, *, cache_root: Path | None = None) -> int:
    """The bytes the model's cache directory currently holds (0 when it holds nothing)."""
    directory = cache_dir_of(model, cache_root=cache_root)
    if not directory.is_dir():
        return 0
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:  # pragma: no cover - a file vanished between listing and stat
            continue
    return total


def model_weight_files(model: str, revision: str | None = None) -> tuple[str, int] | None:
    """What the engine downloads for ``model``: ``(pattern, bytes)``, or ``None`` when the Hub does not say.

    vLLM loads one weight format -- the first of ``*.safetensors``, ``*.bin``, ``*.pt`` the repo ships at
    its root -- plus the small root-level files beside it (configs, tokenizers, the safetensors index),
    and ignores ``original/**``.  Summing every sibling (the old behaviour) also sized the other formats:
    ``openai/gpt-oss-120b`` holds 60.8 GiB of root safetensors plus 60.8 GiB each under ``original/`` and
    ``metal/`` (182.3 GiB against 60.8 GiB loaded), so a node with 62-183 GiB free false-failed the judge.
    Offline-safe: any Hub error returns ``None`` (the caller records "size unknown" and decides).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        info = HfApi().model_info(model, revision=revision, files_metadata=True)
    except Exception:  # noqa: BLE001 - offline, rate-limited or gone: the size is unknown, not fatal
        return None
    files: dict[str, int] = {}
    for sibling in info.siblings or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        size = getattr(sibling, "size", None)
        if name and size:
            files[name] = int(size)
    if not files:
        return None
    for pattern in _WEIGHT_PATTERNS:
        weights = {name: size for name, size in files.items() if "/" not in name and fnmatch.fnmatch(name, pattern)}
        if weights:
            small = sum(
                size
                for name, size in files.items()
                if "/" not in name and not any(fnmatch.fnmatch(name, other) for other in _WEIGHT_PATTERNS)
            )
            return pattern, sum(weights.values()) + small
    # No recognised weight file at the root: size what the loader would fetch minus the ignored copies.
    total = sum(size for name, size in files.items() if not name.startswith("original/"))
    return ("*", total) if total else None


def model_disk_bytes(model: str, revision: str | None = None) -> int | None:
    """The model's weight bytes from the Hub's file metadata; ``None`` when the Hub does not say.

    The bytes the engine downloads (see :func:`model_weight_files`), never the sum of every sibling.
    Offline-safe: any Hub error returns ``None`` (the caller records "size unknown" and decides), so a
    wave without the Hub never fails on the disk check. The download needs ``huggingface_hub`` (the
    ``[hf]`` extra); without it the size is unknown too.
    """
    found = model_weight_files(model, revision)
    return found[1] if found is not None else None


@dataclass(frozen=True)
class Eviction:
    """The result of one eviction: what was removed and what it freed.

    Attributes:
        model: The model whose cache was removed.
        removed: Whether a cache directory existed and was removed.
        freed_bytes: The bytes the removed directory held (measured before the removal; 0 when it
            removed nothing).
        bytes_before, bytes_after: The free disk before and after, in bytes, on the cache root's
            filesystem (other processes write concurrently, so the difference is indicative, not exact).
        error: The eviction's failure, when it failed (an eviction failure never stops a wave; the
            next recipe's disk check sees the disk as it is).
    """

    model: str
    removed: bool
    freed_bytes: int
    bytes_before: int
    bytes_after: int
    error: str | None = None


def evict(model: str, *, cache_root: Path | None = None) -> Eviction:
    """Remove the model's weights from the HF cache (node-runtime item 8).

    Only the exact ``models--<org>--<name>`` directory is removed, and only when its name matches the
    layout (a path that does not is refused, never deleted): the cache root is not touched. An
    ``HF_HUB_CACHE``/``HF_HOME`` from the environment selects the root, as the engines see it.

    Returns:
        The :class:`Eviction` (with the error, when the removal failed; never raises for a cache that
        is absent, already gone or unwritable).
    """
    root = cache_root or hf_cache_root()
    directory = cache_dir_of(model, cache_root=root)
    if not _MODEL_DIR.match(directory.name):
        # The layout check is the rm guard: only an exact models--<org>--<name> directory is ever removed.
        return Eviction(model, False, 0, disk_free_bytes(root), disk_free_bytes(root), "refused: unexpected cache name")
    if not directory.is_dir():
        return Eviction(model, False, 0, disk_free_bytes(root), disk_free_bytes(root), None)
    freed = snapshot_bytes(model, cache_root=root)
    before = disk_free_bytes(root)
    try:
        _remove_tree(directory)
    except OSError as error:
        return Eviction(model, False, 0, before, disk_free_bytes(root), f"{type(error).__name__}: {error}")
    return Eviction(model, True, freed, before, disk_free_bytes(root))


def _remove_tree(directory: Path) -> None:
    """``shutil.rmtree`` that tolerates read-only snapshot files (HF marks blobs read-only)."""
    import os
    import stat

    def _writable(func: Any, path: str, _exc: Any) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IXUSR | stat.S_IRUSR)
        except OSError:  # pragma: no cover - nothing more can be done here
            return
        try:
            func(path)
        except OSError:  # pragma: no cover - a busy filesystem: one retry after a beat
            time.sleep(0.2)
            func(path)

    shutil.rmtree(directory, onerror=_writable)


def will_fit(free_bytes: int, model_bytes: int | None) -> tuple[bool, str]:
    """Whether a model of ``model_bytes`` fits in ``free_bytes`` (with the compile-cache headroom).

    Returns ``(True, "")`` when it fits or the size is unknown (the caller records "size unknown" and
    decides), and ``(False, reason)`` with a one-line reason when it measurably does not.
    """
    if model_bytes is None:
        return True, ""
    needed = model_bytes + _HEADROOM_BYTES
    if free_bytes < needed:
        return False, (
            f"the model needs ~{model_bytes / (1 << 30):.1f} GiB of weights (+1 GiB headroom) but only "
            f"{free_bytes / (1 << 30):.1f} GiB are free; evict other models or use a smaller one"
        )
    return True, ""
