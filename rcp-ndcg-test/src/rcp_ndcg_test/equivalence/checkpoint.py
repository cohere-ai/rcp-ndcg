"""The checkpoint's own files at the recipe's pinned revision: what the engine reads when the recipe pins nothing.

One reader for the harness and the negative controls: :func:`checkpoint_file` resolves a file of the recipe's
``model`` at its 40-hex ``revision`` from a registered **checkpoint store** first (:func:`use_checkpoint_store`:
a directory whose ``index.json`` maps ``<model>@<revision>:<file>`` keys to vendored copies, integrity-checked
against the index's SHA-256 on every read -- the committed corpora vendor the files their replay reads, so the
conformance suite resolves them offline), else through ``huggingface_hub`` (the local Hub cache answers a pinned
revision without a request; the Hub otherwise, unless offline).  An ABSENT file -- the Hub says the repository
has no such file, or the cache recorded that answer (its ``.no_exist`` marker) -- is ``None``, so a caller may
try the next source.  A file the cache does not hold while the Hub cannot be asked is unknown, not absent; that
and any other failure is a :class:`~rcp_ndcg_test.errors.HarnessError` naming the file, never a silent
fall-through to another source.

Public surface: :func:`checkpoint_file`, :func:`checkpoint_pixel_budget`, :func:`checkpoint_text`,
:func:`stored_checkpoint_file`, :func:`use_checkpoint_store`, :data:`PIXEL_BUDGET_FILES`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg_test.errors import HarnessError

__all__ = [
    "PIXEL_BUDGET_FILES",
    "checkpoint_file",
    "checkpoint_pixel_budget",
    "checkpoint_text",
    "stored_checkpoint_file",
    "use_checkpoint_store",
]

PIXEL_BUDGET_FILES = ("preprocessor_config.json", "processor_config.json")
"""Where a checkpoint declares its image processor's budget: the image processor's own config, else the
processor config's ``image_processor`` block."""


_CHECKPOINT_STORES: list[Path] = []
"""Registered checkpoint stores, searched in registration order (see :func:`use_checkpoint_store`)."""


def use_checkpoint_store(directory: str | Path) -> None:
    """Register a **checkpoint store**: a directory whose ``index.json`` maps ``<model>@<revision>:<file>``
    keys to vendored copies of the checkpoint's own files (see the module docstring). Idempotent per
    directory; resolution uses the first match.

    Args:
        directory: The store directory. ``<directory>/index.json`` is a mapping
            ``{"<model>@<revision>:<file>": {"file": "<name>", "sha256": "<hex>"}}`` where ``file`` is
            relative to the store and ``sha256`` hashes the file's content (decompressed for ``.gz``).
    """
    path = Path(directory).resolve()
    if path not in _CHECKPOINT_STORES:
        _CHECKPOINT_STORES.append(path)


def _entry_content(path: Path) -> bytes:
    """A vendored copy's content: a ``.gz`` entry decompresses (the index hashes the content, not the
    compressed envelope), a plain file reads as-is."""
    data = path.read_bytes()
    return gzip.decompress(data) if path.name.endswith(".gz") else data


def stored_checkpoint_file(model: str, revision: str, name: str) -> Path | None:
    """The vendored copy of ``<model>@<revision>:<name>`` from the registered checkpoint stores, or ``None``
    when no store's index names it.

    The first store whose ``index.json`` carries the key answers: its entry is
    ``{"file": "<relative path>", "sha256": "<hex>"}``, the SHA-256 of the file's content (decompressed for
    a ``.gz`` entry), verified on every read.  The returned path may be ``.gz``; read its text through
    :func:`checkpoint_text`, never a bare ``read_text``.

    Raises:
        HarnessError: an index entry lacks ``file`` or ``sha256``, or the vendored copy is missing or does
            not hash to the index's value (refused by name, never served and never fallen through).
    """
    key = f"{model}@{revision}:{name}"
    for store in _CHECKPOINT_STORES:
        index_path = store / "index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = index.get(key) if isinstance(index, dict) else None
        if not isinstance(entry, dict):
            continue
        file, expected = entry.get("file"), entry.get("sha256")
        if not isinstance(file, str) or not isinstance(expected, str):
            raise HarnessError(f"checkpoint store {index_path}: entry for {key!r} needs 'file' and 'sha256'")
        vendored = store / file
        try:
            actual = hashlib.sha256(_entry_content(vendored)).hexdigest()
        except OSError as error:
            raise HarnessError(
                f"checkpoint store {index_path}: the vendored copy {file} for {key!r} cannot be read: {error}"
            ) from error
        if actual != expected:
            raise HarnessError(
                f"checkpoint store {index_path}: {file} hashes to {actual}, the index declares {expected} "
                f"for {key!r}; the vendored copy is corrupt or stale"
            )
        return vendored
    return None


def checkpoint_text(path: Path) -> str:
    """The utf-8 text of one resolved checkpoint file: a ``.gz`` entry (a vendored store copy) decompresses,
    a plain file reads as-is.  One home for the read, so a store entry and a Hub copy read identically.

    Raises:
        HarnessError: the file cannot be read (naming it).
    """
    try:
        return _entry_content(path).decode("utf-8")
    except OSError as error:
        raise HarnessError(f"the checkpoint file {path} cannot be read: {error}") from error


def checkpoint_file(recipe: Recipe, name: str) -> Path | None:
    """The local path of the checkpoint file ``name`` at the recipe's pinned revision, or ``None`` when the
    checkpoint has no such file (the Hub says so, or the cache recorded that it did).

    A registered checkpoint store answers first (:func:`use_checkpoint_store`: the committed corpora vendor
    the files their replay reads, each copy verified against the store index's SHA-256), so a vendored file
    resolves with no Hub cache and no Hub; the Hub otherwise (the local cache answers a pinned revision
    without a request; the Hub itself, unless offline).  The returned path may be a ``.gz`` store entry:
    read its text through :func:`checkpoint_text`.

    Raises:
        HarnessError: a store entry is corrupt (named), the file is not cached and the Hub cannot be asked
            (unknown, not absent), or any other failure (a broken cache, a refused download), naming the file.
    """
    stored = stored_checkpoint_file(recipe.model, recipe.revision, name)
    if stored is not None:
        return stored
    try:
        from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError as error:  # pragma: no cover - huggingface_hub ships with the harness's [test] extra
        raise HarnessError("reading the checkpoint's files needs huggingface_hub") from error
    try:
        return Path(hf_hub_download(recipe.model, name, revision=recipe.revision))
    except LocalEntryNotFoundError as error:  # before EntryNotFoundError, its base class
        if try_to_load_from_cache(recipe.model, name, revision=recipe.revision) is _CACHED_NO_EXIST:
            return None
        raise HarnessError(
            f"recipe {recipe.id}: {name} of {recipe.model}@{recipe.revision} is not in the Hub cache and the Hub "
            "cannot be asked, so whether the checkpoint ships it is unknown: populate the cache, or reach the Hub"
        ) from error
    except EntryNotFoundError:
        return None
    except Exception as error:  # noqa: BLE001 - every other failure is named, never a fall-through
        raise HarnessError(
            f"recipe {recipe.id}: {name} of {recipe.model}@{recipe.revision} could not be read: "
            f"{type(error).__name__}: {error}"
        ) from error


def _budget_of(config: dict[str, Any]) -> tuple[int, int] | None:
    """An image processor config's pixel budget: ``min_pixels``/``max_pixels`` when both are set (transformers'
    Qwen2-VL image processor makes them the resize size, ``_standardize_kwargs``), else ``size``'s
    ``shortest_edge``/``longest_edge``."""
    if isinstance(config.get("min_pixels"), int) and isinstance(config.get("max_pixels"), int):
        return int(config["min_pixels"]), int(config["max_pixels"])
    raw = config.get("size")
    size: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if isinstance(size.get("shortest_edge"), int) and isinstance(size.get("longest_edge"), int):
        return int(size["shortest_edge"]), int(size["longest_edge"])
    return None


def checkpoint_pixel_budget(recipe: Recipe) -> tuple[str, tuple[int, int]]:
    """The pixel budget an engine started WITHOUT a pin resizes images to: the checkpoint's own image processor
    config at the pinned revision (:data:`PIXEL_BUDGET_FILES`, in order).

    Output: ``(source, (min_pixels, max_pixels))``, the source naming ``<model>@<revision>:<file>``.

    Raises:
        HarnessError: no file declares a budget, or a file cannot be read (never a guessed default).
    """
    tried: list[str] = []
    for name in PIXEL_BUDGET_FILES:
        path = checkpoint_file(recipe, name)
        if path is None:
            tried.append(f"{name}: absent")
            continue
        config = json.loads(checkpoint_text(path))
        block = config.get("image_processor") if name == "processor_config.json" else config
        budget = _budget_of(block) if isinstance(block, dict) else None
        if budget is None:
            tried.append(f"{name}: no pixel budget")
            continue
        return f"{recipe.model}@{recipe.revision}:{name}", budget
    raise HarnessError(
        f"recipe {recipe.id}: no pixel budget resolves for {recipe.model} at {recipe.revision} ("
        + "; ".join(tried)
        + "): populate the Hub cache, or reach the Hub"
    )
