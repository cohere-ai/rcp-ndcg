"""The checkpoint's own files at the recipe's pinned revision: what the engine reads when the recipe pins nothing.

One reader for the harness and the negative controls: :func:`checkpoint_file` resolves a file of the recipe's
``model`` at its 40-hex ``revision`` through ``huggingface_hub`` (the local Hub cache answers a pinned revision
without a request; the Hub otherwise, unless offline).  An ABSENT file -- the Hub says the repository has no
such file, or the cache recorded that answer (its ``.no_exist`` marker) -- is ``None``, so a caller may try the
next source.  A file the cache does not hold while the Hub cannot be asked is unknown, not absent; that and any
other failure is a :class:`~rcp_ndcg_test.errors.HarnessError` naming the file, never a silent fall-through to
another source.

Public surface: :func:`checkpoint_file`, :func:`checkpoint_pixel_budget`, :data:`PIXEL_BUDGET_FILES`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg_test.errors import HarnessError

__all__ = ["PIXEL_BUDGET_FILES", "checkpoint_file", "checkpoint_pixel_budget"]

PIXEL_BUDGET_FILES = ("preprocessor_config.json", "processor_config.json")
"""Where a checkpoint declares its image processor's budget: the image processor's own config, else the
processor config's ``image_processor`` block."""


def checkpoint_file(recipe: Recipe, name: str) -> Path | None:
    """The local path of the checkpoint file ``name`` at the recipe's pinned revision, or ``None`` when the
    checkpoint has no such file (the Hub says so, or the cache recorded that it did).

    Raises:
        HarnessError: the file is not cached and the Hub cannot be asked (unknown, not absent), or any other
            failure (a broken cache, a refused download), naming the file.
    """
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
        config = json.loads(path.read_text(encoding="utf-8"))
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
