"""Reading text and JSONL files, local or remote.

Storage-agnostic: any function taking a ``str | Path`` accepts a local path or
a remote URI (``gs://``, ``s3://``, ``az://``, ``hf://``, ``http://``).
They resolve through :func:`rcp_ndcg.storage.cache`, which downloads once and
re-uses the copy until the remote object changes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel
from rcp_ndcg_core._records import RankingExample

from rcp_ndcg import storage
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


def _resolve(path: str | Path) -> str | Path:
    """Resolve a possibly-remote path to a local path via the cache."""
    return storage.cache(path) if storage.is_remote(path) else path


def load_text(path: str | Path) -> str:
    """Load prompts written as plain text files."""
    with open(_resolve(path)) as file:
        return file.read().strip()


def iter_jsonl[BaseModelType: BaseModel](
    file_path: str | Path, example_class: type[BaseModelType] = RankingExample
) -> Iterator[BaseModelType]:
    """Stream a JSONL file one parsed model at a time.

    What :class:`~rcp_ndcg.data.io.base.SourceReader` implementations use: a
    corpus is routinely 10^7 rows, and :func:`load_jsonl` holds every line *and*
    every model in memory at once.  Line numbers appear in parse errors here
    because a reader failing on row 4.3 million is otherwise unfindable.
    """
    with open(_resolve(file_path), encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield example_class(**json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON: {exc}. Line: {line[:200]}") from exc


def load_jsonl[BaseModelType: BaseModel](
    file_path: str | Path, example_class: type[BaseModelType] = RankingExample
) -> Sequence[BaseModelType]:
    """Load a JSONL file into a list of Pydantic models, with GCS caching.

    Args:
        file_path: Local or GCS path to the JSONL file.
        example_class: Pydantic model class for parsing each line (default: RankingExample).

    Returns:
        List of parsed model instances.
    """
    t0 = time.monotonic()
    records = list(iter_jsonl(file_path, example_class=example_class))
    elapsed = time.monotonic() - t0
    logger.debug(f"load_jsonl: {len(records)} records from {Path(str(file_path)).name} in {elapsed:.2f}s")
    return records


def load_jsonl_dicts(file_path: str | Path) -> list[dict]:
    """A JSONL file as a list of plain dicts (no model parsing); local paths and remote URIs alike."""
    with open(_resolve(file_path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ["iter_jsonl", "load_jsonl", "load_jsonl_dicts", "load_text"]
