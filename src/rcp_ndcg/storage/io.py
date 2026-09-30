"""Reading text and JSONL files, local or remote.

Storage-agnostic: any function taking a ``str | Path`` accepts a local path or
a remote URI (``gs://``, ``s3://``, ``az://``, ``hf://``, ``http://``).
They resolve through :func:`rcp_ndcg.storage.cache`, which downloads once and
re-uses the copy until the remote object changes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from rcp_ndcg_core._records import RankingExample

from rcp_ndcg import storage
from rcp_ndcg.errors import DataError


def _resolve(path: str | Path) -> str | Path:
    """Resolve a possibly-remote path to a local path via the cache."""
    return storage.cache(path) if storage.is_remote(path) else path


def load_text(path: str | Path) -> str:
    """Load prompts written as plain text files."""
    with open(_resolve(path)) as file:
        return file.read().strip()


def iter_json_lines(file_path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream the JSON objects of a JSONL file, one per non-blank line.

    The one reader of JSONL rows: every record reader goes through it, so every malformed line is refused the same
    way, with its file and line number (a corpus failing on row 4.3 million is otherwise unfindable).

    Raises:
        DataError: A line is not valid JSON, or not a JSON object.
    """
    for _, row in _numbered_rows(file_path):
        yield row


def iter_jsonl[BaseModelType: BaseModel](
    file_path: str | Path, example_class: type[BaseModelType] = RankingExample
) -> Iterator[BaseModelType]:
    """Stream a JSONL file one parsed model at a time (the rows of :func:`iter_json_lines` as ``example_class``).

    Raises:
        DataError: A line is not valid JSON, or does not validate as ``example_class``.
    """
    for line_number, row in _numbered_rows(file_path):
        try:
            yield example_class(**row)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'row'}: {error['msg']}" for error in exc.errors()
            )
            raise DataError(f"{file_path}:{line_number}: not a {example_class.__name__}: {problems}") from exc


def _numbered_rows(file_path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(_resolve(file_path), encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{file_path}:{line_number}: invalid JSON: {exc}. Line: {line[:200]}") from exc
            if not isinstance(row, dict):
                raise DataError(f"{file_path}:{line_number}: a JSONL row is a JSON object, got {line[:200]}")
            yield line_number, row


__all__ = ["iter_json_lines", "iter_jsonl", "load_text"]
