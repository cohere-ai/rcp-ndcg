"""The census files' shared record I/O: append, read and the torn-tail repair.

One home for the JSONL discipline every judgement-store file shares (the judgement records' append,
the text census, the media census): rows are appended under the sink's writer lock, a writer killed
mid-append leaves a torn last line that the next append cuts first, and a reader skips a torn last
row with a warning while a malformed row further in is a :class:`~rcp_ndcg.errors.DataError` -- the
file is provenance, never read as numbers, and skipping silently would misattribute it.

The text census (:class:`~rcp_ndcg.data.census.TextTruncationCensus`) and the media census
(:class:`~rcp_ndcg.data.prepare.MediaCensus`) both read and append through here, as does the
judgement store's record append (:func:`drop_torn_last_line` under its writer lock).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rcp_ndcg.errors import DataError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


def drop_torn_last_line(path: Path) -> int:
    """Cut a last line the writer did not finish (a process killed mid-append); return the bytes dropped.

    The one torn-tail repair: the judgement records' append, the text census and the media census share the
    discipline -- a census file whose torn row is left in place would merge the next appended row into the
    fragment, and the merged line would be refused as corrupt on every later read. Call it under
    :func:`census_sink_lock` (or the store's writer lock): a peer's cut must not truncate an in-flight line.
    """
    if not path.exists():
        return 0
    size = path.stat().st_size
    with path.open("r+b") as handle:
        if not size:
            return 0
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0
        end = size
        while end > 0:
            start = max(0, end - (1 << 20))
            handle.seek(start)
            block = handle.read(end - start)
            cut = block.rfind(b"\n")
            if cut >= 0:
                keep = start + cut + 1
                break
            end = start
        else:
            keep = 0
        logger.warning("%s: dropping a torn last line (%d bytes)", path, size - keep)
        handle.truncate(keep)
        return size - keep


@contextmanager
def census_sink_lock(sink: Path) -> Iterator[None]:
    """Serialize the writers of one census file: an advisory flock on its directory.

    The judgement store's records append under the same lock (their store directory *is* the census's
    parent), so a store's record append and a pass's census appends never interleave; a killed process
    releases it by closing.
    """
    sink = Path(sink)
    sink.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(sink.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def append_census_rows(sink: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append ``rows`` to a census file (``preprocessing.jsonl``), one JSON line each.

    Under the sink's writer lock, and cutting a killed writer's torn tail first -- on every append, not only
    the writer's first: a peer killed mid-write after this writer started leaves a tail only its next append
    would merge into.
    """
    sink = Path(sink)
    if not rows:
        return
    with census_sink_lock(sink):
        drop_torn_last_line(sink)
        with sink.open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def read_census_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """The JSON rows of a census file (``preprocessing.jsonl``), in order.

    A torn last line -- the writer died mid-append, exactly what the judgement records tolerate -- is skipped
    with a warning, so a killed pass does not poison every resumed one; a malformed row further in is a
    :class:`~rcp_ndcg.errors.DataError` naming the line (the file is not a census, and skipping silently would
    misattribute provenance). The one reader of the file's rows: :class:`TextTruncationCensus`'s and
    :class:`~rcp_ndcg.data.prepare.MediaCensus`'s resumptions both read through it.

    Yields:
        Each row, in file order.

    Raises:
        DataError: a complete line that is not a JSON object.
    """
    path = Path(path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if number == len(lines) and not line.endswith("\n"):
            # A torn last row (the process died mid-write -- its newline never landed, parseable or not): the
            # row is recorded again. The cutters' definition of unfinished governs, so a next append can
            # never silently delete a row this read counted as on record.
            logger.warning("%s:%d: ignoring a torn last census row (%d bytes)", path, number, len(line))
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise DataError(
                f"{path}:{number}: not a census row: {exc}",
                hint="the shared census record of the judging passes is corrupt; repair the line or remove the "
                "file (the cuts are provenance, never read as numbers)",
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("mechanism"), str):
            raise DataError(
                f"{path}:{number}: not a census row (a JSON object with a 'mechanism', got {row!r:.120})",
                hint="the shared census record of the judging passes is corrupt; repair the line or remove the file",
            )
        yield row


__all__ = [
    "append_census_rows",
    "census_sink_lock",
    "drop_torn_last_line",
    "read_census_rows",
]
