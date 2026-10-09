"""Dataset I/O: one contract, one class per format, one table of formats.

Reading a dataset never means knowing its format::

    from rcp_ndcg.data.io import get_reader

    reader = get_reader("beir", uri="/data/nfcorpus")
    for example in reader.examples():
        ...

:data:`READERS` maps each format name to its :class:`~rcp_ndcg.data.io.base.SourceReader`; the name is also the
URI scheme of :func:`rcp_ndcg.data.load_dataset` (``beir:``, ``jsonl:``, ``images:``, ...), except ``hf`` (the Hub
layout is ``hf://``) and ``pdf`` (a PDF has no queries);
:data:`WRITERS` does the same for :class:`~rcp_ndcg.data.io.base.SinkWriter`. Another format is one class and
one entry in the table.

See :mod:`rcp_ndcg.data.io.base` for the contract, and ``tests/data/test_io_contract.py`` for the conformance
suite every reader is run through.
"""

import inspect
from collections.abc import Iterable
from typing import Any

from rcp_ndcg.data.io.base import (
    DataShape,
    DuplicateCounts,
    DuplicatesPolicy,
    Provenance,
    SinkWriter,
    SourceReader,
    grade,
)
from rcp_ndcg.data.io.beir import BeirReader, BeirWriter
from rcp_ndcg.data.io.frame_dir import FrameDirReader
from rcp_ndcg.data.io.hf import HfReader
from rcp_ndcg.data.io.image_dir import ImageDirReader
from rcp_ndcg.data.io.jsonl import JsonlReader, JsonlWriter
from rcp_ndcg.data.io.pdf import PdfReader
from rcp_ndcg.data.io.video_dir import VideoDirReader
from rcp_ndcg.errors import ConfigError

# ``hf`` and ``pdf`` import their heavy dependencies (``datasets``, ``pypdfium2``) inside the methods that need
# them, so listing them here costs nothing.
READERS: dict[str, type[SourceReader]] = {
    reader.name: reader
    for reader in (BeirReader, JsonlReader, HfReader, ImageDirReader, VideoDirReader, FrameDirReader, PdfReader)
}
"""Format name (= URI scheme) -> reader class."""

WRITERS: dict[str, type[SinkWriter]] = {writer.name: writer for writer in (BeirWriter, JsonlWriter)}
"""Format name -> writer class."""


def reader_options(fmt: str, /) -> frozenset[str] | None:
    """The option names the reader of format ``fmt`` takes besides ``uri``; ``None`` when it takes any.

    A reader that forwards extra keywords (``hf`` passes them to ``datasets.load_dataset``) takes any name.
    """
    if fmt not in READERS:
        raise ConfigError(f"unknown dataset format {fmt!r}. Available: {sorted(READERS)}.")
    parameters = inspect.signature(READERS[fmt]).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return frozenset(parameter.name for parameter in parameters if parameter.name != "uri")


def unknown_reader_options(fmt: str, /, options: Iterable[str]) -> list[str]:
    """The names in ``options`` that the reader of format ``fmt`` does not take, sorted."""
    known = reader_options(fmt)
    return [] if known is None else sorted(set(options) - known)


def get_reader(fmt: str, /, **kwargs: Any) -> SourceReader:
    """Instantiate the reader of format ``fmt`` with the reader options ``kwargs`` (``uri`` and, e.g., ``name``).

    Raises:
        ConfigError: The format is unknown, or the reader does not take one of the options.
    """
    unknown = unknown_reader_options(fmt, (key for key in kwargs if key != "uri"))
    if unknown:
        raise ConfigError(
            f"the {fmt} reader takes no option {', '.join(map(repr, unknown))}",
            hint=f"its options: {', '.join(sorted(reader_options(fmt) or ()))}",
        )
    return READERS[fmt](**kwargs)


def get_writer(fmt: str, /) -> SinkWriter:
    """Instantiate the writer of format ``fmt``."""
    if fmt not in WRITERS:
        raise ConfigError(f"unknown export format {fmt!r}. Available: {sorted(WRITERS)}.")
    return WRITERS[fmt]()


def available_readers() -> list[str]:
    """The reader format names, sorted."""
    return sorted(READERS)


def available_writers() -> list[str]:
    """The writer format names, sorted."""
    return sorted(WRITERS)


__all__ = [
    "READERS",
    "WRITERS",
    "BeirReader",
    "BeirWriter",
    "DataShape",
    "FrameDirReader",
    "HfReader",
    "ImageDirReader",
    "JsonlReader",
    "JsonlWriter",
    "PdfReader",
    "SinkWriter",
    "SourceReader",
    "VideoDirReader",
    "available_readers",
    "available_writers",
    "get_reader",
    "get_writer",
    "grade",
    "reader_options",
    "unknown_reader_options",
]
