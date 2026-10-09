"""Dataset I/O: one contract, one class per format, one table of formats.

Reading a dataset never means knowing its format::

    from rcp_ndcg.data.io import get_reader

    reader = get_reader("beir", uri="/data/nfcorpus")
    for example in reader.examples():
        ...

The formats are entry points of the ``rcp_ndcg.readers`` and ``rcp_ndcg.writers`` groups (see
:mod:`rcp_ndcg.data.io.registry`): the built-ins are registered by this package, and another format is one
class and one entry point in its own package. Every name is also the URI scheme of
:func:`rcp_ndcg.data.load_dataset` (``beir:``, ``jsonl:``, ``mteb:``, ``images:``, ...), except ``hf`` (the
Hub layout is ``hf://``) and ``pdf`` (a PDF has no queries).

See :mod:`rcp_ndcg.data.io.base` for the contract, and :func:`rcp_ndcg.testing.io_conformance` for the
conformance suite every reader is run through.
"""

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from rcp_ndcg.data.io.base import (
    DataShape,
    DuplicateCounts,
    DuplicateFold,
    DuplicatesPolicy,
    Provenance,
    SinkWriter,
    SourceReader,
    grade,
)
from rcp_ndcg.data.io.beir import BeirReader, BeirWriter
from rcp_ndcg.data.io.frame_dir import FrameDirReader
from rcp_ndcg.data.io.hub import HubReader
from rcp_ndcg.data.io.image_dir import ImageDirReader
from rcp_ndcg.data.io.jsonl import JsonlReader, JsonlWriter
from rcp_ndcg.data.io.mteb_task import MtebTaskReader
from rcp_ndcg.data.io.pdf import PdfReader
from rcp_ndcg.data.io.registry import (
    READER_GROUP,
    WRITER_GROUP,
    reader_class,
    registered_readers,
    registered_writers,
    writer_class,
)
from rcp_ndcg.data.io.video_dir import VideoDirReader
from rcp_ndcg.errors import ConfigError

__all__ = [
    "READER_GROUP",
    "READERS",
    "WRITER_GROUP",
    "WRITERS",
    "BeirReader",
    "BeirWriter",
    "DataShape",
    "DuplicateCounts",
    "DuplicateFold",
    "DuplicatesPolicy",
    "FrameDirReader",
    "HubReader",
    "ImageDirReader",
    "JsonlReader",
    "JsonlWriter",
    "MtebTaskReader",
    "PdfReader",
    "Provenance",
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


class _ReaderTable(Mapping[str, type[SourceReader]]):
    """The reader table: format name (= URI scheme) -> reader class, resolved on demand from the
    ``rcp_ndcg.readers`` entry points, so a plugin's module is imported only when its name is asked for."""

    def __getitem__(self, fmt: str) -> type[SourceReader]:
        return reader_class(fmt)

    def __contains__(self, fmt: object) -> bool:
        return isinstance(fmt, str) and fmt in registered_readers()

    def __iter__(self):
        return iter(registered_readers())

    def __len__(self) -> int:
        return len(registered_readers())


READERS: Mapping[str, type[SourceReader]] = _ReaderTable()
"""Format name (= URI scheme) -> reader class; a view over the ``rcp_ndcg.readers`` entry points."""


class _WriterTable(Mapping[str, type[SinkWriter]]):
    """The writer table: format name -> writer class, a view over the ``rcp_ndcg.writers`` entry points."""

    def __getitem__(self, fmt: str) -> type[SinkWriter]:
        return writer_class(fmt)

    def __contains__(self, fmt: object) -> bool:
        return isinstance(fmt, str) and fmt in registered_writers()

    def __iter__(self):
        return iter(registered_writers())

    def __len__(self) -> int:
        return len(registered_writers())


WRITERS: Mapping[str, type[SinkWriter]] = _WriterTable()
"""Format name -> writer class; a view over the ``rcp_ndcg.writers`` entry points."""


def available_readers() -> list[str]:
    """The reader format names, sorted."""
    return list(registered_readers())


def available_writers() -> list[str]:
    """The writer format names, sorted."""
    return list(registered_writers())


def reader_options(fmt: str, /) -> frozenset[str] | None:
    """The option names the reader of format ``fmt`` takes besides ``uri``; ``None`` when it takes any.

    A reader that forwards extra keywords takes any name.
    """
    parameters = inspect.signature(reader_class(fmt)).parameters.values()
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
    return reader_class(fmt)(**kwargs)


def get_writer(fmt: str, /) -> SinkWriter:
    """Instantiate the writer of format ``fmt``."""
    return writer_class(fmt)()
