"""Format lookup: a reader or writer name is an entry point of its group.

The built-in formats (``beir``, ``jsonl``, ``hf``, ``mteb``, ``images``, ``videos``, ``frames``, ``pdf``) are
entry points of this package, exactly like the runners (``rcp_ndcg.runners``). A third-party format is one
class in its own package, published under the matching group::

    [project.entry-points."rcp_ndcg.readers"]
    my-format = "my_package.io:MyReader"

``MyReader`` implements :class:`~rcp_ndcg.data.io.base.SourceReader` (``SinkWriter`` for
``rcp_ndcg.writers``), its first constructor parameter is named ``uri``, and it must pass the shared
conformance suite (``rcp_ndcg.testing.io_conformance``). Its name is then the URI scheme of
:func:`rcp_ndcg.data.load_dataset` (``my-format:``) and the ``--format`` of ``rcp-ndcg data convert``.

An entry point is loaded only when its name is asked for, so a plugin's imports cost nothing for users who do
not select it. A broken plugin fails when selected, naming its import error -- it never silently disappears
from the table.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from rcp_ndcg.data.io.base import SinkWriter, SourceReader
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.entrypoints import load_entry_point, provider_of, registered_names

READER_GROUP = "rcp_ndcg.readers"
"""The entry-point group of the dataset readers; the name is also a URI scheme of
:func:`rcp_ndcg.data.load_dataset`."""

WRITER_GROUP = "rcp_ndcg.writers"
"""The entry-point group of the dataset writers."""


def reader_class(fmt: str, /) -> type[SourceReader]:
    """The reader class registered for format *fmt*.

    Raises:
        ConfigError: The format is unknown (the installed readers are named).
    """
    entries = tuple(entry_points(group=READER_GROUP))
    available = registered_names(entries, READER_GROUP)
    if fmt not in available:
        raise ConfigError(
            f"unknown dataset format {fmt!r}. Available: {list(available)}.",
            hint=f"a reader is a {READER_GROUP!r} entry point; install the package that provides it",
        )
    return load_entry_point(provider_of(entries, READER_GROUP, fmt, kind="dataset format"), SourceReader, kind="reader")


def writer_class(fmt: str, /) -> type[SinkWriter]:
    """The writer class registered for format *fmt*.

    Raises:
        ConfigError: The format is unknown (the installed writers are named).
    """
    entries = tuple(entry_points(group=WRITER_GROUP))
    available = registered_names(entries, WRITER_GROUP)
    if fmt not in available:
        raise ConfigError(
            f"unknown export format {fmt!r}. Available: {list(available)}.",
            hint=f"a writer is a {WRITER_GROUP!r} entry point; install the package that provides it",
        )
    return load_entry_point(provider_of(entries, WRITER_GROUP, fmt, kind="export format"), SinkWriter, kind="writer")


def registered_readers() -> tuple[str, ...]:
    """The registered reader format names, sorted (built-ins and plugins)."""
    return registered_names(entry_points(group=READER_GROUP), READER_GROUP)


def registered_writers() -> tuple[str, ...]:
    """The registered writer format names, sorted (built-ins and plugins)."""
    return registered_names(entry_points(group=WRITER_GROUP), WRITER_GROUP)


__all__ = [
    "READER_GROUP",
    "WRITER_GROUP",
    "reader_class",
    "registered_readers",
    "registered_writers",
    "writer_class",
]
