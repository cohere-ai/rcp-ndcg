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

from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points

from rcp_ndcg.data.io.base import SinkWriter, SourceReader
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

READER_GROUP = "rcp_ndcg.readers"
"""The entry-point group of the dataset readers; the name is also a URI scheme of
:func:`rcp_ndcg.data.load_dataset`."""

WRITER_GROUP = "rcp_ndcg.writers"
"""The entry-point group of the dataset writers."""


def _registrations(entries: Iterable[EntryPoint], group: str) -> dict[str, EntryPoint]:
    """The entry points of *group*, by name. A duplicate name is refused at lookup, not hidden."""
    out: dict[str, EntryPoint] = {}
    for entry_point in entries:
        if entry_point.name in out:
            logger.warning(f"entry point group {group!r} declares {entry_point.name!r} twice; the last wins")
        out[entry_point.name] = entry_point
    return out


def reader_class(fmt: str, /) -> type[SourceReader]:
    """The reader class registered for format *fmt*.

    Raises:
        ConfigError: The format is unknown (the installed readers are named).
    """
    found = _registrations(entry_points(group=READER_GROUP), READER_GROUP)
    if fmt not in found:
        raise ConfigError(
            f"unknown dataset format {fmt!r}. Available: {sorted(found)}.",
            hint=f"a reader is a {READER_GROUP!r} entry point; install the package that provides it",
        )
    return _load(found[fmt], SourceReader)


def writer_class(fmt: str, /) -> type[SinkWriter]:
    """The writer class registered for format *fmt*.

    Raises:
        ConfigError: The format is unknown (the installed writers are named).
    """
    found = _registrations(entry_points(group=WRITER_GROUP), WRITER_GROUP)
    if fmt not in found:
        raise ConfigError(
            f"unknown export format {fmt!r}. Available: {sorted(found)}.",
            hint=f"a writer is a {WRITER_GROUP!r} entry point; install the package that provides it",
        )
    return _load(found[fmt], SinkWriter)


def _load[T](entry_point: EntryPoint, base: type[T]) -> type[T]:
    """The class *entry_point* names, checked against *base* (a plugin failing to import fails here, named)."""
    try:
        loaded = entry_point.load()
    except Exception as exc:  # noqa: BLE001 - any plugin failure is the plugin's, and is named
        raise ConfigError(
            f"the {entry_point.name!r} entry point of {entry_point.group!r} could not be loaded: "
            f"{type(exc).__name__}: {exc}",
            hint="fix or uninstall the package that declares it",
        ) from exc
    if not (isinstance(loaded, type) and issubclass(loaded, base)):
        raise ConfigError(
            f"the {entry_point.name!r} entry point of {entry_point.group!r} names {loaded!r}, not a {base.__name__}"
        )
    return loaded


def registered_readers() -> tuple[str, ...]:
    """The registered reader format names, sorted (built-ins and plugins)."""
    return tuple(sorted(_registrations(entry_points(group=READER_GROUP), READER_GROUP)))


def registered_writers() -> tuple[str, ...]:
    """The registered writer format names, sorted (built-ins and plugins)."""
    return tuple(sorted(_registrations(entry_points(group=WRITER_GROUP), WRITER_GROUP)))


__all__ = [
    "READER_GROUP",
    "WRITER_GROUP",
    "reader_class",
    "registered_readers",
    "registered_writers",
    "writer_class",
]
