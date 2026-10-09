"""The io conformance suite: the reader contract as one check a reader's own tests call.

Every reader -- built-in or plugin, registered through the ``rcp_ndcg.readers`` entry-point group -- is held
to the same invariants: a format that loses text, drops media, reorders records between passes, invents
metadata it cannot support, or derives one shape from the other wrongly fails here rather than three phases
downstream.  A plugin's test suite runs::

    from rcp_ndcg.testing import io_conformance

    def test_my_reader_passes_the_conformance(tmp_path):
        io_conformance(MyReader(uri=str(build_fixture(tmp_path))))

and every built-in reader runs through the same call in ``tests/data/test_io_contract.py`` -- one definition
of what a reader must do, the one the base class's docstring promises.
"""

from __future__ import annotations

import inspect

from rcp_ndcg.data.io import available_readers
from rcp_ndcg.data.io.base import Provenance, SourceReader

__all__ = ["io_conformance"]


def io_conformance(reader: SourceReader | type[SourceReader], *, registered: bool = True) -> None:
    """The reader contract, as one check; raises :class:`AssertionError` naming every failed invariant.

    Checks, collected as a list rather than a first failure:

    * ``name`` is a non-empty string and -- with ``registered`` -- a name of the reader table (the URI scheme
      of :func:`rcp_ndcg.data.load_dataset` and the ``--format`` of ``rcp-ndcg data convert``);
    * ``shapes`` declares at least one;
    * the constructor's first parameter is named ``uri`` (what :func:`get_reader` and ``data convert`` pass by
      keyword; a reader that names it otherwise is unreachable through the registry);
    * :meth:`~rcp_ndcg.data.io.base.SourceReader.examples` and
      :meth:`~rcp_ndcg.data.io.base.SourceReader.documents` are re-iterable (a reader is read more than once; a
      generator consumed once is a bug) and the documents are non-empty and uniquely identified;
    * examples exist exactly when something is judged: the ranking shape *is* the judged candidate lists, so a
      corpus with no qrels yields none rather than fabricating a pool;
    * every query and document id is unique, every document carries text, a title or media (a title-only
      document is content a model reads: ``text`` is the body and ``title`` a field of its own), and the qrels
      reference known ids;
    * every example aligns its document fields and ``doc_contents`` never raises (what every encoder and judge
      calls);
    * media references resolve to decodable assets (when Pillow is installed);
    * the widened contract: :attr:`~rcp_ndcg.data.io.base.SourceReader.provenance` is a
      :class:`~rcp_ndcg.data.io.base.Provenance`; :meth:`~rcp_ndcg.data.io.base.SourceReader.candidates` is
      ``None`` or ``{query_id: [doc_id, ...]}``; :meth:`~rcp_ndcg.data.io.base.SourceReader.excluded` is
      ``{query_id: [doc_id, ...]}``; :meth:`~rcp_ndcg.data.io.base.SourceReader.gains` and ``thetas`` are
      ``None`` or ``{query_id: {doc_id: gain}}``; a pool or exclusion names queries the reader's own queries
      have, and a pool never lists a document twice.

    Args:
        reader: The reader instance (or its class, which is instantiated with no arguments -- only possible
            for a reader whose constructor takes none).
        registered: Whether ``name`` must be in the reader table; ``False`` for a reader under development,
            whose package is not installed yet.

    Raises:
        AssertionError: naming every failed check, so a third party's test suite fails with the list -- never
            somewhere inside the pipeline.
    """
    failures: list[str] = []
    cls = reader if isinstance(reader, type) else type(reader)
    instance = reader if not isinstance(reader, type) else _instantiate(cls, failures)
    if instance is None:
        raise AssertionError("the io conformance failed:\n  - " + "\n  - ".join(failures))

    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        failures.append(f"{cls.__name__}.name is not a non-empty string: {name!r}")
    elif registered and name not in available_readers():
        failures.append(
            f"{cls.__name__}.name {name!r} is not registered (a reader is a 'rcp_ndcg.readers' entry point)"
        )
    if not getattr(cls, "shapes", None):
        failures.append(f"{cls.__name__}.shapes is empty; a reader declares which shapes it can serve")
    parameters = list(inspect.signature(cls.__init__).parameters)
    if len(parameters) < 2 or parameters[1] != "uri":
        first = parameters[1] if len(parameters) > 1 else "<none>"
        failures.append(f"{cls.__name__} takes {first!r} rather than 'uri' as its first parameter")

    failures.extend(_shape_invariants(instance))
    failures.extend(_table_invariants(instance))
    if failures:
        raise AssertionError("the io conformance failed:\n  - " + "\n  - ".join(failures))


def _instantiate(cls: type[SourceReader], failures: list[str]) -> SourceReader | None:
    """The class as an instance, constructed with no arguments; a construction failure is a contract failure."""
    try:
        return cls()  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001 - any construction failure is a contract failure
        failures.append(f"{cls.__name__}() did not construct: {type(exc).__name__}: {exc}")
        return None


def _shape_invariants(reader: SourceReader) -> list[str]:
    """The shape invariants: re-iterability, the judged-pool definition of the ranking shape, alignment."""
    failures: list[str] = []
    try:
        examples = list(reader.examples())
        again = [example.id for example in reader.examples()]
    except Exception as exc:  # noqa: BLE001 - any read failure is a contract failure
        failures.append(f"examples() raised {type(exc).__name__}: {exc}")
        return failures
    if [example.id for example in examples] != again:
        failures.append("examples() is not re-iterable: a second read gave different records")
    try:
        documents = list(reader.documents())
        again = [doc.id for doc in reader.documents()]
    except Exception as exc:  # noqa: BLE001
        failures.append(f"documents() raised {type(exc).__name__}: {exc}")
        return failures
    if [doc.id for doc in documents] != again:
        failures.append("documents() is not re-iterable: a second read gave different records")
    if not documents:
        failures.append("documents() yielded nothing: a corpus-only source still has a corpus")
    ids = [doc.id for doc in documents]
    if len(ids) != len(set(ids)):
        failures.append("documents() repeated a document id (folding duplicates is the reader's own job)")
    qrels = reader.qrels()
    if bool(examples) != bool(qrels):
        failures.append(
            "examples exist exactly when something is judged: the ranking shape *is* the judged candidate lists"
        )
    for example in examples:
        if not example.doc_ids:
            failures.append(f"example {example.id!r} has no candidates")
        if example.docs is not None and len(example.docs) != len(example.doc_ids):
            failures.append(f"example {example.id!r}: docs does not align with doc_ids")
        if example.contents is not None and len(example.contents) != len(example.doc_ids):
            failures.append(f"example {example.id!r}: contents does not align with doc_ids")
        try:
            contents = example.doc_contents
        except Exception as exc:  # noqa: BLE001 - what every encoder and judge calls must work
            failures.append(f"example {example.id!r}: doc_contents raised {type(exc).__name__}: {exc}")
            continue
        if len(contents) != len(example.doc_ids):
            failures.append(f"example {example.id!r}: doc_contents does not align with doc_ids")
    for document in documents:
        if not (document.text or document.title or document.has_media):
            failures.append(
                f"document {document.id!r} has no text, no title and no media (a title-only document is "
                "content a model reads)"
            )
    return failures


def _table_invariants(reader: SourceReader) -> list[str]:
    """The provenance, pool and exclusion tables: shapes, and references the reader's own tables have."""
    failures: list[str] = []
    provenance = reader.provenance
    if not isinstance(provenance, Provenance):
        failures.append(f"provenance is a {type(provenance).__name__}, not a Provenance")
    try:
        queries = {query.id for query in reader.queries()}
    except Exception as exc:  # noqa: BLE001
        failures.append(f"queries() raised {type(exc).__name__}: {exc}")
        return failures
    if len(queries) != len(list(reader.queries())):
        failures.append("queries() repeated a query id")
    documents = {doc.id for doc in reader.documents()}

    candidates = reader.candidates()
    if candidates is not None:
        if not isinstance(candidates, dict) or not all(
            isinstance(docs, list) and len(set(docs)) == len(docs) for docs in candidates.values()
        ):
            failures.append("candidates() is None or {query_id: [doc_id, ...]} with no document listed twice")
        else:
            unknown = sorted(set(candidates) - queries)
            if unknown:
                failures.append(f"candidates() names queries the queries() lacks: {unknown[:5]}")
            missing = sorted({doc for docs in candidates.values() for doc in docs} - documents)
            if missing:
                failures.append(f"candidates() names documents the corpus lacks: {missing[:5]}")
    excluded = reader.excluded()
    if not isinstance(excluded, dict):
        failures.append(f"excluded() is a {type(excluded).__name__}, not {{query_id: [doc_id, ...]}}")
    else:
        unknown = sorted(set(excluded) - queries)
        if unknown:
            failures.append(f"excluded() names queries the queries() lacks: {unknown[:5]}")
        missing = sorted({doc for docs in excluded.values() for doc in docs} - documents)
        if missing:
            failures.append(f"excluded() names documents the corpus lacks: {missing[:5]}")
    for name in ("gains", "thetas"):
        table = getattr(reader, name)()
        if table is None:
            continue
        if not isinstance(table, dict) or not all(
            isinstance(judged, dict) and all(isinstance(g, int | float) for g in judged.values())
            for judged in table.values()
        ):
            failures.append(f"{name}() is None or {{query_id: {{doc_id: number}}}}")
            continue
        unknown = sorted(set(table) - queries)
        if unknown:
            failures.append(f"{name}() names queries the queries() lacks: {unknown[:5]}")
        missing = sorted({doc for judged in table.values() for doc in judged} - documents)
        if missing:
            failures.append(f"{name}() names documents the corpus lacks: {missing[:5]}")
    failures.extend(_media_invariants(reader))
    return failures


def _media_invariants(reader: SourceReader) -> list[str]:
    """Every media reference of the corpus resolves to a decodable asset (skipped without Pillow)."""
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return []
    from rcp_ndcg.data.media import MediaResolver, probe_video_header

    failures: list[str] = []
    resolver = MediaResolver()
    for document in reader.documents():
        for ref in document.media:
            try:
                if ref.mime and ref.mime.startswith("video/"):
                    assert probe_video_header(resolver.bytes_of(ref)) is not None, "not a readable container"
                else:
                    assert resolver.image(ref).width > 0, "not a decodable image"
            except Exception as exc:  # noqa: BLE001 - a reference that cannot resolve is a contract failure
                failures.append(f"the media of document {document.id!r} does not resolve: {exc}")
    return failures
