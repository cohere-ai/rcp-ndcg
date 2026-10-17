"""The declared sinks and a helper that makes them visible to the registry.

The lane's environment was built before the ``rcp_ndcg.results`` group existed; the gate rebuilds it from
``pyproject.toml`` (its env hash covers every manifest), and the packaging snapshot pins the declaration. The
tests here exercise the registry's real ``importlib.metadata`` path with the declared entries injected, the way
the adapter tests do (``tests/inference/test_types.py``).
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

#: The declared entry points of ``rcp_ndcg/pyproject.toml``, in the one place a test reads them from.
DECLARED_SINKS = {
    "jsonl": "rcp_ndcg.results:JsonlResultSink",
    "null": "rcp_ndcg.results:NullResultSink",
    "parquet": "rcp_ndcg.results:ParquetResultSink",
}


def inject_sinks(monkeypatch: pytest.MonkeyPatch, *extra: EntryPoint) -> list[EntryPoint]:
    """Point ``rcp_ndcg.results.entry_points`` at the declared sinks plus *extra* (a plugin under test)."""
    import rcp_ndcg.results as module

    entries = [EntryPoint(name=name, value=value, group=module.RESULTS_GROUP) for name, value in DECLARED_SINKS.items()]
    entries.extend(extra)
    monkeypatch.setattr(
        module,
        "entry_points",
        lambda group: entries if group == module.RESULTS_GROUP else [],
    )
    return entries
