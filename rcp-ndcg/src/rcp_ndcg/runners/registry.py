"""Runner lookup: a runner name is an entry point of the ``rcp_ndcg.runners`` group.

The public runners (``local``, ``slurm``, ``kubernetes``) are entry points of this package. Any other runner is a
separate package that publishes one::

    [project.entry-points."rcp_ndcg.runners"]
    my-scheduler = "my_package.runner:MyRunner"

``MyRunner(**options)`` receives the runner's options (``runner.options`` of a run config) and implements
:class:`~rcp_ndcg.runners.base.JobRunner` (``submit``, ``status``, ``logs``, ``cancel``). An entry point is loaded
only when its name is asked for, so a plugin's imports cost nothing for users who do not select it.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners.base import JobRunner

ENTRY_POINT_GROUP = "rcp_ndcg.runners"


def get_runner(name: str, **options: Any) -> JobRunner:
    """Instantiate the runner the ``rcp_ndcg.runners`` entry point ``name`` names, with ``options``.

    Raises:
        ConfigError: ``name`` is no installed runner, or ``options`` are not the runner's options.
    """
    found = {entry_point.name: entry_point for entry_point in entry_points(group=ENTRY_POINT_GROUP)}
    if name not in found:
        raise ConfigError(
            f"unknown runner {name!r}; installed: {sorted(found)}",
            hint=f"a runner is a {ENTRY_POINT_GROUP!r} entry point; install the package that provides it",
        )
    runner = found[name].load()
    try:
        return runner(**options)
    except TypeError as exc:
        raise ConfigError(f"invalid options for the {name!r} runner: {exc}") from exc


__all__ = ["ENTRY_POINT_GROUP", "get_runner"]
