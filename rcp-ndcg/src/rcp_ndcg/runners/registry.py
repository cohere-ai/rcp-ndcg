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
from rcp_ndcg.support.entrypoints import provider_of, registered_names

ENTRY_POINT_GROUP = "rcp_ndcg.runners"


def get_runner(name: str, **options: Any) -> JobRunner:
    """Instantiate the runner the ``rcp_ndcg.runners`` entry point ``name`` names, with ``options``.

    A name provided by more than one installed distribution (or twice by one) is refused: the lookup once kept
    the last one silently, so a plugin publishing a public name (``slurm``) could replace the built-in and
    receive the built-in's typed options.

    Raises:
        ConfigError: ``name`` is no installed runner, its name is ambiguous, or ``options`` are not the
            runner's options.
    """
    entries = tuple(entry_points(group=ENTRY_POINT_GROUP))
    available = registered_names(entries, ENTRY_POINT_GROUP)
    if name not in available:
        raise ConfigError(
            f"unknown runner {name!r}; installed: {list(available)}",
            hint=f"a runner is a {ENTRY_POINT_GROUP!r} entry point; install the package that provides it",
        )
    runner = provider_of(entries, ENTRY_POINT_GROUP, name, kind="runner").load()
    try:
        return runner(**options)
    except (TypeError, ValueError) as exc:
        # A plugin whose constructor rejects the options raises what it raises; built-ins and plugins then
        # fail with the same typed error for the same mistake (JobOptions.parse converts for the built-ins).
        raise ConfigError(f"invalid options for the {name!r} runner: {exc}") from exc


__all__ = ["ENTRY_POINT_GROUP", "get_runner"]
