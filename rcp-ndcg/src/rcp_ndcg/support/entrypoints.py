"""The one entry-point registry: the names in a group, and the refusal of an ambiguous one.

Every plugin seam of the product resolves its names through here -- the dataset readers and writers
(``rcp_ndcg.readers``, ``rcp_ndcg.writers``), the result sinks (``rcp_ndcg.results``) and the job runners
(``rcp_ndcg.runners``) -- so one name has one lookup and one duplicate-name policy.  The caller fetches its
own group (one ``entry_points`` call, its group constant beside the seam it serves) and passes the entries
in; this module owns what happens to them.

The policy: a name **selected** through :func:`provider_of` that more than one installed distribution
provides is refused, naming the providers.  The alternative -- a warning and the last one wins -- let a
plugin publishing a built-in's name (``slurm``, ``beir``) replace it silently, which is how a wrong reader or
runner gets used without anyone seeing a message.  A listing (:func:`registered_names`) still names the
group's names once each: the ambiguity surfaces when the ambiguous name is actually chosen.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import EntryPoint

from rcp_ndcg.errors import ConfigError

__all__ = ["load_entry_point", "provider_of", "registered_names"]


def registered_names(entries: Iterable[EntryPoint], group: str) -> tuple[str, ...]:
    """The names registered in *group*, sorted, each once (a duplicate name is not listed twice)."""
    del group  # named for the caller's error messages; the listing needs no group name
    return tuple(sorted({entry_point.name for entry_point in entries}))


def provider_of(entries: Iterable[EntryPoint], group: str, name: str, *, kind: str) -> EntryPoint:
    """The entry point that provides *name* in *group*.

    Args:
        entries: The group's entry points (the caller's own ``entry_points`` result).
        group: The group's name, for the messages.
        name: The name to resolve.
        kind: What the name selects (``"dataset format"``, ``"result sink"``, ``"runner"``), for the message.

    Returns:
        The entry point. Its ``value`` is the plugin target; loading it is :func:`load_entry_point`'s job (or
        the caller's, when it constructs rather than instantiates a class).

    Raises:
        ConfigError: *name* is registered by more than one installed distribution (or twice by one). Two
            entry points from the same distribution and target are one provider, not a conflict.
    """
    providers = sorted({_provider(entry_point) for entry_point in entries if entry_point.name == name})
    if len(providers) > 1:
        where = ", ".join(f"{distribution} ({value})" for distribution, value in providers)
        raise ConfigError(
            f"the {kind} name {name!r} is provided by more than one installed distribution: {where}",
            hint=f"uninstall all but one of them, or rename one entry point in the {group!r} group",
        )
    if not providers:
        raise ConfigError(
            f"no {kind} named {name!r} is installed in the {group!r} entry-point group",
            hint="install the package that provides it, or drop the name",
        )
    return next(entry_point for entry_point in entries if entry_point.name == name)


def load_entry_point[T](entry_point: EntryPoint, base: type[T], *, kind: str) -> type[T]:
    """The class *entry_point* names, checked against *base*.

    Raises:
        ConfigError: The entry point does not import (the plugin's own failure, named), or what it names is
            not a subclass of *base* (the one contract every plugin of that kind must meet).
    """
    del kind  # the messages name the entry point and its group, which the entry point itself carries
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


def _provider(entry_point: EntryPoint) -> tuple[str, str]:
    """``(distribution, target)`` an entry point comes from; a distribution-less one is named ``<unknown>``."""
    distribution = getattr(getattr(entry_point, "dist", None), "name", None)
    return (distribution or "<unknown>", entry_point.value)
