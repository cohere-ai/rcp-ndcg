"""The adapter: the one seam a third party implements to serve a role over its own protocol (C2).

An adapter turns a role's request into :class:`~rcp_ndcg.inference.types.Call` objects and reads the
:class:`~rcp_ndcg.inference.types.Reply` objects back into the role's result; the transport does everything
around that (routing over replicas, retries, parking on outage, credentials, usage). An adapter is registered
under its ``(role, name)`` -- the registry is scoped by role, so ``cohere`` names a different adapter for the
embed and the rerank roles -- and selected from a config with ``api: <name>``. A third party ships its
adapter in the ``rcp_ndcg.adapters`` entry-point group, each entry named ``<role>.<name>``
(e.g. ``embed.bedrock``), and no config ever names a code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.types import Call, Reply, TokenCount

#: The entry-point group a third-party adapter registers in
#: (``[project.entry-points."rcp_ndcg.adapters"]``, e.g. ``embed.bedrock = "pkg.module:BedrockEmbed"``).
ADAPTER_ENTRY_POINTS = "rcp_ndcg.adapters"

#: The role an adapter speaks; the role fixes the request and result types around an adapter.
AdapterRole = Literal["judge", "embed", "rerank", "multi_vector"]

#: Every role an adapter may serve, in the registry's and the entry-point group's namespace.
ROLES: tuple[str, ...] = ("judge", "embed", "rerank", "multi_vector")

Req = TypeVar("Req", contravariant=True)
"""The request type an adapter consumes (a role's request type); contravariant: adapters are called."""

Res = TypeVar("Res", covariant=True)
"""The result type an adapter produces (a role's result type)."""


@runtime_checkable
class Adapter(Protocol[Req, Res]):
    """One wire protocol for one role: requests to calls, replies to results.

    A class implementing the four members and registered under :func:`register_adapter` (or in the
    :data:`ADAPTER_ENTRY_POINTS` group) is selectable with ``api: <name>``. The transport calls, in order:
    :meth:`calls` once per request, then :meth:`interpret` on the replies it got back (one reply per call, in
    order), and :meth:`usage` per reply for the token accounting.

    An adapter that cannot speak HTTP (an SDK with its own signing) raises
    :class:`~rcp_ndcg.errors.BackendUnavailableError` for an outage and
    :class:`~rcp_ndcg.errors.RequestRejectedError` for a refused request; the transport still routes, parks,
    counts and probes around it.
    """

    name: ClassVar[str]
    """The adapter's name within its role, the value a config's ``api`` field holds (``"openai_chat"``).

    Names are scoped by role: two roles may each register an adapter named ``cohere``, and a config selects
    among its own role's names (``get_adapter(name, role=...)``).
    """

    role: ClassVar[AdapterRole]
    """The role the adapter serves; it fixes which request and result types flow through it."""

    def calls(self, request: Req, *, model: str) -> Sequence[Call]:
        """The HTTP calls ``request`` becomes (usually one; a batched request may be several).

        Args:
            request: The role's request.
            model: The endpoint's served model name, sent as the request's ``model``.

        Returns:
            The calls to send, in order; the transport sends them on one replica without interleaving other
            requests between them.
        """
        ...

    def interpret(self, request: Req, replies: Sequence[Reply]) -> Res:
        """The role's result for ``request``, from the replies of :meth:`calls` (one per call, in order).

        Args:
            request: The request the replies answer.
            replies: One reply per call, in order.

        Returns:
            The role's result type.

        Raises:
            CapabilityError: The endpoint cannot take this request at all (a refused answer schema, a media
                count beyond its per-request limit, a batch cap it answers with HTTP 413).
            RequestRejectedError: The endpoint refused this one request; others would go through.
        """
        ...

    def usage(self, reply: Reply) -> TokenCount | None:
        """The tokens one reply reports, or ``None`` when its API reports none (tokens, not characters)."""
        ...


_BUILTINS: dict[tuple[str, str], type[Adapter[Any, Any]]] = {}
"""The adapters registered in this process, keyed ``(role, name)``; the shipped ones register at their
module's import."""

_PLUGINS: dict[tuple[str, str], type[Adapter[Any, Any]]] | None = None
"""The adapters of the entry-point group, keyed ``(role, name)``, loaded once on first use (``None``: not
loaded yet)."""


def _check_role(role: Any) -> None:
    """Refuse a value that is not an adapter role, so a typo can neither empty a lookup nor widen it.

    Raises:
        ConfigError: ``role`` is not one of :data:`ROLES`.
    """
    if role not in ROLES:
        raise ConfigError(
            f"{role!r} is not an adapter role",
            hint=f"an adapter's role is one of {', '.join(ROLES)}",
            details={"role": role, "known": list(ROLES)},
        )


def register_adapter(cls: type[Adapter[Any, Any]]) -> type[Adapter[Any, Any]]:
    """Register an adapter class under its ``(role, name)`` (a class decorator; a duplicate is refused).

    The registry is scoped by role: the same name may be registered once per role (an embed ``cohere`` and a
    rerank ``cohere``), and a role's configs select only among their own role's names.

    Args:
        cls: The adapter class; its ``name`` and ``role`` must be set.

    Returns:
        ``cls`` unchanged, so the decorator composes.

    Raises:
        ConfigError: ``cls`` has no or an empty ``name``, an unknown ``role``, or its ``(role, name)`` is
            already registered.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{cls.__name__} needs a non-empty `name` to be registered as an adapter")
    role = getattr(cls, "role", None)
    if role not in ROLES:
        raise ConfigError(f"{cls.__name__}.role must be one of {', '.join(ROLES)}, got {role!r}")
    key = (role, name)
    if key in _BUILTINS:
        raise ConfigError(
            f"an adapter named {name!r} is already registered for the {role} role ({_BUILTINS[key].__name__})"
        )
    _BUILTINS[key] = cls
    return cls


def _load_plugins() -> dict[tuple[str, str], type[Adapter[Any, Any]]]:
    """Load the ``rcp_ndcg.adapters`` entry points once; a broken one is an error, never a silent skip.

    Each entry is named ``<role>.<name>`` (``embed.bedrock``); an adapter whose class role disagrees with its
    entry name's prefix is refused, so a typo cannot route one role's requests to another role's adapter.
    A plugin is registered under its class's ``(role, name)``; one that would take a shipped adapter's key is
    refused, so a built-in is never silently shadowed.
    """
    global _PLUGINS
    if _PLUGINS is None:
        loaded: dict[tuple[str, str], type[Adapter[Any, Any]]] = {}
        for entry in entry_points(group=ADAPTER_ENTRY_POINTS):
            prefix, separator, _ = entry.name.partition(".")
            if not separator or prefix not in ROLES:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) must be named <role>.<name> "
                    f"with role one of {', '.join(ROLES)}"
                )
            try:
                adapter = entry.load()
            except Exception as exc:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) failed to import: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            role = getattr(adapter, "role", None)
            if role != prefix:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) names the {prefix!r} role but "
                    f"loads {getattr(adapter, '__name__', adapter)!r}, whose role is {role!r}"
                )
            key = (prefix, getattr(adapter, "name", ""))
            if not isinstance(key[1], str) or not key[1]:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) loads "
                    f"{getattr(adapter, '__name__', adapter)!r}, which has no adapter name to register under"
                )
            if key in _BUILTINS:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) registers {key[1]!r} for the "
                    f"{prefix} role, where the shipped adapter {_BUILTINS[key].__name__} is already registered"
                )
            if key in loaded:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} registers {key[1]!r} for the {prefix} role a "
                    f"second time (already loaded: {loaded[key].__name__})"
                )
            loaded[key] = adapter
        _PLUGINS = loaded
    return _PLUGINS


def known_adapters(role: AdapterRole | None = None) -> tuple[str, ...]:
    """The adapter names a config's ``api`` may name: the built-ins, then the entry-point group.

    Args:
        role: Restrict to one role's names (``"embed"``), as a config's ``api`` field is; ``None`` (the
            default) lists every registered name once, whatever its role.

    Raises:
        ConfigError: ``role`` is not an adapter role (``None`` is the "every role" default).
    """
    if role is not None:
        _check_role(role)
    plugins = _load_plugins()
    if role is None:
        return tuple(sorted({name for _, name in (*_BUILTINS, *plugins)}))
    return tuple(sorted({name for registered_role, name in (*_BUILTINS, *plugins) if registered_role == role}))


def get_adapter(name: str, *, role: AdapterRole) -> type[Adapter[Any, Any]]:
    """The adapter class a config's ``api: name`` selects, within one role's registry.

    Args:
        name: The adapter's name within its role (``"openai_chat"``, ``"rerank"``, ``"cohere"``, or a third
            party's).
        role: The calling config's role (``"embed"``): which namespace the name is resolved in. The same name
            may name a different adapter per role.

    Returns:
        The registered adapter class (not an instance: the caller -- a role client, or a third party --
        instantiates it with the role config its request fields depend on).

    Raises:
        ConfigError: ``role`` is not an adapter role, or no adapter of that name is registered for it; the
            hint lists the names of that role (and, when the name is registered in another role, says so).
    """
    _check_role(role)
    key = (role, name)
    adapter = _BUILTINS.get(key) or _load_plugins().get(key)
    if adapter is None:
        registered = (*_BUILTINS, *_load_plugins())
        known = known_adapters(role)
        elsewhere = sorted(
            {registered_role for registered_role, registered_name in registered if registered_name == name}
        )
        if known:
            hint = f"known {role} adapters: {', '.join(known)}"
        else:
            hint = (
                f"no {role} wire adapter is registered yet; the shipped ones arrive with their role's lane, and a "
                f"third party's in the {ADAPTER_ENTRY_POINTS!r} entry-point group"
            )
        if elsewhere:
            hint += f"; {name!r} is registered for the {', '.join(elsewhere)} role(s)"
        raise ConfigError(f"unknown {role} adapter {name!r}", hint=hint, details={"known": list(known), "role": role})
    return adapter


__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "ROLES",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
