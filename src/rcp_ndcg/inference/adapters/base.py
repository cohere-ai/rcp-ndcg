"""The adapter: the one seam a third party implements to serve a role over its own protocol (C2).

An adapter turns a role's request into :class:`~rcp_ndcg.inference.types.Call` objects and reads the
:class:`~rcp_ndcg.inference.types.Reply` objects back into the role's result; the transport does everything
around that (routing over replicas, retries, parking on outage, credentials, usage). An adapter is registered
under its ``name`` and selected from a config with ``api: <name>``: a third party ships its adapter in the
``rcp_ndcg.adapters`` entry-point group, and no config ever names a code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.types import Call, Reply, TokenCount

#: The entry-point group a third-party adapter registers in
#: (``[project.entry-points."rcp_ndcg.adapters"]``, e.g. ``bedrock_chat = "pkg.module:BedrockChat"``).
ADAPTER_ENTRY_POINTS = "rcp_ndcg.adapters"

#: The role an adapter speaks; the role fixes the request and result types around an adapter.
AdapterRole = Literal["judge", "embed", "rerank", "multi_vector"]

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
    """The adapter's name, the value a config's ``api`` field holds (``"openai_chat"``)."""

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


_BUILTINS: dict[str, type[Adapter[Any, Any]]] = {}
"""The adapters registered in this process; the shipped ones register at import of
:mod:`rcp_ndcg.inference.adapters`."""

_PLUGINS: dict[str, type[Adapter[Any, Any]]] | None = None
"""The adapters of the entry-point group, loaded once on first use (``None``: not loaded yet)."""


def register_adapter(cls: type[Adapter[Any, Any]]) -> type[Adapter[Any, Any]]:
    """Register an adapter class under its ``name`` (a class decorator; a duplicate name is refused).

    Args:
        cls: The adapter class; its ``name`` and ``role`` must be set.

    Returns:
        ``cls`` unchanged, so the decorator composes.

    Raises:
        ConfigError: ``cls`` has no or an empty ``name``, an unknown ``role``, or its ``name`` is already
            registered.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{cls.__name__} needs a non-empty `name` to be registered as an adapter")
    role = getattr(cls, "role", None)
    if role not in ("judge", "embed", "rerank", "multi_vector"):
        raise ConfigError(f"{cls.__name__}.role must be one of judge, embed, rerank, multi_vector, got {role!r}")
    if name in _BUILTINS:
        raise ConfigError(f"an adapter named {name!r} is already registered ({_BUILTINS[name].__name__})")
    _BUILTINS[name] = cls
    return cls


def _load_plugins() -> dict[str, type[Adapter[Any, Any]]]:
    """Load the ``rcp_ndcg.adapters`` entry points once; a broken one is an error, never a silent skip."""
    global _PLUGINS
    if _PLUGINS is None:
        loaded: dict[str, type[Adapter[Any, Any]]] = {}
        for entry in entry_points(group=ADAPTER_ENTRY_POINTS):
            try:
                adapter = entry.load()
            except Exception as exc:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) failed to import: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            loaded[entry.name] = adapter
        _PLUGINS = loaded
    return _PLUGINS


def known_adapters() -> tuple[str, ...]:
    """Every adapter name a config's ``api`` may name: the built-ins, then the entry-point group."""
    return tuple(sorted({*_BUILTINS, *_load_plugins()}))


def get_adapter(name: str) -> type[Adapter[Any, Any]]:
    """The adapter class a config's ``api: name`` selects: a built-in first, then the entry-point group.

    Args:
        name: The adapter's name (``"openai_chat"``, ``"rerank"``, or a third party's).

    Returns:
        The registered adapter class (not an instance: the caller -- a role client, or a third party --
        instantiates it with the role config its request fields depend on).

    Raises:
        ConfigError: No adapter of that name is registered; the hint lists the known names.
    """
    adapter = _BUILTINS.get(name) or _load_plugins().get(name)
    if adapter is None:
        known = known_adapters()
        if known:
            hint = f"known wire adapters: {', '.join(known)}"
        else:
            hint = (
                "no wire adapter is registered in this process; importing ``rcp_ndcg.inference.adapters`` "
                "registers the shipped ones, and a third party's in the "
                f"{ADAPTER_ENTRY_POINTS!r} entry-point group"
            )
        raise ConfigError(f"unknown adapter {name!r}", hint=hint, details={"known": list(known)})
    return adapter


__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
