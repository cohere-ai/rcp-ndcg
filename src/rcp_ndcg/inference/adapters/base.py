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
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.types import Call, Reply, TokenCount

if TYPE_CHECKING:
    from rcp_ndcg.support.serve import EngineRole

#: The entry-point group a third-party adapter registers in
#: (``[project.entry-points."rcp_ndcg.adapters"]``, e.g. ``embed.bedrock = "pkg.module:BedrockEmbed"``).
ADAPTER_ENTRY_POINTS = "rcp_ndcg.adapters"

#: The role an adapter speaks; the role fixes the request and result types around an adapter.
AdapterRole = Literal["judge", "embed", "rerank", "multi_vector"]

#: Every role an adapter may serve, in the registry's and the entry-point group's namespace.
ROLES: tuple[str, ...] = ("judge", "embed", "rerank", "multi_vector")

#: The one written mapping between the two role vocabularies (F7): the adapter roles an engine of each
#: :data:`~rcp_ndcg.support.serve.EngineRole` speaks. An ``encoder`` engine serves either wire of the
#: retrieval configs (a dense checkpoint speaks an ``embed`` adapter, a late-interaction checkpoint a
#: ``multi_vector`` one); a ``reranker`` engine speaks a ``rerank`` adapter; a ``judge`` engine a ``judge``
#: one. Written here -- the layering allows ``inference`` to import ``support``, never the reverse -- so
#: nothing else re-derives it, and :func:`check_engine_api` refuses a config whose ``api`` selects an
#: adapter of a different engine role where the runners and the run config resolve engines.
ENGINE_ADAPTER_ROLES: dict[EngineRole, frozenset[AdapterRole]] = {
    "judge": frozenset({"judge"}),
    "encoder": frozenset({"embed", "multi_vector"}),
    "reranker": frozenset({"rerank"}),
}

Req = TypeVar("Req", contravariant=True)
"""The request type an adapter consumes (a role's request type); contravariant: adapters are called."""

Res = TypeVar("Res", covariant=True)
"""The result type an adapter produces (a role's result type)."""


#: The three members that make a class an adapter; a registration without them fails here, never at the
#: first request.
ADAPTER_MEMBERS: tuple[str, ...] = ("calls", "interpret", "usage")

#: The credential facts every adapter declares -- the ones the role clients read with declared defaults,
#: where a silent default could send a key (or refuse one) behind the author's back. An adapter that
#: subclasses :class:`AdapterBase` inherits them; a third-party class declares them itself.
ADAPTER_FACTS: tuple[str, ...] = ("HOSTED", "API_KEY_ENV", "KEY_REQUIRED", "AUTH_HEADER", "DEFAULT_BASE_URL")


class AdapterBase:
    """The base every adapter subclasses: the credential and capability facts with their declared defaults,
    and the constructor convention.

    A role client instantiates its adapter with the role config -- ``adapter_cls(config)`` -- and the base
    stores it (an adapter whose requests depend on no config field accepts and ignores it; the rerank
    family reads its fields). The class attributes below are the declared contract: a subclass inherits
    them or overrides them, and :func:`register_adapter` refuses a class that declares none -- a missing
    fact used to be silently duck-typed with a default that could be wrong (a ``KEY_REQUIRED`` default of
    ``False`` on a wire that requires a key, an ``OPENAI_API_KEY`` sent to a stranger's host).

    Attributes:
        config: The role config the adapter was built with, when its requests depend on one.
        name: The adapter's name within its role, the value a config's ``api`` field holds; set per concrete
            class (empty names are refused at registration).
        role: The role the adapter serves; it fixes which request and result types flow through it.
        HOSTED: Whether this wire is a hosted vendor profile (its public API root is its default
            ``base_url``, its key is required): declared, never inferred from the default URL.
        API_KEY_ENV: The environment variables that may hold the API key, most preferred first; the config's
            ``api_key_env`` names one instead. The transport resolves the key and sends it in
            :attr:`AUTH_HEADER`; an adapter never touches a key. Empty: the endpoint takes no key.
        KEY_REQUIRED: Whether the API refuses to answer without a key (the hosted profiles) or takes none.
        AUTH_HEADER: The header the key goes in; ``None`` is the OpenAI-standard ``Authorization: Bearer``.
        DEFAULT_BASE_URL: The hosted profile's public API root, used when the config sets no ``base_url``;
            ``None``: ``base_url`` is required (a served endpoint has no public root). The profile's default
            key variables apply only at this host (any other ``base_url`` carries a key only through the
            config's ``api_key_env``).
        MAX_BATCH: The texts/items-per-request cap the API publishes; ``None`` lets the server decide (its
            over-count refusal is mapped to :class:`~rcp_ndcg.errors.CapabilityError`).
        SUPPORTS_DIMENSIONS: Whether this route takes a ``dimensions`` parameter (a Matryoshka cut).
        ENCODING_FORMAT: The ``encoding_format`` request field; ``None`` leaves it out (the routes that have
            no such field).
    """

    name: ClassVar[str]
    role: ClassVar[AdapterRole]

    HOSTED: ClassVar[bool] = False
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ()
    KEY_REQUIRED: ClassVar[bool] = False
    AUTH_HEADER: ClassVar[str | None] = None
    DEFAULT_BASE_URL: ClassVar[str | None] = None
    MAX_BATCH: ClassVar[int | None] = None
    SUPPORTS_DIMENSIONS: ClassVar[bool] = True
    ENCODING_FORMAT: ClassVar[str | None] = None

    def __init__(self, config: Any = None) -> None:
        """Build the adapter for ``config`` -- the role config whose ``api`` selected it (an adapter whose
        requests depend on no config field accepts and ignores it)."""
        self.config = config


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
"""The adapters registered in this process, keyed ``(role, name)``; the shipped ones register at import of
:mod:`rcp_ndcg.inference.adapters`."""

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


def _check_adapter_shape(cls: Any, *, entry: str | None = None) -> None:
    """The registration-time shape of an adapter class: the three members callable, the credential facts
    declared. A class that fails here registers nothing -- before the fix, a member missing was a first
    request's ``AttributeError`` and a missing fact was a silent (possibly wrong) default."""
    where = f"the adapter entry point {entry!r}" if entry else f"{cls.__name__}"
    for member in ADAPTER_MEMBERS:
        if not callable(getattr(cls, member, None)):
            raise ConfigError(
                f"{where} is not an adapter: it has no callable {member}(...)",
                hint="an adapter implements calls, interpret and usage; subclass "
                "rcp_ndcg.inference.adapters.base.AdapterBase for the facts and the constructor convention",
            )
    for fact in ADAPTER_FACTS:
        if not hasattr(cls, fact):
            raise ConfigError(
                f"{where} declares no credential fact {fact}",
                hint=f"declare {', '.join(ADAPTER_FACTS)} (or subclass "
                "rcp_ndcg.inference.adapters.base.AdapterBase, which carries the declared defaults): a "
                "missing fact would be silently duck-typed, and the default could send a key where none "
                "belongs",
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
        ConfigError: ``cls`` has no or an empty ``name``, an unknown ``role``, does not implement the
            adapter members, declares none of the credential facts, or its ``(role, name)`` is already
            registered.
    """
    _check_adapter_shape(cls)
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

    Each entry is named ``<role>.<name>`` (``embed.bedrock``) with ``<name>`` the class's registered name; an
    adapter whose class role or name disagrees with its entry name is refused, so a typo cannot route one
    role's requests to another role's adapter or register a class under a name no entry point spells. A
    plugin is registered under its class's ``(role, name)``; one that would take a shipped adapter's key is
    refused, so a built-in is never silently shadowed.
    """
    global _PLUGINS
    if _PLUGINS is None:
        loaded: dict[tuple[str, str], type[Adapter[Any, Any]]] = {}
        for entry in entry_points(group=ADAPTER_ENTRY_POINTS):
            prefix, separator, suffix = entry.name.partition(".")
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
            _check_adapter_shape(adapter, entry=entry.name)
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
            if key[1] != suffix:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) loads "
                    f"{getattr(adapter, '__name__', adapter)!r}, which registers as {key[1]!r}, not {suffix!r} "
                    "as its entry name's <role>.<name> declares"
                )
            if key in _BUILTINS:
                raise ConfigError(
                    f"the adapter entry point {entry.name!r} ({entry.value}) registers {key[1]!r} for the "
                    f"{prefix} role, where {_BUILTINS[key].__name__} is already registered"
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
                f"no {role} wire adapter is registered in this process; importing ``rcp_ndcg.inference.adapters`` "
                f"registers the shipped ones, and a third party's in the {ADAPTER_ENTRY_POINTS!r} entry-point group"
            )
        if elsewhere:
            hint += f"; {name!r} is registered for the {', '.join(elsewhere)} role(s)"
        raise ConfigError(f"unknown {role} adapter {name!r}", hint=hint, details={"known": list(known), "role": role})
    return adapter


def adapter_roles_of(engine_role: EngineRole) -> frozenset[AdapterRole]:
    """The adapter roles an engine of ``engine_role`` speaks (the values of :data:`ENGINE_ADAPTER_ROLES`).

    Raises:
        ConfigError: ``engine_role`` is not an engine role (``judge``, ``encoder``, ``reranker``).
    """
    try:
        return ENGINE_ADAPTER_ROLES[engine_role]  # type: ignore[index]  # an unknown role is refused below
    except KeyError:
        raise ConfigError(
            f"{engine_role!r} is not an engine role",
            hint=f"an engine role is one of {', '.join(sorted(ENGINE_ADAPTER_ROLES))}",
            details={"engine_role": engine_role, "known": sorted(ENGINE_ADAPTER_ROLES)},
        ) from None


def check_engine_api(api: str | None, *, engine_role: EngineRole, where: str) -> None:
    """Refuse a config whose ``api`` selects no adapter of ``engine_role``'s adapter roles (F7).

    Called where the runners and the run config resolve engines onto configs (the engine overlay), so a
    config whose ``api`` names an adapter of a different role is refused at resolution time, not at the
    first request.

    Args:
        api: The config's ``api`` field; ``None`` (no wire adapter declared -- the retrieval configs' and the
            judge's today) is not checked.
        engine_role: The engine role the config is being resolved for.
        where: What is being resolved, for the error message (``"serve.encoder"``).

    Raises:
        ConfigError: ``api`` names an adapter, and none of the engine role's adapter roles registers it.
    """
    if api is None:
        return
    roles = adapter_roles_of(engine_role)
    for role in sorted(roles):
        if (role, api) in _BUILTINS or (role, api) in _load_plugins():
            return
    registered = (*_BUILTINS, *_load_plugins())
    elsewhere = sorted({registered_role for registered_role, name in registered if name == api})
    hint = (
        f"the {engine_role} role speaks these adapter roles: {', '.join(sorted(roles))}"
        if elsewhere == []
        else f"{api!r} is registered for the {', '.join(elsewhere)} role(s), not for the {engine_role} "
        f"role's adapters ({', '.join(sorted(roles))})"
    )
    raise ConfigError(
        f"{where}: the config's api {api!r} names an adapter outside the {engine_role} engine's roles",
        hint=hint,
        details={"api": api, "engine_role": engine_role, "adapter_roles": sorted(roles)},
    )


__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "ADAPTER_FACTS",
    "ADAPTER_MEMBERS",
    "AdapterBase",
    "ENGINE_ADAPTER_ROLES",
    "Adapter",
    "AdapterRole",
    "ROLES",
    "adapter_roles_of",
    "check_engine_api",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
