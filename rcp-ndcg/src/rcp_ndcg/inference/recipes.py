"""``recipe: <id>``: a role config that names a shipped serving recipe takes its client block from it.

One home for the mapping form (docs-firstcontact Q1). The shipped recipes live in ``rcp-ndcg-vllm``'s package
data (``rcp_ndcg_vllm.recipe``); this module imports that package **lazily** (the layering charter's exemption:
rcp-ndcg may read rcp-ndcg-vllm's recipe data, rcp-ndcg-vllm never imports rcp-ndcg). When it is absent, the
refusal is a typed :class:`~rcp_ndcg.errors.ConfigError` with the exact install line (docs-site OQ-6).

The recipe **file format is the versioned contract** between the two packages (owner decision 18): every
recipe carries ``schema_version``, and this module refuses a recipe whose version it cannot read — the check
runs before the merge, so a newer rcp-ndcg-vllm and an older rcp-ndcg fail with a versioned message instead of
a schema surprise (there is no lockstep version pin between the packages; core and rcp-ndcg keep theirs).

The merge (Q1), over the recipe's plain ``client`` block:

- the CONTENT fields (per the endpoint's ``IDENTITY_ROLES`` declarations) come from the recipe — every CONTENT
  field the config sets explicitly must equal the recipe's declared value or the refusal is a
  :class:`~rcp_ndcg.errors.ConfigError` naming **both** values; a field the recipe leaves undeclared accepts
  the config's explicit value (nothing it declares can conflict);
- ``model`` and ``revision`` come from the recipe's ids and are checked the same way;
- the RUNTIME fields (``base_url`` and the rest) stay on the config — a serve-by-role run leaves ``base_url``
  unset and the engines' URLs reach the step through ``RCP_NDCG_ENGINES``.

The ``recipe`` pointer itself is replaced by the recipe's **identity**: the shipped id, or
``unshipped:sha256:<hex>`` for a recipe loaded from a path — the content hash of its resolved form, so two runs
whose files differ never share a run identity and the config's own spelling of the path is not part of it.

``recipe:`` names a shipped variant id, or **a file of the operator's own**: ``recipe:./my-family``,
``recipe:../my-family/family.yaml`` or ``recipe:/abs/path`` load a family directory through rcp-ndcg-vllm's own
loader (the same schema, families included), marked unshipped with ``status: unverified``.  The
``schema_version`` check below applies to both unchanged.

The CLI shorthand ``--retriever recipe:<id>`` / ``--reranker recipe:<id>`` (Q1) is the same mapping in one
string: :func:`shorthand_config` expands it and ``--set`` fills the runtime fields (e.g.
``--set base_url=http://127.0.0.1:8000/v1``).
"""

from __future__ import annotations

import re
from typing import Any

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.identity import FieldRole, declared_roles

__all__ = [
    "RECIPE_SCHEMA_VERSIONS",
    "available_recipe_ids",
    "expand_role_recipe",
    "recipe_client_data",
    "recipe_role",
    "shorthand_config",
]

_INSTALL_LINE = "pip install rcp-ndcg-vllm"

_IDENTITY_POINTER = re.compile(r"^unshipped:sha256:[0-9a-f]{64}$")
"""An unshipped recipe's identity exactly as :attr:`~rcp_ndcg_vllm.recipe.Recipe.identity` writes it.

A config whose ``recipe`` carries one has already been expanded from the file: every client field of the
recipe is in the mapping, and the file it came from may not even exist on the machine reading the config
again (a run config's resume, a retrieval index's reload), so it is passed through untouched -- there is
nothing left to resolve."""

_MRL_SELECTION_FIELDS = frozenset({"mrl_dim", "dimensions"})
"""The client fields a run selects a Matryoshka width with: ``mrl_dim`` (the client head, both wires) and
``dimensions`` (the engine-side cut, dense only).  A recipe declares the kind and the card's set once and
ships the checkpoint's full width (``dimensions: null``, no ``mrl_dim``); the selection is the run's, so a
``k`` inside the declared set is not a CONTENT disagreement with the recipe's own ``null``."""

RECIPE_SCHEMA_VERSIONS = frozenset({"1"})
"""The recipe file-format versions this rcp-ndcg reads (decision 18: the recipe file format is the versioned
contract, so the reader names the versions it understands instead of pinning the sibling package)."""


def _vllm_recipe_module():
    """``rcp_ndcg_vllm.recipe``, or a typed refusal with the install line when the package is absent."""
    try:
        from rcp_ndcg_vllm import recipe as vllm_recipe  # noqa: PLC0415 -- lazy by the charter's exemption
    except ModuleNotFoundError as error:
        raise ConfigError(
            f"recipe: ... needs the serving recipes of rcp-ndcg-vllm ({error})",
            hint=f"install it alongside rcp-ndcg: {_INSTALL_LINE} (it depends on pydantic and PyYAML only)",
        ) from error
    return vllm_recipe


def available_recipe_ids() -> frozenset[str]:
    """The ids of every shipped recipe (the variant ids of ``rcp-ndcg-vllm``'s family directories).

    Loads each family and collects its variant ids; a family that fails to load is skipped (its error
    surfaces when someone resolves one of its recipes), so one broken family must not hide the rest of
    the catalogue (the wave runner's promise, HARNESS-1). Family ids are never listed: they are never
    served (decision 34).
    """
    module = _vllm_recipe_module()
    root = module.default_recipes_root()
    ids: set[str] = set()
    from rcp_ndcg_vllm.errors import RecipeError

    for directory in sorted(root.iterdir()):
        if not (directory / "family.yaml").is_file():
            continue
        try:
            family = module.load_family(directory)
        except RecipeError:
            continue
        ids.update(variant.id for variant in family.variants)
    return frozenset(ids)


def _looks_like_a_path(recipe_id: str) -> bool:
    """Whether ``recipe_id`` is a filesystem path rather than a shipped recipe id.

    The shipped ids never start with a dot or a slash (``^[a-z0-9][a-z0-9.-]*$``), so the path forms
    ``./dir``, ``../dir`` and ``/abs/dir`` are unambiguous; anything else is an id.  ``~`` is not a path form
    here: nothing expands it, so it is refused like any unknown id.
    """
    return recipe_id.startswith((".", "/"))


def _load(recipe_id: str):
    """The recipe ``recipe_id``: a shipped variant id, or a path of the operator's own.

    A shipped id resolves under the package's recipes root; a path form loads the family directory (or
    ``family.yaml``) through rcp-ndcg-vllm's own loader, which marks it unshipped and unverified and gives it
    the content-hash identity :func:`expand_role_recipe` puts in the config.  Either way the recipe's
    ``schema_version`` must be one this rcp-ndcg reads (:data:`RECIPE_SCHEMA_VERSIONS`): the recipe file format
    is the versioned contract between rcp-ndcg and rcp-ndcg-vllm (decision 18) -- there is no lockstep version
    pin between the two packages, so the check is here, at the read.  Family ids are refused by the resolver
    itself (they are never served, decision 34).
    """
    module = _vllm_recipe_module()
    if _looks_like_a_path(recipe_id):
        try:
            loaded = module.load_recipe(recipe_id)
        except module.RecipeError as error:
            raise ConfigError(
                f"recipe: {recipe_id}: {error}",
                hint="a recipe path names a family directory (or its family.yaml): the file must satisfy the "
                "family schema and name the files it references beside it",
            ) from error
    else:
        try:
            loaded = module.resolve_recipe(recipe_id)
        except module.RecipeError as error:
            known = ", ".join(sorted(available_recipe_ids()))
            raise ConfigError(
                f"recipe: {recipe_id}: no shipped recipe of that id",
                hint=f"the shipped recipes are: {known} (the resolver said: {error})",
            ) from error
    version = getattr(loaded, "schema_version", None)
    if version is not None and str(version) not in RECIPE_SCHEMA_VERSIONS:
        known = ", ".join(sorted(RECIPE_SCHEMA_VERSIONS))
        raise ConfigError(
            f"recipe: {recipe_id}: schema_version {version!r} is not one this rcp-ndcg reads ({known})",
            hint="the recipe file format is the versioned contract between rcp-ndcg and rcp-ndcg-vllm "
            "(decision 18): upgrade rcp-ndcg, or ship recipes of a schema version it reads",
        )
    return loaded


def recipe_role(recipe_id: str) -> str:
    """The role of the recipe ``recipe_id``: ``embed``, ``multi_vector`` or ``rerank``."""
    return str(_load(recipe_id).role)


def recipe_client_data(recipe_id: str) -> dict[str, Any]:
    """The product's client block of the recipe ``recipe_id``: the recipe's plain ``client`` dict with its
    ``model`` and ``revision`` injected (what :func:`expand_role_recipe` merges into a config)."""
    return dict(_load(recipe_id).client)


def _mrl_declaration(client: dict[str, Any]) -> tuple[str, Any] | None:
    """The recipe's declared Matryoshka set or range, as ``(text, membership)``, or ``None``.

    Reads the plain ``client`` block: ``mrl_dims`` (the card's discrete table) or ``mrl_range``
    (``[min, max]``, the card's prose range).  ``None`` when the recipe declares neither -- there is no set
    to select from (the endpoint's own validation names the kind then) -- or when the declaration is not the
    shape this rule reads, which the product's endpoint validation refuses with a better message.
    """
    try:
        dims = client.get("mrl_dims")
        if dims is not None:
            values = tuple(int(dim) for dim in dims)
            return f"mrl_dims {values}", lambda k: k in values
        mrl_range = client.get("mrl_range")
        if mrl_range is not None:
            low, high = int(mrl_range[0]), int(mrl_range[1])
            return f"mrl_range [{low}, {high}]", lambda k: low <= k <= high
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _selected_dimension(key: str, given: Any, config_cls: type) -> int | None:
    """The positive integer ``given`` selects for the field ``key``, or ``None`` when it is not one.

    Validates through the field's own pydantic annotation, so a YAML string (``mrl_dim: "512"``) selects
    like the integer it coerces to and a value the field would refuse falls through to the ordinary merge
    and the endpoint's own validation.
    """
    from pydantic import TypeAdapter, ValidationError

    try:
        value = TypeAdapter(config_cls.model_fields[key].annotation).validate_python(given)
    except ValidationError:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def expand_role_recipe(data: dict[str, Any], *, classes: dict[str, type]) -> dict[str, Any]:
    """The mapping form (Q1): a config mapping whose ``recipe`` names a recipe takes its client block from it.

    Inputs: the parsed config mapping (``recipe`` a shipped variant id or a path: ``./dir``, ``/abs/dir``) and
    the role -> endpoint-class map to read the CONTENT declarations from.  Outputs: a new mapping — the recipe's
    declared CONTENT plus the config's explicit fields (CONTENT equal or refused, with **both** values named;
    RUNTIME untouched), with ``recipe`` replaced by the recipe's identity (its shipped id, or
    ``unshipped:sha256:<hex>``).  One exception, the MRL selection (owner decision 39): the recipe declares its
    Matryoshka kind and the card's set (``mrl_kind`` with ``mrl_dims``/``mrl_range``) and ships the checkpoint's
    full width, so a config's ``mrl_dim``/``dimensions`` is a *selection* — accepted when ``k`` is in the
    declared set, refused naming the set otherwise — not a disagreement with the recipe's declared ``null``.
    A mapping without a ``recipe`` key, or whose ``recipe`` is ``None``, passes through unchanged.  Raises
    :class:`~rcp_ndcg.errors.ConfigError`: unknown recipe id (the shipped ids named), a path rcp-ndcg-vllm's
    loader refuses, rcp-ndcg-vllm absent (the install line), an unreadable recipe ``schema_version``
    (:data:`RECIPE_SCHEMA_VERSIONS`), a CONTENT field that disagrees with the recipe, or an MRL selection
    outside the recipe's declared set.
    """
    if not isinstance(data, dict):
        return data
    recipe_id = data.get("recipe")
    if not isinstance(recipe_id, str) or not recipe_id:
        return data
    if _IDENTITY_POINTER.fullmatch(recipe_id):
        # An already-expanded config (a recorded run config, a written index): the pointer is the recipe's
        # identity and the block beside it is the recipe's own, so there is nothing to load again.
        return data
    loaded = _load(str(recipe_id))
    client = dict(loaded.client)
    config_cls = classes[loaded.role]
    content = {name for name, role in declared_roles(config_cls).items() if role is FieldRole.CONTENT}
    merged: dict[str, Any] = {**client}
    for key, given in data.items():
        if key == "recipe":
            # The pointer is the recipe's identity, never the config's own spelling of it: the shipped id, or
            # ``unshipped:sha256:<hex>`` for a file of the operator's own -- the content hash of its resolved
            # form, so two runs whose files differ never share a run identity (and the path itself is not in
            # the identity: the same file at two paths is one recipe).
            merged[key] = loaded.identity
            continue
        if key in ("model", "revision") or key in content:
            if key in _MRL_SELECTION_FIELDS:
                declaration = _mrl_declaration(client)
                selected = _selected_dimension(key, given, config_cls)
                if selected is not None:
                    if declaration is not None:
                        declared_text, supports = declaration
                        if supports(selected):
                            # A declared selection, not a disagreement with the recipe's own (undeclared)
                            # one: the recipe declares the kind and the card's set, the run selects k.
                            merged[key] = selected
                            continue
                        raise ConfigError(
                            f"{key}: {selected} is not in recipe {recipe_id!r}'s declared {declared_text}",
                            hint=f"select a k in {declared_text}, or drop {key} (the recipe serves the "
                            "checkpoint's full width; the declared set bounds every selection)",
                        )
                    if client.get("mrl_kind") in (None, "none"):
                        raise ConfigError(
                            f"{key}: {selected} selects a Matryoshka output, but recipe {recipe_id!r} "
                            f"declares mrl_kind {client.get('mrl_kind') or 'none'!r}: the checkpoint's card "
                            "declares no Matryoshka head",
                            hint=f"drop {key} (the recipe serves the checkpoint's full width), or name a "
                            "recipe that declares the card's set",
                        )
            expected = client.get(key)
            if key not in client:
                merged[key] = given  # nothing the recipe declares can conflict
            elif _content_equal(given, expected, config_cls.model_fields[key].annotation):
                merged[key] = expected
            else:
                raise ConfigError(
                    f"{key}: {given!r} disagrees with recipe {recipe_id!r}, which declares {expected!r}",
                    hint=f"drop {key} (the client block comes from recipe: {recipe_id}) or name a recipe that "
                    f"declares {given!r}; both values are: config {given!r} vs recipe {expected!r}",
                )
        else:
            merged[key] = given  # RUNTIME (and the endpoint's own machinery): the config's
    merged.setdefault("recipe", recipe_id)
    return merged


def _content_equal(given: Any, expected: Any, annotation: Any) -> bool:
    """Whether a config's explicit CONTENT value agrees with the recipe's declared one.

    Plain equality first; for nested declarations (a template, a media policy) the compared forms may differ
    only in the schema defaults the endpoint model fills (a re-validated dump carries ``anchor_markers: []``
    where the recipe's YAML omits the key), so both sides go through the field's own pydantic type: two
    values the product cannot distinguish agree.
    """
    if given == expected:
        return True
    if isinstance(given, dict) and isinstance(expected, dict) and annotation is not None:
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(annotation)
        try:
            return bool(adapter.validate_python(given) == adapter.validate_python(expected))
        except ValidationError:
            return False
    return False


def shorthand_config(value: str) -> dict[str, Any]:
    """The CLI shorthand (Q1): the string ``recipe:<id-or-path>`` as the mapping ``{"recipe": <id-or-path>}``;
    ``--set`` overrides (``key=value``, dotted) are applied to it by the caller (e.g. ``--set base_url=...``).

    Inputs: the ``--retriever``/``--reranker`` value.  Outputs: the mapping form's dict.  Raises
    :class:`~rcp_ndcg.errors.ConfigError` for an empty id (an id is ``rcp-ndcg-vllm``'s to validate; a path
    form -- ``./dir``, ``/abs/dir`` -- is resolved by :func:`expand_role_recipe`).
    """
    recipe_id = value.split(":", 1)[1].strip()
    if not recipe_id:
        raise ConfigError(
            "recipe:<id> names no recipe",
            hint="the form is recipe:<recipe-id>, e.g. --reranker recipe:qwen3-reranker-0.6b",
        )
    return {"recipe": recipe_id}
