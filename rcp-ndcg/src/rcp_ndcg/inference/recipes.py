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
  the config's explicit value (nothing it declares can conflict), and so does ``recipe`` itself (it is the
  pointer);
- ``model`` and ``revision`` come from the recipe's ids and are checked the same way;
- the RUNTIME fields (``base_url`` and the rest) stay on the config — a serve-by-role run leaves ``base_url``
  unset and the engines' URLs reach the step through ``RCP_NDCG_ENGINES``.

The CLI shorthand ``--retriever recipe:<id>`` / ``--reranker recipe:<id>`` (Q1) is the same mapping in one
string: :func:`shorthand_config` expands it and ``--set`` fills the runtime fields (e.g.
``--set base_url=http://127.0.0.1:8000/v1``).
"""

from __future__ import annotations

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


def _load(recipe_id: str):
    """The shipped recipe ``recipe_id`` (a variant id); a typed refusal naming the shipped ids when it
    is not shipped.

    The recipe's ``schema_version`` must be one this rcp-ndcg reads (:data:`RECIPE_SCHEMA_VERSIONS`): the
    recipe file format is the versioned contract between rcp-ndcg and rcp-ndcg-vllm (decision 18) -- there is
    no lockstep version pin between the two packages, so the check is here, at the read. Family ids are
    refused by the resolver itself (they are never served, decision 34).
    """
    module = _vllm_recipe_module()
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
    """The role of the shipped recipe ``recipe_id``: ``embed``, ``multi_vector`` or ``rerank``."""
    return str(_load(recipe_id).role)


def recipe_client_data(recipe_id: str) -> dict[str, Any]:
    """The product's client block of the shipped recipe: the recipe's plain ``client`` dict with its ``model``
    and ``revision`` injected (what :func:`expand_role_recipe` merges into a config)."""
    return dict(_load(recipe_id).client)


def expand_role_recipe(data: dict[str, Any], *, classes: dict[str, type]) -> dict[str, Any]:
    """The mapping form (Q1): a config mapping whose ``recipe`` names a shipped id takes its client block from
    it.

    Inputs: the parsed config mapping and the role -> endpoint-class map to read the CONTENT declarations
    from.  Outputs: a new mapping — the recipe's declared CONTENT plus the config's explicit fields (CONTENT
    equal or refused, with **both** values named; RUNTIME untouched).  A mapping without a ``recipe`` key, or
    whose ``recipe`` is ``None``, passes through unchanged.  Raises :class:`~rcp_ndcg.errors.ConfigError`:
    unknown recipe id (the shipped ids named), rcp-ndcg-vllm absent (the install line), an unreadable recipe
    ``schema_version`` (:data:`RECIPE_SCHEMA_VERSIONS`), or a CONTENT field that disagrees with the recipe.
    """
    if not isinstance(data, dict):
        return data
    recipe_id = data.get("recipe")
    if not isinstance(recipe_id, str) or not recipe_id:
        return data
    loaded = _load(str(recipe_id))
    client = dict(loaded.client)
    config_cls = classes[loaded.role]
    content = {name for name, role in declared_roles(config_cls).items() if role is FieldRole.CONTENT}
    merged: dict[str, Any] = {**client}
    for key, given in data.items():
        if key == "recipe":
            merged[key] = given  # the pointer: the config's own spelling (usually the recipe id)
            continue
        if key in ("model", "revision") or key in content:
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
    """The CLI shorthand (Q1): the string ``recipe:<id>`` as the mapping ``{"recipe": <id>}``; ``--set``
    overrides (``key=value``, dotted) are applied to it by the caller (e.g. ``--set base_url=...``).

    Inputs: the ``--retriever``/``--reranker`` value.  Outputs: the mapping form's dict.  Raises
    :class:`~rcp_ndcg.errors.ConfigError` for an empty id (an id is ``rcp-ndcg-vllm``'s to validate).
    """
    recipe_id = value.split(":", 1)[1].strip()
    if not recipe_id:
        raise ConfigError(
            "recipe:<id> names no recipe",
            hint="the form is recipe:<recipe-id>, e.g. --reranker recipe:qwen3-reranker-0.6b",
        )
    return {"recipe": recipe_id}
