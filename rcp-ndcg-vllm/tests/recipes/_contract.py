"""Pin a recipe's declared contract: every field of its ``serve``, ``client`` and ``reference`` blocks.

One home for the recipe contract pins (the sweep's shared part of "the contract tests must fail when a
serve or reference field drifts"): a recipe's test calls :func:`assert_recipe_contract` with the EXPECTED
mapping of each block, plus any top-level fields it pins (``licence``, ``revision``, ...). The comparison
is exact and complete in both directions -- a drifted value fails naming the field path, and so does a
field the expected mapping omits (nothing rides unpinned, so a schema default that moves reds here and is
re-pinned deliberately). The blocks are pinned as the product's models resolve them
(``model_dump(mode="json")``): authored values and schema defaults alike. The endpoint's runtime
``base_url`` is the one field excluded (the harness fills it per run).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["assert_recipe_contract"]

_MISSING = object()

_CLIENT_RUNTIME_FIELDS = frozenset({"base_url"})
"""The endpoint's runtime fields, never pinned: the harness fills base_url per run."""


def assert_recipe_contract(
    recipe: Any,
    *,
    serve: Mapping[str, Any],
    client: Mapping[str, Any],
    reference: Mapping[str, Any],
    top: Mapping[str, Any] | None = None,
) -> None:
    """Pin every field of the recipe's resolved ``serve``, ``client`` and ``reference`` blocks.

    Inputs: the loaded recipe and, per block, its full expected mapping (plus optional top-level
    recipe fields). Outputs: none. Raises ``AssertionError`` naming the exact field path of the
    first difference -- a value drift, a field the mapping never pinned, or a key the block does
    not carry.

    Args:
        recipe: the loaded ``Recipe`` (``load_recipe``).
        serve: the full expected ``serve`` block.
        client: the full expected ``client`` block (minus the runtime ``base_url``).
        reference: the full expected ``reference`` block.
        top: optional top-level recipe fields to pin (``licence``, ``id``, ``role``, ...).
    """
    _assert_block("serve", recipe.serve, serve, skips=frozenset())
    _assert_block("client", recipe.client, client, skips=_CLIENT_RUNTIME_FIELDS)
    _assert_block("reference", recipe.reference, reference, skips=frozenset())
    for key, value in (top or {}).items():
        actual = getattr(recipe, key, _MISSING)
        if actual is _MISSING:
            raise AssertionError(f"recipe.{key}: the recipe has no such field")
        if actual != value:
            raise AssertionError(f"recipe.{key} == {actual!r}, expected {value!r}")


def _assert_block(name: str, model: Any, expected: Mapping[str, Any], *, skips: frozenset[str]) -> None:
    """One whole block against its expected mapping, runtime fields excluded.

    The client block is plain data (the lean package validates no endpoint model); it pins as the YAML
    declares it, with the recipe's own ``model`` and ``revision`` injected."""
    actual = dict(model if isinstance(model, Mapping) else model.model_dump(mode="json"))
    for key in skips & set(actual):
        del actual[key]
    _assert_mapping(name, actual, dict(expected))


def _assert_mapping(path: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """A mapping against its expectation: same keys, every leaf equal, nested mappings recursed."""
    unpinned = sorted(set(actual) - set(expected))
    if unpinned:
        raise AssertionError(
            f"{path}: unpinned field(s) {unpinned}: pin every field of the block "
            "(the mapping must carry the recipe's full resolved contract)"
        )
    unknown = sorted(set(expected) - set(actual))
    if unknown:
        raise AssertionError(f"{path}: expected field(s) {unknown} do not exist on the block")
    for key in sorted(actual):
        found, want = actual[key], expected[key]
        if isinstance(found, Mapping) and isinstance(want, Mapping):
            _assert_mapping(f"{path}.{key}", found, want)
        elif found != want:
            raise AssertionError(f"{path}.{key} == {found!r}, expected {want!r}")
