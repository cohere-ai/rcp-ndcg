"""Identity and hashing: the one home for "are these two things interchangeable?".

The digest recipe (canonical JSON bytes, then SHA-256: :func:`hash_payload`,
:func:`hash_text`, :func:`short`) is implemented once in the core and re-exported
here; every identity digest of the pipeline (a judgement family, a store's
identity, a run step's identity, a calibration's) is computed through it.

Declared identities: which config fields may change a key, and which may not.

A key ("store identity", "step identity", "family key") is a claim -- *two
things with this key are interchangeable*. Two ways to break it:

**False sharing** -- an output-relevant field is missing from the key. Two runs
that computed different things share one output path; a resume mixes them; a
pooled fit mixes rubrics. Wrong numbers, no error.

**False splitting** -- an output-irrelevant field is in the key. Retuning a
timeout or moving an endpoint moves the output tree and restarts a multi-hour
run from zero. Lost compute, no wrong numbers.

So every field of every model that feeds a key carries a declared role:

``CONTENT``
    changes *what* is computed -- in the key.
``RUNTIME``
    changes *how fast* or *where*, never *what* -- out of the key.

The roles live next to the fields, as ``IDENTITY_ROLES`` on the model, because
the failure this prevents is *a field added without a decision*: a separate list
of runtime knobs drifts from the models it describes. The declared model is the
judge (:class:`~rcp_ndcg.judging.JudgeConfig`: model, revision and sampling are
CONTENT; endpoint, concurrency, timeouts, retries and capability gates are
RUNTIME); a run's step identities are built from the judge's payload and the
:class:`~rcp_ndcg.runs.RunConfig` sections they depend on.
:func:`check_declarations` turns a missing declaration into a failing test.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from rcp_ndcg_core._hashing import (
    UnhashableValueError,
    canonical,
    combine_digests,
    hash_payload,
    hash_strings,
    hash_text,
    short,
)


class FieldRole(StrEnum):
    """What a config field may do to an identity."""

    CONTENT = "content"
    RUNTIME = "runtime"


class IdentityDeclarationError(TypeError):
    """A model that feeds an identity has an undeclared or stale field role."""


def declared_roles(model_cls: type[BaseModel]) -> dict[str, FieldRole]:
    """Roles declared for ``model_cls``, merged down its MRO.

    A subclass may add fields and may re-declare an
    inherited one; the most derived declaration wins.
    """
    merged: dict[str, FieldRole] = {}
    for klass in reversed(model_cls.__mro__):
        roles = klass.__dict__.get("IDENTITY_ROLES")
        if roles is None:
            continue
        if not isinstance(roles, Mapping):
            raise IdentityDeclarationError(f"{klass.__name__}.IDENTITY_ROLES must be a mapping of field -> FieldRole")
        for name, role in roles.items():
            merged[name] = FieldRole(role)
    return merged


def check_declarations(model_cls: type[BaseModel]) -> None:
    """Raise unless every field of ``model_cls`` has exactly one declared role.

    Both directions are errors:

    * an **undeclared** field -- somebody added a knob and nobody decided
      whether it changes what is computed;
    * a **stale** declaration -- the field it names is gone, so the mapping is
      a record of a model that no longer exists.
    """
    roles = declared_roles(model_cls)
    if not roles:
        raise IdentityDeclarationError(
            f"{model_cls.__name__} feeds an identity but declares no IDENTITY_ROLES; "
            "classify each field CONTENT / RUNTIME."
        )
    fields = set(model_cls.model_fields)
    undeclared = sorted(fields - set(roles))
    stale = sorted(set(roles) - fields)
    problems: list[str] = []
    if undeclared:
        problems.append(
            f"undeclared field(s) {undeclared}: add each to {model_cls.__name__}.IDENTITY_ROLES as "
            "CONTENT (changes what is computed) or RUNTIME (changes only how fast/where)"
        )
    if stale:
        problems.append(f"declared but absent field(s) {stale}: delete them from IDENTITY_ROLES")
    if problems:
        raise IdentityDeclarationError(f"{model_cls.__name__}: " + "; ".join(problems))


def _content_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return identity_payload(value)
    if isinstance(value, Mapping):
        return {key: _content_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_content_value(item) for item in value]
    return value


def identity_payload(model: BaseModel) -> dict[str, Any]:
    """The CONTENT-only payload of ``model``, ready for the canonical serialiser.

    Nested models are recursed through the same rule, so a runtime knob inside a
    nested model cannot reach a key by being nested.

    Fields whose value is ``None`` are omitted: an *unset* optional field is the
    absence of a declaration, and adding a new optional CONTENT field must not
    re-key every run that does not set it. Setting it to anything else does.

    Keys an ``extra="allow"`` model received are CONTENT: a needless re-key costs
    compute, a missed one costs numbers.
    """
    check_declarations(type(model))
    roles = declared_roles(type(model))
    payload: dict[str, Any] = {}
    for name in type(model).model_fields:
        if roles[name] is not FieldRole.CONTENT:
            continue
        value = getattr(model, name, None)
        if value is None:
            continue
        payload[name] = _content_value(value)
    for name, value in (model.__pydantic_extra__ or {}).items():
        if value is not None:
            payload[name] = _content_value(value)
    return payload


def identity_differences(recorded: Any, current: Any, *, path: str = "") -> list[str]:
    """Dotted paths at which two identity payloads differ, most specific first.

    What turns "cache mismatch" into a message an operator can act on: the old
    sidecar check compared two digests and could only say they were not equal.

    Leaves are compared by their **canonical form**, not by ``==``, because this
    function and the key must never disagree about what "the same" means. Python
    says ``1 == 1.0`` and ``True == 1``; the canonical serialiser deliberately
    keeps them apart, because a count of ``1`` and a fraction of ``1.0`` are
    different settings. With ``==`` here, a changed setting would compare equal,
    and a store's identity would be rewritten with the new value.
    """
    if isinstance(recorded, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(recorded) | set(current)):
            here = f"{path}.{key}" if path else key
            if key not in recorded:
                differences.append(f"{here}: absent in record, now {current[key]!r}")
            elif key not in current:
                differences.append(f"{here}: {recorded[key]!r} in record, now absent")
            else:
                differences.extend(identity_differences(recorded[key], current[key], path=here))
        return differences
    if not _same_leaf(recorded, current):
        return [f"{path or '<root>'}: recorded {recorded!r}, now {current!r}"]
    return []


def _same_leaf(recorded: Any, current: Any) -> bool:
    """Whether two identity leaves are the same value *to the key*.

    Both containers recurse: a list elementwise, and a **dict** key by key. Without the
    dict half, ``{"a": [{"x": 1}]}`` and ``{"a": [{"x": 1.0}]}`` would fall through to
    ``==`` on two dicts, which Python calls equal while ``canonical_json`` (correctly)
    does not, so a record and a request that key differently would compare as
    identical. Free-form payloads such as ``chat_template_kwargs`` can hold a dict inside
    a list, and this function must never disagree with the key.

    **Two different shapes are a difference, never a recursion.** ``canonical`` is
    idempotent on the shapes it produces, so recursing on ``None`` against ``{}``, a
    ``str`` against a ``list``, or a ``dict`` against a ``list`` would call this function
    again with the same two values and never end. Such a change comes from an ordinary
    config edit (``chat_template_kwargs: {reasoning: null}`` becoming
    ``{reasoning: {effort: high}}``), and what the user needs is the name of the field
    that changed. Recursion therefore requires both sides to be the *same* container
    type, the only case where recursing establishes anything.

    A value the serialiser refuses (which a record read off disk cannot contain, but a
    caller could pass) is reported as a difference rather than crashing *this*
    comparison.
    """
    if isinstance(recorded, list) and isinstance(current, list):
        return len(recorded) == len(current) and all(
            _same_leaf(left, right) for left, right in zip(recorded, current, strict=True)
        )
    if isinstance(recorded, dict) and isinstance(current, dict):
        return set(recorded) == set(current) and all(_same_leaf(recorded[key], current[key]) for key in recorded)
    try:
        left_canonical = canonical(recorded)
        right_canonical = canonical(current)
    except UnhashableValueError:
        return False
    # ``canonical`` turns a model or a tuple into a container, so a model against a dict
    # still has to recurse -- but only when both sides landed on the *same* container
    # type, and only when something was actually converted. Recursing when only one side
    # is a container re-enters with values ``canonical`` has already reduced to
    # themselves, which is a ``RecursionError`` rather than an answer: that is how a
    # ``chat_template_kwargs: null`` -> ``{}`` edit crashed the identity check instead of
    # reporting a difference.
    left_is_list, right_is_list = isinstance(left_canonical, list), isinstance(right_canonical, list)
    left_is_dict, right_is_dict = isinstance(left_canonical, dict), isinstance(right_canonical, dict)
    if (left_is_list and right_is_list) or (left_is_dict and right_is_dict):
        if left_canonical is recorded and right_canonical is current:
            # Nothing was converted, so recursing would ask the same question again. The
            # branches above already handle list-vs-list and dict-vs-dict, so this is a
            # guard against a future ``canonical`` that returns its input for a new type.
            return left_canonical == right_canonical
        return _same_leaf(left_canonical, right_canonical)
    if type(left_canonical) is not type(right_canonical):
        # ``1`` vs ``1.0`` and ``True`` vs ``1`` are equal in Python but different payloads
        # to ``canonical_json``, therefore different runs -- this check is the only thing
        # that keeps them apart. Shape changes (a container against a scalar, a dict
        # against a list, ``None`` against ``{}``) are rejected here too, but not only
        # here: Python already says ``None != {}``, so the ``==`` below rejects them as well.
        return False
    return left_canonical == right_canonical


__all__ = [
    "FieldRole",
    "IdentityDeclarationError",
    "combine_digests",
    "check_declarations",
    "declared_roles",
    "hash_payload",
    "hash_strings",
    "hash_text",
    "identity_differences",
    "identity_payload",
    "short",
]
