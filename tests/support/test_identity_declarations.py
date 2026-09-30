"""Adding a field to a config that feeds an identity must force a decision.

Every model that feeds a key declares a role per field, and
:func:`rcp_ndcg.support.identity.check_declarations` refuses both an undeclared
field and a declaration whose field is gone. The roots are enumerated through
their fields, so a nested model added to one of them is covered the moment it is
added.
"""

from __future__ import annotations

from typing import Any, ClassVar, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg.llm import JudgeConfig
from rcp_ndcg.support.identity import (
    FieldRole,
    IdentityDeclarationError,
    check_declarations,
    declared_roles,
    identity_payload,
)


def _annotation_models(annotation: Any) -> list[type[BaseModel]]:
    """Every Pydantic model reachable from a field annotation."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if get_origin(annotation) is None:
        return []
    return [model for argument in get_args(annotation) for model in _annotation_models(argument)]


def identity_models(*roots: type[BaseModel]) -> list[type[BaseModel]]:
    """``roots`` plus every Pydantic model reachable through fields that can reach a key.

    A field declared ``RUNTIME`` never enters the payload, so the models nested under it feed no identity and need
    no declarations of their own.
    """
    seen: dict[type[BaseModel], None] = {}
    queue = list(roots)
    while queue:
        model_cls = queue.pop(0)
        if model_cls in seen:
            continue
        seen[model_cls] = None
        roles = declared_roles(model_cls)
        for name, field in model_cls.model_fields.items():
            if roles.get(name) is not FieldRole.RUNTIME:
                queue.extend(nested for nested in _annotation_models(field.annotation) if nested not in seen)
    return list(seen)


#: Every model that feeds a judgement identity or a step identity.
IDENTITY_ROOTS = (JudgeConfig,)


class _Nested(BaseModel):
    """A declared model to nest inside a root."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"seed": FieldRole.CONTENT}

    seed: int = 0


class _Undeclared(BaseModel):
    """A model without declarations, nested under a runtime field."""

    rate: float = 0.0


class _RuntimeNest(BaseModel):
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"model": FieldRole.CONTENT, "gate": FieldRole.RUNTIME}

    model: str = "m"
    gate: _Undeclared = Field(default_factory=_Undeclared)


def test_models_under_a_runtime_field_feed_no_identity() -> None:
    # A runtime field never enters the payload, so the model under it needs no declarations; walking into it
    # demanded roles for fields no key can see.
    reached = set(identity_models(_RuntimeNest))
    assert _Undeclared not in reached


@pytest.mark.parametrize("model_cls", identity_models(*IDENTITY_ROOTS), ids=lambda cls: cls.__name__)
def test_every_field_of_every_identity_model_is_classified(model_cls: type[BaseModel]) -> None:
    check_declarations(model_cls)


@pytest.mark.parametrize("model_cls", identity_models(*IDENTITY_ROOTS), ids=lambda cls: cls.__name__)
def test_no_declaration_names_a_field_that_is_gone(model_cls: type[BaseModel]) -> None:
    stale = sorted(set(declared_roles(model_cls)) - set(model_cls.model_fields))
    assert stale == [], f"{model_cls.__name__}.IDENTITY_ROLES names absent field(s) {stale}"


class TestAnUndeclaredFieldFails:
    """The guard, mutated: a new field on each kind of model must fail."""

    @staticmethod
    def _subclass_with_new_field(base: type[BaseModel]) -> type[BaseModel]:
        # A subclass is the honest simulation of "somebody added a field": the MRO
        # merge in ``declared_roles`` sees the parent's declarations and the child's
        # new field.
        return type(f"{base.__name__}WithNewKnob", (base,), {"__annotations__": {"new_knob": int}, "new_knob": 7})

    @pytest.mark.parametrize("base", [JudgeConfig, _Nested], ids=lambda cls: cls.__name__)
    def test_a_new_field_is_refused(self, base: type[BaseModel]) -> None:
        with pytest.raises(IdentityDeclarationError, match="undeclared field"):
            check_declarations(self._subclass_with_new_field(base))

    def test_a_new_field_inside_a_nested_model_is_refused(self) -> None:
        # The field is not on the root, so a check that only walked the root's own
        # fields would pass.
        nested = self._subclass_with_new_field(_Nested)

        class Judge(JudgeConfig):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"args": FieldRole.CONTENT}

            args: nested | None = None  # type: ignore[valid-type]

        offenders = [cls for cls in identity_models(Judge) if _is_undeclared(cls)]
        assert nested in offenders

    def test_the_message_names_the_field_and_the_choices(self) -> None:
        with pytest.raises(IdentityDeclarationError) as error:
            check_declarations(self._subclass_with_new_field(JudgeConfig))
        message = str(error.value)
        assert "new_knob" in message
        assert "CONTENT" in message and "RUNTIME" in message

    def test_a_model_with_no_declarations_at_all_is_refused(self) -> None:
        class Undeclared(BaseModel):
            knob: int = 1

        with pytest.raises(IdentityDeclarationError, match="declares no IDENTITY_ROLES"):
            check_declarations(Undeclared)


def _is_undeclared(model_cls: type[BaseModel]) -> bool:
    try:
        check_declarations(model_cls)
    except IdentityDeclarationError:
        return True
    return False


class TestAStaleDeclarationFails:
    def test_declaring_a_field_that_does_not_exist_is_refused(self) -> None:
        class Ghost(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"real": FieldRole.CONTENT, "removed": FieldRole.RUNTIME}

            real: int = 1

        with pytest.raises(IdentityDeclarationError, match="declared but absent"):
            check_declarations(Ghost)


class TestThePayloadFollowsTheDeclarations:
    def test_only_content_fields_reach_the_payload(self) -> None:
        class Cfg(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
                "kept": FieldRole.CONTENT,
                "speed": FieldRole.RUNTIME,
            }

            kept: int = 1
            speed: int = 2

        assert identity_payload(Cfg()) == {"kept": 1}

    def test_none_is_omitted(self) -> None:
        class Cfg(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"maybe": FieldRole.CONTENT}

            maybe: int | None = None

        assert identity_payload(Cfg()) == {}
        assert identity_payload(Cfg(maybe=0)) == {"maybe": 0}

    def test_a_subclass_may_override_an_inherited_role(self) -> None:
        class Base(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"knob": FieldRole.RUNTIME}

            knob: int = 1

        class Child(Base):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"knob": FieldRole.CONTENT}

        assert identity_payload(Base()) == {}
        assert identity_payload(Child()) == {"knob": 1}

    def test_nested_models_are_recursed_by_the_same_rule(self) -> None:
        class Inner(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kept": FieldRole.CONTENT, "speed": FieldRole.RUNTIME}

            kept: int = 1
            speed: int = 2

        class Outer(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"inner": FieldRole.CONTENT}

            inner: Inner = Field(default_factory=Inner)

        # A runtime knob cannot reach the key by being nested.
        assert identity_payload(Outer()) == {"inner": {"kept": 1}}

    def test_an_undeclared_extra_field_is_content(self) -> None:
        # ``extra="allow"`` models forward unknown keys to a loader, so the fail-safe direction is to
        # key on them: a needless re-key costs compute, a missed one costs numbers.
        class Cfg(BaseModel):
            model_config = ConfigDict(extra="allow")
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"known": FieldRole.CONTENT}

            known: int = 1

        assert identity_payload(Cfg.model_validate({"known": 1, "surprise": "x"})) == {"known": 1, "surprise": "x"}

    def test_the_payload_builder_itself_refuses_an_undeclared_field(self) -> None:
        # The guard's *production* call site. Every other test here calls
        # ``check_declarations`` directly, so deleting the call inside
        # ``identity_payload`` left them all green -- and the production path then
        # degraded to a ``KeyError`` from inside the key builder instead of the
        # message this module was written to produce.
        class Base(BaseModel):
            IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"known": FieldRole.CONTENT}

            known: int = 1

        class WithNewKnob(Base):
            new_knob: int = 7

        with pytest.raises(IdentityDeclarationError, match="undeclared field"):
            identity_payload(WithNewKnob())


class TestTheShippedDeclarations:
    """Spot checks that the roles on the real models are the intended ones."""

    def test_judge_transport_is_runtime(self) -> None:
        roles = declared_roles(JudgeConfig)
        for field in (
            "base_url",
            "api_key_env",
            "concurrency",
            "timeout_s",
            "connect_timeout_s",
            "max_retries",
            "max_images",
            "max_videos",
            "wait_on_outage_s",
        ):
            assert roles[field] is FieldRole.RUNTIME, field

    def test_judge_content_is_content(self) -> None:
        roles = declared_roles(JudgeConfig)
        for field in ("model", "revision", "temperature", "max_output_tokens", "context_tokens", "decoding"):
            assert roles[field] is FieldRole.CONTENT, field

    def test_moving_the_endpoint_keeps_the_judge_identity(self) -> None:
        judge = JudgeConfig(base_url="http://a/v1", model="m", revision="r")
        moved = judge.model_copy(update={"base_url": "http://b/v1", "concurrency": 3, "timeout_s": 1})
        assert moved.identity() == judge.identity()
        assert judge.model_copy(update={"temperature": 0.7}).identity() != judge.identity()
