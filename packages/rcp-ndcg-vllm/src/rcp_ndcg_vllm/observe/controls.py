"""The negative controls (GPU-VALIDATION.md, "The GPU run is also the test suite's audit", item 5).

Six broken variants of every recipe family -- the chat template removed, an engine-side right cut of
the rendered prompt, ``use_activation`` flipped, the wrong pooling, ``float32`` decoded as ``float16``
and an unpinned ``max_pixels`` -- each generated from its family's recipe and served as its own
(slightly broken) recipe.  Every control must FAIL the ordinary gates; a control that passes is a
blocker (:func:`controls_summary` flags it), because it proves the gates cannot see that breakage
class.  The mutation test (a gate made a no-op) shows the summary flagging exactly that.

Public surface:

- :data:`CONTROLS`, :class:`ControlSpec` -- the six controls and which families they apply to.
- :func:`control_variants` -- one broken recipe per applicable control, derived from a recipe.
- :func:`controls_summary` -- the wave's control report; a passing control is a blocker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["CONTROLS", "ControlSpec", "control_variants", "controls_summary"]

Family = Literal["dense", "multi_vector", "pointwise_rerank", "listwise_rerank", "vl_embed", "vl_rerank"]

FAMILIES: dict[str, tuple[str, ...]] = {
    "embed": ("dense", "vl_embed", "multi_vector"),
    "multi_vector": ("multi_vector",),
    "rerank": ("pointwise_rerank", "listwise_rerank", "vl_reranker"),
}


@dataclass(frozen=True)
class ControlSpec:
    """One deliberate breakage.

    Attributes:
        letter: The control's ``(a)``-``(f)`` label from GPU-VALIDATION.md item 5.
        name: The variant's infix (its recipe id becomes ``<recipe>.<name>``).
        description: What breaks and which gate must catch it.
        applies: Family kinds (``text`` / ``image``) the control applies to.
        mutate: The mutation, applied to the loaded recipe's ``serve``/``client`` blocks.
    """

    letter: str
    name: str
    description: str
    applies: tuple[str, ...]
    mutate: Callable[[dict[str, Any]], dict[str, Any]]


def _drop_template(blocks: dict[str, Any]) -> dict[str, Any]:
    return {**blocks, "serve": {**blocks["serve"], "chat_template": None}}


def _right_cut(blocks: dict[str, Any]) -> dict[str, Any]:
    extra = list(blocks["serve"].get("extra_args") or []) + ["--truncate-prompt-tokens", "32"]
    return {**blocks, "serve": {**blocks["serve"], "extra_args": extra}}


def _flip_use_activation(blocks: dict[str, Any]) -> dict[str, Any]:
    client = dict(blocks["client"])
    if "use_activation" in client:
        client["use_activation"] = not client["use_activation"]
    return {**blocks, "client": client}


def _wrong_pooling(blocks: dict[str, Any]) -> dict[str, Any]:
    pooler = dict(blocks["serve"].get("pooler_config") or {})
    pooler["pooling_type"] = "MEAN" if pooler.get("pooling_type") != "MEAN" else "LAST"
    return {**blocks, "serve": {**blocks["serve"], "pooler_config": pooler}}


def _float32_read_as_float16(blocks: dict[str, Any]) -> dict[str, Any]:
    client = dict(blocks["client"])
    if "embed_dtype" in client:
        client["embed_dtype"] = "float16" if client["embed_dtype"] != "float16" else "float32"
    return {**blocks, "client": client}


def _unpinned_max_pixels(blocks: dict[str, Any]) -> dict[str, Any]:
    mm = dict(blocks["serve"].get("mm_processor_kwargs") or {})
    mm.pop("max_pixels", None)
    mm.pop("min_pixels", None)
    return {**blocks, "serve": {**blocks["serve"], "mm_processor_kwargs": mm}}


CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec(
        "(a)",
        "template-removed",
        "the served chat template file is dropped (vLLM concatenates the spans with no frame): the "
        "anchor and render gates must catch every prompt's missing frame",
        ("text", "image"),
        _drop_template,
    ),
    ControlSpec(
        "(b)",
        "right-cut",
        "an engine-side right cut of the rendered prompt (--truncate-prompt-tokens 32): the anchor "
        "audit and the score/vector gates must catch the dropped anchor",
        ("text", "image"),
        _right_cut,
    ),
    ControlSpec(
        "(c)",
        "use-activation-flipped",
        "use_activation is flipped (sigmoid on a raw logit or the reverse): the probability-scale "
        "gate must catch the score shift",
        ("text", "image"),
        _flip_use_activation,
    ),
    ControlSpec(
        "(d)",
        "wrong-pooling",
        "the pooling is replaced (MEAN where LAST is declared): the vector and score gates must "
        "catch the wrong pooled position",
        ("text", "image"),
        _wrong_pooling,
    ),
    ControlSpec(
        "(e)",
        "float16-read",
        "float32 vectors are decoded as float16 (embed_dtype flipped on the transfer): the vector "
        "gate must catch the precision change",
        ("text", "image"),
        _float32_read_as_float16,
    ),
    ControlSpec(
        "(f)",
        "unpinned-max-pixels",
        "max_pixels is unpinned from mm_processor_kwargs (image tokens per page move): the media "
        "token counts and the vector gates must catch the drift",
        ("image",),
        _unpinned_max_pixels,
    ),
)


def control_variants(recipe: Any) -> list[dict[str, Any]]:
    """Every applicable broken variant of ``recipe`` (from each control's recipe variant).

    Input: a loaded :class:`~rcp_ndcg_vllm.recipe.Recipe`.  Output: one dict per applicable control:
    ``{"control": <letter>, "name": ..., "recipe": <the mutated recipe>, "description": ...}``, the
    mutated recipe's id being ``<recipe.id>.<name>`` (the wave serves it like any recipe, so the
    ordinary gates run over it -- and MUST fail it).
    """
    from ..recipe import load_recipe

    blocks = {
        "serve": recipe.serve.model_dump(),
        "client": recipe.client.model_dump(),
    }
    kinds = ("image",) if "image" in recipe.input else ("text",)
    out: list[dict[str, Any]] = []
    for spec in CONTROLS:
        if not set(spec.applies) & set(kinds):
            continue
        mutated = spec.mutate(blocks)
        if mutated == blocks:
            continue  # the recipe declares nothing this control breaks (its target field is absent)
        directory = recipe._dir
        variant = load_recipe(directory) if directory is not None else recipe
        variant = variant.model_copy(
            update={
                "id": f"{recipe.id}.{spec.name}",
                "serve": recipe.serve.__class__.model_validate(mutated["serve"]),
                "client": recipe.client.__class__.model_validate(
                    {**mutated["client"], "model": recipe.id, "revision": recipe.revision}
                ),
            }
        )
        out.append(
            {
                "control": spec.letter,
                "name": spec.name,
                "recipe": variant,
                "description": spec.description,
            }
        )
    return out


def controls_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The wave's control report: every control must FAIL its gates; a pass is a blocker.

    Inputs: one row per control variant as the wave recorded it (``{"recipe": ..., "control": ...,
    "equivalence": <the stage report>}``, any gate document with a boolean ``passed``).  Output:
    ``{"passed": bool, "blockers": [...], "rows": [...]}`` -- ``passed`` means every control was
    CAUGHT (its gates reported failure); each blocker names the control whose breakage went unseen.
    """
    blockers: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for row in rows:
        gates = row.get("equivalence") or {}
        caught = gates.get("passed") is False
        entry = {
            "control": row.get("control"),
            "name": row.get("name"),
            "caught": caught,
            "gates_passed": gates.get("passed"),
        }
        details.append(entry)
        if not caught:
            blockers.append(
                {
                    "control": row.get("control"),
                    "name": row.get("name"),
                    "reason": "the broken variant PASSED the gates (a no-op or blind gate): the control is a "
                    "blocker until the gate catches this breakage class",
                }
            )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "rows": details,
        "referent": "every negative control must fail its gates; a passing control means the gates cannot "
        "see its breakage (GPU-VALIDATION.md item 5)",
    }
