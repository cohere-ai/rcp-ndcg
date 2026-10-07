"""The negative controls (GPU-VALIDATION.md, "The GPU run is also the test suite's audit", item 5).

Six deliberate breakages per recipe -- (a) the chat template removed, (b) an engine-side right cut of the rendered
prompt, (c) ``use_activation`` flipped, (d) the wrong pooling, (e) float32 decoded as float16 and (f) an unpinned
``max_pixels`` -- each derived from the recipe and served through the ordinary gates, which must FAIL it.  A
control that passes is a blocker (:func:`controls_summary`): its gate cannot see that breakage class.

Every breakage is one vLLM v0.31.0 really exhibits, so a control never "fails" for an unrelated reason (an
engine that refuses its argv would fail every gate and prove nothing):

- **recipe variants** (served as their own engine, client and engine agreeing on the variant's id): (a) drops
  ``serve.chat_template``; (c) flips the client's ``use_activation`` (or the pooler config's); (d) flips the
  pooling key the recipe declares (``seq_pooling_type`` or ``pooling_type`` -- vLLM refuses both at once);
  (f) unpins ``max_pixels``/``min_pixels`` from ``serve.mm_processor_kwargs`` (the nested ``images_kwargs`` pin
  or a flat one) -- inapplicable, said why, when the client prepares every image inside the processor family's
  stock range, which an unpinned engine keeps;
- **wire variants** (the recipe's own engine, the request bodies patched on the way out through
  :func:`rcp_ndcg_vllm.equivalence.wire.patched_wire`): (b) adds the request fields vLLM cuts with
  (``truncate_prompt_tokens`` + ``truncation_side: right`` -- a request field, not a serve flag, in v0.31.0);
  (e) asks ``/pooling`` for the other ``embed_dtype`` than the client decodes, so float32 frames are read as
  float16 (or the reverse).

A control that does not apply to a recipe is listed with the reason (never dropped silently).

Public surface:

- :data:`CONTROLS`, :class:`ControlSpec` -- the six controls.
- :func:`control_variants` -- every control of one recipe: its variant or wire patch, or why it does not apply.
- :func:`controls_summary` -- the wave's control report; a passing control is a blocker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CONTROLS", "ControlSpec", "control_variants", "controls_summary", "right_cut_tokens"]

_ROUTES = {"embed": "/embeddings", "multi_vector": "/pooling", "rerank": "/rerank"}


@dataclass(frozen=True)
class ControlSpec:
    """One deliberate breakage.

    Attributes:
        letter: The control's ``(a)``-``(f)`` label from GPU-VALIDATION.md item 5.
        name: The variant's infix (a recipe variant's id becomes ``<recipe>.<name>``).
        description: What breaks and which gate must catch it.
        derive: ``derive(recipe) -> (kind, change, reason)``: ``kind`` is ``recipe`` (``change``: the
            ``serve``/``client`` blocks), ``wire`` (``change``: the request-body fields per route path suffix)
            or ``None`` with the ``reason`` the control does not apply.
    """

    letter: str
    name: str
    description: str
    derive: Callable[[Any], tuple[str | None, dict[str, Any], str]]


def right_cut_tokens(recipe: Any) -> int:
    """The engine-side cut of control (b): a quarter of the client's budget (at least 4 tokens), so every pairs
    row longer than that -- the length ladder's rows always are -- loses its tail, anchor included."""
    return max(4, int(recipe.client.max_tokens or 64) // 4)


def _blocks(recipe: Any) -> dict[str, Any]:
    return {"serve": recipe.serve.model_dump(), "client": recipe.client.model_dump()}


def _template_removed(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    if recipe.serve.chat_template is None:
        return None, {}, "the recipe serves no template file (the client renders its prompts): nothing to remove"
    blocks = _blocks(recipe)
    blocks["serve"]["chat_template"] = None
    return "recipe", blocks, ""


def _right_cut(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    fields = {"truncate_prompt_tokens": right_cut_tokens(recipe), "truncation_side": "right"}
    return "wire", {_ROUTES[recipe.role]: fields}, ""


def _flip_use_activation(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    if recipe.role != "rerank":
        return None, {}, "an embedder's vectors pass no activation: use_activation shapes rerank scores only"
    blocks = _blocks(recipe)
    if blocks["client"].get("use_activation") is not None:
        blocks["client"]["use_activation"] = not blocks["client"]["use_activation"]
        return "recipe", blocks, ""
    pooler = dict(blocks["serve"].get("pooler_config") or {})
    if pooler.get("use_activation") is not None:
        pooler["use_activation"] = not pooler["use_activation"]
        blocks["serve"]["pooler_config"] = pooler
        return "recipe", blocks, ""
    return None, {}, "neither the client nor the pooler config declares use_activation"


def _wrong_pooling(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    if recipe.role == "multi_vector":
        return (
            None,
            {},
            (
                "token-level pooling has no wrong-but-servable alternative in vLLM v0.31.0 (STEP needs a step tag); "
                "a sequence pooler would change the route's contract, not the pooling"
            ),
        )
    blocks = _blocks(recipe)
    pooler = dict(blocks["serve"].get("pooler_config") or {})
    key = "seq_pooling_type" if "seq_pooling_type" in pooler else "pooling_type" if "pooling_type" in pooler else None
    if key is None:
        # Undeclared: the checkpoint's default applies.  MEAN is set; were MEAN the default, the control would
        # pass and the summary would flag it -- declaring the pooling in the recipe is then the fix.
        pooler["seq_pooling_type"] = "MEAN"
    else:
        pooler[key] = "LAST" if str(pooler[key]).upper() == "MEAN" else "MEAN"
    blocks["serve"]["pooler_config"] = pooler
    return "recipe", blocks, ""


def _float32_read_as_float16(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    if recipe.role != "multi_vector":
        return (
            None,
            {},
            ("only /pooling frames carry an embed_dtype; the dense /v1/embeddings float frames name their numbers"),
        )
    declared = str(getattr(recipe.client, "embed_dtype", "float16") or "float16")
    other = "float32" if declared == "float16" else "float16"
    return "wire", {"/pooling": {"embed_dtype": other}}, ""


_PIXEL_KEYS = ("min_pixels", "max_pixels")


def _unpinned_max_pixels(recipe: Any) -> tuple[str | None, dict[str, Any], str]:
    if "image" not in recipe.input:
        return None, {}, "the recipe is text-only: no image is resized"
    blocks = _blocks(recipe)
    kwargs = dict(blocks["serve"].get("mm_processor_kwargs") or {})
    scoped = dict(kwargs.get("images_kwargs") or {}) if isinstance(kwargs.get("images_kwargs"), dict) else {}
    if not any(key in kwargs or key in scoped for key in _PIXEL_KEYS):
        return None, {}, "the recipe pins no max_pixels or min_pixels to unpin (a finding of its own for a VL recipe)"
    policy = getattr(recipe.client, "image_policy", None)
    processor = getattr(recipe.client, "image_processor", None)
    if policy is not None and processor is not None and policy.max_px is not None and not policy.pinned:
        from rcp_ndcg.data.resolution import PROCESSORS

        geometry = PROCESSORS[processor]
        return (
            None,
            {},
            f"the client prepares every image inside the {processor} stock range ({policy.min_px}-{policy.max_px} px "
            f"within {geometry.min_pixels}-{geometry.max_pixels} px), which an unpinned engine keeps: unpinning "
            "changes nothing the engine reads (the pin binds an image the client sends unprepared, and this "
            "client prepares every image)",
        )
    for key in _PIXEL_KEYS:  # the ONE pin shape is nested images_kwargs; a flat pin is unpinned the same way
        kwargs.pop(key, None)
        scoped.pop(key, None)
    if scoped:
        kwargs["images_kwargs"] = scoped
    else:
        kwargs.pop("images_kwargs", None)
    blocks["serve"]["mm_processor_kwargs"] = kwargs
    return "recipe", blocks, ""


CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec(
        "(a)",
        "template-removed",
        "the served chat template file is dropped (the engine scores the unframed spans): the score gates must "
        "catch every pair's missing frame",
        _template_removed,
    ),
    ControlSpec(
        "(b)",
        "right-cut",
        "an engine-side right cut of the rendered prompt (truncate_prompt_tokens with truncation_side right on "
        "every request): the score and vector gates must catch the dropped tail and its anchor",
        _right_cut,
    ),
    ControlSpec(
        "(c)",
        "use-activation-flipped",
        "use_activation is flipped (a raw logit where a probability is declared, or the reverse): the score-scale "
        "gate must catch the shift",
        _flip_use_activation,
    ),
    ControlSpec(
        "(d)",
        "wrong-pooling",
        "the declared pooling is swapped (MEAN where LAST or CLS is declared, LAST where MEAN is): the vector and "
        "score gates must catch the wrong pooled position",
        _wrong_pooling,
    ),
    ControlSpec(
        "(e)",
        "float16-read",
        "the /pooling request asks for the other embed_dtype than the client decodes (float32 frames read as "
        "float16, or the reverse): the vector gate must catch the garbled vectors",
        _float32_read_as_float16,
    ),
    ControlSpec(
        "(f)",
        "unpinned-max-pixels",
        "max_pixels/min_pixels are unpinned from mm_processor_kwargs (nested images_kwargs or flat): the engine "
        "re-resizes a prepared image outside its stock budget, and the media stage's engine count must catch the "
        "drift",
        _unpinned_max_pixels,
    ),
)


def control_variants(recipe: Any) -> list[dict[str, Any]]:
    """Every control of ``recipe``: the broken variant or wire patch it is served with, or why it does not apply.

    Input: a loaded :class:`~rcp_ndcg_vllm.recipe.Recipe` (loaded from its directory: a variant keeps its
    template and reference files).  Output: one dict per control, in (a)-(f) order: ``{"control", "name",
    "description", "kind"}`` plus, for ``kind: "recipe"``, ``recipe`` (the variant: id ``<recipe>.<name>``, which
    is also its served model name and its client's ``model``, so client and engine agree), for ``kind: "wire"``,
    ``recipe`` (the recipe itself) and ``wire_patch`` (``{route suffix: fields}``), and for an inapplicable
    control ``kind: None`` and ``reason``.
    """
    out: list[dict[str, Any]] = []
    for spec in CONTROLS:
        kind, change, reason = spec.derive(recipe)
        entry: dict[str, Any] = {
            "control": spec.letter,
            "name": spec.name,
            "description": spec.description,
            "kind": kind,
        }
        if kind is None:
            out.append({**entry, "reason": reason})
            continue
        if kind == "wire":
            out.append({**entry, "recipe": recipe, "wire_patch": change})
            continue
        variant_id = f"{recipe.id}.{spec.name}"
        variant = recipe.model_copy(
            update={
                "id": variant_id,
                "serve": recipe.serve.__class__.model_validate(change["serve"]),
                "client": recipe.client.__class__.model_validate({**change["client"], "model": variant_id}),
            }
        )
        out.append({**entry, "recipe": variant})
    return out


def controls_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The wave's control report: every applicable control must FAIL its gates; a pass is a blocker.

    Inputs: one row per control as the wave recorded it (``{"control", "name", "equivalence": <the gate
    document with a boolean "passed">}``, or ``{"control", "name", "reason"}`` for an inapplicable one).
    Output: ``{"passed", "blockers", "rows", "inapplicable"}``: ``passed`` means every applicable control was
    caught (its gates reported failure); each blocker names the control whose breakage went unseen.
    """
    blockers: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    inapplicable: list[dict[str, Any]] = []
    for row in rows:
        if row.get("equivalence") is None:
            inapplicable.append({"control": row.get("control"), "name": row.get("name"), "reason": row.get("reason")})
            continue
        gates = row["equivalence"]
        caught = gates.get("passed") is False
        details.append(
            {
                "control": row.get("control"),
                "name": row.get("name"),
                "caught": caught,
                "gates_passed": gates.get("passed"),
            }
        )
        if not caught:
            blockers.append(
                {
                    "control": row.get("control"),
                    "name": row.get("name"),
                    "reason": "the broken variant PASSED the gates (a no-op or blind gate): the control is a "
                    "blocker until a gate catches this breakage class",
                }
            )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "rows": details,
        "inapplicable": inapplicable,
        "referent": "every applicable negative control must fail its gates; a passing control means the gates "
        "cannot see its breakage (GPU-VALIDATION.md item 5)",
    }
