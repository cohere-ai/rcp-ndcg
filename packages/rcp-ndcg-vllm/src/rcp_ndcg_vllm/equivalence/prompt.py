"""Stage 1: the served prompt's token ids must equal the reference's, with zero tolerance.

The served prompt of a rerank recipe is its ``serve.chat_template`` rendered with jinja2 exactly as the engine
renders chat templates: a sandboxed environment with ``trim_blocks`` and ``lstrip_blocks`` on, a stripped template
trailing newline, and undefined variables refused.  The context is always ``{query, document, instruction}``,
where the values depend on ``client.instruction``:

- ``none``: the bare query, no instruction anywhere;
- ``field``: the raw query, and the recipe's ``default_instruction`` as the ``instruction`` variable (the client
  sends it as the request's ``instruction`` field);
- ``fold``: the query with the instruction folded in (see :func:`~rcp_ndcg_vllm.equivalence.client.fold_instruction`),
  and an empty ``instruction`` variable, so the text appears exactly once.

For an embedding or late-interaction recipe there is no template: the served prompt is the client-side prompt
(``client.doc_prompt`` + the document), and the reference's ``render("", document, None)`` must produce its token
ids.  Both sides are tokenised with the recipe's tokenizer (``client.tokenizer``, ``repo@revision``) without
special tokens, since the rendered text already carries them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..errors import HarnessError
from ..recipe import Recipe
from .client import fold_instruction

__all__ = [
    "TokenizerAdapter",
    "TransformersTokenizerAdapter",
    "load_tokenizer",
    "render_template",
    "served_prompt_text",
]  # noqa: E501

_UNDEFINED_MARK = object()


class TokenizerAdapter(ABC):
    """The tokenizer operations stage 1 needs: ids, token strings, special-token resolution and cutting."""

    @abstractmethod
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids of ``text``; specials only when the recipe's request form applies them."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """The token string of one id, for the first-mismatch report."""

    def special_tokens(self) -> dict[str, int]:
        """The tokenizer's added/special tokens by name (``{"im_end": 151645, ...}``); names as declared."""
        return {}

    def truncate(self, text: str, max_tokens: int) -> str:
        """The text of the first ``max_tokens`` tokens of ``text`` (a right token-boundary cut)."""
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return text
        return self.decode(ids[:max_tokens])

    def decode(self, token_ids: list[int]) -> str:
        """The text of a kept prefix of ids (used to re-attach cut spans into a text prompt)."""
        return " ".join(self.id_to_token(token_id) for token_id in token_ids)


class TransformersTokenizerAdapter(TokenizerAdapter):
    """The adapter around a transformers tokenizer (the recipe's ``client.tokenizer`` on a GPU host)."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids of ``text``, with the tokenizer's specials only when the form applies them."""
        return list(self._tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    def id_to_token(self, token_id: int) -> str:
        """The token string of one id."""
        return str(self._tokenizer.convert_ids_to_tokens(int(token_id)))

    def special_tokens(self) -> dict[str, int]:
        """The tokenizer's added tokens, keyed by their declared name (brackets stripped for lookup)."""
        return {name: int(token_id) for name, token_id in self._tokenizer.get_added_vocab().items()}

    def decode(self, token_ids: list[int]) -> str:
        """The text of a kept prefix of ids."""
        return str(self._tokenizer.decode(list(token_ids)))


def load_tokenizer(recipe: Recipe) -> TokenizerAdapter:
    """Load the recipe's tokenizer (``client.tokenizer``, ``repo@revision``) with transformers.

    Raises :class:`~rcp_ndcg_vllm.errors.HarnessError` with the install hint when transformers is absent (the
    ``reference`` extra carries it).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised only without the reference extra
        raise HarnessError(
            f"stage 1 needs the recipe's tokenizer ({recipe.client.tokenizer}): install rcp-ndcg-vllm[reference]"
        ) from error
    repo, _, revision = recipe.client.tokenizer.partition("@")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    return TransformersTokenizerAdapter(tokenizer)


def render_template(template_text: str, context: dict[str, str]) -> str:
    """Render a recipe chat template the way the engine renders it (the contract of the module docstring)."""
    try:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:  # pragma: no cover - exercised only without jinja2
        raise HarnessError(
            "stage 1 renders the recipe's chat template with jinja2: install rcp-ndcg-vllm[reference] or [test]"
        ) from error
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    return environment.from_string(template_text).render(**context)


def context(recipe: Recipe, query: str, document: str) -> dict[str, str]:
    """The template context for one (query, document) pair, by the recipe's ``client.instruction`` mode."""
    assert recipe.client.instruction is not None  # a rerank recipe always sets it
    if recipe.client.instruction == "fold":
        folded = fold_instruction(recipe.client.default_instruction, query)
        return {"query": folded, "document": document, "instruction": ""}
    if recipe.client.instruction == "field":
        return {"query": query, "document": document, "instruction": recipe.client.default_instruction or ""}
    return {"query": query, "document": document, "instruction": ""}


def render_template_for(recipe: Recipe, query: str, document: str) -> str:
    """The served prompt text for one pair of a templated recipe (rerank)."""
    if recipe.serve.chat_template is None:
        raise HarnessError(f"recipe {recipe.id} has no serve.chat_template to render")
    directory = recipe._dir
    if directory is None:
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory; use load_recipe")
    return render_template(
        (directory / recipe.serve.chat_template).read_text(encoding="utf-8"), context(recipe, query, document)
    )


def served_prompt_text(recipe: Recipe, query: str, document: str, tokenizer: Any, *, shape: str | None = None) -> str:
    """The text the engine tokenises for one (query, document) pair.

    With ``client.template`` the declared shape assembles the prompt (query, document or pair, by role) with
    the anchor-preserving cuts; with ``serve.chat_template`` the recipe's template file renders (the jinja2
    environment of this module, the engine's settings); otherwise the client-side prompt composition
    (``doc_prompt`` + document) stands.  The instruction mode applies to the template and chat paths (``fold``
    into the query; ``field``/``system``/``none`` leave it to the request); the prompted path has none.
    """
    if recipe.client.template is not None:
        shape = shape or ("pair" if recipe.role == "rerank" else "document")
        if recipe.client.instruction == "fold":
            query = fold_instruction(recipe.client.default_instruction, query)
        return assemble_shape_text(recipe, tokenizer, shape, {"query": query, "document": document})
    if recipe.role == "rerank":
        return render_template_for(recipe, query, document)
    return recipe.client.doc_prompt + document


# ``render_template`` is the primitive; ``render_template_for`` is the recipe-aware wrapper.


# ---------------------------------------------------------------------------
# Declared shapes: the template block as data, with anchor-preserving cuts.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Declared shapes as data: assembly with anchor-preserving cuts, and the checks.
#
# The contract (the design's binding finding): a cut applies to content spans only, inside a budget computed
# after reserving every fixed template segment the model reads its output from (the anchors), and the template
# is re-attached after the cut.  A boundary-merge caveat applies to subword tokenizers (the fixtures'
# whitespace tokenizer is exact); stage 1 compares against the reference, so divergence shows up there.
# ---------------------------------------------------------------------------


def _fixed_ids(recipe: Recipe, tokenizer: Any, shape: str) -> list[list[int]]:
    """The resolved ids of every fixed segment of one declared shape, in order."""
    assert recipe.client.template is not None
    segments = getattr(recipe.client.template, shape) or []
    resolved: list[list[int]] = []
    for position, segment in enumerate(segments):
        if segment.content is not None:
            continue
        if segment.special is not None:
            table = tokenizer.special_tokens()
            candidates = (segment.special, f"<|{segment.special}|>", f"<{segment.special}>")
            match = next((table[candidate] for candidate in candidates if candidate in table), None)
            if match is None:
                raise HarnessError(
                    f"recipe {recipe.id}: template.{shape} segment {position} names special token "
                    f"{segment.special!r}, which the tokenizer does not define (known: {sorted(table)[:8]})"
                )
            resolved.append([match])
        elif segment.text is not None:
            ids = tokenizer.encode(segment.text, add_special_tokens=False)
            if not ids:
                raise HarnessError(
                    f"recipe {recipe.id}: template.{shape} segment {position} ({segment.text[:24]!r}) resolves "
                    "to no token ids; a fixed segment must contribute at least one id"
                )
            resolved.append(ids)
        else:
            resolved.append([int(value) for value in segment.ids or []])
    if any(not ids for ids in resolved):
        raise HarnessError(f"recipe {recipe.id}: template.{shape} has a fixed segment that resolves to no ids")
    return resolved


def _specials_ids(recipe: Recipe, tokenizer: Any) -> list[int]:
    """The ids the tokenizer's post-processor appends when the recipe pins ``add_special_tokens``."""
    if recipe.client.add_special_tokens:
        return tokenizer.encode("", add_special_tokens=True)
    return []


def _segment_text(recipe: Recipe, tokenizer: Any, segment: Any, shape: str, position: int) -> str:
    """The verbatim text of one fixed segment (special tokens come from the tokenizer, never from the recipe)."""
    if segment.text is not None:
        return segment.text
    if segment.special is not None:
        table = tokenizer.special_tokens()
        candidates = (segment.special, f"<|{segment.special}|>", f"<{segment.special}>")
        match = next((candidate for candidate in candidates if candidate in table), None)
        if match is None:
            raise HarnessError(f"recipe {recipe.id}: template.{shape} segment {position}: unknown special token")
        return tokenizer.id_to_token(table[match])
    return tokenizer.decode([int(value) for value in segment.ids or []])


def shape_overhead(recipe: Recipe, tokenizer: Any, shape: str) -> int:
    """The token overhead of one declared shape's fixed segments (measured; the reserved part of the budget)."""
    return sum(len(ids) for ids in _fixed_ids(recipe, tokenizer, shape))


def _content_budgets(recipe: Recipe, tokenizer: Any, shape: str) -> dict[str, int]:
    """The per-span budget split: a pair's query gets ``template.query_max_tokens``, the document the rest."""
    assert recipe.client.template is not None
    remaining = (
        recipe.client.max_tokens - shape_overhead(recipe, tokenizer, shape) - len(_specials_ids(recipe, tokenizer))
    )
    if shape == "pair":
        assert recipe.client.template.query_max_tokens is not None  # the pair-shape validator
        return {
            "query": min(recipe.client.template.query_max_tokens, remaining),
            "document": max(remaining - recipe.client.template.query_max_tokens, 0),
        }
    return {"query": remaining, "document": remaining}


def assemble_shape_text(recipe: Recipe, tokenizer: Any, shape: str, contents: dict[str, str]) -> str:
    """The rendered prompt text of one declared shape, with the content spans cut to their budget.

    Fixed segments contribute verbatim text (special tokens by their tokenizer-resolved string); content spans
    are cut at token boundaries from the right via the tokenizer, then the fixed segments are re-attached
    around them.  This is what a ``request_shape: text`` client sends and what the engine tokenises with the
    recipe's ``add_special_tokens`` flag.
    """
    assert recipe.client.template is not None
    segments = getattr(recipe.client.template, shape) or []
    budgets = _content_budgets(recipe, tokenizer, shape)
    parts: list[str] = []
    for segment in segments:
        if segment.content is not None:
            text = contents.get(segment.content, "")
            if (
                recipe.client.on_overflow == "fail"
                and len(tokenizer.encode(text, add_special_tokens=False)) > budgets[segment.content]
            ):
                raise HarnessError(
                    f"recipe {recipe.id}: {segment.content} exceeds its share of client.max_tokens "
                    f"({budgets[segment.content]} tokens) and client.on_overflow is fail"
                )
            parts.append(tokenizer.truncate(text, budgets[segment.content]))
        else:
            parts.append(_segment_text(recipe, tokenizer, segment, shape, 0))
    return "".join(parts)


def assemble_shape_ids(recipe: Recipe, tokenizer: Any, shape: str, contents: dict[str, str]) -> list[int]:
    """The token ids of one declared shape's rendered prompt, as the client sends it (specials per the recipe)."""
    text = assemble_shape_text(recipe, tokenizer, shape, contents)
    return tokenizer.encode(text, add_special_tokens=bool(recipe.client.add_special_tokens))


def anchor_report(recipe: Recipe, tokenizer: Any, shape: str, ids: list[int]) -> dict[str, Any]:
    """The anchor assertion for one rendered prompt: every declared anchor survives the cut, in position.

    ``anchor: last`` requires the rendered ids to end with the tail fixed segment's ids (plus the tokenizer's
    post-processor end token when the recipe pins ``add_special_tokens``); ``anchor: first`` to start with the
    head's; ``anchor: marker`` to contain every declared marker id; ``mean`` has no positional anchor.
    """
    template = recipe.client.template
    assert template is not None
    segments = getattr(template, shape) or []
    fixed = [segment for segment in segments if segment.content is None]
    failures: list[dict[str, Any]] = []
    if template.anchor == "last":
        if fixed:
            tail = _fixed_ids(recipe, tokenizer, shape)[-1] + _specials_ids(recipe, tokenizer)
        else:
            # A content-final shape pins add_special_tokens: true: the tokenizer's post-processor end token IS
            # the anchor. The specials block (everything the tokenizer appends) must sit at the very tail; a
            # post-processor that also prepends a token makes this assertion fail loudly rather than silently.
            tail = _specials_ids(recipe, tokenizer)
        if not tail or ids[-len(tail) :] != tail:
            failures.append({"check": "tail", "expected_tail_ids": tail, "actual_tail_ids": ids[-len(tail) :]})
    elif template.anchor == "first" and fixed:
        head = _fixed_ids(recipe, tokenizer, shape)[0]
        if ids[: len(head)] != head:
            failures.append({"check": "head", "expected_head_ids": head, "actual_head_ids": ids[: len(head)]})
    elif template.anchor == "marker":
        missing = sorted(set(template.anchor_markers) - set(ids))
        if missing:
            failures.append({"check": "markers", "missing_ids": missing})
    return {
        "anchor": template.anchor,
        "shape": shape,
        "checked_ids": len(ids),
        "passed": not failures,
        "failures": failures,
    }


def template_render_check(recipe: Recipe, tokenizer: Any, query: str, document: str) -> dict[str, Any]:
    """Stage 1's extra proof for a served chat template: the declared shapes must render to the same ids.

    The template file is rendered the way the engine renders it (the jinja2 environment of this module), with
    the recipe's ``client.instruction`` context; the declared shape renders through the template block.  Both
    are tokenised with the same tokenizer and flags and compared exactly.
    """
    shape = "pair" if recipe.role == "rerank" else "document"
    folded_query = (
        fold_instruction(recipe.client.default_instruction, query) if recipe.client.instruction == "fold" else query
    )
    from_ids = assemble_shape_ids(recipe, tokenizer, shape, {"query": folded_query, "document": document})
    template_text = render_template_for(recipe, query, document)  # context() applies the fold
    template_ids = tokenizer.encode(template_text, add_special_tokens=bool(recipe.client.add_special_tokens))
    passed = from_ids == template_ids
    report: dict[str, Any] = {
        "shape": shape,
        "passed": passed,
        "template_ids_len": len(template_ids),
        "shape_ids_len": len(from_ids),
    }
    if not passed:
        report["template_ids_head"] = template_ids[:32]
        report["shape_ids_head"] = from_ids[:32]
    return report
