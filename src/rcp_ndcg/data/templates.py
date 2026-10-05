"""The request template as data: what every served role's prompt is made of, declared, never coded.

A role client sends text to a model that reads it through a fixed frame: a system turn, instruction
prefixes, role markers, the end-of-turn marker the score is pooled from. That frame is data here --
:class:`TemplateSpec`, per request shape (``query``, ``document``, ``pair``) an ordered list of
:class:`Segment` objects, each a fixed piece of frame text or a content span:

* a **fixed** segment is template text; a special token inside it is written by name --
  ``{special:im_end}`` -- and resolved at render time from the tokenizer's added tokens. Specials are
  never typed literally in code, tests or docs: a literal would end an agent's turn or drift from the
  tokenizer's own form;
* a **content** segment names a span (``query``, ``document`` or ``instruction``) that
  :func:`rcp_ndcg.data.preprocess.fit` fills and, when the budget says so, cuts.

The template also declares the model's **anchor** (:attr:`TemplateSpec.anchor`): the fixed position the
model reads its output from (``last``, ``first``, ``mean`` or a ``marker`` id). The anchor is why the
budget reserves every fixed token: a cut of the rendered prompt drops exactly these tokens, and a model
pooled from the last token then reads an arbitrary interior one -- the defect class the anchors exist to
prevent. ``fit`` renders, measures, cuts spans only, and re-attaches the frame.

One more declaration rides with the template: ``add_special_tokens`` per shape -- what the engine does
to the rendered string for that route (vLLM's pooling and scoring routes add the tokenizer's
post-processor tokens by default; the chat-embed form does not). The budget reserves those tokens too.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from rcp_ndcg.errors import ConfigError

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

#: The request shapes a template can frame: ``query`` and ``document`` for an embedder's two sides,
#: ``pair`` for a reranker's request (which orders query and document per model -- some rerankers put
#: the document first, and then the query block is an anchor).
RequestShape = Literal["query", "document", "pair"]

#: How the model reads its output: the pooled/scored position, as data. ``last`` -- the trailing fixed
#: segment (or, when ``add_special_tokens`` declares it, the post-processor's appended token);
#: ``first`` -- the leading fixed segment; ``mean`` -- over the content, with every fixed token still
#: reserved; ``marker`` -- at a named special token (``anchor_markers``).
AnchorKind = Literal["last", "first", "mean", "marker"]

#: What a content span carries: one of the request's roles. ``instruction`` is the run-level task text
#: (declared once per run, never cut).
ContentSpan = Literal["query", "document", "instruction"]

SHAPES: tuple[RequestShape, ...] = ("query", "document", "pair")
"""The request shapes a :class:`TemplateSpec` declares, in canonical order."""

_CONTENT_OF_SHAPE: dict[str, frozenset[str]] = {
    "query": frozenset({"query", "instruction"}),
    "document": frozenset({"document", "instruction"}),
    "pair": frozenset({"query", "document", "instruction"}),
}

_SPECIAL = re.compile(r"\{special:([^{}]+)\}")
"""How a fixed segment writes a special token: ``{special:<name>}``, resolved from the tokenizer's
added tokens (both the bare name and the ``<|...|>`` literal form are accepted)."""


class Segment(BaseModel):
    """One segment of a rendered request shape: a piece of fixed frame text or a content span.

    Exactly one of the two is set. A fixed segment writes its special tokens by name
    (``{special:<name>}``); a content span names the role whose text fills it
    (:data:`ContentSpan`).

    Attributes:
        fixed: The frame text, with specials written ``{special:<name>}``.
        content: The role whose text fills the span.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, Any]] = {"fixed": "content", "content": "content"}

    fixed: str | None = None
    content: ContentSpan | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Segment:
        if (self.fixed is None) == (self.content is None):
            raise ValueError(
                "a segment is exactly one of {fixed: <frame text>} or {content: <query|document|instruction>}"
            )
        return self

    def render(self, tokenizer: TextTokenizer, *, query: str = "", document: str = "", instruction: str = "") -> str:
        """The segment's text: the fixed frame with its specials resolved, or the role's text."""
        if self.fixed is not None:
            return _resolve_specials(self.fixed, tokenizer)
        fill = {"query": query, "document": document, "instruction": instruction}[self.content or ""]
        return fill


class TemplateSpec(BaseModel):
    """The frame of every request a role sends, declared as data.

    Per request shape, an ordered list of segments; at render time the fixed segments are joined with
    the content spans filled from the call's texts. The template is content: two configs whose
    templates differ never share an identity (its canonical JSON is what the identity hashes).

    Attributes:
        query: The ``query`` shape's segments (an embedder's query side).
        document: The ``document`` shape's segments (an embedder's document side).
        pair: The ``pair`` shape's segments (a reranker's request; its order is the model's).
        anchor: The position the model reads its output from (``last`` by default: the last-token
            poolers, the pointwise rerankers scored at the last position). Declared so the budget
            knows what must survive and the golden-render tests what to assert.
        anchor_markers: For ``anchor: marker``: the special tokens' *names* (resolved like
            ``{special:...}``), e.g. a listwise reranker's per-passage markers.
        add_special_tokens: What the engine does to the rendered string per shape -- ``True`` (the
            default) when the route adds the tokenizer's post-processor tokens (vLLM's pooling and
            scoring routes), ``False`` where it adds none (the chat-embed form). A bool for every
            shape, or a mapping shape -> bool naming every declared shape. The budget reserves those
            tokens: measured on the empty render, they are part of the fixed overhead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, Any]] = {
        "query": "content",
        "document": "content",
        "pair": "content",
        "anchor": "content",
        "anchor_markers": "content",
        "add_special_tokens": "content",
    }

    query: tuple[Segment, ...] | None = None
    document: tuple[Segment, ...] | None = None
    pair: tuple[Segment, ...] | None = None
    anchor: AnchorKind = "last"
    anchor_markers: tuple[str, ...] = ()
    add_special_tokens: bool | dict[str, bool] = True

    @model_validator(mode="after")
    def _shapes_are_complete(self) -> TemplateSpec:
        declared: list[RequestShape] = [shape for shape in SHAPES if getattr(self, shape) is not None]
        if not declared:
            raise ValueError("a template declares at least one shape (query, document or pair)")
        if isinstance(self.add_special_tokens, dict):
            stale = sorted(set(self.add_special_tokens) - set(declared))
            missing = sorted(set(declared) - set(self.add_special_tokens))
            if stale or missing:
                raise ValueError(
                    f"add_special_tokens must name every declared shape {declared}: "
                    + (f"unknown {stale}; " if stale else "")
                    + (f"missing {missing}" if missing else "")
                )
        for shape in declared:
            segments: tuple[Segment, ...] = getattr(self, shape)
            allowed = _CONTENT_OF_SHAPE[shape]
            contents = {segment.content for segment in segments if segment.content is not None}
            if not contents:
                raise ValueError(f"the {shape!r} shape declares no content segment: nothing would be filled in")
            outside = contents - allowed
            if outside:
                raise ValueError(f"the {shape!r} shape cannot take the content span(s) {sorted(outside)}")
            if self.anchor == "last":
                ends_fixed = segments[-1].fixed is not None
                if not ends_fixed and not self.adds_special_tokens(shape):
                    raise ValueError(
                        f"an 'anchor: last' shape must end with a fixed segment (the anchor the model reads) or "
                        f"declare add_special_tokens for {shape!r}, so the tokenizer's post-processor appends it"
                    )
            if self.anchor == "first" and segments[0].fixed is None and not self.adds_special_tokens(shape):
                raise ValueError(
                    f"an 'anchor: first' shape must open with a fixed segment or declare add_special_tokens "
                    f"for {shape!r}"
                )
        if self.anchor == "marker" and not self.anchor_markers:
            raise ValueError("anchor: marker needs anchor_markers: the named specials the model pools at")
        if self.anchor != "marker" and self.anchor_markers:
            raise ValueError("anchor_markers apply to anchor: marker only")
        return self

    # -- lookup -----------------------------------------------------------------------------------------------

    def shapes(self) -> tuple[RequestShape, ...]:
        """The declared request shapes, in canonical order."""
        return tuple(shape for shape in SHAPES if getattr(self, shape) is not None)

    def segments(self, shape: RequestShape) -> tuple[Segment, ...]:
        """The segments of ``shape``.

        Raises:
            ConfigError: the template does not declare that shape; the message names the ones it does.
        """
        declared = getattr(self, shape)
        if declared is None:
            raise ConfigError(
                f"the template declares no {shape!r} shape (it declares {list(self.shapes())})",
                hint="declare the shape's segments in the template, or fit a shape the template declares",
            )
        return declared

    def adds_special_tokens(self, shape: RequestShape) -> bool:
        """Whether the engine adds the tokenizer's post-processor tokens to this shape's rendered string.

        Raises:
            ConfigError: the template does not declare that shape (the same typed error :meth:`segments`
                raises, so a caller that reads the flag before the segments sees it too).
        """
        if isinstance(self.add_special_tokens, dict):
            try:
                return self.add_special_tokens[shape]
            except KeyError:
                raise ConfigError(
                    f"the template declares no {shape!r} shape (it declares {list(self.shapes())})",
                    hint="declare the shape's segments in the template, or fit a shape the template declares",
                ) from None
        return self.add_special_tokens

    # -- rendering --------------------------------------------------------------------------------------------

    def render(
        self,
        shape: RequestShape,
        tokenizer: TextTokenizer,
        *,
        query: str = "",
        document: str = "",
        instruction: str = "",
    ) -> str:
        """The rendered request string: the fixed frame with its specials resolved, the content spans filled.

        This is the string the client sends; the engine tokenizes it with the shape's
        ``add_special_tokens`` flag. Empty content spans render empty, which is how the fixed
        overhead is measured.
        """
        return "".join(
            segment.render(tokenizer, query=query, document=document, instruction=instruction)
            for segment in self.segments(shape)
        )

    def overhead(self, shape: RequestShape, tokenizer: TextTokenizer, *, instruction: str = "") -> int:
        """The fixed overhead of ``shape``, in tokens of ``tokenizer``: the template rendered once with every
        content span empty (the instruction filled -- it is fixed for the run, never cut) and counted as the
        engine reads it (the shape's ``add_special_tokens`` flag). Measured once per
        (template, shape, tokenizer, instruction) and cached.
        """
        flag = self.adds_special_tokens(shape)
        key = (json.dumps(self.model_dump(mode="json"), sort_keys=True), shape, tokenizer.sha256, flag, instruction)
        cached = _OVERHEAD_CACHE.get(key)
        if cached is None:
            rendered = self.render(shape, tokenizer, instruction=instruction)
            cached = tokenizer.count(rendered, add_special_tokens=flag)
            if len(_OVERHEAD_CACHE) >= 256:
                _OVERHEAD_CACHE.clear()
            _OVERHEAD_CACHE[key] = cached
        return cached


_OVERHEAD_CACHE: dict[tuple[str, str, str, bool, str], int] = {}
"""The measured fixed overhead, keyed by (template canonical JSON, shape, tokenizer SHA-256, flag,
instruction): one measurement per (template, shape) per process."""


def _resolve_specials(text: str, tokenizer: TextTokenizer) -> str:
    """Replace every ``{special:<name>}`` in *text* with the tokenizer's literal form of that added token."""

    def substitute(match: re.Match[str]) -> str:
        return tokenizer.special_text(match.group(1).strip())

    return _SPECIAL.sub(substitute, text)


__all__ = ["AnchorKind", "ContentSpan", "RequestShape", "SHAPES", "Segment", "TemplateSpec"]
