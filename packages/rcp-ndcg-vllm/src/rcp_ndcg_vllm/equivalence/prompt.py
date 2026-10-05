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
    """The two tokenizer operations stage 1 needs: ids for text, and token strings for a mismatch report."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` without special tokens (the rendered prompt carries them already)."""

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """The token string of one id, for the first-mismatch report."""


class TransformersTokenizerAdapter(TokenizerAdapter):
    """The adapter around a transformers tokenizer (the recipe's ``client.tokenizer`` on a GPU host)."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` without special tokens."""
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def id_to_token(self, token_id: int) -> str:
        """The token string of one id."""
        return str(self._tokenizer.convert_ids_to_tokens(int(token_id)))


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


def served_prompt_text(recipe: Recipe, query: str, document: str) -> str:
    """The text the engine tokenises for one (query, document) pair: the template, or the prompted document."""
    if recipe.role == "rerank":
        return render_template_for(recipe, query, document)
    return recipe.client.doc_prompt + document


# ``render_template`` is the primitive; ``render_template_for`` is the recipe-aware wrapper.
