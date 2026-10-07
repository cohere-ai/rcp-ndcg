"""The harness's typed errors. The recipe loader's :class:`RecipeError` keeps its home in
:mod:`rcp_ndcg_test.errors` and is re-exported here so the harness's callers take both from one place.

Every public module declares ``__all__``; every error here is a typed, actionable failure.
"""

from __future__ import annotations

from rcp_ndcg_vllm.errors import RecipeError as RecipeError

__all__ = ["HarnessError", "RecipeError"]


class HarnessError(RuntimeError):
    """A harness failure outside the recipe schema: a missing tokenizer, a failed reference subprocess,
    a wire the capture cannot decode.

    Attributes:
        message: What failed, and what the harness was doing when it did.
    """
