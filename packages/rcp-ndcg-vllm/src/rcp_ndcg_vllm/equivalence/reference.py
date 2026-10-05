"""Loading a recipe's ``reference.py`` (the in-process reference implementation).

The reference module is loaded with :func:`importlib` straight from the recipe directory; it is never imported
through the package, so it can carry any imports the model needs (torch, transformers, the checkpoint's remote
code).  The interface, which every recipe lane implements:

- ``load(device: str) -> object`` — load the model once onto ``device`` (``"cpu"`` or ``"cuda:0"``); the module
  keeps it and later calls use it.
- ``score(query: str, documents: list[str], instruction: str | None) -> list[float]`` — rerank: one score per
  document, on the recipe's ``reference.score_scale``.  The instruction is the recipe's
  ``client.default_instruction``; the reference folds it the way its paper code does.  ``score_query`` is accepted
  as an alias of ``score`` (the name the first recipes used).
- ``embed(texts: list[str], role: str) -> list[numpy.ndarray]`` — embedding roles: one float32/float16 array per
  text, shape ``(dim,)`` for a dense model and ``(n_tokens, dim)`` for a late-interaction model.  ``role`` is
  ``"query"`` or ``"document"``; the reference composes its own prompts.
- ``render(query: str, document: str, instruction: str | None) -> list[int]`` — stage 1: the token ids of the exact
  prompt the engine must see for that pair.  For an embedding recipe, the ids of ``doc_prompt + document``.

An optional fifth member serves CPU-only checking (tests and CI): ``tokenizer() -> object`` returning an object
with ``encode(text) -> list[int]`` and ``id_to_token(id) -> str``; when present the harness uses it for stage 1
instead of loading ``client.tokenizer`` from the Hub with transformers.  Production references omit it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ..errors import HarnessError

__all__ = ["Reference", "load_reference"]


class Reference:
    """One loaded reference module plus the names the harness calls."""

    def __init__(self, module: Any, path: str) -> None:
        self._module = module
        self.path = path

    def load(self, device: str) -> Any:
        """``load(device)`` of the reference module, exactly once per process (later calls are no-ops)."""
        try:
            return self._module.load(device)
        except HarnessError:
            raise
        except Exception as error:
            raise HarnessError(f"{self.path} load({device!r}) failed: {error}") from error

    def score(self, query: str, documents: list[str], instruction: str | None) -> list[float]:
        """``score(query, documents, instruction)`` of the reference module (rerank recipes)."""
        function = getattr(self._module, "score", None) or getattr(self._module, "score_query", None)
        if function is None:
            raise HarnessError(f"{self.path} defines neither score nor score_query(query, documents, instruction)")
        try:
            return [float(value) for value in function(query, documents, instruction)]
        except HarnessError:
            raise
        except Exception as error:
            raise HarnessError(f"{self.path} score(...) failed: {error}") from error

    def embed(self, texts: list[str], role: str) -> list[Any]:
        """``embed(texts, role)`` of the reference module (embedding recipes)."""
        function = getattr(self._module, "embed", None)
        if function is None:
            raise HarnessError(f"{self.path} does not define embed(texts, role); an embedding recipe needs it")
        try:
            return function(texts, role)
        except HarnessError:
            raise
        except Exception as error:
            raise HarnessError(f"{self.path} embed(...) failed: {error}") from error

    def render(self, query: str, document: str, instruction: str | None) -> list[int]:
        """``render(query, document, instruction)`` of the reference module (stage 1, the default shape)."""
        function = getattr(self._module, "render", None)
        if function is None:
            raise HarnessError(f"{self.path} does not define render(query, document, instruction)")
        try:
            return [int(value) for value in function(query, document, instruction)]
        except HarnessError:
            raise
        except Exception as error:
            raise HarnessError(f"{self.path} render(...) failed: {error}") from error

    def has_render_shape(self) -> bool:
        """Whether the reference module implements the optional ``render_shape`` hook (per-shape stage 1)."""
        return callable(getattr(self._module, "render_shape", None))

    def render_shape(self, shape: str, query: str, document: str, instruction: str | None) -> list[int]:
        """``render_shape(shape, query, document, instruction)`` of the reference module, when it defines it.

        The optional per-shape stage-1 hook: recipes whose declared shapes differ beyond the default need it
        for the reference side of the anchor audit.  Raises :class:`HarnessError` when the module lacks it.
        """
        function = getattr(self._module, "render_shape", None)
        if function is None:
            raise HarnessError(f"{self.path} does not define render_shape(shape, query, document, instruction)")
        try:
            return [int(value) for value in function(shape, query, document, instruction)]
        except HarnessError:
            raise
        except Exception as error:
            raise HarnessError(f"{self.path} render_shape(...) failed: {error}") from error


def load_reference(recipe_dir: str | Path, entry: str = "reference.py") -> Reference:
    """Load ``<recipe_dir>/<entry>`` as a module and wrap it; raises :class:`HarnessError` when it cannot."""
    path = Path(recipe_dir) / entry
    if not path.is_file():
        raise HarnessError(f"no reference module at {path}")
    spec = importlib.util.spec_from_file_location("rcp_ndcg_vllm_fixture_reference", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib failure mode
        raise HarnessError(f"cannot load reference module {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise HarnessError(f"{path} failed to import: {error}") from error
    return Reference(module, str(path))
