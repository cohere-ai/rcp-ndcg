"""The qwen3-vl-reranker reference's vision wiring: a media row that cannot be processed fails, never scores.

The card's script re-renders the prompt as a NULL-only user turn when ``process_vision_info`` raises, and
scores that turn -- a plausible-looking score for a row whose image was never read. The reference is the
instrument a gated recipe is compared against, so it refuses the row instead (the declared deviation is in
the recipe's notes). The module is loaded from its file: its top level imports the standard library only.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REFERENCE = Path(__file__).resolve().parents[1] / "src" / "rcp_vllm" / "recipes" / "qwen3-vl-reranker" / "reference.py"


def _reference_module() -> Any:
    """The family's ``reference.py`` as a module (a subprocess script, so it is loaded by path)."""
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker_reference", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Processor:
    """The processor as far as ``tokenize`` reaches it before the vision wiring runs."""

    def __init__(self) -> None:
        self.renders: list[Any] = []

    def apply_chat_template(self, pairs: Any, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        self.renders.append(pairs)
        return "the row's own render"

    def __call__(self, **kwargs: Any) -> Any:
        raise AssertionError(f"the card's NULL fallback scored the row; renders: {self.renders!r}")


def test_a_vision_wiring_failure_refuses_the_row_instead_of_scoring_a_null_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``process_vision_info`` raising means the row's media was never read: a typed refusal naming the
    failure, never the card's silent "NULL" turn (which would gate a score of a document the reference
    did not see)."""
    module = _reference_module()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("cannot identify image file")

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(process_vision_info=broken))
    reference = module.Qwen3VLRerankerReference.__new__(module.Qwen3VLRerankerReference)
    reference.processor = processor = _Processor()

    with pytest.raises(SystemExit, match="process_vision_info"):
        reference.tokenize([[{"role": "user", "content": [{"type": "text", "text": "q"}]}]])

    assert len(processor.renders) == 1, "the row is rendered once and refused; no NULL re-render"
