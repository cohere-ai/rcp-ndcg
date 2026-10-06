"""No in-process model code under ``src/rcp_ndcg/``.

RFC-0001 (option 1, section 6.2): the package carries no model that loads weights. The only torch
left is the core's IRT estimators, reached through the ``[calibrate]`` extra; nothing under
``src/rcp_ndcg/`` may import torch, transformers, accelerate or vllm, at module level or lazily.
The ``[local]`` and ``[vllm]`` extras have left ``pyproject.toml``: every model is served, and the
package installs cleanly next to an engine image without touching it.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg"

_FRAMEWORK_IMPORT = re.compile(r"\b(?:from|import)\s+(?:torch|transformers|accelerate|vllm)\b")

#: The modules allowed to import a framework: the calibration fit (the core's IRT estimators under
#: it, imported through the ``[calibrate]`` extra).
_ALLOWED = frozenset({"calibration"})

#: The in-process modules this lane deleted; no reference to them may remain under ``src/``.
_DELETED = (
    "rcp_ndcg.retrieval._http",
    "rcp_ndcg.retrieval.api_dense",
    "rcp_ndcg.retrieval.vllm_http",
    "rcp_ndcg.retrieval.hf_dense",
    "rcp_ndcg.retrieval.external_rerankers",
    "rcp_ndcg.retrieval.accel",
    "rcp_ndcg.retrieval.cross_encoder",
    "rcp_ndcg.retrieval.encoders",
    "rcp_ndcg.retrieval.encoder",
)


def _modules() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts)


def test_no_module_under_src_imports_a_model_framework() -> None:
    """No torch, transformers, accelerate or vllm import anywhere under ``src/rcp_ndcg/``."""
    offenders = []
    for path in _modules():
        relative = path.relative_to(SRC)
        if relative.parts and relative.parts[0] in _ALLOWED:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if _FRAMEWORK_IMPORT.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, "in-process model imports under src/rcp_ndcg/:\n" + "\n".join(offenders)


def test_the_reference_code_lives_outside_the_package() -> None:
    """The paper's in-process implementations moved to experiments/, and the package never imports them."""
    reference = Path(__file__).resolve().parents[1] / "experiments" / "paper" / "rerankers" / "reference"
    assert reference.is_dir() and list(reference.glob("*.py")), "the reference code is there"
    for path in _modules():
        assert "rerankers.reference" not in path.read_text(encoding="utf-8"), path


def test_nothing_under_src_names_the_deleted_modules() -> None:
    """The in-process modules are deleted: no import, lazy import or docstring path may remain."""
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        for name in _DELETED:
            assert name not in text, f"{path}: {name}"
