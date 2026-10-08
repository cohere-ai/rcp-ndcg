"""The layering charter, as a test: eager imports point inward only.

``AGENTS.md`` fixes the import order
(``rcp_ndcg_core → support → storage → data → inference → retrieval → llm → calibration → eval → runners → runs
→ schemas | mcp → cli``, with ``errors`` below ``support``, the facade just above ``runs``, and ``testing`` and
``examples`` above it). This module parses every module under ``rcp-ndcg/src/rcp_ndcg/`` with :mod:`ast` and
fails when an eager import -- module-level, outside ``if TYPE_CHECKING:`` -- points outward in that order
(toward a layer that typically imports this one).

An import of the facade (``rcp_ndcg``) resolves to the layer of the name it binds (``from rcp_ndcg import
storage`` is a storage import); a name the facade itself defines (``__version__``) resolves to the facade. A
lazily imported module is exempt: its cost is paid when it runs, not when its layer is imported.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "rcp-ndcg/src/rcp_ndcg"

#: The layers, from the one every module may import to the one that may import everything; the charter of
#: ``AGENTS.md`` (Layout and layering), with the two single-file modules it does not name placed where the
#: current tree already satisfies them.
LAYERS: tuple[str, ...] = (
    "rcp_ndcg_core",
    "errors",
    "support",
    "storage",
    "data",
    "inference",
    "retrieval",
    "llm",
    "calibration",
    "eval",
    "runners",
    "runs",
    "rcp_ndcg",  # the facade (its __init__): it re-exports everything up to runs, and cli reads its __version__
    "schemas",
    "mcp",
    "cli",
    "testing",
    "examples",
)

_LAYER = {name: index for index, name in enumerate(LAYERS)}

#: Eager imports the charter would refuse, allowed to stand until the lane that owns their layer moves them.
#: Each entry names its module and the outward target it may keep importing; nothing may be added here without
#: naming it in the lane report. Empty today: the tree the charter was first written against satisfies it.
_ALLOW_LIST: dict[str, frozenset[str]] = {}


def _module_layer(dotted: str) -> str:
    """The layer of a module of ``rcp_ndcg`` (its top-level part; the facade's own ``__init__`` is the facade)."""
    parts = dotted.split(".")
    return parts[0] if parts else "rcp_ndcg"


def _targets_of_import(node: ast.Import | ast.ImportFrom, *, module: str, is_package: bool) -> Iterator[str]:
    """The layers one import statement binds, as layers of ``rcp_ndcg``.

    A relative import is resolved to its absolute module from the importing module's package (``module``: the
    package itself when ``is_package``, i.e. a package's ``__init__``) and ``node.level``, and then checked like
    an absolute one (``from ..retrieval import x`` in a ``data`` module is a ``retrieval`` import).
    """
    if isinstance(node, ast.ImportFrom):
        if node.level:
            # `module` is relative to the package (``data.dataset``): resolve against the real dotted name, whose
            # root is ``rcp_ndcg`` (the facade's own ``__init__`` already carries it).
            real = module if module == "rcp_ndcg" else f"rcp_ndcg.{module}"
            anchor = real if is_package else real.rpartition(".")[0]
            parts = anchor.split(".")
            if node.level - 1 >= len(parts):  # relative beyond the top-level package: an ImportError, not a layer
                return
            parts = parts[: len(parts) - (node.level - 1)]
            if not parts:  # relative beyond the top-level package: unresolvable statically
                return
            module = ".".join([*parts, node.module or ""])
        else:
            module = node.module or ""
        names = [alias.name for alias in node.names]
    else:
        module = ""
        names = [alias.name for alias in node.names]
    for name in names:
        target = f"{module}.{name}" if module else name
        if target.startswith("rcp_ndcg_core"):
            yield "rcp_ndcg_core"
        elif target == "rcp_ndcg" or target.startswith("rcp_ndcg."):
            rest = target[len("rcp_ndcg") :].strip(".")
            top = rest.split(".")[0] if rest else ""
            # ``from rcp_ndcg import storage`` binds the storage package; a name the facade itself defines
            # (``__version__``) binds the facade.
            yield top if top in _LAYER else "rcp_ndcg"


def _eager_imports(tree: ast.Module) -> Iterator[tuple[ast.Import | ast.ImportFrom, int]]:
    """Every eager import of a module's syntax tree, with its line.

    Module scope only: bodies of functions and methods run later, and a ``TYPE_CHECKING`` guard (a bare name or
    an attribute of ``typing``) never executes. Imports directly in a class body do execute at import time, so
    they count.
    """

    def scan(stmts: list[ast.stmt]) -> Iterator[tuple[ast.Import | ast.ImportFrom, int]]:
        for statement in stmts:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if isinstance(statement, ast.ClassDef):
                yield from scan(statement.body)  # a class body executes at import time; its methods do not
                continue
            if _is_type_checking(statement):
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Import | ast.ImportFrom):
                    yield node, node.lineno

    yield from scan(tree.body)


def _is_type_checking(statement: ast.stmt) -> bool:
    """Whether ``statement`` is ``if TYPE_CHECKING:`` (a bare name or an attribute of ``typing``)."""
    if not isinstance(statement, ast.If):
        return False
    test = statement.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def outward_imports() -> list[str]:
    """Every eager import of ``rcp-ndcg/src/rcp_ndcg`` that points outward in the charter order.

    Allow-listed imports excepted.
    """
    findings: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        is_package = path.name == "__init__.py"
        if module == "__init__":
            module = "rcp_ndcg"  # the facade itself
        elif module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        layer = _module_layer(module)
        if layer not in _LAYER:
            raise AssertionError(
                f"{module} is a module of the package but its top-level layer is not placed in LAYERS; "
                "add it where the charter puts it"
            )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, line in _eager_imports(tree):
            for target in _targets_of_import(node, module=module, is_package=is_package):
                if target not in _LAYER or target == layer:
                    continue
                if _LAYER[target] <= _LAYER[layer]:
                    continue
                if target in _ALLOW_LIST.get(module, frozenset()):
                    continue
                findings.append(f"rcp-ndcg/src/rcp_ndcg/{path.relative_to(SRC)}:{line}: {layer} -> {target} (outward)")
    return findings


def test_no_eager_import_points_outward() -> None:
    assert outward_imports() == [], "eager imports point outward in the charter order"


def test_a_relative_import_across_layers_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative import is resolved to its absolute module: `from ..retrieval import x` in a data module fails.

    The mutation injects the outward relative import into a tree under ``tmp_path`` (the checkout is never
    written); the same-layer relative imports beside it stay legal. The module names then read like the real
    tree's (relative to the package).
    """
    for name in ("data", "retrieval"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "data" / "broken.py").write_text("from ..retrieval import anything\n", encoding="utf-8")
    (tmp_path / "data" / "same_layer.py").write_text("from .broken import anything\n", encoding="utf-8")
    (tmp_path / "data" / "up_one.py").write_text("from .. import data\n", encoding="utf-8")
    (tmp_path / "data" / "beyond.py").write_text("from ....retrieval import anything\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path)

    # every finding is prefixed with the checkout's own rcp-ndcg/src/rcp_ndcg, whatever SRC reads now
    assert outward_imports() == ["rcp-ndcg/src/rcp_ndcg/data/broken.py:1: data -> retrieval (outward)"]


def test_a_top_level_module_that_is_not_placed_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new top-level module the charter has not placed is a named instruction, never a KeyError."""
    import sys

    module = sys.modules[__name__]
    shrink = tuple(layer for layer in LAYERS if layer != "data")
    monkeypatch.setattr(module, "LAYERS", shrink)
    monkeypatch.setattr(module, "_LAYER", {name: index for index, name in enumerate(shrink)})
    with pytest.raises(AssertionError, match=r"\bdata\b.*not placed in LAYERS"):
        outward_imports()
