"""The model-plugin contribution shape: every subpackage of ``rcp_vllm.models`` passes the rules the
add-a-model how-to tells a contributor to follow, and a package that breaks one fails here, not in review.

The rules, each with a mutant below:

* the package imports no torch, vLLM or transformers when it is imported: a plugin is registry-lazy, and the
  harness, the recipe loader and the CPU tests all run where the engine is absent;
* a model registers as a ``"module:Class"`` string inside a ``register_<family>()`` function, never as a class
  object and never at module scope, so a process that never serves the model never imports its code;
* the class the string names is defined in the family's own subpackage;
* the architecture has a row in :data:`rcp_vllm.models.ARCHITECTURE_MODULES` that keys the module the qualname
  resolves to (the behaviour fingerprint hashes exactly a row's modules, so a row that omits the model module
  would let a change to the model class leave every recipe's fingerprint where it was);
* the ``register_<family>()`` callable is imported and called by the one entry point,
  :func:`rcp_vllm.models.register`;
* ``__init__.py`` declares ``__all__``.

:func:`shape_violations` and :func:`import_violation` are the checker the how-to's runnable snippet calls
(``docs/how-to/add-a-model.md``), so the skeleton the page hands a contributor is checked by the same rules as
the shipped tree. Nothing here imports vLLM, torch or a model module: the tree is read as source.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import rcp_vllm
from rcp_vllm.models import ARCHITECTURE_MODULES

MODELS_ROOT = Path(rcp_vllm.__file__).resolve().parent / "models"
"""The package the rules are about: one subpackage per model family, plus the one entry point."""

MODELS_PACKAGE = "rcp_vllm.models"
"""The dotted name of :data:`MODELS_ROOT`, the prefix a ``"module:Class"`` string starts with."""

ENGINE_MODULES = ("torch", "vllm", "transformers")
"""The imports a plugin package's import must not make: they exist only inside an engine process."""


@dataclass(frozen=True)
class Registration:
    """One ``ModelRegistry.register_model`` call, read as source.

    Attributes:
        package: The subpackage's directory name (the family).
        architecture: The architecture name, or ``None`` when the first argument is not a string constant.
        qualname: The ``"module:Class"`` string, or ``None`` when the second argument is not one (an eager
            registration).
        function: The enclosing function's name, ``""`` when the call runs at module scope.
    """

    package: str
    architecture: str | None
    qualname: str | None
    function: str

    @property
    def module(self) -> str:
        """The dotted module the qualname names (empty when it is not a qualname)."""
        return self.qualname.split(":", 1)[0] if self.qualname and ":" in self.qualname else ""

    @property
    def class_name(self) -> str:
        """The class the qualname names (empty when it is not a qualname)."""
        return self.qualname.split(":", 1)[1] if self.qualname and ":" in self.qualname else ""


def model_packages(root: Path = MODELS_ROOT) -> tuple[Path, ...]:
    """The model-family subpackages under ``root``, in name order (a directory with an ``__init__.py``)."""
    return tuple(sorted(p for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()))


def _sources(package: Path) -> tuple[Path, ...]:
    """Every Python source of one subpackage."""
    return tuple(sorted(package.rglob("*.py")))


def _constants(tree: ast.Module) -> dict[str, ast.expr]:
    """The module-scope assignments of a parsed file, by name (``AnnAssign`` included: ``__all__: list[str]``)."""
    found: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            found.update({t.id: node.value for t in node.targets if isinstance(t, ast.Name)})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            found[node.target.id] = node.value
    return found


def _string(expression: ast.expr, constants: Mapping[str, ast.expr]) -> str | None:
    """The string an argument stands for: a literal, or a module-scope constant naming one; else ``None``."""
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Name):
        value = constants.get(expression.id)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _registrations_of(tree: ast.Module, package: str) -> list[Registration]:
    """The ``ModelRegistry.register_model`` calls of one parsed file, with their enclosing function."""
    constants = _constants(tree)
    found: list[Registration] = []

    def walk(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, child.name)
                continue
            func = child.func if isinstance(child, ast.Call) else None
            if isinstance(func, ast.Attribute) and func.attr == "register_model":
                architecture = _string(child.args[0], constants) if child.args else None
                qualname = _string(child.args[1], constants) if len(child.args) > 1 else None
                found.append(Registration(package, architecture, qualname, function))
            walk(child, function)

    walk(tree, "")
    return found


def registrations(package: Path) -> tuple[Registration, ...]:
    """The ``ModelRegistry.register_model`` calls of one subpackage, with their enclosing function."""
    return tuple(
        registration
        for path in _sources(package)
        for registration in _registrations_of(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), package.name
        )
    )


def _module_file(package: Path, package_prefix: str, module: str) -> Path | None:
    """The file a dotted module names, when it lives inside ``package``; ``None`` otherwise."""
    own = f"{package_prefix}.{package.name}"
    if module == own:
        return package / "__init__.py"
    if not module.startswith(f"{own}."):
        return None
    path = package / (module[len(own) + 1 :].replace(".", "/") + ".py")
    return path if path.is_file() else None


def _defines_class(path: Path, name: str) -> bool:
    """Whether ``path`` defines a module-scope class ``name``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body)


def _entry_point(root: Path) -> tuple[set[str], set[str]]:
    """The names ``register()`` imports by a relative import, and the ``register_*`` names it calls."""
    init = root / "__init__.py"
    if not init.is_file():
        return set(), set()
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    entry = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "register"),
        None,
    )
    if entry is None:
        return set(), set()
    imported = {
        name.name for node in ast.walk(entry) if isinstance(node, ast.ImportFrom) and node.level for name in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(entry)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("register_")
    }
    return imported, called


def shape_violations(
    root: Path = MODELS_ROOT,
    *,
    package_prefix: str = MODELS_PACKAGE,
    registry: Mapping[str, tuple[str, ...]] = ARCHITECTURE_MODULES,
) -> list[str]:
    """The shape rules a model-family tree under ``root`` breaks, as ``"<family>: <what is wrong>"`` lines.

    Args:
        root: The directory that holds the family subpackages and the package's ``__init__.py`` (the entry
            point's ``register()``).
        package_prefix: The dotted name of ``root``, which a ``"module:Class"`` string starts with.
        registry: The architecture rows the tree must declare -- :data:`rcp_vllm.models.ARCHITECTURE_MODULES`
            for the shipped tree, and the rows a contribution adds for a skeleton.

    Returns:
        One line per broken rule, in family and source order; empty when every family has the shape.
    """
    violations: list[str] = []
    imported, called = _entry_point(root)
    for package in model_packages(root):
        name = package.name
        init = package / "__init__.py"
        init_tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        if "__all__" not in _constants(init_tree):
            violations.append(f"{name}: __init__.py declares no __all__")
        for registration in registrations(package):
            where = f"{name}: {registration.architecture or 'the architecture'}"
            if not registration.function:
                violations.append(f"{where} is registered at import; move the call into register_{name}()")
            elif not registration.function.startswith("register_"):
                violations.append(f"{where} is registered from {registration.function}(); only a register_*() runs")
            elif registration.function not in imported or registration.function not in called:
                violations.append(
                    f"{where}: {registration.function}() is not imported and called by {package_prefix}.register()"
                )
            if not registration.qualname:
                violations.append(f'{where} is registered eagerly; pass a "module:Class" string instead')
                continue
            if not registration.module or not registration.class_name:
                violations.append(f'{where} registers {registration.qualname!r}, which is not a "module:Class"')
                continue
            if registration.architecture is None:
                violations.append(f"{where} registers {registration.qualname!r} under a non-constant name")
            elif registration.architecture not in registry:
                violations.append(f"{where} has no row in ARCHITECTURE_MODULES")
            elif registration.module not in registry[registration.architecture]:
                violations.append(
                    f"{where}: ARCHITECTURE_MODULES[{registration.architecture!r}] does not key {registration.module}"
                )
            path = _module_file(package, package_prefix, registration.module)
            if path is None:
                violations.append(f"{where} registers {registration.qualname!r}: the module is not in {name}/")
            elif not _defines_class(path, registration.class_name):
                violations.append(f"{where} registers {registration.qualname!r}: {path.name} defines no class")
    return violations


_IMPORT_PROBE = """
import importlib
import sys

BLOCKED = set({blocked!r})


class _Blocker:
    \"\"\"Refuse the engine modules before any finder can satisfy them.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is imported at package import")
        return None


sys.meta_path.insert(0, _Blocker())
importlib.import_module({module!r})
print("clean")
"""


def import_violation(module: str, *, pythonpath: Path | None = None) -> str | None:
    """Why importing ``module`` is not engine-free, or ``None`` when the package import is clean.

    Runs in a subprocess (so an earlier import in this process cannot mask the check) with a meta-path hook
    that refuses torch, vLLM and transformers before any real finder sees them: the plugin's package import
    must not need the engine, and a contribution that imports it at package import fails here.

    Args:
        module: The dotted module to import (a family's package, e.g. ``"rcp_vllm.models.topk"``).
        pythonpath: A directory to put first on the child's ``PYTHONPATH`` (a tree the test wrote); the
            installed ``rcp_vllm`` source root is always added.

    Returns:
        ``None`` on a clean import, else the child's failure text.
    """
    source_root = str(Path(rcp_vllm.__file__).resolve().parents[1])
    parts = [str(pythonpath)] if pythonpath else []
    parts.append(source_root)
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        parts.append(inherited)
    script = _IMPORT_PROBE.format(blocked=ENGINE_MODULES, module=module)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(parts)},
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip() == "clean":
        return None
    return (result.stdout + result.stderr).strip()


def _write(root: Path, files: Mapping[str, str]) -> Path:
    """Write a synthetic model-family tree: ``{relative path: source}``, dedented."""
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return root


GOOD = {
    "__init__.py": """
        def register() -> None:
            from .good.plugin import register_good

            register_good()
    """,
    "good/__init__.py": '__all__ = ["register_good"]\n',
    "good/model.py": 'class GoodModel:\n    """The class the registry resolves."""\n',
    "good/plugin.py": """
        ARCHITECTURE = "GoodModel"
        QUALNAME = "mutant.good.model:GoodModel"


        def register_good() -> None:
            from vllm.model_executor.models import ModelRegistry

            ModelRegistry.register_model(ARCHITECTURE, QUALNAME)
    """,
}
"""A minimal plugin tree that passes every rule: the baseline each mutant is one edit away from."""

ROWS = {"GoodModel": ("mutant.good.model",)}
"""The row a contribution adds for :data:`GOOD`'s architecture."""


def test_the_baseline_tree_passes_every_rule(tmp_path: Path) -> None:
    """The mutants below fail because of their one edit, never because the synthetic tree is malformed."""
    root = _write(tmp_path / "mutant", GOOD)
    assert shape_violations(root, package_prefix="mutant", registry=ROWS) == []


def test_every_shipped_subpackage_has_the_plugin_shape() -> None:
    """Every family the wheel ships passes the rules: no exception list, no skip."""
    assert [package.name for package in model_packages()] == ["embedding_gemma2", "pplx", "topk"]
    assert shape_violations() == []


def test_the_shipped_tree_registers_the_three_architectures() -> None:
    """The checker reads the tree it claims to check: three lazy registrations, each in a ``register_*()``."""
    found = {
        registration.architecture: (registration.package, registration.qualname, registration.function)
        for package in model_packages()
        for registration in registrations(package)
    }
    assert found == {
        "TopkEmbedModel": ("topk", "rcp_vllm.models.topk.model:TopkEmbedModel", "register_topk"),
        "PplxContextualModel": ("pplx", "rcp_vllm.models.pplx.model:PplxContextualForPooling", "register_pplx"),
        "Qwen3_5Model": ("pplx", "rcp_vllm.models.pplx.late:PplxLateMultiVectorModel", "register_pplx"),
    }


@pytest.mark.parametrize("package", model_packages(), ids=lambda package: package.name)
def test_every_shipped_subpackage_imports_without_the_engine(package: Path) -> None:
    """Importing a family's package pulls no torch, vLLM or transformers: the plugin is registry-lazy."""
    assert import_violation(f"{MODELS_PACKAGE}.{package.name}") is None


def test_a_package_that_imports_torch_at_package_import_fails(tmp_path: Path) -> None:
    """The mutant: ``__init__.py`` imports torch, so the harness (which has no engine) cannot load it."""
    files = {**GOOD, "good/__init__.py": "import torch\n\n__all__: list[str] = []\n"}
    root = _write(tmp_path / "mutant", files)
    assert shape_violations(root, package_prefix="mutant", registry=ROWS) == []
    message = import_violation("mutant.good", pythonpath=tmp_path)
    assert message is not None
    assert "torch is imported at package import" in message


def test_a_package_that_registers_eagerly_fails(tmp_path: Path) -> None:
    """The mutant: the registry is handed the class object instead of a lazy ``"module:Class"`` string."""
    files = {
        **GOOD,
        "good/plugin.py": """
            ARCHITECTURE = "GoodModel"


            def register_good() -> None:
                from vllm.model_executor.models import ModelRegistry

                from .model import GoodModel

                ModelRegistry.register_model(ARCHITECTURE, GoodModel)
        """,
    }
    root = _write(tmp_path / "mutant", files)
    violations = shape_violations(root, package_prefix="mutant", registry=ROWS)
    assert violations == ['good: GoodModel is registered eagerly; pass a "module:Class" string instead']


def test_a_package_that_registers_at_import_fails(tmp_path: Path) -> None:
    """The mutant: the registration runs at module scope, so importing the family needs the engine."""
    files = {
        **GOOD,
        "good/plugin.py": """
            from vllm.model_executor.models import ModelRegistry

            ARCHITECTURE = "GoodModel"
            QUALNAME = "mutant.good.model:GoodModel"
            ModelRegistry.register_model(ARCHITECTURE, QUALNAME)


            def register_good() -> None:
                return None
        """,
    }
    root = _write(tmp_path / "mutant", files)
    violations = shape_violations(root, package_prefix="mutant", registry=ROWS)
    assert violations == ["good: GoodModel is registered at import; move the call into register_good()"]


def test_a_package_without_a_registry_row_fails(tmp_path: Path) -> None:
    """The mutant: the architecture is registered but has no ``ARCHITECTURE_MODULES`` row to key it."""
    root = _write(tmp_path / "mutant", GOOD)
    violations = shape_violations(root, package_prefix="mutant", registry={})
    assert violations == ["good: GoodModel has no row in ARCHITECTURE_MODULES"]


def test_a_row_that_does_not_key_the_model_module_fails(tmp_path: Path) -> None:
    """The mutant: the row exists but omits the module the qualname resolves, so its fingerprint misses it."""
    root = _write(tmp_path / "mutant", GOOD)
    violations = shape_violations(root, package_prefix="mutant", registry={"GoodModel": ("mutant.good.pooling",)})
    assert violations == [
        "good: GoodModel: ARCHITECTURE_MODULES['GoodModel'] does not key mutant.good.model",
    ]


def test_a_package_without___all___fails(tmp_path: Path) -> None:
    """The mutant: the family's ``__init__.py`` declares no ``__all__`` (the repository's public-module rule)."""
    files = {**GOOD, "good/__init__.py": ""}
    root = _write(tmp_path / "mutant", files)
    violations = shape_violations(root, package_prefix="mutant", registry=ROWS)
    assert violations == ["good: __init__.py declares no __all__"]


def test_a_package_the_entry_point_never_calls_fails(tmp_path: Path) -> None:
    """The mutant: the registration exists but ``register()`` neither imports nor calls it, so no engine runs it."""
    files = {**GOOD, "__init__.py": "def register() -> None:\n    return None\n"}
    root = _write(tmp_path / "mutant", files)
    violations = shape_violations(root, package_prefix="mutant", registry=ROWS)
    assert violations == ["good: GoodModel: register_good() is not imported and called by mutant.register()"]


def test_a_qualname_whose_class_is_not_defined_fails(tmp_path: Path) -> None:
    """The mutant: the string names a class the module does not define (a renamed or moved model class)."""
    files = {**GOOD, "good/model.py": 'class Renamed:\n    """Not the class the registry resolves."""\n'}
    root = _write(tmp_path / "mutant", files)
    violations = shape_violations(root, package_prefix="mutant", registry=ROWS)
    assert violations == [
        "good: GoodModel registers 'mutant.good.model:GoodModel': model.py defines no class",
    ]
