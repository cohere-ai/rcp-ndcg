"""The repo's docstring rule on the record modules: every public class, function, method and
property states its inputs, outputs and units (AGENTS.md, "Rules")."""

from __future__ import annotations

import importlib
import inspect

import pytest

MODULES = ("rcp_ndcg_core.content", "rcp_ndcg_core.records")


def _documented(obj: object) -> bool:
    return bool((inspect.getdoc(obj) or "").strip())


@pytest.mark.parametrize("module_name", MODULES)
def test_every_public_class_and_member_has_a_docstring(module_name: str) -> None:
    """``TextPart``/``ImagePart`` and their ``media_refs``, ``Content``'s constructors and views,
    the ``Query``/``Document``/``RankingExample`` classes and their properties were bare, so the
    generated pages showed no role for them."""
    module = importlib.import_module(module_name)
    missing: list[str] = []
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            if not _documented(obj):
                missing.append(name)
            for member_name, member in inspect.getmembers(obj):
                if member_name.startswith("_"):
                    continue
                if isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member) and member.__module__ == module_name and not _documented(member):
                    missing.append(f"{name}.{member_name}")
        elif inspect.isfunction(obj) and not _documented(obj):
            missing.append(name)
    assert not missing, f"public names without a docstring: {missing}"
