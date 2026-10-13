"""The package's version constant: the distribution manifest is the one source of the version."""

from __future__ import annotations

import tomllib
from pathlib import Path

import rcp_ndcg_vllm


def test_the_version_constant_matches_the_manifest() -> None:
    """``rcp_ndcg_vllm.__version__`` is the version its ``pyproject.toml`` declares.

    The constant is hand-kept (the package must import in the engine image without installed metadata),
    so a version bump that misses it desynchronises silently; this pins the two together.
    """
    manifest = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert rcp_ndcg_vllm.__version__ == manifest["project"]["version"]
