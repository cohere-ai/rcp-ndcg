"""The package's version constant: the distribution manifest is the one source of the version."""

from __future__ import annotations

import re
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


def test_the_declared_expression_covers_every_licence_notice_names() -> None:
    """The one expression a redistributor reads must cover every licence ``NOTICE`` names.

    ``NOTICE`` names Apache-2.0 for most third-party works but CC-BY-NC-SA-4.0 (ContextualAI),
    CC-BY-NC-4.0 (Jina AI) and MIT (Perplexity AI) for the rest, and the wheel bundles all of them.
    Declaring Apache-2.0 alone told a commercial redistributor the opposite of the truth, and PyPI
    cannot be corrected in place.
    """
    package = Path(__file__).resolve().parents[1]
    project = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {token.strip() for token in project["license"].split("AND")}

    named = {
        match.group(1).strip()
        for line in (package / "NOTICE").read_text(encoding="utf-8").splitlines()
        if (match := re.search(r"Licence:\s*([A-Za-z0-9.\-]+)", line))
    }
    assert named, "NOTICE names no licence: the parse is wrong, not the manifest"
    assert not named - declared, f"NOTICE names {sorted(named - declared)} but the expression omits them"
    assert not declared - named, f"the expression declares {sorted(declared - named)}, which NOTICE does not name"


def test_the_licence_texts_the_wheel_ships_are_complete() -> None:
    """Every non-Apache licence the wheel bundles ships its text, and ``license-files`` carries it."""
    package = Path(__file__).resolve().parents[1]
    project = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license-files"] == ["LICENSE", "NOTICE", "LICENSES/*.txt"]
    for name, marker in (
        ("CC-BY-NC-4.0.txt", "Attribution-NonCommercial 4.0 International"),
        ("CC-BY-NC-SA-4.0.txt", "Attribution-NonCommercial-ShareAlike 4.0 International"),
        ("MIT.txt", "Permission is hereby granted"),
    ):
        text = (package / "LICENSES" / name).read_text(encoding="utf-8")
        assert marker in text, f"LICENSES/{name} does not look like its licence text"
        assert len(text.splitlines()) > 8, f"LICENSES/{name} looks truncated"
