"""The shipped prompts are package data, loaded by name, with their criteria read from the text."""

from __future__ import annotations

from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.llm import load_prompt
from rcp_ndcg.llm.prompts import PROMPT_FILES, Prompt, prompt_path, shipped_prompt_name


def test_every_shipped_prompt_loads_from_any_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in PROMPT_FILES:
        prompt = load_prompt(name)
        assert prompt_path(name).is_absolute()  # type: ignore[arg-type]
        assert "{passages_placeholder}" in prompt.text
        expected = ("C1", "C2", "C3", "C4", "C5") if name.startswith("rubric") else ()
        assert prompt.criteria == expected, name


def test_the_shipped_prompt_follows_the_modality() -> None:
    assert shipped_prompt_name("rubric", "text") == "rubric"
    assert shipped_prompt_name("tournament", "image") == "tournament_vision"
    assert shipped_prompt_name("rubric", "video") == "rubric_video"


def test_the_text_and_page_rubrics_are_different_instruments() -> None:
    assert len({load_prompt(name).sha256 for name in PROMPT_FILES}) == len(PROMPT_FILES)


def test_criteria_are_a_contiguous_ladder_or_none() -> None:
    assert Prompt("p", "Judge C1, C2 and C3.").criteria == ("C1", "C2", "C3")
    assert Prompt("p", "Judge C1 and C3.").criteria == ()


def test_a_custom_prompt_loads_from_a_path_and_an_unknown_name_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mine.txt"
    path.write_text("  Judge C1 and C2. {query_placeholder} {passages_placeholder}\n")
    assert load_prompt(str(path)).criteria == ("C1", "C2")
    with pytest.raises(ConfigError, match="tournament, rubric"):
        prompt_path("rubric_v2")  # type: ignore[arg-type]
