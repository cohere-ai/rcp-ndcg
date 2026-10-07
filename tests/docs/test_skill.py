"""The shipped skill: its front matter, its length, and its exit-code table (the documented one, for every code).

Every ``rcp-ndcg`` command in it is checked against the command tree by ``test_commands``.
"""

from __future__ import annotations

import re

import yaml

from rcp_ndcg.errors import ExitCode
from tests.docs._markdown import ROOT

(SKILL,) = (ROOT / "skills" / "rcp-ndcg").glob("*.md")  # the one skill file


def _exit_table(text: str) -> list[str]:
    if "## Exit codes" not in text:
        return []  # the table has one home now (the reference); the skill links it
    section = text[text.index("## Exit codes") :]
    return [line for line in section.splitlines() if re.match(r"\| \d+ \|", line)]


def test_the_skill_has_front_matter_and_fits_the_budget() -> None:
    text = SKILL.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["name"] == "rcp-ndcg" and len(front["description"]) > 50
    assert len(text.splitlines()) <= 200
    assert not list(ROOT.glob("SKILL*.md"))  # one skill, in skills/


def test_the_exit_code_table_is_the_documented_one_and_covers_every_code() -> None:
    reference = _exit_table((ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8"))
    skill = SKILL.read_text(encoding="utf-8")
    assert _exit_table(skill) == [], "the skill links the table; it does not carry a second copy"
    assert "(../../docs/reference/cli.md#exit-codes)" in skill
    rows = [(int(cells[0].strip("| ")), cells[1].strip("`")) for cells in (line.split(" | ") for line in reference)]
    assert rows == [(code.value, code.name) for code in ExitCode]


def test_the_polling_instruction_names_every_status_a_run_ends_in() -> None:
    # A run that ran only some of its steps ends `partial` with exit 0; polling for `completed` alone never ends.
    from rcp_ndcg.runs import RunStatus

    text = " ".join(SKILL.read_text(encoding="utf-8").split())
    poll = text[text.index("Poll `run status`") :].split(". ")[0]
    final = set(RunStatus) - {RunStatus.SUBMITTED, RunStatus.RUNNING}
    assert all(f"`{status.value}`" in poll for status in final), poll


def test_the_reference_names_every_warning_code() -> None:
    from rcp_ndcg.errors import WARNING_CODES

    reference = (ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    listed = reference[reference.index("rcp_ndcg.errors.WarningCode") :].split(")")[0]
    assert sorted(re.findall(r"`([A-Z0-9_]+)`", listed)) == sorted(WARNING_CODES)
