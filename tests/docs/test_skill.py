"""The shipped skill: its front matter, its length, and its exit-code table (the documented one, for every code).

Every ``rcp-ndcg`` command in it is checked against the command tree by ``test_commands``.
"""

from __future__ import annotations

import re

import yaml

from rcp_ndcg.errors import ExitCode
from rcp_ndcg.examples import run_config_path
from tests.docs._markdown import ROOT

(SKILL,) = (ROOT / "skills" / "rcp-ndcg").glob("*.md")  # the one skill file


def _exit_table(text: str) -> list[str]:
    section = text[text.index("## Exit codes") :]
    return [line for line in section.splitlines() if re.match(r"\| \d+ \|", line)]


def test_the_skill_has_front_matter_and_fits_the_budget() -> None:
    text = SKILL.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["name"] == "rcp-ndcg" and len(front["description"]) > 50
    assert len(text.splitlines()) <= 200
    assert not list(ROOT.glob("SKILL*.md"))  # one skill, in skills/


def test_the_exit_code_table_is_the_documented_one_and_covers_every_code() -> None:
    skill = _exit_table(SKILL.read_text(encoding="utf-8"))
    reference = _exit_table((ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8"))
    assert skill == reference
    rows = [(int(cells[0].strip("| ")), cells[1].strip("`")) for cells in (line.split(" | ") for line in skill)]
    assert rows == [(code.value, code.name) for code in ExitCode]


def test_the_polling_instruction_names_every_status_a_run_ends_in() -> None:
    # A run stopped by its --budget-usd ceiling ends `partial` with exit 0; polling for `completed` alone never ends.
    from rcp_ndcg.runs import RunStatus

    text = " ".join(SKILL.read_text(encoding="utf-8").split())
    poll = text[text.index("Poll `run status`") :].split(". ")[0]
    final = set(RunStatus) - {RunStatus.SUBMITTED, RunStatus.RUNNING}
    assert all(f"`{status.value}`" in poll for status in final), poll


def test_every_documented_budget_on_an_ad_hoc_judge_states_its_price() -> None:
    """A budget needs a price, and a --judge-url judge has none unless --set judge.price.* gives it."""
    pages = [
        SKILL,
        ROOT / "README.md",
        ROOT / "docs" / "quickstart.md",
        run_config_path("rejudge_nfcorpus"),
    ]
    for page in pages:
        commands = page.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
        for command in commands:
            if "--judge-url" in command and "--budget-usd" in command:
                assert "judge.price.input_usd_per_mtok" in command and "judge.price.output_usd_per_mtok" in command, (
                    page.name,
                    command,
                )


def test_the_reference_names_every_warning_code() -> None:
    from rcp_ndcg.errors import WARNING_CODES

    reference = (ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    listed = reference[reference.index("rcp_ndcg.errors.WarningCode") :].split(")")[0]
    assert sorted(re.findall(r"`([A-Z0-9_]+)`", listed)) == sorted(WARNING_CODES)
