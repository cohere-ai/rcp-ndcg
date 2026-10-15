"""The GitHub workflows parse: a YAML error disables a whole workflow on GitHub, not only the broken step."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda path: path.name)
def test_the_workflow_parses_and_names_its_jobs(workflow: Path) -> None:
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and document.get("jobs"), f"{workflow.name} declares no jobs"
    for name, job in document["jobs"].items():
        steps = job.get("steps", [])
        assert all(isinstance(step, dict) for step in steps), f"{workflow.name}: a step of {name} is not a mapping"


def test_the_release_workflow_checks_the_citation_version() -> None:
    """``CITATION.cff`` carries the release's version and no workflow checked it: a tag could ship a citation
    naming another release. The build job reads the file and compares its ``version:`` with the tag."""
    release = next(path for path in _WORKFLOWS if path.name == "release.yml")
    document = yaml.safe_load(release.read_text(encoding="utf-8"))
    scripts = "\n".join(str(step.get("run", "")) for step in document["jobs"]["build"]["steps"])
    assert "CITATION.cff" in scripts, "the release workflow must check CITATION.cff"
    assert "GITHUB_REF_NAME#v" in scripts, "the check compares the file with the tag"


def _ci() -> tuple[Path, dict]:
    ci = next(path for path in _WORKFLOWS if path.name == "ci.yml")
    return ci, yaml.safe_load(ci.read_text(encoding="utf-8"))


def test_the_test_pkg_job_does_not_combine_quiet_and_verbose() -> None:
    """g-ci-2: ``-q`` and ``-v`` count into pytest's one verbosity, so ``-q -v`` nets to the default and
    prints no test names: the 474fdec crash diagnostic cannot name the crashing test (the cause is fixed in
    observe.requests.main).  The command must not combine them again, and no comment may cite the removed
    ``bin/gate`` script."""
    ci, document = _ci()
    commands = [str(step.get("run", "")) for step in document["jobs"]["test-pkg"]["steps"]]
    pytest_commands = [command for command in commands if "pytest" in command]
    assert pytest_commands, "the test-pkg job runs pytest"
    for command in pytest_commands:
        flags = command.split()
        assert not ("-q" in flags and "-v" in flags), f"-q and -v cancel out: {command}"
    assert "bin/gate" not in ci.read_text(encoding="utf-8"), "bin/gate does not exist in the repository"
