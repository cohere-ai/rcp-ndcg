"""Config validation errors name the field, the given value, the expected type, a did-you-mean and their source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runs.config import RunConfig


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text("dataset: jsonl:rows.jsonl\njudge: fake\nsteps: [tournament]\n")
    return path


def _error(config: Path, *overrides: str) -> ConfigError:
    with pytest.raises(ConfigError) as caught:
        RunConfig.load(config, overrides=overrides)
    return caught.value


def test_a_set_typo_blames_set_and_proposes_the_known_key(config: Path) -> None:
    error = _error(config, "evaluation.kk=5")

    (problem,) = error.details["errors"]
    assert problem == {
        "field": "evaluation.kk",
        "problem": "unknown key",
        "input": 5,
        "did_you_mean": "evaluation.k",
        "source": "--set",
    }
    assert error.message.startswith("--set evaluation.kk: unknown key; did you mean 'evaluation.k'?")
    assert "YAML literals" in (error.hint or "")
    assert "errors.pydantic.dev" not in error.message and str(config) not in error.message


def test_a_file_typo_blames_the_file(config: Path) -> None:
    config.write_text(config.read_text() + "tournment: {window: 4}\n")
    (problem,) = _error(config).details["errors"]
    assert (problem["field"], problem["did_you_mean"], problem["source"]) == ("tournment", "tournament", str(config))


def test_a_wrong_type_names_the_expected_one(config: Path) -> None:
    (problem,) = _error(config, "limit=many").details["errors"]
    assert problem["input"] == "many" and problem["source"] == "--set"
    assert "integer" in problem["expected"]
    (problem,) = _error(config, "steps=[tournament, calibrat]").details["errors"]
    assert problem["field"] == "steps.1" and "'calibrate'" in problem["expected"]


def test_the_cli_carries_the_structured_errors(config: Path) -> None:
    args = ["run", "start", str(config), "--set", "tournament.windw=4", "--estimate", "--json"]
    result = CliRunner().invoke(cli, args)

    error = json.loads(result.stdout)["error"]
    assert result.exit_code == 3, result.output
    (problem,) = error["details"]["errors"]
    assert (problem["field"], problem["did_you_mean"], problem["source"]) == (
        "tournament.windw",
        "tournament.window",
        "--set",
    )
