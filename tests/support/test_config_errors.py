"""Config validation errors name the field, the given value, the expected type, a did-you-mean and their source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.support.config import config_error, validation_problems


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


@pytest.mark.parametrize(
    ("override", "field", "did_you_mean"),
    [
        ("runner={name: slurm, options: {partitoin: gpu}}", "runner.options.partitoin", "runner.options.partition"),
        ("candidates={from: retrieval, retrieval: {kind: bm25, kk: 1}}", "candidates.retrieval.kk", None),
        ("judge={base_url: http://x/v1, model: m, tempreature: 1}", "judge.tempreature", "judge.temperature"),
    ],
)
def test_a_union_members_tag_is_not_part_of_the_field_path(
    config: Path, override: str, field: str, did_you_mean: str | None
) -> None:
    """The path read 'runner.slurm.options.partitoin' and blamed the file; following its hint failed again."""
    problem = _error(config, override).details["errors"][0]
    assert (problem["field"], problem["source"], problem.get("did_you_mean")) == (field, "--set", did_you_mean)


def test_a_runner_option_typo_in_the_file_names_the_key_the_file_holds(config: Path) -> None:
    config.write_text(config.read_text() + "runner: {name: kubernetes, options: {namspace: eval}}\n")
    error = _error(config)
    (problem,) = error.details["errors"]
    assert (problem["field"], problem["source"]) == ("runner.options.namspace", str(config))
    assert "--set runner.options.namspace=..." in (error.cli_hint or "")


def test_a_non_pydantic_error_is_never_read_as_one() -> None:
    """A validator that raised something other than a ``ValidationError`` is re-raised as itself: the reader
    never calls ``errors()`` on it, so a bug inside a validator is not reported as a broken config reader."""
    with pytest.raises(KeyError, match="judge"):
        validation_problems(KeyError("judge"))
    with pytest.raises(KeyError, match="judge"):
        config_error(KeyError("judge"))


def test_a_refused_secret_is_never_echoed() -> None:
    """``details.errors[].input`` (and the message) must not carry a secret: the details reach ``--json``
    output, CI logs and MCP tool results. A refused env token, a credential-bearing URL in a sibling error
    and a credentialed engine list are the three shapes."""
    with pytest.raises(ConfigError) as caught:
        RunConfig.from_data(
            {
                "dataset": {"uri": "x"},
                "runner": {"name": "local", "options": {"env": {"HF_TOKEN": "hf_SECRETVALUE123"}}},
            }
        )
    assert "hf_SECRETVALUE123" not in json.dumps(caught.value.to_dict())

    with pytest.raises(ConfigError) as caught:
        RunConfig.from_data({"dataset": {"uri": "x"}, "judge": {"base_url": "http://user:pw-SECRET@host/v1"}})
    assert "pw-SECRET" not in json.dumps(caught.value.to_dict())

    from rcp_ndcg.support.serve import parse_engines_env

    with pytest.raises(ConfigError) as caught:
        parse_engines_env('{"judge": {"urls": ["https://user:pw-SECRET@gw.example/v1", ""]}}')
    assert "pw-SECRET" not in json.dumps(caught.value.to_dict())


def test_a_secret_under_an_env_or_in_a_command_is_never_echoed() -> None:
    """The redactor's own rule: every value under an ``env`` mapping is a credential by convention, whatever
    its name, and a credential flag's value in a ``command`` is one too. The final review reproduced both
    leaking: a model-level error dumps the whole config (``env.MYVAR``'s value included), the refusal of a
    key-bearing env value echoes the mapping, and the serve command's refusal echoes its argv."""
    with pytest.raises(ConfigError) as caught:
        RunConfig.from_data(
            {"dataset": {"uri": "x"}, "runner": {"name": "local", "options": {"env": {"MYVAR": "hunter2"}}}}
        )
    assert "hunter2" not in json.dumps(caught.value.to_dict())

    with pytest.raises(ConfigError) as caught:
        RunConfig.from_data(
            {
                "dataset": {"uri": "x"},
                "runner": {"name": "local", "options": {"env": {"AZURE_CONN": "AccountKey=SECRETKEYVALUE"}}},
            }
        )
    assert "SECRETKEYVALUE" not in json.dumps(caught.value.to_dict())

    base = {
        "dataset": {"uri": "x"},
        "candidates": {
            "from": "retrieval",
            "retrieval": {
                "kind": "dense",
                "encoder": {"api": "openai_embeddings", "model": "e", "tokenizer": "t", "max_tokens": 512},
            },
        },
        "steps": ["retrieve"],
    }
    for command in (
        ["vllm", "serve", "e", "--api-key=sk-live-XYZ"],
        ["vllm", "serve", "e", "--api-key", "sk-live-XYZ"],
        "vllm serve e --hf-token=sk-live-XYZ",
    ):
        data = dict(base, serve={"encoder": {"image": "registry.example.com/vllm-openai:v1", "command": command}})
        with pytest.raises(ConfigError) as caught:
            RunConfig.from_data(data)
        assert "sk-live-XYZ" not in json.dumps(caught.value.to_dict()), command


def test_a_control_character_refusal_names_the_character_not_the_value() -> None:
    """A mirror URI can carry a presigned signature; the refusal names the character and its position, so
    the value (signature included) does not ride out in the message."""
    from rcp_ndcg.support.resources import no_control_characters

    with pytest.raises(ValueError) as caught:
        no_control_characters("gs://b/x?sig=SECRETSIG\n")
    assert "SECRETSIG" not in str(caught.value)
    assert "U+000A" in str(caught.value)
