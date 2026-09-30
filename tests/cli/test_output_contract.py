"""The output contract: one ``rcp-ndcg.cli.v1`` JSON document on stdout for every ``--json`` exit path.

Also the exit codes of the guard, typed warnings, and where tracebacks go (``--log-file``, or stderr with
``-vv``; never by default).
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from rcp_ndcg import __version__
from rcp_ndcg.cli.introspect import walk
from rcp_ndcg.cli.main import cli
from rcp_ndcg.cli.output import CliEnvelope, ErrorCode
from rcp_ndcg.errors import ExitCode


def _leaves_with_json() -> list[str]:
    return sorted(
        path.removeprefix("rcp-ndcg ")
        for path, cmd in walk()
        if not isinstance(cmd, click.Group) and any("--json" in param.opts for param in cmd.params)
    )


def _declared_leaves() -> list[str]:
    return sorted(path.removeprefix("rcp-ndcg ") for path, cmd in walk() if hasattr(cmd, "spec"))


def _one_document(stdout: str) -> dict:
    """Exactly one JSON document: ``json.loads`` refuses trailing data, so a second document fails here."""
    document = json.loads(stdout)
    CliEnvelope.model_validate(document)
    return document


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("RCP_NDCG_RUNS_DIR", str(tmp_path / "runs"))
    return CliRunner()


@pytest.mark.parametrize("path", _leaves_with_json())
def test_a_usage_error_is_one_envelope(runner: CliRunner, path: str) -> None:
    result = runner.invoke(cli, [*path.split(), "--json", "--no-such-option"])

    document = _one_document(result.stdout)
    assert result.exit_code == 2
    assert document["ok"] is False
    assert document["command"] == path
    assert document["error"]["code"] == "USAGE"


@pytest.mark.parametrize("path", _declared_leaves())
def test_a_bare_invocation_is_one_envelope(runner: CliRunner, path: str) -> None:
    """With no arguments a command either succeeds offline or names what is missing: one document either way."""
    result = runner.invoke(cli, [*path.split(), "--json"])

    document = _one_document(result.stdout)
    assert result.exit_code in (0, 2), result.output
    assert document["ok"] is (result.exit_code == 0)


def test_the_success_envelope_carries_data_and_meta(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["schema", "list", "--json"])

    document = _one_document(result.stdout)
    assert result.exit_code == 0
    assert document["schema"] == "rcp-ndcg.cli.v1"
    assert document["data"]["schema"] == "rcp-ndcg.schema-list.v1"
    assert document["meta"]["version"] == __version__
    assert document["meta"]["elapsed_s"] >= 0
    assert "error" not in document


def test_a_typed_failure_keeps_its_exit_code(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["run", "show", "--run", str(tmp_path / "nope"), "--json"])

    document = _one_document(result.stdout)
    assert result.exit_code == 4
    assert document["error"]["code"] == "MISSING_INPUT"
    assert document["error"]["hint"]


def test_every_leaf_is_declared_over_a_request_model() -> None:
    """Only ``mcp serve`` (a server loop, no result) is a plain click command."""
    leaves = {path.removeprefix("rcp-ndcg ") for path, cmd in walk() if not isinstance(cmd, click.Group)}
    assert leaves - set(_declared_leaves()) == {"mcp serve"}


def test_warnings_are_typed(runner: CliRunner, tmp_path: Path) -> None:
    broken = tmp_path / "runs" / "20200101-000000-broken-aaaaaa"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")

    result = runner.invoke(cli, ["run", "list", "--json"])

    document = _one_document(result.stdout)
    assert result.exit_code == 0
    assert [warning["code"] for warning in document["warnings"]] == ["UNREADABLE_RUN"]


@pytest.fixture
def broken_schema_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode():
        raise RuntimeError("boom")

    monkeypatch.setattr("rcp_ndcg.schemas.entries", explode)


@pytest.mark.usefixtures("broken_schema_list")
def test_an_unexpected_failure_is_exit_1_without_a_traceback(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "log" / "cli.log"

    result = runner.invoke(cli, ["--log-file", str(log), "schema", "list"])

    assert result.exit_code == 1
    assert "error [INTERNAL]: RuntimeError: boom" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" in log.read_text(encoding="utf-8")


@pytest.mark.usefixtures("broken_schema_list")
def test_vv_prints_the_traceback(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["-vv", "schema", "list", "--json"])

    assert result.exit_code == 1
    assert _one_document(result.stdout)["error"]["code"] == "INTERNAL"
    assert "Traceback" in result.stderr


def test_env_file_is_loaded_only_when_named(runner: CliRunner, tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "vars.env"
    env_file.write_text(f"RCP_NDCG_RUNS_DIR={tmp_path / 'elsewhere'}\n", encoding="utf-8")
    monkeypatch.delenv("RCP_NDCG_RUNS_DIR")
    monkeypatch.chdir(tmp_path)

    plain = runner.invoke(cli, ["run", "list", "--json"])
    loaded = runner.invoke(cli, ["--env-file", str(env_file), "run", "list", "--json"])

    assert _one_document(plain.stdout)["data"]["runs_dir"] == "runs"
    assert _one_document(loaded.stdout)["data"]["runs_dir"] == str(tmp_path / "elsewhere")


def test_the_envelope_error_codes_are_the_exit_code_names() -> None:
    assert set(typing.get_args(ErrorCode)) == {code.name for code in ExitCode} - {"SUCCESS"}


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout
