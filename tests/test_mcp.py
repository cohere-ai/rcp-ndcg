"""The MCP server: tools generated from the CLI's request and result models, structured results, typed errors."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from rcp_ndcg import mcp
from rcp_ndcg.cli.introspect import command_specs
from rcp_ndcg.data import SUITES, Rankings
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.pipeline import Pipeline

READ_ONLY = {
    "describe", "schema_show", "data_inspect", "eval_compare", "eval_explain", "calibration_show",
    "run_list", "run_show", "run_status",
}  # fmt: skip


def test_run_start_is_served_with_its_config_as_the_one_required_input() -> None:
    tools = mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]
    (start,) = [tool for tool in tools if tool["name"] == "run_start"]

    assert start["annotations"] == {"readOnlyHint": False, "destructiveHint": False}
    assert start["inputSchema"]["required"] == ["config"]
    assert not any("budget" in name or "usd" in name for name in start["inputSchema"]["properties"])
    missing = mcp.call_tool("run_start", {"config": "/nonexistent/run.yaml"})["structuredContent"]
    assert missing["code"] == "MISSING_INPUT"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mcp")
    rankings = root / "input.jsonl"
    rankings.write_text(json.dumps({"id": "q0", "query": "q", "doc_ids": ["d1", "d2"], "qrels": {"d1": 1}}) + "\n")
    config = RunConfig.model_validate(
        {
            "dataset": f"jsonl:{rankings}",
            "candidates": {"from": "rankings", "rankings": str(rankings)},
            "steps": ["retrieve"],
        }
    )
    pipeline = Pipeline(config, runs_dir=str(root / "runs"))
    pipeline.run()
    return Path(pipeline.layout.root)


def test_the_read_only_tools_mirror_their_commands() -> None:
    manifest = mcp.tool_manifest().model_dump(mode="json", by_alias=True)
    tools = {tool["name"]: tool for tool in manifest["tools"]}
    specs = command_specs()

    assert set(tools) == READ_ONLY | {"eval_score", "run_cancel", "estimate", "run_start"}
    for tool in tools.values():
        spec = specs[tool["_meta"]["x-cli-command"]]
        assert set(tool["inputSchema"]["properties"]) <= set(spec.request.model_fields)
    assert all(tools[name]["annotations"] == {"readOnlyHint": True, "destructiveHint": False} for name in READ_ONLY)
    assert tools["eval_score"]["annotations"] == {"readOnlyHint": False, "destructiveHint": False}
    assert tools["run_cancel"]["annotations"] == {"readOnlyHint": False, "destructiveHint": True}
    assert tools["estimate"]["inputSchema"]["required"] == ["config"]
    assert tools["describe"]["inputSchema"]["properties"] == {}
    assert tools["run_show"]["inputSchema"]["required"] == ["run"]
    assert tools["run_list"]["outputSchema"]["x-rcp-ndcg-schema"] == "rcp-ndcg.run-list.v1"


def test_eval_score_writes_out_so_it_is_not_read_only() -> None:
    """`eval_score` takes `out` and overwrites it with the full report: its readOnlyHint is false, so an agent
    client knows the call leaves an artifact behind."""
    manifest = mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]
    (score,) = [tool for tool in manifest if tool["name"] == "eval_score"]

    assert score["annotations"] == {"readOnlyHint": False, "destructiveHint": False}
    assert "out" in score["inputSchema"]["properties"]


def test_a_call_returns_structured_content(run_dir: Path) -> None:
    listed = mcp.call_tool("run_list", {"runs_dir": str(run_dir.parent)})
    shown = mcp.call_tool("run_show", {"run": str(run_dir)})

    assert listed["isError"] is False
    assert listed["structuredContent"]["runs"][0]["run_id"] == run_dir.name
    assert json.loads(listed["content"][0]["text"]) == listed["structuredContent"]
    assert shown["structuredContent"]["schema"] == "rcp-ndcg.run-summary.v1"
    described = mcp.call_tool("describe")["structuredContent"]
    assert described["schema"] == "rcp-ndcg.command-index.v1"
    flags = {flag["flag"]: flag for flag in described["commands"]["eval score"]["flags"]}
    assert flags["--rankings"]["required"] and flags["--rankings"]["help"]
    assert flags["--per-query"]["type"] == "boolean"


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("run_show", {"run": "/nonexistent/run"}, "MISSING_INPUT"),
        ("run_show", {}, "USAGE"),
        ("run_list", {"limit": "many"}, "USAGE"),
        ("teleport", {}, "USAGE"),
    ],
)
def test_a_failure_is_a_tool_error_carrying_the_error_object(name: str, arguments: dict, code: str) -> None:
    result = mcp.call_tool(name, arguments)

    assert result["isError"] is True
    assert result["structuredContent"]["code"] == code
    assert set(result["structuredContent"]) == {"code", "exit_code", "message", "hint", "retryable", "details"}


def test_a_typed_warning_reaches_the_server_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A typed warning a tool call collected is logged, the MCP analogue of the CLI's stderr line."""
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    repo = SUITES["vidore"].repo
    snapshot = cache / f"datasets--{repo.replace('/', '--')}" / "snapshots" / ("4" * 40)
    (snapshot / "hr__english").mkdir(parents=True)
    pd.DataFrame({"query-id": ["q1"], "corpus-id": ["a"], "score": [1.0], "gain": [1.0], "theta": [1.0]}).to_parquet(
        snapshot / "hr__english/qrels.parquet"
    )
    pd.DataFrame({"query-id": ["q1"], "corpus-ids": [["a"]]}).to_parquet(snapshot / "hr__english/top_ranked.parquet")
    no_exist = cache / f"datasets--{repo.replace('/', '--')}" / ".no_exist" / ("4" * 40) / "hr__english"
    no_exist.mkdir(parents=True)
    (no_exist / "excluded.parquet").touch()
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    with caplog.at_level(logging.WARNING, logger="rcp_ndcg.mcp"):
        mcp.call_tool("eval_score", {"rankings": str(rankings), "suite": "vidore", "subset": "hr__english"})

    assert "UNPINNED_REVISION" in caplog.text


def test_the_builtin_loop_speaks_json_rpc(run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_show", "arguments": {"run": "/x"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(json.dumps(request) for request in requests)))

    mcp._serve_stdio()

    responses = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4]
    assert responses[0]["result"]["serverInfo"]["name"] == "rcp-ndcg"
    assert len(responses[1]["result"]["tools"]) == len(READ_ONLY) + 4
    assert responses[2]["result"]["isError"] is True
    assert responses[2]["result"]["structuredContent"]["code"] == "MISSING_INPUT"
    assert responses[3]["error"]["code"] == -32601


def test_a_malformed_tools_call_is_answered_and_the_loop_goes_on(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A tools/call whose arguments are not a JSON object is an invalid-params error, and the next request is
    still answered: the loop once died on the first malformed request, leaving a client without a server."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "run_list", "arguments": "oops"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "run_list", "arguments": [1, 2]}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_list", "arguments": []}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": "not an object"},
        {"jsonrpc": "2.0", "id": 5, "method": "ping"},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(json.dumps(request) for request in requests)))

    mcp._serve_stdio()

    responses = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4, 5], "every request is answered, in order"
    assert all(response["error"]["code"] == -32602 for response in responses[:4])
    assert responses[4] == {"jsonrpc": "2.0", "id": 5, "result": {}}


def test_a_line_that_is_not_an_object_is_answered_not_fatal(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    requests = ["[1, 2]", {"jsonrpc": "2.0", "id": 1, "method": "ping"}]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(json.dumps(request) for request in requests)))

    mcp._serve_stdio()

    responses = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [response["id"] for response in responses] == [None, 1]
    assert responses[0]["error"]["code"] == -32600
    assert responses[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_a_failure_inside_a_handler_is_answered_not_fatal(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The loop answers -32603 for whatever a handler raises, and answers the next request too."""

    def explode(*_: object) -> dict:
        raise RuntimeError("boom")

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "run_list", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(json.dumps(request) for request in requests)))
    monkeypatch.setattr(mcp, "call_tool", explode)

    mcp._serve_stdio()

    responses = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[0]["error"]["code"] == -32603
    assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


@pytest.mark.parametrize("arguments", ["oops", [1, 2], "", 0, 3.5])
def test_call_tool_refuses_arguments_that_are_not_an_object(arguments: object) -> None:
    """Non-object arguments (falsy ones included) are a typed USAGE tool error, not an exception and not `{}`."""
    result = mcp.call_tool("run_list", arguments)  # type: ignore[arg-type]

    assert result["isError"] is True
    assert result["structuredContent"]["code"] == "USAGE"
    assert "JSON object" in result["structuredContent"]["message"]


def test_the_sdk_server_returns_the_same_results() -> None:
    types = pytest.importorskip("mcp.types")
    import anyio

    server = mcp.sdk_server()
    listed = anyio.run(server.get_request_handler("tools/list").handler, None, None)
    params = types.CallToolRequestParams(name="run_show", arguments={"run": "/x"})
    result = anyio.run(server.get_request_handler("tools/call").handler, None, params)

    wire = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    # The SDK server and the built-in loop answer through the same tool_manifest, so both list the same tools.
    assert {tool.name for tool in listed.tools} == {tool.name for tool in mcp.tool_manifest().tools}
    assert {tool.name for tool in listed.tools} == READ_ONLY | {"eval_score", "run_cancel", "estimate", "run_start"}
    assert wire["isError"] is True
    assert wire["structuredContent"] == mcp.call_tool("run_show", {"run": "/x"})["structuredContent"]


def test_eval_score_takes_the_system_argument(tmp_path: Path) -> None:
    """`eval_score` exposes `system` and scores only the named systems of a multi-system file."""
    manifest = mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]
    (score,) = [tool for tool in manifest if tool["name"] == "eval_score"]
    assert "system" in score["inputSchema"]["properties"]

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({"id": "q1", "query": "q", "doc_ids": ["a", "b"], "qrels": {"a": 1, "b": 0}}) + "\n")
    rankings = tmp_path / "mixed.jsonl"
    Rankings.concat(
        [
            Rankings.from_orders({"q1": ["a", "b"]}, system="good"),
            Rankings.from_orders({"q1": ["x1"]}, system="broken", dataset="zzz"),
        ]
    ).save(rankings)

    result = mcp.call_tool(
        "eval_score",
        {"rankings": str(rankings), "dataset": f"jsonl:{dataset}", "metrics": ["qrel_ndcg"], "system": ["good"]},
    )

    assert result["isError"] is False, result
    assert [row["system"] for row in result["structuredContent"]["summary"]] == ["good"]


def test_eval_score_takes_the_judgements_argument_for_count_ndcg(tmp_path: Path) -> None:
    """`count_ndcg` is offered, so the rubric stores its gains come from must be reachable: the tool takes
    `judgements`, and a call without one gets the CLI's own usage error naming the flag."""
    manifest = mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]
    (score,) = [tool for tool in manifest if tool["name"] == "eval_score"]
    assert "judgements" in score["inputSchema"]["properties"]
    assert "count_ndcg" in score["inputSchema"]["properties"]["metrics"]["items"]["enum"]

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({"id": "q1", "query": "q", "doc_ids": ["a", "b"], "qrels": {"a": 1, "b": 0}}) + "\n")
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a", "b"]}, system="good").save(rankings)

    result = mcp.call_tool(
        "eval_score",
        {"rankings": str(rankings), "dataset": f"jsonl:{dataset}", "metrics": ["count_ndcg"], "judgements": []},
    )

    assert result["isError"] is True
    assert result["structuredContent"]["code"] == "USAGE"
    assert "--judgements" in result["structuredContent"]["message"] + (result["structuredContent"].get("hint") or "")


def test_a_call_refuses_an_argument_the_tool_does_not_take() -> None:
    """An argument the tool's input schema does not list is refused (isError, USAGE), never passed through."""
    result = mcp.call_tool("run_list", {"nope": 1})

    assert result["isError"] is True
    assert result["structuredContent"]["code"] == "USAGE"
    assert "nope" in result["structuredContent"]["message"]
