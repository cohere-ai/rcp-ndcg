"""The MCP server: tools generated from the CLI's request and result models, structured results, typed errors."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from rcp_ndcg import mcp
from rcp_ndcg.cli.introspect import command_specs
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.pipeline import Pipeline

READ_ONLY = {
    "describe", "schema_show", "data_inspect", "eval_score", "eval_compare", "eval_explain", "calibration_show",
    "run_list", "run_show", "run_status",
}  # fmt: skip


def test_run_start_needs_allow_spend_and_a_budget_unless_the_judge_is_offline(tmp_path: Path) -> None:
    spending = mcp.tool_manifest(allow_spend=True).model_dump(mode="json", by_alias=True)["tools"]
    (start,) = [tool for tool in spending if tool["name"] == "run_start"]
    offline = mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]
    (served,) = [tool for tool in offline if tool["name"] == "run_start"]

    assert start["annotations"] == {"readOnlyHint": False, "destructiveHint": False}
    assert start["inputSchema"]["required"] == ["config", "budget_usd"]
    assert served["inputSchema"]["required"] == ["config"] and "judge: fake" in served["description"]
    assert mcp.call_tool("run_start", {"config": "x.yaml"}, allow_spend=True)["structuredContent"]["code"] == "USAGE"

    fake, real = tmp_path / "fake.yaml", tmp_path / "real.yaml"
    fake.write_text("dataset: jsonl:rows.jsonl\njudge: fake\nsteps: [tournament]\n")
    real.write_text("dataset: jsonl:rows.jsonl\njudge: gpt_oss_120b\nsteps: [tournament]\n")
    assert mcp._offline_run({"config": str(fake)})
    assert mcp._offline_run({"config": str(real), "judge": "fake"})
    assert mcp._offline_run({"config": str(real), "set": ["judge=fake"]})
    assert not mcp._offline_run({"config": str(real)})
    assert not mcp._offline_run({"config": str(fake), "judge_url": "http://x/v1", "judge_model": "m"})
    refused = mcp.call_tool("run_start", {"config": str(real)})["structuredContent"]
    assert refused["code"] == "USAGE" and "--allow-spend" in refused["hint"]


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

    assert set(tools) == READ_ONLY | {"run_cancel", "estimate", "run_start"}
    for tool in tools.values():
        spec = specs[tool["_meta"]["x-cli-command"]]
        assert set(tool["inputSchema"]["properties"]) <= set(spec.request.model_fields)
    assert all(tools[name]["annotations"] == {"readOnlyHint": True, "destructiveHint": False} for name in READ_ONLY)
    assert tools["run_cancel"]["annotations"] == {"readOnlyHint": False, "destructiveHint": True}
    assert tools["estimate"]["inputSchema"]["required"] == ["config"]
    assert tools["describe"]["inputSchema"]["properties"] == {}
    assert tools["run_show"]["inputSchema"]["required"] == ["run"]
    assert tools["run_list"]["outputSchema"]["x-rcp-ndcg-schema"] == "rcp-ndcg.run-list.v1"


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


def test_spending_tools_need_allow_spend_and_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class SpendRequest(BaseModel):
        budget_usd: float | None = None

    specs = command_specs()
    spending = replace(specs["run list"], request=SpendRequest, spends=True, read_only=False, handler=lambda r: {})
    monkeypatch.setattr(mcp, "_specs", lambda: {**specs, "spend": spending})
    monkeypatch.setattr(mcp, "TOOLS", (*mcp.TOOLS, mcp.Tool("spend", "spend")))

    served = {tool.name for tool in mcp.available_tools()}
    offered = mcp.tool_manifest(allow_spend=True).model_dump(mode="json", by_alias=True)["tools"]
    spend = next(tool for tool in offered if tool["name"] == "spend")

    assert "spend" not in served
    assert mcp.call_tool("spend", {"budget_usd": 1.0})["structuredContent"]["code"] == "USAGE"
    assert spend["inputSchema"]["required"] == ["budget_usd"]
    assert mcp.call_tool("spend", {}, allow_spend=True)["structuredContent"]["code"] == "USAGE"
    assert mcp.call_tool("spend", {"budget_usd": 1.0}, allow_spend=True)["isError"] is False


def test_the_builtin_loop_speaks_json_rpc(run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_show", "arguments": {"run": "/x"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(json.dumps(request) for request in requests)))

    mcp._serve_stdio(allow_spend=False)

    responses = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4]
    assert responses[0]["result"]["serverInfo"]["name"] == "rcp-ndcg"
    assert len(responses[1]["result"]["tools"]) == len(READ_ONLY) + 3
    assert responses[2]["result"]["isError"] is True
    assert responses[2]["result"]["structuredContent"]["code"] == "MISSING_INPUT"
    assert responses[3]["error"]["code"] == -32601


def test_the_sdk_server_returns_the_same_results() -> None:
    types = pytest.importorskip("mcp.types")
    import anyio

    server = mcp.sdk_server()
    listed = anyio.run(server.get_request_handler("tools/list").handler, None, None)
    params = types.CallToolRequestParams(name="run_show", arguments={"run": "/x"})
    result = anyio.run(server.get_request_handler("tools/call").handler, None, params)

    wire = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert {tool.name for tool in listed.tools} == READ_ONLY | {"run_cancel", "estimate"}
    assert wire["isError"] is True
    assert wire["structuredContent"] == mcp.call_tool("run_show", {"run": "/x"})["structuredContent"]
