"""``rcp-ndcg docs api``: one API page per public module, generated from the pinned surface snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from rcp_ndcg.cli.docs import docs_group
from rcp_ndcg.support.api_docs import PUBLIC_MODULES, page_name


def _invoke(*args: str) -> dict:
    result = CliRunner().invoke(docs_group, [*args, "--json"])
    return {"exit_code": result.exit_code, **json.loads(result.stdout)}


def test_docs_api_writes_one_page_per_public_module(tmp_path: Path) -> None:
    payload = _invoke("api", "--out", str(tmp_path))
    assert payload["exit_code"] == 0, payload
    written = sorted(Path(path).name for path in payload["data"]["written"])
    assert written == sorted(page_name(module) for module in PUBLIC_MODULES)
    page = (tmp_path / page_name("rcp_ndcg_core.metric")).read_text(encoding="utf-8")
    assert page.startswith("# `rcp_ndcg_core.metric`")
    assert "## `ndcg`" in page


def test_docs_api_refuses_a_missing_snapshot(tmp_path: Path) -> None:
    payload = _invoke("api", "--out", str(tmp_path), "--snapshot", str(tmp_path / "missing.json"))
    assert payload["exit_code"] != 0
    assert "no surface snapshot" in payload["error"]["message"]
