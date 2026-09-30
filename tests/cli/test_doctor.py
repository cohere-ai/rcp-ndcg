"""``rcp-ndcg doctor``: the credential variables it reports are the ones the shipped configs and backends read."""

from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner

from rcp_ndcg.cli.doctor import doctor_cmd

ROOT = Path(__file__).resolve().parents[2]


def _reported_variables() -> set[str]:
    result = CliRunner().invoke(doctor_cmd, ["--json"])
    assert result.exit_code == 0, result.output
    checks = json.loads(result.stdout)["data"]["checks"]
    return {check["name"][1:] for check in checks if check["name"].startswith("$")}


def test_doctor_reports_the_variables_the_shipped_configs_read() -> None:
    from rcp_ndcg.retrieval.api_dense import VENDORS

    shipped = [*ROOT.joinpath("configs").rglob("*.yaml"), *ROOT.joinpath("src/rcp_ndcg/llm/judges").glob("*.yaml")]
    read = {m for path in shipped for m in re.findall(r"api_key_env:\s*([A-Z_]+)", path.read_text(encoding="utf-8"))}
    read |= {vendor.api_key_env[0] for vendor in VENDORS.values()}
    assert "CO_API_KEY" in read
    reported = _reported_variables()
    assert read <= reported, read - reported
    assert "JINA_API_KEY" not in reported
