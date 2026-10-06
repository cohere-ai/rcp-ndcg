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
    from rcp_ndcg.inference.adapters.embeddings import (
        CohereEmbeddings,
        GeminiEmbeddings,
        OpenAIEmbeddings,
        VoyageEmbeddings,
    )

    shipped = [*ROOT.joinpath("configs").rglob("*.yaml"), *ROOT.joinpath("src/rcp_ndcg/llm/judges").glob("*.yaml")]
    read = {m for path in shipped for m in re.findall(r"api_key_env:\s*([A-Z_]+)", path.read_text(encoding="utf-8"))}
    read |= {cls.API_KEY_ENV[0] for cls in (OpenAIEmbeddings, CohereEmbeddings, VoyageEmbeddings, GeminiEmbeddings)}
    assert "CO_API_KEY" in read
    reported = _reported_variables()
    assert read <= reported, read - reported
    assert "JINA_API_KEY" not in reported


def test_doctor_checks_every_module_the_error_map_names_under_its_extra() -> None:
    """doctor kept its own extras table: it lacked vllm, and proved [hf] without tokenizers."""
    from rcp_ndcg.cli.doctor import _EXTRAS
    from rcp_ndcg.errors import EXTRA_FOR_MODULE

    assert {(module, extra) for extra, modules in _EXTRAS.items() for module in modules} == set(
        EXTRA_FOR_MODULE.items()
    )
    assert set(_EXTRAS["hf"]) == {"huggingface_hub", "tokenizers"}
    assert "vllm" in _EXTRAS and "sglang" not in EXTRA_FOR_MODULE
