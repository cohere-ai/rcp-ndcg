"""Every shipped config validates against the model that reads it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

from rcp_ndcg.examples import run_config_names
from rcp_ndcg.judging import JudgeConfig
from rcp_ndcg.judging.judges import judge_names
from rcp_ndcg.retrieval import RerankerConfig, RetrieverConfig
from rcp_ndcg.runs import RunConfig
from tests.docs._markdown import ROOT

PAPER = ROOT / "experiments" / "paper"
FOLDERS = {"retrieval": PAPER / "retrieval", "reranker": PAPER / "rerankers"}


def _files(kind: str) -> list[Path]:
    return sorted(FOLDERS[kind].glob("*.yaml"))


def test_every_config_folder_has_a_reader() -> None:
    assert {p.name for p in PAPER.iterdir() if p.is_dir()} == {"retrieval", "rerankers"}
    assert all(_files(kind) for kind in FOLDERS)
    assert not (ROOT / "configs").exists(), "the starter run configs ship in the package (rcp_ndcg.examples)"


@pytest.mark.parametrize("name", judge_names())
def test_shipped_judge_configs_load_by_name(name: str) -> None:
    judge = JudgeConfig.load(name)
    assert judge.temperature is None  # the shipped judges use the server's default sampling, as the paper did
    assert judge.decoding == "json_schema"  # vLLM and the OpenAI API constrain answers to a JSON schema


@pytest.mark.parametrize("name", run_config_names())
def test_packaged_run_configs_load_by_name_from_any_directory(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # paths in a run config are relative to the config file
    config = RunConfig.load(name)
    if config.dataset.uri.startswith("jsonl:"):
        assert Path(config.dataset.uri.removeprefix("jsonl:")).is_file()


@pytest.mark.parametrize("path", _files("retrieval"), ids=lambda p: p.stem)
def test_retriever_configs_load(path: Path) -> None:
    TypeAdapter(RetrieverConfig).validate_python(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", _files("reranker"), ids=lambda p: p.stem)
def test_reranker_configs_load(path: Path) -> None:
    TypeAdapter(RerankerConfig).validate_python(yaml.safe_load(path.read_text(encoding="utf-8")))
