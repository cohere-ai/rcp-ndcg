"""Every shipped config validates against the model that reads it, and the paper's engine scripts are valid bash."""

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
    assert {p.name for p in PAPER.iterdir() if p.is_dir()} == {"serve", "retrieval", "rerankers"}
    assert all(_files(kind) for kind in FOLDERS)
    assert not (ROOT / "configs").exists(), "the starter run configs ship in the package (rcp_ndcg.examples)"


@pytest.mark.parametrize("name", judge_names())
def test_shipped_judge_configs_load_by_name(name: str) -> None:
    judge = JudgeConfig.load(name)
    assert judge.temperature is None  # the shipped judges use the server's default sampling, as the paper did
    assert judge.decoding == "json_schema"  # vLLM, SGLang and the OpenAI API constrain answers to a JSON schema


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


@pytest.mark.parametrize("path", sorted((PAPER / "serve").glob("*.sh")), ids=lambda p: p.name)
def test_the_papers_engine_scripts_are_valid_bash_and_name_the_presets_model(path: Path) -> None:
    """Each script pins its image and its weights revision, and serves the model name of the judge preset it is
    named after."""
    import re
    import subprocess

    from tests.runners.shell import assert_shellcheck_clean

    script = path.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    assert_shellcheck_clean(script)
    judge = JudgeConfig.load(path.name.split(".")[0])
    assert f"--served-model-name {judge.model}\n" in script
    image = next(line for line in script.splitlines() if line.startswith("IMAGE="))
    assert ":latest" not in image and ":v" in image  # a pinned release tag
    revisions = re.findall(r"^\s*--revision\b(.*)$", script, re.M)
    assert revisions and all(re.fullmatch(r"\s+[0-9a-f]{40}", rev) for rev in revisions), (
        "every --revision line pins a live 40-hex Hub revision (never a branch, never a second revision line)"
    )
    if judge.context_tokens is not None:
        assert f" {judge.context_tokens}\n" in script  # the served context is the preset's
