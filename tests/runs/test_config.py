"""The run config: typed sections, unknown keys refused, ``extends:``, ``key=value`` overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.llm import JudgeConfig, RubricSchedule, TournamentSchedule
from rcp_ndcg.runs import RunConfig
from rcp_ndcg.support.config import apply_overrides, load_config

DATASET = "jsonl:rows.jsonl"


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestComposition:
    def test_extends_merges_deeply_relative_to_the_extending_file(self, tmp_path: Path) -> None:
        (tmp_path / "base").mkdir()
        _write(tmp_path / "base" / "base.yaml", {"dataset": DATASET, "evaluation": {"k": 5, "systems": {"a": "a"}}})
        child = _write(tmp_path / "child.yaml", {"extends": "base/base.yaml", "evaluation": {"k": 20}})
        assert load_config(child) == {"dataset": DATASET, "evaluation": {"k": 20, "systems": {"a": "a"}}}

    def test_an_extends_cycle_is_refused(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
        _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
        with pytest.raises(ConfigError, match="extends itself"):
            load_config(tmp_path / "a.yaml")

    def test_overrides_parse_their_values_as_yaml(self) -> None:
        original = {"a": {"b": 1}}
        merged = apply_overrides(original, ["a.b=2", "a.c=null", "d.e=[1, 2]", "f=text"])
        assert merged == {"a": {"b": 2, "c": None}, "d": {"e": [1, 2]}, "f": "text"}
        assert original == {"a": {"b": 1}}  # the input (e.g. a manifest's recorded config) is left alone
        with pytest.raises(ConfigError, match="not key=value"):
            apply_overrides({}, ["novalue"])
        with pytest.raises(ConfigError, match="is not a mapping"):
            apply_overrides({"a": 1}, ["a.b=2"])


class TestValidation:
    def test_an_unknown_key_is_refused_at_load(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "run.yaml", {"dataset": DATASET, "num_criteria": 5})
        with pytest.raises(ConfigError, match="num_criteria"):
            RunConfig.load(path)
        with pytest.raises(ConfigError, match="depht"):
            RunConfig.load(_write(tmp_path / "r2.yaml", {"dataset": DATASET}), overrides=["candidates.depht=10"])

    def test_steps_run_in_pipeline_order(self) -> None:
        config = RunConfig(dataset=DATASET, judge="fake", steps=["evaluate", "rubric", "calibrate", "tournament"])
        assert config.ordered_steps == ["tournament", "rubric", "calibrate", "evaluate"]

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"steps": ["retrieve"]}, "retrieve step needs"),
            ({"steps": ["rerank"]}, "rerank step needs"),
            ({"steps": ["tournament"]}, "need a judge"),
            ({"candidates": {"from": "rankings"}}, "needs candidates.rankings"),
            ({"candidates": {"from": "retrieval"}}, "needs candidates.retrieval"),
            ({"steps": ["teleport"]}, "teleport"),
        ],
    )
    def test_steps_need_their_inputs(self, fields: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            RunConfig.model_validate({"dataset": DATASET, **fields})

    def test_schedules_default_to_none_and_validate_when_given(self) -> None:
        config = RunConfig(dataset=DATASET, judge="fake", tournament={"window": 5}, rubric={"windows_per_query": 200})
        assert isinstance(config.tournament, TournamentSchedule) and config.tournament.window == 5
        assert isinstance(config.rubric, RubricSchedule) and config.rubric.windows_per_query == 200
        assert RunConfig(dataset=DATASET, judge="fake").tournament is None


class TestJudge:
    def test_fake_is_the_offline_judge_with_the_run_seed(self) -> None:
        judge = RunConfig(dataset=DATASET, judge="fake", seed=3).judge_config()
        assert judge.is_fake and judge == JudgeConfig.fake(3)

    def test_a_path_and_an_inline_mapping(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "judge.yaml", {"base_url": "http://h/v1", "model": "m"})
        assert RunConfig(dataset=DATASET, judge=str(path)).judge_config().model == "m"
        inline = RunConfig.model_validate({"dataset": DATASET, "judge": {"base_url": "http://h/v1", "model": "n"}})
        assert inline.judge_config().model == "n"

    def test_a_judge_path_takes_overrides_of_its_fields(self, tmp_path: Path) -> None:
        """``--set judge.base_url=...`` on a config whose judge is a file once failed on the string."""
        (tmp_path / "judges").mkdir()
        _write(tmp_path / "judges" / "j.yaml", {"base_url": "http://h/v1", "model": "m", "concurrency": 4})
        path = _write(tmp_path / "run.yaml", {"dataset": DATASET, "judge": "judges/j.yaml"})

        config = RunConfig.load(path, overrides=["judge.base_url=http://other:8000/v1"])

        judge = config.judge_config()
        assert (judge.base_url, judge.model, judge.concurrency) == ("http://other:8000/v1", "m", 4)
        assert RunConfig.load(path).judge == str(tmp_path / "judges" / "j.yaml"), "relative to the config file"

    def test_paths_in_a_config_are_relative_to_the_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config once read its dataset relative to the working directory, so it ran only from one place."""
        (tmp_path / "cfg").mkdir()
        path = _write(
            tmp_path / "cfg" / "run.yaml",
            {
                "dataset": "jsonl:../data/rows.jsonl",
                "candidates": {"from": "rankings", "rankings": "pools.parquet"},
                "evaluation": {"systems": {"mine": "systems.jsonl#bm25"}},
                "steps": ["retrieve"],
            },
        )
        monkeypatch.chdir(tmp_path / "cfg")
        here = RunConfig.load("run.yaml")
        monkeypatch.chdir("/")
        anywhere = RunConfig.load(path)
        assert here == anywhere
        assert anywhere.dataset.uri == f"jsonl:{tmp_path / 'data' / 'rows.jsonl'}"
        assert anywhere.candidates.rankings == str(tmp_path / "cfg" / "pools.parquet")
        assert anywhere.evaluation.systems == {"mine": f"{tmp_path / 'cfg' / 'systems.jsonl'}#bm25"}
        overridden = RunConfig.load(path, overrides=["dataset=jsonl:other.jsonl"])
        assert overridden.dataset.uri == "jsonl:other.jsonl", "--set paths stay relative to the working directory"

    def test_a_shipped_judge_by_name_takes_overrides_from_any_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(tmp_path / "run.yaml", {"dataset": DATASET, "judge": "gpt_oss_120b"})
        assert RunConfig.load(path).judge_config().model == "gpt-oss-120b"
        judge = RunConfig.load(path, overrides=["judge.concurrency=3"]).judge_config()
        assert (judge.model, judge.concurrency) == ("gpt-oss-120b", 3)

    def test_no_judge_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match="no judge"):
            RunConfig(dataset=DATASET, steps=["evaluate"]).judge_config()


def test_the_shipped_configs_validate() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "configs" / "run").glob("*.yaml")):
        config = RunConfig.load(path)
        assert isinstance(config.judge, str) and isinstance(config.judge_config(), JudgeConfig)


class TestRunnerOptions:
    @pytest.mark.parametrize("name", ["local", "slurm", "kubernetes"])
    def test_the_typed_options_are_the_runners_own_model(self, name: str) -> None:
        # One model per runner: the config's options are exactly what the runner validates.
        from rcp_ndcg.runners import get_runner

        config = RunConfig.model_validate({"dataset": DATASET, "steps": ["evaluate"], "runner": {"name": name}})
        runner = get_runner(name)
        assert type(config.runner.options) is type(runner.options)

    def test_a_misspelt_runner_option_fails_at_load(self) -> None:
        with pytest.raises(ValidationError, match="partiton"):
            RunConfig.model_validate({"dataset": DATASET, "runner": {"name": "slurm", "options": {"partiton": "g"}}})
        mine = {"name": "mine", "options": {"x": 1}}
        plugin = RunConfig.model_validate({"dataset": DATASET, "steps": ["evaluate"], "runner": mine})
        assert plugin.runner.option_values() == {"x": 1}
