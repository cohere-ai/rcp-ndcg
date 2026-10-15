"""The run's fuse step: RRF over the systems a config names, and the refusals that keep a declared fusion
from silently becoming one system's ranking.

A system is a rankings file (a ``retrieval search`` output, another run's rankings, ...) or one system of a
file that holds several (``<file>#<system>``). Fusion reads scores and orders only, so it is modality-free:
a text system and a video system fuse by the same arithmetic (``rcp_ndcg.retrieval.fusion``, the one home of
the rule; ``retrieval fuse`` calls it too).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from rcp_ndcg.data import Rankings, load_rankings
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError
from rcp_ndcg.retrieval import fuse
from rcp_ndcg.runs import Pipeline, RunConfig, RunManifest, RunStatus
from rcp_ndcg.testing import TINY_TOURNAMENT, tiny_rows
from tests.conftest import SESSION_TOKENIZER
from tests.runs.conftest import tiny_config

FUSED_STEPS = ["fuse", "tournament", "rubric", "calibrate", "evaluate"]

_SERVED_RERANK_BUDGET = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192, "use_activation": False}


def _orders() -> dict[str, dict[str, list[str]]]:
    """A forward and a reversed order over the tiny dataset's documents, one ranking per query and system."""
    rows, _ = tiny_rows()
    return {
        "text": {row.id: row.doc_ids[:6] for row in rows},
        "video": {row.id: list(reversed(row.doc_ids[:6])) for row in rows},
    }


def _files(tmp_path: Path, systems: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """One rankings file per system, named after the system; ``{name: location}``."""
    locations = {}
    for name, orders in systems.items():
        path = tmp_path / f"{name}.jsonl"
        Rankings.from_orders(orders, system=name).save(path)
        locations[name] = str(path)
    return locations


def _fused_config(
    data: Path, locations: dict[str, str], *, rrf_k: int = 60, steps: list[str] = FUSED_STEPS, **fields
) -> RunConfig:
    return tiny_config(
        data,
        candidates={"from": "fuse", "fuse": {"systems": locations, "rrf_k": rrf_k}},
        steps=steps,
        **fields,
    )


class TestTheFusedCandidates:
    def test_the_fuse_step_writes_the_fused_candidates_and_records_the_fusion(self, data: Path, tmp_path: Path) -> None:
        systems = _orders()
        locations = _files(tmp_path, systems)
        config = _fused_config(data, locations, rrf_k=60)
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        manifest = pipeline.run()

        assert manifest.status is RunStatus.COMPLETED
        written = load_rankings(pipeline.layout.candidates)
        assert written.systems == ["rrf"]
        expected = fuse(
            [Rankings.from_orders(systems[name], system=name) for name in systems],
            rrf_k=60,
            depth=config.candidates.depth,
        )
        assert written.queries() == expected.queries(system="rrf")

        record = manifest.step("fuse")
        assert record is not None and record.succeeded
        assert record.identity["fusion"] == {"systems": locations, "rrf_k": 60}
        assert record.identity["depth"] == config.candidates.depth
        assert [ref.path for ref in record.inputs] == list(locations.values())
        assert [ref.path for ref in record.outputs] == ["candidates.parquet"]

    def test_the_fusion_is_the_1_over_k_plus_rank_arithmetic_and_consults_no_modality(
        self, data: Path, tmp_path: Path
    ) -> None:
        """A text system and a video system fuse identically: the step reads scores and orders, never documents."""
        systems = _orders()
        locations = _files(tmp_path, systems)
        pipeline = Pipeline(_fused_config(data, locations, rrf_k=10), runs_dir=str(tmp_path / "runs"))
        pipeline.run()

        fused = load_rankings(pipeline.layout.candidates)
        for query, text_order in systems["text"].items():
            video_order = systems["video"][query]
            scores = fused.for_query(query, system="rrf")
            assert set(scores) == set(text_order) | set(video_order)
            for rank, doc in enumerate(text_order, start=1):
                expected = 1 / (10 + rank) + 1 / (10 + video_order.index(doc) + 1)
                assert scores[doc] == pytest.approx(expected)

    def test_a_changed_systems_file_re_runs_the_fuse_step(self, data: Path, tmp_path: Path) -> None:
        systems = _orders()
        locations = _files(tmp_path, systems)
        pipeline = Pipeline(_fused_config(data, locations), runs_dir=str(tmp_path / "runs"))
        pipeline.run()

        changed = {query: list(reversed(docs)) for query, docs in systems["text"].items()}
        Rankings.from_orders(changed, system="text").save(locations["text"])
        resumed = Pipeline.resume(pipeline.layout.root)
        assert resumed.plan()[0] == {"step": "fuse", "status": "would run"}
        assert resumed.run().status is RunStatus.COMPLETED

    def test_the_estimate_of_a_fused_run_reads_the_systems_before_the_step_runs(
        self, data: Path, tmp_path: Path
    ) -> None:
        systems = _orders()
        locations = _files(tmp_path, systems)
        config = _fused_config(data, locations, steps=["fuse", "tournament", "rubric"])
        runs = str(tmp_path / "runs")

        before = Pipeline(config, runs_dir=runs).estimate()
        assert before.calls > 0 and not (tmp_path / "runs").exists()
        pipeline = Pipeline(config, runs_dir=runs)
        pipeline.run()
        after = Pipeline(config, layout=pipeline.layout).estimate()
        assert {stage: entry.calls for stage, entry in before.stages.items()} == {
            stage: entry.calls for stage, entry in after.stages.items()
        }

    def test_a_fused_run_with_a_rerank_step_rescores_the_fused_first_stage(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fuse step writes the first stage the reranker reads (``work/first_stage.parquet``), as retrieve does."""

        def score_by_position(self, examples, *, instruction=None, checkpoint=None):
            for example in examples:
                scores = tuple(float(i) for i in range(len(example.doc_ids)))
                if checkpoint is not None:
                    checkpoint(str(example.id), scores)
            return []

        monkeypatch.setattr("rcp_ndcg.retrieval._api.RerankClient.rerank_many", score_by_position)
        systems = _orders()
        locations = _files(tmp_path, systems)
        config = tiny_config(
            data,
            candidates={
                "from": "fuse",
                "fuse": {"systems": locations, "rrf_k": 60},
                "rerank": {"api": "rerank", "model": "stub", "base_url": "http://stub:8000", **_SERVED_RERANK_BUDGET},
                "depth": 4,
            },
            steps=["fuse", "rerank", "tournament"],
            tournament=TINY_TOURNAMENT.model_copy(update={"window": 4}).model_dump(),
        )
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        assert pipeline.run().status is RunStatus.COMPLETED

        first = load_rankings(Path(pipeline.layout.work) / "first_stage.parquet")
        expected = fuse(
            [Rankings.from_orders(systems[name], system=name) for name in systems],
            rrf_k=60,
            depth=config.candidates.depth,
        )
        assert first.queries() == expected.queries(system="rrf")
        assert load_rankings(pipeline.layout.candidates).systems == ["candidates"], (
            "the reranker's order is what the judging steps read"
        )


class TestTheRefusals:
    def test_a_fusion_of_fewer_than_two_systems_is_refused(self, tmp_path: Path) -> None:
        only = tmp_path / "only.jsonl"
        Rankings.from_orders({"q1": ["d1", "d2"]}, system="only").save(only)
        with pytest.raises(ConfigError, match="at least two systems"):
            RunConfig.from_data(
                {
                    "dataset": "jsonl:rows.jsonl",
                    "judge": "fake",
                    "candidates": {"from": "fuse", "fuse": {"systems": {"only": str(only)}}},
                    "steps": ["fuse"],
                }
            )

    @pytest.mark.parametrize(
        ("candidates", "steps", "message"),
        [
            ({"from": "fuse"}, ["fuse"], "needs candidates.fuse"),
            ({"from": "dataset"}, ["fuse"], "fuse step needs candidates.from: fuse"),
            (
                {"from": "fuse", "fuse": {"systems": {"a": "a.jsonl", "b": "b.jsonl"}}},
                ["tournament"],
                "needs the fuse step",
            ),
            (
                {"from": "fuse", "rankings": "x.jsonl", "fuse": {"systems": {"a": "a.jsonl", "b": "b.jsonl"}}},
                ["fuse"],
                "candidates.rankings",
            ),
            (
                {"from": "rankings", "rankings": "x.jsonl", "fuse": {"systems": {"a": "a.jsonl", "b": "b.jsonl"}}},
                ["retrieve"],
                "set candidates.from: fuse, or drop candidates.fuse",
            ),
        ],
    )
    def test_the_fuse_step_and_its_block_need_each_other(
        self, candidates: dict, steps: list[str], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            RunConfig.model_validate(
                {"dataset": "jsonl:rows.jsonl", "judge": "fake", "candidates": candidates, "steps": steps}
            )

    def test_a_fusion_whose_system_rankings_are_missing_is_refused_and_writes_no_candidates(
        self, data: Path, tmp_path: Path
    ) -> None:
        locations = _files(tmp_path, _orders())
        locations["video"] = str(tmp_path / "not-retrieved.jsonl")
        pipeline = Pipeline(_fused_config(data, locations), runs_dir=str(tmp_path / "runs"))

        with pytest.raises(MissingInputError) as caught:
            pipeline.run()
        assert "video" in str(caught.value) and "not-retrieved.jsonl" in str(caught.value)
        assert not Path(pipeline.layout.candidates).exists(), "nothing is written: a system that cannot fuse is refused"
        assert RunManifest.load(pipeline.layout.root).status is RunStatus.FAILED

    def test_a_system_that_was_not_retrieved_into_its_file_is_refused(self, data: Path, tmp_path: Path) -> None:
        systems = _orders()
        locations = _files(tmp_path, systems)
        both = tmp_path / "both.jsonl"
        Rankings.concat(
            [
                Rankings.from_orders(systems["text"], system="text"),
                Rankings.from_orders(systems["video"], system="video"),
            ]
        ).save(both)
        pipeline = Pipeline(
            _fused_config(data, {"picked": f"{both}#other", "video": locations["video"]}),
            runs_dir=str(tmp_path / "runs"),
        )

        with pytest.raises(DataError) as caught:
            pipeline.run()
        assert "'other'" in str(caught.value) and "not retrieved" in str(caught.value)
        assert not Path(pipeline.layout.candidates).exists()

    def test_a_multi_system_file_without_a_selector_is_refused_not_fused_whole(
        self, data: Path, tmp_path: Path
    ) -> None:
        systems = _orders()
        both = tmp_path / "both.jsonl"
        Rankings.concat(
            [
                Rankings.from_orders(systems["text"], system="text"),
                Rankings.from_orders(systems["video"], system="video"),
            ]
        ).save(both)
        pipeline = Pipeline(
            _fused_config(data, {"text": str(both), "video": _files(tmp_path, systems)["video"]}),
            runs_dir=str(tmp_path / "runs"),
        )

        with pytest.raises(DataError, match="holds 2 systems"):
            pipeline.run()
        assert not Path(pipeline.layout.candidates).exists()


def test_run_start_rrf_k_sets_the_fusions_constant(data: Path, tmp_path: Path) -> None:
    systems = _orders()
    locations = _files(tmp_path, systems)
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(_fused_config(data, locations).resolved()), encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["run", "start", str(config), "--rrf-k", "10", "--runs-dir", str(tmp_path / "runs"), "--json"]
    )
    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.stdout)["data"]["run_dir"])
    recorded = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert recorded["config"]["candidates"]["fuse"]["rrf_k"] == 10
    expected = fuse(
        [Rankings.from_orders(systems[name], system=name) for name in systems],
        rrf_k=10,
        depth=150,
    )
    assert load_rankings(run_dir / "candidates.parquet").queries() == expected.queries(system="rrf")


def test_run_start_rrf_k_without_a_fused_config_is_a_usage_error(data: Path, tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")

    result = CliRunner().invoke(cli, ["run", "start", str(config), "--rrf-k", "10", "--json"])

    assert result.exit_code == 2, result.output
    assert "candidates.fuse" in result.stdout
