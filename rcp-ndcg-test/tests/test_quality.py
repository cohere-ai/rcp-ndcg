"""The T3 quality stage on CPU: the task matrix, the served command lines, the gates, the run and the golden capture.

The served argv are parsed by the product's own click tree (a flag the CLI does not have fails here, not on the
node), the configs by the product's own retriever/reranker validators, and the stage runs end to end with a
fake subprocess runner that writes what the real commands write.  The golden-replay proxy runs against the stub
engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.quality import (
    QUALITY_TOLERANCE,
    TASK_MATRIX,
    QualityTask,
    RecordingProxy,
    _parser,
    comparison_rows,
    reference_argv,
    reference_scores,
    reranker_config,
    retriever_config,
    run_quality,
    served_commands,
    tasks_for,
)
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from tests.conftest import RECIPES, TOKENIZER, start_stub

REV = "4517f2cb9e342479725bf0931a329998b4d35038"


def _task(view: str, *, golden: bool = False, subset: str = "NanoNQRetrieval") -> QualityTask:
    return QualityTask(model="fixture", suite="nanobeir", subset=subset, view=view, revision=REV, golden=golden)


def _parse_with_the_product_cli(argv: list[str]) -> None:
    """Resolve ``rcp-ndcg <group> <command> <options>`` through the product's click tree; unknown flags fail."""
    import click

    from rcp_ndcg.cli.main import cli

    command: click.Command = cli
    args = argv[1:]
    name = "rcp-ndcg"
    while isinstance(command, click.Group):
        context = command.make_context(name, list(args), resilient_parsing=False)
        # The served argv carry no group option, so the remaining tokens start with the subcommand's name.
        name, sub, args = command.resolve_command(context, list(args))
        assert sub is not None, argv
        command = sub
    command.make_context(name, list(args))


@pytest.mark.parametrize(
    ("recipe_id", "view"),
    [("fixture-embed", "retrieval"), ("fixture-multi-vector", "retrieval"), ("fixture-rerank-pointwise", "reranking")],
)
def test_every_served_command_is_a_valid_product_command_line(tmp_path: Path, recipe_id: str, view: str) -> None:
    """The served path is the product's CLI with the CLI's own flags (the earlier argv used ``--encoder-url``,
    ``--rerank-url`` and a ``jsonl:`` dataset no step wrote -- none of which the CLI has)."""
    recipe = load_recipe(RECIPES / recipe_id)
    commands = served_commands(_task(view), recipe, "http://127.0.0.1:9", tmp_path)
    assert commands[-1][:3] == ["rcp-ndcg", "eval", "score"]
    for argv in commands:
        if argv[0] == "rcp-ndcg":
            _parse_with_the_product_cli(argv)
        else:
            assert argv[1:3] == ["-m", "rcp_ndcg_test.quality"]
            _parser().parse_args(argv[3:])
    _parser().parse_args(reference_argv(_task(view), recipe, tmp_path, "python")[3:])


def test_the_configs_are_the_recipes_endpoints_validated_by_the_product(tmp_path: Path) -> None:
    dense = retriever_config(load_recipe(RECIPES / "fixture-embed"), "http://127.0.0.1:9")
    assert dense["kind"] == "dense" and dense["encoder"]["base_url"] == "http://127.0.0.1:9/v1"
    late = retriever_config(load_recipe(RECIPES / "fixture-multi-vector"), "http://127.0.0.1:9")
    assert late["kind"] == "late_interaction" and late["encoder"]["api"] == "vllm_pooling"
    reranker = reranker_config(load_recipe(RECIPES / "fixture-rerank-pointwise"), "http://127.0.0.1:9")
    assert reranker["api"] == "rerank" and reranker["model"] == "fixture-rerank-pointwise"
    with pytest.raises(HarnessError, match="needs a reranker"):
        reranker_config(load_recipe(RECIPES / "fixture-embed"), "http://127.0.0.1:9")


def test_the_task_matrix_covers_every_recipe_once_per_view() -> None:
    listed = [model for row in TASK_MATRIX for model in row["models"]]
    recipe_ids = sorted(path.name for path in default_recipes_root().iterdir() if (path / "recipe.yaml").is_file())
    assert sorted(set(listed)) == recipe_ids
    assert len(listed) == len(set(listed)) + 1, "topk-embed-v1-small: visual documents and late interaction, text"
    assert QUALITY_TOLERANCE == 0.005


def test_a_recipes_tasks_are_every_subset_at_the_pinned_commits() -> None:
    from rcp_ndcg_test.observe.requests import PINNED_DATASET_COMMITS

    reranker = tasks_for("qwen3-reranker-0.6b")
    assert len(reranker) == 13 + 12 + 2 and {task.view for task in reranker} == {"reranking"}
    assert {task.revision for task in reranker} <= set(PINNED_DATASET_COMMITS.values())
    assert [task.key for task in reranker if task.golden] == ["nanobeir/NanoArguAnaRetrieval"]
    topk = tasks_for("topk-embed-v1-small")
    assert len(topk) == 8 + 13 + 12 and {task.view for task in topk} == {"retrieval"}
    assert sorted(task.suite for task in topk if task.golden) == ["nanobeir", "vidore"]
    with pytest.raises(HarnessError, match="not in the T3 task matrix"):
        tasks_for("no-such-recipe")


def test_the_gates_cover_the_reference_and_the_papers_numbers() -> None:
    ok = comparison_rows({"a": 0.4120}, {"a": 0.4100}, paper={"a": 0.4095}, published={"a": 0.45})
    assert ok["passed"] is True and "unexplained deviation" in ok["rows"][0]["deviation_note"]
    off_paper = comparison_rows({"a": 0.4120}, {"a": 0.4100}, paper={"a": 0.40})
    assert off_paper["passed"] is False, "a paper model more than 0.5 points off the paper's number fails"
    assert comparison_rows({"a": 0.40}, {"a": 0.41})["passed"] is False
    missing = comparison_rows({"a": 0.40, "b": 0.5}, {"a": 0.40})
    assert missing["passed"] is False, "a subset without its reference is never a pass"
    noted = comparison_rows({"a": 0.41}, {"a": 0.41}, published={"a": 0.45}, deviation_note={"a": "pool differs"})
    assert noted["rows"][0]["deviation_note"] == "pool differs"


def test_the_reference_scores_are_read_from_an_mteb_task_result() -> None:
    result = {
        "task_name": "NanoNQ",
        "scores": {"test": [{"hf_subset": "default", "ndcg_at_10": 0.51, "ndcg_float_at_10": 0.47}]},
    }
    assert reference_scores(result) == {"qrel_ndcg": 0.51, "rcp_ndcg": 0.47}
    retrieval_view = {"task_name": "NanoNQ", "scores": {"test": [{"ndcg_at_10": 0.51}]}}
    assert reference_scores(retrieval_view) == {"qrel_ndcg": 0.51}


def _eval_report(subset: str, values: dict[str, float]) -> dict[str, Any]:
    return {
        "schema": "rcp-ndcg.eval-report.v1",
        "protocol": {"name": "nanobeir", "ties": "group_mean"},
        "gains_source": "dataset",
        "metrics": sorted(values),
        "k": [10],
        "per_query": [],
        "per_dataset": [
            {"system": "s", "dataset": subset, "metric": metric, "k": 10, "value": value, "num_queries": 50}
            for metric, value in values.items()
        ],
        "summary": [],
    }


class FakeRunner:
    """Stands in for the subprocesses: writes ``scores.json`` for ``eval score`` and ``reference.json`` for the
    mteb run, and -- for the golden task -- sends one request through the endpoint its config names."""

    def __init__(self, served: dict[str, float], reference: dict[str, float], *, fail: str | None = None) -> None:
        self.served, self.reference, self.fail = served, reference, fail
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_: Any) -> Any:
        import subprocess

        self.calls.append(argv)
        if self.fail and self.fail in argv:
            return subprocess.CompletedProcess(argv, 3, "", "boom")
        if argv[:3] == ["rcp-ndcg", "retrieval", "index"]:
            config = yaml.safe_load(Path(argv[argv.index("--retriever") + 1]).read_text(encoding="utf-8"))
            base = config["encoder"]["base_url"]
            httpx.post(f"{base}/embeddings", json={"model": "fixture-embed", "input": ["doc: x [END]"]}, timeout=30)
        if argv[:3] == ["rcp-ndcg", "eval", "score"]:
            subset = argv[argv.index("--subset") + 1]
            out = Path(argv[argv.index("--out") + 1])
            out.write_text(json.dumps(_eval_report(subset, self.served)), encoding="utf-8")
        if "reference" in argv:
            out = Path(argv[argv.index("--out") + 1])
            row = {"ndcg_at_10": self.reference["qrel_ndcg"], "ndcg_float_at_10": self.reference.get("rcp_ndcg")}
            out.write_text(json.dumps({"task_name": "t", "scores": {"test": [row]}}), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_the_stage_runs_every_task_gates_it_and_captures_the_golden_replay(tmp_path: Path) -> None:
    """End to end on CPU: the reranking view gates both metrics; the golden task's served run goes through the
    recording proxy into an observation corpus the product's reader accepts."""
    from rcp_ndcg_test.corpus import integrity_mismatches, load_corpus

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        tasks = [_task("retrieval", golden=True), _task("retrieval", subset="NanoFEVERRetrieval")]
        runner = FakeRunner({"qrel_ndcg": 0.51, "rcp_ndcg": 0.47}, {"qrel_ndcg": 0.512})
        document = run_quality(
            recipe,
            engine_url=engine.base_url,
            tasks=tasks,
            work_dir=tmp_path / "q",
            reference_python="py",
            runner=runner,
        )
    finally:
        engine.stop()
    assert document["passed"] is True, document
    assert document["comparisons"]["rcp_ndcg"]["gated"] is False, "mteb's retrieval view reports no RCP-nDCG"
    assert document["comparisons"]["qrel_ndcg"]["n_subsets"] == 2
    golden = load_corpus(document["golden_replay"][0])
    assert integrity_mismatches(golden) == [] and len(golden.records) == 1
    assert golden.records[0]["request"]["path"] == "/v1/embeddings"
    assert golden.records[0]["response"]["status"] == 200
    assert (tmp_path / "q" / "QUALITY.md").read_text(encoding="utf-8").count("Verdict:") == 2


def test_the_stage_fails_a_deviation_and_a_failing_command(tmp_path: Path) -> None:
    recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    tasks = [_task("reranking")]
    off = run_quality(
        recipe,
        engine_url="http://127.0.0.1:9",
        tasks=tasks,
        work_dir=tmp_path / "off",
        reference_python="py",
        runner=FakeRunner({"qrel_ndcg": 0.51, "rcp_ndcg": 0.47}, {"qrel_ndcg": 0.51, "rcp_ndcg": 0.46}),
    )
    assert off["passed"] is False and off["comparisons"]["rcp_ndcg"]["passed"] is False
    broken = run_quality(
        recipe,
        engine_url="http://127.0.0.1:9",
        tasks=tasks,
        work_dir=tmp_path / "broken",
        reference_python="py",
        runner=FakeRunner({"qrel_ndcg": 0.5}, {"qrel_ndcg": 0.5}, fail="rerank"),
    )
    assert broken["passed"] is False and "exited 3" in broken["errors"]["nanobeir/NanoNQRetrieval"]


def test_the_recording_proxy_forwards_and_records_the_raw_bytes(tmp_path: Path) -> None:
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        with RecordingProxy(engine.base_url) as proxy:
            direct = httpx.post(f"{engine.base_url}/v1/embeddings", json={"model": "m", "input": ["a b"]})
            proxied = httpx.post(f"{proxy.url}/v1/embeddings", json={"model": "m", "input": ["a b"]})
    finally:
        engine.stop()
    assert proxied.status_code == direct.status_code == 200 and proxied.content == direct.content
    assert len(proxy.exchanges) == 1 and proxy.exchanges[0]["status"] == 200


def test_the_wave_runs_the_quality_stage_and_fails_a_recipe_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_wave --quality`` calls the stage with the served engine's URL and the recipe's tasks; a failed stage
    fails the recipe, and a recipe outside the task matrix fails it with the reason."""
    import sys

    from rcp_ndcg_test import quality as t3
    from rcp_ndcg_test.jobs.run_wave import run_wave

    from tests.conftest import sample_pairs, write_pairs

    pairs = tmp_path / "pairs"
    pairs.mkdir()
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    stub = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}"
    kwargs: dict[str, Any] = {
        "recipe_ids": ["fixture-embed"],
        "recipes_root": RECIPES,
        "gpus": 1,
        "pairs_dir": pairs,
        "reference_python": sys.executable,
        "vllm_cmd": stub,
        "port_base": 0,
        "quality": True,
    }
    outside = run_wave(out_dir=tmp_path / "outside", **kwargs)["recipes"][0]
    assert outside["state"] == "failed" and "not in the T3 task matrix" in outside["steps"]["quality"]["error"]

    seen: dict[str, Any] = {}

    def fake_run(recipe: Any, **options: Any) -> dict[str, Any]:
        seen.update(options, recipe=recipe.id)
        return {"passed": False, "errors": {"nanobeir/NanoNQRetrieval": "0.9 points off"}}

    monkeypatch.setattr(t3, "tasks_for", lambda recipe_id: [_task("retrieval")])
    monkeypatch.setattr(t3, "run_quality", fake_run)
    failed = run_wave(out_dir=tmp_path / "failed", **kwargs)["recipes"][0]
    assert failed["state"] == "failed" and failed["steps"]["quality"]["state"] == "failed"
    assert seen["recipe"] == "fixture-embed" and seen["engine_url"].startswith("http://127.0.0.1:")
    assert seen["golden_manifest"]["engine"]["serve_argv"], "the golden corpus carries the wave's engine block"
