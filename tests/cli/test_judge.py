"""``rcp-ndcg judge tournament|rubric``: the offline judge end to end, and the estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.judge import judge_group
from rcp_ndcg.testing import tiny_rows


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    rows, _ = tiny_rows()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps({"query_id": r.id, "query": r.query, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    return f"jsonl:{path}"


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(judge_group, list(args))
    return result.exit_code, result.stdout


def test_the_offline_judge_fills_a_store_and_a_rerun_asks_nothing(dataset: str, tmp_path: Path) -> None:
    store = tmp_path / "store"
    args = ("tournament", "--dataset", dataset, "--judge", "fake", "--set", "schedule.window=4", "--out", str(store))

    code, first = _invoke(*args, "--json")
    _, again = _invoke(*args, "--json")

    assert code == 0, first
    first_data, again_data = json.loads(first)["data"], json.loads(again)["data"]
    assert first_data["schema"] == "rcp-ndcg.judge-report.v1" and first_data["mode"] == "judged"
    assert first_data["judgements"] > 0 and first_data["stored"] == 0
    assert again_data["stored"] == first_data["judgements"]
    assert (store / "identity.json").is_file()


def test_the_estimate_labels_token_counts_as_approximations(dataset: str, tmp_path: Path) -> None:
    code, text = _invoke("rubric", "--dataset", dataset, "--judge", "fake", "--out", str(tmp_path / "s"), "--estimate")

    assert code == 0, text
    assert "~" in text and "input tokens approximate" in text
    assert not (tmp_path / "s").exists()


def test_a_shipped_judge_is_named_from_any_directory(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "s")
    code, text = _invoke("tournament", "--dataset", dataset, "--judge", "gpt5_hosted", "--out", out, "--estimate")
    assert code == 0, text
    code, text = _invoke("tournament", "--dataset", dataset, "--judge", "no_such_judge", "--out", out, "--json")
    assert code == 4 and "gpt_oss_120b" in json.loads(text)["error"]["hint"]


def test_the_estimate_of_docs_prices_only_those_documents(dataset: str, tmp_path: Path) -> None:
    args = ("rubric", "--dataset", dataset, "--judge", "fake", "--docs", "q2:q2-d10", "--docs", "q2:q2-d11")
    args += ("--out", str(tmp_path / "s"), "--json")

    code, priced = _invoke(*args, "--estimate")
    _, judged = _invoke(*args)

    assert code == 0, priced
    assert json.loads(priced)["data"]["estimate"]["calls"] == json.loads(judged)["data"]["usage"]["requests"]


def test_there_is_no_dry_run_of_a_judging_stage(dataset: str, tmp_path: Path) -> None:
    code, _ = _invoke("rubric", "--dataset", dataset, "--judge", "fake", "--out", str(tmp_path / "s"), "--dry-run")
    assert code == 2


def test_restating_a_default_schedule_field_reuses_the_store(dataset: str, tmp_path: Path) -> None:
    args = ("rubric", "--dataset", dataset, "--judge", "fake", "--out", str(tmp_path / "s"), "--json")
    code, first = _invoke(*args)
    assert code == 0, first

    code, again = _invoke(*args, "--set", "schedule.window=10")

    assert code == 0, again
    assert json.loads(again)["data"]["usage"]["requests"] == 0


def test_an_estimate_into_a_store_of_another_identity_gives_the_judges_refusal(tmp_path: Path) -> None:
    from rcp_ndcg.cli.main import cli
    from rcp_ndcg.examples import tiny

    rows = tiny() / "rows.jsonl"
    store = tmp_path / "store"
    base = ["judge", "rubric", "--dataset", f"jsonl:{rows}", "--judge", "fake", "--out", str(store), "--json",
            "--set", "schedule.window=4", "--set", "schedule.windows_per_query=4",
            "--set", "schedule.random_windows=2"]  # fmt: skip
    assert CliRunner().invoke(cli, base).exit_code == 0
    identity = (store / "identity.json").read_text(encoding="utf-8")

    other = [*base, "--seed", "7"]
    real = CliRunner().invoke(cli, other)
    estimate = CliRunner().invoke(cli, [*other, "--estimate"])

    assert real.exit_code == estimate.exit_code == 11
    assert "--force" in json.loads(estimate.stdout)["error"]["hint"]
    assert (store / "identity.json").read_text(encoding="utf-8") == identity
