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
            json.dumps({"query_id": r.id, "query": r.text, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
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
    hint = json.loads(text)["error"]["hint"]
    assert code == 4 and "gpt5_hosted" in hint and "recipe:" in hint


def test_a_judge_recipe_resolves_through_the_recipe_path() -> None:
    """``--judge <recipe-id>`` and ``--judge recipe:<id>`` are the same path as ``reranker: recipe:<id>``."""
    for source in ("gpt-oss-120b", "recipe:gpt-oss-120b"):
        args = ("check", "--judge", source, "--set", "judge.base_url=fake://seed/7", "--json")
        code, text = _invoke(*args)
        assert code == 0, text
        data = json.loads(text)["data"]
        assert data["schema"] == "rcp-ndcg.judge-check-report.v1"
        assert data["ok"] is True and data["recipe"] == "gpt-oss-120b"
        assert all(check["schema_sent"] and check["parsed"] for check in data["checks"])


def test_judge_check_probes_the_offline_judge() -> None:
    code, text = _invoke("check", "--judge", "fake", "--json")
    assert code == 0, text
    data = json.loads(text)["data"]
    assert data["ok"] is True and data["model"] == "fake"
    assert [check["stage"] for check in data["checks"]] == ["tournament", "rubric"]
    assert all(not check["schema_sent"] for check in data["checks"])


def test_judge_check_refuses_a_non_judge_override() -> None:
    code, text = _invoke("check", "--judge", "fake", "--set", "schedule.window=4", "--json")
    assert code == 2 and "only judge." in text


def test_the_estimate_of_docs_counts_only_those_documents(dataset: str, tmp_path: Path) -> None:
    args = ("rubric", "--dataset", dataset, "--judge", "fake", "--docs", "q2:q2-d10", "--docs", "q2:q2-d11")
    args += ("--out", str(tmp_path / "s"), "--json")

    code, projected = _invoke(*args, "--estimate")
    _, judged = _invoke(*args)

    assert code == 0, projected
    assert json.loads(projected)["data"]["estimate"]["calls"] == json.loads(judged)["data"]["usage"]["requests"]


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
            "--set", "schedule.window=4", "--set", "schedule.placements_per_doc=2.0",
            "--set", "schedule.random_share=0.5"]  # fmt: skip
    assert CliRunner().invoke(cli, base).exit_code == 0
    identity = (store / "identity.json").read_text(encoding="utf-8")

    other = [*base, "--seed", "7"]
    real = CliRunner().invoke(cli, other)
    estimate = CliRunner().invoke(cli, [*other, "--estimate"])

    assert real.exit_code == estimate.exit_code == 11
    assert "--force" in json.loads(estimate.stdout)["error"]["hint"]
    assert (store / "identity.json").read_text(encoding="utf-8") == identity


def test_the_offline_judge_runs_under_every_judge_setting_given(dataset: str, tmp_path: Path) -> None:
    """--judge fake with --judge-model and --set judge.*: the pass records what was set, as for a real judge."""
    store = tmp_path / "store"
    code, out = _invoke(
        "rubric", "--dataset", dataset, "--judge", "fake", "--judge-model", "fake-b", "--limit", "1",
        "--set", "judge.decoding=json_schema", "--set", "judge.context_tokens=64000", "--set", "judge.max_images=2",
        "--set", "schedule.window=4", "--out", str(store), "--json",
    )  # fmt: skip
    assert code == 0, out
    entry = json.loads((store / "identity.json").read_text(encoding="utf-8"))["stages"]["rubric"]
    assert (entry["family"]["judge_model"], entry["family"]["decoding"]) == ("fake-b", "json_schema")
    assert entry["identity"]["judge"]["context_tokens"] == 64000


def test_only_the_tournament_takes_a_plan(dataset: str, tmp_path: Path) -> None:
    """An insertion plan holds tournament windows: `judge rubric` has no --plan to refuse."""
    rubric, _ = _invoke("rubric", "--dataset", dataset, "--judge", "fake", "--plan", "p.json", "--out", str(tmp_path))
    assert rubric == 2  # no such option
    help_text = CliRunner().invoke(judge_group, ["tournament", "--help"]).output
    assert "--plan" in help_text and "--plan" not in CliRunner().invoke(judge_group, ["rubric", "--help"]).output


def test_the_mirror_interval_flag_reaches_the_mirror(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mirror-interval`` matches the run config's ``mirror_interval_s``; the default is its constant."""
    import rcp_ndcg.cli.judge as judge_module
    from rcp_ndcg.runs.mirror import DEFAULT_INTERVAL_S

    seen: list[float] = []
    real = judge_module.mirrored

    def spy(root, remote, *, interval_s, state_file=None):
        seen.append(interval_s)
        return real(root, remote, interval_s=interval_s, state_file=state_file)

    monkeypatch.setattr(judge_module, "mirrored", spy)
    args = ("rubric", "--dataset", dataset, "--judge", "fake", "--set", "schedule.window=4",
            "--out", str(tmp_path / "store"), "--mirror", "memory://mirror/store")  # fmt: skip
    code, out = _invoke(*args, "--json")
    assert code == 0, out
    assert seen == [DEFAULT_INTERVAL_S]

    seen.clear()
    code, out = _invoke(*args, "--mirror-interval", "0.5", "--json")
    assert code == 0, out
    assert seen == [0.5]
