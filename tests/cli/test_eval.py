"""``rcp-ndcg eval score|compare|explain`` over a dataset URI, a rankings file and a calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.eval import eval_group
from rcp_ndcg.data import SUITES, Rankings

VIDORE_REPO = SUITES["vidore"].repo
SHA = "4" * 40


def _invoke(*args: str) -> dict:
    result = CliRunner().invoke(eval_group, [*args, "--json"])
    return {"exit_code": result.exit_code, **json.loads(result.stdout)}


def _staged_vidore_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hub cache an online run filled for one ViDoRe subset: snapshots only, no recorded ref."""
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    snapshot = cache / f"datasets--{VIDORE_REPO.replace('/', '--')}" / "snapshots" / SHA
    (snapshot / "hr__english").mkdir(parents=True)
    pd.DataFrame({"query-id": ["q1"], "corpus-id": ["a"], "score": [1.0], "gain": [1.0], "theta": [1.0]}).to_parquet(
        snapshot / "hr__english/qrels.parquet"
    )
    pd.DataFrame({"query-id": ["q1"], "corpus-ids": [["a"]]}).to_parquet(snapshot / "hr__english/top_ranked.parquet")
    no_exist = cache / f"datasets--{VIDORE_REPO.replace('/', '--')}" / ".no_exist" / SHA / "hr__english"
    no_exist.mkdir(parents=True)
    (no_exist / "excluded.parquet").touch()
    return cache


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    """One query with float grades (the released gains are floats; they once went through ``int()``)."""
    path = tmp_path / "rows.jsonl"
    row = {"query_id": "q1", "query": "q", "doc_ids": ["a", "b", "c"], "docs": ["A", "B", "C"],
           "qrels": {"a": 0.9, "b": 0.4, "c": 0.1}}  # fmt: skip
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return f"jsonl:{path}"


def _summary(document: dict, metric: str, k: int, system: str) -> float:
    (row,) = [r for r in document["data"]["summary"] if (r["metric"], r["k"], r["system"]) == (metric, k, system)]
    return row["value"]


def test_an_unpinned_offline_run_carries_the_warning_in_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline with no revision to resolve, the warning is in the JSON envelope's warnings, not only stderr."""
    _staged_vidore_cache(tmp_path, monkeypatch)
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    document = _invoke("score", "--rankings", str(rankings), "--suite", "vidore", "--subset", "hr__english")

    assert document["ok"] is False
    assert [w["code"] for w in document["warnings"]] == ["UNPINNED_REVISION"]
    assert "--revision" in document["warnings"][0]["message"]


def test_the_unpinned_warning_prints_on_stderr_without_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --json the same warning prints on stderr, as every typed warning does."""
    _staged_vidore_cache(tmp_path, monkeypatch)
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    result = CliRunner().invoke(
        eval_group, ["score", "--rankings", str(rankings), "--suite", "vidore", "--subset", "hr__english"]
    )

    assert result.exit_code == 4
    assert "warning [UNPINNED_REVISION]" in result.stderr


def test_a_pinned_offline_run_resolves_without_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinned offline run works and carries no warning: the revision is already exact."""
    _staged_vidore_cache(tmp_path, monkeypatch)
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    document = _invoke(
        "score", "--rankings", str(rankings), "--suite", "vidore", "--subset", "hr__english", "--revision", SHA
    )

    assert document["ok"] is True, document
    assert document["warnings"] == []


def test_float_grades_are_scored_as_floats(dataset: str, tmp_path: Path) -> None:
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)

    document = _invoke("score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg", "--k", "3")

    assert document["exit_code"] == 0, document
    assert document["data"]["schema"] == "rcp-ndcg.eval-score.v1"
    dcg = 0.4 + 0.9 / math.log2(3) + 0.1 / 2.0
    ideal = 0.9 + 0.4 / math.log2(3) + 0.1 / 2.0
    assert _summary(document, "qrel_ndcg", 3, "mine") == pytest.approx(dcg / ideal)


@pytest.mark.parametrize(("protocol", "expected"), [("plain", 0.5 + 0.5 / math.log2(3)), ("bright", 1 / math.log2(3))])
def test_tied_scores_follow_the_protocols_tie_rule(tmp_path: Path, protocol: str, expected: float) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"query_id": "q1", "query": "q", "doc_ids": ["a", "b", "c"], "docs": ["A", "B", "C"], "qrels": {"a": 1.0}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rankings = tmp_path / "run.jsonl"
    Rankings.from_scores({"q1": {"a": 1.0, "b": 1.0, "c": 0.0}}, system="tied").save(rankings)

    document = _invoke(
        "score", "--rankings", str(rankings), "--dataset", f"jsonl:{path}", "--protocol", protocol,
        "--metrics", "qrel_ndcg", "--k", "2",
    )  # fmt: skip

    assert _summary(document, "qrel_ndcg", 2, "tied") == pytest.approx(expected)


@pytest.fixture
def scored(tmp_path: Path) -> dict:
    """Two systems scored against the tiny world's calibration, with the full report written by --out."""
    from rcp_ndcg.calibration import Calibration
    from rcp_ndcg.testing import build_tiny_world, tiny_rows

    world = build_tiny_world(tmp_path / "world")
    rows, _ = tiny_rows()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps({"query_id": r.id, "query": r.query, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
            + "\n"
            for r in rows
        ),  # fmt: skip
        encoding="utf-8",
    )
    pools = {row.query_id: [] for row in Calibration.load(world.calibration).thetas}
    for row in Calibration.load(world.calibration).thetas:
        pools[row.query_id].append(row.doc_id)
    rankings = tmp_path / "run.parquet"
    Rankings.concat(
        [
            Rankings.from_orders({q: sorted(d) for q, d in pools.items()}, system="forward"),
            Rankings.from_orders({q: sorted(d, reverse=True) for q, d in pools.items()}, system="reverse"),
        ]
    ).save(rankings)
    report = tmp_path / "report.json"
    args = ["--rankings", str(rankings), "--dataset", f"jsonl:{path}", "--calibration", str(world.calibration)]
    return {"args": args, "report": report, "document": _invoke("score", *args, "--out", str(report))}


def test_a_calibration_gives_rcp_ndcg_and_a_report_to_compare(scored: dict) -> None:
    report = scored["report"]
    compared = _invoke("compare", "--report", str(report), "--baseline", "forward", "--bootstrap", "100")
    scored = scored["document"]

    assert scored["data"]["gains_source"] == "calibration"
    assert _summary(scored, "rcp_ndcg", 10, "reverse") > _summary(scored, "rcp_ndcg", 10, "forward")
    assert compared["data"]["schema"] == "rcp-ndcg.comparison.v1"
    (pair,) = compared["data"]["pairs"]
    assert (pair["system_a"], pair["system_b"]) == ("forward", "reverse") and pair["delta"] > 0


def test_score_needs_exactly_one_data_source(tmp_path: Path) -> None:
    document = _invoke("score", "--rankings", str(tmp_path / "x.parquet"))

    assert document["exit_code"] == 2
    assert "--suite" in document["error"]["message"]


def test_rankings_of_another_dataset_exit_12_not_a_table_of_zeros(dataset: str, tmp_path: Path) -> None:
    """A `dataset` column that names no scored subset (issue #5, case 1): the error envelope, exit 12."""
    rankings = tmp_path / "wrong_dataset.parquet"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine", dataset="hr").save(rankings)

    document = _invoke("score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg")

    assert document["exit_code"] == 12 and document["ok"] is False
    assert document["error"]["code"] == "DATA"
    assert "no rankings of dataset" in document["error"]["message"]
    assert "exact dataset name" in document["error"]["hint"]


def test_rankings_of_another_corpus_exit_12_not_a_table_of_zeros(dataset: str, tmp_path: Path) -> None:
    """Doc ids that match no pool or label (issue #5, case 2): the error envelope, exit 12."""
    rankings = tmp_path / "no_prefix.parquet"
    Rankings.from_orders({"q1": ["x1", "x2", "x3"]}, system="mine").save(rankings)

    document = _invoke("score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg")

    assert document["exit_code"] == 12 and document["ok"] is False
    assert document["error"]["code"] == "DATA"
    assert "no ranked document is in the pools or labels" in document["error"]["message"]


def test_explain_refuses_a_report_whose_rankings_stopped_matching(scored: dict) -> None:
    """The checks live in evaluate(), so `eval explain --report` gets them: a rankings file rewritten after
    scoring is a data error, not a table of zeros."""
    Rankings.from_orders({"q1": ["other-1"], "q2": ["other-2"]}, system="forward").save(scored["args"][1])

    document = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1")

    assert document["exit_code"] == 12 and document["ok"] is False
    assert "no ranked document is in the pools or labels" in document["error"]["message"]


def test_count_ndcg_is_not_offered_by_eval_score(dataset: str, tmp_path: Path) -> None:
    # Count-nDCG needs count gains, which no option of eval score supplies: the value is refused at parse.
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)

    from rcp_ndcg.cli.main import cli

    args = ["eval", "score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "count_ndcg", "--json"]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 2 and json.loads(result.stdout)["error"]["code"] == "USAGE", result.output


def test_score_json_is_lean_and_the_full_report_goes_to_out(scored: dict) -> None:
    """The per-query rows (539 KB on NanoBEIR) stay out of stdout unless asked for; --out holds everything."""
    data = scored["document"]["data"]
    assert "per_query" not in data
    assert {"summary", "per_dataset", "warnings", "protocol", "inputs", "out"} <= set(data)
    full = json.loads(scored["report"].read_text())
    assert full["schema"] == "rcp-ndcg.eval-report.v1" and full["per_query"]
    assert full["inputs"]["calibration"] == data["inputs"]["calibration"]

    with_rows = _invoke("score", *scored["args"], "--per-query", "--bootstrap", "0")
    assert len(with_rows["data"]["per_query"]) == len(full["per_query"])
    only = _invoke("score", *scored["args"], "--fields", "summary", "--bootstrap", "0")
    assert set(only["data"]) == {"schema", "summary"}
    typo = _invoke("score", *scored["args"], "--fields", "sumary")
    assert typo["exit_code"] == 2 and typo["error"]["details"]["did_you_mean"] == "summary"


def test_explain_reads_a_saved_report_and_leaves_texts_out_unless_asked(scored: dict) -> None:
    """Systems scored with `eval score --out` can be explained: the report records its inputs."""
    report = str(scored["report"])
    explained = _invoke("explain", "--report", report, "--query-id", "q1", "--k", "3")
    assert explained["exit_code"] == 0, explained
    data = explained["data"]
    assert [system["system"] for system in data["systems"]] == ["forward", "reverse"]
    assert len(data["items"]) == 5 and "gamma" not in data["systems"][0]["top"][0]["criteria"][0]
    assert data["query"] is None and data["texts"] == {}
    (delta,) = data["deltas"]
    assert delta["total"] == pytest.approx(delta["selection"] + delta["ordering"])

    with_text = _invoke("explain", "--report", report, "--query-id", "q1", "--k", "3", "--include-text")["data"]
    assert with_text["query"] and all(with_text["texts"].values())

    missing = _invoke("explain", "--report", report, "--query-id", "q9")
    assert missing["exit_code"] == 12
    assert "known: q1, q2" in missing["error"]["message"]
