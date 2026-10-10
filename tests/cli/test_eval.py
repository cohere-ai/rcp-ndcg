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
    """A hub cache an online run filled for one ViDoRe subset: snapshots only, no recorded ref. The card is
    staged with the tables, the way the released repositories declare them."""
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
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n"
        "- config_name: hr__english-qrels\n  data_files:\n  - split: test\n    path: hr__english/qrels.parquet\n"
        "- config_name: hr__english-top_ranked\n"
        "  data_files:\n  - split: test\n    path: hr__english/top_ranked.parquet\n"
        "---\n"
    )
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
            json.dumps({"query_id": r.id, "query": r.text, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
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
    scoring is refused, not a table of zeros."""
    Rankings.from_orders({"q1": ["other-1"], "q2": ["other-2"]}, system="forward").save(scored["args"][1])

    document = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1")

    assert document["exit_code"] == 3, document
    assert "reverse" in document["error"]["message"], "the report scored reverse, the file no longer holds it"

    narrowed = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1", "--system", "forward")

    assert narrowed["exit_code"] == 12, narrowed
    assert "no ranked document is in the pools or labels" in narrowed["error"]["message"]


def _rubric_store(tmp_path: Path, dataset: str, windows: dict[str, list[list[tuple[str, list[int]]]]]) -> Path:
    """A rubric judgement store of ``dataset`` (the minimal identity entry ``read_judgements`` reads)."""
    from datetime import UTC, datetime

    from rcp_ndcg_core.schemas import Judgement, JudgementFamily, Placement, criterion_labels, judgement_record_id

    from rcp_ndcg.judging.store import JudgementStore

    family = JudgementFamily(
        stage="rubric", judge_model="m", prompt_hash="0" * 64, criteria=criterion_labels(5), parse_version=1
    )
    store = JudgementStore(tmp_path / "rubric-store")
    store.root.mkdir(parents=True, exist_ok=True)
    store.claim("rubric", {}, family)
    for query_id, query_windows in windows.items():
        for seq, window in enumerate(query_windows):
            placements = tuple(
                Placement(
                    position=position,
                    doc_id=doc_id,
                    criteria=dict(zip(family.criteria, map(int, verdicts), strict=True)),
                )
                for position, (doc_id, verdicts) in enumerate(window, start=1)
            )
            ids = [p.unit_id for p in placements]
            store.append(
                Judgement(
                    record_id=judgement_record_id(family.key, query_id, "rubric", seq, ids, dataset=dataset),
                    dataset=dataset,
                    query_id=query_id,
                    stage="rubric",
                    family_key=family.key,
                    window_seq=seq,
                    placements=placements,
                    recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
    return store.root


def test_count_ndcg_scores_the_gains_derived_from_the_rubric_store(dataset: str, tmp_path: Path) -> None:
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)
    store = _rubric_store(
        tmp_path,
        "rows",
        {"q1": [[("a", [1, 1, 1, 1, 1]), ("b", [0, 0, 0, 0, 0])], [("c", [1, 0, 0, 0, 0])]]},
    )

    document = _invoke(
        "score",
        "--rankings",
        str(rankings),
        "--dataset",
        dataset,
        "--metrics",
        "count_ndcg",
        "--judgements",
        str(store),
        "--out",
        str(tmp_path / "report.json"),
    )

    assert document["exit_code"] == 0, document
    # Count gains: a = 5/5, b = 0, c = 1/5. The order b, a, c scores 1/log2(3) + 0.2/2 over 1 + 0.2/log2(3).
    assert _summary(document, "count_ndcg", 10, "mine") == pytest.approx(
        (1 / math.log2(3) + 0.2 / 2) / (1.0 + 0.2 / math.log2(3))
    )
    # The report records the rubric store, so `eval explain --report` re-scores the count gains.
    explained = _invoke("explain", "--report", str(tmp_path / "report.json"), "--query-id", "q1")
    assert explained["exit_code"] == 0, explained


def test_count_ndcg_without_the_rubric_store_names_the_flag(dataset: str, tmp_path: Path) -> None:
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)

    document = _invoke("score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "count_ndcg")

    assert document["exit_code"] == 2 and document["error"]["code"] == "USAGE", document
    assert "--judgements" in document["error"]["hint"]


def test_a_rubric_store_of_another_dataset_is_refused(dataset: str, tmp_path: Path) -> None:
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)
    store = _rubric_store(tmp_path, "elsewhere", {"q1": [[("a", [1, 0, 0, 0, 0])]]})

    document = _invoke(
        "score",
        "--rankings",
        str(rankings),
        "--dataset",
        dataset,
        "--metrics",
        "count_ndcg",
        "--judgements",
        str(store),
    )

    assert document["exit_code"] == 12 and "no verdicts of" in document["error"]["message"], document


def test_a_broken_system_no_longer_stops_the_others_with_system(dataset: str, tmp_path: Path) -> None:
    """One broken system of a multi-system file fails the whole command; --system scores the rest (issue #5)."""
    good = Rankings.from_orders({"q1": ["b", "a", "c"]}, system="good")
    broken = Rankings.from_orders({"q1": ["x1", "x2"]}, system="broken", dataset="hr")
    rankings = tmp_path / "mixed.parquet"
    Rankings.concat([good, broken]).save(rankings)

    document = _invoke(
        "score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg", "--system", "good"
    )

    assert document["exit_code"] == 0, document
    assert [row["system"] for row in document["data"]["summary"]] == ["good"]
    dcg = 0.4 + 0.9 / math.log2(3) + 0.1 / 2.0
    ideal = 0.9 + 0.4 / math.log2(3) + 0.1 / 2.0
    assert _summary(document, "qrel_ndcg", 10, "good") == pytest.approx(dcg / ideal)

    everything = _invoke("score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg")
    assert everything["exit_code"] == 12, everything
    assert "system 'broken'" in everything["error"]["message"]
    assert "--system" in everything["error"]["hint"]


def test_an_unknown_system_is_a_usage_error_listing_the_systems(dataset: str, tmp_path: Path) -> None:
    """An unknown `--system` value is a command-line mistake (exit 2), like an unknown `--fields` name — not a
    config one: there is no YAML here. The message keeps the systems the file names."""
    rankings = tmp_path / "run.parquet"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)

    document = _invoke(
        "score", "--rankings", str(rankings), "--dataset", dataset, "--metrics", "qrel_ndcg", "--system", "nobody"
    )

    assert document["exit_code"] == 2, document
    assert document["error"]["code"] == "USAGE"
    assert "nobody" in document["error"]["message"] and "mine" in document["error"]["message"]


def test_an_unknown_baseline_is_a_usage_error(scored: dict) -> None:
    document = _invoke("compare", "--report", str(scored["report"]), "--baseline", "nobody")

    assert document["exit_code"] == 2, document
    assert document["error"]["code"] == "USAGE"
    assert "nobody" in document["error"]["message"] and "forward" in document["error"]["message"]


def test_explain_report_re_scores_the_systems_the_report_scored(scored: dict) -> None:
    """`eval explain --report` defaults to the systems the report scored: a system added to the rankings file
    after scoring does not kill the explanation, and `--system` narrows the re-score."""
    from rcp_ndcg.data import load_rankings

    rankings = Rankings.concat(
        [
            Rankings.from_scores({"q1": {"a": 1.0}}, system="broken", dataset="zzz"),
            load_rankings(scored["args"][1]),
        ]
    )
    rankings.save(scored["args"][1])

    document = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1")

    assert document["exit_code"] == 0, document
    assert [system["system"] for system in document["data"]["systems"]] == ["forward", "reverse"], "broken never scores"

    narrowed = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1", "--system", "forward")

    assert narrowed["exit_code"] == 0, narrowed
    assert [system["system"] for system in narrowed["data"]["systems"]] == ["forward"]


def test_score_json_is_lean_and_the_full_report_goes_to_out(scored: dict) -> None:
    """The per-query rows (539 KB on NanoBEIR) stay out of stdout unless asked for; --out holds everything."""
    data = scored["document"]["data"]
    assert "per_query" not in data
    assert {"summary", "per_dataset", "warnings", "protocol", "inputs", "out"} <= set(data)
    full = json.loads(scored["report"].read_text())
    assert full["schema"] == "rcp-ndcg.eval-report.v1" and full["per_query"]
    assert full["inputs"]["calibration"] == data["inputs"]["calibration"]
    assert full["inputs"]["split"] == "test" and full["inputs"]["task"] is None

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


def test_explain_run_refuses_the_system_option(scored: dict) -> None:
    """`--system` re-scores a saved report; with `--run` it is refused, never silently ignored."""
    document = _invoke(
        "explain", "--run", str(scored["report"].parent / "no-run"), "--query-id", "q1", "--system", "forward"
    )

    assert document["exit_code"] == 2, document
    assert document["error"]["code"] == "USAGE"
    assert "--system" in document["error"]["message"] and "--report" in document["error"]["hint"]


def test_a_partial_snapshot_listing_carries_its_warning_in_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline corpus read whose listing came from the local snapshot warns SNAPSHOT_LISTING in --json."""
    cache = _staged_vidore_cache(tmp_path, monkeypatch)
    snapshot = next((cache / f"datasets--{VIDORE_REPO.replace('/', '--')}").glob("snapshots/*"))
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: hr__english-corpus\n  data_files:\n  - path: hr__english/corpus/*.parquet\n---\n"
    )
    (snapshot / "hr__english" / "corpus").mkdir()
    pd.DataFrame({"id": ["a"], "text": ["alpha"]}).to_parquet(snapshot / "hr__english/corpus/part-0.parquet")
    pd.DataFrame({"id": ["q1"], "text": ["q"]}).to_parquet(snapshot / "hr__english/queries.parquet")
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    scored = _invoke(
        "score", "--rankings", str(rankings), "--suite", "vidore", "--subset", "hr__english",
        "--revision", SHA, "--out", str(tmp_path / "report.json"),
    )  # fmt: skip
    assert scored["ok"] is True, scored
    assert scored["warnings"] == [], "scoring the qrels reads no corpus, so no snapshot listing"

    explained = _invoke("explain", "--report", str(tmp_path / "report.json"), "--query-id", "q1", "--include-text")

    assert explained["ok"] is True, explained
    assert [warning["code"] for warning in explained["warnings"]] == ["SNAPSHOT_LISTING"]
    assert "partial cache" in explained["warnings"][0]["message"]
    assert explained["data"]["texts"] == {"a": "alpha"}, "the corpus came from the snapshot"


def test_explain_report_of_an_empty_summary_refuses_like_before(dataset: str, tmp_path: Path) -> None:
    """A report whose only metric matched no labelled query has no systems: the query refusal, exit 12."""
    rows = tmp_path / "no_qrels.jsonl"
    rows.write_text(json.dumps({"query_id": "q1", "query": "q", "doc_ids": ["a"], "docs": ["A"], "qrels": {}}) + "\n")
    rankings = tmp_path / "run.jsonl"
    Rankings.from_orders({"q1": ["a"]}, system="mine").save(rankings)

    scored = _invoke("score", "--rankings", str(rankings), "--dataset", f"jsonl:{rows}", "--metrics", "qrel_ndcg",
                     "--out", str(tmp_path / "empty.json"))  # fmt: skip
    assert scored["ok"] is True, scored
    assert scored["data"]["summary"] == []

    explained = _invoke("explain", "--report", str(tmp_path / "empty.json"), "--query-id", "q1")

    assert explained["exit_code"] == 12, explained
    assert "not in the report" in explained["error"]["message"]


def test_explain_run_refuses_the_subset_option(scored: dict) -> None:
    """`--subset` disambiguates a suite report's datasets; a run explanation has no such input, so it is
    refused like `--system` instead of silently ignored."""
    document = _invoke(
        "explain", "--run", str(scored["report"].parent / "no-run"), "--query-id", "q1", "--subset", "hr__english"
    )

    assert document["exit_code"] == 2, document
    assert document["error"]["code"] == "USAGE"
    assert "--subset" in document["error"]["message"]


def test_score_text_prints_the_per_query_values(dataset: str, tmp_path: Path, capsys) -> None:
    """`--per-query` promises every per-query value: the text renderer prints them too, not only --json."""
    from click.testing import CliRunner

    from rcp_ndcg.cli.main import cli

    rankings = tmp_path / "run.parquet"
    Rankings.from_orders({"q1": ["b", "a", "c"]}, system="mine").save(rankings)

    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "score",
            "--rankings",
            str(rankings),
            "--dataset",
            dataset,
            "--metrics",
            "qrel_ndcg",
            "--per-query",
            "--k",
            "2",
        ],  # fmt: skip
    )

    assert result.exit_code == 0, result.output
    assert "per query" in result.output
    assert "q1" in result.output and "mine" in result.output


def test_explain_report_refuses_an_unknown_system(scored: dict) -> None:
    """An unknown `--system` value is a usage error on the report branch too, like on `eval score`."""
    document = _invoke("explain", "--report", str(scored["report"]), "--query-id", "q1", "--system", "nobody")

    assert document["exit_code"] == 2, document
    assert document["error"]["code"] == "USAGE"
    assert "nobody" in document["error"]["message"] and "forward" in document["error"]["message"]
