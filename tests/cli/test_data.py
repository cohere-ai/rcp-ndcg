"""``rcp-ndcg data``: convert through the reader registry, leaving a self-describing dataset; inspect it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.data import data_group
from rcp_ndcg.examples import tiny


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> str:
    pytest.importorskip("pypdfium2")
    PIL = pytest.importorskip("PIL.Image")
    PIL.init()
    pages = [PIL.new("RGB", (612, 792), color=(255, 255 - index * 40, 255)) for index in range(2)]
    target = tmp_path / "report.pdf"
    pages[0].save(target, save_all=True, append_images=pages[1:], resolution=72.0)
    return str(target)


@pytest.fixture
def unjudged_pages(tmp_path: Path) -> str:
    """Page images with no queries and no qrels: a corpus, nothing to search it for."""
    PIL = pytest.importorskip("PIL.Image")
    root = tmp_path / "pages"
    root.mkdir()
    for index in range(2):
        PIL.new("RGB", (40, 60)).save(root / f"page_{index}.png")
    return str(root)


@pytest.fixture
def ranking_jsonl(tmp_path: Path) -> str:
    target = tmp_path / "pool.jsonl"
    rows = [
        {"query_id": "q1", "query": "hello", "doc_ids": ["d1", "d2"], "docs": ["a", "b"], "qrels": {"d1": 1}},
        {"query_id": "q2", "query": "world", "doc_ids": ["d3"], "docs": ["c"], "qrels": {"d3": 2}},
    ]
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(target)


class TestFormats:
    def test_lists_the_registered_readers_and_writers(self, runner: CliRunner) -> None:
        result = runner.invoke(data_group, ["formats", "--json"])
        payload = json.loads(result.stdout)["data"]

        assert result.exit_code == 0
        assert {"pdf", "beir", "jsonl", "hf", "images"} <= set(payload["readers"])
        assert {"jsonl", "beir"} <= set(payload["writers"])


class TestIngestPdf:
    def _ingest(self, runner: CliRunner, source: str, out: Path, *extra: str):
        args = ["convert", "--format", "pdf", "--source", source, "--out", str(out), "--json", *extra]
        return runner.invoke(data_group, args)

    def test_pages_become_a_corpus(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        result = self._ingest(runner, two_page_pdf, out, "--set", "dpi=72")

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["data"]["written"] == 2
        rows = [json.loads(line) for line in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [row["id"] for row in rows] == ["report#p1", "report#p2"]
        assert rows[0]["content"][0]["ref"]["width"] == 612, "a 612-point page rendered at 72 dpi"

    def test_dry_run_writes_nothing(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        result = self._ingest(runner, two_page_pdf, out, "--set", "dpi=72", "--dry-run")

        assert result.exit_code == 0, result.output
        assert "reader" in result.output and "pdf" in result.output
        assert not out.exists()

    def test_limit_stops_early(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        self._ingest(runner, two_page_pdf, out, "--set", "dpi=72", "--limit", "1")

        assert len((out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    def test_reader_options_pass_through_and_coerce(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        self._ingest(runner, two_page_pdf, out, "--set", "dpi=72", "--set", "image_format=jpeg")

        rows = [json.loads(line) for line in (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[0]["content"][0]["ref"]["mime"] == "image/jpeg"

    def test_a_malformed_reader_option_is_refused(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        result = self._ingest(runner, two_page_pdf, tmp_path / "d", "--set", "dpi")

        assert result.exit_code == 2
        assert "KEY=VALUE" in json.loads(result.stdout)["error"]["message"]


class TestIngestShapes:
    def test_a_ranking_source_round_trips(self, runner: CliRunner, ranking_jsonl: str, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        result = runner.invoke(
            data_group,
            [
                "convert",
                "--format",
                "jsonl",
                "--source",
                ranking_jsonl,
                "--out",
                str(out),
                "--shape",
                "ranking",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["data"]["written"] == 2

    def test_a_ranking_source_can_be_written_as_a_corpus(
        self, runner: CliRunner, ranking_jsonl: str, tmp_path: Path
    ) -> None:
        """The derivation the registry exists for: a judged pool becomes a corpus
        without the writer knowing where it came from."""
        out = tmp_path / "corpus"
        result = runner.invoke(
            data_group,
            ["convert", "--format", "jsonl", "--source", ranking_jsonl, "--out", str(out), "--shape", "corpus"],
        )

        assert result.exit_code == 0, result.output
        assert len((out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        assert len((out / "queries.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    def test_a_jsonl_corpus_loads_back(self, runner: CliRunner, ranking_jsonl: str, tmp_path: Path) -> None:
        """The corpus layout ``--to jsonl`` writes (corpus, queries, qrels) is a dataset ``load_dataset`` reads."""
        from rcp_ndcg.data import load_dataset

        out = tmp_path / "corpus"
        args = ["convert", "--format", "jsonl", "--source", ranking_jsonl, "--out", str(out), "--shape", "corpus"]
        assert runner.invoke(data_group, args).exit_code == 0

        dataset = load_dataset(f"jsonl:{out}")

        assert set(dataset.corpus) == {"d1", "d2", "d3"}
        assert {q: row.text for q, row in dataset.queries.items()} == {"q1": "hello", "q2": "world"}
        assert dataset.qrels == {"q1": {"d1": 1.0}, "q2": {"d3": 2.0}}
        assert dataset.candidates is None, "a corpus has no candidate pools"

    def test_a_shape_the_reader_cannot_serve_is_refused(
        self, runner: CliRunner, two_page_pdf: str, tmp_path: Path
    ) -> None:
        """A PDF has no queries, so a ranking shape from it would be fabricated."""
        result = runner.invoke(
            data_group,
            [
                "convert",
                "--format",
                "pdf",
                "--source",
                two_page_pdf,
                "--out",
                str(tmp_path / "d"),
                "--shape",
                "ranking",
            ],
        )

        assert result.exit_code == 2
        assert "cannot serve" in result.stderr

    def test_an_ingest_that_would_write_nothing_is_refused(
        self, runner: CliRunner, unjudged_pages: str, tmp_path: Path
    ) -> None:
        """An empty JSONL is the artefact that gets mistaken for real data weeks later.

        An image directory with no qrels has no candidate lists to give, so asking
        it for the ranking shape must say so rather than write nothing.
        """
        out = tmp_path / "pool.jsonl"
        result = runner.invoke(
            data_group,
            ["convert", "--format", "images", "--source", unjudged_pages, "--out", str(out), "--shape", "ranking"],
        )

        assert result.exit_code == 12
        assert "yielded no" in result.stderr
        assert "--shape corpus" in result.stderr

    def test_the_refusal_leaves_nothing_behind(self, runner: CliRunner, unjudged_pages: str, tmp_path: Path) -> None:
        """A partial artefact from a failed ingest makes the retry ambiguous."""
        out = tmp_path / "pool.jsonl"
        runner.invoke(
            data_group,
            ["convert", "--format", "images", "--source", unjudged_pages, "--out", str(out), "--shape", "ranking"],
        )

        assert not out.exists()

    def test_the_name_option_names_the_dataset(self, ranking_jsonl: str) -> None:
        """``name`` is a reader option like any other; it must not collide with the reader's format argument."""
        from rcp_ndcg.data import load_dataset

        assert load_dataset(f"jsonl:{ranking_jsonl}", name="tinyx").name == "tinyx"

    def test_an_option_the_reader_does_not_take_is_a_usage_error(
        self, runner: CliRunner, ranking_jsonl: str, tmp_path: Path
    ) -> None:
        """A JSONL reader renders nothing, so ``dpi`` means nothing to it: say so, and name what it takes."""
        args = ["convert", "--format", "jsonl", "--source", ranking_jsonl, "--out", str(tmp_path / "d")]
        result = runner.invoke(data_group, [*args, "--set", "dpi=100", "--json"])

        error = json.loads(result.stdout)["error"]
        assert result.exit_code == 2, result.output
        assert "dpi" in error["message"]
        assert "name" in error["hint"]

    def test_an_unknown_format_lists_what_is_available(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            data_group, ["convert", "--format", "parquet_of_dreams", "--source", "x", "--out", str(tmp_path / "d")]
        )

        assert result.exit_code == 2
        assert "jsonl" in result.stderr


class TestInspect:
    def test_any_dataset_uri_is_inspected_through_load_dataset(self, runner: CliRunner) -> None:
        """``--dataset`` means the same URI here as on every other command."""
        rows = tiny() / "rows.jsonl"
        result = runner.invoke(data_group, ["inspect", "--dataset", f"jsonl:{rows}", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)["data"]
        assert (data["kind"], data["queries"]) == ("dataset", 3)
        assert data["pool_depth"] == {"min": 8.0, "mean": 8.0, "max": 8.0}

    def test_validate_checks_rankings_against_any_dataset(self, runner: CliRunner, tmp_path: Path) -> None:
        rows = tiny() / "rows.jsonl"
        rankings = tmp_path / "run.jsonl"
        rankings.write_text(json.dumps({"query_id": "q9", "doc_ids": ["x"]}) + "\n", encoding="utf-8")

        args = ["validate", "--dataset", f"jsonl:{rows}", "--rankings", str(rankings), "--json"]
        result = runner.invoke(data_group, args)

        assert result.exit_code == 12, result.output
        assert [c["code"] for c in json.loads(result.stdout)["error"]["details"]["checks"]] == ["UNKNOWN_QUERIES"]

    def test_subset_and_revision_are_refused_where_they_do_not_apply(self, runner: CliRunner) -> None:
        rows = tiny() / "rows.jsonl"
        result = runner.invoke(data_group, ["inspect", "--dataset", f"jsonl:{rows}", "--subset", "x", "--json"])

        assert result.exit_code == 3, result.output

    def test_a_dataset_directory_is_enough(self, runner: CliRunner, two_page_pdf: str, tmp_path: Path) -> None:
        out = tmp_path / "dataset"
        runner.invoke(
            data_group, ["convert", "--format", "pdf", "--source", two_page_pdf, "--out", str(out), "--set", "dpi=72"]
        )

        result = runner.invoke(data_group, ["inspect", "--dataset", f"jsonl:{out}", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["data"]["subset"] == "dataset"

    def test_a_suite_without_a_subset_lists_its_subsets(self, runner: CliRunner) -> None:
        result = runner.invoke(data_group, ["inspect", "--dataset", "suite:trecdl", "--json"])

        data = json.loads(result.stdout)["data"]
        assert (data["kind"], data["subsets"]) == ("suite-subsets", ["trec_dl_2019", "trec_dl_2020"])


def _subset(**overrides):
    from rcp_ndcg.data import Dataset

    fields = {
        "name": "NanoTiny",
        "protocol": "nanobeir",
        "gains": {"q1": {"d1": 0.9, "d2": 0.2}, "q2": {"d3": 0.5}},
        "qrels": {"q1": {"d1": 2.0, "d2": 0.0}, "q2": {"d3": 1.0}},
        "candidates": {"q1": ["d1", "d2", "d9"], "q2": ["d3"]},
        "excluded": {"q2": ["d4"]},
    }
    return Dataset(**{**fields, **overrides})


class TestSuites:
    """Offline: the released subset is replaced by a small one built in the test."""

    def test_inspect_summarises_pools_labels_and_gains(self, runner: CliRunner, monkeypatch) -> None:
        monkeypatch.setattr("rcp_ndcg.data.load_dataset", lambda *a, **k: _subset())

        result = runner.invoke(data_group, ["inspect", "--dataset", "suite:nanobeir", "--subset", "NanoTiny", "--json"])

        data = json.loads(result.stdout)["data"]
        assert result.exit_code == 0, result.output
        assert data["schema"] == "rcp-ndcg.dataset-summary.v1"
        assert (data["queries"], data["judged_documents"], data["positive_labels"], data["excluded"]) == (2, 3, 2, 1)
        assert data["labels"] == {"1": 1, "2": 1}
        assert data["pool_depth"] == {"min": 1.0, "mean": 2.0, "max": 3.0}

    def test_validate_reports_warnings_and_passes(self, runner: CliRunner, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr("rcp_ndcg.data.load_dataset", lambda *a, **k: _subset())
        rankings = tmp_path / "run.jsonl"
        rankings.write_text(json.dumps({"query_id": "q2", "doc_ids": ["d3", "d4", "d8"]}) + "\n", encoding="utf-8")

        args = ["validate", "--dataset", "suite:nanobeir", "--subset", "NanoTiny", "--rankings", str(rankings)]
        result = runner.invoke(data_group, [*args, "--json"])

        data = json.loads(result.stdout)["data"]
        assert result.exit_code == 0, result.output
        assert data["ok"] is True
        codes = {check["code"]: check["count"] for check in data["checks"]}
        assert codes == {"UNJUDGED_CANDIDATES": 1, "OUTSIDE_POOL": 1, "EXCLUDED_RANKED": 1}

    def test_validate_refuses_data_that_would_give_wrong_numbers(self, runner: CliRunner, monkeypatch) -> None:
        broken = _subset(gains={"q1": {"d1": 1.5, "d2": 0.2}, "q2": {"d3": 0.5}}, excluded={"q1": ["d1"]})
        monkeypatch.setattr("rcp_ndcg.data.load_dataset", lambda *a, **k: broken)

        args = ["validate", "--dataset", "suite:nanobeir", "--subset", "NanoTiny", "--json"]
        result = runner.invoke(data_group, args)

        error = json.loads(result.stdout)["error"]
        assert result.exit_code == 12
        assert error["code"] == "DATA"
        assert {check["code"] for check in error["details"]["checks"]} >= {"GAIN_RANGE", "EXCLUDED_RELEVANT"}

    def test_a_missing_extra_names_the_install_command(self, runner: CliRunner) -> None:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("huggingface_hub is installed")

        result = runner.invoke(data_group, ["fetch", "--dataset", "nanobeir", "--json"])

        error = json.loads(result.stdout)["error"]
        assert result.exit_code == 10
        assert error["hint"] == 'pip install "rcp-ndcg[hf]"'


def test_fetch_tiny_copies_the_packaged_example(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(data_group, ["fetch", "--dataset", "tiny", "--out", str(tmp_path / "tiny"), "--json"])

    data = json.loads(result.stdout)["data"]
    assert result.exit_code == 0, result.output
    assert (data["repo_id"], data["files"]) == (None, 3)
    assert (tmp_path / "tiny" / "systems.jsonl").read_text() == (tiny() / "systems.jsonl").read_text()


def test_a_limited_conversion_records_the_limit(runner: CliRunner, ranking_jsonl: str, tmp_path: Path) -> None:
    """A `--limit` smoke conversion records the cap, so its record is not mistaken for a complete small corpus."""
    limited = runner.invoke(
        data_group,
        [
            "convert",
            "--format",
            "jsonl",
            "--source",
            ranking_jsonl,
            "--out",
            str(tmp_path / "limited.parquet"),
            "--shape",
            "ranking",
            "--limit",
            "1",
            "--json",
        ],  # fmt: skip
    )
    full = runner.invoke(
        data_group,
        [
            "convert",
            "--format",
            "jsonl",
            "--source",
            ranking_jsonl,
            "--out",
            str(tmp_path / "full.parquet"),
            "--shape",
            "ranking",
            "--json",
        ],  # fmt: skip
    )

    assert limited.exit_code == 0 and full.exit_code == 0, (limited.output, full.output)
    assert json.loads(limited.stdout)["data"]["limit"] == 1
    assert json.loads(full.stdout)["data"]["limit"] is None
    assert json.loads(limited.stdout)["data"]["written"] == 1
