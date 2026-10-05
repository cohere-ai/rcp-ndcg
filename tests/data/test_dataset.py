"""``load_dataset`` over every URI scheme, and ``load_rankings`` over every run format."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rcp_ndcg.data import SUITES, Rankings, load_dataset, load_rankings
from rcp_ndcg.data import dataset as dataset_module
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError

REPO = SUITES["nanobeir"].repo
SUBSET = "NanoArguAnaRetrieval"
SHA = "b" * 40  # the fake hub's commit for a branch: a full sha pins the identity without any lookup


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    """A stand-in for the Hub: ``{"<repo>/<path>": table}``, with a two-query NanoArguAna in the public layout."""
    tables = {
        f"{SUBSET}/qrels.parquet": pd.DataFrame(
            {
                "query-id": ["q1", "q1", "q1", "q1", "q2"],
                "corpus-id": ["a", "b", "c", "outside", "a"],
                "score": [0.0, 1.0, 0.5, 1.0, 1.0],
                "gain": [0.9, 0.5, 0.1, 0.0, 0.7],
                "theta": [2.0, 0.0, -2.0, None, 1.0],
            }
        ),
        f"{SUBSET}/top_ranked.parquet": pd.DataFrame(
            {"query-id": ["q1", "q2"], "corpus-ids": [["a", "b", "c"], ["a"]]}
        ),
        f"{SUBSET}/excluded.parquet": pd.DataFrame({"query-id": ["q1"], "excluded-corpus-ids": [["copy"]]}),
        f"{SUBSET}/queries.parquet": pd.DataFrame({"id": ["q1", "q2"], "text": ["first", "second"]}),
        "corpus/part-0.parquet": pd.DataFrame({"id": ["a", "b"], "title": ["A", ""], "text": ["alpha", "beta"]}),
    }
    files = {f"{REPO}/{path}": table for path, table in tables.items()}
    card = tmp_path / "README.md"
    card.write_text(
        f"---\nconfigs:\n- config_name: {SUBSET}-corpus\n"
        "  data_files:\n  - split: train\n    path: corpus/*.parquet\n---\n"
    )

    # The fake hub has resolved its repos' branches to a commit, as an online run leaves a cache: the datasets
    # load pinned, and no test trips the UNPINNED_REVISION warning that an unpinned resolution raises.
    cache = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    for repo in (REPO, SUITES["vidore"].repo):
        ref = cache / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(SHA)
    from rcp_ndcg.data.revisions import resolve_revision

    resolve_revision.cache_clear()

    def read_table(repo: str, path: str, revision: str | None, *, optional: bool = False) -> pd.DataFrame | None:
        if f"{repo}/{path}" in files:
            return files[f"{repo}/{path}"]
        if optional:
            return None
        raise MissingInputError(f"hf://{repo}: {path} does not exist")

    monkeypatch.setattr(dataset_module, "_read_hub_table", read_table)
    monkeypatch.setattr(dataset_module, "_hub_file", lambda repo, path, rev: card if path == "README.md" else None)
    monkeypatch.setattr(dataset_module, "_hub_listing", lambda repo, rev: [k.split("/", 2)[2] for k in files])
    return files


def test_the_public_layout_loads_float_qrels_gains_pools_and_exclusions(hub: dict) -> None:
    dataset = load_dataset(f"hf://{REPO}/{SUBSET}")

    assert dataset.protocol == "nanobeir", "a public suite's repository carries its protocol"
    assert dataset.qrels["q1"] == {"a": 0.0, "b": 1.0, "c": 0.5, "outside": 1.0}, "grades stay floats"
    assert dataset.gains == {"q1": {"a": 0.9, "b": 0.5, "c": 0.1}, "q2": {"a": 0.7}}, "gains of the judged pool only"
    assert dataset.thetas is not None and dataset.thetas["q1"]["c"] == -2.0
    assert dataset.candidates == {"q1": ["a", "b", "c"], "q2": ["a"]}
    assert dataset.excluded == {"q1": ["copy"]}


def test_queries_and_corpus_are_read_on_demand_from_the_card_paths(hub: dict) -> None:
    dataset = load_dataset(f"hf://{REPO}", subset=SUBSET)

    assert {q: query.text for q, query in dataset.queries.items()} == {"q1": "first", "q2": "second"}
    assert {d: doc.text for d, doc in dataset.corpus.items()} == {"a": "A\n\nalpha", "b": "beta"}


def test_hub_data_is_read_at_the_commit_its_revision_resolves_to(hub: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The files were once read at the moving branch while the identity recorded the commit resolved earlier."""
    import os

    from rcp_ndcg.data.revisions import resolve_revision

    commit = "c" * 40
    ref = Path(os.environ["HF_HUB_CACHE"]) / f"datasets--{REPO.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)
    resolve_revision.cache_clear()
    read_at = []
    read_table = dataset_module._read_hub_table
    monkeypatch.setattr(
        dataset_module,
        "_read_hub_table",
        lambda repo, path, revision, **kw: read_at.append(revision) or read_table(repo, path, revision, **kw),
    )

    dataset = load_dataset(f"hf://{REPO}/{SUBSET}")

    assert dataset.revision == commit
    assert set(read_at) == {commit}


def test_a_suite_loads_every_subset(hub: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SUITES, "nanobeir", SUITES["nanobeir"]._replace(subsets=(SUBSET,)))

    suite = load_dataset("suite:nanobeir")

    assert [part.name for part in suite.parts] == [SUBSET]
    assert suite.protocol == "nanobeir" and not suite.qrels
    with pytest.raises(DataError, match="is a suite"):
        _ = suite.queries


def test_a_vidore_domain_selects_its_native_language_subset(hub: dict) -> None:
    hub[f"{SUITES['vidore'].repo}/energy__french/qrels.parquet"] = pd.DataFrame(
        {"query-id": ["q"], "corpus-id": ["p"], "score": [2]}
    )

    dataset = load_dataset("suite:vidore", subset="energy")

    assert dataset.name == "energy__french"
    assert dataset.gains is None and dataset.candidates is None, "a layout without gains or pools says so"


@pytest.mark.parametrize(
    ("uri", "kwargs", "error", "match"),
    [
        ("nanobeir", {}, ConfigError, "no scheme"),
        ("parquet:/x", {}, ConfigError, "unknown dataset URI scheme"),
        ("pdf:/report.pdf", {}, ConfigError, "unknown dataset URI scheme"),
        ("suite:msmarco", {}, ConfigError, "unknown suite"),
        (f"hf://{REPO}/{SUBSET}", {"subset": "other"}, ConfigError, "two subsets"),
        (f"hf://{REPO}/{SUBSET}@v1", {"revision": "v2"}, ConfigError, "two revisions"),
        ("hf://someone/else", {"revision": SHA}, ConfigError, "name the subset"),
        (f"hf://{REPO}/NanoNQRetrieval", {}, MissingInputError, "does not exist"),
        ("beir:/x", {"revision": "v1"}, ConfigError, "subset and revision"),
    ],
)
def test_bad_uris_are_refused_with_the_reason(hub: dict, uri: str, kwargs: dict, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        load_dataset(uri, **kwargs)


def test_reader_schemes_load_through_the_readers(tmp_path: Path) -> None:
    beir = tmp_path / "beir"
    (beir / "qrels").mkdir(parents=True)
    (beir / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "text": "tortoises"}) + "\n")
    (beir / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "slow"}) + "\n")
    (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t0.25\n")
    ranking = tmp_path / "pool.jsonl"
    ranking.write_text(
        json.dumps({"query_id": "q1", "query": "slow", "doc_ids": ["d2", "d1"], "docs": ["x", "y"]}) + "\n"
    )

    from_beir = load_dataset(f"beir:{beir}")
    from_jsonl = load_dataset(f"jsonl:{ranking}")

    assert from_beir.qrels == {"q1": {"d1": 0.25}} and from_beir.candidates is None
    assert from_beir.corpus["d1"].text == "tortoises"
    assert from_jsonl.candidates == {"q1": ["d2", "d1"]}, "a ranking source carries its pools"


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


def test_rankings_load_from_every_format_to_the_same_table(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "model": ["m", "m", "n"],
            "query-id": ["q1", "q1", "q1"],
            "corpus-id": ["a", "b", "a"],
            "score": [2.0, 1.0, 0.5],
        }
    )
    frame.to_csv(tmp_path / "runs.tsv", sep="\t", index=False)
    (tmp_path / "runs.trec").write_text("q1 Q0 a 1 2.0 m\nq1 Q0 b 2 1.0 m\nq1 Q0 a 1 0.5 n\n")
    (tmp_path / "runs.csv").write_text("system,query_id,doc_id,score\nm,q1,a,2\nm,q1,b,1\nn,q1,a,0.5\n")
    (tmp_path / "runs.jsonl").write_text(
        json.dumps({"system": "m", "query_id": "q1", "scores": {"a": 2, "b": 1}})
        + "\n"
        + json.dumps({"system": "n", "query_id": "q1", "doc_id": "a", "score": 0.5})
        + "\n"
    )

    loaded = [load_rankings(tmp_path / name) for name in ("runs.tsv", "runs.trec", "runs.csv", "runs.jsonl")]

    expected = Rankings.concat(
        [
            Rankings.from_scores({"q1": {"a": 2.0, "b": 1.0}}, system="m"),
            Rankings.from_scores({"q1": {"a": 0.5}}, system="n"),
        ]
    )
    assert loaded == [expected] * 4
    assert loaded[0].for_query("q1", system="n") == {"a": 0.5}
    assert list(loaded[0].to_pandas().columns) == ["system", "dataset", "query_id", "doc_id", "score"]


def test_an_order_without_scores_keeps_its_order(tmp_path: Path) -> None:
    (tmp_path / "order.json").write_text(json.dumps({"q1": ["c", "a", "b"]}))

    rankings = load_rankings(tmp_path / "order.json")

    assert rankings.for_query("q1") == {"c": 3.0, "a": 2.0, "b": 1.0}
    assert Rankings.from_orders({"q1": ["c", "a", "b"]}) == rankings
    assert rankings.top(2).for_query("q1") == {"c": 3.0, "a": 2.0}


def test_bad_rankings_are_data_errors(tmp_path: Path) -> None:
    (tmp_path / "nan.jsonl").write_text(json.dumps({"query_id": "q1", "scores": {"a": float("nan")}}) + "\n")
    (tmp_path / "twice.trec").write_text("q1 Q0 a 1 2.0 m\nq1 Q0 a 2 1.0 m\n")
    (tmp_path / "short.trec").write_text("q1 a 2.0\n")

    with pytest.raises(DataError, match="not finite"):
        load_rankings(tmp_path / "nan.jsonl")
    with pytest.raises(DataError, match="duplicate score"):
        load_rankings(tmp_path / "twice.trec")
    with pytest.raises(DataError, match="6 fields"):
        load_rankings(tmp_path / "short.trec")
    (tmp_path / "flat.jsonl").write_text(json.dumps({"query_id": "q1", "doc_id": "a"}) + "\n")
    with pytest.raises(DataError, match="flat.jsonl:1: .*needs its 'score'"):
        load_rankings(tmp_path / "flat.jsonl")
    with pytest.raises(MissingInputError):
        load_rankings(tmp_path / "absent.parquet")
    with pytest.raises(DataError, match="name one of"):
        Rankings.concat([Rankings.from_scores({"q": {"d": 1.0}}, system=s) for s in "mn"]).queries()


def test_the_accessors_read_the_one_dataset_present_and_refuse_to_guess_among_several() -> None:
    tagged = Rankings.from_scores({"q1": {"a": 1.0}}, dataset="tiny")
    suite = Rankings.concat([tagged, Rankings.from_scores({"q1": {"b": 1.0}}, dataset="other")])

    assert tagged.queries() == {"q1": {"a": 1.0}}
    assert tagged.for_query("q1") == {"a": 1.0}
    assert suite.for_query("q1", dataset="other") == {"b": 1.0}
    with pytest.raises(DataError, match="span 2 datasets") as refused:
        suite.queries()
    assert refused.value.details["datasets"] == ["tiny", "other"]
    with pytest.raises(DataError, match="no rankings of dataset 'absent'"):
        suite.queries(dataset="absent")


def test_rows_that_name_no_dataset_rank_any_dataset() -> None:
    undivided = Rankings.from_scores({"q1": {"a": 1.0}})

    assert undivided.resolve_dataset("biology") == ""
    assert undivided.queries(dataset="biology") == {"q1": {"a": 1.0}}
    assert Rankings.from_scores({"q1": {"a": 1.0}}, dataset="x").resolve_dataset("biology") is None


def test_a_trec_run_of_one_subset_is_tagged_with_its_dataset(tmp_path: Path) -> None:
    (tmp_path / "biology.trec").write_text("0 Q0 a 1 2.0 m\n")
    (tmp_path / "tagged.csv").write_text("dataset,query_id,doc_id,score\nbiology,0,a,2\n")

    rankings = load_rankings(tmp_path / "biology.trec", dataset="biology")

    assert rankings.datasets == ["biology"]
    assert rankings.for_query("0", dataset="biology") == {"a": 2.0}
    with pytest.raises(DataError, match="already name the datasets"):
        load_rankings(tmp_path / "tagged.csv", dataset="earth_science")


def test_rankings_load_from_parquet_with_the_hub_column_names(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pd.DataFrame({"model": ["m"], "query-id": ["q1"], "corpus-id": ["a"], "score": [2.0]}).to_parquet(
        tmp_path / "r.parquet"
    )

    assert load_rankings(tmp_path / "r.parquet") == Rankings.from_scores({"q1": {"a": 2.0}}, system="m")


def test_a_compressed_jsonl_is_refused_with_the_fix(tmp_path: Path) -> None:
    """A gzip file read as text fails deep in a decoder; the reader names what to do instead."""
    import gzip

    rows = tmp_path / "rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "q1", "query": "a", "doc_ids": ["d1"], "docs": ["x"]}) + "\n")

    with pytest.raises(ConfigError, match="compressed") as caught:
        load_dataset(f"jsonl:{rows}")
    assert "gunzip" in (caught.value.hint or "")
