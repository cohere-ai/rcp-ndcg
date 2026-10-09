#!/usr/bin/env python3
"""Re-lay the published RCP-nDCG datasets in MTEB's exact push layout (decision 31), for the owner to push.

For every published repository (``rcp-ndcg-nanobeir``, ``rcp-ndcg-bright``, ``rcp-ndcg-trecdl``,
``rcp-ndcg-vidore-v3``) this re-writes every subset with :class:`rcp_ndcg.data.io.mteb.MtebWriter` -- one
config per (subset, part) at the eval split ``test`` (the published repositories still say ``train``) -- and
validates each written directory by loading it back with mteb's own ``RetrievalDatasetLoader`` and comparing
against the loaded :class:`~rcp_ndcg.data.Dataset`: the qrels as integers, the queries, the corpus, and the
pool mteb reads (``top_ranked``, exclusions folded out).

Usage::

    python tools/republish_mteb.py --out /tmp/republish
    python tools/republish_mteb.py --out /tmp/republish --repo rcp-ndcg-nanobeir --no-card

The reads go to the Hugging Face Hub at the revisions the checks were published against
(``experiments/fetch_data.py``); nothing is pushed anywhere: the owner reviews ``--out`` and pushes each
repository themselves (``hf upload <owner>/<repo> <out>/<repo> . --repo-type dataset``) together with the move
to a Hugging Face organisation. The written card comes from the repository's published task metadata (mteb's
own template); the current repositories' hand-written usage cards are not reproduced -- the owner may merge
them. ``rcp-ndcg-vidore-v3`` is a page-image corpus: the text layout cannot hold it, and the writer refuses it
(exporting media is its own work, deferred with the media decisions).

The task definitions (each published ``rcp_ndcg_tasks.py``) still name split ``train`` and this repository's
owner; aligning them with the owner's local mteb PR is an open item and not done here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rcp_ndcg.data.dataset import Dataset, load_dataset
from rcp_ndcg.data.io.mteb import MtebWriter
from rcp_ndcg.errors import DataError

OWNER = "fabianschmidt-cohere"
"""The owner the published repositories live under today; the push moves them to an organisation."""

REPOS: tuple[str, ...] = ("rcp-ndcg-nanobeir", "rcp-ndcg-bright", "rcp-ndcg-trecdl", "rcp-ndcg-vidore-v3")
"""The published MTEB retrieval repositories (``rcp-ndcg-external-validation`` holds no retrieval dataset)."""

EVAL_SPLIT = "test"
"""The eval split the republished datasets carry (the published ones say ``train``)."""

_REVISIONS: dict[str, str] = {}
"""repo -> the revision the checks were published at, read lazily from ``experiments/fetch_data.py``."""


def _published_revision(repo: str) -> str | None:
    global _REVISIONS
    if not _REVISIONS:
        import experiments.fetch_data as fetch

        _REVISIONS = dict(fetch.DATASETS)
    return _REVISIONS.get(repo)


def republish(
    repo: str, out: Path, *, owner: str = OWNER, revision: str | None = None, card: bool = True
) -> dict[str, Any]:
    """Re-lay one published repository under ``out/<repo>`` and validate it with mteb's loader.

    Args:
        repo: The published repository name (a member of :data:`REPOS`, or any other with the same layout).
        out: The output directory; ``out/<repo>`` is written.
        owner: The Hub owner the published repositories live under.
        revision: The revision to read; the default is the one the checks were published at.
        card: Render the README's card from the repository's published task metadata (mteb's template).

    Returns:
        ``{"repo", "revision", "subsets": {name: {"queries", "documents", "qrels"}}}``.

    Raises:
        DataError: The written layout does not load back to the dataset mteb's own loader reads.
        ConfigError: A subset cannot be written (media, for instance).
    """
    uri = f"hf://{owner}/{repo}"
    dataset = _load(uri, revision=revision or _published_revision(repo))
    target = out / repo
    written = MtebWriter().write_dataset(
        dataset, str(target), split=EVAL_SPLIT, card=_card(repo, dataset) if card else None
    )
    if written == 0:
        raise DataError(f"{repo}: no corpus rows were written")
    subsets = {part.name: _validate(part, target) for part in dataset.parts}
    return {"repo": repo, "revision": dataset.revision, "documents": written, "subsets": subsets}


def _load(uri: str, revision: str | None) -> Dataset:
    """The published dataset at one commit: every subset of the repository."""
    return load_dataset(uri, revision=revision)


def _card(repo: str, dataset: Dataset) -> Any:
    """The published task metadata of the repository's first subset (name-sorted, deterministic), pointed at
    the target repository: mteb's own card template renders it."""
    from rcp_ndcg.eval.mteb import _hub_text, task_metadata

    _, table = task_metadata(_hub_text(f"{OWNER}/{repo}", "rcp_ndcg_tasks.py"))
    name = sorted(table)[0]
    fields = dict(table[name])
    fields["dataset"] = {"path": f"{OWNER}/{repo}", "revision": dataset.revision}
    return fields


def _validate(part: Dataset, target: Path) -> dict[str, Any]:
    """The subset's written layout must load back, through mteb's own ``RetrievalDatasetLoader``, to what we
    hold: the qrels as integers, the queries (mteb keeps the qrels-bearing ones), the corpus, and the pool
    (the candidates, exclusions folded out)."""
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    loaded = RetrievalDatasetLoader(hf_repo=str(target), revision="main", split=EVAL_SPLIT, config=part.name).load()
    integer_qrels = {q: {d: int(g) for d, g in docs.items()} for q, docs in part.qrels.items()}
    if loaded["relevant_docs"] != integer_qrels:
        raise DataError(
            f"{part.name}: mteb's loader reads qrels that differ from the dataset's "
            f"({len(loaded['relevant_docs'])} queries vs {len(integer_qrels)})"
        )
    queries = dict(zip(loaded["queries"]["id"], loaded["queries"]["text"], strict=True))
    ours = {query_id: query.text for query_id, query in part.queries.items() if query_id in part.qrels}
    if queries != ours:
        raise DataError(f"{part.name}: mteb's loader reads {len(queries)} queries, ours are {len(ours)}")
    corpus = dict(zip(loaded["corpus"]["id"], loaded["corpus"]["text"], strict=True))
    expected_corpus = {doc_id: document.text for doc_id, document in part.corpus.items()}
    if corpus != expected_corpus:
        raise DataError(f"{part.name}: mteb's loader reads {len(corpus)} documents, ours are {len(expected_corpus)}")
    expected_pool = (
        None
        if part.candidates is None and not part.excluded
        else {
            query_id: [
                doc_id
                for doc_id in (part.candidates.get(query_id) if part.candidates is not None else list(part.corpus))
                if doc_id not in part.excluded.get(query_id, ())
            ]
            for query_id in part.qrels
        }
    )
    if loaded["top_ranked"] != expected_pool:
        raise DataError(f"{part.name}: mteb's loader reads a pool that differs from ours")
    return {
        "queries": len(integer_qrels),
        "documents": len(expected_corpus),
        "qrels": sum(len(docs) for docs in integer_qrels.values()),
        "pool": len(expected_pool) if expected_pool else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the output directory; one subdirectory per repo")
    parser.add_argument(
        "--repo", action="append", choices=sorted(REPOS), help="a repo to convert (repeatable; default: all)"
    )
    parser.add_argument("--revision", default=None, help="the revision to read; default: the published one")
    parser.add_argument("--owner", default=OWNER, help="the Hub owner the published repositories live under")
    parser.add_argument("--no-card", action="store_true", help="write the configs front matter only, no card")
    args = parser.parse_args(argv)

    failures: dict[str, str] = {}
    for repo in args.repo or REPOS:
        try:
            summary = republish(repo, args.out, owner=args.owner, revision=args.revision, card=not args.no_card)
        except Exception as exc:  # noqa: BLE001 - one repo's failure must not stop the others
            failures[repo] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {repo}: {failures[repo]}", file=sys.stderr, flush=True)
            continue
        parts = ", ".join(f"{name}: {part['queries']} queries" for name, part in summary["subsets"].items())
        print(f"[ok] {summary['repo']} @ {summary['revision']}: {summary['documents']} documents ({parts})")
    if failures:
        print(f"{len(failures)} of {len(args.repo or REPOS)} repositories failed; nothing was pushed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
