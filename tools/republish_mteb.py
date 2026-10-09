#!/usr/bin/env python3
"""Re-lay the published RCP-nDCG datasets in MTEB's exact push layout (decision 40), for the owner to push.

For every published repository (``rcp-ndcg-nanobeir``, ``rcp-ndcg-bright``, ``rcp-ndcg-trecdl``,
``rcp-ndcg-vidore-v3``) this re-writes every subset with :class:`rcp_ndcg.data.io.mteb.MtebWriter`, each at
the split its published task definition pins -- NanoBEIR ``train``, BRIGHT ``standard``, ViDoRe v3 ``test``
(the PR's split names; decision 40 dropped the earlier re-lay-everything-to-``test`` rule) -- and validates each
written directory by loading it back with mteb's own ``RetrievalDatasetLoader`` and comparing against the
loaded :class:`~rcp_ndcg.data.Dataset`: the qrels as integers, the queries, the corpus (media included), and
the pool mteb reads (``top_ranked``, exclusions folded out).

Usage::

    python tools/republish_mteb.py --out /tmp/republish
    python tools/republish_mteb.py --out /tmp/republish --repo rcp-ndcg-nanobeir --no-card

The reads go to the Hugging Face Hub at the revisions the checks were published against
(``experiments/fetch_data.py``); nothing is pushed anywhere: the owner reviews ``--out`` and pushes each
repository themselves (``hf upload <owner>/<repo> <out>/<repo> . --repo-type dataset``) together with the move
to a Hugging Face organisation. The written card comes from the repository's published task metadata (mteb's
own template); the current repositories' hand-written usage cards are not reproduced -- the owner may merge
them.

The task definitions (each published ``rcp_ndcg_tasks.py``) carry the task prompt (mteb's
``TaskMetadata.prompt``) and the real split; the converter refuses a definition whose subset or split does not
match the data (mteb itself silently falls back to a config's only split, so the mismatch would otherwise be a
wrong label, not an error). The owner bumps the task file's ``_REVISION``/``dataset.revision`` to the pushed
commit after uploading: the new SHA cannot exist before the push.
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

TASK_FILE = "rcp_ndcg_tasks.py"
"""The published task definitions of a repository: read as data, never executed."""

_REVISIONS: dict[str, str] = {}
"""repo -> the revision the checks were published at, read lazily from ``experiments/fetch_data.py``."""


def _published_revision(repo: str) -> str | None:
    global _REVISIONS
    if not _REVISIONS:
        import importlib.util

        # tools/ is not a package and the script's sys.path[0] is tools/, so the import cannot go through the
        # package tree: the file is loaded by path from the repository root this tool ships in.
        source = Path(__file__).resolve().parents[1] / "experiments" / "fetch_data.py"
        spec = importlib.util.spec_from_file_location("rcp_fetch_data", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REVISIONS = dict(module.DATASETS)
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
    source = _task_source(repo)
    _check_task_definitions(repo, dataset, source)
    target = out / repo
    written = MtebWriter().write_dataset(dataset, str(target), card=_card(repo, dataset, source) if card else None)
    if written == 0:
        raise DataError(f"{repo}: no corpus rows were written")
    subsets = {part.name: _validate(part, target) for part in dataset.parts}
    return {"repo": repo, "revision": dataset.revision, "documents": written, "subsets": subsets}


def _load(uri: str, revision: str | None) -> Dataset:
    """The published dataset at one commit: every subset of the repository."""
    return load_dataset(uri, revision=revision)


def _task_source(repo: str) -> str:
    """The text of the repository's published ``rcp_ndcg_tasks.py`` (the task definitions, read as data)."""
    from rcp_ndcg.eval.mteb import _hub_text

    return _hub_text(f"{OWNER}/{repo}", TASK_FILE)


def _definitions(source: str, subset: str) -> list[dict[str, Any]]:
    """Every task definition of *subset* in a published task file, in file order.

    The 2026-10 files key ``_TASK_METADATA`` by published task name and ship ``_SUBSETS``; older files key it
    by the subset names themselves. A task's ``eval_langs`` keys are the language subsets (ViDoRe v3's are
    ``domain__language``), so a dataset part matches by its own name; the alias map covers a subset the
    languages do not spell out. ViDoRe v3 maps a subset to two tasks (the page-image one and its OCR view), so
    every match is returned and every one's split is checked.
    """
    from rcp_ndcg.eval.mteb import task_metadata, task_subsets

    _, table = task_metadata(source)
    aliases = task_subsets(source)
    matches = [fields for fields in table.values() if subset in (fields.get("eval_langs") or {})]
    task_name = aliases.get(subset)
    if task_name in table:
        matches.append(table[task_name])
    if subset in table:  # an older file keyed by the subset names themselves
        matches.append(table[subset])
    return matches


def _subsets(source: str) -> list[str]:
    """The subset names a published task file defines, name-sorted (the card's deterministic pick)."""
    from rcp_ndcg.eval.mteb import task_metadata, task_subsets

    _, table = task_metadata(source)
    aliases = task_subsets(source)
    names = set(aliases)
    for fields in table.values():
        languages = fields.get("eval_langs")
        if isinstance(languages, dict):
            names.update(languages)
    return sorted(names)


def _check_task_definitions(repo: str, dataset: Dataset, source: str) -> None:
    """Refuse a task definition whose subset or split does not match the data (decision 40, review R7)."""
    for part in dataset.parts:
        matches = _definitions(source, part.name)
        if not matches:
            raise DataError(
                f"{repo}: the published task file defines no task for subset {part.name!r}",
                hint=f"available subsets: {_subsets(source)}",
            )
        for fields in matches:
            splits = [str(split) for split in fields.get("eval_splits") or []]
            if part.split not in splits:
                raise DataError(
                    f"{repo}: subset {part.name!r} is published at split {part.split!r}, and its task "
                    f"definition {fields.get('name')!r} names {splits}",
                    hint="fix the task definition's `eval_splits` (the split the data really has), or the data",
                )


def _card(repo: str, dataset: Dataset, source: str) -> Any:
    """The published task metadata of the repository's first subset (name-sorted, deterministic), pointed at
    the target repository: mteb's own card template renders it."""
    name = _subsets(source)[0]
    fields = dict(_definitions(source, name)[0])
    fields["dataset"] = {"path": f"{OWNER}/{repo}", "revision": dataset.revision}
    return fields


def _validate(part: Dataset, target: Path) -> dict[str, Any]:
    """The subset's written layout must load back, through mteb's own ``RetrievalDatasetLoader``, to what we
    hold: the qrels as integers, the queries (mteb keeps the qrels-bearing ones), the corpus (media included),
    and the pool (the candidates, exclusions folded out)."""
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    loaded = RetrievalDatasetLoader(hf_repo=str(target), revision="main", split=part.split, config=part.name).load()
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
    if "instruction" in loaded["queries"].column_names:
        instructions = dict(zip(loaded["queries"]["id"], loaded["queries"]["instruction"], strict=True))
        expected_instructions = {
            query_id: query.instruction
            for query_id, query in part.queries.items()
            if query_id in part.qrels and query.instruction is not None
        }
        if instructions != expected_instructions:
            raise DataError(f"{part.name}: mteb's loader reads instructions that differ from the dataset's")
    corpus = dict(zip(loaded["corpus"]["id"], loaded["corpus"]["text"], strict=True))
    expected_corpus = {doc_id: document.text for doc_id, document in part.corpus.items()}
    if corpus != expected_corpus:
        raise DataError(f"{part.name}: mteb's loader reads {len(corpus)} documents, ours are {len(expected_corpus)}")
    for column in ("image", "video"):
        if column not in loaded["corpus"].column_names:
            continue
        written_media = _written_media(loaded, column)
        expected_media = _expected_media(part, column)
        if written_media != expected_media:
            raise DataError(
                f"{part.name}: mteb's loader reads {len(written_media)} {column} assets, ours are {len(expected_media)}"
            )
    if part.candidates is None and not part.excluded:
        expected_pool = None
    else:
        # the pool mteb reads: the candidates, exclusions folded out; without candidates, the corpus minus
        # them (the writer's derivation)
        pooled = part.candidates if part.candidates is not None else {q: list(part.corpus) for q in part.qrels}
        expected_pool = {
            query_id: [doc_id for doc_id in ids if doc_id not in part.excluded.get(query_id, ())]
            for query_id, ids in pooled.items()
        }
    if loaded["top_ranked"] != expected_pool:
        raise DataError(f"{part.name}: mteb's loader reads a pool that differs from ours")
    return {
        "queries": len(integer_qrels),
        "documents": len(expected_corpus),
        "qrels": sum(len(docs) for docs in integer_qrels.values()),
        "pool": len(expected_pool) if expected_pool else 0,
    }


def _written_media(loaded: dict[str, Any], column: str) -> dict[str, bytes]:
    """``{doc_id: bytes}`` of one media column, as mteb's loader reads it back (raw, no decode)."""
    from datasets import Image, Video

    feature = Image if column == "image" else Video
    raw = loaded["corpus"].cast_column(column, feature(decode=False))
    return {str(doc_id): cell["bytes"] for doc_id, cell in zip(raw["id"], raw[column], strict=True) if cell is not None}


def _expected_media(part: Dataset, column: str) -> dict[str, bytes]:
    """``{doc_id: bytes}`` of one media column, from the dataset's content parts (what the writer resolves)."""
    from rcp_ndcg_core.content import ImagePart, VideoPart

    from rcp_ndcg.data.media import default_resolver

    part_type = ImagePart if column == "image" else VideoPart
    expected: dict[str, bytes] = {}
    for doc_id, document in part.corpus.items():
        content = document.content
        if content is None:
            continue
        refs = [p.ref for p in content.parts if isinstance(p, part_type) and p.ref is not None]
        if refs:
            expected[str(doc_id)] = default_resolver().bytes_of(refs[0])
    return expected


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
