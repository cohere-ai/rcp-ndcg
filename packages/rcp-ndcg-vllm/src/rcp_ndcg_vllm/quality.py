"""The T3 quality stage: the model through rcp-ndcg's served path and through ``mteb``, compared.

GPU-VALIDATION.md T3 ("MTEB across modalities") runs every model of the task matrix
(:data:`TASK_MATRIX`) through **rcp-ndcg's own served path** (``rcp-ndcg retrieval index|search|rerank``
then ``rcp-ndcg eval score``) and, on the same tasks, through the reference implementation
(``mteb`` with the HF/sentence-transformers model).  This module is the stage: the task matrix as
data, the exact command lines the node runs (:func:`served_commands`, :func:`reference_command`), the
comparison table with its gates and its deviation-note column (:func:`comparison_rows`), the
``QUALITY.md`` writer, and the golden-replay selection -- the FULL served exchanges of one NanoBEIR
and one ViDoRe subset (:func:`golden_replay_selection`), so a later replay reproduces the run's
nDCG@10 and RCP-nDCG@10 to 1e-9.

The runs themselves need the node (a served engine and the reference environment); everything here
computes and renders on CPU.

Public surface:

- :data:`TASK_MATRIX`, :data:`QUALITY_TOLERANCE`.
- :func:`served_commands`, :func:`reference_command`.
- :func:`comparison_rows`, :func:`quality_md`, :func:`golden_replay_selection`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "QUALITY_TOLERANCE",
    "TASK_MATRIX",
    "comparison_rows",
    "golden_replay_selection",
    "quality_md",
    "reference_command",
    "served_commands",
]

QUALITY_TOLERANCE = 0.005
"""The T3 gate: served vs reference nDCG@10 within 0.5 points per subset (a 0-to-1 scale)."""

TASK_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "family": "text embedders",
        "view": "retrieval",
        "models": [
            "zembed-1-embedding",
            "pplx-embed-v2-context-9b-preview",
            "qwen3-embedding-0.6b",
            "jina-embeddings-v5-text-small",
            "octen-embedding-8b",
        ],
        "suites": ["nanobeir (13 subsets)", "bright (all subsets)", "trecdl (2019, 2020)"],
    },
    {
        "family": "text rerankers",
        "view": "reranking (the paper's released pools)",
        "models": [
            "zerank-1-reranker",
            "zerank-1-small-reranker",
            "zerank-2-reranker",
            "qwen3-reranker-0.6b",
            "qwen3-reranker-4b",
            "qwen3-reranker-8b",
            "ctxl-rerank-v2-instruct-multilingual-1b",
            "ctxl-rerank-v2-instruct-multilingual-2b",
            "ctxl-rerank-v2-instruct-multilingual-6b",
            "jina-reranker-v3",
        ],
        "suites": ["nanobeir", "bright", "trecdl"],
    },
    {
        "family": "visual documents",
        "view": "retrieval (Qwen3-VL-Embedding-2B, topk-embed-v1-small) / reranking (Qwen3-VL-Reranker-2B)",
        "models": ["qwen3-vl-embedding-2b", "qwen3-vl-reranker-2b", "topk-embed-v1-small"],
        "suites": ["vidore v3 (all subsets; multilingual)"],
    },
    {
        "family": "late interaction, text",
        "view": "retrieval",
        "models": ["topk-embed-v1-small"],
        "suites": ["nanobeir", "bright"],
    },
)
"""GPU-VALIDATION.md's T3 task matrix as data (the public and paper recipes; the private recipes run
the same suites in their own repository)."""


def served_commands(
    *,
    view: str,
    model: str,
    engine_url: str,
    dataset: str,
    out_dir: str | Path,
    rerank: bool = False,
) -> list[list[str]]:
    """The exact served-path argv list the node runs for one task (index/search/rerank, then score).

    Inputs: the matrix view (``retrieval`` or ``reranking``), the recipe id, the engine's base URL, a
    dataset URI (``suite:nanobeir`` and friends) and the working directory.  Output: one argv per
    command -- ``rcp-ndcg retrieval index`` + ``search`` for the retrieval view (``search`` only with
    ``rerank`` beside an index), ``retrieval rerank`` for the reranking view, and
    ``rcp-ndcg eval score`` over the produced rankings.
    """
    out = Path(out_dir)
    if view.startswith("retrieval"):
        index = [
            "rcp-ndcg",
            "retrieval",
            "index",
            "--dataset",
            dataset,
            "--encoder-url",
            engine_url,
            "--out",
            str(out / "index"),
        ]
        search = [
            "rcp-ndcg",
            "retrieval",
            "search",
            "--dataset",
            dataset,
            "--index",
            str(out / "index"),
            "--encoder-url",
            engine_url,
            "--out",
            str(out / "rankings.jsonl"),
        ]
        if rerank:
            search += ["--rerank-url", engine_url]
        return [index, search, _score_argv(out / "rankings.jsonl", out / "dataset.jsonl")]
    rerank_run = [
        "rcp-ndcg",
        "retrieval",
        "rerank",
        "--dataset",
        dataset,
        "--rerank-url",
        engine_url,
        "--out",
        str(out / "rankings.jsonl"),
    ]
    return [rerank_run, _score_argv(out / "rankings.jsonl", out / "dataset.jsonl")]


def _score_argv(rankings: Path, dataset: Path) -> list[str]:
    return [
        "rcp-ndcg",
        "eval",
        "score",
        "--rankings",
        str(rankings),
        "--dataset",
        f"jsonl:{dataset}",
        "--metrics",
        "qrel_ndcg",
        "rcp_ndcg",
        "--k",
        "10",
        "--json",
        "--out",
        str(rankings.parent / "scores.json"),
    ]


def reference_command(*, model: str, task: str, out_dir: str | Path, revision: str | None = None) -> list[str]:
    """The mteb reference run's argv (the HF/sentence-transformers model on the same task)."""
    spec = model if revision is None else f"{model}@{revision}"
    return ["mteb", "run", "-m", spec, "-t", task, "--output-folder", str(out_dir)]


def comparison_rows(
    served: dict[str, float],
    reference: dict[str, float],
    *,
    paper: dict[str, float] | None = None,
    published: dict[str, float] | None = None,
    published_source: str = "",
    deviation_note: dict[str, str] | None = None,
    tolerance: float = QUALITY_TOLERANCE,
) -> dict[str, Any]:
    """The T3 comparison table (served vs reference vs the paper's numbers vs published ones).

    Inputs: per-subset nDCG@10 on a 0-to-1 scale for the served run and the mteb reference, optionally
    the paper's stored per-subset numbers (paper models) and a published column (model cards, the
    leaderboard) with its source and any pre-known deviation notes.  Output: ``{"rows": [...],
    "passed": bool, "n_subsets": ...}`` -- each row carries both deltas and a ``deviation_note``
    column; a published row that deviates keeps its explanation (a deviation note is required for
    every published mismatch), and ``passed`` gates the served-vs-reference delta at
    :data:`QUALITY_TOLERANCE` per subset (paper deltas are informational and reported).
    """
    notes = deviation_note or {}
    rows: list[dict[str, Any]] = []
    for subset in sorted(set(served) | set(reference)):
        served_value = served.get(subset)
        reference_value = reference.get(subset)
        delta = (
            abs(served_value - reference_value) if served_value is not None and reference_value is not None else None
        )
        paper_value = (paper or {}).get(subset)
        published_value = (published or {}).get(subset)
        note = notes.get(subset, "")
        if published_value is not None and served_value is not None:
            published_delta = abs(served_value - published_value)
            if published_delta > tolerance and not note:
                note = (
                    f"unexplained deviation from the published number ({published_source or 'model card'}): "
                    f"{published_delta * 100:.2f} points"
                )
        rows.append(
            {
                "subset": subset,
                "served_ndcg10": served_value,
                "reference_ndcg10": reference_value,
                "delta_vs_reference": delta,
                "paper_ndcg10": paper_value,
                "delta_vs_paper": abs(served_value - paper_value)
                if served_value is not None and paper_value is not None
                else None,
                "published_ndcg10": published_value,
                "published_source": published_source,
                "deviation_note": note,
                "within_tolerance": bool(delta is not None and delta <= tolerance),
            }
        )
    passed = bool(rows) and all(row["within_tolerance"] for row in rows)
    return {"rows": rows, "passed": passed, "n_subsets": len(rows), "tolerance": tolerance}


def quality_md(model: str, comparison: dict[str, Any]) -> str:
    """``QUALITY.md``, one table row per subset with its deviation note and the gates' verdict."""
    lines = [
        f"# Quality report: {model}",
        "",
        f"- tolerance: served vs reference within {comparison['tolerance'] * 100:.1f} points of nDCG@10 per subset",
        "",
        "| subset | served | reference | delta | paper | delta | published | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison["rows"]:
        lines.append(
            "| {subset} | {served} | {ref} | {delta} | {paper} | {paper_delta} | {published} | {note} |".format(
                subset=row["subset"],
                served=_fmt(row["served_ndcg10"]),
                ref=_fmt(row["reference_ndcg10"]),
                delta=_fmt(row["delta_vs_reference"]),
                paper=_fmt(row["paper_ndcg10"]),
                paper_delta=_fmt(row["delta_vs_paper"]),
                published=_fmt(row["published_ndcg10"]),
                note=str(row["deviation_note"]).replace("|", "\\|"),
            )
        )
    lines += ["", f"Verdict: **{'PASS' if comparison['passed'] else 'FAIL'}**"]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def golden_replay_selection(
    records: list[dict[str, Any]],
    *,
    subsets: tuple[str, ...] = ("one nanobeir subset", "one vidore subset"),
) -> dict[str, Any]:
    """The golden-replay selection: the FULL served exchanges of one NanoBEIR and one ViDoRe subset.

    Inputs: the corpus's records and the subset names to keep (their ``inputs.source.subset``).  The
    selection keeps every record of the chosen subsets (probe and tokenize records excluded -- the
    replay re-derives them), so a replay against the verified emulators reproduces the run's nDCG@10
    and RCP-nDCG@10 to 1e-9.  Output: ``{"records": [...], "count": ..., "subsets": [...]}``.
    """
    chosen = [
        record
        for record in records
        if str((record.get("inputs", {}).get("source") or {}).get("subset")) in subsets
        and record.get("inputs", {}).get("probe") in (None, "ok")
    ]
    return {"records": chosen, "count": len(chosen), "subsets": sorted(subsets)}


def write_golden_replay(selection: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the golden-replay capture (``golden-replay.jsonl`` + its ``index.json``) and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "golden-replay.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selection["records"]),
        encoding="utf-8",
    )
    (out / "index.json").write_text(
        json.dumps({"count": selection["count"], "subsets": selection["subsets"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
