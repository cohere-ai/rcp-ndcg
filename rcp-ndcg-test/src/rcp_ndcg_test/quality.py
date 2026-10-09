"""The T3 quality stage: the model through rcp-ndcg's served path and through ``mteb``, compared.

GPU-VALIDATION.md T3 ("MTEB across modalities") runs every model of the task matrix (:data:`TASK_MATRIX`)
through **rcp-ndcg's own served path** -- ``rcp-ndcg retrieval index`` + ``search`` (embedders, the retrieval
view) or ``retrieval rerank`` over the suite's released pools (rerankers, the reranking view), then
``rcp-ndcg eval score`` -- and, on the same tasks, through the reference implementation: ``mteb`` with the
HF/sentence-transformers model, over the product's own task definitions (:func:`rcp_ndcg.eval.mteb.get_tasks`),
run as a subprocess in the reference environment (``python -m rcp_ndcg_test.quality reference``).  The stage
compares served vs reference per subset and metric (RCP-nDCG@10 and qrel-nDCG@10 within 0.5 points), and for
paper models served vs the paper's stored per-subset numbers (also gated); published numbers are a sanity
column whose every deviation carries a note.  The served exchanges of one NanoBEIR and one ViDoRe subset are
captured in full through a recording proxy (:class:`RecordingProxy`) as the golden-replay corpus.

Public surface:

- :data:`TASK_MATRIX`, :data:`QUALITY_TOLERANCE`, :class:`QualityTask`, :func:`tasks_for`.
- :func:`retriever_config`, :func:`reranker_config`, :func:`served_commands`, :func:`reference_argv`.
- :func:`served_scores`, :func:`reference_scores`, :func:`comparison_rows`, :func:`quality_md`.
- :func:`run_quality` (the stage), :class:`RecordingProxy` (the golden-replay capture).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

from .errors import HarnessError

__all__ = [
    "METRICS",
    "QUALITY_TOLERANCE",
    "TASK_MATRIX",
    "QualityTask",
    "RecordingProxy",
    "comparison_rows",
    "quality_md",
    "reference_argv",
    "reference_scores",
    "reranker_config",
    "retriever_config",
    "run_quality",
    "served_commands",
    "served_scores",
    "tasks_for",
]

QUALITY_TOLERANCE = 0.005
"""The T3 gate: within 0.5 points of nDCG@10 per subset (a 0-to-1 scale)."""

METRICS: tuple[str, ...] = ("rcp_ndcg", "qrel_ndcg")
"""The metrics compared at k = 10 (RCP-nDCG@10 and qrel-nDCG@10)."""

TASK_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "family": "text embedders",
        "view": "retrieval",
        "models": [
            "zembed-1-embedding",
            "pplx-embed-v2-context-9b-preview",
            "qwen3-embedding-0.6b",
            "jina-embeddings-v5-text-nano",
            "jina-embeddings-v5-text-small",
            "octen-embedding-0.6b",
            "octen-embedding-4b",
            "octen-embedding-8b",
            "harrier-oss-v1-270m",
            "harrier-oss-v1-0.6b",
            "harrier-oss-v1-27b",
        ],
        "suites": ["nanobeir", "bright", "trecdl"],
    },
    {
        "family": "text rerankers",
        "view": "reranking",
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
        "family": "visual documents (retrieval)",
        "view": "retrieval",
        "models": [
            "qwen3-vl-embedding-2b",
            "embeddinggemma-2",
            "topk-embed-v1-xsmall",
            "topk-embed-v1-small",
            "pplx-embed-v2-late-0.6b",
        ],
        "suites": ["vidore"],
    },
    {
        "family": "visual documents (reranking)",
        "view": "reranking",
        "models": ["qwen3-vl-reranker-2b"],
        "suites": ["vidore"],
    },
    {
        "family": "late interaction, text",
        "view": "retrieval",
        "models": ["topk-embed-v1-xsmall", "topk-embed-v1-small", "pplx-embed-v2-late-0.6b"],
        "suites": ["nanobeir", "bright"],
    },
)
"""GPU-VALIDATION.md's T3 task matrix as data: per family, the view and the suites (every subset of each, at the
generator's pinned dataset commits).  The private recipes run the same suites in their own repository."""

SEARCH_DEPTH = 100
"""Documents retrieved per query in the retrieval view (nDCG@10 reads the top 10)."""

RERANK_DEPTH = 1000
"""Candidates reranked per query in the reranking view: every document of the released pool."""


@dataclass(frozen=True)
class QualityTask:
    """One T3 task: a model on one subset of one suite, in one view, at a pinned data revision.

    Attributes:
        model: The recipe id.
        suite: ``nanobeir``, ``bright``, ``trecdl`` or ``vidore``.
        subset: The suite's subset (its dataset name).
        view: ``retrieval`` (full-corpus search) or ``reranking`` (the released pools).
        revision: The dataset commit (the request generator's pin).
        golden: Whether the served exchanges are captured in full for the golden replay.
    """

    model: str
    suite: str
    subset: str
    view: str
    revision: str
    golden: bool = False

    @property
    def key(self) -> str:
        """The task's directory name."""
        return f"{self.suite}/{self.subset}"


def tasks_for(recipe_id: str) -> list[QualityTask]:
    """Every T3 task of one recipe, from :data:`TASK_MATRIX` (each suite's subsets, the pinned commits).

    The first NanoBEIR and the first ViDoRe subset of the recipe's plan are its golden-replay tasks.  Raises
    :class:`HarnessError` for a recipe the matrix does not name.
    """
    from .observe.requests import PINNED_DATASET_COMMITS
    from .observe.sources import SUITE_REPOS, SUITE_SUBSETS

    tasks: list[QualityTask] = []
    golden_done: set[str] = set()
    for family in TASK_MATRIX:
        if recipe_id not in family["models"]:
            continue
        for suite in family["suites"]:
            for subset in SUITE_SUBSETS[suite]:
                golden = suite in ("nanobeir", "vidore") and suite not in golden_done
                golden_done.add(suite)
                tasks.append(
                    QualityTask(
                        model=recipe_id,
                        suite=suite,
                        subset=subset,
                        view=str(family["view"]),
                        revision=PINNED_DATASET_COMMITS[SUITE_REPOS[suite]],
                        golden=golden,
                    )
                )
    if not tasks:
        raise HarnessError(f"recipe {recipe_id} is not in the T3 task matrix (quality.TASK_MATRIX)")
    return tasks


def _endpoint(recipe: Recipe, engine_url: str) -> dict[str, Any]:
    """The recipe's endpoint config for the product's CLI, pointed at the engine (the client's own config).

    A shipped recipe keeps its ``recipe: <id>`` pointer (the product re-resolves the block at the read, the
    versioned contract, decision 18); a fixture recipe's id names nothing shipped, so the pointer is dropped
    (it would be read as the mapping form and refused at the resolution)."""
    from rcp_ndcg_vllm.recipe import client_config

    from .equivalence.fitting import resolved_tokenizer_spec
    from .equivalence.wire import _openai_base

    url = _openai_base(engine_url) if recipe.role == "embed" else engine_url.rstrip("/")
    data = client_config(recipe, base_url=url)
    data["tokenizer"] = resolved_tokenizer_spec(recipe)
    if recipe.id not in _shipped_variant_ids():
        data.pop("recipe", None)
    return {key: value for key, value in data.items() if value is not None}


def _shipped_variant_ids() -> frozenset[str]:
    """The shipped recipes' variant ids (decision 34: the families' variants, never the family ids)."""
    from rcp_ndcg_vllm.errors import RecipeError
    from rcp_ndcg_vllm.recipe import default_recipes_root, load_family

    root = default_recipes_root()
    ids: set[str] = set()
    for directory in sorted(root.iterdir()):
        if not (directory / "family.yaml").is_file():
            continue
        try:
            ids.update(variant.id for variant in load_family(directory).variants)
        except RecipeError:
            continue
    return frozenset(ids)


def retriever_config(recipe: Recipe, engine_url: str) -> dict[str, Any]:
    """The retriever YAML data of an embedder recipe, validated by the product's own parser.

    ``kind: dense`` with the recipe's ``openai_embeddings`` endpoint, or ``kind: late_interaction`` with its
    ``vllm_pooling`` one.  Raises :class:`HarnessError` for a reranker recipe or a config the product refuses.
    """
    from rcp_ndcg.errors import RcpNdcgError
    from rcp_ndcg.retrieval.config import validate_retriever

    kinds = {"embed": "dense", "multi_vector": "late_interaction"}
    if recipe.role not in kinds:
        raise HarnessError(f"recipe {recipe.id} is a {recipe.role} recipe: the retrieval view needs an embedder")
    data = {"kind": kinds[recipe.role], "encoder": _endpoint(recipe, engine_url)}
    try:
        validate_retriever(data)
    except RcpNdcgError as error:
        raise HarnessError(f"recipe {recipe.id}: the product refuses its retriever config: {error}") from error
    return data


def reranker_config(recipe: Recipe, engine_url: str) -> dict[str, Any]:
    """The reranker YAML data of a rerank recipe (its ``rerank`` endpoint), validated by the product's parser."""
    from rcp_ndcg.errors import RcpNdcgError
    from rcp_ndcg.retrieval.config import validate_reranker

    if recipe.role != "rerank":
        raise HarnessError(f"recipe {recipe.id} is a {recipe.role} recipe: the reranking view needs a reranker")
    data = _endpoint(recipe, engine_url)
    try:
        validate_reranker(data)
    except RcpNdcgError as error:
        raise HarnessError(f"recipe {recipe.id}: the product refuses its reranker config: {error}") from error
    return data


def _dataset_args(task: QualityTask) -> list[str]:
    return ["--dataset", f"suite:{task.suite}", "--subset", task.subset, "--revision", task.revision]


def served_commands(task: QualityTask, recipe: Recipe, engine_url: str, work_dir: str | Path) -> list[list[str]]:
    """The served-path argv list of one task, the product's CLI only; writes the config YAML it names.

    Retrieval view: ``retrieval index`` (the recipe's retriever YAML), ``retrieval search`` over that index.
    Reranking view: the suite's released pools as a rankings file (``python -m rcp_ndcg_test.quality pools``,
    through the product's dataset loader), then ``retrieval rerank`` with the recipe's reranker YAML.  Then
    ``eval score`` of the rankings under the suite's protocol, both metrics at k = 10, the report to
    ``scores.json``.  Every flag is the CLI's own (a test parses each argv with the product's click tree).
    """
    import yaml

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rankings = work / "rankings.jsonl"
    commands: list[list[str]]
    if task.view == "retrieval":
        config = work / "retriever.yaml"
        config.write_text(yaml.safe_dump(retriever_config(recipe, engine_url), sort_keys=True), encoding="utf-8")
        commands = [
            [
                "rcp-ndcg", "retrieval", "index", *_dataset_args(task), "--retriever", str(config),
                "--out", str(work / "index"),
            ],
            [
                "rcp-ndcg", "retrieval", "search", *_dataset_args(task), "--index", str(work / "index"),
                "--out", str(rankings), "--depth", str(SEARCH_DEPTH),
            ],
        ]  # fmt: skip
    elif task.view == "reranking":
        config = work / "reranker.yaml"
        config.write_text(yaml.safe_dump(reranker_config(recipe, engine_url), sort_keys=True), encoding="utf-8")
        pools = work / "pools.jsonl"
        commands = [
            [
                sys.executable, "-m", "rcp_ndcg_test.quality", "pools", "--suite", task.suite, "--subset", task.subset,
                "--revision", task.revision, "--out", str(pools),
            ],
            [
                "rcp-ndcg", "retrieval", "rerank", *_dataset_args(task), "--rankings", str(pools),
                "--reranker", str(config), "--out", str(rankings), "--depth", str(RERANK_DEPTH),
            ],
        ]  # fmt: skip
    else:
        raise HarnessError(f"unknown T3 view {task.view!r}: retrieval or reranking")
    commands.append(
        [
            "rcp-ndcg", "eval", "score", "--rankings", str(rankings), "--suite", task.suite, "--subset", task.subset,
            "--revision", task.revision, "--metrics", "rcp_ndcg", "--metrics", "qrel_ndcg", "--k", "10",
            "--out", str(work / "scores.json"), "--json",
        ]
    )  # fmt: skip
    return commands


def reference_argv(task: QualityTask, recipe: Recipe, work_dir: str | Path, reference_python: str) -> list[str]:
    """The reference run's argv: ``mteb`` with the HF model on the same task, in the reference environment."""
    return [
        reference_python, "-m", "rcp_ndcg_test.quality", "reference", "--model", recipe.model,
        "--revision", recipe.revision, "--suite", task.suite, "--subset", task.subset, "--view", task.view,
        "--data-revision", task.revision, "--out", str(Path(work_dir) / "reference.json"),
    ]  # fmt: skip


def served_scores(report_path: str | Path, subset: str) -> dict[str, float]:
    """``{metric: value}`` at k = 10 for ``subset`` from an ``eval score`` report (the product's EvalReport)."""
    from rcp_ndcg.eval.evaluate import EvalReport

    try:
        report = EvalReport.model_validate_json(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HarnessError(f"{report_path}: not an eval score report: {error}") from error
    values = {
        row.metric: float(row.value)
        for row in report.per_dataset
        if row.k == 10 and row.value is not None and row.dataset in (subset, "")
    }
    if not values:
        raise HarnessError(
            f"{report_path}: no k=10 value for {subset!r} (datasets: {sorted({r.dataset for r in report.per_dataset})})"
        )
    return values


_MTEB_KEYS = {"qrel_ndcg": "ndcg_at_10", "rcp_ndcg": "ndcg_float_at_10"}
"""mteb's names of the two metrics: ``ndcg_at_10`` over the integer qrels and ``ndcg_float_at_10`` over the
released continuous gains (``rcp_ndcg.eval.mteb``; the retrieval view reports only the integer-qrels metric)."""


def reference_scores(result: dict[str, Any]) -> dict[str, float]:
    """``{metric: value}`` from one mteb task result (``TaskResult.to_dict()``): the test split's first score row."""
    scores = result.get("scores") or {}
    rows = scores.get("test") or next(iter(scores.values()), [])
    if not rows:
        raise HarnessError(f"the mteb result for {result.get('task_name')!r} holds no score row")
    row = rows[0]
    return {metric: float(row[key]) for metric, key in _MTEB_KEYS.items() if row.get(key) is not None}


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
    """The T3 comparison table of one metric: served vs reference vs the paper vs published numbers.

    Inputs: per-subset values (0-to-1) of the served run and the mteb reference, optionally the paper's stored
    per-subset numbers (paper models), a published column (model cards, the leaderboard) with its source, and
    pre-known deviation notes.  Output: ``{"rows", "passed", "n_subsets", "tolerance"}``.  Gates: served vs
    reference within ``tolerance`` per subset, and -- where the paper's number exists -- served vs paper within
    ``tolerance``; a subset missing either side fails (never a vacuous pass).  The published column gates
    nothing, but a published mismatch over the tolerance without a note gets an "unexplained deviation" note.
    """
    notes = deviation_note or {}
    rows: list[dict[str, Any]] = []
    for subset in sorted(set(served) | set(reference)):
        served_value, reference_value = served.get(subset), reference.get(subset)
        delta = (
            abs(served_value - reference_value) if served_value is not None and reference_value is not None else None
        )
        paper_value = (paper or {}).get(subset)
        paper_delta = abs(served_value - paper_value) if served_value is not None and paper_value is not None else None
        published_value = (published or {}).get(subset)
        note = notes.get(subset, "")
        if published_value is not None and served_value is not None and not note:
            published_delta = abs(served_value - published_value)
            if published_delta > tolerance:
                note = (
                    f"unexplained deviation from the published number ({published_source or 'model card'}): "
                    f"{published_delta * 100:.2f} points"
                )
        within_reference = bool(delta is not None and delta <= tolerance)
        within_paper = paper_value is None or bool(paper_delta is not None and paper_delta <= tolerance)
        rows.append(
            {
                "subset": subset,
                "served_ndcg10": served_value,
                "reference_ndcg10": reference_value,
                "delta_vs_reference": delta,
                "paper_ndcg10": paper_value,
                "delta_vs_paper": paper_delta,
                "published_ndcg10": published_value,
                "published_source": published_source,
                "deviation_note": note,
                "within_tolerance": within_reference and within_paper,
            }
        )
    passed = bool(rows) and all(row["within_tolerance"] for row in rows)
    return {"rows": rows, "passed": passed, "n_subsets": len(rows), "tolerance": tolerance}


def quality_md(model: str, comparisons: dict[str, dict[str, Any]]) -> str:
    """``QUALITY.md``: per metric, one table row per subset with its deltas, deviation note and the verdict."""
    lines = [f"# Quality report: {model}", ""]
    for metric, comparison in comparisons.items():
        lines += [
            f"## {metric}@10",
            "",
            f"- tolerance: served vs reference (and vs the paper, where stored) within "
            f"{comparison['tolerance'] * 100:.1f} points per subset",
            "",
            "| subset | served | reference | delta | paper | delta | published | note |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in comparison["rows"]:
            lines.append(
                f"| {row['subset']} | {_fmt(row['served_ndcg10'])} | {_fmt(row['reference_ndcg10'])} | "
                f"{_fmt(row['delta_vs_reference'])} | {_fmt(row['paper_ndcg10'])} | {_fmt(row['delta_vs_paper'])} | "
                f"{_fmt(row['published_ndcg10'])} | {str(row['deviation_note']).replace('|', chr(92) + '|')} |"
            )
        lines += ["", f"Verdict: **{'PASS' if comparison['passed'] else 'FAIL'}**", ""]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _run(runner: Runner, argv: Sequence[str], log: Path) -> None:
    completed = runner(list(argv), capture_output=True, text=True, check=False)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(argv)}\n{completed.stdout}{completed.stderr}\n")
    if completed.returncode != 0:
        raise HarnessError(f"{' '.join(argv[:4])} ... exited {completed.returncode} (see {log})")


def run_quality(
    recipe: Recipe,
    *,
    engine_url: str,
    tasks: list[QualityTask],
    work_dir: str | Path,
    reference_python: str,
    paper: dict[str, dict[str, float]] | None = None,
    published: dict[str, dict[str, float]] | None = None,
    published_source: str = "",
    golden_manifest: dict[str, Any] | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """The T3 stage for one recipe: every task served and referenced, the comparison per metric, ``QUALITY.md``.

    Inputs: the recipe, the engine's URL, its tasks (:func:`tasks_for`), the working directory, the reference
    environment's Python, optionally the paper's stored numbers and published numbers per metric
    (``{metric: {subset: value}}``), the provenance blocks of the golden-replay corpus, and the subprocess
    runner.  A golden task's served commands talk to the engine through a :class:`RecordingProxy`, whose
    exchanges are written as an observation corpus under ``<work>/golden/<suite>/<subset>/``.  A task whose
    commands fail is recorded with its error and fails the stage.  Output: ``quality.json``'s document
    (``passed``, the comparisons, the task errors), also written beside ``QUALITY.md``.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    served: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    reference: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    errors: dict[str, str] = {}
    golden: list[str] = []
    for task in tasks:
        task_dir = work / task.key
        task_dir.mkdir(parents=True, exist_ok=True)
        log = task_dir / "commands.log"
        try:
            if task.golden:
                with RecordingProxy(engine_url) as proxy:
                    for argv in served_commands(task, recipe, proxy.url, task_dir):
                        _run(runner, argv, log)
                if not proxy.exchanges:
                    raise HarnessError(f"{task.key}: the golden-replay run sent no request through the proxy")
                golden_dir = work / "golden" / task.key
                proxy.write(golden_dir, recipe=recipe, task=task, manifest=golden_manifest or {})
                golden.append(str(golden_dir))
            else:
                for argv in served_commands(task, recipe, engine_url, task_dir):
                    _run(runner, argv, log)
            for metric, value in served_scores(task_dir / "scores.json", task.subset).items():
                served.setdefault(metric, {})[task.subset] = value
            _run(runner, reference_argv(task, recipe, task_dir, reference_python), log)
            result = json.loads((task_dir / "reference.json").read_text(encoding="utf-8"))
            for metric, value in reference_scores(result).items():
                reference.setdefault(metric, {})[task.subset] = value
        except (HarnessError, OSError, ValueError) as error:
            errors[task.key] = str(error)
    comparisons: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        if not reference.get(metric) and not served.get(metric):
            continue
        if metric == "rcp_ndcg" and not reference.get(metric) and all(task.view == "retrieval" for task in tasks):
            # mteb's retrieval view reports the integer-qrels metric only (gains exist for the pooled documents):
            # the served RCP-nDCG@10 is listed, not gated -- said so in the table, never a silent pass.
            listed = comparison_rows(served.get(metric, {}), {})
            comparisons[metric] = {
                **listed,
                "passed": True,
                "gated": False,
                "reason": "mteb's retrieval view reports no RCP-nDCG (rcp_ndcg.eval.mteb); served values listed",
            }
            continue
        comparisons[metric] = comparison_rows(
            served.get(metric, {}),
            reference.get(metric, {}),
            paper=(paper or {}).get(metric),
            published=(published or {}).get(metric),
            published_source=published_source,
        )
    document = {
        "recipe": recipe.id,
        "tasks": [task.key for task in tasks],
        "errors": errors,
        "comparisons": comparisons,
        "golden_replay": golden,
        "passed": not errors and bool(comparisons) and all(c["passed"] for c in comparisons.values()),
    }
    (work / "quality.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (work / "QUALITY.md").write_text(quality_md(recipe.id, comparisons), encoding="utf-8")
    return document


# ---------------------------------------------------------------------------
# the golden-replay capture: a recording reverse proxy in front of the engine
# ---------------------------------------------------------------------------


class RecordingProxy:
    """A reverse proxy on ``127.0.0.1`` that forwards every request to the engine and records the exchange.

    The served path's commands talk to :attr:`url` instead of the engine, so every exchange of a whole served
    run -- index, search or rerank -- is captured raw (request and reply bytes, the headers that matter) and
    :meth:`write` stores it as an observation corpus: the golden replay's input, from which the verified fake
    engines reproduce the run's nDCG@10 and RCP-nDCG@10.  Use it as a context manager.
    """

    def __init__(self, upstream: str, *, timeout_s: float = 600.0) -> None:
        self.upstream = upstream.rstrip("/")
        self.timeout_s = timeout_s
        self.exchanges: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
                return

            def _forward(self) -> None:
                proxy._forward(self)

            do_GET = do_POST = _forward  # noqa: N815 - stdlib names

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> RecordingProxy:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        import base64

        import httpx

        length = int(handler.headers.get("content-length") or 0)
        body = handler.rfile.read(length) if length else b""
        headers = {"content-type": handler.headers["content-type"]} if handler.headers.get("content-type") else {}
        response = httpx.request(
            handler.command,
            f"{self.upstream}{handler.path}",
            content=body or None,
            headers=headers,
            timeout=self.timeout_s,
        )
        try:
            parsed_request: Any = json.loads(body) if body else None
        except ValueError:
            parsed_request = None
        try:
            parsed_response: Any = response.json()
        except ValueError:
            parsed_response = None
        with self._lock:
            self.exchanges.append(
                {
                    "url": f"http://engine{handler.path}",
                    "method": handler.command,
                    "request_bytes": base64.b64encode(body).decode("ascii"),
                    "request_body": parsed_request,
                    "status": response.status_code,
                    "headers": {key: response.headers.get(key, "") for key in ("content-type", "server", "metadata")},
                    "response_bytes": base64.b64encode(response.content).decode("ascii"),
                    "response_json": parsed_response,
                    "latency_s": response.elapsed.total_seconds(),
                }
            )
        handler.send_response(response.status_code)
        for key in ("content-type", "metadata"):
            if key in response.headers:
                handler.send_header(key, response.headers[key])
        handler.send_header("content-length", str(len(response.content)))
        handler.end_headers()
        handler.wfile.write(response.content)

    def write(self, out_dir: str | Path, *, recipe: Recipe, task: QualityTask, manifest: dict[str, Any]) -> Path:
        """Write the captured exchanges as an observation corpus (one served run, in sending order).

        Inputs: the corpus directory, the recipe and task it served, and the provenance blocks (the wave's
        engine and collector facts; the model and recipe blocks are filled from the recipe).  Output: the
        corpus directory.
        """
        from .observe.corpus import build_record, summarise_nondeterminism, write_corpus
        from .observe.provenance import model_facts, recipe_facts
        from .record import entry_from_exchange

        records = []
        for sequence, exchange in enumerate(self.exchanges):
            request, response = entry_from_exchange(exchange)
            records.append(
                build_record(
                    sequence=sequence,
                    repetition="same_process_1",
                    batch_context={"size": 1, "request_ids": [f"golden:{sequence}"], "positions": [0]},
                    server_run_id=str(manifest.get("server_run_id", "golden")),
                    request=request,
                    response=response,
                    inputs={
                        "request_id": f"golden:{sequence}",
                        "stratum": f"golden:{task.key}",
                        "probe": "ok",
                        "layer": "model",
                        "source": {"suite": task.suite, "subset": task.subset, "commit": task.revision},
                    },
                )
            )
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "nondeterminism.json").write_text(
            json.dumps(summarise_nondeterminism(records, dim=recipe.client.get("dim")), indent=2) + "\n",
            encoding="utf-8",
        )
        blocks = {
            "model": model_facts(recipe),
            "recipe": recipe_facts(recipe),
            **{key: value for key, value in manifest.items() if key in ("engine", "collector")},
            "golden_replay": {"suite": task.suite, "subset": task.subset, "revision": task.revision, "view": task.view},
            "plan": {"request_ids": [f"golden:{index}" for index in range(len(records))]},
        }
        write_corpus(directory, blocks, records)
        return directory


# ---------------------------------------------------------------------------
# the subprocess entries: the pools file (client environment) and the mteb reference (reference environment)
# ---------------------------------------------------------------------------


def _write_pools(suite: str, subset: str, revision: str, out: Path) -> int:
    """The suite subset's released pools as a rankings file (``{"query_id", "doc_ids", "dataset", "system"}``
    rows, pool order), read through the product's dataset loader.  Returns the number of queries written."""
    from rcp_ndcg.data.dataset import load_dataset

    dataset = load_dataset(f"suite:{suite}", subset=subset, revision=revision)
    pools = dataset.candidates or {}
    if not pools:
        raise HarnessError(f"suite:{suite}/{subset}@{revision} carries no released pools (top_ranked)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps({"query_id": query_id, "doc_ids": list(doc_ids), "dataset": subset, "system": "pool"}) + "\n"
            for query_id, doc_ids in sorted(pools.items())
        ),
        encoding="utf-8",
    )
    return len(pools)


def _reference(args: argparse.Namespace) -> None:
    """The mteb reference run (reference environment): the HF model on the product's task definition."""
    import mteb  # type: ignore[import-not-found]

    from rcp_ndcg.eval.mteb import get_tasks

    tasks = get_tasks(args.suite, [args.subset], mode=args.view, revision=args.data_revision)
    model = mteb.get_model(args.model, revision=args.revision)
    results = mteb.evaluate(model, tasks)
    task_results = list(getattr(results, "task_results", results))
    if len(task_results) != 1:
        raise HarnessError(f"mteb returned {len(task_results)} task results for one task")
    Path(args.out).write_text(json.dumps(task_results[0].to_dict(), indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    """The ``pools`` and ``reference`` subcommands' parser."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.quality")
    commands = parser.add_subparsers(dest="command", required=True)
    pools = commands.add_parser("pools", help="write a suite subset's released pools as a rankings file")
    reference = commands.add_parser("reference", help="the mteb reference run of one task")
    for sub in (pools, reference):
        sub.add_argument("--suite", required=True)
        sub.add_argument("--subset", required=True)
        sub.add_argument("--out", required=True)
    pools.add_argument("--revision", required=True)
    reference.add_argument("--model", required=True)
    reference.add_argument("--revision", required=True)
    reference.add_argument("--view", required=True, choices=["retrieval", "reranking"])
    reference.add_argument("--data-revision", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m rcp_ndcg_test.quality pools|reference ...``: the two subprocess steps of a T3 task."""
    args = _parser().parse_args(argv)
    if args.command == "pools":
        _write_pools(args.suite, args.subset, args.revision, Path(args.out))
    else:
        _reference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
