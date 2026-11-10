#!/usr/bin/env python3
"""W13(c)'s gate: the late-interaction MaxSim scoring path, as its own wave.

The release's equivalence waves for ``pplx-embed-v2-late`` run stages 1-2 only (``run_wave.py``'s
equivalence step passes ``stages=[1, 2]``), whose gate for a ``multi_vector`` recipe is the per-token
cosine floor -- stage 3 (the stage that scores rankings with ``rcp eval score``) never runs, and the
opt-in ``--quality`` step (the one that exercises the product's ``kind: late_interaction`` retriever) is
never passed.  So the release had no evidence for the product's late-interaction **scoring path** --
``rcp.retrieval.maxsim``, the client-side scorer the retrieval view and ``eval score`` both consume.
This probe is that gate, as its own job (the probe pattern: ``sdpa_probe.sh``/``topk_probe.sh``).

Per variant, on the node (``maxsim_probe.sh`` builds the client mechanism, the plugin wheel and the
family's reference venv):

1. serve the recipe's engine once (:func:`rcp_ndcg_vllm.recipe.serve_argv`), encode the pairs file's
   TEXT rows through the product's role client (the served side, ragged per-token matrices kept, exactly
   the wire stage 2 sends), and stop the engine;
2. run the recipe's reference subprocess (``--mode embed``) in the family's reference venv on its own
   GPU (the recipe declares ``reference.device: cuda``), for the reference's per-token vectors;
3. score MaxSim **with the product's own scorer** (:func:`rcp.retrieval.maxsim.maxsim_topk`, the
   call ``score_topk`` dispatches to for a ragged index -- what a ``late_interaction`` retrieval index
   runs) over the pooled text corpus (every text row's documents, each query ranked over all of them:
   the retrieval view's own shape), and compute the **expected MaxSim rows** directly in float64 from
   the reference's vectors -- the card's definition ``s(Q, D) = sum_i max_j q_i . d_j``, implemented
   here independently of the blocked scorer so a scorer defect cannot cancel against itself;
4. gate and record: the per-pair ``|served - expected|`` against the bound the recipe's declared
   per-token floor implies, the median per-query Kendall tau against the published rank gate, and the
   expected MaxSim rows themselves as data in the report (``maxsim-rows.json`` alongside).

The bound is derived, never assumed: for unit vectors
``|q.d - q'.d'| <= |q.d - q'.d| + |q'.d - q'.d'| <= ||q - q'|| + ||d - d'||``, and both the query and
the document tokens move at most ``sqrt(2 * (1 - vec_min_cosine))`` (the stage-2 vector gate's
per-token floor, :data:`rcp_ndcg_test.equivalence.gates.ResolvedGates.vec_min_cosine`, the family's
measured floor), so a pair's MaxSim delta -- summed over the query's tokens -- is at most
``2 * n_query_tokens * sqrt(2 * (1 - vec_min_cosine))``.  A delta beyond it is the scoring path's,
not the vectors' (stage 2 already gates the vectors in the same campaign).  The rank gate is the
published median-Kendall-tau default (:func:`rcp_ndcg_test.equivalence.gates.resolve_gates`).

The pairs file's media rows are excluded with the reason recorded (the media stage owns them; the
scoring route's ``late_interaction_params`` is exercised by every rerank family's equivalence), and a
text the client changed (its processing records -- the declared over-cap deviation) is reported
separately, non-gating: the reference renders over-cap inputs its own way by declaration.

    "$STATE/client" python maxsim_probe.py --out <dir> --recipe pplx-embed-v2-late-0.6b \\
        --pairs <stage>/pairs/<recipe>.jsonl --reference-python <state>/reference/pplx-embed-v2-late/bin/python \\
        [--rows 3,4] [--port-base 8100] [--gpus 0] [--reference-gpus 1]

    # the wrapper's plan, running nothing:
    MAXSIM_DRY=1 maxsim_probe.sh <RC_STAGE_URI> <OUT_URI>

The report (``maxsim-report.json``, ``maxsim-report.md``, ``maxsim-rows.json``) carries the gate with its
sources, the engine's and the reference's argv/state, the corpus shape, the per-query rows (served,
expected, delta, tau), the gate rows, the verdict, and the expected MaxSim rows.
"""

from __future__ import annotations

__all__ = [
    "BOUND_DERIVATION",
    "GATE_SOURCE",
    "SCHEMA",
    "build_parser",
    "build_report",
    "corpus_doc",
    "expected_matrix",
    "main",
    "maxsim_gate",
    "plan",
    "reference_query_matrix",
    "score_matrix",
    "text_rows",
    "verdict",
    "write_report",
]

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_vllm.recipe import Recipe, load_recipe, serve_argv

from rcp_ndcg_test.errors import HarnessError

SCHEMA = "rcp-maxsim-gate/1"
"""The report's format version (the artifact the operator adjudicates on)."""

BOUND_DERIVATION = (
    "per-pair bound = 2 * n_query_tokens * sqrt(2 * (1 - vec_min_cosine)): for unit vectors "
    "|q.d - q'.d'| <= |q.d - q'.d| + |q'.d - q'.d'| <= ||q - q'|| + ||d - d'||, and both the query and the "
    "document tokens move at most sqrt(2 * (1 - vec_min_cosine)) (the stage-2 vector gate's per-token "
    "floor), so the pair's MaxSim delta, summed over the query's tokens, is at most twice the "
    "single-sided bound; a delta beyond it is the scoring path's, not the vectors'"
)
"""The bound's derivation, carried in every report: no number here is typed without it."""

GATE_SOURCE = (
    "PLAN W13(c) (the late-interaction MaxSim gate as its own wave; the reviews' D4): the bound derives "
    "from the recipe's declared gates.vec_min_cosine (the family.yaml overrides' measured floor), and the "
    "rank gate is rcp_ndcg_test/equivalence/gates.py's published tau_min default"
)
"""The gate's provenance, exported so the tests and the report name one string."""


def build_parser() -> argparse.ArgumentParser:
    """The probe's parser: the wrapper builds its argv by hand, so the flags must match here."""
    parser = argparse.ArgumentParser(
        description="W13(c)'s gate: the product's MaxSim scoring path, served vectors against the reference's"
    )
    parser.add_argument("--out", required=True, help="where the report, the serve log and the reference output land")
    parser.add_argument("--recipe", required=True, help="the variant id to probe")
    parser.add_argument("--pairs", required=True, help="the recipe's pairs file (its text rows are probed)")
    parser.add_argument("--rows", default="", help="comma-separated row indices (default: every text row)")
    parser.add_argument("--port-base", type=int, default=8100, help="the port the engine serves on")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES for the engine")
    parser.add_argument("--reference-gpus", default="1", help="CUDA_VISIBLE_DEVICES for the reference subprocess")
    parser.add_argument("--reference-python", required=True, help="the family reference venv's python")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0, help="the engine's load budget")
    return parser


def text_rows(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """The pairs rows this gate scores: every row without a ``media`` key, with its original index.

    The media rows are the media stage's (stage 2's own split); the report records their exclusion, it
    never counts them as scored or skipped silently."""
    return [(index, row) for index, row in enumerate(rows) if not row.get("media")]


def corpus_doc(all_rows: list[dict[str, Any]], rows: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """The corpus facts the report records: the pairs file's rows, the rows this gate scores (after any
    ``--rows`` selection), and the media rows excluded -- counted from the whole pairs file, before the
    filter, so the operator's row selection never shows up as excluded media (the media rows are the
    media stage's own split).  Inputs: the pairs file's rows and the selected text rows.  Output: the
    ``corpus`` document.  Units: row counts."""
    return {
        "pairs_rows": len(all_rows),
        "text_rows": len(rows),
        "media_rows_excluded": len(all_rows) - len(text_rows(all_rows)),
        "note": "the media rows are the media stage's; this gate scores the text corpus",
    }


def plan(recipe: Recipe, *, port_base: int = 8100, gpus: str = "0", reference_gpus: str = "1") -> dict[str, Any]:
    """The run's plan: the engine's argv and env, the reference subprocess's argv and env.

    Inputs: the loaded recipe, the engine's port and the two GPU sets.  Outputs: ``{"serve": {argv, env,
    port}, "reference": {argv, env}}`` -- the argv from :func:`rcp_ndcg_vllm.recipe.serve_argv` (the one
    renderer the wave runner and the serve console use too) and the reference's ``--mode embed`` shape.
    The reference's interpreter and paths are the node's own (placeholders here); the run records the
    executed argv in its step document.  Units: none.
    """
    device = recipe.reference.device if recipe.reference is not None else "cpu"
    return {
        "recipe": recipe.id,
        "serve": {
            "argv": serve_argv(recipe, port=port_base, served_model_name=recipe.id),
            "env": {"CUDA_VISIBLE_DEVICES": gpus, "VLLM_PORT": str(port_base + 1000), "TMPDIR": "<out>/tmp-serve"},
            "port": port_base,
        },
        "reference": {
            "argv": [
                "<reference-python>",
                "<family>/reference.py",
                "--mode",
                "embed",
                "--pairs",
                "<out>/maxsim-pairs.jsonl",
                "--out",
                "<out>/reference-embed.json",
                "--tokenizer",
                "<recipe tokenizer spec>",
                "--recipe",
                "<resolved recipe json>",
                "--device",
                device,
            ],
            "env": {"CUDA_VISIBLE_DEVICES": reference_gpus},
            "device": device,
        },
    }


def score_matrix(docs: Any, queries: Any) -> np.ndarray:
    """The product's own MaxSim scores, one row per query over the whole corpus in document order.

    Inputs: the corpus and the queries as :class:`rcp.inference.types.Embeddings` (the documents
    ragged: this is the late-interaction layout a ``late_interaction`` index stores).  The scorer is
    :func:`rcp.retrieval.maxsim.maxsim_topk` with ``k`` = every document -- exactly the call
    :func:`rcp.retrieval.topk.score_topk` dispatches to for a ragged index, with the product's tie
    rule.  Output: the ``(n_queries, n_documents)`` score matrix in document order.  Units: MaxSim scores
    (sums of per-token cosines of unit vectors, in ``[-n_query_tokens, n_query_tokens]``).
    """
    from rcp.retrieval.maxsim import maxsim_topk

    k = docs.num_items
    scores, indices = maxsim_topk(docs, queries, k=k)
    matrix = np.full((queries.num_items, k), np.nan, dtype=np.float64)
    rows = np.repeat(np.arange(queries.num_items), k)
    matrix[rows, np.asarray(indices).reshape(-1)] = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.isnan(matrix).any():
        raise HarnessError("the product's MaxSim scorer returned an incomplete score matrix")
    return matrix


def expected_matrix(doc_mats: list[np.ndarray], query_mats: list[np.ndarray]) -> np.ndarray:
    """The expected MaxSim rows: the card's definition, direct in float64, on the reference's matrices.

    ``s(Q, D) = sum_i max_j q_i . d_j`` (the product scorer's own docstring is the definition's home) --
    implemented here independently of the blocked scorer, so a scorer defect cannot cancel against
    itself.  Inputs: one token-vector matrix per document and per query.  Output: the
    ``(n_queries, n_documents)`` matrix in the same order.  Units: MaxSim scores as above.
    """
    out = np.zeros((len(query_mats), len(doc_mats)), dtype=np.float64)
    for i, query in enumerate(query_mats):
        q = np.asarray(query, dtype=np.float64)
        for j, document in enumerate(doc_mats):
            d = np.asarray(document, dtype=np.float64)
            if q.shape[0] and d.shape[0]:
                out[i, j] = float((q @ d.T).max(axis=1).sum())
    return out


def reference_query_matrix(row: dict[str, Any], position: int) -> np.ndarray:
    """One pairs row's query matrix, as the reference's embed output carries it: unwrapped.

    The family reference writes ``query_vectors`` as a ONE-element list -- one (n_kept, width) matrix
    per query text, positionally paired, and a pairs row carries a single query -- so ``np.asarray``
    gives a ``(1, n_kept, width)`` array.  Fed straight into :func:`expected_matrix`, that shape
    broadcasts the matmul and ``max(axis=1)`` reduces over the query tokens: the transposed MaxSim,
    which reds an equivalent engine (measured delta 6.73 against the bound 2.26) and records wrong
    numbers in ``maxsim-rows.json``.  The one element is unwrapped; a bare ``(n_kept, width)`` array
    (the shape :func:`expected_matrix` takes) passes through unchanged; any other shape is a
    :class:`HarnessError` -- never a silent ``[0]``.

    Inputs: the reference's embed row and its pairs-row index (for the error).  Output: the query's
    ``(n_kept, width)`` token-vector matrix, dtype float64.  Units: token vectors as the reference
    wrote them.
    """
    query = np.asarray(row["query_vectors"], dtype=np.float64)
    if query.ndim == 3 and query.shape[0] == 1:  # the wire shape: one matrix per query text
        query = query[0]
    if query.ndim != 2:
        raise HarnessError(
            f"pairs row {position}: the reference's query_vectors is {query.shape}, not one (n_kept, width) matrix"
        )
    return query


def maxsim_gate(
    recipe: Recipe,
    served: np.ndarray,
    expected: np.ndarray,
    *,
    query_tokens: list[int],
    gating: list[list[bool]],
) -> dict[str, Any]:
    """The gate over the MaxSim rows: the per-pair delta against the derived bound, the rank gate.

    Inputs: the recipe (its declared gates supply both bounds), the served and expected score matrices
    (``(n_queries, n_documents)``, document order), each query's served token count (the bound's
    per-query scale), and the gating mask (a pair gates only when the client changed neither of its
    texts: under the recipe's declared over-cap deviation the reference renders changed inputs its own
    way, so they are reported separately, never gated).  Output: ``{rows, gates, passed, ...}`` with one
    row per query (its worst delta, its tau over the gating documents) and the two gate rows.  A gate
    that read no gating pair fails loudly (the anchor audit's rule: an audit that checked nothing proves
    nothing).  Units: MaxSim scores as above; the bound in the same units.
    """
    from rcp_ndcg_test.equivalence.gates import kendall_tau_b, resolve_gates

    gates = resolve_gates(recipe)
    floor = gates.vec_min_cosine
    if served.shape != expected.shape:
        raise HarnessError(f"the MaxSim rows are not comparable: served {served.shape}, expected {expected.shape}")
    failures: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    n_gating = n_over_cap = 0
    for i in range(served.shape[0]):
        indices = [j for j in range(served.shape[1]) if gating[i][j]]
        n_over_cap += served.shape[1] - len(indices)
        deltas = np.abs(served[i, indices] - expected[i, indices]) if indices else np.zeros(0)
        bound = 2.0 * query_tokens[i] * float(np.sqrt(2.0 * (1.0 - floor)))
        worst = float(deltas.max()) if deltas.size else None
        tau = kendall_tau_b(served[i, indices].tolist(), expected[i, indices].tolist()) if len(indices) >= 2 else None
        per_query.append(
            {
                "query": i,
                "n_gating_documents": len(indices),
                "max_abs_delta": worst,
                "bound": bound,
                "kendall_tau": tau,
                "n_query_tokens": query_tokens[i],
            }
        )
        n_gating += len(indices)
        for j, delta in zip(indices, deltas, strict=True):
            if delta > bound:
                failures.append(
                    {
                        "query": i,
                        "document": j,
                        "abs_delta": float(delta),
                        "bound": bound,
                        "n_query_tokens": query_tokens[i],
                    }
                )
    if n_gating == 0:
        failures.append(
            {
                "check": "nothing_checked",
                "note": "no gating pair: every pair's query or document was changed by the client "
                "(the declared over-cap deviation) or the corpus was empty",
            }
        )
    taus = [row["kendall_tau"] for row in per_query if row["kendall_tau"] is not None]
    median_tau = float(np.median(taus)) if taus else None
    gate_rows = [
        {
            "gate": "maxsim_max_abs",
            "passed": bool(n_gating > 0 and not failures),
            "value": max((row["max_abs_delta"] for row in per_query if row["max_abs_delta"] is not None), default=None),
            "bound_derivation": BOUND_DERIVATION,
            "referent": "every gating pair's |served - expected| MaxSim delta <= "
            f"2 * n_query_tokens * sqrt(2 * (1 - vec_min_cosine={floor}))",
        },
        {
            "gate": "maxsim_tau_median",
            "passed": bool(median_tau is not None and median_tau >= gates.tau_min),
            "value": median_tau,
            "bound": gates.tau_min,
            "referent": "median per-query Kendall tau between served and expected MaxSim rankings "
            "(the retrieval view's own shape: each query over the pooled corpus)",
        },
    ]
    return {
        "rows": per_query,
        "n_gating_pairs": n_gating,
        "n_over_cap_pairs": n_over_cap,
        "max_abs_delta": gate_rows[0]["value"],
        "kendall_tau_median": median_tau,
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows)),
        "failures": failures,
    }


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """The gate's verdict as a pure function of the report: ``equivalent`` | ``differs`` |
    ``nothing_checked`` | ``unmeasured``.

    ``unmeasured`` when the engine or the reference did not run (nothing was scored) or no gate row was
    produced; ``differs`` when a gate row failed on a measured pair (its reason names the first failing
    pair) or the rank gate failed on measured ranks (its reason names the median tau);
    ``nothing_checked`` when the gating mask excluded every pair (every text client-changed or
    ``--rows`` selected only changed rows) -- the designed named failure: the reason carries the
    ``nothing_checked`` record's note, and ``main`` exits non-zero because the state is not
    ``equivalent``; ``equivalent`` when every gate row passed."""
    if (report.get("serve") or {}).get("state") != "served" or (report.get("reference") or {}).get("state") != "ran":
        return {
            "state": "unmeasured",
            "reason": "the engine did not serve or the reference did not run; nothing was scored",
        }
    gates = report.get("gates") or []
    if not gates:
        return {"state": "unmeasured", "reason": "no gate row was produced"}
    failed = [row["gate"] for row in gates if not row["passed"]]
    if failed:
        failures = (report.get("rows_summary") or {}).get("failures") or []
        pairs = [item for item in failures if "abs_delta" in item]
        if not pairs:
            tau = (report.get("rows_summary") or {}).get("kendall_tau_median")
            if tau is not None and "maxsim_tau_median" in failed:
                # a measured rank failure: the deltas sat inside the bound, the ranks did not --
                # nothing_checked is the no-gating-pair case's own state, and there the tau is None
                bound = next(row["bound"] for row in gates if row["gate"] == "maxsim_tau_median")
                return {
                    "state": "differs",
                    "reason": f"maxsim_tau_median failed: the median tau {tau:.4f} is below the "
                    f"published rank gate {bound}",
                }
            notes = (
                "; ".join(str(item.get("note") or item.get("check") or "unexplained failure") for item in failures)
                or "no failure record was written"
            )
            return {"state": "nothing_checked", "reason": f"{', '.join(failed)} failed: {notes}"}
        first = pairs[0]
        where = (
            f" (query {first.get('query')}, document {first.get('document')}, delta "
            f"{first.get('abs_delta'):.3g} > bound {first.get('bound'):.3g})"
        )
        return {"state": "differs", "reason": f"{', '.join(failed)} failed{where}"}
    tau = report["rows_summary"]["kendall_tau_median"]
    return {
        "state": "equivalent",
        "reason": f"every gating pair is inside the derived bound and the median tau {tau:.4f} is at or "
        "above the published rank gate",
    }


def build_report(
    recipe: Recipe,
    *,
    serve: dict[str, Any],
    reference: dict[str, Any],
    served_scores: np.ndarray,
    expected_scores: np.ndarray,
    query_tokens: list[int],
    gating: list[list[bool]],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """The report: the gate as data (both bounds and their derivation), the rows, the recorded expected
    MaxSim rows, and the verdict.

    Inputs: the recipe, the serve and reference step documents, the served and expected score matrices,
    each query's served token count, the gating mask, and the corpus facts (the pairs rows, the text rows,
    the excluded media rows).  Output: the report document (``verdict`` is :func:`verdict` applied to
    it).  Units: MaxSim scores as above.
    """
    gate = maxsim_gate(recipe, served_scores, expected_scores, query_tokens=query_tokens, gating=gating)
    from rcp_ndcg_test.equivalence.gates import resolve_gates

    resolved = resolve_gates(recipe)
    report: dict[str, Any] = {
        "probe": "maxsim-gate",
        "schema": SCHEMA,
        "recipe": {"id": recipe.id, "model": recipe.model, "revision": recipe.revision},
        "engine": {"image": recipe.engine.image, "min_version": recipe.engine.min_version},
        "gate": {
            "source": GATE_SOURCE,
            "vec_min_cosine": resolved.vec_min_cosine,
            "tau_min": resolved.tau_min,
            "bound_derivation": BOUND_DERIVATION,
        },
        "serve": serve,
        "reference": reference,
        "corpus": corpus,
        "expected_rows": {
            "n_queries": int(expected_scores.shape[0]),
            "n_documents": int(expected_scores.shape[1]),
            "note": "the expected MaxSim rows: the card's definition (direct, float64) on the reference's "
            "per-token vectors; the served rows are the product's scorer on the client's vectors",
            "scores": np.asarray(expected_scores, dtype=np.float64).tolist(),
        },
        "rows_summary": {
            "rows": gate["rows"],
            "n_gating_pairs": gate["n_gating_pairs"],
            "n_over_cap_pairs": gate["n_over_cap_pairs"],
            "max_abs_delta": gate["max_abs_delta"],
            "kendall_tau_median": gate["kendall_tau_median"],
            "failures": gate["failures"],
        },
        "gates": gate["gates"],
    }
    report["verdict"] = verdict(report)
    return report


def write_report(out: Path, report: dict[str, Any]) -> None:
    """Write ``maxsim-report.json``, ``maxsim-report.md`` and ``maxsim-rows.json`` (the expected MaxSim
    rows as their own data artifact) under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "maxsim-report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "maxsim-rows.json").write_text(json.dumps(report["expected_rows"], indent=1) + "\n", encoding="utf-8")
    gate = report["gate"]
    summary = report["rows_summary"]
    lines = [
        "# W13(c) gate: the late-interaction MaxSim scoring path",
        "",
        f"- recipe: `{report['recipe']['id']}` @ `{report['recipe']['revision']}`",
        f"- engine: `{report['engine'].get('image')}`",
        f"- corpus: {json.dumps(report['corpus'], default=str)}",
        f"- gate: {gate['bound_derivation']}",
        f"- rank gate: median tau >= {gate['tau_min']} ({gate['source']})",
        f"- measured: max |delta| `{summary['max_abs_delta']}`, median tau `{summary['kendall_tau_median']}`, "
        f"gating pairs `{summary['n_gating_pairs']}`, over-cap pairs `{summary['n_over_cap_pairs']}`",
        f"- **verdict: `{report['verdict']['state']}`** -- {report['verdict']['reason']}",
        "",
        "| query | gating docs | max abs delta | bound | kendall tau |",
        "|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['query']} | {row['n_gating_documents']} | {row['max_abs_delta']} | {row['bound']} | "
            f"{row['kendall_tau']} |"
        )
    lines += ["", "## raw", "", "```json", json.dumps(report, indent=1, default=str), "```", ""]
    (out / "maxsim-report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# the node: one engine start, the product's client, the reference subprocess
# ---------------------------------------------------------------------------


def _tail(log: Path, lines: int = 40, width: int = 300) -> list[str]:
    """The last ``lines`` of a serve log, each clipped: a dead engine's evidence."""
    if not log.is_file():
        return []
    return [line.strip()[:width] for line in log.read_text(errors="replace").splitlines()[-lines:]]


def _encode_corpus(recipe: Recipe, base_url: str, items: list[tuple[str, Any, Any]]) -> dict[str, Any]:
    """One ragged matrix per side through the product's role client (the wire stage 2 sends).

    Inputs: the recipe, the engine's URL, and ``(key, role, content)`` items (:func:`texts_of`'s layout:
    ``<row>:query`` then ``<row>:document:<position>``, one client call per text).  Output: per side
    ``{matrices, changed}`` -- the client's ragged per-token matrices (the transfer dtype the client
    applied stays as received) and, per key, whether the client changed the text (its processing
    records; a changed text's pairs are the declared over-cap deviation's, reported, not gated).
    """
    from rcp_ndcg_test.equivalence.wire import role_client

    client, _capture = role_client(recipe, base_url, full_width=True)
    sides: dict[str, Any] = {"query": {"matrices": [], "changed": {}}, "document": {"matrices": [], "changed": {}}}
    for key, role, content in items:
        side = "query" if key.endswith(":query") else "document"
        start = len(client.processing)
        embeddings = client.encode([content], role)
        if embeddings.offsets is None:
            raise HarnessError(
                f"{key}: the client returned one pooled vector; a late-interaction model returns one "
                "token-vector matrix per text, and scoring a pooled answer would be a different retrieval method"
            )
        matrix = np.asarray(embeddings.vectors)
        if matrix.ndim != 2 or int(embeddings.offsets[-1]) != matrix.shape[0]:
            raise HarnessError(f"{key}: the client returned {matrix.shape} with offsets {embeddings.offsets}")
        sides[side]["matrices"].append(matrix)
        sides[side]["changed"][key] = any(record.changed for record in client.processing[start:])
    return sides


def _unmeasured_report(
    recipe: Recipe, serve: dict[str, Any], reference: dict[str, Any], corpus: dict[str, Any]
) -> dict[str, Any]:
    """The report for a run that measured nothing: the steps' states and the loud ``unmeasured`` verdict."""
    from rcp_ndcg_test.equivalence.gates import resolve_gates

    resolved = resolve_gates(recipe)
    report: dict[str, Any] = {
        "probe": "maxsim-gate",
        "schema": SCHEMA,
        "recipe": {"id": recipe.id, "model": recipe.model, "revision": recipe.revision},
        "engine": {"image": recipe.engine.image, "min_version": recipe.engine.min_version},
        "gate": {
            "source": GATE_SOURCE,
            "vec_min_cosine": resolved.vec_min_cosine,
            "tau_min": resolved.tau_min,
            "bound_derivation": BOUND_DERIVATION,
        },
        "serve": serve,
        "reference": reference,
        "corpus": corpus,
        "expected_rows": None,
        "rows_summary": {
            "rows": [],
            "n_gating_pairs": 0,
            "n_over_cap_pairs": 0,
            "max_abs_delta": None,
            "kendall_tau_median": None,
            "failures": [],
        },  # fmt: skip
        "gates": [],
    }
    report["verdict"] = verdict(report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write the report; 0 when a verdict was reached, 1 when nothing was measured."""
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe(args.recipe)
    if recipe.role != "multi_vector":
        raise SystemExit(f"maxsim_probe: {recipe.id} is a {recipe.role} recipe; this gate scores MaxSim")
    from rcp_ndcg_test.equivalence.fitting import load_pairs, resolved_tokenizer_spec
    from rcp_ndcg_test.equivalence.reference import reference_entry, reference_of, run_reference
    from rcp_ndcg_test.jobs.sdpa_probe import _stop, _wait_ready, texts_of

    all_rows = load_pairs(args.pairs)
    rows = text_rows(all_rows)
    if args.rows:
        wanted = {int(value) for value in args.rows.split(",") if value.strip()}
        rows = [(index, row) for index, row in rows if index in wanted]
    if not rows:
        raise HarnessError(f"no text rows to score under {args.pairs}")
    corpus = corpus_doc(all_rows, rows)
    plan_doc = plan(recipe, port_base=args.port_base, gpus=args.gpus, reference_gpus=args.reference_gpus)

    # The reference subprocess: its own venv, its own GPU (the recipe declares reference.device: cuda).
    reference_spec = reference_of(recipe)
    reference_device = str(reference_spec.device or "cpu")
    reference_row: dict[str, Any] = {"state": "ran", "device": reference_device}
    pairs_path = out / "maxsim-pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for _index, row in rows), encoding="utf-8")
    try:
        document = run_reference(
            args.reference_python,
            reference_entry(recipe),
            mode="embed",
            pairs_path=pairs_path,
            out_path=out / "reference-embed.json",
            tokenizer_spec=resolved_tokenizer_spec(recipe),
            recipe=recipe,
            device=reference_device,
            cuda_visible_devices=args.reference_gpus,
        )
    except HarnessError as error:
        reference_row = {**reference_row, "state": "failed", "error": str(error)}
        document = {"rows": []}

    # The served side: the engine, the product's client over the text rows, the engine stops.
    serve_env = {k: v for k, v in plan_doc["serve"]["env"].items() if "<" not in str(v)}
    serve_env["TMPDIR"] = str(out / "tmp-serve")
    Path(serve_env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = out / "serve.log"
    serve_row: dict[str, Any] = {"state": "failed", "argv": plan_doc["serve"]["argv"], "serve_log_tail": []}
    popen = None
    items: list[tuple[str, Any, Any]] = []
    sides: dict[str, Any] = {}
    try:
        with log.open("wb") as handle:
            popen = subprocess.Popen(  # noqa: S603 - the recipe's own rendered argv
                plan_doc["serve"]["argv"], stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, **serve_env},
                start_new_session=True,
            )  # fmt: skip
        base_url = f"http://127.0.0.1:{args.port_base}"
        if _wait_ready(base_url, popen, args.startup_timeout_s):
            serve_row = {**serve_row, "state": "served", "base_url": base_url, "serve_log_tail": _tail(log)}
            items = texts_of([row for _index, row in rows])
            sides = _encode_corpus(recipe, base_url, items)
        else:
            serve_row["serve_log_tail"] = _tail(log)
    finally:
        if popen is not None:
            _stop({"popen": popen})

    if serve_row["state"] != "served" or not document.get("rows"):
        report = _unmeasured_report(recipe, serve_row, reference_row, corpus)
        write_report(out, report)
        print(f"maxsim_probe: verdict {report['verdict']['state']} ({report['verdict']['reason']})")
        return 1

    # The MaxSim gate: the product's scorer on the served vectors, the card's definition on the
    # reference's, the per-pair gating mask from the client's own change records.
    from rcp.inference.types import Embeddings

    query_keys = [key for key, _role, _content in items if key.endswith(":query")]
    document_keys = [key for key, _role, _content in items if not key.endswith(":query")]
    by_index = {int(row["index"]): row for row in document.get("rows", []) if isinstance(row.get("index"), int)}
    reference_docs: list[np.ndarray] = []
    reference_queries: list[np.ndarray] = []
    for position in range(len(rows)):
        row = by_index.get(position)
        if row is None:
            raise HarnessError(f"the reference wrote no row for pairs row {position}")
        reference_queries.append(reference_query_matrix(row, position))
        documents = row.get("document_vectors") or []
        if len(documents) != len(rows[position][1]["documents"]):
            raise HarnessError(
                f"the reference wrote {len(documents)} document matrices for pairs row {position}, "
                f"the row carries {len(rows[position][1]['documents'])}"
            )
        reference_docs.extend(np.asarray(matrix, dtype=np.float64) for matrix in documents)

    served = score_matrix(
        Embeddings(
            vectors=np.concatenate(sides["document"]["matrices"]),
            offsets=np.cumsum([0, *(m.shape[0] for m in sides["document"]["matrices"])]),
        ),
        Embeddings(
            vectors=np.concatenate(sides["query"]["matrices"]),
            offsets=np.cumsum([0, *(m.shape[0] for m in sides["query"]["matrices"])]),
        ),
    )
    expected = expected_matrix(reference_docs, reference_queries)
    if served.shape != expected.shape:
        raise HarnessError(
            f"the sides do not agree on the corpus: the client encoded {served.shape[1]} documents and "
            f"{served.shape[0]} queries, the reference wrote {expected.shape[1]} and {expected.shape[0]}"
        )
    query_tokens = [int(m.shape[0]) for m in sides["query"]["matrices"]]
    gating = [
        [
            not (sides["query"]["changed"][query_keys[i]] or sides["document"]["changed"][document_keys[j]])
            for j in range(len(document_keys))
        ]
        for i in range(len(query_keys))
    ]
    report = build_report(
        recipe,
        serve=serve_row,
        reference={**reference_row, "rows": len(document.get("rows", []))},
        served_scores=served,
        expected_scores=expected,
        query_tokens=query_tokens,
        gating=gating,
        corpus=corpus,
    )
    write_report(out, report)
    print(f"maxsim_probe: verdict {report['verdict']['state']} ({report['verdict']['reason']})")
    return 0 if report["verdict"]["state"] == "equivalent" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main() on the node
    sys.exit(main())
