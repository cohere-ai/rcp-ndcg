"""The stage functions: stage 1 (prompts, CPU, zero tolerance) and stage 2 (scores or vectors against the gates).

Both take the loaded recipe, the pairs to check and the loaded reference, and return plain dictionaries for the
report (every number with its referent).  Stage 2 talks to the engine through
:class:`~rcp_ndcg_vllm.equivalence.client.EngineClient`; the reference runs in this process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import HarnessError
from ..recipe import Recipe
from .client import EngineClient, fold_instruction
from .gates import ResolvedGates, kendall_tau_b, resolve_gates
from .prompt import TokenizerAdapter, served_prompt_text
from .reference import Reference

__all__ = ["load_pairs", "stage1_prompts", "stage2_scores"]

_SNIPPET = 80


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """The pairs file: JSONL, one object per query, ``{"query": str, "documents": [str, ...]}``.

    The sampled pairs are the wave's stage-1 and stage-2 inputs; a wave's ``--pairs-dir`` supplies one file per
    recipe (``<id>.jsonl``), falling back to ``default.jsonl``.
    """
    pairs: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"{path}:{number} is not a JSON object: {error}") from error
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("query"), str)
                or not isinstance(row.get("documents"), list)
            ):
                raise HarnessError(f'{path}:{number}: expected {{"query": str, "documents": [str, ...]}}')
            if not all(isinstance(document, str) for document in row["documents"]):
                raise HarnessError(f"{path}:{number}: documents must be strings")
            pairs.append(row)
    if not pairs:
        raise HarnessError(f"{path} holds no pairs")
    return pairs


def stage1_prompts(
    recipe: Recipe,
    pairs: list[dict[str, Any]],
    reference: Reference,
    tokenizer: TokenizerAdapter,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Stage 1: compare served and reference token ids for every (query, document) pair; zero tolerance.

    Output: a report dict with ``pairs``, ``checked``, ``passed`` and, on the first mismatch, ``mismatch`` carrying
    both token-id lists and their token strings.  ``limit`` truncates the run (a quick CPU check).
    """
    checked = 0
    rows = pairs if limit is None else pairs[:limit]
    for row_index, row in enumerate(rows):
        for document_index, document in enumerate(row["documents"]):
            served_ids = tokenizer.encode(served_prompt_text(recipe, row["query"], document))
            reference_ids = reference.render(row["query"], document, recipe.client.default_instruction)
            if served_ids != reference_ids:
                return {
                    "pairs": len(pairs),
                    "checked": checked,
                    "passed": False,
                    "mismatch": {
                        "query_index": row_index,
                        "document_index": document_index,
                        "query": row["query"],
                        "document": document[:_SNIPPET],
                        "served_ids": served_ids,
                        "reference_ids": reference_ids,
                        "served_tokens": [tokenizer.id_to_token(i) for i in served_ids],
                        "reference_tokens": [tokenizer.id_to_token(i) for i in reference_ids],
                    },
                }
            checked += 1
    return {"pairs": len(pairs), "checked": checked, "passed": True, "mismatch": None}


def stage2_scores(
    recipe: Recipe,
    base_url: str,
    pairs: list[dict[str, Any]],
    reference: Reference,
    *,
    served_model_name: str | None = None,
    timeout_s: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Stage 2: served scores or vectors against the in-process reference, under the recipe's gates.

    A rerank recipe sends one ``/rerank`` request per sampled query (the whole candidate set, ``top_n`` equal to
    the number of documents); an ``embed`` recipe compares dense vectors from ``/v1/embeddings``; a
    ``multi_vector`` recipe compares per-token vectors from ``/pooling`` after the same float16 cast.  The report
    carries every number with its referent and one row per gate; ``passed`` is true only when every gate holds.
    The reference's model is loaded first (``device``, ``cpu`` by default: the wave's engines hold the GPUs).
    """
    if recipe.reference.kind == "stored_scores":
        raise HarnessError(
            "stage 2 needs a runnable reference; reference.kind=stored_scores supports stage 1 and stage 3 only"
        )
    reference.load(device)
    gates = resolve_gates(recipe)
    with EngineClient(recipe, base_url, served_model_name=served_model_name, timeout_s=timeout_s or 300.0) as client:
        if recipe.role == "rerank":
            return _rerank_stage2(recipe, pairs, reference, client, gates)
        return _vector_stage2(recipe, pairs, reference, client, gates)


def _instruction_field(recipe: Recipe) -> str | None:
    """The ``instruction`` request field (only in ``field`` mode; a fold puts it in the query text)."""
    assert recipe.client.instruction is not None
    return recipe.client.default_instruction if recipe.client.instruction == "field" else None


def _served_query(recipe: Recipe, query: str) -> str:
    """The query text the client sends to the engine (folded when ``client.instruction`` is ``fold``)."""
    assert recipe.client.instruction is not None
    if recipe.client.instruction == "fold":
        return fold_instruction(recipe.client.default_instruction, query)
    return query


def _score_bound(gates: ResolvedGates, scale: str, reference_score: float) -> float:
    """The per-document |delta| bound on one score scale, at the reference's score."""
    if scale == "probability":
        return gates.prob_max_abs
    if scale == "logit":
        return gates.logit_rel_abs * (1.0 + abs(reference_score))
    return gates.cos_max_abs


def _rerank_stage2(
    recipe: Recipe,
    pairs: list[dict[str, Any]],
    reference: Reference,
    client: EngineClient,
    gates: ResolvedGates,
) -> dict[str, Any]:
    """Scores from the served /rerank against the reference, with the scale's gates and the per-query tau."""
    scale = recipe.reference.score_scale
    per_document: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for row_index, row in enumerate(pairs):
        served = client.rerank(
            _served_query(recipe, row["query"]),
            row["documents"],
            instruction=_instruction_field(recipe),
            use_activation=recipe.client.use_activation,
        )
        values = reference.score(row["query"], row["documents"], recipe.client.default_instruction)
        for document_index, (served_score, reference_score) in enumerate(zip(served, values, strict=True)):
            delta = abs(served_score - reference_score)
            bound = _score_bound(gates, scale, reference_score)
            per_document.append(
                {
                    "query_index": row_index,
                    "document_index": document_index,
                    "query": row["query"],
                    "document": row["documents"][document_index][:_SNIPPET],
                    "served": served_score,
                    "reference": reference_score,
                    "abs_delta": delta,
                    "bound": bound,
                    "within": bool(delta <= bound),
                }
            )
        tau = kendall_tau_b(served, values)
        per_query.append(
            {
                "query_index": row_index,
                "query": row["query"],
                "documents": len(served),
                "kendall_tau": tau,
                "within": bool(tau is not None and tau >= gates.tau_min),
            }
        )
    return _rerank_summary(per_document, per_query, gates, scale)


def _rerank_summary(
    per_document: list[dict[str, Any]], per_query: list[dict[str, Any]], gates: ResolvedGates, scale: str
) -> dict[str, Any]:
    """Aggregate the per-document deltas and per-query taus into the gate rows."""
    deltas = [entry["abs_delta"] for entry in per_document]
    p99 = float(np.percentile(deltas, 99)) if deltas else 0.0
    worst = max(deltas) if deltas else 0.0
    taus = [entry["kendall_tau"] for entry in per_query if entry["kendall_tau"] is not None]
    median_tau = float(np.median(taus)) if taus else None
    within_p99 = sum(1 for entry in per_document if entry["abs_delta"] <= gates.prob_p99_abs) / max(
        len(per_document), 1
    )
    gate_rows = [
        {
            "gate": "p99_abs_delta",
            "passed": bool(p99 <= gates.prob_p99_abs),
            "value": p99,
            "bound": gates.prob_p99_abs,
            "referent": "|served - reference|, 99th percentile over all documents",
        },
        {
            "gate": "max_abs_delta",
            "passed": bool(worst <= gates.prob_max_abs),
            "value": worst,
            "bound": gates.prob_max_abs,
            "referent": "max |served - reference| over all documents",
        },
        {
            "gate": "kendall_tau_median",
            "passed": bool(median_tau is not None and median_tau >= gates.tau_min),
            "value": median_tau,
            "bound": gates.tau_min,
            "referent": "median per-query Kendall tau between served and reference scores",
        },
    ]
    return {
        "score_scale": scale,
        "n_queries": len(per_query),
        "n_documents": len(per_document),
        "per_document": per_document,
        "per_query": per_query,
        "abs_delta_p99": p99,
        "abs_delta_max": worst,
        "within_p99_fraction": within_p99,
        "kendall_tau_median": median_tau,
        "kendall_tau_defined_queries": len(taus),
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows)),
    }


def _vector_stage2(
    recipe: Recipe,
    pairs: list[dict[str, Any]],
    reference: Reference,
    client: EngineClient,
    gates: ResolvedGates,
) -> dict[str, Any]:
    """Vectors from the served endpoint against the reference: cosine floor per vector (per token)."""
    multi = recipe.role == "multi_vector"
    per_vector: list[dict[str, Any]] = []
    for row_index, row in enumerate(pairs):
        query = row["query"]
        documents: list[str] = row["documents"]
        served_texts = [recipe.client.doc_prompt + document for document in documents]
        served_texts.append(recipe.client.query_prompt + query)
        served_items: list[tuple[np.ndarray, str, list[int] | None]] = (
            client.pooling(served_texts)
            if multi
            else [(np.asarray(v, dtype=np.float32), "float32", None) for v in client.embeddings(served_texts)]
        )
        # The reference gets the raw texts and composes its own prompts; the role tells the two sides apart.
        reference_items = reference.embed(documents, "document") + reference.embed([query], "query")
        referents = [f"query {row_index} document {i}" for i in range(len(documents))]
        referents.append(f"query {row_index} query")
        _compare_vectors(served_items, reference_items, referents, multi, gates, per_vector)
    return _vector_summary(recipe, per_vector, gates, multi)


def _compare_vectors(
    served_items: list[tuple[np.ndarray, str, list[int] | None]],
    reference_items: list[Any],
    referents: list[str],
    multi: bool,
    gates: ResolvedGates,
    per_vector: list[dict[str, Any]],
) -> None:
    """Pair served with reference vectors positionally and record the cosine rows (per vector, or per token)."""
    if len(served_items) != len(reference_items):
        raise HarnessError(f"engine returned {len(served_items)} vectors, reference produced {len(reference_items)}")
    for position, (served_item, reference_vector) in enumerate(zip(served_items, reference_items, strict=True)):
        flat, _dtype, shape = served_item
        reference_array = np.asarray(reference_vector)
        if multi:
            served = _reshape_tokens(flat, shape, reference_array.shape[0], reference_array.shape[1])
            reference_f16 = reference_array.astype(np.float16).astype(np.float32)
            for token_index in range(min(served.shape[0], reference_f16.shape[0])):
                cosine = _cosine(served[token_index], reference_f16[token_index])
                per_vector.append(
                    {
                        "referent": f"{referents[position]} token {token_index}",
                        "cosine": cosine,
                        "within": bool(cosine >= gates.vec_min_cosine),
                    }
                )
            if served.shape[0] != reference_f16.shape[0]:
                per_vector.append(
                    {
                        "referent": f"{referents[position]} token count",
                        "cosine": None,
                        "within": False,
                        "note": f"engine returned {served.shape[0]} tokens, reference {reference_f16.shape[0]}",
                    }
                )
        else:
            cosine = _cosine(flat, reference_array)
            per_vector.append(
                {"referent": referents[position], "cosine": cosine, "within": bool(cosine >= gates.vec_min_cosine)}
            )


def _reshape_tokens(flat: np.ndarray, shape: list[int] | None, n_tokens: int, dim: int) -> np.ndarray:
    """Reshape one /pooling payload into ``(n_tokens, dim)``, by the response's shape or the reference's shape."""
    if shape and len(shape) == 2:
        return flat.reshape(shape[0], shape[1])
    if flat.size != n_tokens * dim:
        raise HarnessError(
            f"/pooling returned {flat.size} values; expected {n_tokens} tokens x {dim} dimensions "
            "(send shape metadata or check the reference)"
        )
    return flat.reshape(n_tokens, dim)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The cosine of two vectors, in float64; 0.0 when either is zero."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / norm) if norm else 0.0


def _vector_summary(
    recipe: Recipe, per_vector: list[dict[str, Any]], gates: ResolvedGates, multi: bool
) -> dict[str, Any]:
    """Aggregate the per-vector cosines into the gate row."""
    cosines = [entry["cosine"] for entry in per_vector if entry["cosine"] is not None]
    worst = min(cosines) if cosines else None
    gate_rows = [
        {
            "gate": "min_cosine",
            "passed": bool(worst is not None and worst >= gates.vec_min_cosine),
            "value": worst,
            "bound": gates.vec_min_cosine,
            "referent": f"cosine per {'token' if multi else 'vector'}, after the same float16 cast"
            if multi
            else "cosine per vector",
        }
    ]
    return {
        "score_scale": recipe.reference.score_scale,
        "embed_dtype": gates.embed_dtype,
        "multi_vector": multi,
        "n_vectors": len(per_vector),
        "per_vector": per_vector,
        "cosine_min": worst,
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows)),
    }
