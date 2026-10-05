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
from .prompt import TokenizerAdapter, anchor_report, served_prompt_text, template_render_check
from .reference import Reference

__all__ = ["load_pairs", "stage1_prompts", "stage2_scores"]

_SNIPPET = 80


def stage1_prompts(
    recipe: Recipe,
    pairs: list[dict[str, Any]],
    reference: Reference,
    tokenizer: TokenizerAdapter,
    *,
    limit: int | None = None,
    over_length_per_shape: int = 20,
) -> dict[str, Any]:
    """Stage 1: the served prompt's token ids must equal the reference's, with zero tolerance.

    Inputs: the recipe, the sampled pairs, the loaded reference and the recipe's tokenizer (a reference-provided
    ``tokenizer()`` hook or the recipe's ``client.tokenizer``).  In addition to the in-budget pairs, every
    declared shape is sampled on purpose with over-length inputs (at least ``over_length_per_shape`` per shape,
    longer than ``client.max_tokens``): every anchor must survive the client's cut, on the served render and on
    ``reference.render``.  The report carries ``anchor_check`` separately from the token-id mismatches, and
    ``template_render_check`` when ``serve.chat_template`` is set (the declared shapes must render to the same
    token ids as the template file).  ``limit`` truncates the in-budget run (a quick CPU check).
    """
    checked = 0
    rows = pairs if limit is None else pairs[:limit]
    for row_index, row in enumerate(rows):
        for document_index, document in enumerate(row["documents"]):
            served_ids = tokenizer.encode(
                served_prompt_text(recipe, row["query"], document, tokenizer),
                add_special_tokens=bool(recipe.client.add_special_tokens),
            )
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
                    "anchor_check": None,
                    "template_render_check": None,
                }
            checked += 1
    anchor_check = (
        stage1_anchor_check(recipe, reference, tokenizer, pairs, over_length_per_shape=over_length_per_shape)
        if recipe.client.template is not None
        else {"anchor": None, "passed": True, "failures": [], "note": "no declared shapes to audit"}
    )
    template_check = stage1_template_check(recipe, tokenizer, pairs)
    return {
        "pairs": len(pairs),
        "checked": checked,
        "passed": bool(anchor_check["passed"] and (template_check is None or template_check["passed"])),
        "mismatch": None,
        "anchor_check": anchor_check,
        "template_render_check": template_check,
    }


__all__ = ["load_pairs", "stage1_prompts", "stage2_scores"]

_SNIPPET = 80
_DTYPE_NUMPY = {"float16": np.float16, "float32": np.float32}


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


def stage2_scores(
    recipe: Recipe,
    base_url: str,
    pairs: list[dict[str, Any]],
    reference: Reference,
    *,
    served_model_name: str | None = None,
    timeout_s: float | None = None,
    device: str = "cpu",
    tokenizer: TokenizerAdapter | None = None,
) -> dict[str, Any]:
    """Stage 2: served scores or vectors against the in-process reference, under the recipe's gates.

    A rerank recipe sends one ``/rerank`` request per sampled query (the whole candidate set, ``top_n`` equal to
    the number of documents); an ``embed`` recipe compares dense vectors from ``/v1/embeddings``; a
    ``multi_vector`` recipe compares per-token vectors from ``/pooling`` after the same float16 cast.  The report
    carries every number with its referent and one row per gate; ``passed`` is true only when every gate holds.
    The reference's model is loaded first (``device``, ``cpu`` by default: the wave's engines hold the GPUs).
    A recipe with declared shapes needs the ``tokenizer`` (the same adapter stage 1 used) to assemble the
    served prompts.
    """
    if recipe.reference.kind == "stored_scores":
        raise HarnessError(
            "stage 2 needs a runnable reference; reference.kind=stored_scores supports stage 1 and stage 3 only"
        )
    reference.load(device)
    if recipe.client.template is not None and tokenizer is None:
        raise HarnessError(
            f"recipe {recipe.id}: stage 2 renders the declared shapes, which needs the recipe's tokenizer "
            "(pass the stage-1 adapter, or install rcp-ndcg-vllm[reference])"
        )
    gates = resolve_gates(recipe)
    with EngineClient(recipe, base_url, served_model_name=served_model_name, timeout_s=timeout_s or 300.0) as client:
        if recipe.role == "rerank":
            return _rerank_stage2(recipe, pairs, reference, client, gates)
        return _vector_stage2(recipe, pairs, reference, client, gates, tokenizer)


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


def _pair_over_cap(recipe: Recipe, tokenizer: TokenizerAdapter, query: str, document: str) -> int:
    """The token length of one pair's UNCUT served prompt (the wire never carries more than the budget)."""
    served_prompt_text_len = len(
        tokenizer.encode(
            served_prompt_text(recipe, _served_query(recipe, query), document, tokenizer),
            add_special_tokens=bool(recipe.client.add_special_tokens),
        )
    )
    return served_prompt_text_len


def _rerank_stage2(
    recipe: Recipe,
    pairs: list[dict[str, Any]],
    reference: Reference,
    client: EngineClient,
    gates: ResolvedGates,
) -> dict[str, Any]:
    """Scores from the served /rerank against the reference, with the scale's gates and the per-query tau.

    With ``reference.known_deviations: [anchor_drop_over_cap]``, pairs whose uncut prompt exceeds
    ``client.max_tokens`` are excluded from the gates (the reference deliberately drops its anchors there, so
    served and reference may differ by design) and reported in a separate, non-gating ``over_cap`` table.
    """
    scale = recipe.reference.score_scale
    deviation = "anchor_drop_over_cap" in recipe.reference.known_deviations
    per_document: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    for row_index, row in enumerate(pairs):
        served = client.rerank(
            _served_query(recipe, row["query"]),
            row["documents"],
            instruction=_instruction_field(recipe),
            use_activation=recipe.client.use_activation,
        )
        values = reference.score(row["query"], row["documents"], recipe.client.default_instruction)
        for document_index, (served_score, reference_score) in enumerate(zip(served, values, strict=True)):
            document = row["documents"][document_index]
            uncut_tokens = _pair_tokens(recipe, row["query"], document, reference)
            over = uncut_tokens > recipe.client.max_tokens
            delta = abs(served_score - reference_score)
            entry = {
                "query_index": row_index,
                "document_index": document_index,
                "query": row["query"],
                "document": document[:_SNIPPET],
                "served": served_score,
                "reference": reference_score,
                "abs_delta": delta,
                "bound": _score_bound(gates, scale, reference_score),
                "within": bool(delta <= _score_bound(gates, scale, reference_score)),
                "over_cap": over,
            }
            if over and deviation:
                over_cap.append(entry)
                continue
            per_document.append(entry)
        if not deviation or any(
            _pair_tokens(recipe, row["query"], document, reference) <= recipe.client.max_tokens
            for document in row["documents"]
        ):
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
    summary = _rerank_summary(per_document, per_query, gates, scale)
    summary["over_cap"] = {
        "known_deviation": deviation,
        "n_pairs": len(over_cap),
        "gating": False,
        "pairs": over_cap,
        "passed": True,
        "referent": "pairs whose uncut prompt exceeds client.max_tokens; served and reference may differ by "
        "design when reference.known_deviations declares anchor_drop_over_cap",
    }
    return summary


def _pair_tokens(recipe: Recipe, query: str, document: str, reference: Reference) -> int:
    """The uncut pair's token count, measured from the reference's own render."""
    return len(reference.render(_served_query(recipe, query), document, recipe.client.default_instruction))


def _rerank_summary(
    per_document: list[dict[str, Any]], per_query: list[dict[str, Any]], gates: ResolvedGates, scale: str
) -> dict[str, Any]:
    """Aggregate the per-document deltas and per-query taus into the gate rows of the score's scale."""
    deltas = [entry["abs_delta"] for entry in per_document]
    p99 = float(np.percentile(deltas, 99)) if deltas else 0.0
    worst = max(deltas) if deltas else 0.0
    within_p99_fraction = sum(1 for entry in per_document if entry["abs_delta"] <= gates.prob_p99_abs) / max(
        len(per_document), 1
    )
    taus = [entry["kendall_tau"] for entry in per_query if entry["kendall_tau"] is not None]
    median_tau = float(np.median(taus)) if taus else None
    tau_row = {
        "gate": "kendall_tau_median",
        "passed": bool(median_tau is not None and median_tau >= gates.tau_min),
        "value": median_tau,
        "bound": gates.tau_min,
        "referent": "median per-query Kendall tau between served and reference scores",
    }
    if scale == "probability":
        gate_rows = [
            {
                "gate": "p99_documents_within",
                "passed": bool(within_p99_fraction >= 0.99),
                "value": within_p99_fraction,
                "bound": 0.99,
                "referent": f"fraction of documents with |served - reference| <= {gates.prob_p99_abs}",
            },
            {
                "gate": "max_abs_delta",
                "passed": bool(worst <= gates.prob_max_abs),
                "value": worst,
                "bound": gates.prob_max_abs,
                "referent": "max |served - reference| over all documents",
            },
            tau_row,
        ]
    elif scale == "logit":
        ratios = [
            entry["abs_delta"] / (1.0 + abs(entry["reference"])) if entry["reference"] is not None else 0.0
            for entry in per_document
        ]
        worst_ratio = max(ratios) if ratios else 0.0
        gate_rows = [
            {
                "gate": "max_relative_delta",
                "passed": bool(worst_ratio <= gates.logit_rel_abs),
                "value": worst_ratio,
                "bound": gates.logit_rel_abs,
                "referent": "max |served - reference| / (1 + |reference|) over all documents",
            },
            tau_row,
        ]
    else:
        gate_rows = [
            {
                "gate": "max_abs_delta",
                "passed": bool(worst <= gates.cos_max_abs),
                "value": worst,
                "bound": gates.cos_max_abs,
                "referent": "max |served - reference| over all documents",
            },
            tau_row,
        ]
    return {
        "score_scale": scale,
        "n_queries": len(per_query),
        "n_documents": len(per_document),
        "per_document": per_document,
        "per_query": per_query,
        "abs_delta_p99": p99,
        "abs_delta_max": worst,
        "within_p99_fraction": within_p99_fraction,
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
    tokenizer: TokenizerAdapter | None = None,
) -> dict[str, Any]:
    """Vectors from the served endpoint against the reference: cosine floor per vector (per token)."""
    multi = recipe.role == "multi_vector"
    per_vector: list[dict[str, Any]] = []
    for row_index, row in enumerate(pairs):
        query = row["query"]
        documents: list[str] = row["documents"]
        if recipe.client.template is not None:
            served_texts = [
                served_prompt_text(recipe, "", document, tokenizer, shape="document") for document in documents
            ]
            served_texts.append(served_prompt_text(recipe, query, "", tokenizer, shape="query"))  # fmt: skip
        else:
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
            reference_cast = reference_array.astype(_DTYPE_NUMPY[gates.embed_dtype]).astype(np.float32)
            for token_index in range(min(served.shape[0], reference_cast.shape[0])):
                cosine = _cosine(served[token_index], reference_cast[token_index])
                per_vector.append(
                    {
                        "referent": f"{referents[position]} token {token_index}",
                        "cosine": cosine,
                        "within": bool(cosine >= gates.vec_min_cosine),
                    }
                )
            if served.shape[0] != reference_cast.shape[0]:
                per_vector.append(
                    {
                        "referent": f"{referents[position]} token count",
                        "cosine": None,
                        "within": False,
                        "note": f"engine returned {served.shape[0]} tokens, reference {reference_cast.shape[0]}",
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
        "passed": bool(all(row["passed"] for row in gate_rows) and all(entry["within"] for entry in per_vector)),
    }


def _shape_names(recipe: Recipe) -> list[str]:
    """The recipe's declared shapes, in the fixed order (query, document, pair)."""
    template = recipe.client.template
    assert template is not None  # anchor checking only runs for templated recipes
    return [name for name in ("query", "document", "pair") if getattr(template, name) is not None]


def _over_length_text(seed: str, max_tokens: int, tokenizer: TokenizerAdapter, index: int) -> str:
    """One over-length sample: the seed text repeated deterministically past the cap."""
    unit = f"{seed} part {index}"
    one = len(tokenizer.encode(unit))
    repetitions = max(max_tokens // max(one, 1) + 2, 2)
    return " ".join([unit] * repetitions)


def stage1_anchor_check(
    recipe: Recipe,
    reference: Reference,
    tokenizer: TokenizerAdapter,
    pairs: list[dict[str, Any]],
    *,
    over_length_per_shape: int = 20,
) -> dict[str, Any]:
    """The anchor audit: over-length inputs must keep every anchor after the client's cut.

    For every declared shape, ``over_length_per_shape`` inputs are padded beyond ``client.max_tokens``; the
    assembled render and ``reference.render(...)`` are both asserted to carry every anchor — the tail (or head)
    fixed segments plus the post-processor end token, or the declared markers.  The result is reported
    separately from the token-id mismatches in ``equivalence.json``.
    """
    seed = pairs[0] if pairs else {"query": "anchor check", "documents": ["anchor check document"]}
    failures: list[dict[str, Any]] = []
    for shape in _shape_names(recipe):
        for index in range(max(over_length_per_shape, 1)):
            document = _over_length_text(seed["documents"][0], recipe.client.max_tokens, tokenizer, index)
            served_ids = tokenizer.encode(
                served_prompt_text(recipe, seed["query"], document, tokenizer),
                add_special_tokens=bool(recipe.client.add_special_tokens),
            )
            served_report = anchor_report(recipe, tokenizer, shape, served_ids)
            if not served_report["passed"]:
                failures.append({"side": "served", "shape": shape, "index": index, **served_report})
            reference_ids = reference.render(seed["query"], document, recipe.client.default_instruction)
            reference_report = anchor_report(recipe, tokenizer, shape, reference_ids)
            if not reference_report["passed"]:
                failures.append(
                    {
                        "side": "reference",
                        "shape": shape,
                        "index": index,
                        "passed": False,
                        "failures": reference_report["failures"],
                        "note": "reference.render lost an anchor on an over-cap pair",
                    }
                )
    return {
        "anchor": recipe.client.template.anchor if recipe.client.template else None,
        "over_length_per_shape": over_length_per_shape,
        "shapes": _shape_names(recipe),
        "passed": not failures,
        "failures": failures,
    }


def stage1_template_check(
    recipe: Recipe, tokenizer: TokenizerAdapter, pairs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """When ``serve.chat_template`` and ``client.template`` are both set: the shapes must match the file.

    A template file without a declared shape block has nothing to prove the shapes against, so the check is
    skipped (reported ``None``); the token-id equality against the reference still applies.
    """
    if recipe.serve.chat_template is None or recipe.client.template is None or not pairs:
        return None
    row = pairs[0]
    return template_render_check(recipe, tokenizer, row["query"], row["documents"][0])
