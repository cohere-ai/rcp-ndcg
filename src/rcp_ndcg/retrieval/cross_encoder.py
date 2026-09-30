"""Cross-encoder reranking: a served engine, or a vendor's own implementation.

Two paths, because there are only two kinds of reranker in practice:

* ``framework: vllm`` -- the model is served by vLLM and scored through ``/rerank``. The engine owns tokenisation,
  batching, truncation, padding and (for late-interaction checkpoints) MaxSim on the worker. This module sends
  ``(query, documents)`` and reads floats back; truncation budgets travel as request fields
  (:data:`MAX_SEQ_LENGTH`, :data:`MAX_QUERY_LENGTH`), so truncation happens where the tokeniser is.
* everything else -- a vendor reranker with its own ``predict(query, docs)`` implementation (Qwen3-Reranker, ZeRank,
  ctxl-rerank, Jina v3) or a hosted API (Cohere, Voyage), where the scoring *is* the vendor's code and
  reimplementing it would change the numbers. The in-process ones run on the torch path with ``accelerate``
  sharding.

:func:`rcp_ndcg.retrieval.rerank` builds the :class:`RerankSettings` from a
:class:`~rcp_ndcg.retrieval.RerankerConfig` and calls :func:`rerank_examples`, which scores a batch of
:class:`RankingExample` records with per-query crash checkpointing on the served path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rcp_ndcg_core._records import RankingExample

from rcp_ndcg.retrieval.accel import AccelState
from rcp_ndcg.retrieval.external_rerankers import EXTERNAL_FRAMEWORKS
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: The framework name for the served path.
VLLM_FRAMEWORK = "vllm"

MAX_SEQ_LENGTH = 8192
"""Combined ``(query, document)`` token budget: ``truncate_prompt_tokens`` on the served path, the model's
``max_seq_len`` in process. Must not exceed the server's ``--max-model-len``."""

MAX_QUERY_LENGTH = 4096
"""Query tokens on the served path (``max_tokens_per_query``)."""


@dataclass(frozen=True)
class RerankSettings:
    """What :func:`rerank_examples` needs to score with one reranker.

    Attributes:
        model_name: The HuggingFace repo id or local path, or the served or API model id.
        framework: ``vllm`` (a served ``/rerank`` endpoint), or one of
            :data:`~rcp_ndcg.retrieval.external_rerankers.EXTERNAL_FRAMEWORKS`.
        revision: The Hub revision an in-process reranker loads.
        api_base: The root URL of the vLLM server, or a vendor API's override.
        api_key: The bearer token of the server or vendor API.
        concurrency: Served rerank requests in flight (one request is one query's candidate set).
        external_batch_size: Documents per forward pass (Qwen3-Reranker, ctxl-rerank).
        request_size: Documents per API request (Cohere, Voyage).
        timeout_s, connect_timeout_s, max_retries: The request policy of a served or hosted reranker.
    """

    model_name: str
    framework: str = VLLM_FRAMEWORK
    revision: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    concurrency: int = 8
    external_batch_size: int = 8
    request_size: int = 100
    timeout_s: float = 600.0
    connect_timeout_s: float = 5.0
    max_retries: int = 4


# ---------------------------------------------------------------------------
# Checkpointing -- framework-agnostic, so a crash costs one query.
# ---------------------------------------------------------------------------


def _iter_checkpoint_records(ckpt_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield score records from every ``rank*.jsonl`` in *ckpt_dir*.

    Tolerates a truncated trailing line -- the common crash signature is a process
    that died mid-flush, and one unparseable line should not invalidate the work
    before it.
    """
    if not ckpt_dir.exists():
        return
    for shard in sorted(ckpt_dir.glob("rank*.jsonl")):
        with shard.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _checkpoint_scored_queries(ckpt_dir: Path) -> set[str]:
    """Query ids already fully scored.

    Per *query* rather than per pair: a rerank request is atomic at the query
    level, so a partially-written query has nothing usable to resume from.
    """
    return {str(record["q"]) for record in _iter_checkpoint_records(ckpt_dir)}


def _checkpoint_scores(ckpt_dir: Path) -> dict[str, dict[str, float]]:
    """``{query_id: {doc_id: score}}`` across all rank checkpoints."""
    by_qid: dict[str, dict[str, float]] = {}
    for record in _iter_checkpoint_records(ckpt_dir):
        by_qid.setdefault(str(record["q"]), {}).update(
            {str(doc_id): float(score) for doc_id, score in record["s"].items()}
        )
    return by_qid


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def rerank_examples(
    examples: Sequence[RankingExample],
    cfg: RerankSettings,
    *,
    checkpoint_dir: Path | str | None = None,
) -> list[RankingExample] | None:
    """Rerank *examples*, returning them with ``scores`` set.

    Documents must already be populated (``docs`` or ``contents``). ``doc_ids`` order is never changed: scores
    come back aligned with it, and callers sort downstream if they want to. Each rank of an ``accelerate launch``
    scores a disjoint slice.

    Args:
        examples: Ranking examples to rerank.
        cfg: The reranker.
        checkpoint_dir: Served path only: each rank appends one record per scored query to
            ``<dir>/rank<NNN>.jsonl`` and flushes immediately, so a crash costs at most the queries in flight; a
            rerun skips the queries already there.

    Returns:
        On rank 0 (or single-process): the rescored examples in input order. On other ranks: ``None``.
    """
    state = AccelState()
    if cfg.framework != VLLM_FRAMEWORK and cfg.framework not in EXTERNAL_FRAMEWORKS:
        raise ValueError(
            f"unknown reranker framework {cfg.framework!r}; supported: "
            f"{', '.join(sorted({VLLM_FRAMEWORK, *EXTERNAL_FRAMEWORKS}))}. A sequence-classification "
            f"cross encoder is served: `vllm serve {cfg.model_name} --runner pooling`, then backend: http."
        )
    if cfg.framework != VLLM_FRAMEWORK:
        return _rerank_examples_external(examples, cfg, state=state)
    return _rerank_examples_vllm(examples, cfg, state=state, checkpoint_dir=checkpoint_dir)


def _rerank_examples_vllm(
    examples: Sequence[RankingExample],
    cfg: RerankSettings,
    *,
    state: AccelState,
    checkpoint_dir: Path | str | None,
) -> list[RankingExample] | None:
    """Score every example through a served ``/rerank`` endpoint.

    Sharded per *query*, not per pair: one request carries a whole candidate set,
    which is what lets the engine reuse the query's prefix across the documents.
    Concurrency is a thread pool rather than a process pool because the work is a
    socket wait.
    """
    from rcp_ndcg.retrieval.vllm_http import VllmPoolingClient

    if not cfg.api_base:
        raise ValueError("a served reranker needs the server's base_url")
    client = VllmPoolingClient(
        cfg.api_base,
        model=cfg.model_name,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        connect_timeout_s=cfg.connect_timeout_s,
        max_retries=cfg.max_retries,
    )

    local_examples, _ = state.shard(examples)
    ckpt_file, ckpt_dir = _prepare_checkpoint(checkpoint_dir, state=state)
    if ckpt_dir is not None:
        already = _checkpoint_scored_queries(ckpt_dir)
        if already:
            before = len(local_examples)
            local_examples = [example for example in local_examples if str(example.id) not in already]
            if state.is_main_process:
                logger.info(f"rerank resume: {before - len(local_examples)}/{before} queries already scored")

    if state.is_main_process:
        logger.info(
            f"vLLM rerank: {len(local_examples):,} local queries "
            f"(rank={state.process_index}/{state.num_processes}, concurrency={cfg.concurrency}, "
            f"model={cfg.model_name} at {cfg.api_base})"
        )

    def score_one(example: RankingExample) -> tuple[str, dict[str, float]]:
        return str(example.id), _score_example(client, example)

    local_scores: dict[str, dict[str, float]] = {}
    ckpt_fh = ckpt_file.open("a") if ckpt_file is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            for query_id, scores in pool.map(score_one, local_examples):
                if ckpt_fh is None:
                    local_scores[query_id] = scores
                    continue
                ckpt_fh.write(json.dumps({"q": query_id, "s": scores}) + "\n")
                # Flush + fsync per query so a crash on the next one keeps this one.
                ckpt_fh.flush()
                os.fsync(ckpt_fh.fileno())
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()

    state.wait_for_everyone()
    if ckpt_dir is not None:
        if not state.is_main_process:
            return None
        by_qid = _checkpoint_scores(ckpt_dir)
    else:
        gathered = state.gather_object(local_scores)
        if not state.is_main_process:
            return None
        by_qid = {}
        for chunk in gathered:
            by_qid.update(chunk)

    return _apply_scores(examples, by_qid)


def _score_example(client: Any, example: RankingExample) -> dict[str, float]:
    """One query's scores, keyed by doc id.

    The query carries its instruction (``Task: ...``) when the example has one. Text is sent as a plain string, so a
    text-only server receives exactly the request it would without content parts.
    """
    if not example.doc_ids:
        return {}
    query = example.format_content()
    scores = client.rerank(
        query if query.has_media else query.text,
        [content if content.has_media else content.text for content in example.doc_contents],
        max_tokens_per_query=MAX_QUERY_LENGTH,
        truncate_prompt_tokens=MAX_SEQ_LENGTH,
    )
    return {str(doc_id): float(score) for doc_id, score in zip(example.doc_ids, scores, strict=True)}


def _apply_scores(
    examples: Sequence[RankingExample],
    by_qid: dict[str, dict[str, float]],
) -> list[RankingExample]:
    """Attach scores to *examples* in input order.

    A document with no score gets ``-inf`` so it sorts last rather than being
    dropped: the candidate set is part of the run's identity, and silently
    shortening it would change every metric computed downstream.
    """
    out: list[RankingExample] = []
    for example in examples:
        qid_scores = by_qid.get(str(example.id), {})
        scores = [qid_scores.get(str(doc_id), float("-inf")) for doc_id in example.doc_ids]
        out.append(example.model_copy(update={"scores": scores}))
    return out


def _prepare_checkpoint(checkpoint_dir: Path | str | None, *, state: AccelState) -> tuple[Path | None, Path | None]:
    """``(this rank's checkpoint file, the checkpoint dir)``, or ``(None, None)``."""
    if checkpoint_dir is None:
        return None, None
    ckpt_dir = Path(checkpoint_dir)
    state.wait_for_everyone()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"rank{state.process_index:03d}.jsonl", ckpt_dir


def _rerank_examples_external(
    examples: Sequence[RankingExample],
    cfg: RerankSettings,
    *,
    state: AccelState,
) -> list[RankingExample] | None:
    """Rerank via a vendor's own ``predict(query, docs)`` implementation.

    Sharded per *example*: these backends own their batching, and splitting a
    query's documents across ranks would break the ones that score a candidate set
    jointly.
    """
    from rcp_ndcg.retrieval.external_rerankers import load_external_model

    model = load_external_model(cfg, device=str(state.device))
    model.to(str(state.device))

    local_examples, local_indices = state.shard(examples)
    local_out: list[RankingExample] = []
    for example in local_examples:
        if not example.docs:
            local_out.append(example.model_copy(update={"scores": [float("-inf")] * len(example.doc_ids)}))
            continue
        scores = model.predict(example.query, list(example.docs))
        if len(scores) != len(example.doc_ids):
            raise RuntimeError(
                f"{cfg.framework} reranker returned {len(scores)} scores for {len(example.doc_ids)} docs "
                f"(query_id={example.query_id})."
            )
        local_out.append(example.model_copy(update={"scores": [float(score) for score in scores]}))

    state.wait_for_everyone()
    gathered_items = state.gather_object(local_out)
    gathered_indices = state.gather_object(local_indices)
    if not state.is_main_process:
        return None
    return state.reorder_global(gathered_items, gathered_indices, len(examples))


__all__ = ["MAX_QUERY_LENGTH", "MAX_SEQ_LENGTH", "VLLM_FRAMEWORK", "RerankSettings", "rerank_examples"]
