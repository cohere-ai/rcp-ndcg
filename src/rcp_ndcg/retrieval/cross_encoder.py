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

from rcp_ndcg.errors import DataError
from rcp_ndcg.retrieval.accel import AccelState
from rcp_ndcg.retrieval.external_rerankers import EXTERNAL_FRAMEWORKS
from rcp_ndcg.support.identity import hash_payload, short
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


def _checkpoint_key(cfg: RerankSettings, example: RankingExample) -> str:
    """What a checkpointed query's scores are valid for: the reranker, the budgets, the query and its candidates.

    A rerun with another reranker, another depth or other candidates has another key and is scored again.
    """
    payload = {
        "model": cfg.model_name,
        "framework": cfg.framework,
        "revision": cfg.revision,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_query_length": MAX_QUERY_LENGTH,
        "query_id": str(example.id),
        "doc_ids": [str(doc_id) for doc_id in example.doc_ids],
    }
    return short(hash_payload(payload), 16)


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


def _checkpoint_scores(ckpt_dir: Path) -> dict[str, dict[str, float]]:
    """``{checkpoint key: {doc_id: score}}`` across all rank checkpoints (:func:`_checkpoint_key`).

    Per *query* rather than per pair: a rerank request is atomic at the query level, so a partially-written query
    has nothing usable to resume from. Records without a key are not reused.
    """
    by_key: dict[str, dict[str, float]] = {}
    for record in _iter_checkpoint_records(ckpt_dir):
        if isinstance(record, dict) and isinstance(record.get("k"), str) and isinstance(record.get("s"), dict):
            by_key[record["k"]] = {str(doc_id): float(score) for doc_id, score in record["s"].items()}
    return by_key


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
            rerun skips the queries already scored by the same reranker over the same candidates.

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

    keys = [_checkpoint_key(cfg, example) for example in examples]
    key_of = {id(example): key for example, key in zip(examples, keys, strict=True)}
    local_examples, _ = state.shard(examples)
    ckpt_file, ckpt_dir = _prepare_checkpoint(checkpoint_dir, state=state)
    if ckpt_dir is not None:
        already = _checkpoint_scores(ckpt_dir)
        if already:
            before = len(local_examples)
            local_examples = [example for example in local_examples if key_of[id(example)] not in already]
            if state.is_main_process:
                logger.info(f"rerank resume: {before - len(local_examples)}/{before} queries already scored")

    if state.is_main_process:
        logger.info(
            f"vLLM rerank: {len(local_examples):,} local queries "
            f"(rank={state.process_index}/{state.num_processes}, concurrency={cfg.concurrency}, "
            f"model={cfg.model_name} at {cfg.api_base})"
        )

    def score_one(example: RankingExample) -> tuple[str, str, dict[str, float]]:
        return str(example.id), key_of[id(example)], _score_example(client, example)

    local_scores: dict[str, dict[str, float]] = {}
    ckpt_fh = ckpt_file.open("a") if ckpt_file is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            for query_id, key, scores in pool.map(score_one, local_examples):
                if ckpt_fh is None:
                    local_scores[key] = scores
                    continue
                ckpt_fh.write(json.dumps({"q": query_id, "k": key, "s": scores}) + "\n")
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
        by_key = _checkpoint_scores(ckpt_dir)
    else:
        gathered = state.gather_object(local_scores)
        if not state.is_main_process:
            return None
        by_key = {}
        for chunk in gathered:
            by_key.update(chunk)

    return _apply_scores(examples, keys, by_key)


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
    keys: Sequence[str],
    by_key: dict[str, dict[str, float]],
) -> list[RankingExample]:
    """Attach scores to *examples* in input order (``keys``: each example's :func:`_checkpoint_key`).

    Raises:
        DataError: A document has no score. The candidate set is part of the run's identity, so a document is
            neither dropped nor given a made-up score.
    """
    out: list[RankingExample] = []
    for example, key in zip(examples, keys, strict=True):
        scored = by_key.get(key, {})
        missing = [str(doc_id) for doc_id in example.doc_ids if str(doc_id) not in scored]
        if missing:
            raise DataError(
                f"query {example.id!r}: the reranker returned no score for {len(missing)} documents, e.g. "
                f"{missing[:3]}",
                hint="rerun the rerank; scored queries are kept in the checkpoint directory",
            )
        out.append(example.model_copy(update={"scores": [scored[str(doc_id)] for doc_id in example.doc_ids]}))
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
        if not example.doc_ids:
            local_out.append(example.model_copy(update={"scores": []}))
            continue
        if example.docs is None:
            raise DataError(f"query {example.id!r} has no document bodies; hydrate it before reranking")
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
