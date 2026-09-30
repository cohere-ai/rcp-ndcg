"""Rerankers that score with the vendor's own code: in-process causal-LM rerankers and hosted rerank APIs.

A sequence-classification cross encoder is served (``vllm serve --runner pooling``) and scored through ``/rerank``
by :mod:`rcp_ndcg.retrieval.cross_encoder`. Many strong rerankers do not fit that shape -- they are causal LMs scored
on a single "yes"/"no" token, or hosted APIs -- and this module gives them one interface::

    predict(query: str, docs: list[str]) -> list[float]

returning one score per document, aligned with ``docs`` (higher is more relevant). :func:`load_external_model` maps
a :class:`~rcp_ndcg.retrieval.cross_encoder.RerankSettings` to the backend by its ``framework``:

* ``qwen_og``   -- Qwen3-Reranker (causal LM, yes/no token log-softmax).
* ``zerank``    -- ZeroEntropy ZeRank (chat template + "Yes"-token logit).
* ``contextual``-- ContextualAI ctxl-rerank (causal LM, vocab-pos-0 logit).
* ``jina_hf``   -- Jina reranker v3 (local ``model.rerank()`` API).
* ``voyage``    -- Voyage AI rerank API (``rerank-2.5``, ``rerank-2.5-lite``) over its public HTTP API.
* ``cohere``    -- Cohere Rerank (``rerank-v4.0-pro``, ``rerank-v4.0-fast``) over its public HTTP API.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from rcp_ndcg.retrieval.accel import resolve_torch_dtype
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.retrieval.cross_encoder import RerankSettings

logger = get_logger(__name__)

#: Frameworks routed through this module (everything that is not the served ``vllm`` path).
EXTERNAL_FRAMEWORKS = frozenset({"qwen_og", "zerank", "contextual", "jina_hf", "voyage", "cohere"})

#: The dtype the in-process rerankers load their weights in.
DTYPE = "bfloat16"


class EvalCrossEncoder(ABC):
    """Abstract base for ``predict(query, docs)``-style rerankers."""

    @abstractmethod
    def predict(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per document, aligned with ``docs``."""

    def to(self, device: str) -> EvalCrossEncoder:
        """Move the model to ``device``.  Default is a no-op (API backends)."""
        return self


# ---------------------------------------------------------------------------
# Qwen3-Reranker (yes/no token scoring).
# ---------------------------------------------------------------------------


class QwenOGRerank(EvalCrossEncoder):
    """Qwen3-Reranker scored on yes/no token log-softmax.

    Reproduces the released Qwen3-Reranker recipe (its model card's usage code, Apache-2.0, see NOTICE): a causal
    LM is prompted with ``<Instruct>/<Query>/<Document>`` and the relevance score is
    ``softmax([no_logit, yes_logit])[yes]`` at the final position.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-8B",
        max_seq_len: int = 8192,
        batch_size: int = 8,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        dtype: Any = None,
        attn_implementation: str | None = "flash_attention_2",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", revision=revision)
        self.max_length = max_seq_len
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.instruction = instruction

        model_kwargs: dict[str, Any] = {"dtype": dtype or torch.float16, "revision": revision}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def to(self, device: str) -> QwenOGRerank:
        self.model.to(device)
        self.device = device
        return self

    def _process_inputs(self, pairs: list[str]) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_scores(self, inputs: dict) -> list[float]:
        import torch

        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return scores.float().tolist()

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        import torch

        try:
            return self._compute_scores(self._process_inputs(pairs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise
            mid = len(pairs) // 2
            return self._score_pairs(pairs[:mid]) + self._score_pairs(pairs[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        import torch

        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in docs]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                scores.extend(self._score_pairs(pairs[i : i + self.batch_size]))
        if len(scores) != len(docs):
            raise RuntimeError(f"QwenOGRerank: expected {len(docs)} scores, got {len(scores)}")
        return scores


# ---------------------------------------------------------------------------
# ZeroEntropy ZeRank (chat template + "Yes" token logit).
# ---------------------------------------------------------------------------


class ZerankRerank(EvalCrossEncoder):
    """ZeRank reranker: chat template (system=query, user=doc) + Yes logit.

    Adapted from https://huggingface.co/zeroentropy/zerank-1 (Apache-2.0, see NOTICE).  The score is
    ``sigmoid(yes_logit / 5)`` at the final token position.
    """

    def __init__(
        self,
        model_name_or_path: str = "zeroentropy/zerank-2",
        max_seq_len: int = 8192,
        batch_size_tokens: int = 15_000,
        dtype: Any = None,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.max_seq_length = max_seq_len
        self.batch_size_tokens = batch_size_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype or torch.bfloat16, revision=revision
        )
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> ZerankRerank:
        self.model.to(device)
        self.device = device
        return self

    def _format_inputs(self, query: str, docs: list[str]) -> list[str]:
        texts: list[str] = []
        for doc in docs:
            messages = [
                {"role": "system", "content": query.strip()},
                {"role": "user", "content": doc.strip()},
            ]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return texts

    def _batch_logits(self, batch: list[str]) -> list[float]:
        import torch

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch, padding=True, return_tensors="pt", truncation=True, max_length=self.max_seq_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, use_cache=False)
                attention_mask = inputs["attention_mask"]
                last_positions = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(outputs.logits.shape[0], device=self.device)
                last_logits = outputs.logits[batch_indices, last_positions]
                yes_logits = last_logits[:, self.yes_token_id]
                return [float(v) / 5.0 for v in yes_logits]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._batch_logits(batch[:mid]) + self._batch_logits(batch[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        texts = self._format_inputs(query, docs)

        # Length-descending permutation for tight token buckets.
        permutation = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        sorted_texts = [texts[i] for i in permutation]

        batches: list[list[str]] = []
        max_length = 0
        for text in sorted_texts:
            text_len = len(text)
            if not batches or (len(batches[-1]) + 1) * max(max_length, text_len) > self.batch_size_tokens:
                batches.append([])
                max_length = 0
            batches[-1].append(text)
            max_length = max(max_length, text_len)

        all_logits: list[float] = []
        for batch in batches:
            all_logits.extend(self._batch_logits(batch))

        from scipy.special import expit

        scores = expit(all_logits).tolist()

        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, scores, strict=True):
            result[orig_idx] = score
        return result


# ---------------------------------------------------------------------------
# ContextualAI ctxl-rerank (vocab-position-0 logit).
# ---------------------------------------------------------------------------


class ContextualRerank(EvalCrossEncoder):
    """ContextualAI ctxl-rerank: causal LM scored on vocab-position-0 logit.

    Based on
    https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b.
    """

    def __init__(
        self,
        model_name_or_path: str = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
        max_seq_len: int = 32768,
        batch_size: int = 32,
        batch_size_tokens: int = 15_000,
        instruction: str | None = None,
        dtype: Any = None,
        attn_implementation: str | None = "flash_attention_2",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.max_seq_len = max_seq_len
        # ``batch_size`` is a hard cap on docs per forward; ``batch_size_tokens``
        # is a padded-area budget (docs * max_char_len) that keeps long-document
        # batches small.  ctxl is a 32k-context causal LM, so a fixed batch of
        # near-max-length docs OOMs -- the token budget prevents that, and the
        # per-batch OOM backoff in ``predict`` is a final safety net.
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "dtype": dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            "revision": revision,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> ContextualRerank:
        self.model.to(device)
        self.device = device
        return self

    def _format_prompts(self, query: str, docs: list[str]) -> list[str]:
        instruction = f" {self.instruction}" if self.instruction else ""
        return [
            f"Check whether a given document contains information helpful to answer the query.\n"
            f"<Document> {doc}\n<Query> {query}{instruction} ??"
            for doc in docs
        ]

    def _forward_scores(self, batch: list[str]) -> list[float]:
        import torch

        try:
            with torch.no_grad():
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Left-padded, so the final position is the last real token for
                # every row; vocab index 0 is ctxl's relevance logit.
                return out.logits[:, -1, 0].float().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._forward_scores(batch[:mid]) + self._forward_scores(batch[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        prompts = self._format_prompts(query, docs)

        # Length-descending permutation for tight token buckets; longest (most
        # OOM-prone) prompts are front-loaded so the backoff triggers early.
        permutation = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        sorted_prompts = [prompts[i] for i in permutation]

        # Padded-area budget: keep ``docs * max_char_len`` under the token budget
        # and never exceed ``batch_size`` docs.  Docs are scored independently, so
        # batching/order only affect throughput -- not the per-doc score.
        batches: list[list[str]] = []
        max_len = 0
        for text in sorted_prompts:
            text_len = len(text)
            over_tokens = bool(batches) and (len(batches[-1]) + 1) * max(max_len, text_len) > self.batch_size_tokens
            over_count = bool(batches) and len(batches[-1]) >= self.batch_size
            if not batches or over_tokens or over_count:
                batches.append([])
                max_len = 0
            batches[-1].append(text)
            max_len = max(max_len, text_len)

        sorted_scores: list[float] = []
        for batch in batches:
            sorted_scores.extend(self._forward_scores(batch))

        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, sorted_scores, strict=True):
            result[orig_idx] = score
        return result


# ---------------------------------------------------------------------------
# Jina reranker v3 (local model.rerank() API).
# ---------------------------------------------------------------------------


class JinaRerank(EvalCrossEncoder):
    """Jina local reranker (v2 / v3) via the model's built-in ``rerank()``."""

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-reranker-v3",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name_or_path
        self.is_v3 = "v3" in model_name_or_path

        model_cls = AutoModel if self.is_v3 else AutoModelForSequenceClassification
        self.model = model_cls.from_pretrained(
            model_name_or_path, dtype="auto", trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, revision=revision)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> JinaRerank:
        self.model.to(device)
        self.device = device
        return self

    def predict(self, query: str, docs: list[str]) -> list[float]:
        non_empty = [(i, d) for i, d in enumerate(docs) if d.strip()]
        if not non_empty:
            return [0.0] * len(docs)

        indices, filtered_docs = zip(*non_empty, strict=True)
        results = self.model.rerank(query, documents=list(filtered_docs))
        scored = [0.0] * len(filtered_docs)
        for r in results:
            scored[r["index"]] = r["relevance_score"]

        output = [0.0] * len(docs)
        for orig_idx, score in zip(indices, scored, strict=True):
            output[orig_idx] = score
        return output


# ---------------------------------------------------------------------------
# Hosted rerank APIs (Cohere, Voyage), over plain HTTP.
# ---------------------------------------------------------------------------


class _HostedRerank(EvalCrossEncoder):
    """A rerank API reached over plain HTTP rather than its SDK: the request is three fields.

    Empty documents are not sent (the APIs refuse them) and score 0. The rest go ``request_size`` per request,
    each request retried with backoff (:func:`rcp_ndcg.retrieval._http.post_json`).
    """

    BASE_URL: ClassVar[str]
    KEY_ENV: ClassVar[tuple[str, ...]]
    VENDOR: ClassVar[str]
    PAUSE_S: ClassVar[float] = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "",
        request_size: int = 100,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        max_retries: int = 8,
    ) -> None:
        from rcp_ndcg.retrieval._http import api_key as resolve_key

        self._api_key = resolve_key(api_key, self.KEY_ENV, what=f"{self.VENDOR} rerank")
        self.model_name = model_name
        self.request_size = request_size
        self.url = f"{(base_url or self.BASE_URL).rstrip('/')}/rerank"
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_retries = max_retries

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        return {"model": self.model_name, "query": query, "documents": docs}

    def _results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _rerank_batch(self, query: str, docs: list[str]) -> list[float]:
        from rcp_ndcg.retrieval._http import post_json

        body = post_json(
            self.url,
            self._payload(query, docs),
            headers={"Authorization": f"Bearer {self._api_key}"},
            what=f"{self.VENDOR} rerank",
            timeout_s=self.timeout_s,
            connect_timeout_s=self.connect_timeout_s,
            max_retries=self.max_retries,
        )
        scores = [0.0] * len(docs)
        for row in self._results(body):
            scores[int(row["index"])] = float(row["relevance_score"])
        return scores

    def predict(self, query: str, docs: list[str]) -> list[float]:
        sent = [index for index, doc in enumerate(docs) if doc.strip()]
        scores = [0.0] * len(docs)
        for start in range(0, len(sent), self.request_size):
            batch = sent[start : start + self.request_size]
            if self.PAUSE_S:
                time.sleep(self.PAUSE_S)
            for index, score in zip(batch, self._rerank_batch(query, [docs[i] for i in batch]), strict=True):
                scores[index] = score
        return scores


class VoyageRerank(_HostedRerank):
    """Voyage AI rerank (``rerank-2.5``, ``rerank-2.5-lite``, ...): ``POST /v1/rerank``.

    The key comes from ``api_key``, else ``VOYAGE_API_KEY``. Requests are spaced by half a second: Voyage enforces
    strict rate limits.
    """

    BASE_URL = "https://api.voyageai.com/v1"
    KEY_ENV = ("VOYAGE_API_KEY",)
    VENDOR = "Voyage"
    PAUSE_S = 0.5

    def _results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return body["data"]


class CohereRerank(_HostedRerank):
    """Cohere Rerank (``rerank-v4.0-pro``, ``rerank-v4.0-fast``, ...): ``POST /v2/rerank``.

    The key comes from ``api_key``, else ``CO_API_KEY`` or ``COHERE_API_KEY``.
    """

    BASE_URL = "https://api.cohere.com/v2"
    KEY_ENV = ("CO_API_KEY", "COHERE_API_KEY")
    VENDOR = "Cohere"

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        return {**super()._payload(query, docs), "top_n": len(docs)}

    def _results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return body["results"]


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def load_external_model(cfg: RerankSettings, device: str | None = None) -> EvalCrossEncoder:
    """Instantiate the reranker selected by ``cfg.framework``.

    The in-process rerankers load ``cfg.revision`` in :data:`DTYPE`, with a
    :data:`~rcp_ndcg.retrieval.cross_encoder.MAX_SEQ_LENGTH`-token budget where the model takes one.
    """
    from rcp_ndcg.retrieval.cross_encoder import MAX_SEQ_LENGTH

    framework = cfg.framework
    if framework == "qwen_og":
        return QwenOGRerank(
            model_name_or_path=cfg.model_name,
            max_seq_len=MAX_SEQ_LENGTH,
            batch_size=cfg.external_batch_size,
            dtype=resolve_torch_dtype(DTYPE),
            device=device,
            revision=cfg.revision,
        )
    if framework == "zerank":
        return ZerankRerank(
            model_name_or_path=cfg.model_name,
            max_seq_len=MAX_SEQ_LENGTH,
            dtype=resolve_torch_dtype(DTYPE),
            device=device,
            revision=cfg.revision,
        )
    if framework == "contextual":
        return ContextualRerank(
            model_name_or_path=cfg.model_name,
            max_seq_len=MAX_SEQ_LENGTH,
            batch_size=cfg.external_batch_size,
            dtype=resolve_torch_dtype(DTYPE),
            device=device,
            revision=cfg.revision,
        )
    if framework == "jina_hf":
        return JinaRerank(model_name_or_path=cfg.model_name, device=device, revision=cfg.revision)
    if framework in ("voyage", "cohere"):
        hosted = VoyageRerank if framework == "voyage" else CohereRerank
        return hosted(
            api_key=cfg.api_key,
            model_name=cfg.model_name,
            request_size=cfg.request_size,
            base_url=cfg.api_base,
            timeout_s=cfg.timeout_s,
            connect_timeout_s=cfg.connect_timeout_s,
            max_retries=cfg.max_retries,
        )
    raise ValueError(
        f"Unknown external framework {framework!r}. "
        f"Supported: {', '.join(sorted(EXTERNAL_FRAMEWORKS))}, or 'vllm' for a served cross encoder."
    )


__all__ = [
    "DTYPE",
    "EXTERNAL_FRAMEWORKS",
    "CohereRerank",
    "ContextualRerank",
    "EvalCrossEncoder",
    "JinaRerank",
    "QwenOGRerank",
    "VoyageRerank",
    "ZerankRerank",
    "load_external_model",
]
