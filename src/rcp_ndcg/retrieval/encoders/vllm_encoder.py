"""vLLM as an encoder backend, in two modes.

Which mode to use is a transport question, and the answer differs by *stage* of
the same job:

* ``mode: offline`` -- an in-process ``vllm.LLM`` with ``runner="pooling"``.
  For building a corpus index. A late-interaction page embedding is ~1030 vectors
  at 128 dims, so a 40k-page corpus is ~21 GB of vectors; JSON-encoding that over
  a socket is the wrong transport regardless of how fast the server is.
* ``mode: http`` -- ``/pooling`` on an already-running server. For queries, small
  corpora, and any context where the weights are already resident somewhere else
  and loading a second copy is what would cost.

Both produce the same :class:`~rcp_ndcg.retrieval.encoder.Embeddings`, and the
pooling task decides the layout: ``token_embed`` is ragged (one vector per
token/patch, MaxSim downstream), ``embed`` is one vector per item.

Multimodal input goes through the model's own HuggingFace chat template rather
than a template string written here. The template is part of the checkpoint, and
a hand-copied one drifts from the weights silently -- producing embeddings that
are wrong in a way no shape check catches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import DependencyError
from rcp_ndcg.retrieval.encoder import Embeddings, Encoder, EncodeRole
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: Pooling tasks this encoder knows how to read back. ``token_embed`` returns one
#: vector per token; ``embed`` returns one per item.
MULTI_VECTOR_TASK = "token_embed"
SINGLE_VECTOR_TASK = "embed"
POOLING_TASKS = (MULTI_VECTOR_TASK, SINGLE_VECTOR_TASK)

DEFAULT_BATCH_SIZE = 64
"""Items per ``LLM.encode`` call. vLLM does its own continuous batching, so this
only bounds how much decoded media is resident at once -- which for page images is
the constraint that actually binds."""


class VllmEncoder(Encoder):
    """An embedding model served by vLLM, offline in-process or over HTTP.

    Args:
        model_name: HuggingFace repo id or local path. Also the model id sent to
            the server in ``http`` mode.
        mode: ``offline`` (in-process engine) or ``http`` (running server).
        revision: The Hub revision the offline engine loads; ``http`` mode refuses one (the server's weights are
            what they are).
        pooling_task: ``token_embed`` for late interaction, ``embed`` for a single
            vector per item. Required: a config says which kind of index it is
            building, and nothing defaults it to multi-vector.
        api_base: Server root; required for ``http`` mode.
        api_key: Bearer token for the server, when it has one.
        batch_size: Items per offline ``LLM.encode`` call, or per ``/pooling`` request.
        timeout_s / connect_timeout_s / max_retries: ``http`` mode's request policy (the client's defaults when
            ``None``).
        query_prefix / document_prefix: Role instructions, as the checkpoint's
            recipe defines them (ColPali-family models use ``"Query: "`` /
            ``"Passage: "``). Applied by role rather than by a flag, so the two
            sides cannot be configured inconsistently.

    The output is L2-normalised. The offline engine runs with ``dtype="auto"``,
    ``gpu_memory_utilization=0.85`` and the checkpoint's remote code trusted.
    """

    def __init__(
        self,
        *,
        model_name: str,
        pooling_task: str,
        mode: str = "offline",
        revision: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_prefix: str = "",
        document_prefix: str = "",
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        if mode not in ("offline", "http"):
            raise ValueError(f"unknown vllm mode {mode!r}; expected 'offline' or 'http'")
        if pooling_task not in POOLING_TASKS:
            raise ValueError(f"unknown pooling_task {pooling_task!r}; expected one of {POOLING_TASKS}")
        if mode == "http" and not api_base:
            raise ValueError("mode='http' needs api_base, e.g. api_base: http://localhost:8000")
        if mode == "http" and revision is not None:
            raise ValueError("mode='http' encodes with the server's weights; pin the revision where it is served")

        self.model_name = model_name
        self.mode = mode
        self.revision = revision
        self.pooling_task = pooling_task
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._http = {
            key: value
            for key, value in (
                ("timeout_s", timeout_s),
                ("connect_timeout_s", connect_timeout_s),
                ("max_retries", max_retries),
            )
            if value is not None
        }

        self.is_multi_vector = pooling_task == MULTI_VECTOR_TASK
        # vLLM's own multimodal registry is the authority on whether this
        # checkpoint takes images, but answering that needs the engine loaded.
        # Until then the encoder claims media support and lets vLLM reject it,
        # rather than pre-emptively refusing a corpus it could have handled.
        self.supports_media = True

        self._engine: Any = None
        self._client: Any = None
        self._processor: Any = None

    # -- encoding ----------------------------------------------------------
    def encode(
        self,
        contents: Sequence[Content],
        *,
        role: EncodeRole,
        batch_size: int | None = None,
    ) -> Embeddings:
        if not contents:
            return Embeddings.empty(0, multi_vector=self.is_multi_vector)

        prefixed = [self._with_role_prefix(content, role) for content in contents]
        if self.mode == "http":
            size = batch_size or self.batch_size
            client = self._client_for_http()
            chunks = [client.pooling(prefixed[i : i + size]) for i in range(0, len(prefixed), size)]
            embeddings = chunks[0]
            for chunk in chunks[1:]:
                embeddings = embeddings.concat(chunk)
        else:
            embeddings = self._encode_offline(prefixed, batch_size=batch_size or self.batch_size)

        if embeddings.num_items != len(contents):
            raise RuntimeError(
                f"vLLM returned {embeddings.num_items} embeddings for {len(contents)} inputs; "
                "refusing to return misaligned vectors"
            )
        return embeddings.l2_normalized()

    def _with_role_prefix(self, content: Content, role: EncodeRole) -> Content:
        prefix = self.query_prefix if role is EncodeRole.QUERY else self.document_prefix
        return content.with_text_prefix(prefix) if prefix else content

    def _encode_offline(self, contents: Sequence[Content], *, batch_size: int) -> Embeddings:
        engine = self._engine_for_offline()
        per_item: list[Any] = []
        for offset in range(0, len(contents), batch_size):
            batch = contents[offset : offset + batch_size]
            outputs = engine.encode(
                [self._offline_prompt(content) for content in batch],
                pooling_task=self.pooling_task,
                use_tqdm=False,
            )
            per_item.extend(output.outputs.data for output in outputs)
            if len(contents) > batch_size:
                logger.info(f"[vllm] encoded {min(offset + len(batch), len(contents)):,}/{len(contents):,}")

        import numpy as np

        arrays = [np.asarray(item.float().cpu().numpy(), dtype=np.float32) for item in per_item]
        if self.is_multi_vector:
            return Embeddings.ragged([array.reshape(-1, array.shape[-1]) for array in arrays])
        return Embeddings.single(np.stack([array.reshape(-1) for array in arrays]))

    def _offline_prompt(self, content: Content) -> Any:
        """One offline prompt: plain text, or chat-templated text plus images.

        The template comes from the checkpoint's own processor. Writing the
        ``<|vision_start|>``-style placeholders by hand is what the vLLM examples
        do, and it is exactly the kind of thing that keeps working after a
        checkpoint changes its template while producing different embeddings.
        """
        from vllm.inputs import TextPrompt

        if not content.has_media:
            return TextPrompt(prompt=content.text)

        from rcp_ndcg.data.media import default_resolver

        images = default_resolver().images_of(content)
        return TextPrompt(
            prompt=self._templated_prompt(content, num_images=len(images)),
            multi_modal_data={"image": images},
        )

    def _templated_prompt(self, content: Content, *, num_images: int) -> str:
        processor = self._hf_processor()
        parts: list[dict[str, Any]] = []
        for part in content.parts:
            if part.type == "text":
                if part.text:
                    parts.append({"type": "text", "text": part.text})
                continue
            parts.extend({"type": "image"} for _ in part.media_refs())
        messages = [{"role": "user", "content": parts}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not isinstance(prompt, str):
            raise RuntimeError(
                f"{self.model_name}'s chat template returned {type(prompt).__name__}, not a string; "
                "cannot build an offline prompt for it"
            )
        logger.debug(f"[vllm] templated prompt with {num_images} image(s)")
        return prompt

    # -- lazily-built collaborators ---------------------------------------
    def _engine_for_offline(self) -> Any:
        """The in-process engine, built once.

        Deferred to first use so that constructing the encoder does not claim a GPU.
        """
        if self._engine is not None:
            return self._engine
        try:
            from vllm import LLM
            from vllm.config import PoolerConfig
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise DependencyError(
                "an encoder with provider local and engine vllm runs vLLM in process, which is not installed",
                hint='pip install "rcp-ndcg[vllm]", or serve the model and use provider: openai_compatible',
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "revision": self.revision,
            "runner": "pooling",
            "pooler_config": PoolerConfig(task=self.pooling_task),
            "gpu_memory_utilization": 0.85,
            "dtype": "auto",
            "trust_remote_code": True,
        }

        logger.info(f"[vllm] loading {self.model_name} offline (task={self.pooling_task})")
        self._engine = LLM(**kwargs)
        return self._engine

    def _client_for_http(self) -> Any:
        if self._client is None:
            from rcp_ndcg.retrieval.vllm_http import VllmPoolingClient

            assert self.api_base is not None
            self._client = VllmPoolingClient(self.api_base, model=self.model_name, api_key=self.api_key, **self._http)
        return self._client

    def _hf_processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
            )
        return self._processor


__all__ = ["DEFAULT_BATCH_SIZE", "MULTI_VECTOR_TASK", "POOLING_TASKS", "SINGLE_VECTOR_TASK", "VllmEncoder"]
