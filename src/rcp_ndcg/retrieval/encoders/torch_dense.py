"""In-process HuggingFace weights: the reference backend (``provider: local`` with ``engine: hf``).

vLLM is faster and is what a corpus-scale index build should use, but it reaches the same embeddings through a
different attention kernel and batching regime, so its numbers can differ slightly from these.

Text-only by construction: pooling a hidden state at the last position says nothing about how to feed a page
image to a tower that may not have one.
"""

from __future__ import annotations

from collections.abc import Sequence

from rcp_ndcg_core.content import Content

from rcp_ndcg.retrieval.encoder import Embeddings, Encoder, EncodeRole

DEFAULT_BATCH_SIZE = 32


class TorchDenseEncoder(Encoder):
    """A HuggingFace ``AutoModel`` with last-token pooling and L2-normalised output, on one device.

    One replica on one device (:attr:`~rcp_ndcg.retrieval.accel.AccelState.device`), so the same object works
    under ``accelerate launch`` and in a plain process. Texts are truncated at
    :data:`~rcp_ndcg.retrieval.hf_dense.MAX_LENGTH` tokens.

    Args:
        model_name: The HuggingFace repo id or a local path.
        revision: The Hub revision to load; ``None`` is the repository's default branch.
        batch_size: Texts per forward pass.
        document_prefix: Prepended to documents only, as the model's recipe defines it (Octen: ``"- "``);
            ``None`` prepends nothing.
    """

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        document_prefix: str | None = None,
    ) -> None:
        from rcp_ndcg.retrieval.hf_dense import load_model_and_tokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.document_prefix = document_prefix
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(model_name, revision=revision)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Output width, from the model config, falling back to one forward pass."""
        if self._dim is None:
            hidden = getattr(getattr(self.model, "config", None), "hidden_size", None)
            self._dim = int(hidden) if hidden else int(self.encode_texts([""], role=EncodeRole.QUERY).dim)
        return self._dim

    def encode(
        self,
        contents: Sequence[Content],
        *,
        role: EncodeRole,
        batch_size: int | None = None,
    ) -> Embeddings:
        self.check_media(contents)
        texts = [content.text for content in contents]
        if role is EncodeRole.DOCUMENT and self.document_prefix:
            texts = [f"{self.document_prefix}{text}" for text in texts]
        if not texts:
            return Embeddings.empty(self.dim)

        from rcp_ndcg.retrieval.accel import AccelState
        from rcp_ndcg.retrieval.hf_dense import encode_text_batches

        vectors = encode_text_batches(
            self.model,
            self.tokenizer,
            texts,
            device=self.device,
            batch_size=batch_size or self.batch_size,
            show_progress=AccelState().is_main_process,
        )
        return Embeddings.single(vectors)


__all__ = ["DEFAULT_BATCH_SIZE", "TorchDenseEncoder"]
