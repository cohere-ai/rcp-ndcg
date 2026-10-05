"""Octen-Embedding-8B, in process: HuggingFace ``AutoModel`` weights, last-token pooling, L2, the ``"- "``
document prefix, and 8192-token right truncation: the paper's exact implementation.

Assembled unchanged in behaviour from the deleted ``src/rcp_ndcg/retrieval/hf_dense.py`` (loading,
tokenizing, batched encoding) and ``src/rcp_ndcg/retrieval/encoders/torch_dense.py`` (the encoder
with the role-dependent document prefix) when the package stopped carrying in-process models
(RFC-0001, option 1). The scoring is what the paper ran: bfloat16 weights, left padding so the
last position of every row is a real token, an 8192-token right truncation, and documents encoded
as ``"- " + text`` while queries are encoded as they are.

This module is not part of the package and must never be imported by it; see the README beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Every row scaled to unit L2 norm (a zero row stays zero), in float32, returned as float32.

    The deleted ``rcp_ndcg.inference.types.l2_normalize`` verbatim, so the vectors are bit for bit what the
    package's encoders computed."""
    source = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    return source / np.maximum(norms, 1e-12)


MAX_LENGTH = 8192
"""Tokens per text; longer texts are truncated on the right."""


def load_model_and_tokenizer(model_name: str, *, revision: str | None = None) -> tuple[Any, Any, str]:
    """Load a HuggingFace model and tokenizer onto the one device this process uses.

    Args:
        model_name: The HuggingFace repo id or a local path.
        revision: The Hub revision (commit, tag or branch) to load; ``None`` is the repository's default branch.

    Returns:
        ``(model, tokenizer, device)``, the model in eval mode on the device.

    Raises:
        ImportError: torch or transformers is not installed (see requirements.txt).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", revision=revision)
    model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16, revision=revision)
    model = model.to(device_str).eval()
    return model, tokenizer, device_str


def _encode_batch(model: Any, tokenizer: Any, texts: list[str], device: str) -> np.ndarray:
    """One batch of texts as L2-normalised last-token embeddings, ``(len(texts), D)`` float32."""
    import torch

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return _l2_normalize(outputs.last_hidden_state[:, -1, :].float().cpu().numpy())


def encode_text_batches(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode ``texts`` into L2-normalised ``(N, D)`` float32 embeddings with last-token pooling.

    The role-dependent document prefix is not applied here: that belongs to
    :class:`~rcp_ndcg.retrieval.encoders.torch_dense.TorchDenseEncoder`, which knows the role.

    Args:
        model: The HuggingFace model (already on ``device``).
        tokenizer: Its tokenizer (left padding, so the last position of every row is a real token).
        texts: The texts, already carrying any role prefix.
        device: The torch device string.
        batch_size: Texts per forward pass.
        show_progress: Whether to render a progress bar (typically on rank 0 only).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    iterator: range | tqdm[int] = range(0, len(texts), batch_size)
    if show_progress and len(texts) > batch_size:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Encoding")
    batches = [_encode_batch(model, tokenizer, texts[i : i + batch_size], device) for i in iterator]
    return np.vstack(batches).astype(np.float32, copy=False)


DEFAULT_BATCH_SIZE = 32


class TorchDenseEncoder:
    """A HuggingFace ``AutoModel`` with last-token pooling and L2-normalised output, on one device.

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
            self._dim = (
                int(hidden)
                if hidden
                else int(encode_text_batches(self.model, self.tokenizer, [""], device=self.device).shape[1])
            )
        return self._dim

    def encode(
        self,
        texts: list[str],
        *,
        document: bool = False,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode ``texts``: L2-normalised ``(N, D)`` float32 rows, the document prefix when ``document``."""
        if document and self.document_prefix:
            texts = [f"{self.document_prefix}{text}" for text in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return encode_text_batches(
            self.model,
            self.tokenizer,
            texts,
            device=self.device,
            batch_size=batch_size or self.batch_size,
            show_progress=show_progress,
        )


__all__ = ["MAX_LENGTH", "TorchDenseEncoder", "encode_text_batches", "load_model_and_tokenizer"]
