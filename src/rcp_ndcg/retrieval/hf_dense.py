"""Encoding text with a HuggingFace transformer embedding model: loading it, tokenizing, and batched encoding.

The torch dense encoder (:mod:`rcp_ndcg.retrieval.encoders.torch_dense`) encodes through these helpers, e.g. for
Octen/Octen-Embedding-8B (Qwen3-8B-based, 4096-dim, last-token pooling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from rcp_ndcg.errors import DependencyError
from rcp_ndcg.retrieval.accel import AccelState
from rcp_ndcg.retrieval.encoder import l2_normalize
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from torch import device as TorchDevice

logger = get_logger(__name__)

MAX_LENGTH = 8192
"""Tokens per text; longer texts are truncated on the right."""


def load_model_and_tokenizer(
    model_name: str,
    device: str | TorchDevice | None = None,
    *,
    revision: str | None = None,
) -> tuple[Any, Any, str]:
    """Load a HuggingFace model and tokenizer onto the rank's device.

    Args:
        model_name: The HuggingFace repo id or a local path.
        device: The torch device; defaults to :attr:`AccelState.device`, so that under ``accelerate launch`` each
            rank lands on its assigned GPU.
        revision: The Hub revision (commit, tag or branch) to load; ``None`` is the repository's default branch.

    Returns:
        ``(model, tokenizer, device)``, the model in eval mode on ``device``.

    Raises:
        DependencyError: torch or transformers is not installed (the ``[local]`` extra).
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "the hf encoder needs torch and transformers, which are not installed",
            hint='pip install "rcp-ndcg[local]"',
        ) from exc

    device_str = str(AccelState().device if device is None else device)
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
    return l2_normalize(outputs.last_hidden_state[:, -1, :].float().cpu().numpy())


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


__all__ = ["MAX_LENGTH", "encode_text_batches", "load_model_and_tokenizer"]
