"""The engine-side keep-rule: the declared rule read from the engine config and applied to the token ids the
pooler sees.

The rule's home is the recipe. A late-interaction checkpoint's own mask keeps a subset of a document's
positions (``document_skip_token_ids`` on the client: pplx-embed-v2-late's 32 ASCII punctuation ids, resolved
from the checkpoint's ``MultiVectorMask``; ``media_keep_token_ids``: topk-embed-v1's image-patch token, the
only positions its reference keeps for an image document); the recipe declares each rule once and renders it
for the engine in ``serve.hf_overrides.document_skip_token_ids`` / ``document_keep_token_ids`` (the loader
cross-checks the two halves). vLLM v0.31.0's pooling route cannot return the engine's per-position token ids,
so the rules are applied HERE, engine-side, by the plugin's own pooler
(:mod:`rcp_ndcg_vllm.models.keep_pooler`): the per-token vectors at the excluded positions are dropped before
the reply, and the wire carries only kept vectors -- for a text document and for a media document's
chat-template render alike (the plugin sees the render's own ids, the vision markers and the trained head
included). The client cannot recompute the kept set from the reply, so it checks the reply's per-item count
against the declared kept count instead (``PoolingEndpoint.document_skip_engine_side`` /
``PoolingEndpoint.media_keep_token_ids``).

Two gates select the rows a rule applies to, both declared by the recipe's engine half:

* the DOCUMENT role prefix (``document_skip_prefix_token_id``, pplx-embed-v2-late: the ``[D] `` added token):
  the checkpoint's mask is document-side (``skiplist_tasks: ["document"]``), so a row that does not open with
  it is a query prompt and keeps every position;
* the MEDIA allowlist itself (``document_keep_token_ids``): a row that carries one of the allowlist's ids is a
  media document (topk-embed-v1's image render carries the image-patch token, which is not the text render's
  leading token), and only the allowlist's positions are kept -- the chat template's structural tokens and a
  caption are dropped, which is exactly what that checkpoint's reference keeps.

This module imports no vLLM and no torch: the rules' semantics are testable with fake token-id rows, and the
pooler that calls them stays a thin adapter over :func:`kept_positions`. It duplicates no product logic: the
engine image installs this wheel with ``--no-deps`` and ``rcp_ndcg`` is absent there, so the mask rules have
one home in the engine (here) and one in the client (:func:`rcp_ndcg.data.postprocess.skip_keep_mask`,
:func:`rcp_ndcg.data.postprocess.kept_vector_count`), both fed by the recipe's one declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "declared_keep_ids",
    "declared_skip_ids",
    "declared_skip_prefix_id",
    "kept_positions",
]

#: The HF-config key the recipe's engine half of the TEXT skip rule is declared under
#: (``serve.hf_overrides.document_skip_token_ids``; the same name as the client's field, so the loader's
#: cross-check compares like with like).
SKIP_IDS_KEY = "document_skip_token_ids"

#: The HF-config key of the TEXT rule's role gate: the leading token id a DOCUMENT prompt opens with
#: (``serve.hf_overrides.document_skip_prefix_token_id``; pplx-embed-v2-late: the ``[D] `` added token,
#: 248078, which every text document render and the media head open with). The checkpoint's mask is
#: document-side (its ``skiplist_tasks: ["document"]``), so the rule must not touch a query prompt: a row
#: that does not open with this id keeps every position.
PREFIX_ID_KEY = "document_skip_prefix_token_id"

#: The HF-config key the recipe's engine half of the MEDIA allowlist is declared under
#: (``serve.hf_overrides.document_keep_token_ids``; the same name as the client's field).
KEEP_IDS_KEY = "document_keep_token_ids"


def declared_skip_prefix_id(model_config: Any) -> int | None:
    """The document role prefix the served recipe declared for the engine, from the model's HF config.

    Args:
        model_config: vLLM's ``ModelConfig`` (only ``hf_config`` is read): the recipe's
            ``serve.hf_overrides.document_skip_prefix_token_id`` is merged into the HF config, so the gate
            travels with the engine's own startup config.

    Returns:
        The declared id, or ``None`` when the recipe declares none.

    Raises:
        ValueError: the declared value is not a token id (the gate would otherwise never match, silently
            applying the rule to queries too).
    """
    value = getattr(getattr(model_config, "hf_config", None), PREFIX_ID_KEY, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"the served model's hf_overrides declares {PREFIX_ID_KEY} as {type(value).__name__} ({value!r}), "
            "not a token id; the engine-side keep-rule's document gate cannot be applied"
        )
    return value


def declared_skip_ids(model_config: Any) -> tuple[int, ...]:
    """The TEXT document skip rule the served recipe declared for the engine, from the model's HF config.

    Args:
        model_config: vLLM's ``ModelConfig`` (only ``hf_config`` is read): the recipe's
            ``serve.hf_overrides.document_skip_token_ids`` is merged into the HF config, so the rule travels
            with the engine's own startup config.

    Returns:
        The declared token ids, in declaration order; ``()`` when the recipe declares none (the engine then
        returns every prompt token's vector, as before).

    Raises:
        ValueError: the declared value is not a sequence of ints (a malformed ``hf_overrides`` entry would
            otherwise drop arbitrary positions or fail obscurely mid-forward).
    """
    return _declared_ids(model_config, SKIP_IDS_KEY)


def declared_keep_ids(model_config: Any) -> tuple[int, ...]:
    """The MEDIA allowlist the served recipe declared for the engine, from the model's HF config.

    Args:
        model_config: vLLM's ``ModelConfig`` (only ``hf_config`` is read): the recipe's
            ``serve.hf_overrides.document_keep_token_ids`` is merged into the HF config.

    Returns:
        The declared token ids, in declaration order; ``()`` when the recipe declares none (a media render is
        then treated like any other row).

    Raises:
        ValueError: the declared value is not a sequence of ints (a malformed ``hf_overrides`` entry would
            otherwise keep arbitrary positions or fail obscurely mid-forward).
    """
    return _declared_ids(model_config, KEEP_IDS_KEY)


def _declared_ids(model_config: Any, key: str) -> tuple[int, ...]:
    """One declared token-id list from the model's HF config, with the malformed cases refused by name."""
    value = getattr(getattr(model_config, "hf_config", None), key, None)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"the served model's hf_overrides declares {key} as {type(value).__name__} ({value!r}), not a "
            "list of token ids; the engine-side keep-rule cannot be applied"
        )
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"the served model's hf_overrides declares {key} with a non-integer entry ({item!r}); the "
                "engine-side keep-rule is a list of token ids"
            )
        ids.append(item)
    return tuple(ids)


def kept_positions(
    token_ids: Sequence[int],
    skip_ids: Sequence[int],
    *,
    document_prefix_id: int | None = None,
    keep_ids: Sequence[int] = (),
) -> list[int]:
    """The positions of ``token_ids`` the declared rule keeps, ascending -- the vectors the wire carries.

    The rules are by token id alone. The MEDIA allowlist comes first: with ``keep_ids`` given, a row that
    carries one of the allowlist's ids is a media document -- those ids ARE the render's media positions
    (topk-embed-v1's image-patch token) -- and only the allowlist's positions are kept; the row's other
    positions (the chat template's structural tokens, a caption) are dropped, which is exactly what that
    checkpoint's reference keeps. A row without an allowlist id is a text document or a query, and the TEXT
    skip rule applies: every position whose id is in ``skip_ids`` is dropped, however often the id recurs,
    and everything else is kept -- so a rule that names a structural id drops exactly that position. A row
    fully inside a rule keeps nothing (the reply carries no vector for it, which the client's declared count
    then expects).

    The text rule is DOCUMENT-side (the checkpoint's own mask declares ``skiplist_tasks: ["document"]``), so
    with ``document_prefix_id`` given a row that does not open with it is not a document prompt: a query keeps
    every position (its vectors are the checkpoint's own query output, which the reference's ``encode_query``
    keeps whole). Without a declared prefix the text rule has no role signal and applies to every non-media
    row -- which is why the recipe's engine half must declare the prefix whenever it declares the skip rule
    (the loader refuses the rule alone, and
    :func:`~rcp_ndcg_vllm.models.keep_pooler.build_keep_pooler` refuses it too).

    Args:
        token_ids: One sequence's prompt token ids, as the engine rendered it (the pooler's
            ``PoolingMetadata.get_prompt_token_ids_cpu()`` row).
        skip_ids: The ids the declared text rule excludes.
        document_prefix_id: The leading token id a document prompt opens with (``None``: no gate).
        keep_ids: The media allowlist's ids (``()``: no allowlist; a row carrying one is a media document
            and keeps only its allowlist positions).

    Returns:
        The kept positions, ascending (``[]`` when a rule excludes every position).
    """
    keep = frozenset(keep_ids)
    if keep and any(token_id in keep for token_id in token_ids):
        return [position for position, token_id in enumerate(token_ids) if token_id in keep]
    if document_prefix_id is not None and (not token_ids or token_ids[0] != document_prefix_id):
        return list(range(len(token_ids)))
    skip = frozenset(skip_ids)
    return [position for position, token_id in enumerate(token_ids) if token_id not in skip]
