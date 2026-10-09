"""The pplx-late engine-side keep-rule: the declared skip list, read from the engine config and applied to
the token ids the pooler sees.

The rule's home is the recipe. A late-interaction checkpoint's own mask excludes a set of token ids from a
document's scored positions (``document_skip_token_ids`` on the client: pplx-embed-v2-late's 32 ASCII
punctuation ids, resolved from the checkpoint's ``MultiVectorMask``); the recipe declares it once and
renders it for the engine in ``serve.hf_overrides.document_skip_token_ids`` (the loader cross-checks the two
halves). vLLM v0.31.0's pooling route cannot return the engine's per-position token ids, so the rule is
applied HERE, engine-side, by the plugin's own pooler
(:mod:`rcp_ndcg_vllm.models.pplx.late_pooler`): the per-token vectors at the excluded positions are dropped
before the reply, and the wire carries only kept vectors -- for a text document and for a media document's
chat-template render alike (the plugin sees the render's own ids, the vision markers and the trained head
included). The checkpoint's mask is DOCUMENT-side (``skiplist_tasks: ["document"]``), so the engine half also
declares the document role prefix (``document_skip_prefix_token_id``, the ``[D] `` added token): a row that
does not open with it is a query prompt and keeps every position. The client cannot recompute the kept set
from the reply, so it checks the reply's per-item count against the declared kept count instead
(``PoolingEndpoint.document_skip_engine_side``).

This module imports no vLLM and no torch: the rule's semantics are testable with fake token-id rows, and
the pooler that calls them stays a thin adapter over :func:`kept_positions`. It duplicates no product logic:
the engine image installs this wheel with ``--no-deps`` and ``rcp_ndcg`` is absent there, so the mask rule
has one home in the engine (here) and one in the client (:func:`rcp_ndcg.data.postprocess.skip_keep_mask`),
both fed by the recipe's one declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["declared_skip_ids", "declared_skip_prefix_id", "kept_positions"]

#: The HF-config key the recipe's engine half is declared under (``serve.hf_overrides.document_skip_token_ids``;
#: the same name as the client's field, so the loader's cross-check compares like with like).
SKIP_IDS_KEY = "document_skip_token_ids"

#: The HF-config key of the role gate: the leading token id a DOCUMENT prompt opens with
#: (``serve.hf_overrides.document_skip_prefix_token_id``; pplx-embed-v2-late: the ``[D] `` added token,
#: 248078, which every text document render and the media head open with). The checkpoint's mask is
#: document-side (its ``skiplist_tasks: ["document"]``), so the rule must not touch a query prompt: a row
#: that does not open with this id keeps every position.
PREFIX_ID_KEY = "document_skip_prefix_token_id"


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
    """The document skip rule the served recipe declared for the engine, from the model's HF config.

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
    value = getattr(getattr(model_config, "hf_config", None), SKIP_IDS_KEY, None)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"the served model's hf_overrides declares {SKIP_IDS_KEY} as {type(value).__name__} "
            f"({value!r}), not a list of token ids; the engine-side keep-rule cannot be applied"
        )
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"the served model's hf_overrides declares {SKIP_IDS_KEY} with a non-integer entry "
                f"({item!r}); the engine-side keep-rule is a list of token ids"
            )
        ids.append(item)
    return tuple(ids)


def kept_positions(
    token_ids: Sequence[int], skip_ids: Sequence[int], *, document_prefix_id: int | None = None
) -> list[int]:
    """The positions of ``token_ids`` the declared rule keeps, ascending -- the vectors the wire carries.

    The rule is by token id alone: every position whose id is in ``skip_ids`` is dropped, however often the
    id recurs, and everything else is kept -- a text document's punctuation positions and a media render's
    structural positions (the trained head, the vision markers) are treated alike, so a rule that names a
    structural id drops exactly that position. A row fully inside the rule keeps nothing (the reply carries
    no vector for it, which the client's declared count then expects).

    The rule is DOCUMENT-side (the checkpoint's own mask declares ``skiplist_tasks: ["document"]``), so with
    ``document_prefix_id`` given a row that does not open with it is not a document prompt: a query keeps
    every position (its vectors are the checkpoint's own query output, which the reference's ``encode_query``
    keeps whole). Without a declared prefix the rule has no role signal and applies to every row -- which is
    why the recipe's engine half must declare the prefix whenever it declares the rule (the loader refuses
    the rule alone, and :func:`~rcp_ndcg_vllm.models.pplx.late_pooler.build_late_pooler` refuses it too).

    Args:
        token_ids: One sequence's prompt token ids, as the engine rendered it (the pooler's
            ``PoolingMetadata.get_prompt_token_ids_cpu()`` row).
        skip_ids: The ids the declared rule excludes.
        document_prefix_id: The leading token id a document prompt opens with (``None``: no gate).

    Returns:
        The kept positions, ascending (``[]`` when the rule excludes every position).
    """
    if document_prefix_id is not None and (not token_ids or token_ids[0] != document_prefix_id):
        return list(range(len(token_ids)))
    skip = frozenset(skip_ids)
    return [position for position, token_id in enumerate(token_ids) if token_id not in skip]
