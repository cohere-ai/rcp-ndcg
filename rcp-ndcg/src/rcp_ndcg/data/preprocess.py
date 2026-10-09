"""The preprocessing aggregation: what happens to text before a model reads it, re-exported from its homes.

The module was split (one home per concept); every public name keeps its import path here:

* the judge's text policy, the chunking and the pass's effective policy:
  :mod:`rcp_ndcg.data.text_policy` (:class:`TextPolicy`, :class:`ChunkPolicy`,
  :func:`apply_text_policy`, :func:`chunk_ranking_example`, :func:`token_prefix`,
  :class:`Preprocessing`);
* the cut record and the census: :mod:`rcp_ndcg.data.census` (:class:`TextCutRecord`,
  :class:`TextTruncationCensus`);
* the served roles' text budget and the fit: :mod:`rcp_ndcg.data.text_budget` (:class:`TextBudget`,
  :func:`fit`, :class:`FitResult`, :class:`ProcessingRecord`);
* the postprocess of model output (L2 normalisation, the chunk-score aggregation, the Matryoshka cut,
  the late-interaction skip ids): :mod:`rcp_ndcg.data.postprocess`;
* the census files' record I/O (append, read, the torn-tail repair): :mod:`rcp_ndcg.storage.census`.

Nothing is defined here: import from the home module in new code, or from this module for the
documented public path (both work, and the contract snapshot pins this module's surface).
"""

from rcp_ndcg.data.census import (
    CUT_CAUSES,
    CutCause,
    TextCutRecord,
    TextTruncationCensus,
)
from rcp_ndcg.data.postprocess import (
    document_id_for_chunk,
    document_ids_from_chunks,
    max_pool_rubric_window_by_document,
    max_pool_scores_by_document,
)
from rcp_ndcg.data.text_budget import (
    BUDGET_DOC_ID,
    CHANGE_MECHANISMS,
    ChangeMechanism,
    ContentParts,
    FitResult,
    ProcessingRecord,
    TextBudget,
    TextBudgetExceededError,
    fit,
    fixed_overhead,
    processing_records,
    rendered_pair_tokens,
    rendered_request,
)
from rcp_ndcg.data.text_policy import (
    CHUNK_ID_SEPARATOR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEXT_POLICY,
    ChunkPolicy,
    DocumentOverCapError,
    OnOverflow,
    Preprocessing,
    TextPolicy,
    apply_text_policy,
    chunk_ranking_example,
    needs_tokenizer,
    require_tokenizer,
    split_into_chunks,
    token_prefix,
)
from rcp_ndcg.storage.census import (
    append_census_rows,
    census_sink_lock,
    drop_torn_last_line,
    read_census_rows,
)

__all__ = [
    "BUDGET_DOC_ID",
    "fixed_overhead",
    "rendered_pair_tokens",
    "rendered_request",
    "CHANGE_MECHANISMS",
    "CHUNK_ID_SEPARATOR",
    "CUT_CAUSES",
    "ChangeMechanism",
    "ChunkPolicy",
    "CutCause",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEXT_POLICY",
    "DocumentOverCapError",
    "FitResult",
    "Preprocessing",
    "TextBudget",
    "TextBudgetExceededError",
    "TextCutRecord",
    "ContentParts",
    "OnOverflow",
    "ProcessingRecord",
    "TextTruncationCensus",
    "TextPolicy",
    "apply_text_policy",
    "chunk_ranking_example",
    "append_census_rows",
    "census_sink_lock",
    "document_id_for_chunk",
    "document_ids_from_chunks",
    "drop_torn_last_line",
    "fit",
    "max_pool_rubric_window_by_document",
    "max_pool_scores_by_document",
    "needs_tokenizer",
    "processing_records",
    "read_census_rows",
    "require_tokenizer",
    "split_into_chunks",
    "token_prefix",
]
