"""The preprocessing aggregation: what happens to text before a model reads it, re-exported from its homes.

The module was split (one home per concept); the names below keep their import path here, and nothing else
is re-exported (import every other name from its home):

* the judge's text policy, the chunking and the pass's effective policy:
  :mod:`rcp_ndcg.data.text_policy` (:class:`TextPolicy`, :class:`ChunkPolicy`,
  :func:`apply_text_policy`, :func:`chunk_ranking_example`, :func:`token_prefix`,
  :class:`Preprocessing`, :data:`DEFAULT_MAX_TOKENS`, :data:`DEFAULT_TEXT_POLICY`,
  :data:`CHUNK_ID_SEPARATOR`, :class:`DocumentOverCapError`, :class:`OnOverflow`,
  :func:`needs_tokenizer`, :func:`require_tokenizer`, :func:`split_into_chunks`);
* the cut record and the census: :mod:`rcp_ndcg.data.census` (:class:`TextCutRecord`,
  :class:`TextTruncationCensus`, :data:`CUT_CAUSES`, :class:`CutCause`);
* the served roles' text budget and the fit: :mod:`rcp_ndcg.data.text_budget` (:class:`TextBudget`,
  :func:`fit`, :class:`FitResult`, :class:`ProcessingRecord`, :func:`processing_records`,
  :data:`BUDGET_DOC_ID`, :data:`CHANGE_MECHANISMS`, :class:`ChangeMechanism`, :class:`ContentParts`,
  :class:`TextBudgetExceededError`, :func:`fixed_overhead`, :func:`rendered_pair_tokens`,
  :func:`rendered_request`);
* the chunk-score aggregation's names (:func:`document_id_for_chunk`, :func:`document_ids_from_chunks`,
  :func:`max_pool_scores_by_document`, :func:`max_pool_rubric_window_by_document`):
  :mod:`rcp_ndcg.data.postprocess` -- its other names (L2 normalisation, the late-interaction skip ids)
  and the Matryoshka head (:mod:`rcp_ndcg.data.mrl`) are NOT re-exported here;
* the census files' record I/O (append, read, the torn-tail repair): :mod:`rcp_ndcg.storage.census`
  (:func:`append_census_rows`, :func:`census_sink_lock`, :func:`drop_torn_last_line`,
  :func:`read_census_rows`).

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
