"""The observation request generator and corpus records (OBSERVATIONS-SPEC).

The deterministic request set, the recorded exchanges, the manifest and the change handling behind the
GPU waves' observations and the verified fake engines.  See
:mod:`rcp_ndcg_vllm.observe.requests` for the generator and ``schema/observation-corpus.md`` for the
corpus schema.

Public surface:

- :data:`~rcp_ndcg_vllm.observe.requests.GENERATOR_VERSION`,
  :data:`~rcp_ndcg_vllm.observe.requests.SEED`,
  :data:`~rcp_ndcg_vllm.observe.requests.PINNED_DATASET_COMMITS`.
- :func:`~rcp_ndcg_vllm.observe.requests.plan_recipe`,
  :func:`~rcp_ndcg_vllm.observe.requests.pairs_jsonl`,
  :func:`~rcp_ndcg_vllm.observe.requests.write_pairs_file`,
  :func:`~rcp_ndcg_vllm.observe.requests.write_manifest`.
- :func:`~rcp_ndcg_vllm.observe.corpus.build_corpus`, :func:`~rcp_ndcg_vllm.observe.corpus.verify_corpus`
  (the observation-corpus format: ``RECORD_SCHEMA`` records and the hash-chained manifest), and
  :func:`~rcp_ndcg_vllm.observe.corpus.behaviour_fingerprint` (GPU-VALIDATION.md item 8).
"""

from __future__ import annotations

from typing import Any

from .adversarial import CONTENT_KINDS, SYNTHETIC_TEXTS, special_token_spellings, synthetic_text

_REQUESTS_EXPORTS = frozenset(
    {
        "GENERATOR_VERSION",
        "PINNED_DATASET_COMMITS",
        "SEED",
        "PlannedRow",
        "RecipePlan",
        "pairs_jsonl",
        "plan_recipe",
        "write_manifest",
        "write_pairs_file",
    }
)


def __getattr__(name: str) -> Any:
    """Re-export the generator's names lazily.

    ``python -m rcp_ndcg_vllm.observe.requests`` runs the module as ``__main__``; an eager import
    here would put it in ``sys.modules`` first and make runpy execute it twice (its warning names
    exactly that).  The names resolve identically at attribute access.
    """
    if name in _REQUESTS_EXPORTS:
        from . import requests

        return getattr(requests, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REQUESTS_EXPORTS))


__all__ = [
    "CONTENT_KINDS",
    "GENERATOR_VERSION",
    "PINNED_DATASET_COMMITS",
    "SEED",
    "SYNTHETIC_TEXTS",
    "PlannedRow",
    "RecipePlan",
    "pairs_jsonl",
    "plan_recipe",
    "special_token_spellings",
    "synthetic_text",
    "write_manifest",
    "write_pairs_file",
]
