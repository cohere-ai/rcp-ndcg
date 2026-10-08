"""Reference cases and one conformance suite for rcp-ndcg recipes.

Every case runs through the **product's** role clients
(:class:`rcp_ndcg.inference.clients.EmbeddingClient`, ``PoolingClient``, ``RerankClient``) built from the
recipe's ``client`` block -- never raw HTTP, never a copy of the client. Two targets: a live engine
(``base_url`` of a serving vLLM) or a recipe-level fake engine (:mod:`rcp_ndcg_test.fakes`).

* :mod:`rcp_ndcg_test.cases` -- the case format and its validation (:func:`load_cases`).
* :mod:`rcp_ndcg_test.conformance` -- the runner and its typed report (:func:`run_suite`).
* :mod:`rcp_ndcg_test.fakes` -- the fake-engine seam and the registry by recipe id.
* :mod:`rcp_ndcg_test.plugin` -- pytest integration (opt-in; parametrised tests).

The package is **unpublished** (never on PyPI): the repository's CI, the product's pytest suite and the
GPU waves install it from the workspace or the staged wheelhouse. No published package names it.
"""

from __future__ import annotations

from .errors import CaseError, ConformanceError

__version__ = "0.0.1"

__all__ = ["CaseError", "ConformanceError", "__version__"]
