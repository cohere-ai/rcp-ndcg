"""RCP-nDCG core: the dependency-light distribution ``rcp-ndcg-core`` (numpy and pydantic).

It is the **stable surface** of the library:

* :mod:`rcp_ndcg_core.schemas` -- the public judgement records (``ItemParams``,
  ``QueryParams``, ``Judgement``, ``Placement``, ``JudgementFamily``, ``JudgementSet``,
  ``DocumentEstimate``).
* :mod:`rcp_ndcg_core.records` -- the pipeline records (``Document``, ``Query``,
  ``RankingExample``, ``Text``, ``Input``, ``ID``): what a reader yields, a writer
  takes, and retrieval, judging and the role clients pass around.
* :mod:`rcp_ndcg_core.metric` -- ``ndcg`` with float gains and the tie rules,
  ``dcg``.
* :mod:`rcp_ndcg_core.gain` -- the gains (``gain``, ``pass_probabilities``,
  ``count_gain``, ``qrel_gain``).
* :mod:`rcp_ndcg_core.protocol` -- the paper's scoring protocol (suite candidate
  rules, tie rules, ideals, macro averages).
* :mod:`rcp_ndcg_core.irt` -- IRT estimators and 2PL calibrators (torch on first use).
* the content parts (``Content``, ``TextPart``, ``ImagePart``, ``VideoPart``, ``MediaRef``).

Importing the package does not pull in ``torch``.
"""

from rcp_ndcg_core.content import (
    Content,
    ImagePart,
    MediaRef,
    Modality,
    Part,
    TextPart,
    VideoPart,
)
from rcp_ndcg_core.gain import Gains, count_gain, gain, pass_probabilities, qrel_gain
from rcp_ndcg_core.metric import TieRule, dcg, ndcg
from rcp_ndcg_core.protocol import PROTOCOLS, MetricName, Protocol, aggregate, score_query
from rcp_ndcg_core.records import ID, Document, Input, Query, RankingExample, Text
from rcp_ndcg_core.schemas import (
    ItemParams,
    Judgement,
    JudgementFamily,
    JudgementSet,
    Placement,
    QueryParams,
)

__all__ = [
    "JudgementFamily",
    "ItemParams",
    "Judgement",
    "JudgementSet",
    "Placement",
    "QueryParams",
    "PROTOCOLS",
    "Content",
    "Document",
    "ID",
    "ImagePart",
    "Input",
    "MediaRef",
    "Modality",
    "Part",
    "Protocol",
    "Query",
    "RankingExample",
    "Text",
    "TextPart",
    "TieRule",
    "VideoPart",
    "aggregate",
    "Gains",
    "MetricName",
    "count_gain",
    "dcg",
    "gain",
    "qrel_gain",
    "ndcg",
    "pass_probabilities",
    "score_query",
]
