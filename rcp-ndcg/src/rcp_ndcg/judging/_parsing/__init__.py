"""Parsers of the judge's answers.

* :mod:`rcp_ndcg.judging._parsing.common` -- the one decoder both parsers use (the judge's JSON object, never
  digits from prose), the error it raises and :data:`~rcp_ndcg.judging._parsing.common.PARSE_VERSION`;
* :mod:`rcp_ndcg.judging._parsing.listwise` -- the tournament answer (ranking and scores) and its comparisons;
* :mod:`rcp_ndcg.judging._parsing.rubric` -- the rubric answer (criteria C1..Cn per document);
* :mod:`rcp_ndcg.judging._parsing.schema` -- the JSON schema of each stage's answer, for structured output.

Document identifiers in prompts are 1-based (``doc_1``, ``doc_2``, ...). A parser raises
:class:`~rcp_ndcg.judging._parsing.common.UnparseableAnswer` with an
:data:`~rcp_ndcg_core.schemas.InvalidCategory` when an answer is not a complete observation of its window.
"""
