"""Runtime bounds on the request generator: it stays linear on long documents (the BRIGHT stall).

An earlier generator ran for 40+ minutes at 100% CPU on long BRIGHT documents: it grew a padded text toward
its target length by re-tokenizing the whole candidate string at every step.  The bounds here count the
work the tokenizer does -- the characters every ``encode`` call reads, through the backend the product's
:class:`~rcp_ndcg.data.tokenizer.TextTokenizer` funnels every count, id and offset through -- so they are
deterministic (no wall clock) and fail on any growth loop: a quadratic pad reads the padded text once per
step, thousands of times its length.  The harness sampler's over-length padding
(``equivalence.stages._over_length``) has its own bound, in ``tests/test_equivalence.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_test.observe.requests import _pad_to_tokens, plan_recipe
from rcp_ndcg_test.observe.sources import SourceCorpus, SourceDoc, SourceQuery
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES

WORK_FACTOR = 12
"""The bound: characters tokenized per character the generator reads or writes.  The planned rows and the
source documents are each read a small, fixed number of times (counts, one offset pass, a galloping
search over token boundaries; measured: under 4x on the long-document plan below); a growth loop reads the
padded text once per step."""

PLAN_CALLS = 120
"""The bound on ``encode`` calls for the long-document plan below (measured: 51).  A growth loop calls once
per step: the earlier generator's one-pad-word steps made ~32k calls for one 32k-token row."""


class CountingBackend:
    """A ``tokenizers.Tokenizer`` stand-in that tallies the characters every ``encode`` reads."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.calls = 0
        self.chars = 0

    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate one encoding and record its work (characters read)."""
        self.calls += 1
        self.chars += len(text)
        return self._backend.encode(text, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _counting_tokenizer(recipe: Any) -> tuple[Any, CountingBackend]:
    """The recipe's real tokenizer with its backend wrapped in a :class:`CountingBackend`."""
    tokenizer = tokenizer_of(recipe)
    counter = CountingBackend(tokenizer.backend)
    return dataclasses.replace(tokenizer, backend=counter), counter


def _long_document(words: int) -> str:
    """A long word-heavy synthetic document (the shape of a long BRIGHT item): varied words, no structure."""
    vocabulary = ("alpha", "beta", "gamma", "delta", "epsilon", "theory", "proof", "lemma", "river", "city")
    return " ".join(vocabulary[(index * 7 + index // 3) % len(vocabulary)] for index in range(words))


def test_the_generator_plans_a_long_document_suite_in_linear_tokenizer_work() -> None:
    """``plan_recipe`` over a BRIGHT-like source with long documents and a 32768-token budget.

    The length ladder pads ``at_90`` and ``at_budget`` rows to ~29k and ~32k tokens, the median row to the
    long documents' median, and the source rows carry the long documents themselves: the tokenizer's total
    work must stay within :data:`WORK_FACTOR` times the characters the plan reads and writes.
    """
    base = load_recipe(RECIPES / "fixture-embed")
    recipe = base.model_copy(update={"client": {**base.client, "max_tokens": 32768}})
    tokenizer, counter = _counting_tokenizer(recipe)
    documents = {f"d{index}": SourceDoc(f"d{index}", _long_document(9000 + 500 * index)) for index in range(4)}
    corpus = SourceCorpus(
        suite="bright",
        subset="biology",
        commit="0" * 40,
        queries={
            "q1": SourceQuery("q1", "how do cells divide", None, ("d0", "d1")),
            "q2": SourceQuery("q2", "what limits enzyme speed", None, ("d2", "d3")),
        },
        docs=documents,
    )
    plan = plan_recipe(recipe, tokenizer, {"bright": [corpus]})
    assert plan.strata["length:at_budget"]["present"] is True, plan.strata["length:at_budget"]
    read = sum(len(doc.text) for doc in documents.values())
    written = sum(len(row.query) + sum(len(document) for document in row.documents) for row in plan.rows)
    assert written > 200_000, "the plan must carry the long rows the bound is about"
    assert counter.calls <= PLAN_CALLS, f"{counter.calls} encode calls: a growth loop counts once per step"
    assert counter.chars <= WORK_FACTOR * (read + written), (
        f"the generator tokenized {counter.chars} characters for {read + written} read and written "
        f"({counter.calls} encode calls): a growth loop re-tokenizes its candidate at every step"
    )


def test_the_exact_length_pad_reads_a_long_seed_a_bounded_number_of_times() -> None:
    """``_pad_to_tokens`` on a long seed toward a long target: bounded calls and linear characters."""
    tokenizer, counter = _counting_tokenizer(load_recipe(RECIPES / "fixture-embed"))
    seed = _long_document(4000)
    target = tokenizer.count(seed) + 20000
    counter.calls = counter.chars = 0
    text = _pad_to_tokens(tokenizer, seed, target)
    assert abs(tokenizer.count(text) - target) <= 2
    assert text.startswith(seed), "the pad never rewrites the seed"
    assert counter.calls <= 40, counter.calls
    # The pool is sized in pad WORDS for a gap in TOKENS (each pad word is several tokens), so one offset
    # pass reads a few times the padded text: linear, at a larger constant than the whole plan's.
    assert counter.chars <= 3 * WORK_FACTOR * len(text), (counter.chars, len(text))
