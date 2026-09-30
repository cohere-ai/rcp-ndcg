"""The offline judge, answering both stages from a hidden ability per document.

Public as ``rcp_ndcg.testing.FakeJudge``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from rcp_ndcg.llm.client import FAKE_URL_SCHEME, Completion, CompletionInput, JudgeClient, JudgeConfig
from rcp_ndcg.llm.cost import Price
from rcp_ndcg.llm.tokens import approx_tokens

#: Criterion difficulties (logits) of the fake rubric, C1 easiest to C5 hardest.
DEFAULT_DIFFICULTIES = (-1.5, -0.5, 0.0, 0.8, 1.8)

#: Slope of every criterion's response curve in the fake rubric.
DISCRIMINATION = 1.5

_DOC_BLOCK = re.compile(r'<doc id="doc_(\d+)">\n(.*?)\n</doc>', re.S)
_DOCUMENTS = re.compile(r"<documents>.*?</documents>", re.S)
_CRITERION = re.compile(r"\bC([1-9][0-9]?)\b")


def _uniform(*parts: object) -> float:
    """A deterministic draw in [0, 1) from *parts* (the same on every machine and in every call order)."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _hidden_ability(seed: int) -> Callable[[str], float]:
    """A standard-normal ability per document text, fixed by ``seed``."""

    def ability(text: str) -> float:
        u1, u2 = max(_uniform(seed, "ability", text, 1), 1e-12), _uniform(seed, "ability", text, 2)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return ability


class FakeJudge(JudgeClient):
    """A deterministic judge for offline runs, examples and tests.

    It reads the documents out of the real rendered prompt (the ``<doc id="doc_N">``
    blocks every judging prompt uses) and answers in the JSON the real parsers
    read, so everything downstream runs exactly as with a model:

    * a **rubric** prompt (one that asks for criteria ``C1..CK``): criterion ``k``
      passes when a uniform draw falls below
      ``sigmoid(1.5 * (ability - severity - difficulty_k))`` with the difficulties
      :data:`DEFAULT_DIFFICULTIES` (spread evenly when the prompt asks for another
      number of criteria) -- a 2PL world with known items;
    * a **tournament** prompt: each document scores ``ability + noise * (2u - 1)``,
      clipped to [-5, 5], and the ranking follows the scores.

    Every draw is a hash of ``(seed, window, document, criterion)``: the same
    window gets the same answer on every machine, and a document in two windows
    gets independent draws, like two placements of a real judge.

    Args:
        ability: The hidden ability (logits) of a document, from its text as shown in the
            prompt: a callable or a mapping. Default: a standard-normal draw per text, fixed by ``seed``.
        severity: Added to every difficulty; a lenient judge has a negative severity.
        noise: Half-width of the tournament score noise, in logits.
        seed: Changes every draw.
        name: The judge's model name, recorded in the judgement family (``"fake"`` by default).
        price: Token prices to bill the approximate tokens of each call at (budgets offline);
            ``None`` leaves the cost unknown, as for a real judge without a price.
    """

    def __init__(
        self,
        ability: Callable[[str], float] | Mapping[str, float] | None = None,
        *,
        severity: float = 0.0,
        noise: float = 0.5,
        seed: int = 0,
        name: str = "fake",
        price: Price | None = None,
    ) -> None:
        super().__init__(
            JudgeConfig(base_url=f"{FAKE_URL_SCHEME}seed/{seed}", model=name, temperature=None, price=price)
        )
        if ability is None:
            self.ability: Callable[[str], float] = _hidden_ability(seed)
        elif isinstance(ability, Mapping):
            self.ability = ability.__getitem__
        else:
            self.ability = ability
        self.severity = severity
        self.noise = noise
        self.seed = seed

    @classmethod
    def from_config(cls, config: JudgeConfig) -> FakeJudge:
        """The fake judge of ``JudgeConfig.fake(seed)``."""
        tail = config.urls[0].removeprefix(FAKE_URL_SCHEME)
        seed = int(tail.rsplit("/", 1)[-1]) if tail.rsplit("/", 1)[-1].lstrip("-").isdigit() else 0
        return cls(seed=seed, name=config.model, price=config.price)

    @staticmethod
    def _difficulties(num_criteria: int) -> tuple[float, ...]:
        if num_criteria == len(DEFAULT_DIFFICULTIES):
            return DEFAULT_DIFFICULTIES
        low, high = min(DEFAULT_DIFFICULTIES), max(DEFAULT_DIFFICULTIES)
        step = (high - low) / max(num_criteria - 1, 1)
        return tuple(low + step * k for k in range(num_criteria))

    async def _send(self, request: CompletionInput, replica: Any = None) -> Completion:
        prompt = request.user_prompt
        docs = {int(index): body for index, body in _DOC_BLOCK.findall(prompt)}
        if not docs:
            raise ValueError("FakeJudge found no <doc id=...> blocks in the prompt")
        window = "|".join(docs[index] for index in sorted(docs))
        labels = {int(k) for k in _CRITERION.findall(_DOCUMENTS.sub("", prompt))}
        if labels and labels == set(range(1, max(labels) + 1)):
            payload: dict = {
                f"doc_{index}": {
                    "criteria": {
                        f"C{k + 1}": int(
                            _uniform(self.seed, "rubric", window, body, k)
                            < 1.0 / (1.0 + math.exp(-DISCRIMINATION * (self.ability(body) - self.severity - b)))
                        )
                        for k, b in enumerate(self._difficulties(max(labels)))
                    }
                }
                for index, body in docs.items()
            }
        else:
            scores = {
                str(index): round(
                    max(-5.0, min(5.0, self.ability(body) + self.noise * (2 * _uniform(self.seed, window, body) - 1))),
                    4,
                )
                for index, body in docs.items()
            }
            ranking = sorted(scores, key=lambda index: (-scores[index], int(index)))
            payload = {"ranking": [int(index) for index in ranking], "scores": scores}
        response = json.dumps(payload)
        input_tokens, output_tokens = approx_tokens(len(prompt)), approx_tokens(len(response))
        self._bill(input_tokens, output_tokens)
        return Completion(
            response=response, finish_reason="stop", input_tokens=input_tokens, output_tokens=output_tokens
        )


__all__ = ["DEFAULT_DIFFICULTIES", "FakeJudge"]
