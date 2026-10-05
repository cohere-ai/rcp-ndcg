"""The offline judge, answering both stages from a hidden ability per document.

Public as ``rcp_ndcg.testing.FakeJudge``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from rcp_ndcg_core.gain import sigmoid

from rcp_ndcg.inference.fake import fake_uniform as _uniform
from rcp_ndcg.inference.fake import hidden_ability as _hidden_ability
from rcp_ndcg.llm.client import FAKE_URL_SCHEME, Completion, CompletionInput, JudgeClient, JudgeConfig
from rcp_ndcg.llm.tokens import approx_tokens

#: Criterion difficulties (logits) of the fake rubric, C1 easiest to C5 hardest.
DEFAULT_DIFFICULTIES = (-1.5, -0.5, 0.0, 0.8, 1.8)

#: Slope of every criterion's response curve in the fake rubric.
DISCRIMINATION = 1.5

#: Half-width of the fake tournament's score noise, in logits.
TOURNAMENT_NOISE = 0.5

_DOC_BLOCK = re.compile(r'<doc id="doc_(\d+)">\n(.*?)\n</doc>', re.S)
_DOCUMENTS = re.compile(r"<documents>.*?</documents>", re.S)
_CRITERION = re.compile(r"\bC([1-9][0-9]?)\b")


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
    * a **tournament** prompt: each document scores ``ability + 0.5 * (2u - 1)``,
      clipped to [-5, 5], and the ranking follows the scores.

    Every draw is a hash of ``(seed, window, document, criterion)``: the same
    window gets the same answer on every machine, and a document in two windows
    gets independent draws, like two placements of a real judge.

    Args:
        ability: The hidden ability (logits) of a document, from its text as shown in the
            prompt: a callable or a mapping. Default: a standard-normal draw per text, fixed by ``seed``.
        severity: Added to every difficulty; a lenient judge has a negative severity.
        seed: Changes every draw.
        name: The judge's model name, recorded in the judgement family (``"fake"`` by default).
        config: The judge config to run under (a ``fake://`` one, as :meth:`from_config` passes): every field of
            it applies, as for a real judge (its tokenizer, context, media limits, decoding and concurrency);
            ``name`` is then its ``model``. ``None`` builds the default offline config for ``seed`` and ``name``.
    """

    def __init__(
        self,
        ability: Callable[[str], float] | Mapping[str, float] | None = None,
        *,
        severity: float = 0.0,
        seed: int = 0,
        name: str = "fake",
        config: JudgeConfig | None = None,
    ) -> None:
        if config is not None and not config.is_fake:
            raise ValueError(f"the fake judge runs under a fake:// config, not {config.urls[0]!r}")
        super().__init__(config or JudgeConfig(base_url=f"{FAKE_URL_SCHEME}seed/{seed}", model=name, temperature=None))
        if ability is None:
            self.ability: Callable[[str], float] = _hidden_ability(seed)
        elif isinstance(ability, Mapping):
            self.ability = ability.__getitem__
        else:
            self.ability = ability
        self.severity = severity
        self.seed = seed

    @classmethod
    def from_config(cls, config: JudgeConfig) -> FakeJudge:
        """The fake judge of a ``fake://`` config (``JudgeConfig.fake(seed)`` with any fields set on it).

        The seed is the URL's; every other field of ``config`` applies, so a pass on the fake judge records the
        model, tokenizer, context, media limits and decoding it was given.
        """
        tail = config.urls[0].removeprefix(FAKE_URL_SCHEME)
        seed = int(tail.rsplit("/", 1)[-1]) if tail.rsplit("/", 1)[-1].lstrip("-").isdigit() else 0
        return cls(seed=seed, config=config)

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
                            < sigmoid(DISCRIMINATION * (self.ability(body) - self.severity - b))
                        )
                        for k, b in enumerate(self._difficulties(max(labels)))
                    }
                }
                for index, body in docs.items()
            }
        else:
            scores = {
                str(index): round(
                    max(
                        -5.0,
                        min(5.0, self.ability(body) + TOURNAMENT_NOISE * (2 * _uniform(self.seed, window, body) - 1)),
                    ),
                    4,
                )
                for index, body in docs.items()
            }
            ranking = sorted(scores, key=lambda index: (-scores[index], int(index)))
            payload = {"ranking": [int(index) for index in ranking], "scores": scores}
        response = json.dumps(payload)
        input_tokens, output_tokens = approx_tokens(len(prompt)), approx_tokens(len(response))
        self._count_usage(input_tokens, output_tokens)
        return Completion(
            response=response, finish_reason="stop", input_tokens=input_tokens, output_tokens=output_tokens
        )


__all__ = ["DEFAULT_DIFFICULTIES", "FakeJudge"]
