"""The offline judge: the ``fake://`` chat-completions route, and :class:`FakeJudge` for tests that map abilities
themselves.

The route (:func:`register_fake_route`, registered at import of this module) answers
``POST /chat/completions`` on a ``fake://`` endpoint the way a served model would:
it reads the documents out of the real rendered prompt (the ``<doc id="doc_N">``
blocks every judging prompt uses) and answers in the JSON the real parsers read,
so ``JudgeConfig.fake(seed)`` runs a real :class:`~rcp_ndcg.judging.JudgeClient` over
the real transport with no endpoint. Public as ``rcp_ndcg.testing.FakeJudge``;
the route is what that stands in for.

Both the route and :class:`FakeJudge` answer from one hidden ability per document:

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
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from rcp_ndcg_core.gain import sigmoid

from rcp_ndcg.inference.fake import (
    fake_uniform as _uniform,
)
from rcp_ndcg.inference.fake import (
    hidden_ability as _hidden_ability,
)
from rcp_ndcg.inference.fake import (
    register_fake_route,
)
from rcp_ndcg.judging.client import FAKE_URL_SCHEME, JudgeClient, JudgeConfig
from rcp_ndcg.judging.prompts import criterion_labels_in
from rcp_ndcg.judging.tokens import approx_tokens

#: Criterion difficulties (logits) of the fake rubric, C1 easiest to C5 hardest.
DEFAULT_DIFFICULTIES = (-1.5, -0.5, 0.0, 0.8, 1.8)

#: Slope of every criterion's response curve in the fake rubric.
DISCRIMINATION = 1.5

#: Half-width of the fake tournament's score noise, in logits.
TOURNAMENT_NOISE = 0.5

_DOC_BLOCK = re.compile(r'<doc id="doc_(\d+)">\n(.*?)\n</doc>', re.S)
_DOCUMENTS = re.compile(r"<documents>.*?</documents>", re.S)


def _difficulties(num_criteria: int) -> tuple[float, ...]:
    """The rubric's criterion difficulties, spread evenly for a prompt asking for another number."""
    if num_criteria == len(DEFAULT_DIFFICULTIES):
        return DEFAULT_DIFFICULTIES
    low, high = min(DEFAULT_DIFFICULTIES), max(DEFAULT_DIFFICULTIES)
    step = (high - low) / max(num_criteria - 1, 1)
    return tuple(low + step * k for k in range(num_criteria))


def _answer_text(seed: int, ability: Callable[[str], float], severity: float, prompt: str) -> str:
    """The fake judge's answer for one rendered prompt: rubric verdicts or a tournament ranking, as JSON.

    Raises:
        ValueError: the prompt carries no ``<doc id="doc_N">`` blocks, so there is nothing to judge.
    """
    docs = {int(index): body for index, body in _DOC_BLOCK.findall(prompt)}
    if not docs:
        raise ValueError("FakeJudge found no <doc id=...> blocks in the prompt")
    window = "|".join(docs[index] for index in sorted(docs))
    labels = criterion_labels_in(_DOCUMENTS.sub("", prompt))
    if labels:
        payload: dict[str, Any] = {
            f"doc_{index}": {
                "criteria": {
                    f"C{k + 1}": int(
                        _uniform(seed, "rubric", window, body, k)
                        < sigmoid(DISCRIMINATION * (ability(body) - severity - b))
                    )
                    for k, b in enumerate(_difficulties(len(labels)))
                }
            }
            for index, body in docs.items()
        }
    else:
        scores = {
            str(index): round(
                max(-5.0, min(5.0, ability(body) + TOURNAMENT_NOISE * (2 * _uniform(seed, window, body) - 1))),
                4,
            )
            for index, body in docs.items()
        }
        ranking = sorted(scores, key=lambda index: (-scores[index], int(index)))
        payload = {"ranking": [int(index) for index in ranking], "scores": scores}
    return json.dumps(payload)


def _prompt_text(messages: Any) -> str:
    """The rendered prompt of a chat-completions request's ``messages``.

    A text prompt is the string ``content`` itself. A media prompt is rebuilt from
    its blocks: each ``image_url``/``video_url`` block stood for one
    ``<media index="N"/>`` marker of the rendered prompt, at its position among the
    text blocks, so the prompt the fake reads is the one the window rendered.

    A video sent as sampled frames lowers to one block per frame where the prompt
    carried one marker, so a framed clip shifts the fake's draws; a text prompt,
    every page and every whole container round-trip exactly.
    """
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return ""
    content = messages[0].get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    media = 0
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        else:
            parts.append(f'<media index="{media}"/>')
            media += 1
    return "".join(parts)


def _chat_answer(
    body: dict[str, Any], *, seed: int, ability: Callable[[str], float], severity: float, model: str
) -> httpx.Response:
    """The chat-completions answer of the fake judge: the payload, with its approximate token counts."""
    prompt = _prompt_text(body.get("messages"))
    answer = _answer_text(seed, ability, severity, prompt)
    input_tokens, output_tokens = approx_tokens(len(prompt)), approx_tokens(len(answer))
    return httpx.Response(
        200,
        json={
            "id": "fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


def _fake_chat_completions(request: httpx.Request, endpoint: Any) -> httpx.Response:
    """The ``fake://`` chat-completions route: the offline judge's answers, at the seed the URL names."""
    try:
        body = json.loads(request.content)
    except ValueError:
        body = {}
    return _chat_answer(
        body if isinstance(body, dict) else {},
        seed=endpoint.seed,
        ability=_hidden_ability(endpoint.seed),
        severity=0.0,
        model=endpoint.model,
    )


register_fake_route("POST", "/chat/completions", _fake_chat_completions)


class FakeJudge(JudgeClient):
    """A deterministic judge for offline runs, examples and tests.

    It runs as a real :class:`~rcp_ndcg.judging.JudgeClient` over the real transport, with the endpoint faked
    below it: a ``fake://`` config (``JudgeConfig.fake(seed)``, or :meth:`from_config`) is answered by the
    chat route registered above, and a judge with its own ability mapping answers through its own in-process
    endpoint (:meth:`_answer`), with the same answer logic as the route.

    Args:
        ability: The hidden ability (logits) of a document, from its text as shown in the
            prompt: a callable or a mapping. Default: a standard-normal draw per text, fixed by ``seed``.
        severity: Added to every difficulty; a lenient judge has a negative severity.
        seed: Changes every draw.
        name: The judge's model name, recorded in the judgement family (``"fake"`` by default).
        config: The judge config to run under (a ``fake://`` one): every field of it applies, as for a real
            judge (its tokenizer, context, media limits, decoding and concurrency); ``name`` is then its
            ``model``. ``None`` builds the default offline config for ``seed`` and ``name``.
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
            where = config.urls[0] if config.urls else "(no base_url)"
            raise ValueError(f"the fake judge runs under a fake:// config, not {where!r}")
        if config is not None and config.fake_seed is not None:
            # The config is the instrument: its URL's seed decides the draws (and the identity), so an
            # explicit ``seed`` beside a config is overridden rather than silently recorded against it.
            seed = config.fake_seed
        self._fake_config = config or JudgeConfig(
            base_url=f"{FAKE_URL_SCHEME}seed/{seed}", model=name, temperature=None
        )
        if ability is None:
            self.ability: Callable[[str], float] = _hidden_ability(seed)
        elif isinstance(ability, Mapping):
            self.ability = ability.__getitem__
        else:
            self.ability = ability
        self.severity = severity
        self.seed = seed
        super().__init__(self._fake_config, httpx_transport=httpx.MockTransport(self._answer))

    @classmethod
    def from_config(cls, config: JudgeConfig) -> FakeJudge:
        """The fake judge of a ``fake://`` config with any fields set on it.

        The seed is the config's own (its URL's, parsed once by :meth:`JudgeConfig.fake_seed`); every other
        field of ``config`` applies, so a pass on the fake judge records the model, tokenizer, context, media
        limits and decoding it was given. (``JudgeConfig.fake(seed)`` itself builds a plain
        :class:`~rcp_ndcg.judging.JudgeClient` over the registered route, which answers identically for the
        default ability.)
        """
        seed = config.fake_seed if config.fake_seed is not None else 0
        return cls(seed=seed, config=config)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        """The in-process endpoint below the transport: the chat route's answer, at this judge's ability.

        Test doubles override it to refuse, garble or drop answers at the wire, exactly as an endpoint would.
        """
        try:
            body = json.loads(request.content)
        except ValueError:
            body = {}
        return _chat_answer(
            body if isinstance(body, dict) else {},
            seed=self.seed,
            ability=self.ability,
            severity=self.severity,
            model=self.config.model,
        )


__all__ = ["DEFAULT_DIFFICULTIES", "FakeJudge"]
