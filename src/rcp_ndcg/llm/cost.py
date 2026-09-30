"""What judging will cost, before it runs: calls, tokens, USD and wall time.

:func:`estimate` walks the same queries, schedules and window budgets the
judging pass uses (:mod:`rcp_ndcg.llm.judging`), so the call count is the
schedule's own (216 calls per query of 150 candidates for the paper's
tournament, 100 for its rubric). Text tokens are counted exactly with the judge's
tokenizer (``JudgeConfig.tokenizer``) when it names one, and otherwise approximated
from characters (:mod:`rcp_ndcg.llm.tokens`, labelled as such); image tokens come from the
pass's ``preprocessing.image`` pixel budget under the judge's ``image_processor``, and
are approximated (labelled, :data:`~rcp_ndcg.llm.tokens.APPROX_TOKENS_PER_IMAGE` each) for a judge
that declares none and whose pass sends documents whole. The cost is ``None`` -- unknown, never 0 --
when the judge has no price.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule, schedule_for
from rcp_ndcg.llm.tokens import APPROXIMATION_NOTE, IMAGE_APPROXIMATION_NOTE, approx_media_tokens, approx_tokens

if TYPE_CHECKING:
    from rcp_ndcg.data.preprocess import Preprocessing
    from rcp_ndcg.llm.client import JudgeClient, JudgeConfig

#: Assumed completion tokens per document in a tournament window (its score, rank and share of the reasoning line).
TOURNAMENT_OUTPUT_TOKENS_PER_DOC = 24

#: Assumed completion tokens per document in a rubric window: a reasoning line and one verdict per criterion.
RUBRIC_OUTPUT_TOKENS_PER_DOC = 40

#: Assumed seconds per judge call, for the wall-time estimate.
SECONDS_PER_CALL = 20.0

#: Seconds per call of the offline fake judge, which answers in-process.
FAKE_SECONDS_PER_CALL = 0.005


class Price(BaseModel):
    """Token prices of the judge, in USD per million tokens.

    Attributes:
        input_usd_per_mtok: Prompt tokens.
        output_usd_per_mtok: Completion tokens (reasoning included).
        cached_input_usd_per_mtok: Prompt tokens served from the vendor's cache; ``None`` bills them as input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    cached_input_usd_per_mtok: float | None = Field(default=None, ge=0)

    def cost_usd(self, *, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        """The cost of one call, in USD."""
        cached_rate = (
            self.cached_input_usd_per_mtok if self.cached_input_usd_per_mtok is not None else self.input_usd_per_mtok
        )
        billable = max(input_tokens - cached_input_tokens, 0)
        return (
            billable * self.input_usd_per_mtok
            + cached_input_tokens * cached_rate
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


class StageEstimate(BaseModel):
    """One stage's share of a :class:`CostEstimate`."""

    model_config = ConfigDict(frozen=True)

    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None


class CostEstimate(BaseModel):
    """A projected judging pass.

    Attributes:
        calls: Judge calls (an upper bound: the adaptive phase can stop early).
        input_tokens: Prompt tokens (exact or approximate: ``input_token_count``).
        output_tokens: Completion tokens (approximate).
        usd: Cost in USD, or ``None`` when the judge has no price.
        wall_s: Wall time in seconds at the judge's concurrency (approximate).
        stages: Per stage.
        input_token_count: ``"exact"`` when the text was counted with the judge's tokenizer, ``"approximate"``
            when it was estimated from characters (the judge names no tokenizer).
        assumptions: What the numbers rest on.
    """

    model_config = ConfigDict(frozen=True, validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.cost-estimate.v1"] = Field(default="rcp-ndcg.cost-estimate.v1", alias="schema")
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None
    wall_s: float
    stages: dict[str, StageEstimate]
    input_token_count: Literal["exact", "approximate"]
    assumptions: list[str]


def _usd(price: Price | None, input_tokens: int, output_tokens: int) -> float | None:
    return None if price is None else price.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)


def estimate(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    judge: JudgeConfig | JudgeClient,
    *,
    stages: Sequence[Literal["tournament", "rubric"]] = ("tournament", "rubric"),
    schedules: Mapping[str, TournamentSchedule | RubricSchedule] | None = None,
    preprocessing: Preprocessing | None = None,
    docs: Mapping[str, Sequence[str]] | None = None,
    windows: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> CostEstimate:
    """Project the calls, tokens, cost and wall time of judging ``candidates`` with ``judge``.

    The projection is of a pass into an empty store: windows a store already holds
    are reused by the pass and cost nothing, which this estimate does not subtract.

    Args:
        dataset: The dataset, as for :func:`~rcp_ndcg.llm.judging.judge`.
        candidates: Each query's pool (``None``: the dataset's own pools).
        judge: The judge; its price, concurrency, context window and output cap enter the estimate.
        stages: The stages to price.
        schedules: ``{stage: schedule}``; a stage left out uses the paper's schedule for the corpus's modality.
        preprocessing: The text policy, chunking and image policy the pass would apply.
        docs: ``{query_id: [doc_id, ...]}`` -- price judging only these documents of each pool, as
            ``judge(docs=...)`` does (the rubric scales its window count to the subset).
        windows: ``{query_id: [[doc_id, ...], ...]}`` -- price exactly these windows, as ``judge(windows=...)``
            asks them (each twice when the tournament schedule mirrors).

    Returns:
        The :class:`CostEstimate` (no call is made).

    Raises:
        ConfigError: the documents carry images whose token cost cannot be counted (no pixel budget, or a judge
            without an ``image_processor``) and the pass would budget text on it (the judge declares
            ``context_tokens`` and a ``tokenizer``); the pass refuses the same. Without a text budget their tokens
            are approximated instead, and the assumptions say so.
    """
    from rcp_ndcg.data.preprocess import document_ids_from_chunks
    from rcp_ndcg.llm._templates import rendered_text
    from rcp_ndcg.llm.client import JudgeClient
    from rcp_ndcg.llm.judging import (
        CHAT_TEMPLATE_TOKENS,
        _effective_preprocessing,
        _judge_tokenizer,
        _media_tokens,
        _modality,
        _queries,
        prompt_overhead_tokens,
        window_tokens,
    )
    from rcp_ndcg.llm.prompts import load_prompt, shipped_prompt_name

    config = judge.config if isinstance(judge, JudgeClient) else judge
    effective = _effective_preprocessing(preprocessing, config)
    tokenizer = _judge_tokenizer(config)
    if windows is not None:
        docs = {query: list(dict.fromkeys(doc for window in rows for doc in window)) for query, rows in windows.items()}
    _name, queries, _source = _queries(dataset, candidates, docs, effective, tokenizer=tokenizer)
    counted: dict[str, int] = {}
    images_approximated = False

    def text_tokens(text: str) -> int:
        """A document's tokens as the prompt carries it (each distinct text counted once)."""
        if text not in counted:
            counted[text] = tokenizer.count(rendered_text(text)) if tokenizer is not None else 0
        return counted[text]

    modality = _modality(queries)
    policy, video = effective.image, effective.video
    per_stage: dict[str, StageEstimate] = {}
    for stage in stages:
        schedule = (schedules or {}).get(stage) or schedule_for(stage, modality)
        prompt = load_prompt(schedule.prompt or shipped_prompt_name(stage, modality))
        window = schedule.window
        calls = input_tokens = output_tokens = 0
        for query in queries:
            n = len(query.units)
            w = min(window, n)
            if w == 0:
                continue
            if windows is not None:
                planned = windows[query.query_id]
                w = max(len(rows) for rows in planned)
                mirrored = isinstance(schedule, TournamentSchedule) and schedule.mirror
                n_calls = len(planned) * (2 if mirrored else 1)
            elif isinstance(schedule, TournamentSchedule):
                n_calls = schedule.calls_per_query(n)
            else:
                documents = len(document_ids_from_chunks(query.units, query.chunk_mapping))
                pool_size = query.pool_size if documents < query.pool_size else None
                n_calls = schedule.calls_per_query(documents, pool_size=pool_size, n_units=n)
            budgeted = tokenizer is not None and config.context_tokens is not None
            counted_media = _media_tokens(query.contents.values(), effective, strict=budgeted)
            if counted_media is None:
                images_approximated = True
            media = (
                counted_media
                if counted_media is not None
                else max((approx_media_tokens(c, video) for c in query.contents.values()), default=0)
            )
            if tokenizer is not None:
                overhead = prompt_overhead_tokens(prompt, stage, query.text, w, tokenizer)
                cap = window_tokens(config, w, overhead_tokens=overhead, media_tokens_per_doc=media)
                tokens = [text_tokens(c.text) for c in query.contents.values()]
                tokens = [min(t, cap) for t in tokens] if cap is not None else tokens
                per_call = overhead + math.ceil(w * sum(tokens) / max(len(tokens), 1)) + w * media
            else:
                # No tokenizer: documents are sent whole, and their tokens are approximated from characters.
                if counted_media is not None:  # known images must fit, as in the pass
                    window_tokens(config, w, overhead_tokens=0, media_tokens_per_doc=counted_media)
                chars = [len(c.text) for c in query.contents.values()]
                mean_chars = sum(chars) / len(chars) if chars else 0.0
                per_call = approx_tokens(len(prompt.text) + len(query.text) + w * mean_chars) + w * media
            per_doc_out = TOURNAMENT_OUTPUT_TOKENS_PER_DOC if stage == "tournament" else RUBRIC_OUTPUT_TOKENS_PER_DOC
            out = w * per_doc_out
            if config.max_output_tokens is not None:
                out = min(out, config.max_output_tokens)
            calls += n_calls
            input_tokens += n_calls * per_call
            output_tokens += n_calls * out
        per_stage[stage] = StageEstimate(
            calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=_usd(config.price, input_tokens, output_tokens),
        )
    seconds_per_call = FAKE_SECONDS_PER_CALL if config.is_fake else SECONDS_PER_CALL
    calls = sum(s.calls for s in per_stage.values())
    input_tokens = sum(s.input_tokens for s in per_stage.values())
    output_tokens = sum(s.output_tokens for s in per_stage.values())
    assumptions = [
        f"input tokens counted exactly with the judge's tokenizer {tokenizer.name}, plus {CHAT_TEMPLATE_TOKENS} per "
        "call reserved for the chat template"
        if tokenizer is not None
        else f"input tokens {APPROXIMATION_NOTE}; the judge names no tokenizer",
        f"output tokens assumed at {TOURNAMENT_OUTPUT_TOKENS_PER_DOC} per document per tournament window and "
        f"{RUBRIC_OUTPUT_TOKENS_PER_DOC} per document per rubric window"
        + (f", capped at max_output_tokens={config.max_output_tokens}" if config.max_output_tokens else "")
        + "; a model that reasons before answering emits more",
        f"wall time assumes {seconds_per_call:g} s per call at concurrency {config.concurrency}",
        "calls are an upper bound: the adaptive tournament phase stops early when no boundary is informative",
    ]
    if images_approximated:
        assumptions.append(f"image tokens {IMAGE_APPROXIMATION_NOTE}")
    elif policy is not None:
        assumptions.append(f"image tokens from the image policy {policy.descriptor}")
    free = config.price is not None and config.price.cost_usd(input_tokens=1, output_tokens=1) == 0.0
    if config.is_fake and free:
        assumptions.append("the offline fake judge spends nothing and answers in-process")
    elif config.price is None:
        assumptions.append(f"no price for {config.model}: usd is unknown")
    return CostEstimate(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=_usd(config.price, input_tokens, output_tokens),
        wall_s=math.ceil(calls / config.concurrency) * seconds_per_call,
        stages=per_stage,
        input_token_count="exact" if tokenizer is not None else "approximate",
        assumptions=assumptions,
    )


__all__ = ["CostEstimate", "Price", "StageEstimate", "estimate"]
