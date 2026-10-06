"""What judging will take, before it runs: calls, tokens and wall time.

:func:`estimate` walks the same queries, schedules and window budgets the
judging pass uses (:mod:`rcp_ndcg.llm.judging`), so the call count is the
schedule's own (216 calls per query of 150 candidates for the paper's
tournament, 100 for its rubric). Text tokens are counted exactly with the judge's
tokenizer (``JudgeConfig.tokenizer``) when it names one, and otherwise approximated
from characters (:mod:`rcp_ndcg.llm.tokens`, labelled as such); image tokens come from the
pass's ``preprocessing.image`` pixel budget under the judge's ``image_processor``, and
are approximated (labelled, :data:`~rcp_ndcg.llm.tokens.APPROX_TOKENS_PER_IMAGE` each) for a judge
that declares none and whose pass sends documents whole.
"""

from __future__ import annotations

import math
import warnings
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


class StageEstimate(BaseModel):
    """One stage's share of a :class:`CostEstimate`."""

    model_config = ConfigDict(frozen=True)

    calls: int
    input_tokens: int
    output_tokens: int


class CostEstimate(BaseModel):
    """A projected judging pass.

    Attributes:
        calls: Judge calls (an upper bound: the adaptive phase can stop early).
        input_tokens: Prompt tokens (exact or approximate: ``input_token_count``).
        output_tokens: Completion tokens (approximate).
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
    wall_s: float
    stages: dict[str, StageEstimate]
    input_token_count: Literal["exact", "approximate"]
    assumptions: list[str]


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
    """Project the calls, tokens and wall time of judging ``candidates`` with ``judge``.

    The projection is of a pass into an empty store: windows a store already holds
    are reused by the pass and are not asked again, which this estimate does not subtract.

    Args:
        dataset: The dataset, as for :func:`~rcp_ndcg.llm.judging.judge`.
        candidates: Each query's pool (``None``: the dataset's own pools).
        judge: The judge; its concurrency, tokenizer, context window and output cap enter the estimate.
        stages: The stages to estimate.
        schedules: ``{stage: schedule}``; a stage left out uses the paper's schedule for the corpus's modality.
        preprocessing: The text policy, chunking and image policy the pass would apply.
        docs: ``{query_id: [doc_id, ...]}`` -- estimate judging only these documents of each pool, as
            ``judge(docs=...)`` does (the window counts follow the subset's size).
        windows: ``{query_id: [[doc_id, ...], ...]}`` -- estimate exactly these windows, as ``judge(windows=...)``
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
        media_marker_tokens,
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
    # The template's per-part media marker, measured with the judge's tokenizer; 0 where no tokenizer is at
    # hand (no text budget is computed then, and the media are approximated instead).
    marker = media_marker_tokens(tokenizer) if tokenizer is not None else 0

    def text_tokens(text: str) -> int:
        """A document's tokens as the prompt carries it (each distinct text counted once)."""
        if text not in counted:
            counted[text] = tokenizer.count(rendered_text(text)) if tokenizer is not None else 0
        return counted[text]

    modality = _modality(queries)
    policy, video = effective.image, effective.video
    per_stage: dict[str, StageEstimate] = {}
    images_approximated = False
    for stage in stages:
        schedule = (schedules or {}).get(stage) or schedule_for(stage, modality)
        prompt = load_prompt(schedule.prompt or shipped_prompt_name(stage, modality))
        calls = input_tokens = output_tokens = 0
        per_doc_out = TOURNAMENT_OUTPUT_TOKENS_PER_DOC if stage == "tournament" else RUBRIC_OUTPUT_TOKENS_PER_DOC
        for query in queries:
            n = len(query.units)
            if n == 0:
                continue
            # (calls, documents per window): each group of calls is counted at its own window's budget.
            if windows is not None:
                planned = windows[query.query_id]
                mirrored = isinstance(schedule, TournamentSchedule) and schedule.mirror
                groups = [(len(planned) * (2 if mirrored else 1), max(len(rows) for rows in planned))]
            elif isinstance(schedule, TournamentSchedule):
                fixed, adaptive = schedule.phase_calls(n)
                groups = [(fixed, min(schedule.window, n)), (adaptive, min(schedule.adaptive_window, n))]
            else:
                documents = len(document_ids_from_chunks(query.units, query.chunk_mapping))
                groups = [(schedule.calls_per_query(documents, n_units=n), min(schedule.window, n))]
            budgeted = tokenizer is not None and config.context_tokens is not None
            counted_media = _media_tokens(query.contents.values(), effective, strict=budgeted, marker_tokens=marker)
            if counted_media is None:
                images_approximated = True
            media = (
                counted_media
                if counted_media is not None
                else max((approx_media_tokens(c, video) for c in query.contents.values()), default=0)
            )
            for n_calls, w in groups:
                if n_calls == 0:
                    continue
                if tokenizer is not None:
                    overhead = prompt_overhead_tokens(prompt, stage, query.text, w, tokenizer)
                    cap = window_tokens(config, w, overhead_tokens=overhead, media_tokens_per_doc=media)
                    tokens = [text_tokens(c.text) for c in query.contents.values()]
                    tokens = [min(t, cap) for t in tokens] if cap is not None else tokens
                    per_call = overhead + math.ceil(w * sum(tokens) / max(len(tokens), 1)) + w * media
                else:
                    # No tokenizer: documents are sent whole, and their tokens are approximated from characters.
                    # counted_media is already the strict=False, marker-free count (no tokenizer: marker 0).
                    if counted_media is not None:
                        window_tokens(config, w, overhead_tokens=0, media_tokens_per_doc=counted_media)
                    chars = [len(c.text) for c in query.contents.values()]
                    mean_chars = sum(chars) / len(chars) if chars else 0.0
                    per_call = approx_tokens(len(prompt.text) + len(query.text) + w * mean_chars) + w * media
                out = w * per_doc_out
                if config.max_output_tokens is not None:
                    out = min(out, config.max_output_tokens)
                calls += n_calls
                input_tokens += n_calls * per_call
                output_tokens += n_calls * out
        per_stage[stage] = StageEstimate(calls=calls, input_tokens=input_tokens, output_tokens=output_tokens)
    seconds_per_call = FAKE_SECONDS_PER_CALL if config.is_fake else SECONDS_PER_CALL
    calls = sum(s.calls for s in per_stage.values())
    input_tokens = sum(s.input_tokens for s in per_stage.values())
    output_tokens = sum(s.output_tokens for s in per_stage.values())
    per_window = {
        "tournament": f"{TOURNAMENT_OUTPUT_TOKENS_PER_DOC} per document per tournament window",
        "rubric": f"{RUBRIC_OUTPUT_TOKENS_PER_DOC} per document per rubric window",
    }
    assumptions = [
        f"input tokens counted exactly with the judge's tokenizer {tokenizer.name}, plus {CHAT_TEMPLATE_TOKENS} per "
        "call reserved for the chat template"
        if tokenizer is not None
        else f"input tokens {APPROXIMATION_NOTE}; the judge names no tokenizer",
        "output tokens assumed at "
        + " and ".join(per_window[stage] for stage in per_stage)
        + (f", capped at max_output_tokens={config.max_output_tokens}" if config.max_output_tokens else "")
        + "; a model that reasons before answering emits more",
        f"wall time assumes {seconds_per_call:g} s per call at concurrency {config.concurrency}",
    ]
    if "tournament" in per_stage and windows is None:
        assumptions.append(
            "calls are an upper bound: the adaptive tournament phase stops early when no boundary is informative"
        )
    if images_approximated:
        assumptions.append(f"image tokens {IMAGE_APPROXIMATION_NOTE}")
        from rcp_ndcg.errors import RcpNdcgWarning

        warnings.warn(
            RcpNdcgWarning("APPROXIMATE_IMAGE_TOKENS", f"the estimate's image tokens are {IMAGE_APPROXIMATION_NOTE}"),
            stacklevel=2,
        )
    elif policy is not None:
        assumptions.append(f"image tokens from the image policy {policy.descriptor}")
    if config.is_fake:
        assumptions.append("the offline fake judge answers in-process")
    return CostEstimate(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_s=math.ceil(calls / config.concurrency) * seconds_per_call,
        stages=per_stage,
        input_token_count="exact" if tokenizer is not None else "approximate",
        assumptions=assumptions,
    )


__all__ = ["CostEstimate", "StageEstimate", "estimate"]
