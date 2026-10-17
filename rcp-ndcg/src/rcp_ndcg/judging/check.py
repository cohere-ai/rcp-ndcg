"""``judge check``: a short conformance probe of a judge endpoint, before a long run.

The probe sends the shipped stage prompts over **fixed probe windows** and reports, per stage, the three
things a judging pass depends on:

* **structured output accepted** -- with ``decoding: json_schema`` the request carried the stage's answer
  schema as ``response_format`` and the endpoint answered it (an endpoint that refuses the schema is a
  :class:`~rcp_ndcg.errors.CapabilityError` naming it);
* **parses** -- the answer parses with the stage's own parser into a placement (tournament) or the criteria
  C1..C5 (rubric);
* **the reasoning parser separates the answer** -- the endpoint reported a reasoning channel beside the
  answer (``separated``), or it reported none (``absent``; with ``decoding: json_schema`` that can mean the
  engine runs without the model's reasoning parser, or that the model does not reason -- the same advisory
  the judging pass logs after its first answers, available here before the run starts).

It is a *probe*, not a reference: two windows, no store, no schedule, nothing written.  ``judge check``
runs it and reports; a failed check is a non-zero ``ok`` in the report, like ``rcp-ndcg doctor``.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError, RequestRejectedError
from rcp_ndcg.judging._parsing.schema import response_format as _response_format
from rcp_ndcg.judging.client import CompletionInput, JudgeClient, JudgeConfig, Usage
from rcp_ndcg.judging.judging import parse_window
from rcp_ndcg.judging.prompts import load_prompt, shipped_prompt_name

__all__ = ["JudgeCheck", "JudgeCheckReport", "check_judge"]

#: The probe window's query (fixed: the same request every run, so two runs' answers are comparable).
PROBE_QUERY = "probe query"

#: The probe window's documents (fixed; two documents are the smallest window both stages accept).
PROBE_DOCUMENTS = ("probe document one", "probe document two")

#: The probe window's query id (only the parsers read it, and they only echo it).
PROBE_QUERY_ID = "probe"


class JudgeCheck(BaseModel):
    """What one stage's probe window showed.

    Attributes:
        stage: ``tournament`` (Stage A) or ``rubric`` (Stage B, criteria C1..C5).
        schema_sent: Whether the request carried the stage's answer schema as ``response_format``
            (``decoding: json_schema``); ``False`` for a free-decoding judge.
        accepted: Whether the endpoint answered the request (an answer, or a refusal it reported).
        parsed: Whether the answer parsed with the stage's own parser.
        reasoning_channel: ``separated`` when the endpoint reported a reasoning channel beside the answer,
            ``absent`` when it reported none.
        detail: The refusal or parse failure, or the reasoning-channel advisory; empty when nothing to say.
    """

    stage: Literal["tournament", "rubric"]
    schema_sent: bool
    accepted: bool
    parsed: bool
    reasoning_channel: Literal["separated", "absent"]
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether this stage's window passed: the request was answered and the answer parsed."""
        return self.accepted and self.parsed


class JudgeCheckReport(BaseModel):
    """The probe's result: one :class:`JudgeCheck` per stage, and ``ok`` when every one passed."""

    ok: bool = Field(description="True when every probe window was answered and parsed.")
    model: str = Field(description="The judge config's served model name.")
    revision: str | None = Field(default=None, description="The checkpoint revision the config pins, when any.")
    decoding: str = Field(description="The judge config's decoding: json_schema or free.")
    recipe: str | None = Field(default=None, description="The recipe the config came from, when it named one.")
    checks: list[JudgeCheck] = Field(description="The probe windows, one per stage, in stage order.")
    usage: Usage = Field(description="Calls and tokens of the probe itself.")


async def _probe_stage(client: JudgeClient, stage: Literal["tournament", "rubric"]) -> JudgeCheck:
    """One stage's fixed probe window: render, ask, parse; report instead of raising.

    The rendering, the answer schema and the parser are the judging pass's own functions, so a pass that
    would refuse the prompt or fail to parse fails here too.
    """
    prompt = load_prompt(shipped_prompt_name(stage, "text"))
    template = prompt.template(with_num_documents=stage == "tournament")
    documents = [Content.from_text(text) for text in PROBE_DOCUMENTS]
    resolved = template.resolve_prompt(query=PROBE_QUERY, documents=documents)
    units = [f"probe_{index}" for index in range(len(documents))]
    schema_sent = client.config.decoding == "json_schema"
    response_format = _response_format(stage, len(units), prompt.criteria) if schema_sent else None
    request = CompletionInput(user_prompt=resolved.text, user_content=resolved.content, response_format=response_format)
    try:
        completion = await client.complete(request)
    except (CapabilityError, RequestRejectedError) as error:
        return JudgeCheck(
            stage=stage, schema_sent=schema_sent, accepted=False, parsed=False,
            reasoning_channel="absent", detail=f"{type(error).__name__}: {error}",
        )  # fmt: skip
    reasoning = "separated" if (completion.reasoning or "").strip() else "absent"
    detail = ""
    if reasoning == "absent" and schema_sent:
        detail = (
            "the endpoint reported no reasoning channel: the engine may run without the model's reasoning "
            "parser, or the model does not reason (the judging pass logs the same advisory after its first "
            "answers)"
        )
    try:
        parse_window(stage, PROBE_QUERY_ID, completion, units, len(prompt.criteria))
    except Exception as error:  # noqa: BLE001 - any parser refusal is this window's report, never the command's
        return JudgeCheck(
            stage=stage, schema_sent=schema_sent, accepted=True, parsed=False,
            reasoning_channel=reasoning, detail=f"{type(error).__name__}: {error}",
        )  # fmt: skip
    return JudgeCheck(
        stage=stage, schema_sent=schema_sent, accepted=True, parsed=True, reasoning_channel=reasoning, detail=detail
    )


def check_judge(config: JudgeConfig) -> JudgeCheckReport:
    """Probe ``config``'s endpoint with the fixed tournament and rubric windows and report.

    Inputs: the judge config (a served judge's URL must be set; a recipe-derived config without one is
    refused by the client, with the same typed error a judging pass gives).  Output: the
    :class:`JudgeCheckReport` -- ``ok`` when every stage's window was answered and parsed.  Raises
    :class:`~rcp_ndcg.errors.ConfigError`: the config names no URL and no adapter default, or names an
    unknown wire adapter; the transport's own typed errors (credentials, an unreachable endpoint) propagate.
    """
    client = JudgeClient.from_config(config)
    checks = asyncio.run(_probe(client))
    return JudgeCheckReport(
        ok=all(check.ok for check in checks),
        model=config.model,
        revision=config.revision,
        decoding=config.decoding,
        recipe=config.recipe,
        checks=checks,
        usage=client.usage,
    )


async def _probe(client: JudgeClient) -> list[JudgeCheck]:
    """The two stage probes, in stage order."""
    return [await _probe_stage(client, "tournament"), await _probe_stage(client, "rubric")]
