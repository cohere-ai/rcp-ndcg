"""The judge's prompt formatting: the document's title join and the two instructions (decisions 27, 33).

The judge reads a document as MTEB's dataloader does -- ``(title + " " + body).strip()``, the body alone
without a title -- and its query slot carries the task instruction's generic prefix and the per-query
instruction's append, each once. ``title: separate`` on the judge config takes the title as its own part
instead. The prompt is captured at the fake judge's own endpoint, so the assertion is on what a model
would read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from rcp_ndcg.data import Dataset
from rcp_ndcg.judging import judge
from rcp_ndcg.testing import TINY_RUBRIC, FakeJudge

TASK_INSTRUCTION = "Given a claim, find documents that refute the claim"


class _Recording(FakeJudge):
    """The fake judge, recording every user prompt it is sent (the text a real judge would read)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompts: list[str] = []

    def _answer(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        for message in body.get("messages", []):
            if message.get("role") == "user":
                content = message.get("content")
                self.prompts.append(content if isinstance(content, str) else json.dumps(content))
        return super()._answer(request)


def _dataset() -> Dataset:
    return Dataset.from_records(
        name="sample",
        corpus=[
            {"doc_id": "d1", "title": "Tortoises", "text": "a tortoise is a reptile"},
            {"doc_id": "d2", "title": "  ", "text": "  a haiku about ponds  "},
        ],
        queries=[{"query_id": "q1", "text": "find docs", "instruction": "about turtles"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1.0}],
        candidates={"q1": ["d1", "d2"]},
        task_instruction=TASK_INSTRUCTION,
    )


def _judge(tmp_path: Path, **update: Any) -> _Recording:
    fake = _Recording()
    if update:
        fake.config = fake.config.model_copy(update=update)
    judge(_dataset(), None, fake, stage="rubric", out=tmp_path / "store", schedule=TINY_RUBRIC)
    return fake


def test_the_judges_documents_read_the_mteb_join(tmp_path: Path) -> None:
    fake = _judge(tmp_path)

    prompt = "\n".join(fake.prompts)
    assert '<doc id="doc_1">\nTortoises a tortoise is a reptile\n</doc>' in prompt
    assert "a haiku about ponds" in prompt
    assert "  a haiku about ponds  " not in prompt, "the body is stripped, as mteb's dataloader strips it"


def test_the_judges_query_carries_both_instructions_once(tmp_path: Path) -> None:
    fake = _judge(tmp_path)

    assert f"Task: {TASK_INSTRUCTION}\nQuery: find docs about turtles" in "\n".join(fake.prompts)


def test_the_judge_can_take_the_title_separately(tmp_path: Path) -> None:
    fake = _judge(tmp_path, title="separate")

    assert "Tortoises\na tortoise is a reptile" in "\n".join(fake.prompts)


def test_the_judging_identity_records_the_task_instruction(tmp_path: Path) -> None:
    """Which instructions a model saw is part of the pass's identity: two in-memory datasets that differ only
    in their task instruction never share a store (the instruction is in the content digest)."""
    _judge(tmp_path)
    first = json.loads((tmp_path / "store" / "identity.json").read_text())

    other = _Recording()
    dataset = _dataset().model_copy(update={"task_instruction": "another instruction"})
    judge(dataset, None, other, stage="rubric", out=tmp_path / "other", schedule=TINY_RUBRIC)
    second = json.loads((tmp_path / "other" / "identity.json").read_text())

    first_dataset = first["stages"]["rubric"]["identity"]["dataset"]
    second_dataset = second["stages"]["rubric"]["identity"]["dataset"]
    assert first_dataset["content_sha256"] != second_dataset["content_sha256"]
    assert first["stages"]["rubric"]["identity"] != second["stages"]["rubric"]["identity"]


def test_a_dataset_without_a_task_instruction_records_none(tmp_path: Path) -> None:
    fake = _Recording()
    dataset = _dataset().model_copy(update={"task_instruction": None})
    judge(dataset, None, fake, stage="rubric", out=tmp_path / "store", schedule=TINY_RUBRIC)

    identity = json.loads((tmp_path / "store" / "identity.json").read_text())
    dataset = identity["stages"]["rubric"]["identity"]["dataset"]
    assert set(dataset) == {"name", "content_sha256"}
    assert "Task: " not in "\n".join(fake.prompts)


def test_the_judge_reads_the_query_side_of_a_per_side_instruction(tmp_path: Path) -> None:
    """A ``{"query": ..., "document": ...}`` instruction names its sides: the judge's prompt is the query's,
    so the query-side text is prefixed and the document side's is not the judge's input (it frames the
    RETRIEVAL role's documents, which the judge does not serve)."""
    dataset = _dataset().model_copy(
        update={"task_instruction": {"query": TASK_INSTRUCTION, "document": "a passage instruction"}}
    )
    fake = _Recording()
    judge(dataset, None, fake, stage="rubric", out=tmp_path / "store", schedule=TINY_RUBRIC)

    prompts = "\n".join(fake.prompts)
    assert f"Task: {TASK_INSTRUCTION}\nQuery: find docs about turtles" in prompts
    assert "a passage instruction" not in prompts
