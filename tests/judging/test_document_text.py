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
from rcp_ndcg.judging import JudgementStore, judge
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


def test_title_joined_and_separate_documents_never_pool(tmp_path: Path) -> None:
    """``title`` decides the string the judge reads, so it is part of the judgement family and of the record
    ids: the two passes share no family key and no window, and a merge keeps both stores' judgements."""
    joined = _Recording()
    judge(_dataset(), None, joined, stage="rubric", out=tmp_path / "join", schedule=TINY_RUBRIC)
    separate = _Recording()
    separate.config = separate.config.model_copy(update={"title": "separate"})
    judge(_dataset(), None, separate, stage="rubric", out=tmp_path / "separate", schedule=TINY_RUBRIC)

    first = JudgementStore(tmp_path / "join").read()
    second = JudgementStore(tmp_path / "separate").read()
    (joined_family,), (separate_family,) = first.families.values(), second.families.values()
    assert joined_family.title is None and separate_family.title == "separate"
    assert joined_family.key != separate_family.key
    assert joined_family.rubric_key != separate_family.rubric_key
    assert not {j.record_id for j in first.judgements} & {j.record_id for j in second.judgements}


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


def test_a_loaded_dataset_without_a_task_instruction_keeps_its_identity_key(tmp_path: Path) -> None:
    """An absent instruction is the absence of a declaration: a loaded dataset that carries none keeps the
    identity it had (the key is added only when one is declared, as ``identity_payload`` does)."""
    from rcp_ndcg.data import load_dataset

    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "title": "T", "text": "a body"}) + "\n")
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "find docs"}) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    dataset = load_dataset(f"beir:{root}")
    assert dataset.task_instruction is None

    judge(dataset, None, _Recording(), stage="rubric", out=tmp_path / "store", schedule=TINY_RUBRIC)

    identity = json.loads((tmp_path / "store" / "identity.json").read_text())
    assert set(identity["stages"]["rubric"]["identity"]["dataset"]) == {"name", "uri", "revision"}


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
