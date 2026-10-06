"""The shipped judging prompts, as package data, loaded by name.

``tournament`` is Stage A, ``rubric`` is Stage B (criteria C1-C5); the
``_vision`` and ``_video`` variants read page images and videos. The files ship
inside the package, so the prompts load from any working directory and after
``pip install``. A path or URI loads a custom prompt instead; it is a different
instrument and hashes to a different judgement family.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rcp_ndcg.errors import ConfigError

if TYPE_CHECKING:
    from rcp_ndcg.llm._templates import Template
    from rcp_ndcg.llm.schedule import Modality

PromptName = Literal["tournament", "rubric", "tournament_vision", "rubric_vision", "tournament_video", "rubric_video"]

#: Prompt name -> file in this package.
PROMPT_FILES: dict[str, str] = {
    "tournament": "tournament.txt",
    "rubric": "rubric.txt",
    "tournament_vision": "tournament_vision.txt",
    "rubric_vision": "rubric_vision.txt",
    "tournament_video": "tournament_video.txt",
    "rubric_video": "rubric_video.txt",
}

_CRITERION = re.compile(r"\bC([1-9][0-9]?)\b")


def criterion_labels_in(text: str) -> tuple[str, ...]:
    """The criterion labels ``C1..Cn`` ``text`` names, when they form a contiguous ladder; empty otherwise.

    The one derivation of "which criteria does this prompt ask for" (it is what the judge answers):
    :attr:`Prompt.criteria` and the fake judge's rubric answers read it.
    """
    labels = {int(match) for match in _CRITERION.findall(text)}
    if not labels or labels != set(range(1, max(labels) + 1)):
        return ()
    return tuple(f"C{k}" for k in range(1, max(labels) + 1))

#: The slots every judging prompt must have: the query, and the window's documents.
REQUIRED_PLACEHOLDERS: tuple[str, ...] = ("query_placeholder", "passages_placeholder")


@dataclass(frozen=True)
class Prompt:
    """A judging prompt: its name (a shipped name or the path it was read from) and its text."""

    name: str
    text: str

    @cached_property
    def sha256(self) -> str:
        """SHA-256 of the text: the prompt's identity in a judgement family."""
        from rcp_ndcg.support.identity import hash_text

        return hash_text(self.text)

    @cached_property
    def criteria(self) -> tuple[str, ...]:
        """The criterion labels the prompt asks for (``("C1", ..., "C5")`` for the shipped rubric).

        Derived from the text, which is what the judge answers (see :func:`criterion_labels_in`): empty for
        the tournament prompts.
        """
        return criterion_labels_in(self.text)

    def template(self, *, with_num_documents: bool) -> Template:
        """The prompt as a :class:`~rcp_ndcg.llm._templates.Template` over the query and the documents.

        Placeholder order is ``[Query, (NumDocuments), Passages]``; the tournament prompts
        take the number of documents, the rubric prompts do not.
        """
        from rcp_ndcg.llm._templates import NumDocumentsPlaceholder, PassagesPlaceholder, QueryPlaceholder, Template

        placeholders: list = [QueryPlaceholder()]
        if with_num_documents:
            placeholders.append(NumDocumentsPlaceholder())
        placeholders.append(PassagesPlaceholder())
        return Template(template_str=self.text, placeholders=placeholders)


def prompt_path(name: PromptName) -> Path:
    """The installed file of the shipped prompt *name*.

    Raises:
        ConfigError: *name* is not a shipped prompt; the message lists the names.
    """
    if name not in PROMPT_FILES:
        raise ConfigError(f"no shipped prompt {name!r}; known: {', '.join(PROMPT_FILES)}")
    return Path(str(files(__name__) / PROMPT_FILES[name]))


def load_prompt(name: str) -> Prompt:
    """The shipped prompt *name*, or the prompt at a path or URI; text stripped of surrounding whitespace.

    Raises:
        ConfigError: the prompt lacks a slot of :data:`REQUIRED_PLACEHOLDERS` (``{query_placeholder}``,
            ``{passages_placeholder}``): a judge asked with it would never see the query or the documents.
    """
    if name in PROMPT_FILES:
        text = prompt_path(name).read_text(encoding="utf-8").strip()  # type: ignore[arg-type]
    else:
        from rcp_ndcg.storage.io import load_text

        text = load_text(name).strip()
    missing = [slot for slot in REQUIRED_PLACEHOLDERS if "{" + slot + "}" not in text]
    if missing:
        raise ConfigError(
            f"the prompt {name!r} lacks {', '.join('{' + slot + '}' for slot in missing)}: the judge would not see "
            "the query or the documents",
            hint="put {query_placeholder} and {passages_placeholder} where the shipped prompts have them",
            details={"prompt": name, "missing": missing},
        )
    return Prompt(name=name, text=text)


def shipped_prompt_name(stage: Literal["tournament", "rubric"], modality: Modality) -> PromptName:
    """The shipped prompt of a stage for text, page images or videos."""
    suffix = {"text": "", "image": "_vision", "video": "_video"}[modality]
    return f"{stage}{suffix}"  # type: ignore[return-value]


def shipped_prompts_digest(stage: Literal["tournament", "rubric"]) -> str:
    """SHA-256 over the shipped prompts of ``stage`` (text, page images and videos), by content.

    A run that leaves the schedule's prompt unset has a pass resolve the shipped one from the corpus's
    modality at judging time, so a step identity pins the stage's whole shipped set by content instead of
    naming one: any edited shipped prompt re-keys the step (the judgement family still carries the exact
    prompt's hash, and the names are runtime).
    """
    from rcp_ndcg_core._hashing import hash_payload

    names = sorted(name for name in PROMPT_FILES if name.startswith(stage))
    return hash_payload({name: load_prompt(name).text for name in names})


__all__ = [
    "PROMPT_FILES",
    "REQUIRED_PLACEHOLDERS",
    "Prompt",
    "PromptName",
    "criterion_labels_in",
    "load_prompt",
    "prompt_path",
    "shipped_prompt_name",
    "shipped_prompts_digest",
]
