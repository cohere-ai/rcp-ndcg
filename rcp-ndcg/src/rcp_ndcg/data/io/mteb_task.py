"""The ``mteb:<TaskName>[/<subset>][@split]`` reader: a task loaded through mteb itself (the ``[mteb]`` extra).

The ``hf://`` reader follows the layout rules of a dataset card, which is all a repository in the standard
layout needs. 113 tasks instead load their data through custom task code -- ViDoRe v1's id prefixes, BRIGHT's
``top_ranked`` built as the corpus minus its exclusions, the arXivQA id remapping -- and only mteb's own
:meth:`~mteb.abstasks.AbsTask.load_data` knows those rules.  This reader runs it and converts
``task.dataset[subset][split]`` (a :class:`~mteb.abstasks.retrieval_dataset_loaders.RetrievalSplitData`) into
the canonical records, so a custom-loaded task reads through the same contract as every other source:

* the ids and texts are mteb's own, whatever the task code did to them;
* the task name, subset, split and the pinned ``dataset_revision`` are recorded in the provenance, and the
  task's name and its ``TaskMetadata.prompt`` become the dataset's :attr:`Dataset.task` and
  :attr:`Dataset.task_instruction`;
* mteb's own loader has already applied its rules upstream (a repeated pair keeps the last value, queries are
  cut to those with qrels), so there is nothing left for the duplicates policy to do here.

The mteb import happens inside the methods that need it, so listing this reader costs nothing without the
extra (``pip install 'rcp-ndcg[mteb]'``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, cast

from rcp_ndcg_core._records import ID, Document, Query
from rcp_ndcg_core.content import Content, ImagePart, TextPart, VideoPart

from rcp_ndcg.data.io.base import DataShape, Provenance, SourceReader
from rcp_ndcg.data.io.hub import _encode_media
from rcp_ndcg.data.media import image_dimensions, store_media
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


class MtebTaskReader(SourceReader):
    """Reads an mteb task's data as mteb's own loader loads it.

    Args:
        uri: ``<TaskName>[/<subset>][@split]`` (the locator after the ``mteb:`` scheme). A split in the URI
            overrides the ``split`` option.
        split: The split to load; ``None`` is the task's first evaluation split.
        name: The dataset name; defaults to the subset's name, or the task's name for the ``default`` subset.

    Raises:
        ConfigError: The locator names no task, a task whose subsets are not named (a multilingual task), a
            subset or split the task does not have, or two splits.
    """

    name = "mteb"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    def __init__(self, uri: str, *, split: str | None = None, name: str | None = None) -> None:
        locator, _, uri_split = uri.partition("@")
        if uri_split and split and uri_split != split:
            raise ConfigError(f"two splits for one task: {uri_split!r} in the URI and {split!r}")
        self.split = split or uri_split or None
        parts = locator.split("/", 1)
        if not parts[0].strip():
            raise ConfigError(f"an mteb dataset is 'mteb:<TaskName>[/<subset>][@split]', got 'mteb:{uri}'")
        self.task_name = parts[0].strip()
        self.subset = parts[1].strip() if len(parts) > 1 else None
        self._name = name
        self._loaded: tuple[Any, str, str] | None = None

    # -- the task ----------------------------------------------------------
    def _load(self) -> tuple[Any, str, str]:
        """The loaded task, its subset and its split (once per reader; mteb caches nothing across tasks)."""
        if self._loaded is None:
            import mteb

            try:
                task = mteb.get_task(self.task_name)
            except (KeyError, ValueError) as exc:
                raise ConfigError(
                    f"the mteb task {self.task_name!r} could not be loaded: {exc}",
                    hint="a task is named as mteb names it (mteb.get_task); `mteb:<TaskName>[/<subset>][@split]`",
                ) from exc
            available = _subsets_of(task.metadata)
            if self.subset is None:
                if available != ["default"]:
                    raise ConfigError(
                        f"name the subset of the mteb task {self.task_name!r}",
                        hint=f"its subsets (mteb's hf_subsets): {available}",
                    )
                subset = "default"
            elif self.subset not in available:
                raise ConfigError(
                    f"the mteb task {self.task_name!r} has no subset {self.subset!r}",
                    hint=f"its subsets: {available}",
                )
            else:
                subset = self.subset
            declared = list(task.metadata.eval_splits)
            if self.split is not None and self.split not in declared:
                raise ConfigError(
                    f"the mteb task {self.task_name!r} declares the evaluation splits {declared}, not {self.split!r}",
                    hint="name a split the task evaluates (its TaskMetadata.eval_splits)",
                )
            split = self.split or declared[0]
            task = task.filter_eval_splits(eval_splits=[split])
            task = task.filter_languages(None, None, hf_subsets=[subset])
            task.load_data()
            # A v1-style loader fills `corpus`/`queries`/`relevant_docs` instead of `dataset`; mteb converts
            # those in `evaluate`, so the reader converts them too (the call is a no-op for a v2 loader).
            task.convert_v1_dataset_format_to_v2(num_proc=None)
            splits = sorted(task.dataset.get(subset, {}))  # type: ignore[union-attr,union-attr]
            if split not in splits:
                raise ConfigError(
                    f"the mteb task {self.task_name!r} has no split {split!r} for the subset {subset!r}",
                    hint=f"loaded splits: {splits}",
                )
            dataset_name = self._name or (subset if subset != "default" else task.metadata.name)
            self.dataset_name = dataset_name
            self._loaded = (task, subset, split)
            logger.info(f"loaded mteb task {task.metadata.name} ({subset}/{split}) from {task.metadata.dataset}")
        return self._loaded

    # -- the corpus shape --------------------------------------------------
    def documents(self) -> Iterator[Document]:
        """The task's corpus, as the task's loader built it (ids and texts as mteb serves them)."""
        task, subset, split = self._load()
        corpus = task.dataset[subset][split]["corpus"]  # type: ignore[index]
        for row in corpus:
            yield self._document(row)

    def _document(self, row: Mapping[str, Any]) -> Document:
        doc_id = str(row["id"])
        title = row.get("title")
        body = _text_of(row, what="a corpus row")
        media = self._media_of(row)
        if not media:
            title = title if isinstance(title, str) and title else None
            return Document(doc_id=doc_id, title=title, text=body)
        parts: list[TextPart | ImagePart | VideoPart] = [TextPart(text=body)] if body else []
        parts.extend(media)
        title = title if isinstance(title, str) and title else None
        return Document(doc_id=doc_id, title=title, content=Content.from_parts(parts))

    def queries(self) -> Iterator[Query]:
        """The task's queries (mteb's loader has already cut them to those with qrels); a per-query
        ``instruction`` column stays a field of its own, never merged into the text at load."""
        task, subset, split = self._load()
        queries = task.dataset[subset][split]["queries"]  # type: ignore[index]
        for row in queries:
            yield self._query(row)

    def _query(self, row: Mapping[str, Any]) -> Query:
        query_id = str(row["id"])
        text = _text_of(row, what="a query row")
        instruction = row.get("instruction")
        media = self._media_of(row)
        if not media:
            return Query(
                query_id=query_id,
                query=text,
                instruction=instruction if isinstance(instruction, str) and instruction else None,
            )
        parts: list[TextPart | ImagePart | VideoPart] = [TextPart(text=text)] if text else []
        parts.extend(media)
        return Query(
            query_id=query_id,
            content=Content.from_parts(parts),
            instruction=instruction if isinstance(instruction, str) and instruction else None,
        )

    def qrels(self) -> dict[ID, dict[ID, float]]:
        """``{query_id: {doc_id: grade}}``: mteb's ``relevant_docs`` (integer grades, cast to float)."""
        task, subset, split = self._load()
        relevant = task.dataset[subset][split]["relevant_docs"]  # type: ignore[index]
        return {
            query_id: {doc_id: float(grade) for doc_id, grade in judged.items()}
            for query_id, judged in relevant.items()
        }

    def candidates(self) -> dict[ID, list[ID]] | None:
        """Each query's pool in pool order (the task's ``top_ranked``), or ``None`` when the task has none."""
        task, subset, split = self._load()
        top_ranked = task.dataset[subset][split].get("top_ranked")  # type: ignore[index,union-attr]
        if top_ranked is None:
            return None
        return {query_id: [str(doc_id) for doc_id in docs] for query_id, docs in top_ranked.items()}

    # -- provenance --------------------------------------------------------
    def revision_payload(self) -> dict[str, Any]:
        """The identity payload of the task's pinned dataset revision: ``{"repo", "commit", "verified"}``.

        What :func:`rcp_ndcg.data.revisions.dataset_uri_revision` returns for an ``hf://`` URI, for a task
        whose repository only its metadata names (the task is read without loading its data).
        """
        import mteb

        task = mteb.get_task(self.task_name)
        dataset = task.metadata.dataset
        commit = dataset.get("revision")
        return {
            "repo": str(dataset.get("path") or task.metadata.name),
            "commit": commit,
            "verified": bool(commit),
        }

    @property
    def provenance(self) -> Provenance:
        """The task, subset and split read, and the pinned revision mteb's task metadata declares.

        ``duplicates`` is ``None``: mteb's own loader has already applied its rules upstream (a repeated pair
        keeps the last value, mteb 2.x), so there was nothing left for this reader's policy to decide.
        """
        task, subset, split = self._load()
        dataset = task.metadata.dataset
        return Provenance(
            source_uri=f"mteb:{task.metadata.name}/{subset}@{split}",
            revision=dataset.get("revision"),
            subset=subset,
            split=split,
        )

    @property
    def task(self) -> str | None:
        """The canonical mteb task name (the name the task registry knows, after any rename)."""
        return self._load()[0].metadata.name

    @property
    def task_instruction(self) -> str | dict[Literal["query", "document"], str] | None:
        """The task's ``TaskMetadata.prompt``: one instruction for the whole task, as a string or per side."""
        prompt = self._load()[0].metadata.prompt
        if prompt is None or isinstance(prompt, str):
            return prompt
        # mteb's PromptDict keys are exactly mteb's PromptType values ("query", "document").
        return cast("dict[Literal['query', 'document'], str]", dict(prompt))

    # -- helpers -----------------------------------------------------------
    def _media_of(self, row: Mapping[str, Any]) -> list[ImagePart | VideoPart]:
        """A row's media columns as content parts, persisted once, content-addressed."""
        parts: list[ImagePart | VideoPart] = []
        for column, part_type in (("image", ImagePart), ("video", VideoPart)):
            cell = row.get(column)
            if cell is None:
                continue
            cells = cell if isinstance(cell, list) else [cell]
            for value in cells:
                if value is None:
                    continue
                payload, extension, mime = _encode_media(value, column=column)
                width, height = getattr(value, "width", None), getattr(value, "height", None)
                if width is None and height is None and column == "image":
                    width, height = image_dimensions(payload)
                parts.append(part_type(ref=store_media(payload, extension, width=width, height=height, mime=mime)))
        return parts


def _subsets_of(metadata: Any) -> list[str]:
    """The subsets a task would load: the ``eval_langs`` keys of a multilingual task, else ``default``."""
    eval_langs = getattr(metadata, "eval_langs", None)
    if isinstance(eval_langs, Mapping):
        return sorted(eval_langs)
    return ["default"]


def _text_of(row: Mapping[str, Any], *, what: str) -> str:
    """A row's ``text`` column: a string; mteb's conversation form (a list of turns) is deferred."""
    value = row.get("text")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise DataError(
        f"a {what}'s 'text' is a {type(value).__name__}, not text",
        hint="mteb's conversation-style queries are not read yet (workstream 10 defers them): read the "
        "turns yourself, or wait for the conversation reader",
    )


__all__ = ["MtebTaskReader"]
