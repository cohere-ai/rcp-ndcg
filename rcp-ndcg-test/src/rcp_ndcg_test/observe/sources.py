"""The source catalogs the request generator samples from (OBSERVATIONS-SPEC section 1, sources).

Real items are sampled from the public suites **by id at pinned commits** (:data:`PINNED_DATASET_COMMITS`
in :mod:`rcp_ndcg_test.observe.requests`): NanoBEIR, BRIGHT (long documents), ViDoRe v3 (pages) and
TREC DL.  This module adapts the product's own dataset loader
(:func:`rcp_ndcg.data.dataset.load_dataset`) into small frozen catalogs the generator samples from --
the harness never re-implements a dataset reader (R30).

Public surface:

- :class:`SourceQuery`, :class:`SourceDoc`, :class:`SourceCorpus` — the catalog records.
- :func:`load_corpora` — load named sources through the product's dataset loader at pinned commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUITE_REPOS: dict[str, str] = {
    "nanobeir": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
    "bright": "fabianschmidt-cohere/rcp-ndcg-bright",
    "vidore": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
    "trecdl": "fabianschmidt-cohere/rcp-ndcg-trecdl",
}

SUITE_SUBSETS: dict[str, tuple[str, ...]] = {
    "nanobeir": tuple(
        f"Nano{name}Retrieval"
        for name in (
            "ArguAna",
            "ClimateFever",
            "DBPedia",
            "FEVER",
            "FiQA2018",
            "HotpotQA",
            "MSMARCO",
            "NFCorpus",
            "NQ",
            "Quora",
            "SCIDOCS",
            "SciFact",
            "Touche2020",
        )
    ),
    "bright": (
        "aops",
        "biology",
        "earth_science",
        "economics",
        "leetcode",
        "pony",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "theoremqa_questions",
        "theoremqa_theorems",
    ),
    "vidore": (
        "computer_science__english",
        "energy__french",
        "finance_en__english",
        "finance_fr__french",
        "hr__english",
        "industrial__english",
        "pharmaceuticals__english",
        "physics__french",
    ),
    "trecdl": ("trec_dl_2019", "trec_dl_2020"),
}

__all__ = [
    "SUITE_REPOS",
    "SUITE_SUBSETS",
    "SourceCorpus",
    "SourceDoc",
    "SourceMedia",
    "SourceQuery",
    "load_corpora",
]


@dataclass(frozen=True)
class SourceMedia:
    """One media asset a source document carries, addressed where it came from.

    The pairs file records media by **source coordinates** (suite, subset, document id, part) plus the
    content-addressed descriptors; the bytes are resolved through the product's dataset loader and
    :class:`~rcp_ndcg.data.media.MediaResolver` (never re-fetched by the harness).  A machine-local
    cache path is deliberately not recorded: it would not resolve on any other machine.

    Attributes:
        suite: The suite name (``vidore`` for the public pages).
        subset: The dataset subset the document comes from.
        doc_id: The document id in the subset's corpus.
        part: The part index of the image inside the document's content.
        sha256: The image bytes' SHA-256 (the content-addressed cache key).
        mime: The declared MIME type (``image/png`` and friends).
        num_bytes: The image bytes' size (bytes), when the source carries it.
    """

    suite: str
    subset: str
    doc_id: str
    part: int
    sha256: str | None
    mime: str | None
    num_bytes: int | None

    def to_json(self) -> dict[str, Any]:
        """This media entry as the pairs file's JSON object (deterministic key order)."""
        return {
            "suite": self.suite,
            "subset": self.subset,
            "doc_id": self.doc_id,
            "part": self.part,
            "sha256": self.sha256,
            "mime": self.mime,
            "num_bytes": self.num_bytes,
        }


@dataclass(frozen=True)
class SourceQuery:
    """One source query: its id, text and (BRIGHT) its instruction, with its pooled candidate ids."""

    query_id: str
    text: str
    instruction: str | None
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class SourceDoc:
    """One source document: its id, its text and its media (ViDoRe pages)."""

    doc_id: str
    text: str
    media: tuple[SourceMedia, ...] = ()


@dataclass(frozen=True)
class SourceCorpus:
    """One named dataset subset at its pinned commit, ready for deterministic sampling."""

    suite: str
    subset: str
    commit: str
    queries: dict[str, SourceQuery] = field(default_factory=dict)
    docs: dict[str, SourceDoc] = field(default_factory=dict)


def load_corpora(suite: str, subsets: tuple[str, ...], commits: dict[str, str]) -> list[SourceCorpus]:
    """Load the named subsets of one suite through the product's dataset loader, at the pinned commit.

    Inputs: the suite name (:data:`SUITE_REPOS`), the subset names and the pinned commits
    (:data:`~rcp_ndcg_test.observe.requests.PINNED_DATASET_COMMITS`).  Output: one
    :class:`SourceCorpus` per subset, sorted by subset name.  Needs network access to the Hub (or a
    populated cache); generation-time only -- tests build catalogs inline and never call this.
    """
    from rcp_ndcg.data.dataset import load_dataset

    pinned = commits[SUITE_REPOS[suite]]
    corpora: list[SourceCorpus] = []
    for subset in sorted(subsets):
        dataset = load_dataset(f"hf://{SUITE_REPOS[suite]}/{subset}@{pinned}")
        docs: dict[str, SourceDoc] = {}
        for doc_id, row in dataset.corpus.items():
            media = _media_of(suite, subset, doc_id, row)
            docs[doc_id] = SourceDoc(doc_id=doc_id, text=row.text, media=media)
        queries: dict[str, SourceQuery] = {}
        pool = dataset.candidates or {}
        for query_id, row in dataset.queries.items():
            queries[query_id] = SourceQuery(
                query_id=query_id,
                text=row.text,
                instruction=row.instruction,
                candidates=tuple(pool.get(query_id) or ()),
            )
        corpora.append(SourceCorpus(suite=suite, subset=subset, commit=pinned, queries=queries, docs=docs))
    return corpora


def _media_of(suite: str, subset: str, doc_id: str, row: Any) -> tuple[SourceMedia, ...]:
    """The document row's media as source coordinates (the resolver's cache path is never recorded)."""
    from rcp_ndcg_core.content import ImagePart

    out: list[SourceMedia] = []
    content = getattr(row, "content", None)
    if content is None:
        return ()
    part_index = 0
    for part in content.root:
        if isinstance(part, ImagePart):
            ref = part.ref
            out.append(
                SourceMedia(
                    suite=suite,
                    subset=subset,
                    doc_id=doc_id,
                    part=part_index,
                    sha256=ref.sha256,
                    mime=ref.mime,
                    num_bytes=ref.num_bytes,
                )
            )
        part_index += 1
    return tuple(out)
