"""Offline helpers for examples and tests: a deterministic judge and a tiny judged world.

:class:`FakeJudge` stands in for an LLM endpoint (``JudgeConfig.fake(seed)`` builds
one): it reads the documents out of the real rendered prompt and answers in the
JSON the real parsers read, so judging, calibration and evaluation run offline
exactly as they do with a model. :func:`build_tiny_world` uses it to produce a
complete small example on disk: a calibrated fit, a re-judged subset, the
windows of an inserted document and a second, more lenient judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rcp_ndcg_core._records import RankingExample

from rcp_ndcg.llm._fake import DEFAULT_DIFFICULTIES, FakeJudge
from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule

if TYPE_CHECKING:
    from rcp_ndcg.data import Dataset, Rankings

#: The small schedules the tiny world is judged with: at its pools of ten documents, 4 random, 2 stratified and 1
#: adaptive tournament windows of 5, and 8 rubric windows of 5 (4 random).
TINY_TOURNAMENT = TournamentSchedule(
    window=5, random_placements=2.0, stratified_placements=1.0, adaptive_batches=1, adaptive_placements=1.0
)
TINY_RUBRIC = RubricSchedule(window=5, placements_per_doc=4.0, random_share=0.5)


@dataclass(frozen=True)
class TinyWorld:
    """Paths and facts of a :func:`build_tiny_world` directory.

    Attributes:
        root: The directory.
        dataset: The two queries with their candidates, qrels and document bodies (``jsonl:<root>/dataset.jsonl``,
            named ``dataset``).
        rankings: One system that ranks each query's candidates in pool order.
        abilities: The hidden ability (logits) of every document.
        judgements: The judgement store of the fit (both stages), plus the re-judged subset.
        calibration: The saved calibration.
        insertion: The judgement store of the inserted document's windows.
        lenient: The rubric judgement store of a second, more lenient judge.
        rejudged_docs: Documents of ``q2`` left out of the fit and judged again on the rubric alone.
        inserted_doc: The new document of ``q1``, judged in windows against documents of the fit.
    """

    root: Path
    dataset: Dataset
    rankings: Rankings
    abilities: dict[str, float]
    judgements: Path
    calibration: Path
    insertion: Path
    lenient: Path
    rejudged_docs: tuple[str, ...]
    inserted_doc: str


def tiny_rows() -> tuple[tuple[RankingExample, ...], dict[str, float]]:
    """Two queries (``q1``: ten documents and ``q1-new``; ``q2``: twelve) and each document's hidden ability.

    A document's text ends with its id, and its ability rises with its index.
    """
    abilities: dict[str, float] = {}
    rows = []
    for query, n_docs in (("q1", 10), ("q2", 12)):
        doc_ids = [f"{query}-d{index:02d}" for index in range(n_docs)]
        if query == "q1":
            doc_ids.append("q1-new")
        for index, doc_id in enumerate(doc_ids):
            abilities[doc_id] = 0.7 if doc_id == "q1-new" else -2.5 + 5.0 * index / (n_docs - 1)
        rows.append(
            RankingExample(
                query_id=query,
                query=f"tiny query {query}",
                doc_ids=doc_ids,
                docs=[f"tiny document {doc_id}" for doc_id in doc_ids],
                qrels={doc_id: int(abilities[doc_id] > 0.0) for doc_id in doc_ids},
            )
        )
    return tuple(rows), abilities


def build_tiny_world(root: str | Path, *, seed: int = 0) -> TinyWorld:
    """A small, fully judged example world on disk; deterministic for a given ``seed``.

    Two queries judged by a :class:`FakeJudge` on both stages and calibrated with
    the tournament. Around them, the three situations the extension primitives
    exist for: ``q2-d10``/``q2-d11`` were left out of the fit and are re-judged on
    the rubric alone; ``q1-new`` is a new candidate of ``q1``, judged against
    opponents the calibration picks; and a second, more lenient judge answered the
    rubric on the same pools.
    """
    from rcp_ndcg.calibration import calibrate, read_judgements, select_opponents
    from rcp_ndcg.calibration.fit import Calibration
    from rcp_ndcg.data import Rankings, load_dataset
    from rcp_ndcg.llm.judging import judge

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    examples, abilities = tiny_rows()
    rows = root / "dataset.jsonl"
    rows.write_text(
        "".join(
            json.dumps(
                {"query_id": ex.id, "query": ex.query, "doc_ids": ex.doc_ids, "docs": ex.docs, "qrels": ex.qrels}
            )
            + "\n"
            for ex in examples
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(f"jsonl:{rows}")
    rejudged, inserted = ("q2-d10", "q2-d11"), "q1-new"
    pools = {ex.id: [d for d in ex.doc_ids if d not in {*rejudged, inserted}] for ex in examples}

    def fake(**kwargs) -> FakeJudge:
        return FakeJudge(lambda text: abilities[text.split()[-1]], seed=seed, **kwargs)

    tournament = TINY_TOURNAMENT.model_copy(update={"seed": seed})
    rubric = TINY_RUBRIC.model_copy(update={"seed": seed})
    store = root / "judgements"
    judge(dataset, pools, fake(), stage="tournament", out=store, schedule=tournament)
    judge(dataset, pools, fake(), stage="rubric", out=store, schedule=rubric)
    calibration: Calibration = calibrate(read_judgements(store))
    calibration.save(root / "calibration")

    judge(dataset, None, fake(), stage="rubric", out=store, schedule=rubric, docs={"q2": list(rejudged)})
    (window,) = select_opponents(calibration, "q1", inserted, n=9)
    insertion = root / "insertion"
    judge(
        dataset,
        None,
        fake(),
        stage="tournament",
        out=insertion,
        docs={"q1": window},
        schedule=TournamentSchedule(
            window=10, random_placements=12.0, stratified_placements=0.0, adaptive_batches=0, seed=seed
        ),
    )
    lenient = root / "lenient"
    judge(dataset, pools, fake(name="fake-lenient", severity=-0.6), stage="rubric", out=lenient, schedule=rubric)
    return TinyWorld(
        root=root,
        dataset=dataset,
        rankings=Rankings.from_orders({ex.id: list(ex.doc_ids) for ex in examples}),
        abilities=abilities,
        judgements=store,
        calibration=root / "calibration",
        insertion=insertion,
        lenient=lenient,
        rejudged_docs=rejudged,
        inserted_doc=inserted,
    )


__all__ = [
    "DEFAULT_DIFFICULTIES",
    "TINY_RUBRIC",
    "TINY_TOURNAMENT",
    "FakeJudge",
    "TinyWorld",
    "build_tiny_world",
    "tiny_rows",
]
