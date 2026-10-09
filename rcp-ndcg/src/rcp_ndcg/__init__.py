"""RCP-nDCG: calibrated relevance for retrieval evaluation.

The facade, ``import rcp_ndcg as rcp``::

    # data
    rcp.load_dataset(uri, *, subset=None, revision=None) -> Dataset
    rcp.load_rankings(path, *, format="auto", dataset=None) -> Rankings
    # evaluation (no LLM)
    rcp.evaluate(rankings, *, suite=None, dataset=None, gains=None, protocol=None, k=10, metrics=...) -> EvalReport
    rcp.compare(report, *, baseline=None, metric="rcp_ndcg", ...) -> Comparison
    # first stage and reranking
    rcp.retrieve(dataset, retriever, *, depth=150) -> Rankings
    rcp.rerank(dataset, rankings, reranker, *, depth=150) -> Rankings
    rcp.fuse(rankings, *, rrf_k=60, depth=150) -> Rankings
    # judging
    rcp.estimate(dataset, candidates, judge, *, stages=("tournament", "rubric")) -> CostEstimate
    rcp.judge(dataset, candidates, judge, *, stage, out, docs=None, schedule=None) -> JudgementSet
    # calibration and its primitives
    rcp.calibrate(judgements, *, mode="auto", judges="single") -> Calibration
    rcp.score_documents(calibration, judgements) -> Extension
    rcp.insert_documents(calibration, judgements, *, max_gain_shift=0.01) -> Extension
    # the pipeline
    rcp.run(config, *, runner=None, resume=True, estimate=False, runs_dir=None) -> Run | CostEstimate
    # the results export
    rcp.records_from_report(report, ...) -> list[ResultRecord]
    rcp.records_from_run(run_dir, ...) -> list[ResultRecord]
    # the two formulas
    rcp.ndcg, rcp.gain

Every name is re-exported from its one home module. Errors live in :mod:`rcp_ndcg.errors`.

Importing the package has no side effects: it reads no ``.env`` file, looks for no project root and creates no
directory. Paths come from arguments or the ``RCP_NDCG_*`` variables (:mod:`rcp_ndcg.support.paths`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("rcp-ndcg")
except PackageNotFoundError:  # a source tree on sys.path without an install
    __version__ = "0+unknown"

from rcp_ndcg_core.gain import Gains, gain  # noqa: E402
from rcp_ndcg_core.metric import ndcg  # noqa: E402
from rcp_ndcg_core.protocol import Protocol  # noqa: E402
from rcp_ndcg_core.schemas import JudgementSet  # noqa: E402

from rcp_ndcg.calibration import Calibration, Extension, calibrate, insert_documents, score_documents  # noqa: E402
from rcp_ndcg.data import Dataset, Rankings, load_dataset, load_rankings  # noqa: E402
from rcp_ndcg.data.text_policy import Preprocessing  # noqa: E402
from rcp_ndcg.eval import Comparison, EvalReport, compare, evaluate  # noqa: E402
from rcp_ndcg.inference.endpoint import Endpoint  # noqa: E402
from rcp_ndcg.judging import (  # noqa: E402
    CostEstimate,
    JudgeConfig,
    RubricSchedule,
    TournamentSchedule,
    estimate,
    judge,
)
from rcp_ndcg.results import (  # noqa: E402
    ResultArtifact,
    ResultDataset,
    ResultMetric,
    ResultRecord,
    ResultsSink,
    ResultSubject,
    records_from_report,
    records_from_run,
)
from rcp_ndcg.retrieval import RerankerConfig, RetrieverConfig, fuse, rerank, retrieve  # noqa: E402
from rcp_ndcg.runs import RunConfig  # noqa: E402
from rcp_ndcg.runs.execution import run  # noqa: E402
from rcp_ndcg.runs.run import Run  # noqa: E402

__all__ = [
    "Calibration",
    "Comparison",
    "CostEstimate",
    "Dataset",
    "Endpoint",
    "EvalReport",
    "Extension",
    "Gains",
    "JudgeConfig",
    "JudgementSet",
    "Preprocessing",
    "Protocol",
    "Rankings",
    "RerankerConfig",
    "ResultArtifact",
    "ResultDataset",
    "ResultMetric",
    "ResultRecord",
    "ResultSubject",
    "ResultsSink",
    "RetrieverConfig",
    "RubricSchedule",
    "Run",
    "RunConfig",
    "TournamentSchedule",
    "__version__",
    "calibrate",
    "compare",
    "estimate",
    "evaluate",
    "fuse",
    "gain",
    "insert_documents",
    "judge",
    "load_dataset",
    "load_rankings",
    "ndcg",
    "records_from_report",
    "records_from_run",
    "rerank",
    "retrieve",
    "run",
    "score_documents",
]
