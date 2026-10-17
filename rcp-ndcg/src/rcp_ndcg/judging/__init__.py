"""LLM judging: the judge client, the schedules of the two stages, the one judging path, and estimates.

* :class:`JudgeConfig`, :class:`JudgeClient` -- one OpenAI-compatible endpoint
  (:mod:`rcp_ndcg.judging.client`); ``JudgeConfig.fake(seed)`` is the offline judge;
* :class:`TournamentSchedule`, :class:`RubricSchedule` -- the window schedules, the
  paper's settings by default (:mod:`rcp_ndcg.judging.schedule`);
* :func:`judge` -- run a stage over candidate pools into an append-only judgement
  store (:mod:`rcp_ndcg.judging.judging`, :mod:`rcp_ndcg.judging.store`);
* :func:`check_judge` -- the conformance probe of an endpoint, before a long run
  (:mod:`rcp_ndcg.judging.check`, ``rcp-ndcg judge check``);
* :func:`reparse` -- read a store's stored answers again with the current parser, into a new store
  (:mod:`rcp_ndcg.judging.reparse`);
* :func:`estimate` -- calls, tokens and wall time before judging (:mod:`rcp_ndcg.judging.cost`);
* :func:`load_prompt` -- the shipped prompts, by name (:mod:`rcp_ndcg.judging.prompts`).
"""

from rcp_ndcg.judging.check import check_judge
from rcp_ndcg.judging.client import JudgeClient, JudgeConfig, Usage
from rcp_ndcg.judging.cost import CostEstimate, estimate
from rcp_ndcg.judging.judging import judge
from rcp_ndcg.judging.prompts import Prompt, load_prompt
from rcp_ndcg.judging.reparse import reparse
from rcp_ndcg.judging.schedule import RubricSchedule, TournamentSchedule
from rcp_ndcg.judging.store import JudgementStore

__all__ = [
    "CostEstimate",
    "JudgeClient",
    "JudgeConfig",
    "JudgementStore",
    "Prompt",
    "RubricSchedule",
    "TournamentSchedule",
    "Usage",
    "check_judge",
    "estimate",
    "judge",
    "load_prompt",
    "reparse",
]
