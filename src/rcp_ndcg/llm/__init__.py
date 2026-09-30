"""LLM judging: the judge client, the schedules of the two stages, the one judging path, and cost.

* :class:`JudgeConfig`, :class:`JudgeClient` -- one OpenAI-compatible endpoint
  (:mod:`rcp_ndcg.llm.client`); ``JudgeConfig.fake(seed)`` is the offline judge;
* :class:`TournamentSchedule`, :class:`RubricSchedule` -- the window schedules, the
  paper's settings by default (:mod:`rcp_ndcg.llm.schedule`);
* :func:`judge` -- run a stage over candidate pools into an append-only judgement
  store (:mod:`rcp_ndcg.llm.judging`, :mod:`rcp_ndcg.llm.store`);
* :func:`reparse` -- read a store's stored answers again with the current parser, into a new store
  (:mod:`rcp_ndcg.llm.reparse`);
* :func:`estimate` -- calls, tokens, cost and wall time before judging (:mod:`rcp_ndcg.llm.cost`);
* :func:`load_prompt` -- the shipped prompts, by name (:mod:`rcp_ndcg.llm.prompts`).
"""

from rcp_ndcg.llm.client import JudgeClient, JudgeConfig, Usage
from rcp_ndcg.llm.cost import CostEstimate, Price, estimate
from rcp_ndcg.llm.judging import judge
from rcp_ndcg.llm.prompts import Prompt, load_prompt
from rcp_ndcg.llm.reparse import reparse
from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule
from rcp_ndcg.llm.store import JudgementStore

__all__ = [
    "CostEstimate",
    "JudgeClient",
    "JudgeConfig",
    "JudgementStore",
    "Price",
    "Prompt",
    "RubricSchedule",
    "TournamentSchedule",
    "Usage",
    "estimate",
    "judge",
    "load_prompt",
    "reparse",
]
