"""LLM judging: the judge client, the schedules of the two stages, the one judging path, and estimates.

* :class:`JudgeConfig`, :class:`JudgeClient` -- one OpenAI-compatible endpoint
  (:mod:`rcp.judging.client`); ``JudgeConfig.fake(seed)`` is the offline judge;
* :class:`TournamentSchedule`, :class:`RubricSchedule` -- the window schedules, the
  paper's settings by default (:mod:`rcp.judging.schedule`);
* :func:`judge` -- run a stage over candidate pools into an append-only judgement
  store (:mod:`rcp.judging.judging`, :mod:`rcp.judging.store`);
* :func:`check_judge` -- the conformance probe of an endpoint, before a long run
  (:mod:`rcp.judging.check`, ``rcp judge check``);
* :func:`reparse` -- read a store's stored answers again with the current parser, into a new store
  (:mod:`rcp.judging.reparse`);
* :func:`estimate` -- calls, tokens and wall time before judging (:mod:`rcp.judging.cost`);
* :func:`load_prompt` -- the shipped prompts, by name (:mod:`rcp.judging.prompts`).
"""

from rcp.judging.check import check_judge
from rcp.judging.client import JudgeClient, JudgeConfig, Usage
from rcp.judging.cost import CostEstimate, estimate
from rcp.judging.judging import judge
from rcp.judging.prompts import Prompt, load_prompt
from rcp.judging.reparse import reparse
from rcp.judging.schedule import RubricSchedule, TournamentSchedule
from rcp.judging.store import JudgementStore

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
