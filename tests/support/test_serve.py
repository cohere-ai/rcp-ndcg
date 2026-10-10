"""The phase plan (:func:`rcp_ndcg.support.serve.plan_phases`) and the engines overlay it is planned from.

``plan_phases`` is a pure function of the steps, the engines and their use; every grouping below is the one
RFC-0001 5.2 option B asks for: consecutive steps that call the same served engines share a phase, steps that
call no served engine run in a phase without one, and the paper run becomes four phases.
"""

from __future__ import annotations

import json

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import (
    ENGINES_ENV,
    EngineConfig,
    EngineRole,
    EngineURLs,
    Phase,
    ServeByRole,
    ServeConfig,
    parse_engines_env,
    plan_phases,
)

ENGINE = EngineConfig(command=("vllm", "serve", "m"))

#: ``uses`` tables of the runs the table plans: per step, the engine roles the step calls.
_PAPER_USES = {
    "retrieve": frozenset[EngineRole](["encoder"]),
    "rerank": frozenset[EngineRole](["reranker"]),
    "tournament": frozenset[EngineRole](["judge"]),
    "rubric": frozenset[EngineRole](["judge"]),
}
STEPS = ("retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate")


class TestThePhasePlan:
    """One table per run shape; each row is one planned run."""

    def test_the_paper_run_gives_four_phases(self) -> None:
        serve = ServeByRole(encoder=ENGINE, reranker=ENGINE, judge=ENGINE)
        assert plan_phases(STEPS, serve, _PAPER_USES) == [
            Phase(engines=frozenset({"encoder"}), steps=("retrieve",)),
            Phase(engines=frozenset({"reranker"}), steps=("rerank",)),
            Phase(engines=frozenset({"judge"}), steps=("tournament", "rubric")),
            Phase(engines=frozenset(), steps=("calibrate", "evaluate")),
        ]

    def test_a_judge_only_run_gives_two_phases(self) -> None:
        uses = {step: roles for step, roles in _PAPER_USES.items() if "judge" in roles}
        assert plan_phases(STEPS[2:], ServeByRole(judge=ENGINE), uses) == [
            Phase(engines=frozenset({"judge"}), steps=("tournament", "rubric")),
            Phase(engines=frozenset(), steps=("calibrate", "evaluate")),
        ]

    def test_encoder_and_judge_without_a_reranker(self) -> None:
        serve = ServeByRole(encoder=ENGINE, judge=ENGINE)
        uses = {step: roles for step, roles in _PAPER_USES.items() if step != "rerank"}
        steps = (step for step in STEPS if step != "rerank")
        assert plan_phases(list(steps), serve, uses) == [
            Phase(engines=frozenset({"encoder"}), steps=("retrieve",)),
            Phase(engines=frozenset({"judge"}), steps=("tournament", "rubric")),
            Phase(engines=frozenset(), steps=("calibrate", "evaluate")),
        ]

    def test_a_hosted_encoder_formed_no_engine_phase(self) -> None:
        """A hosted encoder is reached at its vendor's API: the retrieve step runs with the rest, engine-free."""
        serve = ServeByRole(judge=ENGINE)
        assert plan_phases(list(STEPS), serve, _PAPER_USES) == [
            Phase(engines=frozenset(), steps=("retrieve", "rerank")),
            Phase(engines=frozenset({"judge"}), steps=("tournament", "rubric")),
            Phase(engines=frozenset(), steps=("calibrate", "evaluate")),
        ]

    def test_bm25_retrieval_uses_no_engine(self) -> None:
        uses = {step: roles for step, roles in _PAPER_USES.items() if step != "retrieve"}
        serve = ServeByRole(judge=ENGINE)
        steps = ("retrieve", "tournament", "rubric", "calibrate", "evaluate")
        assert plan_phases(list(steps), serve, uses) == [
            Phase(engines=frozenset(), steps=("retrieve",)),
            Phase(engines=frozenset({"judge"}), steps=("tournament", "rubric")),
            Phase(engines=frozenset(), steps=("calibrate", "evaluate")),
        ]

    def test_a_subset_of_steps_plans_a_subset_of_phases(self) -> None:
        """``--only`` narrows the steps the plan covers, like the run's ``--only``."""
        serve = ServeByRole(judge=ENGINE)
        assert plan_phases(["tournament"], serve, _PAPER_USES) == [
            Phase(engines=frozenset({"judge"}), steps=("tournament",))
        ]
        assert plan_phases(["rubric", "calibrate"], serve, _PAPER_USES) == [
            Phase(engines=frozenset({"judge"}), steps=("rubric",)),
            Phase(engines=frozenset(), steps=("calibrate",)),
        ]

    def test_a_role_the_step_uses_but_no_engine_serves_is_ignored(self) -> None:
        """Bring-your-own URLs: the step runs in the shared engine-free phase."""
        serve = ServeByRole(judge=ENGINE)
        assert plan_phases(["rerank", "tournament"], serve, _PAPER_USES) == [
            Phase(engines=frozenset(), steps=("rerank",)),
            Phase(engines=frozenset({"judge"}), steps=("tournament",)),
        ]

    def test_the_same_engines_merge_only_when_consecutive(self) -> None:
        serve = ServeByRole(judge=ENGINE)
        uses = {"a": frozenset({"judge"}), "b": frozenset(), "c": frozenset({"judge"})}
        assert plan_phases(["a", "b", "c"], serve, uses) == [
            Phase(engines=frozenset({"judge"}), steps=("a",)),
            Phase(engines=frozenset(), steps=("b",)),
            Phase(engines=frozenset({"judge"}), steps=("c",)),
        ]

    def test_no_steps_plan_no_phases(self) -> None:
        assert plan_phases([], ServeByRole(judge=ENGINE), _PAPER_USES) == []

    def test_a_step_absent_from_uses_is_engine_free(self) -> None:
        assert plan_phases(["calibrate"], ServeByRole(judge=ENGINE), {}) == [
            Phase(engines=frozenset(), steps=("calibrate",))
        ]

    def test_a_step_that_uses_two_roles_co_locates_their_engines(self) -> None:
        """A caller whose ``uses`` names two roles gets one phase with both; a run's own ``serve:`` names one
        role per step, so its phases hold at most one engine."""
        serve = ServeByRole(encoder=ENGINE, reranker=ENGINE)
        assert plan_phases(["retrieve"], serve, {"retrieve": frozenset({"encoder", "reranker"})}) == [
            Phase(engines=frozenset({"encoder", "reranker"}), steps=("retrieve",))
        ]


class TestTheEngineCommand:
    """The command's own device flags and port must agree with the fields the runner renders."""

    def test_a_tensor_parallel_size_must_match_the_declared_gpus(self) -> None:
        with pytest.raises(ValueError, match="resources.gpus"):
            ServeConfig(command=("vllm", "serve", "m", "--tensor-parallel-size", "4"), resources=Resources(gpus=1))
        # The world size is TP x DP, and the equals form is the same flag.
        assert (
            ServeConfig(
                command=("vllm", "serve", "m", "--tensor-parallel-size=4", "--data-parallel-size", "2"),
                resources=Resources(gpus=8),
            ).resources.gpus
            == 8
        )
        with pytest.raises(ValueError, match="8"):
            ServeConfig(command=("vllm", "serve", "m", "--data-parallel-size", "3"), resources=Resources(gpus=8))

    def test_a_port_must_match_the_declared_port(self) -> None:
        with pytest.raises(ValueError, match="--port"):
            ServeConfig(command=("vllm", "serve", "m", "--port", "9999"), port=8000)
        assert ServeConfig(command=("vllm", "serve", "m", "--port=8001"), port=8001).port == 8001

    def test_the_parallelism_aliases_and_pipeline_parallel_are_checked(self) -> None:
        """vLLM spells ``-tp``/``-dp``/``-pp`` and its world size includes pipeline parallelism: a check that
        reads only the long spellings and only TP x DP lets an under-reserved engine through."""
        with pytest.raises(ValueError, match="resources.gpus"):
            ServeConfig(command=("vllm", "serve", "m", "-tp", "4"), resources=Resources(gpus=1))
        with pytest.raises(ValueError, match="resources.gpus"):
            ServeConfig(
                command=("vllm", "serve", "m", "--tensor-parallel-size", "4", "--pipeline-parallel-size", "2"),
                resources=Resources(gpus=4),
            )
        assert (
            ServeConfig(
                command=("vllm", "serve", "m", "-tp", "4", "-pp", "2"), resources=Resources(gpus=8)
            ).resources.gpus
            == 8
        )

    def test_a_non_integer_or_non_positive_parallel_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not an integer"):
            ServeConfig(command=("vllm", "serve", "m", "--tensor-parallel-size", "0x4"), resources=Resources(gpus=1))
        with pytest.raises(ValueError, match="positive"):
            ServeConfig(command=("vllm", "serve", "m", "--tensor-parallel-size", "0"), resources=Resources(gpus=0))

    def test_a_command_without_the_flags_is_left_verbatim(self) -> None:
        engine = ServeConfig(command=("python3", "-m", "encoder", "--host", "0.0.0.0"))
        assert engine.resources.gpus == 0 and engine.port == 8000


class TestParseEnginesEnv:
    """``RCP_NDCG_ENGINES`` carries the current phase's engines to the coordinator."""

    def test_one_role_with_its_urls_and_outage_wait(self) -> None:
        engines = parse_engines_env(json.dumps({"judge": {"urls": ["http://n1:8000/v1"], "wait_on_outage_s": 900}}))
        assert engines == {"judge": EngineURLs(urls=("http://n1:8000/v1",), wait_on_outage_s=900.0)}

    def test_urls_are_trailing_slash_stripped(self) -> None:
        engines = parse_engines_env(json.dumps({"encoder": {"urls": ["http://n1:8000/v1/"]}}))
        assert engines["encoder"].urls == ("http://n1:8000/v1",)

    def test_an_empty_object_is_a_phase_without_engines(self) -> None:
        assert parse_engines_env("{}") == {}

    @pytest.mark.parametrize(
        "text, match",
        [
            ("not json", "not valid JSON"),
            ("[1]", "must be a JSON object"),
            ('{"gpu": {"urls": ["http://n:8000/v1"]}}', "unknown engine role"),
        ],
    )
    def test_bad_values_are_config_errors(self, text: str, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            parse_engines_env(text)

    def test_a_role_with_no_urls_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="urls"):
            parse_engines_env('{"judge": {"urls": []}}')

    def test_the_error_names_the_variable(self) -> None:
        with pytest.raises(ConfigError) as error:
            parse_engines_env("not json")
        assert error.value.details and error.value.details["variable"] == ENGINES_ENV
