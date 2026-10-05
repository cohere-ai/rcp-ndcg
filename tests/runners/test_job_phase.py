import pytest
from pydantic import ValidationError

from rcp_ndcg.runners.base import JobPhase, JobSpec
from rcp_ndcg.support.serve import ServeConfig

ENGINE = ServeConfig(command=("vllm", "serve", "m"))


def test_a_phase_names_its_engines_by_role() -> None:
    phase = JobPhase(engines={"encoder": ENGINE}, argv=("rcp-ndcg", "run", "resume"))
    assert phase.engines["encoder"] is ENGINE
    assert JobPhase(argv=("true",)).engines == {}


def test_a_phase_needs_a_command() -> None:
    with pytest.raises(ValidationError, match="argv must not be empty"):
        JobPhase(argv=())


def test_a_job_with_phases_sets_no_serve() -> None:
    phases = (JobPhase(engines={"judge": ENGINE}, argv=("a",)), JobPhase(argv=("b",)))
    assert JobSpec(name="run", argv=("b",), phases=phases).phases == phases
