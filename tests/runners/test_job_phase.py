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


def test_a_job_round_trips_its_phases() -> None:
    phases = (JobPhase(engines={"judge": ENGINE}, argv=("a",)), JobPhase(argv=("b",)))
    job = JobSpec(name="run", phases=phases)
    assert job.phases == phases and job.argv is None


def test_a_job_takes_exactly_one_of_argv_or_phases() -> None:
    """A job without phases runs ``argv``; a phased job's commands are its phases' ``argv``; both or neither
    are refused, naming which."""
    phases = (JobPhase(argv=("a",)),)
    with pytest.raises(ValidationError, match="a phased job takes no argv"):
        JobSpec(name="run", argv=("b",), phases=phases)
    with pytest.raises(ValidationError, match="a job takes argv, or phases"):
        JobSpec(name="run")
    with pytest.raises(ValidationError, match="argv must not be empty"):
        JobSpec(name="run", argv=())


def test_a_phased_job_runs_each_phase_s_argv_in_order(tmp_path) -> None:
    """A phased job has no command of its own: the local runner runs the phases' argv, in order."""
    from rcp_ndcg.runners import LocalRunner

    phases = (JobPhase(argv=("echo", "one")), JobPhase(argv=("echo", "two")))
    job = JobSpec(name="phased", phases=phases)
    script = LocalRunner(log_dir=str(tmp_path)).render([job])["phased"]
    assert "echo one" in script and "echo two" in script and script.index("echo one") < script.index("echo two")
