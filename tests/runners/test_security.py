"""Injection is refused at the config boundary and at the render site; Kubernetes pods are hardened.

A newline in any config- or recipe-derived string used to break out of the job script's heredoc, of a
``#SBATCH`` directive or of a shell word built from config. ``$(...)`` and backticks are inert where they are
shell-quoted (``shlex``); a newline is not: a quoted value keeps it, and the heredoc reader is line-based.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners import JobPhase, JobSpec, KubernetesRunner, ServeConfig, SlurmRunner
from rcp_ndcg.runners.script import heredoc
from tests.runners.k8s_schema import check_objects

#: A value whose second line is the fixed heredoc terminator ``RCP_NDCG_WORKER``.
BREAKOUT = "x\nRCP_NDCG_WORKER\ntouch /tmp/pwned\n: '"
#: A newline alone: enough to end a ``#SBATCH`` directive line or a heredoc body line.
NEWLINE = "a\nb"


@pytest.mark.parametrize("value", [NEWLINE, "a\rb", "a\x00b", "\x1b[31m"])
def test_a_control_character_in_a_name_or_an_image_is_refused(value: str) -> None:
    with pytest.raises(ValidationError, match="control character"):
        JobSpec(name="j", argv=("true",), image=f"img:{value}")
    with pytest.raises(ValidationError, match="control character"):
        ServeConfig(image=f"vllm/vllm-openai:v0.31.0{value}", command=("vllm", "serve"))


@pytest.mark.parametrize("value", ["a\x00b"])
def test_a_nul_in_a_free_form_value_is_refused(value: str) -> None:
    """A NUL truncates the rendered script; a newline is allowed and quoted inert."""
    with pytest.raises(ValidationError, match="NUL"):
        JobSpec(name="j", argv=("rcp-ndcg", "run", "resume", value))
    with pytest.raises(ValidationError, match="NUL"):
        JobSpec(name="j", argv=("true",), env={"ODD": value})
    with pytest.raises(ValidationError, match="NUL"):
        JobPhase(argv=("true", value))
    with pytest.raises(ValidationError, match="NUL"):
        ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", value))
    with pytest.raises(ValidationError, match="NUL"):
        ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve"), env={"HF_HOME": value})


def test_a_newline_in_a_free_form_value_round_trips_inert(tmp_path: Path) -> None:
    """An argv word and an env value are quoted as one shell word: the newline stays inside it."""
    hostile = "line1\nline2\n$(touch pwned)"
    argv = ("python3", "-c", "import json, sys; print(json.dumps(sys.argv[1:]))", hostile)
    script = SlurmRunner(log_dir=str(tmp_path)).render_job(JobSpec(name="j", argv=argv, env={"ODD": hostile}))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [hostile]
    assert not (tmp_path / "pwned").exists()


def test_a_heredoc_body_holding_its_terminator_is_refused_at_the_render_site() -> None:
    """A value that forges the terminator line is refused at the render site, whatever the model boundary let
    through: the rest of the value would run as top-level script."""
    with pytest.raises(ConfigError, match="terminator|RCP_NDCG_WORKER"):
        heredoc("WORKER", "echo hi\nRCP_NDCG_WORKER\ntouch /tmp/pwned\n")
    with pytest.raises(ConfigError, match="terminator|RCP_NDCG_WORKER"):
        SlurmRunner(log_dir="logs").render_job(
            JobSpec(name="j", argv=("true",), env={"ODD": "x\nRCP_NDCG_WORKER\ntouch /tmp/pwned\n"})
        )
    # A body that merely mentions the tag inside a line is fine (the reader ends only on a line of its own).
    assert heredoc("WORKER", "echo RCP_NDCG_WORKER_1\n")[-1] == "RCP_NDCG_WORKER"


def test_a_dollar_or_backtick_value_round_trips_inert(tmp_path: Path) -> None:
    """``$(...)`` and backticks are not a heredoc escape: they stay inside the single quotes shlex gives them."""
    hostile = ["$(touch pwned)", "`touch pwned`", "a'b", '{"k": "$(touch pwned)"}']
    argv = ("python3", "-c", "import json, sys; print(json.dumps(sys.argv[1:]))", *hostile)
    script = SlurmRunner(log_dir=str(tmp_path)).render_job(JobSpec(name="j", argv=argv, env={"ODD": hostile[0]}))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == hostile
    assert not (tmp_path / "pwned").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"log_dir": NEWLINE},
        {"log_dir": "/tmp/my logs"},
        {"partition": NEWLINE},
        {"partition": "gpu big"},
        {"account": NEWLINE},
        {"qos": NEWLINE},
        {"sbatch_args": ["--gres=gpu:1\ntouch /tmp/pwned"]},
        {"sbatch_args": ["--gres=gpu:1 extra"]},
    ],
)
def test_sbatch_directive_values_are_refused_not_rendered(options: dict) -> None:
    """``#SBATCH`` is parsed whitespace-separated by sbatch, so quoting does not help: a space misdirects the
    directive and a newline ends it, turning the rest into a script line."""
    with pytest.raises(ConfigError, match="SBATCH|whitespace|control"):
        SlurmRunner(**options)


def test_a_rendered_sbatch_script_has_no_line_from_a_directive_value() -> None:
    """The refusal above is the boundary; the render site must not emit a payload either."""
    with pytest.raises(ConfigError):
        SlurmRunner(partition=BREAKOUT)


@pytest.mark.parametrize("namespace", ["Bad_NS", "a;touch x", "a\nb", "-lead", "trail-", "a" * 64, ""])
def test_a_namespace_that_is_not_a_dns_label_is_refused(namespace: str) -> None:
    with pytest.raises(ConfigError):
        KubernetesRunner(namespace=namespace)


def test_a_valid_namespace_and_its_engine_hosts_render_inert() -> None:
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    rendered = KubernetesRunner(namespace="eval", secrets=["hf-token"]).render([job])["j"]
    check_objects(list(yaml.safe_load_all(rendered)))
    assert "judge-engine" in rendered or "engine-judge" in rendered


def test_the_pod_is_hardened_by_default() -> None:
    """No service-account token, no privilege escalation, a RuntimeDefault seccomp profile and a non-root user:
    the security context the image's own default user cannot weaken."""
    pod = KubernetesRunner().manifest(JobSpec(name="j", argv=("true",)))["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}}
    assert pod["containers"][0]["securityContext"] == {"allowPrivilegeEscalation": False}


def test_the_engine_pods_are_hardened_too() -> None:
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    objects = KubernetesRunner().engine_objects(job)
    stateful_set = next(obj for obj in objects if obj["kind"] == "StatefulSet")
    pod = stateful_set["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}}
    assert pod["containers"][0]["securityContext"] == {"allowPrivilegeEscalation": False}


def test_an_image_that_needs_root_and_a_declared_token_opt_out() -> None:
    """``run_as_non_root: false`` is the declared opt-out for the stock vLLM image (its ``USER`` is root);
    ``automount_service_account_token: true`` mounts the token of a job that talks to the API server."""
    pod = KubernetesRunner(run_as_non_root=False, automount_service_account_token=True).manifest(
        JobSpec(name="j", argv=("true",))
    )["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is True
    assert pod["securityContext"]["runAsNonRoot"] is False
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}


def test_render_emits_applyable_objects_without_the_submit_uid_placeholder() -> None:
    """The rendered multi-document stream is what ``kubectl apply`` gets: an owner reference with the
    placeholder uid is an invalid object, so ``render`` leaves the reference to ``submit``."""
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    rendered = KubernetesRunner(namespace="eval").render([job])["j"]
    assert "<the Job's uid" not in rendered
    assert "blockOwnerDeletion" not in rendered
    check_objects(list(yaml.safe_load_all(rendered)))


def test_the_engine_image_must_be_pinned() -> None:
    """A floating tag (or none at all: an implicit ``latest``) runs whatever the registry serves at job time."""
    for image in ("vllm/vllm-openai:latest", "vllm/vllm-openai", "registry.example:5000/vllm"):
        with pytest.raises(ValidationError, match="pin|digest|tag"):
            ServeConfig(image=image, command=("vllm", "serve", "m"))
    for image in ("vllm/vllm-openai:v0.31.0", "registry.example:5000/vllm:v1", "repo@sha256:" + "0" * 64):
        assert ServeConfig(image=image, command=("vllm", "serve", "m")).image == image
