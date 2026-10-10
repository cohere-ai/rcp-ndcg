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
from rcp_ndcg.runs import RunConfig
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


@pytest.mark.parametrize(
    "options",
    [{"image": "img\n:x"}, {"context": "ctx\ntouch x"}, {"service_account": "sa\nx"}, {"secrets": ["sec\nret"]}],
)
def test_kubernetes_option_names_refuse_control_characters(options: dict) -> None:
    with pytest.raises(ConfigError, match="control character"):
        KubernetesRunner(**options)


@pytest.mark.parametrize("options", [{"image": "img\n:x"}, {"workdir": "/w\nx"}, {"container_mounts": ["/a\nb:/c"]}])
def test_slurm_option_values_refuse_control_characters(options: dict) -> None:
    with pytest.raises(ConfigError, match="control character"):
        SlurmRunner(**options)


@pytest.mark.parametrize(
    "argument", ["-o/tmp/x", "-Jpwn", "--out=/tmp/x", "--job-n=pwn", "--err=/tmp/x", "-o", "--output=x"]
)
def test_sbatch_args_may_not_take_over_the_rendered_directives(argument: str) -> None:
    """Slurm's getopt accepts attached short values and unambiguous long abbreviations, so the guard checks
    prefixes, not the exact flag: `run logs` resolves the output through ``log_dir``."""
    with pytest.raises(ConfigError, match="may not set"):
        SlurmRunner(log_dir="logs", sbatch_args=[argument])
    # An ordinary constraint still renders (the guard is not a blanket refusal of short or long options).
    assert SlurmRunner(sbatch_args=["--constraint=a100&h100"]).options.sbatch_args


def test_a_mirror_uri_with_a_control_character_is_refused() -> None:
    with pytest.raises(ValidationError, match="control character"):
        RunConfig.model_validate(
            {"dataset": "jsonl:rows.jsonl", "steps": ["evaluate"], "mirror": "s3://b/x\nRCP_NDCG_WORKER"}
        )


def test_a_valid_namespace_and_its_engine_hosts_render_inert() -> None:
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    rendered = KubernetesRunner(namespace="eval", secrets=["hf-token"]).render([job])["j"]
    check_objects(list(yaml.safe_load_all(rendered)))
    assert "judge-engine" in rendered or "engine-judge" in rendered


def test_engine_hosts_are_quoted_where_they_become_shell_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replicas' stable names become one shell word each in ``wait_for_replicas``; a hostile host must be
    inert. The namespace is DNS-validated, so the quoting is the render site's own guard."""
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    monkeypatch.setattr(KubernetesRunner, "_engine_hosts", lambda self, serve, role, job: ["h;touch /tmp/pwned"])
    rendered = KubernetesRunner().render([job])["j"]
    manifest = next(obj for obj in yaml.safe_load_all(rendered) if obj["kind"] == "Job")
    script = manifest["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID_REMOTE 1800 8000 /v1/models 'h;touch /tmp/pwned'" in script


def test_the_pod_is_hardened_by_default() -> None:
    """No service-account token, no privilege escalation and a RuntimeDefault seccomp profile, whatever the
    image; the non-root user is declared per image (the stock images run as root)."""
    pod = KubernetesRunner().manifest(JobSpec(name="j", argv=("true",)))["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {"runAsNonRoot": False, "seccompProfile": {"type": "RuntimeDefault"}}
    assert pod["containers"][0]["securityContext"] == {"allowPrivilegeEscalation": False}


def test_the_engine_pods_are_hardened_too() -> None:
    serve = ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), replicas=2)
    job = JobSpec(name="j", phases=(JobPhase(engines={"judge": serve}, argv=("true",)),))
    objects = KubernetesRunner(run_as_non_root=True).engine_objects(job)
    stateful_set = next(obj for obj in objects if obj["kind"] == "StatefulSet")
    pod = stateful_set["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}}
    assert pod["containers"][0]["securityContext"] == {"allowPrivilegeEscalation": False}


def test_an_image_that_allows_non_root_and_a_declared_token_opt_in() -> None:
    """``run_as_non_root: true`` is the declared opt-in for an image with a non-root ``USER`` (the stock vLLM
    image runs as root, and the kubelet refuses it with ``runAsNonRoot``);
    ``automount_service_account_token: true`` mounts the token of a job that talks to the API server (or to a
    store through the cluster's workload identity)."""
    pod = KubernetesRunner(run_as_non_root=True, automount_service_account_token=True).manifest(
        JobSpec(name="j", argv=("true",))
    )["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is True
    assert pod["securityContext"]["runAsNonRoot"] is True
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
