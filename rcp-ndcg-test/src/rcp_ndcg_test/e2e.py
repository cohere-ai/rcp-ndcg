"""T4 end to end: the run scenarios of GPU-VALIDATION.md, driven inside the pod (not pytest).

One scenario is one declarative ``scenarios/<id>.yaml`` (the :class:`Scenario` schema below): the run
config the product runs (its dataset, pools, steps and judge), the engines the run starts by role, and
the scenario's own routine (``mode: run | outage | identity``).  The stage materializes the product's
:class:`~rcp_ndcg.runs.config.RunConfig` from it (R30: the product validates and the product computes the
role configs -- the recipe's ``client`` block IS the product's endpoint config), renders the run's phased
job script with the product's SLURM renderer (``container_runtime: none``: the engines run as processes of
the node, which in the pod is the one container), and runs that script in the pod::

    python -m rcp_ndcg_test.e2e --scenarios <dir | ids | @list> --recipes-root <dir> --out <dir> \
        --wheelhouse <dir> --constraints <file> --version <release>

What the driver adds around the product's rendering (nothing of the product is copied):

* **the client mechanism** -- each phase's coordinator command is wrapped with the product's own
  :func:`~rcp_ndcg.runners.script.install_argv` from the staged wheelhouse (node-runtime item 2: the
  ``uvx --find-links <wheelhouse> --no-index`` install source the runners' ``wheelhouse``/``constraints``
  options describe), so the coordinator never runs from the engine environment;
* **the srun shim** -- the SLURM renderer starts each engine step under ``srun``; the pod has no SLURM
  client, so the driver puts a tiny ``srun`` on ``PATH`` that runs the step's command in its own session
  and stops that session when it is stopped (the same semantics the root suite's supervision tests pin
  for their stub).  The engine's ``CUDA_VISIBLE_DEVICES``, ``VLLM_PORT`` and ``TMPDIR`` come from the
  scenario's ``serve`` blocks (node-runtime item 7), never from a dropped ``--gres`` flag;
* **the process-boundary probe** (node-runtime item 10) -- the phase workers export ``PYTHONPATH`` with a
  one-file ``sitecustomize`` that appends ``sys.prefix``, the executable and the versions each interpreter
  saw to ``client-probe.jsonl``, and the driver records the engine environment's ``python3`` beside it.
  After the run the coordinator's records and the engine record must name different environments and the
  coordinator's versions must be the staged release's.  Engines never see the shim: only the phase
  workers (the coordinators) get ``PYTHONPATH``.

Scenarios (``scenarios/``):

* ``text-four-phases`` -- NanoBEIR one subset, served encoder -> served reranker -> served judge
  (tournament + rubric) -> calibrate + evaluate: the four phases of one run.  Run twice into two run
  directories and compared (:func:`compare_runs`: identical identities and outputs; judgement values are
  counts and families only), judgements may differ at temperature > 0 -- see :func:`run_facts`);
* ``outage`` -- the same text run with the judge engine killed mid-tournament.  The judge is a run-scoped
  engine the driver owns (the product's model: engines the supervision script started end the job when
  they exit; engines that live elsewhere are "restarted by whatever runs them, and the judge bounds the
  outage instead"), which the driver restarts: one sub-run parks and recovers and finishes, a second
  outlives ``wait_on_outage_s`` and fails with ``BackendUnavailableError``, then a resume finishes it;
* ``identity`` -- rerun scenario 1's run with different engine ports; nothing recomputes (``base_url``
  and ``wait_on_outage_s`` are runtime fields and never reach an identity);
* ``vidore`` -- one ViDoRe v3 subset through a served VL encoder and a served VL judge, media budgets on.

The judge models and their revisions are the owner's (GPU-VALIDATION.md, "Judges for T4"): the four-phase
and visual runs take ``nvidia/Qwen3.8-Flash-Next-NVFP4`` when a T0 smoke proves vLLM v0.31.0 serves it on
this node, else their ``fallback`` (``Qwen/Qwen3.8-Flash-Next-FP8``); the outage and identity runs take
``Qwen/Qwen3.8-27B-FP8``.  Every scenario names its engines verbatim (``judge.command``: the user's
``vllm serve`` argv, ``{port}`` filled) -- the package never builds an engine's flags.

Public helpers:

- :func:`load_scenario`, :func:`iter_scenarios`, :func:`scenario_json_schema` -- the scenario configs.
- :func:`build_run_config` -- the product's ``RunConfig`` one scenario stands for.
- :func:`render_phased_script` -- the phased job script as the SLURM runner renders it (the golden file).
- :func:`run_scenario` -- drive one scenario in this pod and return its report document.
- :func:`run_facts`, :func:`compare_runs` -- what two identical runs are compared on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rcp_ndcg_vllm.recipe import Recipe, load_recipe, serve_argv

from rcp_ndcg.errors import RcpNdcgError
from rcp_ndcg.runners.script import bootstrap_uv, install_argv
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.execution import job_for, stage_run
from rcp_ndcg.runs.layout import RunLayout
from rcp_ndcg.runs.manifest import RunManifest, StepStatus
from rcp_ndcg.runs.run import prepare
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import ServeByRole, ServeConfig

from .errors import HarnessError, RecipeError

__all__ = [
    "EngineSlot",
    "JudgeEngine",
    "JudgeModel",
    "ManagedEngine",
    "Scenario",
    "ScenarioResult",
    "build_run_config",
    "compare_runs",
    "default_scenarios_root",
    "iter_scenarios",
    "load_scenario",
    "main",
    "render_phased_script",
    "run_facts",
    "run_scenario",
    "scenario_json_schema",
    "stage_run_dir",
]

#: The ``srun`` the pod runs instead of SLURM's: one step's command in its own session, its session stopped
#: when the step is stopped (the semantics the root suite's ``tests/runners/test_supervise.py`` pins).  The
#: flags SLURM gives the step (``--overlap``, ``--gres``, ``--kill-on-bad-exit``) are its scheduling; the
#: GPU partitioning they would imply comes from the scenario's ``serve`` env (``CUDA_VISIBLE_DEVICES``).
SRUN_SHIM = """\
#!/usr/bin/env bash
# The T4 driver's stand-in for srun in a pod without a SLURM client (rcp_ndcg_test.e2e).
while (($#)) && [[ "$1" == --* ]]; do shift; done
setsid "$@" &
step=$!
trap 'kill -TERM -- -"$step" 2>/dev/null; wait "$step" 2>/dev/null; exit 143' TERM
wait "$step"
"""

#: The coordinator's process-boundary probe (node-runtime item 10), as the ``sitecustomize`` the phase
#: workers put on ``PYTHONPATH``.  Every interpreter it starts records ``sys.prefix``, the executable and
#: the versions it sees; the driver reads the records as the coordinator's own witness.  Imports nothing
#: but the standard library (it runs before the entry point).
PROBE_SITECUSTOMISE = '''\
"""Record this interpreter's prefix and versions (rcp_ndcg_test.e2e's process-boundary probe)."""
import importlib.metadata as _metadata
import json as _json
import os as _os
import platform as _platform
import sys as _sys

_record = _os.environ.get("RCP_E2E_PROBE_JSONL")
if _record:
    _versions = {}
    for _dist in ("rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm", "pydantic", "numpy", "tokenizers"):
        try:
            _versions[_dist] = _metadata.version(_dist)
        except _metadata.PackageNotFoundError:
            _versions[_dist] = None
    _line = _json.dumps(
        {
            "pid": _os.getpid(),
            "argv": _sys.argv,
            "sys_prefix": _sys.prefix,
            "executable": _sys.executable,
            "python": _platform.python_version(),
            "platform": _platform.platform(),
            "versions": _versions,
        }
    )
    with open(_record, "a", encoding="utf-8") as _handle:
        _handle.write(_line + "\\n")
'''

#: Seconds between the outaged judge's rest starts (the outage scenario's ``sleep``s).
POLL_S = 1.0


def _no_extra() -> ConfigDict:
    """The common model config: frozen, unknown fields refused."""
    return ConfigDict(extra="forbid", frozen=True)


class EngineSlot(BaseModel):
    """One role's engine placement on the node (node-runtime item 7): port, internal port, GPU slice.

    Attributes:
        port: The HTTP port the engine serves on.
        vllm_port: The engine's internal ``VLLM_PORT`` (a second internal port vLLM opens), distinct per
            engine when several share a node.
        cuda_visible_devices: The device slice the engine process sees, as its ``CUDA_VISIBLE_DEVICES``.
            A phase runs one engine at a time (the phases reuse the GPUs), so the slices may overlap; they
            may not be left unset -- a node grants every device to a process that sees them all.
        gpus: The GPUs the engine occupies.  For a recipe engine it equals the recipe's own
            ``resources.gpus`` (the recipe renders its ``--tensor-parallel-size``; the slot sizes the
            driver's ``--gres`` and the device slice around it, checked in :func:`build_serve`); for a
            judge command it matches the command's ``--tensor-parallel-size``.
        startup_timeout_s: Seconds the script waits for the engine's readiness path.
        outage_timeout_s: Seconds a step waits while the engine is down before
            :class:`~rcp_ndcg.errors.BackendUnavailableError` (the role config's ``wait_on_outage_s``).
    """

    model_config = _no_extra()

    port: int = Field(ge=1, le=65535)
    vllm_port: int = Field(default=8100, ge=1, le=65535)
    cuda_visible_devices: str = Field(min_length=1)
    gpus: int = Field(default=1, ge=1)
    startup_timeout_s: int = Field(default=3600, gt=0)
    outage_timeout_s: int = Field(default=300, ge=0)

    def env(self, scratch: Path) -> dict[str, str]:
        """The engine process's environment: its device slice, its internal ports and its own ``TMPDIR``."""
        return {
            "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            "VLLM_PORT": str(self.vllm_port),
            "TMPDIR": str(scratch / f"tmp-{self.port}"),
        }


class JudgeModel(BaseModel):
    """A judge checkpoint to serve: the revision the run pins.

    ``tokenizer`` is the judge config's tokenizer **when this checkpoint serves** (a quantized release of
    one base shares its tokenizer files; name it explicitly when it does not) -- the fallback partner's
    tokenizer must follow the winner, since the tokenizer's SHA-256 enters the judgement family.
    """

    model_config = _no_extra()

    model: str = Field(min_length=1, description="the Hub repository id")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="the commit served, 40 hex")
    tokenizer: str | None = None


class JudgeEngine(BaseModel):
    """The judge engine and the judge config of a scenario.

    Attributes:
        served_name: The ``--served-model-name`` the command serves and the judge config's ``model``.
        command: The ``vllm serve`` argv that starts one judge replica, verbatim; every ``{port}`` is
            filled with the slot's port.  The package never builds engine flags (this is the user's
            command).
        candidate: The checkpoint the command serves (its id and pinned revision).
        fallback: The command and checkpoint used when the T0 smoke of ``candidate`` fails on this node
            (GPU-VALIDATION.md, "Judges for T4": the NVFP4 candidate falls back to FP8).
        fallback_command: The ``vllm serve`` argv of ``fallback``.
        config: The product's :class:`~rcp_ndcg.judging.client.JudgeConfig` fields, minus ``model`` (the
            driver fills ``served_name``), ``revision`` (filled from the served candidate) and ``base_url``
            (the slot's URL).  ``wait_on_outage_s`` here bounds the outage scenario's expiry path.
        slot: The engine's placement.
    """

    model_config = _no_extra()

    served_name: str = Field(default="judge", min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    candidate: JudgeModel
    fallback: JudgeModel | None = None
    fallback_command: tuple[str, ...] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    slot: EngineSlot

    @model_validator(mode="after")
    def _a_fallback_is_a_pair(self) -> JudgeEngine:
        if (self.fallback is None) != (self.fallback_command is None):
            raise ValueError("a fallback judge gives both `fallback` (the checkpoint) and `fallback_command`")
        return self

    def render_command(self, command: Sequence[str], *, port: int) -> list[str]:
        """``command`` with every ``{port}`` replaced by the port to serve on."""
        return [part.replace("{port}", str(port)) for part in command]


class Scenario(BaseModel):
    """One T4 end-to-end scenario: the run the product runs, its engines, and the routine around it.

    Attributes:
        id: The scenario identifier, equal to the file name.
        description: What the scenario proves (it goes into the report).
        mode: ``run`` (a plain phased run; ``pair`` identical runs are compared), ``outage`` (the judge
            engine is killed mid-tournament: parks and recovers once, the ``wait_on_outage_s`` expiry path
            once, then a resume finishes the failed run), or ``identity`` (the run, then the same run
            directory again with ``identity_port_offset`` added to every engine port: nothing recomputes).
        pair: In ``run`` mode, how many identical runs into separate run directories (2 compares them on
            identities and outputs).
        identity_port_offset: In ``identity`` mode, what the rerun adds to every engine port.
        dataset: The dataset (``uri``, ``subset``, ``revision``), as the run config's.
        limit: Judge only the first ``limit`` queries.
        depth: Candidates per query kept for judging.
        steps: The run's steps, in run order.
        encoder_recipe: The recipe of the retrieval encoder (``None``: no retrieval step).
        rerank_recipe: The recipe of the reranker (``None``: no rerank step).
        judge: The judge engine and config.
        slots: The encoder's and reranker's engine placements (the judge's is its own ``slot``).
        tournament, rubric, calibration, evaluation, preprocessing, seed: Passed to the run config as
            they are (the product validates each).
        runner_options: The run config's ``runner.options`` on top of the driver's (a ``time_limit_s``).
    """

    model_config = _no_extra()

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    description: str = Field(min_length=1)
    mode: Literal["run", "outage", "identity"] = "run"
    pair: int = Field(default=1, ge=1, le=2)
    identity_port_offset: int = Field(default=0, ge=0)
    dataset: dict[str, Any]
    limit: int | None = Field(default=None, ge=1)
    depth: int = Field(default=30, ge=1)
    steps: tuple[Literal["retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate"], ...]
    encoder_recipe: str | None = None
    rerank_recipe: str | None = None
    judge: JudgeEngine
    slots: dict[str, EngineSlot] = Field(default_factory=dict)
    tournament: dict[str, Any] | None = None
    rubric: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    preprocessing: dict[str, Any] | None = None
    seed: int | None = None
    runner_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("slots")
    @classmethod
    def _known_slots(cls, value: dict[str, EngineSlot]) -> dict[str, EngineSlot]:
        unknown = sorted(set(value) - {"encoder", "reranker"})
        if unknown:
            raise ValueError(f"slots: unknown role(s) {unknown}; a slot is declared for encoder or reranker")
        return value

    @model_validator(mode="after")
    def _steps_have_their_engines(self) -> Scenario:
        for step, recipe in (("retrieve", self.encoder_recipe), ("rerank", self.rerank_recipe)):
            if (step in self.steps) != (recipe is not None):
                raise ValueError(f"steps list {step!r} {step in self.steps} but {step}_recipe is {recipe!r}")
        if "encoder" in self.slots and self.encoder_recipe is None:
            raise ValueError("slots.encoder: this scenario retrieves with no served encoder")
        if "reranker" in self.slots and self.rerank_recipe is None:
            raise ValueError("slots.reranker: this scenario reranks with no served reranker")
        if self.mode == "identity" and self.identity_port_offset == 0:
            raise ValueError("an identity rerun needs identity_port_offset: the rerun's ports differ")
        if "retrieve" in self.steps and self.encoder_recipe is not None and "encoder" not in self.slots:
            raise ValueError("steps.retrieve: declare the encoder's slot (ports and devices live there)")
        if "rerank" in self.steps and self.rerank_recipe is not None and "reranker" not in self.slots:
            raise ValueError("steps.rerank: declare the reranker's slot (ports and devices live there)")
        seen_ports: dict[int, str] = {}
        seen_internal: dict[int, str] = {}
        for role, slot in {"judge": self.judge.slot, **self.slots}.items():
            if slot.port in seen_ports:
                raise ValueError(
                    f"the {role} and {seen_ports[slot.port]} slots share the port {slot.port}: one engine per "
                    "phase still needs one port per role"
                )
            if slot.vllm_port in seen_internal:
                raise ValueError(
                    f"the {role} and {seen_internal[slot.vllm_port]} slots share the vllm_port "
                    f"{slot.vllm_port}: each engine opens its own internal port"
                )
            seen_ports[slot.port] = role
            seen_internal[slot.vllm_port] = role
        return self


class ScenarioResult(BaseModel):
    """One scenario's report row: the runs it made, every check and what failed."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    description: str = ""
    mode: str
    status: Literal["passed", "failed"]
    runs: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    judge: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def default_scenarios_root() -> Path:
    """The package checkout's own ``scenarios/`` directory (where the scenario lanes write).

    Like the recipes, scenario configs are staged to the node and named with ``--scenarios-root``; they
    are not inside the wheel (the wheel holds code, the stage holds the scenarios' data).
    """
    return Path(__file__).resolve().parents[2] / "scenarios"


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate one scenario from a YAML file.

    Inputs: ``path`` to the YAML.  Outputs: a frozen :class:`Scenario`.  Raises :class:`RecipeError`
    (with the file path and the validator message) when the YAML does not satisfy the schema, or when
    ``id`` differs from the file stem.
    """
    path = Path(path)
    if not path.is_file():
        raise RecipeError(f"no scenario at {path}: expected {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RecipeError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise RecipeError(f"{path} must contain a YAML mapping of the Scenario schema, got {type(data).__name__}")
    try:
        scenario = Scenario.model_validate(data)
    except Exception as error:
        raise RecipeError(f"{path}: {error}") from error
    if path.stem != scenario.id:
        raise RecipeError(f"{path}: id {scenario.id!r} must equal the file name {path.stem!r}")
    return scenario


def iter_scenarios(root: str | Path | None = None) -> list[Scenario]:
    """Load every scenario under ``root`` (default: the checkout's ``scenarios/``), sorted by id."""
    root = Path(root) if root is not None else default_scenarios_root()
    if not root.is_dir():
        raise HarnessError(f"no scenarios root at {root}")
    return [load_scenario(path) for path in sorted(root.glob("*.yaml"))]


def scenario_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Scenario` (exported to ``schema/scenario.schema.json``, kept by a test)."""
    return Scenario.model_json_schema()


def find_recipe(roots: Sequence[Path], recipe_id: str) -> Recipe:
    """The recipe ``recipe_id``, looked up in each root in order.

    Raises:
        RecipeError: no root holds it (the message names the roots).
    """
    for root in roots:
        directory = root / recipe_id
        if (directory / "recipe.yaml").is_file():
            return load_recipe(directory)
    raise RecipeError(f"no recipe {recipe_id!r} under {[str(root) for root in roots]}")


def _client_without_base_url(recipe_id: str, roots: Sequence[Path]) -> dict[str, Any]:
    """The recipe's product endpoint config dict, without ``base_url``: the job's engine sets it at
    runtime (a served role's config naming a URL is refused by the run config's validator)."""
    recipe = find_recipe(roots, recipe_id)
    data = {key: value for key, value in recipe.client.items() if key != "base_url"}
    data.setdefault("recipe", recipe.id)
    return data


def build_serve(
    scenario: Scenario,
    *,
    recipes_root: Sequence[Path],
    port_offset: int = 0,
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None = None,
) -> ServeByRole:
    """The engines this scenario's job starts by role (role configs at their runtime URLs).

    The encoder's and reranker's commands are the recipes' :func:`~rcp_ndcg_vllm.recipe.serve_argv`; the
    judge's is its ``command`` with the slot's port filled.  Each engine gets its own ``CUDA_VISIBLE_DEVICES``,
    ``VLLM_PORT`` and ``TMPDIR`` (node-runtime item 7).  ``mode: outage`` starts no judge engine: the
    driver owns the judge (run-scoped, restarted by whatever runs it -- see :class:`ManagedEngine`).

    Inputs: the scenario, the recipe roots, ``port_offset`` (the identity rerun's shift), and the judge the
    T0 smoke chose (``(checkpoint, command)``; default the candidate).  Output: the
    :class:`~rcp_ndcg.support.serve.ServeByRole` the run config is materialized with.
    """
    scratch = Path(f"/tmp/rcp-e2e-{scenario.id}")
    engines: dict[str, ServeConfig] = {}

    def shifted(slot: EngineSlot) -> EngineSlot:
        # The rerun's ports move together: the command's --port, ServeConfig.port (the readiness wait and
        # the RCP_NDCG_ENGINES URLs), VLLM_PORT and the per-slot TMPDIR are one story, and a script whose
        # engines listen elsewhere than it waits on can never pass.
        return slot.model_copy(update={"port": slot.port + port_offset, "vllm_port": slot.vllm_port + port_offset})

    if scenario.encoder_recipe is not None and "encoder" in scenario.slots:
        slot = shifted(scenario.slots["encoder"])
        recipe = find_recipe(recipes_root, scenario.encoder_recipe)
        _check_slot_gpus(scenario.encoder_recipe, slot, recipe)
        engines["encoder"] = _serve_config(
            serve_argv(recipe, port=slot.port, served_model_name=recipe.id), slot, scratch
        )
    if scenario.rerank_recipe is not None and "reranker" in scenario.slots:
        slot = shifted(scenario.slots["reranker"])
        recipe = find_recipe(recipes_root, scenario.rerank_recipe)
        _check_slot_gpus(scenario.rerank_recipe, slot, recipe)
        engines["reranker"] = _serve_config(
            serve_argv(recipe, port=slot.port, served_model_name=recipe.id), slot, scratch
        )
    if scenario.mode != "outage":
        _, judge_command = judge_choice or (scenario.judge.candidate, scenario.judge.command)
        slot = shifted(scenario.judge.slot)
        engines["judge"] = _serve_config(
            scenario.judge.render_command(judge_command, port=slot.port),
            slot,
            scratch,
        )
    return ServeByRole.model_validate(engines)


def _check_slot_gpus(recipe_id: str, slot: EngineSlot, recipe: Recipe) -> None:
    """The slot and the recipe name the same devices: the recipe renders ``--tensor-parallel-size`` from
    its own ``resources.gpus``, and the slot sizes the driver's ``--gres`` and ``CUDA_VISIBLE_DEVICES``
    around it -- two numbers here is one incoherent engine."""
    if slot.gpus != recipe.resources.gpus:
        raise HarnessError(
            f"slot for {recipe_id}: gpus {slot.gpus} but the recipe serves on {recipe.resources.gpus} "
            "(set the slot's gpus to the recipe's, so --gres matches --tensor-parallel-size)"
        )


def _serve_config(command: Sequence[str], slot: EngineSlot, scratch: Path) -> ServeConfig:
    """One engine's :class:`~rcp_ndcg.support.serve.ServeConfig` on the node (no image: the command runs
    as a process of the node, which is the pod's one container)."""
    return ServeConfig(
        command=tuple(command),
        env=slot.env(scratch),
        resources=Resources(gpus=slot.gpus),
        port=slot.port,
        readiness_path="/v1/models",
        startup_timeout_s=slot.startup_timeout_s,
        outage_timeout_s=slot.outage_timeout_s,
    )


def build_run_config(
    scenario: Scenario,
    *,
    recipes_root: Sequence[Path],
    port_offset: int = 0,
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None = None,
) -> RunConfig:
    """The product's run config this scenario stands for.

    Inputs: the scenario, the recipe roots, ``port_offset``, and the judge the T0 smoke chose (its pinned
    checkpoint enters the judgement family).  Output: the validated product
    :class:`~rcp_ndcg.runs.config.RunConfig` -- the role configs are the recipes' product endpoint dumps,
    the judge is the product's :class:`~rcp_ndcg.judging.client.JudgeConfig` at the slot's URL, and the
    ``serve:`` block is built by :func:`build_serve`.  Raises the product's
    :class:`~rcp_ndcg.errors.ConfigError` for a scenario whose run would be refused (the product's
    messages).  In ``outage`` mode the judge config's URL is the driver-managed judge's (no
    ``serve.judge``: the supervision script must not watch an engine the driver restarts).
    """
    from rcp_ndcg.judging.client import JudgeConfig

    judge_slot = scenario.judge.slot
    judge_model, _ = judge_choice or (scenario.judge.candidate, scenario.judge.command)
    judge_data = {
        **scenario.judge.config,
        "model": scenario.judge.served_name,
        "revision": judge_model.revision,
        "base_url": f"http://127.0.0.1:{judge_slot.port + port_offset}/v1",
    }
    if judge_model.tokenizer is not None:  # the winner's tokenizer, never the loser's
        judge_data["tokenizer"] = judge_model.tokenizer
    judge = JudgeConfig.model_validate(judge_data)
    data: dict[str, Any] = {
        "label": scenario.id,
        "dataset": dict(scenario.dataset),
        "candidates": {
            "from": "retrieval" if scenario.encoder_recipe is not None else "dataset",
            "depth": scenario.depth,
        },
        "judge": judge.model_dump(mode="json", exclude_none=False),
        "steps": list(scenario.steps),
    }
    if scenario.limit is not None:
        data["limit"] = scenario.limit
    if scenario.encoder_recipe is not None:
        encoder = _client_without_base_url(scenario.encoder_recipe, recipes_root)
        # The api picks the kind (a multi-vector recipe speaks ``vllm_pooling``); the product re-validates.
        kind = "late_interaction" if encoder.get("api") == "vllm_pooling" else "dense"
        data["candidates"]["retrieval"] = {"kind": kind, "encoder": encoder}
    if scenario.rerank_recipe is not None:
        data["candidates"]["rerank"] = _client_without_base_url(scenario.rerank_recipe, recipes_root)
    serve = build_serve(scenario, recipes_root=recipes_root, port_offset=port_offset, judge_choice=judge_choice)
    data["serve"] = {
        role: engine.model_dump(mode="json")
        for role in ("judge", "encoder", "reranker")
        if (engine := getattr(serve, role)) is not None
    }
    if not data["serve"]:
        del data["serve"]
    for name in ("tournament", "rubric", "calibration", "evaluation", "preprocessing"):
        value = getattr(scenario, name)
        if value is not None:
            data[name] = value
    if scenario.seed is not None:
        data["seed"] = scenario.seed
    data["runner"] = {"name": "slurm", "options": {"container_runtime": "none", **scenario.runner_options}}
    return RunConfig.from_data(data)


def stage_run_dir(pipeline: Any) -> None:
    """Create the run directory ``pipeline`` will run, as :func:`~rcp_ndcg.runs.execution.submit_run`
    stages it before it hands a job to a runner.

    The driver runs the job itself (the phased script, in this pod) instead of submitting to a scheduler,
    but the job re-enters the product's ``run resume`` on the same run directory, so the directory must be
    staged exactly as the product stages one: the layout, the product's own ``run.yaml`` writer and the
    manifest in status ``submitted``.
    """
    stage_run(pipeline)  # the product's one staging step (R30: consume, never copy)


def render_phased_script(
    pipeline: Any,
    *,
    options: Mapping[str, Any] | None = None,
    wheelhouse: str,
    constraints: str | None = None,
    version: str | None = None,
    run_dir: str | Path | None = None,
) -> str:
    """The run's phased job script as the product's SLURM runner renders it (``container_runtime: none``).

    Inputs: a prepared pipeline (its ``serve:`` plan), the runner options (``log_dir``, ``workdir``, the
    coordinator's ``resources`` and the probe's ``env``), and the install source: the staged wheelhouse
    and constraints file every phase's coordinator installs the release from through
    :func:`~rcp_ndcg.runners.script.install_argv` (node-runtime item 2; the pod's engine environment is
    never asked).  ``run_dir`` points the phases at another run directory than the pipeline's own (the
    identity rerun drives an existing run again, on new ports).  Output: the rendered script (one
    supervision block per phase).  This exact text is the golden file the tests pin and the byte stream
    the driver executes.

    Raises:
        ConfigError: the runner's options or the job's fields do not validate (the product's message).
    """
    backend, job, _ = job_for(pipeline, "slurm", dict(options or {}))

    def repoint(argv: tuple[str, ...]) -> tuple[str, ...]:
        if run_dir is not None and "--run" in argv:
            at = argv.index("--run")
            return (*argv[: at + 1], str(run_dir), *argv[at + 2 :])
        return argv

    def wrap(argv: tuple[str, ...]) -> tuple[str, ...]:
        return install_argv(repoint(argv), version=version, wheelhouse=wheelhouse, constraints=constraints)

    wrapped = job.model_copy(
        update={"phases": tuple(phase.model_copy(update={"argv": wrap(phase.argv)}) for phase in job.phases)}
    )
    render = getattr(backend, "render", None)  # the slurm runner's own (the product's render_run pattern)
    if render is None:
        raise HarnessError("the slurm runner cannot render what it would submit")
    return render([wrapped])[wrapped.name]


def probe_paths(out: Path) -> tuple[Path, Path]:
    """The process-boundary probe's files' paths (pure): the ``sitecustomize`` directory and the JSONL it
    appends to.  The directory is what the phase workers put on ``PYTHONPATH``; their coordinators record
    (one JSON line per interpreter that starts with it on the path)."""
    return out / "probe-site", out / "client-probe.jsonl"


def write_probe(out: Path) -> tuple[Path, Path]:
    """The probe's files, written (:func:`probe_paths` is the pure path computation)."""
    site, probe_jsonl = probe_paths(out)
    site.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(PROBE_SITECUSTOMISE, encoding="utf-8")
    return site, probe_jsonl


def write_srun_shim(out: Path) -> Path:
    """The ``srun`` shim directory for the pod (see :data:`SRUN_SHIM`): put it first on the job's ``PATH``."""
    bin_dir = out / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "srun"
    shim.write_text(SRUN_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    return bin_dir


def job_env(out: Path, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """One launcher environment for everything the driver runs (the job script and the managed engines):
    the driver's ``srun`` shim first on the path, and no probe ``PYTHONPATH`` -- the probe belongs to the
    phase workers only (node-runtime item 10), so an engine process never imports the probe's
    ``sitecustomize``."""
    merged = {**os.environ, **dict(env or {})}
    merged.pop("PYTHONPATH", None)
    merged["PATH"] = f"{write_srun_shim(out)}:{merged.get('PATH', '')}"
    return merged


def launch_job_script(script: str, *, out: Path, env: Mapping[str, str] | None = None) -> subprocess.Popen[bytes]:
    """Run the rendered phased job script in this pod: ``bash`` with the driver's ``srun`` on ``PATH``
    (:func:`job_env`).  The script's phase workers export the probe environment themselves.

    Inputs: the rendered script, the scenario's output directory (``job.sh``, ``job.log`` and the shim
    land there) and extra environment.  Output: the script's process (its ``job.log`` captures its
    output); :func:`run_job_script` waits for it, the outage driver watches it.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.sh").write_text(script, encoding="utf-8")
    log = open(out / "job.log", "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the job
    log.write(f"\n===== run job.sh at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n")
    log.flush()
    return subprocess.Popen(["bash", str(out / "job.sh")], env=job_env(out, env), stdout=log, stderr=subprocess.STDOUT)


def run_job_script(script: str, *, out: Path, env: Mapping[str, str] | None = None) -> tuple[int, Path]:
    """Run the rendered phased job script and wait: ``(exit status, job.log path)``.

    See :func:`launch_job_script` for the launcher (the srun shim and the probe hygiene).
    """
    process = launch_job_script(script, out=out, env=env)
    return process.wait(), out / "job.log"


def resume_run(run_dir: Path, *, wheelhouse: str, constraints: str | None, version: str | None, out: Path) -> int:
    """Resume a run directory through the client mechanism (what a submitted job would run): the
    product's ``run resume`` wrapped with :func:`~rcp_ndcg.runners.script.install_argv`, with the
    launcher's hygiene (:func:`job_env`: the driver's ``PYTHONPATH`` out) and the probe in (the resumed
    coordinator records like the script's ones)."""
    site, probe_jsonl = write_probe(out)
    env = job_env(out)
    env["PYTHONPATH"] = str(site)  # the probe replaces every inherited entry (job_env dropped them)
    env["RCP_E2E_PROBE_JSONL"] = str(probe_jsonl)
    command = install_argv(
        ("rcp-ndcg", "run", "resume", "--run", str(run_dir)),
        version=version,
        wheelhouse=wheelhouse,
        constraints=constraints,
    )
    return subprocess.call(list(command), env=env)


def _wait_http(url: str, timeout_s: float) -> bool:
    """Whether ``url`` answers 2xx within ``timeout_s`` seconds (the standard library only)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as reply:  # noqa: S310 - the run's own engine
                if 200 <= reply.status < 300:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(POLL_S)
    return False


class ManagedEngine:
    """A run-scoped judge engine the driver owns: started beside the job, killed on command, restarted.

    The product's model (see :func:`~rcp_ndcg.runners.script.supervise`): engines the supervision script
    started end the job when they exit, while replicas that live elsewhere are "restarted by whatever runs
    them, and the judge bounds the outage instead" (``wait_on_outage_s``).  The outage scenario's judge
    lives here so a killed engine is an outage the run parks through, not a job failure.

    Attributes:
        command: The engine's argv (already ported).
        env: The engine process's environment (its ``CUDA_VISIBLE_DEVICES`` and ports).
        log_path: Where the engine's stdout and stderr go (appended).
    """

    def __init__(self, command: Sequence[str], *, env: Mapping[str, str], log_path: Path) -> None:
        self.command = list(command)
        self.env = dict(env)
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Start the engine (its own session, so :meth:`kill` takes its whole process group), with the
        launcher's environment hygiene (:func:`job_env` without the shim: the engine runs no ``srun``
        step): no probe ``PYTHONPATH`` reaches an engine process."""
        self.env.setdefault("TMPDIR", "/tmp")
        Path(self.env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        clean = {**os.environ, **self.env}
        clean.pop("PYTHONPATH", None)  # the probe belongs to the coordinators only (node-runtime item 10)
        log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the engine
        self.process = subprocess.Popen(
            self.command, env=clean, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def kill(self) -> None:
        """SIGKILL the engine's whole process group ("the engine is killed")."""
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        self.process = None

    def restart(self) -> None:
        """``end`` and ``start`` again ("the engine restarts")."""
        self.kill()
        self.start()

    def wait_ready(self, url: str, timeout_s: float) -> bool:
        """Wait until the engine's readiness URL answers 2xx (``False`` at the timeout)."""
        if self.process is not None and self.process.poll() is not None:
            return False
        return _wait_http(url, timeout_s)

    def stop(self) -> None:
        """SIGTERM the engine's session and reap it."""
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            self.kill()
        self.process = None


def t0_smoke(
    engine: ManagedEngine,
    *,
    url: str,
    served_name: str,
    completion_body: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """T0 smoke for a judge candidate (GPU-VALIDATION.md's T0 row): boot, ``GET /v1/models``, one request.

    Inputs: the engine to boot, its base URL (``.../v1``), the name the command serves and the body of the
    one chat completion to send.  Output: a report dict (``state``: ``verified`` or ``failed``, what
    ``/v1/models`` said: the engine version, ``max_model_len``, ``dtype``, and the error when it failed).
    The smoke is what decides the Flash-Next NVFP4 vs FP8 question on this node.
    """
    report: dict[str, Any] = {"state": "failed", "url": url, "served_model_name": served_name}
    engine.start()
    try:
        if not engine.wait_ready(f"{url}/models", timeout_s):
            report["error"] = f"the engine answered no {url}/models within {timeout_s:g} s"
            return report
        with urllib.request.urlopen(f"{url}/models", timeout=30) as reply:  # noqa: S310
            models = json.loads(reply.read().decode("utf-8"))
        listed = [row.get("id") for row in models.get("data", [])]
        report["models"] = listed
        if served_name not in listed:
            report["error"] = f"{url}/models lists {listed}, expected {served_name!r}"
            return report
        entry = next((row for row in models.get("data", []) if row.get("id") == served_name), {})
        report["max_model_len"] = entry.get("max_model_len")
        report["dtype"] = entry.get("dtype")
        report["owned_by"] = entry.get("owned_by")
        request = urllib.request.Request(
            f"{url}/chat/completions",
            data=json.dumps(dict(completion_body)).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as reply:  # noqa: S310 - the run's own engine
            answer = json.loads(reply.read().decode("utf-8"))
        report["completion"] = {
            "model": answer.get("model"),
            "usage": answer.get("usage"),
            "finish_reason": (answer.get("choices") or [{}])[0].get("finish_reason"),
        }
        report["state"] = "verified"
        return report
    except (urllib.error.URLError, OSError, ValueError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    finally:
        engine.stop()


def read_probe(probe_jsonl: Path) -> list[dict[str, Any]]:
    """The probe's records (one dict per interpreter that started with the probe on its path)."""
    if not probe_jsonl.exists():
        return []
    return [json.loads(line) for line in probe_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_process_boundary(
    records: Sequence[Mapping[str, Any]], *, engine_python: Mapping[str, Any], version: str | None
) -> dict[str, Any]:
    """The node-runtime item-10 assertion: the coordinator ran outside the engine environment.

    Inputs: the probe's records, the engine ``python3``'s record (:func:`engine_python_facts`) and the
    staged release version.  Output: a check dict -- ``ok`` when every coordinator record (its argv is
    ``run resume``) names a ``sys.prefix`` and executable the engine environment does not hold, and the
    versions it saw are the staged release's.
    """
    coordinators = [row for row in records if tuple(row.get("argv", ())[1:3]) == ("run", "resume")]
    problems: list[str] = []
    if not coordinators:
        problems.append("no coordinator record: the probe's sitecustomize ran in no coordinator process")
    engine_prefix = engine_python.get("sys_prefix")
    engine_executable = engine_python.get("executable")
    for row in coordinators:
        if row.get("sys_prefix") == engine_prefix or row.get("executable") == engine_executable:
            problems.append(f"the coordinator ran the engine environment's interpreter: {row}")
        if version is not None and row.get("versions", {}).get("rcp-ndcg") != version:
            problems.append(f"the coordinator saw rcp-ndcg {row.get('versions', {}).get('rcp-ndcg')}, staged {version}")
    return {
        "name": "process boundaries",
        "ok": not problems,
        "detail": "; ".join(problems)
        or f"{len(coordinators)} coordinator process(es) outside the engine environment ({engine_prefix})",
    }


def engine_python_facts() -> dict[str, Any]:
    """What the engine environment's ``python3`` is and sees (``sys.prefix``, the executable, its versions).

    This is the engine side of the process-boundary check: vLLM's interpreter, queried in the way the
    engines start it (``python3`` on this ``PATH``).
    """
    program = (
        "import json, platform, sys\n"
        "from importlib import metadata\n"
        "versions = {}\n"
        "for dist in ('vllm', 'torch', 'transformers'):\n"
        "    try:\n"
        "        versions[dist] = metadata.version(dist)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        versions[dist] = None\n"
        "print(json.dumps({'sys_prefix': sys.prefix, 'executable': sys.executable,"
        " 'python': platform.python_version(), 'versions': versions}))\n"
    )
    completed = subprocess.run(["python3", "-c", program], capture_output=True, text=True)
    if completed.returncode != 0:
        return {"sys_prefix": None, "executable": None, "error": completed.stderr.strip()}
    return json.loads(completed.stdout)


def run_facts(run_dir: Path) -> dict[str, Any]:
    """What two runs are compared on (and what a rerun must leave alone).

    One fact block per step: its identity (the payload's fields and its ``identity_hash``), and its
    artifacts by ``(path, sha256)``.  **What is compared** (the brief's "judgements may differ at
    temperature > 0: state what is compared"): ids and identities for every step; outputs **byte for byte**
    for the input-deterministic steps (``retrieve``, ``rerank``); and, for the judge-derived steps, the
    stored window count and the judgement families' keys -- both determined by the schedule and the judge
    identity -- never the judged values themselves (a judge sampling at temperature > 0 may answer
    differently in two identical runs, and the report says so).
    """
    manifest = RunManifest.load(RunLayout.at(run_dir))
    facts: dict[str, Any] = {"steps": [], "windows": {}, "families": sorted(manifest.families)}
    for record in manifest.steps:
        steps_facts: dict[str, Any] = {
            "name": record.name,
            "status": record.status.value,
            "identity_hash": record.identity_hash,
            "identity": record.identity,
        }
        if record.name in ("retrieve", "rerank"):
            steps_facts["outputs"] = sorted((ref.path, ref.sha256) for ref in record.outputs)
            steps_facts["inputs"] = sorted((ref.path, ref.sha256) for ref in record.inputs)
        else:
            steps_facts["outputs"] = sorted(ref.path for ref in record.outputs)
        facts["steps"].append(steps_facts)
    for name in ("tournament", "rubric"):
        store = run_dir / "judgements" / f"{name}.jsonl"
        facts["windows"][name] = sum(1 for _ in store.open(encoding="utf-8")) if store.is_file() else 0
    return facts


def full_facts(run_dir: Path) -> dict[str, Any]:
    """Everything a rerun must leave alone: :func:`run_facts` plus every step's times, requests and usage,
    the judge stores' sizes, the step outputs' bytes and the metrics (what "nothing recomputes" is decided
    on)."""
    manifest = RunManifest.load(RunLayout.at(run_dir))
    facts = run_facts(run_dir)
    facts["records"] = [
        {
            "name": record.name,
            "duration_s": record.duration_s,
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            "usage": record.usage.model_dump(mode="json") if record.usage else None,
        }
        for record in manifest.steps
    ]
    facts["usage"] = manifest.usage.model_dump(mode="json")
    facts["metrics"] = manifest.metrics
    facts["stores"] = {path.name: path.stat().st_size for path in sorted((run_dir / "judgements").glob("*.jsonl"))}
    facts["artifacts"] = {
        ref.path: _sha256(run_dir / ref.path)
        for record in manifest.steps
        for ref in record.outputs
        if (run_dir / ref.path).is_file()
    }
    return facts


def _sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes (what two identical runs' outputs are compared on)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def compare_runs(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """The comparison of two identical runs (:func:`run_facts` documents what is compared).

    Output: a check dict -- ``ok`` when every compared fact is equal, and ``mismatches`` naming each
    difference when not.
    """
    mismatches: list[str] = []
    for key in ("families", "windows", "steps"):
        if a.get(key) != b.get(key):
            mismatches.append(f"{key} differ")
    for left_row, right_row in zip(a.get("steps", ()), b.get("steps", ()), strict=False):
        if left_row != right_row:
            mismatches.append(f"step {left_row.get('name')!r} differs")
    return {
        "name": "two identical runs give identical identities and outputs",
        "ok": not mismatches,
        "detail": "; ".join(sorted(set(mismatches)))
        or "identities, the deterministic steps' outputs and the judgement windows' counts and families match "
        "(judged values are never compared: judgements may differ at temperature > 0)",
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _steps_all_completed(run_dir: Path, *, expected: Sequence[str]) -> dict[str, Any]:
    """The brief's "every step completes": each configured step's manifest record is ``completed``."""
    manifest = RunManifest.load(RunLayout.at(run_dir))
    seen = {record.name: record.status for record in manifest.steps}
    problems = [
        f"{name}: {seen[name].value if name in seen else 'missing'}"
        for name in expected
        if seen.get(name) is not StepStatus.COMPLETED
    ]
    return _check(
        "every step completes",
        not problems,
        "; ".join(problems) or f"all {len(expected)} step(s) completed",
    )


def _check_phase_engines(run_dir: Path, phases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The brief's "the manifest records each phase's engines": the rendered phases' engine blocks and the
    manifest's engine records agree (judging steps record what their endpoints said they serve)."""
    manifest = RunManifest.load(RunLayout.at(run_dir))
    problems: list[str] = []
    for phase in phases:
        roles = sorted(phase.get("engines", {}))
        for step in phase.get("steps", ()):
            record = manifest.step(step)
            if record is None:
                problems.append(f"phase of {roles}: step {step} is not in the manifest")
            elif "judge" in roles and not record.engines:
                problems.append(f"phase of {roles}: step {step} records no engine")
    return _check(
        "the manifest records each phase's engines",
        not problems,
        "; ".join(problems)
        or f"{len(phases)} phase(s) such that every judging step names what its engine said it serves (the "
        "product records EngineInfo for judging steps only -- runs/pipeline.py -- and the encoder/reranker "
        "phases' engine blocks are recorded here from the phase plan: "
        + ", ".join("/".join(sorted(phase.get("engines", {}))) or "none" for phase in phases)
        + ")",
    )


def _check_backend_unavailable(run_dir: Path, outputs: Sequence[Path]) -> dict[str, Any]:
    """The brief's "the wait_on_outage_s expiry path fails with BackendUnavailableError": the run's records
    and logs name the error class."""
    manifest = RunManifest.load(RunLayout.at(run_dir))
    hay = " ".join(
        [record.error or "" for record in manifest.steps]
        + [path.read_text(encoding="utf-8", errors="replace") for path in outputs if path.is_file()]
    ).lower()
    ok = "backendunavailable" in hay or "backend unavailable" in hay
    detail = "the failed run names BackendUnavailableError" if ok else "no BackendUnavailableError found"
    return _check("wait_on_outage_s expiry fails with BackendUnavailableError", ok, detail)


def _snapshot_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """The brief's "the identity rerun recomputes nothing": every recorded fact is identical."""
    same = json.dumps(before, sort_keys=True, default=str) == json.dumps(after, sort_keys=True, default=str)
    return _check(
        "the identity rerun recomputes nothing",
        same,
        "every step record, usage count, store size and metric is identical after the rerun"
        if same
        else "the rerun changed the recorded facts (see status.json for both snapshots)",
    )


class ScenarioResultBuilder:
    """The scenario's report row, appended to as checks land."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.checks: list[dict[str, Any]] = []
        self.runs: list[str] = []
        self.error: str | None = None
        self.judge: dict[str, Any] = {}

    def add(self, check: dict[str, Any]) -> None:
        self.checks.append(check)

    def build(self) -> ScenarioResult:
        ok = self.error is None and all(check["ok"] for check in self.checks)
        return ScenarioResult(
            scenario=self.scenario.id,
            mode=self.scenario.mode,
            status="passed" if ok else "failed",
            runs=self.runs,
            checks=self.checks,
            judge=self.judge,
            error=self.error,
        )


def _phase_facts(job_text_pipeline: Any) -> list[dict[str, Any]]:
    """What the run's phases are, engines per phase, from the product's own phase plan (records the
    manifest's per-phase engine block against it)."""
    from rcp_ndcg.support.serve import ServeByRole, plan_phases

    config = job_text_pipeline.config
    serve = config.serve or ServeByRole()
    phases = plan_phases(config.ordered_steps, serve, config.engine_uses())
    return [
        {"steps": list(phase.steps), "engines": {role: "served" for role in sorted(phase.engines)}} for phase in phases
    ]


def _judge_completion_body(scenario: Scenario) -> dict[str, Any]:
    """The one request the T0 smoke sends: the judge's own model name and one short exchange."""
    return {
        "model": scenario.judge.served_name,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 16,
    }


def run_scenario(
    scenario: Scenario,
    *,
    out_dir: str | Path,
    recipes_root: Sequence[str | Path],
    wheelhouse: str,
    constraints: str | None = None,
    version: str | None = None,
    runs_dir: str | Path | None = None,
) -> ScenarioResult:
    """Drive one scenario in this pod and return its report document.

    Inputs: the scenario, the output directory, the recipe roots and the install source the coordinators
    install the release from (``wheelhouse``, ``constraints``, ``version`` -- node-runtime item 2).
    Output: a :class:`ScenarioResult`; the run directories live under ``runs_dir`` (default
    ``<out_dir>/runs``, so one upload carries the report and the runs), the
    rendered scripts and job logs beside it.  Raises :class:`HarnessError` only for a scenario the driver
    cannot set up; a scenario's own failures are its result's checks and error.
    """
    roots = [Path(root) for root in recipes_root]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    builder = ScenarioResultBuilder(scenario)
    judge_engine, judge_choice, judge_report = _pick_judge(
        scenario, out=out, wheelhouse=wheelhouse, constraints=constraints, version=version
    )
    builder.judge = judge_report
    if judge_choice is None:
        builder.error = f"no judge candidate booted: {judge_report.get('error')}"
        return builder.build()
    install = {"wheelhouse": wheelhouse, "constraints": constraints, "version": version}
    try:
        if scenario.mode == "outage":
            _run_outage(scenario, builder, judge_engine, out=out, roots=roots, runs_dir=runs_dir,
                        install=install, judge_choice=judge_choice)  # fmt: skip
        elif scenario.mode == "identity":
            _run_identity(scenario, builder, out=out, roots=roots, runs_dir=runs_dir,
                          install=install, judge_choice=judge_choice)  # fmt: skip
        else:
            _run_text(scenario, builder, out=out, roots=roots, runs_dir=runs_dir,
                      install=install, judge_choice=judge_choice)  # fmt: skip
    except RcpNdcgError as error:
        builder.error = f"{type(error).__name__}: {error}"
    except HarnessError as error:
        builder.error = str(error)
    finally:
        if judge_engine is not None:
            judge_engine.stop()
    return builder.build()


def _pick_judge(
    scenario: Scenario, *, out: Path, wheelhouse: str, constraints: str | None, version: str | None
) -> tuple[ManagedEngine | None, tuple[JudgeModel, tuple[str, ...]] | None, dict[str, Any]]:
    """The judge engine of this scenario: the T0 smoke's winner (``candidate`` or its ``fallback``).

    GPU-VALIDATION.md's "Judges for T4": the four-phase run takes the NVFP4 candidate when vLLM v0.31.0
    serves it (the source says it parses the checkpoint's ``hf_quant_config.json`` and serves
    ``Qwen4ExpForConditionalGeneration``; the T0 smoke proves this node does), else the FP8 fallback.  A
    ``mode: outage`` judge is started fresh here as the driver's run-scoped engine.

    Returns:
        ``(engine, (checkpoint, command), summary)``: the engine is started only in ``outage`` mode (the
        run-scoped judge the driver restarts); in the other modes the run's own job starts the winner's
        command through ``serve.judge``.
    """
    slot = scenario.judge.slot
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[JudgeModel | None, tuple[str, ...]]] = [(scenario.judge.candidate, scenario.judge.command)]
    if scenario.judge.fallback is not None and scenario.judge.fallback_command is not None:
        candidates.append((scenario.judge.fallback, scenario.judge.fallback_command))
    attempts: list[dict[str, Any]] = []
    for model, command in candidates:
        assert model is not None  # the candidate is always named
        rendered = scenario.judge.render_command(command, port=slot.port)
        engine = ManagedEngine(
            rendered, env=slot.env(scratch), log_path=out / f"judge-t0-{model.model.rsplit('/', 1)[-1]}.log"
        )
        report = t0_smoke(
            engine,
            url=f"http://127.0.0.1:{slot.port}/v1",
            served_name=scenario.judge.served_name,
            completion_body=_judge_completion_body(scenario),
            timeout_s=slot.startup_timeout_s,
        )
        report["judge_model"] = model.model
        report["judge_revision"] = model.revision
        report["command"] = rendered
        attempts.append(report)
        if report["state"] == "verified":
            if scenario.mode == "outage":  # the T0 smoke stopped its engine; the run gets a fresh one
                engine.start()
            summary: dict[str, Any] = {"state": "verified", "attempts": attempts}
            for key in ("judge_model", "judge_revision", "models", "max_model_len", "dtype"):
                if key in report:
                    summary[key] = report[key]
            return engine, (model, tuple(command)), summary
    return (
        None,
        None,
        {
            "state": "unresolved",
            "attempts": attempts,
            "error": "; ".join(row.get("error", "") for row in attempts),
        },
    )


def _prepare_run(
    scenario: Scenario,
    *,
    roots: Sequence[Path],
    out: Path,
    runs_dir: str | Path | None,
    port_offset: int,
    install: Mapping[str, Any],
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None = None,
    run_dir: Path | None = None,
    stage: bool = True,
) -> tuple[Any, Path, str]:
    """One run's staged directory and its rendered phased job script.

    Output: ``(pipeline, run_dir, script)``.  The script carries the phase workers' probe environment
    (``PYTHONPATH``, ``RCP_E2E_PROBE_JSONL``) and the uv bootstrap (a pod's engine image has pip, and the
    coordinator runs through ``uvx`` from the wheelhouse -- node-runtime items 1 and 2).  ``run_dir``
    renders the phases against an existing run directory (the identity rerun) and ``stage=False`` stages
    none of its own.
    """
    config = build_run_config(scenario, recipes_root=roots, port_offset=port_offset, judge_choice=judge_choice)
    pipeline = prepare(config, runs_dir=str(runs_dir) if runs_dir else None, label=scenario.id)
    if stage:
        stage_run_dir(pipeline)
    staged = run_dir or Path(pipeline.layout.root)
    write_probe(out)  # the workers' PYTHONPATH names it; without the file the probe would never record
    script = render_phased_script(
        pipeline,
        options=render_options(out, staged, pipeline=pipeline),
        wheelhouse=str(install["wheelhouse"]),
        constraints=str(install.get("constraints")),
        version=str(install.get("version")) if install.get("version") else None,
        run_dir=run_dir,
    )
    return pipeline, staged, script


def render_options(out: Path, workdir: Path, *, pipeline: Any) -> dict[str, Any]:
    """The runner options every driven render uses -- the golden file pins this exact set (the tests and
    :func:`_prepare_run` share it, so the golden is the byte stream the run executes).

    The setup brings uv to an image that lacks it (:func:`~rcp_ndcg.runners.script.bootstrap_uv`) and
    creates each engine slot's ``TMPDIR`` (node-runtime item 7 -- without the directory Python's
    ``tempfile`` silently falls back to the shared ``/tmp`` and ``mktemp`` fails).
    """
    site, probe_jsonl = probe_paths(out)  # pure: rendering writes nothing
    serve = pipeline.config.serve or ServeByRole()
    tmpdirs = sorted(
        {
            engine.env["TMPDIR"]
            for role in ("judge", "encoder", "reranker")
            if (engine := getattr(serve, role, None)) is not None
        }
    )
    setup = [*bootstrap_uv()]
    if tmpdirs:
        setup.append("mkdir -p " + " ".join(shlex.quote(path) for path in tmpdirs))
    return {
        "log_dir": str(out / "logs"),
        "workdir": str(workdir),
        "setup": setup,
        "env": {"PYTHONPATH": str(site), "RCP_E2E_PROBE_JSONL": str(probe_jsonl)},
    }


def _run_text(
    scenario: Scenario,
    builder: ScenarioResultBuilder,
    *,
    out: Path,
    roots: Sequence[Path],
    runs_dir: str | Path | None,
    install: Mapping[str, Any],
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None,
) -> None:
    """``mode: run``: the phased run (``pair`` of them), with the phase and boundary checks."""
    facts: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    for repeat in range(scenario.pair):
        name = scenario.id if scenario.pair == 1 else f"{scenario.id}-{repeat + 1}"
        run_out = out / name
        run_out.mkdir(parents=True, exist_ok=True)
        pipeline, run_dir, script = _prepare_run(
            scenario, roots=roots, out=run_out, runs_dir=runs_dir, port_offset=0,
            install=install, judge_choice=judge_choice,
        )  # fmt: skip
        phases = _phase_facts(pipeline)
        builder.runs.append(str(run_dir))
        status, log_path = run_job_script(script, out=run_out)
        builder.add(_check(f"{name}: the job script exits 0", status == 0, f"exit {status} ({log_path})"))
        builder.add(_steps_all_completed(run_dir, expected=scenario.steps))
        builder.add(_check_phase_engines(run_dir, phases))
        builder.add(_boundary_check(run_out, install, name))
        roles = ", ".join("/".join(sorted(phase["engines"])) or "none" for phase in phases)
        builder.add(_check(f"{name}: phases recorded", True, f"{roles} ({len(phases)} phases)"))
        facts.append(run_facts(run_dir))
    if scenario.pair == 2:
        builder.add(compare_runs(facts[0], facts[1]))


def _boundary_check(run_out: Path, install: Mapping[str, Any], label: str = "") -> dict[str, Any]:
    """The process-boundary check of one driven job (node-runtime item 10): its probe records beside the
    engine environment's own ``python3`` facts.  Every mode drives jobs under it -- the entry's claim
    applies to all of them."""
    _, probe_jsonl = write_probe(run_out)  # idempotent: returns the probe's paths
    version = str(install["version"]) if install.get("version") else None
    check = check_process_boundary(read_probe(probe_jsonl), engine_python=engine_python_facts(), version=version)
    if label:
        check["name"] = f"process boundaries ({label})"
        check["detail"] = f"{label}: {check['detail']}"
    return check


def _run_identity(
    scenario: Scenario,
    builder: ScenarioResultBuilder,
    *,
    out: Path,
    roots: Sequence[Path],
    runs_dir: str | Path | None,
    install: Mapping[str, Any],
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None,
) -> None:
    """``mode: identity``: the run, then the same run directory again on different ports: nothing recomputes."""
    pipeline, run_dir, script = _prepare_run(
        scenario, roots=roots, out=out, runs_dir=runs_dir, port_offset=0,
        install=install, judge_choice=judge_choice,
    )  # fmt: skip
    builder.runs.append(str(run_dir))
    phases = _phase_facts(pipeline)
    status, log_path = run_job_script(script, out=out)
    builder.add(_check("first run: the job script exits 0", status == 0, f"exit {status} ({log_path})"))
    builder.add(_steps_all_completed(run_dir, expected=scenario.steps))
    builder.add(_check_phase_engines(run_dir, phases))
    builder.add(_boundary_check(out, install, "first run"))
    before = full_facts(run_dir)
    (out / "facts-before.json").write_text(json.dumps(before, indent=2, default=str) + "\n", encoding="utf-8")

    rerun_out = out / "rerun"
    rerun_out.mkdir(parents=True, exist_ok=True)
    _, _, rerun_script = _prepare_run(
        scenario, roots=roots, out=rerun_out, runs_dir=runs_dir,
        port_offset=scenario.identity_port_offset, install=install, judge_choice=judge_choice,
        run_dir=run_dir, stage=False,  # the same run directory, on the rerun's (new) ports
    )  # fmt: skip
    status, log_path = run_job_script(rerun_script, out=rerun_out)
    builder.add(_check("the rerun (new ports) exits 0", status == 0, f"exit {status} ({log_path})"))
    builder.add(_boundary_check(rerun_out, install, "rerun"))
    after = full_facts(run_dir)
    (out / "facts-after.json").write_text(json.dumps(after, indent=2, default=str) + "\n", encoding="utf-8")
    builder.add(_snapshot_equal(before, after))
    unchanged = before.get("artifacts") == after.get("artifacts") and before.get("stores") == after.get("stores")
    builder.add(
        _check(
            "the rerun recomputed no judgement or output",
            unchanged,
            "every step output and judgement store is byte-identical after the rerun"
            if unchanged
            else "the rerun rewrote an output or a judgement store",
        )
    )


def _run_outage(
    scenario: Scenario,
    builder: ScenarioResultBuilder,
    judge: ManagedEngine | None,
    *,
    out: Path,
    roots: Sequence[Path],
    runs_dir: str | Path | None,
    install: Mapping[str, Any],
    judge_choice: tuple[JudgeModel, tuple[str, ...]] | None,
) -> None:
    """``mode: outage``: the run's judge engine is killed mid-tournament.

    Two sub-runs (both stages of GPU-VALIDATION.md's outage scenario) and a resume:

    * **recovers** -- the killed judge restarts after a short outage; the run parks through it and finishes;
    * **expires** -- the killed judge stays down past the role's ``wait_on_outage_s``; the run fails with
      :class:`~rcp_ndcg.errors.BackendUnavailableError`, and after the engine returns a ``run resume``
      finishes it (the T4 criterion "resume after a killed engine parks and recovers").
    """
    from rcp_ndcg.judging.client import JudgeConfig

    assert judge is not None  # the outage scenario's judge is the driver's own
    probe = JudgeConfig.model_validate(
        {**scenario.judge.config, "model": scenario.judge.served_name, "base_url": "http://127.0.0.1:1/v1"}
    )
    wait_s = float(probe.wait_on_outage_s or 300.0)
    # (a) the outage it recovers from
    run_out = out / "outage-recovers"
    run_out.mkdir(parents=True, exist_ok=True)
    pipeline, run_dir, script = _prepare_run(
        scenario, roots=roots, out=run_out, runs_dir=runs_dir, port_offset=0,
        install=install, judge_choice=judge_choice,
    )  # fmt: skip
    builder.runs.append(str(run_dir))
    phases = _phase_facts(pipeline)
    status, parked = _outage_pass(
        scenario, run_out=run_out, script=script, run_dir=run_dir, judge=judge,
        outage_s=min(20.0, max(5.0, wait_s / 6)), restart=True,
    )  # fmt: skip
    builder.add(
        _check(
            "the outaged run parks and recovers",
            status == 0 and parked,
            f"parked through the outage: {parked}; the job script exited {status}",
        )
    )
    builder.add(_steps_all_completed(run_dir, expected=scenario.steps))
    builder.add(_check_phase_engines(run_dir, phases))
    builder.add(_boundary_check(run_out, install, "outage: the recovers sub-run"))

    # (b) the wait_on_outage_s expiry path
    run_out = out / "outage-expires"
    run_out.mkdir(parents=True, exist_ok=True)
    _, run_dir, script = _prepare_run(
        scenario, roots=roots, out=run_out, runs_dir=runs_dir, port_offset=0,
        install=install, judge_choice=judge_choice,
    )  # fmt: skip
    builder.runs.append(str(run_dir))
    status, parked = _outage_pass(
        scenario, run_out=run_out, script=script, run_dir=run_dir, judge=judge,
        outage_s=wait_s + 30.0, restart=False,
    )  # fmt: skip
    expired = _check(
        "the outage outlasting wait_on_outage_s fails the run",
        status != 0,
        f"the job script exited {status} (the run parked through the outage: {parked})",
    )
    builder.add(expired)
    builder.add(_check_backend_unavailable(run_dir, [run_out / "job.log", Path(run_dir) / "logs" / "run.log"]))
    builder.add(_boundary_check(run_out, install, "outage: the expires sub-run"))

    # and the resume finishes it (the T4 criterion "resume after a killed engine parks and recovers")
    judge.restart()
    judge.wait_ready(f"http://127.0.0.1:{scenario.judge.slot.port}/v1/models", scenario.judge.slot.startup_timeout_s)
    resumed = resume_run(run_dir, out=run_out, **install)  # type: ignore[arg-type]
    builder.add(_check("resume after the killed engine finishes the run", resumed == 0, f"run resume exited {resumed}"))
    builder.add(_steps_all_completed(run_dir, expected=scenario.steps))
    builder.add(_boundary_check(run_out, install, "the resume"))


def _outage_pass(
    scenario: Scenario,
    *,
    run_out: Path,
    script: str,
    run_dir: Path,
    judge: ManagedEngine,
    outage_s: float,
    restart: bool,
) -> tuple[int, bool]:
    """One outage sub-run: the script runs to the middle of the tournament, the judge is killed, and
    ``restart`` decides whether it comes back after ``outage_s`` or stays down past it.  Observes the
    parking: the run is alive and unfinished while the judge is down.  Output: ``(the script's exit
    status, the parked observation)``."""
    url = f"http://127.0.0.1:{scenario.judge.slot.port}/v1"
    if judge.process is None:
        judge.start()
        judge.wait_ready(f"{url}/models", scenario.judge.slot.startup_timeout_s)
    process = launch_job_script(script, out=run_out)  # the driver's launcher: the srun shim, no probe
    store = run_dir / "judgements" / "tournament.jsonl"
    deadline = time.monotonic() + scenario.judge.slot.startup_timeout_s + 1800
    while time.monotonic() < deadline and process.poll() is None:
        if store.is_file() and sum(1 for _ in store.open(encoding="utf-8")) >= 1:
            break
        time.sleep(POLL_S)
    judge.kill()  # the judge engine is killed mid-tournament
    killed_at = time.monotonic()
    time.sleep(outage_s)
    parked = process.poll() is None  # the run is still going: it parked through the outage
    if restart:
        judge.start()
        judge.wait_ready(f"{url}/models", scenario.judge.slot.startup_timeout_s)
    status = process.wait()
    record = {
        "killed_at": killed_at,
        "outage_s": outage_s,
        "restart": restart,
        "parked": parked,
        "exit_status": status,
    }
    (run_out / "outage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return status, parked


def write_report(out: Path, results: Sequence[ScenarioResult]) -> Path:
    """``E2E.md`` and ``status.json`` of the T4 stage: per scenario, its judge decision and every check."""
    document = {
        "schema": "rcp-ndcg-vllm.e2e-report.v1",
        "passed": all(row.status == "passed" for row in results),
        "scenarios": [row.model_dump(mode="json") for row in results],
    }
    (out / "status.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    lines = ["# T4 end to end", ""]
    for row in results:
        lines += [f"## {row.scenario}: {row.status.upper()}", "", row.description, ""]
        judge = row.judge or {}
        if judge.get("judge_model"):
            revision = judge.get("judge_revision", "?")
            lines.append(f"Judge: `{judge['judge_model']}` @ `{revision}` ({judge.get('state')}).")
        for attempt in judge.get("attempts", []):
            if attempt.get("error"):
                lines.append(f"- T0 smoke of `{attempt.get('judge_model')}`: failed -- {attempt['error']}")
            else:
                lines.append(f"- T0 smoke of `{attempt.get('judge_model')}`: verified")
        lines.append("")
        for check in row.checks:
            lines.append(f"- [{'x' if check['ok'] else ' '}] {check['name']}: {check['detail']}")
        if row.error:
            lines.append(f"- error: {row.error}")
        lines += ["", f"Runs: {', '.join(row.runs) or '(none)'}", ""]
    lines.append(
        "What two identical runs are compared on: every step's identity and its deterministic outputs, and the "
        "judgement windows' counts and families (judged values are never compared: judgements may differ at "
        "temperature > 0)."
    )
    (out / "E2E.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "E2E.md"


def _resolve_scenarios(names: Sequence[str], roots: Sequence[Path]) -> list[Path]:
    """Scenario files: a name is a path, or ``<id>`` looked up under each root (a directory loads all)."""
    resolved: list[Path] = []
    for name in names:
        path = Path(name)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.yaml")))
            continue
        if path.is_file():
            resolved.append(path)
            continue
        for root in roots:
            candidate = root / f"{name}.yaml"
            if candidate.is_file():
                resolved.append(candidate)
                break
        else:
            raise HarnessError(f"no scenario {name!r} under {[str(root) for root in roots]}")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI: drive the scenarios named on the command line in this pod.  Exit 0 when every scenario
    passed, 1 when one failed, 2 for a bad request."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.e2e", description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--scenarios", required=True, help="scenario ids, files, directories, or @file listing ids/paths (one per line)"
    )
    parser.add_argument("--scenarios-root", action="append", default=[], help="where scenario ids resolve")
    parser.add_argument(
        "--recipes-root",
        action="append",
        required=True,
        help="the staged recipes root(s), where the encoder and reranker recipes live",
    )
    parser.add_argument("--out", required=True, help="the T4 output directory (E2E.md, status.json, run dirs)")
    parser.add_argument("--runs-dir", default=None, help="where the run directories land (default: <out>/runs)")
    parser.add_argument("--wheelhouse", required=True, help="the staged wheelhouse every coordinator installs from")
    parser.add_argument("--constraints", default=None, help="the staged constraints file")
    parser.add_argument("--version", default=None, help="the staged release version (default: the installed one)")
    parser.add_argument(
        "--upload", default=None, help="gs:// URI: the report and the run directories upload there after the scenarios"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenario_roots = [Path(root) for root in args.scenarios_root] or [default_scenarios_root()]
    out = Path(args.out)
    runs_dir = Path(args.runs_dir) if args.runs_dir else out / "runs"
    try:
        names = _scenario_ids(args.scenarios) if args.scenarios.startswith("@") else args.scenarios.split(",")
        paths = _resolve_scenarios([name for name in names if name], scenario_roots)
        results = [
            run_scenario(
                load_scenario(path),
                out_dir=out / path.stem,
                recipes_root=[Path(root) for root in args.recipes_root],
                wheelhouse=args.wheelhouse,
                constraints=args.constraints,
                version=args.version,
                runs_dir=runs_dir,
            )
            for path in paths
        ]
    except (HarnessError, RecipeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    write_report(out, results)
    if args.upload:
        from .jobs import run_wave as _run_wave

        _run_wave._upload(out, args.upload)  # noqa: SLF001 - the wave runner's transfer path: one home
    for row in results:
        print(f"{row.scenario}: {row.status}")
    print(f"e2e: {'PASS' if all(row.status == 'passed' for row in results) else 'FAIL'}")
    return 0 if all(row.status == "passed" for row in results) else 1


def _scenario_ids(value: str) -> list[str]:
    """``@file`` (one scenario id or path per line, ``#`` comments allowed) into a list (the wave
    runner's file rule).

    Raises:
        HarnessError: the file cannot be read (a bad request, not a crash).
    """
    try:
        lines = Path(value[1:]).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HarnessError(f"cannot read the scenario list {value[1:]}: {error}") from error
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


if __name__ == "__main__":
    raise SystemExit(main())
