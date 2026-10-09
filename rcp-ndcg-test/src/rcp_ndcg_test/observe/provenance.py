"""The corpus manifest's provenance (OBSERVATIONS-SPEC section 4): engine, model, recipe and collector facts.

Every key :data:`rcp_ndcg_test.corpus.PROVENANCE_KEYS` names is filled with its value or with an explicit
:func:`unavailable` reason (``{"unavailable": "..."}``) -- never left out -- so a later difference between two
corpora can be explained and a missing fact is visible, not silent.  The engine facts are probed on the node
(``nvidia-smi`` and the engine environment's own Python, each under a timeout); the model and recipe facts come
from the recipe, its files and the Hugging Face cache; the collector facts from this package and the run.
Secrets never enter: the engine environment is filtered to the behaviour-affecting ``VLLM_*`` variables minus
anything named like a credential, and the node's hostname is recorded hashed.

Public surface:

- :func:`unavailable`.
- :func:`engine_facts`, :func:`model_facts`, :func:`recipe_facts`, :func:`collector_facts`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

__all__ = ["collector_facts", "engine_facts", "model_facts", "recipe_facts", "unavailable"]

_SECRET_NAME = re.compile(r"(?i)(token|secret|key|password|passwd|credential|auth)")

_ENGINE_PROBE = (
    "import json, torch, vllm\n"
    "print(json.dumps({'vllm_version': getattr(vllm, '__version__', None),"
    " 'vllm_commit': getattr(vllm, '__commit__', None),"
    " 'torch': torch.__version__, 'cuda': torch.version.cuda}))"
)
"""What the engine environment's Python reports about itself (run with ``-c``)."""

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def unavailable(reason: str) -> dict[str, str]:
    """An explicit stand-in for a provenance fact that could not be collected, with the reason."""
    return {"unavailable": reason}


def _run(runner: Runner, argv: list[str], timeout_s: float) -> tuple[str | None, str]:
    """``(stdout, "")`` of one probe, or ``(None, reason)`` when it cannot run or fails."""
    try:
        completed = runner(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{argv[0]} could not run: {type(error).__name__}: {error}"
    if completed.returncode != 0:
        return None, f"{argv[0]} exited {completed.returncode}: {(completed.stderr or '').strip()[:200]}"
    return completed.stdout, ""


def engine_facts(
    *,
    image: str,
    serve_argv: list[str],
    engine_python: str | None,
    environ: Mapping[str, str] | None = None,
    started: str | None,
    ready_wait_s: float | None,
    runner: Runner = subprocess.run,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """The engine block: image and digest, vLLM version and commit, torch, CUDA, driver, GPUs, argv, env, times.

    Inputs: the image reference the wave serves, the exact ``vllm serve`` argv executed, the engine
    environment's Python (``None``: not probed, said so), the environment the engine ran with, its start time
    and readiness wait (seconds), and the subprocess runner (tests inject one).  The image digest comes from
    ``RCP_IMAGE_DIGEST`` (the pod cannot read its own digest; the submit step exports it).  Output: the block,
    every key a value or an :func:`unavailable` reason.
    """
    env = dict(os.environ if environ is None else environ)
    facts: dict[str, Any] = {
        "name": "vllm",
        "image": image,
        "version": image.rpartition(":")[2].removeprefix("v"),
        "image_digest": env.get("RCP_IMAGE_DIGEST")
        or unavailable("RCP_IMAGE_DIGEST is not set (the pod cannot read its own image digest)"),
        "serve_argv": list(serve_argv),
        "env": {
            name: value
            for name, value in sorted(env.items())
            if name.startswith("VLLM_") and not _SECRET_NAME.search(name)
        },
        "started": started or unavailable("the engine's start time was not recorded"),
        "ready_wait_s": ready_wait_s
        if ready_wait_s is not None
        else unavailable("the readiness wait was not measured"),
    }
    stdout, reason = _run(runner, ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout_s)
    if stdout is None:
        facts["gpus"] = facts["driver"] = unavailable(reason)
    else:
        rows = [line.split(",") for line in stdout.strip().splitlines() if line.strip()]
        names = sorted({row[0].strip() for row in rows})
        drivers = sorted({row[1].strip() for row in rows if len(row) > 1})
        facts["gpus"] = {"model": names, "count": len(rows)}
        facts["driver"] = drivers[0] if len(drivers) == 1 else drivers
    if engine_python is None:
        missing = unavailable("the engine environment's Python was not given to the collector")
        for key in ("vllm_version", "vllm_commit", "torch", "cuda"):
            facts[key] = missing
        return facts
    stdout, reason = _run(runner, [engine_python, "-c", _ENGINE_PROBE], timeout_s)
    reported: dict[str, Any] = {}
    if stdout is not None:
        try:
            reported = json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            reason = f"the engine probe printed no JSON: {error}"
    for key in ("vllm_version", "vllm_commit", "torch", "cuda"):
        value = reported.get(key)
        facts[key] = value if value is not None else unavailable(reason or f"the engine environment reports no {key}")
    return facts


def _hub_snapshot(model: str, revision: str, hub_cache: Path | None) -> Path | None:
    """The Hugging Face cache's snapshot directory of ``model`` at ``revision``, when it is there."""
    root = hub_cache
    if root is None:
        home = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
        )
        root = Path(home)
    snapshot = root / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def _weights(model: str, revision: str, hub_cache: Path | None) -> Any:
    """``{file: sha256}`` of the snapshot's weight files and their index, read from the cache's content-addressed
    blob names (a Hub LFS blob is named by its SHA-256): exact and cheap, nothing re-hashed."""
    snapshot = _hub_snapshot(model, revision, hub_cache)
    if snapshot is None:
        return unavailable(f"{model}@{revision} is not in the Hugging Face cache on this machine")
    out: dict[str, str] = {}
    for path in sorted(snapshot.rglob("*")):
        name = path.relative_to(snapshot).as_posix()
        if not (name.endswith((".safetensors", ".bin", ".pt")) or name.endswith(".index.json")):
            continue
        target = path.resolve()
        blob = target.name
        out[name] = blob if re.fullmatch(r"[0-9a-f]{64}", blob) else hashlib.sha256(target.read_bytes()).hexdigest()
    return out or unavailable(f"{model}@{revision}: the cached snapshot holds no weight files")


def model_facts(
    recipe: Recipe,
    *,
    hub_cache: str | Path | None = None,
    plugin_wheel: str | Path | None = None,
) -> dict[str, Any]:
    """The model block: id, revision, weight hashes, tokenizer and template hashes, plugin, served settings.

    Inputs: the recipe, the Hugging Face hub cache (default: the environment's) and the plugin wheel the
    engine installed (its SHA-256 is recorded).  Output: the block; the tokenizer hash is the behaviour
    fingerprint's own (:func:`rcp_ndcg_test.fingerprint.tokenizer_sha256`).
    """
    from rcp_ndcg_test.errors import HarnessError

    from ..fingerprint import tokenizer_sha256

    serve = recipe.serve
    try:
        tokenizer: Any = tokenizer_sha256(recipe)
    except HarnessError as error:
        tokenizer = unavailable(str(error))
    template: Any = None
    if serve.chat_template is None:
        template = unavailable("the recipe serves no template file (the model's own chat template applies)")
    elif recipe._dir is not None:
        template = hashlib.sha256((Path(recipe._dir) / serve.chat_template).read_bytes()).hexdigest()
    plugin: Any
    if serve.plugin is None:
        plugin = unavailable("the recipe installs no plugin")
    else:
        wheel = Path(plugin_wheel) if plugin_wheel is not None else None
        plugin = {
            "name": serve.plugin,
            "io_processor_plugin": serve.io_processor_plugin,
            "wheel": wheel.name if wheel is not None else unavailable("the plugin wheel's path was not given"),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()
            if wheel is not None and wheel.is_file()
            else unavailable("the plugin wheel is not readable here"),
        }
    return {
        "id": recipe.model,
        "revision": recipe.revision,
        "weights": _weights(recipe.model, recipe.revision, Path(hub_cache) if hub_cache is not None else None),
        "tokenizer_sha256": tokenizer,
        "template_sha256": template,
        "plugin": plugin,
        "hf_overrides": serve.hf_overrides,
        "pooler_config": serve.pooler_config,
        "mm_processor_kwargs": serve.mm_processor_kwargs
        if serve.mm_processor_kwargs is not None
        else unavailable("the recipe pins no mm_processor_kwargs"),
        "dtype": serve.dtype,
    }


def recipe_facts(recipe: Recipe) -> dict[str, Any]:
    """The recipe block: id, the recipe file's SHA-256, the behaviour fingerprint and its named inputs, status.

    The fingerprint is :func:`rcp_ndcg_test.fingerprint.behaviour_fingerprint` (the one key of the model layer)
    with :func:`~rcp_ndcg_test.fingerprint.fingerprint_inputs`, so a staleness failure names what changed;
    ``declared_dim`` is the client's token-vector width (the acceptance check decodes vectors at it).
    """
    from ..fingerprint import behaviour_fingerprint, fingerprint_inputs

    recipe_file = Path(recipe._dir) / "recipe.yaml" if recipe._dir is not None else None
    return {
        "id": recipe.id,
        "file_sha256": hashlib.sha256(recipe_file.read_bytes()).hexdigest()
        if recipe_file is not None and recipe_file.is_file()
        else unavailable("the recipe was not loaded from a directory"),
        "behaviour_fingerprint": behaviour_fingerprint(recipe),
        "fingerprint_inputs": fingerprint_inputs(recipe),
        "status": recipe.status.model_dump(mode="json"),
        "declared_dim": recipe.client.get("dim"),
    }


def collector_facts(
    *,
    wave_id: str | None,
    job_id: str | None,
    started: str,
    finished: str | None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The collector block: this package's version and commit, the generator's identity, the run's ids and times.

    Inputs: the wave and job ids (``None``: unavailable, said so), the recording's start and end times, and the
    environment (``RCP_SOURCE_COMMIT`` names the commit the release candidate was built from).  Output: the
    block, with the node's hostname as its SHA-256 only.
    """
    from importlib.metadata import PackageNotFoundError, version

    from rcp_ndcg_test.corpus import RECORD_SCHEMA

    from .requests import CORPUS_PLAN_VERSION, GENERATOR_VERSION, PINNED_DATASET_COMMITS, SEED

    env = dict(os.environ if environ is None else environ)
    try:
        package_version: Any = version("rcp-ndcg-vllm")
    except PackageNotFoundError:
        package_version = unavailable("rcp-ndcg-vllm is not installed as a distribution")
    return {
        "package": "rcp-ndcg-vllm",
        "version": package_version,
        "commit": env.get("RCP_SOURCE_COMMIT")
        or unavailable("RCP_SOURCE_COMMIT is not set (the release candidate's manifest records the commit)"),
        "generator_version": GENERATOR_VERSION,
        "corpus_plan_version": CORPUS_PLAN_VERSION,
        "record_schema": RECORD_SCHEMA,
        "seed": SEED,
        "dataset_commits": dict(PINNED_DATASET_COMMITS),
        "wave_id": wave_id or unavailable("the wave id was not given"),
        "job_id": job_id or env.get("RCP_JOB_ID") or unavailable("RCP_JOB_ID is not set"),
        "started": started,
        "finished": finished or unavailable("the recording did not finish"),
        "host_sha256": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest(),
    }
