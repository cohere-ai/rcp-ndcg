"""The wave runner: many recipes on one node's GPUs, one failing recipe never stops the wave.

For every recipe it packs the engine onto ``resources.gpus`` GPUs (``--tensor-parallel-size`` follows the recipe),
starts one ``vllm serve`` per slot from :func:`~rcp_ndcg_vllm.recipe.serve_argv` with its own
``CUDA_VISIBLE_DEVICES``, port (``--port-base`` + slot; default 8100; ``--port-base 0`` gives every engine port 0),
``VLLM_PORT`` and ``TMPDIR`` (one home per slot: two engines cannot collide), waits for ``GET /v1/models`` within the
recipe's ``engine.startup_timeout_s`` (an engine that exits early fails that recipe only), then runs smoke,
equivalence (stages 1 and 2) and — with ``--record`` — the recorder, stops the engine's process group, and moves
on.  A recipe that cannot run at all — it fails validation when loaded, or the bootstrap recorded its
``serve.plugin`` among ``--failed-plugins`` (installed from the staged tree or wheelhouse only) — is a failed row
in the wave report with the validation message or the plugin's exact name; the wave runs the rest.  The pod has no
persistent volume (node-runtime item 8): before each recipe the runner measures the free
disk and the model's Hub size and fails the recipe early when it measurably cannot fit (on a fresh pod the cache
does not exist yet, so the measurement lands on the nearest existing parent); after a recipe whose
model no later recipe reuses, the model's weights are evicted from the HF cache.  It writes
``<out>/<id>/{serve.log, equivalence.json, EQUIVALENCE.md, status.json}``, a wave summary
(``wave.json`` and ``WAVE.md``), and with ``--upload`` copies ``<out>`` to the URI after each recipe
(``gcloud storage cp -r``, then a ``gsutil -m cp -r`` fallback, then the product's own
:mod:`rcp_ndcg.storage` - the stock engine image ships neither CLI).

Test mode: ``--vllm-cmd "python tests/stub_engine.py"`` replaces the ``vllm serve`` launcher with that command
(the rest of the rendered argv is appended, so a stub engine receives the real flags and may ignore them), and
``--port-base 0`` gives every engine ``--port 0``; such an engine must announce its bound port by printing
``RCPS_STUB_PORT=<n>`` on stdout, which the runner reads instead of guessing a port.

Run it on the node with ``python -m rcp_ndcg_vllm.jobs.run_wave`` (the node bootstrap's wave mode does; see
``jobs/bootstrap.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..equivalence import run as run_equivalence
from ..errors import HarnessError, RecipeError
from ..recipe import Recipe, default_recipes_root, serve_argv
from ..record import record as record_exchanges
from .plugins import spec_of as _plugin_spec_of
from .wavelist import load_wave, parse_ids

__all__ = ["main", "run_wave"]

_POLL_S = 2.0
_ANNOUNCE_TIMEOUT_S = 60.0

_ZMQ_IPC_SUFFIX_CHARS = 37
"""One vLLM ZMQ IPC socket path under a slot's TMPDIR: ``/`` plus the 36-character uuid.  AF_UNIX's
``sun_path`` caps total paths at 107 characters, so a slot's TMPDIR must leave this much room
(``<slot tmpdir>`` + this <= 107)."""


def _slot_tmp_dir(slot: int) -> Path:
    """One engine slot's TMPDIR: short, unique per wave and slot, outside the output tree.

    vLLM's ZMQ IPC sockets live under the slot's TMPDIR as ``<uuid>`` and AF_UNIX caps paths at 107
    characters - a TMPDIR of ``<out>/<recipe-id>/tmp`` blows the cap for long recipe ids (an engine
    died on exactly that path shape once).  The directory is ``<system temp>/rcp-s<pid>-<slot>`` (≈ 22
    characters): whatever the recipe id and the state prefix are.  The runner removes it with its
    engine (it is scratch).  Inputs: the slot index.  Output: the directory (not yet created).
    Units: none.
    """
    return Path(tempfile.gettempdir()) / f"rcp-s{os.getpid()}-{slot}"


def run_wave(
    recipe_ids: list[str],
    recipes_root: str | Path | None = None,
    *,
    gpus: int,
    out_dir: str | Path,
    upload: str | None = None,
    record: bool = False,
    record_corpus: bool = False,
    quality: bool = False,
    paper_numbers: str | Path | None = None,
    controls: bool = False,
    changed_since_index: str | Path | None = None,
    pairs_dir: str | Path | None = None,
    reference_python: str | None = None,
    vllm_cmd: str | None = None,
    port_base: int = 8100,
    failed_plugins: Iterable[str] = (),
) -> dict[str, Any]:
    """Run one wave: every recipe on the node's GPUs, as parallel as the GPUs allow.

    Inputs: the recipe ids (directories under ``recipes_root``; empty means every recipe there), the GPU count,
    the output directory, and the plugin specs the bootstrap could not install (the recipes that name one fail
    early, with its exact name).  Output: the wave document (also ``wave.json`` and ``WAVE.md`` under ``out_dir``);
    a recipe's own failure — including a recipe that fails validation at load — is recorded in its status and
    never raises.  Raises :class:`HarnessError` only for a bad wave request: a missing recipe root, or a wave
    list with no recipes at all (an unknown or invalid id is a failed row, not a wave abort).

    ``record_corpus`` writes one observation corpus per recipe under ``<out>/<id>/corpus/`` (the request
    plan's rows twice in one process and once after an engine restart -- the runner stops and restarts
    the engine between the passes).  ``--changed-since <index>`` re-records only the recipes whose
    behaviour fingerprint differs from the index (a previous ``wave.json`` or corpus index), listing
    the rest as ``skipped_unchanged`` (OBSERVATIONS-SPEC section 7).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _MODEL_SIZES.clear()  # each wave asks the Hub for its models' sizes once
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    recipes, load_failures = _resolve_recipes(recipe_ids, root)
    not_installed = frozenset(failed_plugins)
    skipped_unchanged: list[str] = []
    change_verdict: dict[str, Any] | None = None
    if changed_since_index is not None:
        recipes, skipped_unchanged, change_verdict = _filter_changed(recipes, Path(changed_since_index), vllm_cmd)
    results: dict[str, dict[str, Any]] = {}
    # The two early-failure classes that know no engine: a recipe that fails validation, and a recipe
    # whose plugin the bootstrap could not install.  Both are failed rows; the wave runs the rest.
    for recipe_id, message in load_failures.items():
        _record_status(results, out, _failed_row(recipe_id, message))
    for recipe in list(recipes):
        spec = _uninstalled_plugin(recipe, not_installed, root)
        if spec is not None:
            error = f"the recipe's plugin {spec} is not staged and not in the staged wheelhouse"
            _record_status(
                results,
                out,
                _status(recipe, "failed", error=error, steps={"serve": {"state": "failed", "error": error}}),
            )
            recipes.remove(recipe)
    used_gpus: set[int] = set()
    pending = list(recipes)
    running: list[_EngineRun] = []
    slot = 0
    while pending or running:
        progressed = False
        for recipe in list(pending):
            need = recipe.resources.gpus
            if need > gpus:
                row = _status(recipe, "failed", error=f"needs {need} GPUs, the wave has {gpus}", steps={})
                _record_status(results, out, row)
                pending.remove(recipe)
                progressed = True
                continue
            if len(used_gpus) + need <= gpus:
                # Node-runtime item 8: the pod has no persistent volume; a model that measurably cannot
                # fit fails here, before its engine has started and downloaded anything.
                disk = _disk_check(recipe)
                if disk["error"] is not None:
                    row = _status(
                        recipe,
                        "failed",
                        error=disk["error"],
                        steps={"serve": {"state": "failed", "error": disk["error"]}},
                        disk=disk,
                    )
                    _record_status(results, out, row)
                    pending.remove(recipe)
                    progressed = True
                    continue
                assigned = _lowest_free(used_gpus, need)
                used_gpus.update(assigned)
                try:
                    run = _start(recipe, assigned, slot, out, vllm_cmd, port_base, disk=disk)
                except HarnessError as start_error:
                    # An engine that cannot even start (no vllm binary) fails that recipe only.
                    row = _status(recipe, "failed", error=str(start_error), steps={"serve": {"state": "failed"}})
                    _record_status(results, out, row)
                    used_gpus.difference_update(assigned)
                    pending.remove(recipe)
                    progressed = True
                    continue
                running.append(run)
                slot += 1
                pending.remove(recipe)
        for run in list(running):
            error: str | None = None
            if run.exited():
                error = f"the engine exited early (code {run.return_code()})"
            elif run.timed_out():
                error = f"GET /v1/models not ready within {run.timeout_s:.0f}s"
            if run.exited() or run.timed_out() or run.ready():
                # One home per concept (item 8): the model's weights stay while any other recipe in this
                # wave still needs them (queued or already served); otherwise they are evicted below.
                reuse = any(other.model == run.recipe.model for other in pending) or any(
                    other.recipe.model == run.recipe.model for other in running if other is not run
                )
                _finalise(
                    run, results, out, pairs_dir=pairs_dir, record=record, error=error,
                    reference_python=reference_python, reuse=reuse,
                    record_corpus=record_corpus, vllm_cmd=vllm_cmd, port_base=port_base,
                    quality=quality, paper_numbers=paper_numbers, controls=controls,
                )  # fmt: skip
                running.remove(run)
                used_gpus.difference_update(run.gpus)
                progressed = True
                break
        if not progressed:
            time.sleep(_POLL_S)
        if upload is not None and progressed:
            _upload(out, upload)
    if upload is not None:
        _upload(out, upload)
    document = _wave_document(gpus, results, skipped_unchanged=skipped_unchanged, change_verdict=change_verdict)
    (out / "wave.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (out / "WAVE.md").write_text(_wave_markdown(document), encoding="utf-8")
    return document


class _EngineRun:
    """One running engine: its process, its log, its readiness and its recipe's status dict."""

    def __init__(
        self,
        recipe: Recipe,
        gpus: list[int],
        port: int,
        popen: subprocess.Popen[bytes],
        log_path: Path,
        out_dir: Path,
        disk: dict[str, Any] | None = None,
        tmpdir: Path | None = None,
    ) -> None:
        self.recipe = recipe
        self.gpus = gpus
        self.port = port
        self.popen = popen
        self.log_path = log_path
        self.out_dir = out_dir
        self.tmpdir = Path(tmpdir) if tmpdir is not None else log_path.parent / "tmp"
        self.disk: dict[str, Any] = disk or {}
        self.env: dict[str, str] = {}
        self.started = time.monotonic()
        self.timeout_s = float(recipe.engine.startup_timeout_s)
        self.status: dict[str, Any] = _status(recipe, "running", port=port, gpus=gpus, steps={})
        self._announced_port: int | None = None
        self._pump_done = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Tee the engine's stdout into serve.log and catch a stub's RCPS_STUB_PORT announcement."""
        assert self.popen.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log:
            for raw in iter(self.popen.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                log.write(line)
                log.flush()
                if line.startswith("RCPS_STUB_PORT="):
                    try:
                        self._announced_port = int(line.strip().split("=", 1)[1])
                    except ValueError:
                        pass
        self._pump_done.set()

    def exited(self) -> bool:
        """Whether the engine process is gone."""
        return self.popen.poll() is not None

    def return_code(self) -> int | None:
        """The engine's exit code, when it is gone."""
        return self.popen.poll()

    def ready(self) -> bool:
        """Whether the assigned port answers ``GET /v1/models`` with 200 (test mode resolves the port first)."""
        if self.port == 0:
            announced = self.announced_port(0.0)
            if announced is None:
                return False
            self.port = announced
        return self._models_ok(self.port)

    def timed_out(self) -> bool:
        """Whether the engine outlived the recipe's startup timeout without becoming ready."""
        return time.monotonic() - self.started > self.timeout_s

    def announced_port(self, timeout_s: float) -> int | None:
        """The port the engine announced (test mode), waiting up to ``timeout_s`` for the line."""
        deadline = time.monotonic() + timeout_s
        while self._announced_port is None:
            if self.exited() or time.monotonic() > deadline:
                return None
            time.sleep(0.2)
        return self._announced_port

    def _models_ok(self, port: int) -> bool:
        import httpx

        try:
            response = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=10.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        """Stop the engine's whole process group: SIGTERM, then SIGKILL after a grace period."""
        if self.popen.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.popen.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - the engine already died
            self.popen.kill()
        try:
            self.popen.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - a stuck engine
            try:
                os.killpg(os.getpgid(self.popen.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.popen.wait()
        self._thread.join(timeout=5)


def _failed_row(recipe_id: str, error: str) -> dict[str, Any]:
    """A failed wave-report row for a recipe that never ran (no :class:`Recipe` could be loaded for it)."""
    return {"recipe": recipe_id, "state": "failed", "gpus": None, "started": _now(), "error": error, "steps": {}}


def _uninstalled_plugin(recipe: Recipe, failed_plugins: frozenset[str], root: Path) -> str | None:
    """The recipe's plugin exact name when the bootstrap recorded it as not installable.

    The match is exactly the form ``jobs.plugins`` collects for THIS recipe (its staged file as
    ``<recipe-id>/<file>``, else the bare spec), so one recipe's failed bare name never fails a recipe
    whose own collected form installed fine.  The row's message carries the exact name from the
    recipe.  Units: none.
    """
    spec = _plugin_spec_of(recipe, root)
    return recipe.serve.plugin if spec is not None and spec in failed_plugins else None


def _record_status(results: dict[str, dict[str, Any]], out: Path, row: dict[str, Any]) -> None:
    """Record one recipe's report row and its ``<out>/<id>/status.json``, creating the directory."""
    results[row["recipe"]] = row
    directory = out / row["recipe"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def _lowest_free(used: set[int], count: int) -> list[int]:
    """The lowest GPU indices not in use."""
    free: list[int] = []
    candidate = 0
    while len(free) < count:
        if candidate not in used:
            free.append(candidate)
        candidate += 1
    return free


def _start(
    recipe: Recipe,
    gpus: list[int],
    slot: int,
    out: Path,
    vllm_cmd: str | None,
    port_base: int,
    *,
    disk: dict[str, Any] | None = None,
) -> _EngineRun:
    """Start one engine on the given GPUs; the port is ``port_base + slot``, or 0 (announced) in test mode.

    Node-runtime item 7: every slot gets its own ``CUDA_VISIBLE_DEVICES``, HTTP port, ``VLLM_PORT`` (the
    engine's internal port) and ``TMPDIR``, so two engines on one node cannot collide on any of them.
    """
    port = port_base if port_base == 0 else port_base + slot
    argv = serve_argv(recipe, port=port, served_model_name=recipe.id)
    if vllm_cmd:
        argv = [*shlex.split(vllm_cmd), *argv[2:]]
    directory = out / recipe.id
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    # One home per slot, kept SHORT and outside the output tree: the slot's TMPDIR carries vLLM's ZMQ
    # IPC sockets, whose paths must fit AF_UNIX's 107 characters whatever the recipe id is.
    tmpdir = _slot_tmp_dir(slot)
    tmpdir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmpdir)
    if port_base != 0:
        # The engine's internal port, distinct per slot (test mode leaves it to the stub).
        env["VLLM_PORT"] = str(port_base + 1000 + slot)
    try:
        popen = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    except OSError as error:
        shutil.rmtree(tmpdir, ignore_errors=True)  # a start that never ran leaves no scratch behind
        raise HarnessError(f"cannot start the engine for {recipe.id} ({' '.join(argv[:2])} ...): {error}") from error
    run = _EngineRun(
        recipe,
        gpus,
        port,
        popen,
        directory / "serve.log",
        directory,
        disk=disk,
        tmpdir=tmpdir,
    )
    run.status["serve_argv"] = argv
    run.env = env
    run.status["steps"]["serve"] = {"state": "running", "port": port, "gpus": gpus, "tmpdir": str(tmpdir)}
    return run


_MODEL_SIZES: dict[str, int | None] = {}
"""The Hub size of each model, asked once per wave (the same model does not download twice)."""


def _disk_check(recipe: Recipe) -> dict[str, Any]:
    """The pre-serve disk check (node-runtime item 8): free disk, the model's size, the verdict.

    A model whose Hub metadata is unreachable has an unknown size: the check records ``unknown`` and
    the wave proceeds (never silently - the status says so); a model that measurably does not fit is
    the recipe's early failure, with the one-line reason.
    """
    from . import weights

    free = weights.disk_free_bytes(weights.hf_cache_root())
    size = _model_size(recipe)
    ok, reason = weights.will_fit(free, size)
    document: dict[str, Any] = {
        "free_disk_bytes": free,
        "model_bytes": size,
        "fits": ok if size is not None else None,
    }
    document["error"] = reason if not ok else None
    if size is None:
        document["note"] = "the model's size is unknown (the Hub did not answer); the disk check is a record only"
    return document


def _model_size(recipe: Recipe) -> int | None:
    """The recipe model's weight bytes, asked once per model per process (the Hub's file metadata)."""
    from . import weights

    key = f"{recipe.model}@{recipe.revision}"
    if key not in _MODEL_SIZES:
        _MODEL_SIZES[key] = weights.model_disk_bytes(recipe.model, recipe.revision)
    return _MODEL_SIZES[key]


def _evict(recipe: Recipe, *, reuse: bool) -> dict[str, Any]:
    """The post-recipe eviction (node-runtime item 8), recorded for the recipe's status."""
    from . import weights

    if reuse:
        return {"evicted": False, "reason": "a later recipe in this wave serves the same model"}
    eviction = weights.evict(recipe.model)
    document: dict[str, Any] = {
        "evicted": eviction.removed,
        "freed_bytes": eviction.freed_bytes,
        "free_disk_bytes": eviction.bytes_after,
    }
    if eviction.error is not None:
        document["error"] = eviction.error
    return document


def _mark_serve_step(run: _EngineRun, state: str) -> None:
    """Record the serve step's final state (and its slot's TMPDIR), keeping any error the failure
    path recorded."""
    step = run.status["steps"].get("serve") or {}
    run.status["steps"]["serve"] = {
        "state": state,
        "port": run.port,
        "gpus": run.gpus,
        "tmpdir": str(run.tmpdir),
        **({"error": step["error"]} if step.get("error") else {}),
    }


def _finalise(
    run: _EngineRun,
    results: dict[str, dict[str, Any]],
    out: Path,
    *,
    pairs_dir: str | Path | None = None,
    record: bool = False,
    record_corpus: bool = False,
    quality: bool = False,
    paper_numbers: str | Path | None = None,
    controls: bool = False,
    vllm_cmd: str | None = None,
    port_base: int = 8100,
    error: str | None = None,
    reference_python: str | None = None,
    reuse: bool = False,
) -> None:
    """Take one engine to its end state: run the steps, or record the failure, then stop it.

    Unless ``reuse`` (a later recipe in this wave serves the same model), the model's weights are evicted
    from the HF cache when the engine has stopped (node-runtime item 8: the pod has no persistent
    volume), and the disk before/after is recorded with the recipe's status.  The observation-corpus
    step restarts the engine between the in-process passes and the after-restart pass (its own
    ``server_run_id`` per engine run).
    """
    restarted: list[_EngineRun] = []
    run.status["ready_wait_s"] = round(time.monotonic() - run.started, 3)
    try:
        if error is None and run.port == 0:
            announced = run.announced_port(_ANNOUNCE_TIMEOUT_S)
            if announced is None:
                error = "the engine did not announce RCPS_STUB_PORT (test mode)"
            elif not run._models_ok(announced):
                error = f"GET /v1/models not ready on the announced port {announced}"
            else:
                run.port = announced
                run.status["port"] = announced
        if error is None:
            base_url = f"http://127.0.0.1:{run.port}"
            run.status["steps"]["smoke"] = _smoke(run.recipe, base_url)
            served: list[dict[str, Any]] = []
            run.status["steps"]["equivalence"] = _equivalence(
                run.recipe, base_url, out, pairs_dir, reference_python, recorder=served if record_corpus else None
            )
            if record:
                run.status["steps"]["record"] = _record(run.recipe, base_url, out)
            if quality:
                # Before the corpus step: that one restarts the engine for its after-restart pass.
                run.status["steps"]["quality"] = _quality(run, base_url, out, reference_python, paper_numbers, vllm_cmd)
            if record_corpus:
                step, fingerprint = _observe_corpus(
                    run,
                    out,
                    pairs_dir,
                    vllm_cmd=vllm_cmd,
                    port_base=port_base,
                    restarted=restarted,
                    equivalence_exchanges=served if run.status["steps"]["equivalence"].get("stages") else None,
                )
                run.status["steps"]["observation_corpus"] = step
                run.status["behaviour_fingerprint"] = fingerprint
                run.status["engine_version"] = _engine_version(run.recipe, vllm_cmd)
            if controls:
                # Last: the recipe-variant controls take the slot's GPUs one engine at a time (one owner).
                run.status["steps"]["controls"] = _controls(
                    run, out, pairs_dir, reference_python, vllm_cmd=vllm_cmd, port_base=port_base, restarted=restarted
                )
            steps = run.status["steps"]
            record_ok = not record or steps["record"].get("state") == "passed"
            controls_ok = not controls or steps["controls"].get("state") == "passed"
            corpus_ok = not record_corpus or steps["observation_corpus"].get("state") != "failed"
            quality_ok = not quality or steps["quality"].get("state") == "passed"
            run.status["state"] = (
                "verified"
                if steps["smoke"].get("state") == "passed"
                and steps["equivalence"].get("passed")
                and record_ok
                and corpus_ok
                and quality_ok
                and controls_ok
                else "failed"
            )
        else:
            run.status["state"] = "failed"
            run.status["error"] = error
            run.status["steps"]["serve"] = {"state": "failed", "error": error}
    except Exception as step_error:  # noqa: BLE001 - one recipe's failure never stops the wave
        run.status["state"] = "failed"
        run.status["error"] = f"{type(step_error).__name__}: {step_error}"
    finally:
        # The serve step records ITS outcome: "the engine answered and was stopped
        # cleanly" is a success, whatever a later step's verdict is - a clean stop is not a failure.
        _mark_serve_step(run, "passed" if error is None else "failed")
        if run.status["state"] == "failed" and not run.status.get("error"):
            # A row never fails bare: the steps that failed are named with their errors, and any skip
            # that stands between the row and "verified" is named too.
            failed_steps = [
                f"{name}: {step.get('error') or 'failed'}"
                for name, step in run.status["steps"].items()
                if isinstance(step, dict) and step.get("state") == "failed"
            ]
            skipped = [
                f"{name} ({step['reason']})" if step.get("reason") else name
                for name, step in run.status["steps"].items()
                if isinstance(step, dict) and step.get("state") == "skipped"
            ]
            if failed_steps:
                run.status["error"] = "; ".join(failed_steps + [f"skipped {name}" for name in skipped])
            elif skipped:
                run.status["error"] = f"verification incomplete: {', '.join(skipped)}"
        run.stop()
        for extra in restarted:
            extra.stop()
        shutil.rmtree(run.tmpdir, ignore_errors=True)  # the slot's scratch TMPDIR leaves with its engine
        run.status["disk"] = {**run.disk, **_evict(run.recipe, reuse=reuse)}
        run.status["finished"] = _now()
        _write_status(run)
        results[run.recipe.id] = run.status


def _write_status(run: _EngineRun) -> None:
    """One recipe's final status file."""
    path = run.out_dir / "status.json"
    path.write_text(json.dumps(run.status, indent=2) + "\n", encoding="utf-8")


def _smoke(recipe: Recipe, base_url: str) -> dict[str, Any]:
    """One minimal request per role: the engine serves, the route answers, the body parses."""
    import httpx

    try:
        root = base_url.rstrip("/")
        if root.endswith(("/v1", "/v2")):
            root = root.rsplit("/", 1)[0]
        with httpx.Client(base_url=root, timeout=120.0) as http:
            if recipe.role == "rerank":
                reply = http.post("/rerank", json={"model": recipe.id, "query": "smoke query",
                                                   "documents": ["smoke document"], "top_n": 1})  # fmt: skip
            elif recipe.role == "embed":
                reply = http.post("/v1/embeddings", json={"model": recipe.id, "input": ["smoke text"]})
            else:
                reply = http.post("/pooling", json={"model": recipe.id, "input": ["smoke text"], "task": "token_embed"})
        ok = reply.status_code == 200
    except httpx.HTTPError as error:
        return {"state": "failed", "error": str(error)}
    return {"state": "passed" if ok else "failed"}


def _equivalence(
    recipe: Recipe,
    base_url: str,
    out: Path,
    pairs_dir: str | Path | None,
    reference_python: str | None,
    *,
    recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stages 1 and 2 for one recipe, written to ``<out>/<id>/equivalence.json``; ``recorder`` collects stage 2's
    captured exchanges (the corpus step checks its replies against them)."""
    pairs_path = _pairs_path(recipe, pairs_dir)
    if pairs_path is None:
        return {"state": "skipped", "reason": "no pairs file; give --pairs-dir"}
    try:
        document = run_equivalence(
            recipe,
            base_url=base_url,
            pairs_path=str(pairs_path),
            out_dir=str(Path(out) / recipe.id),
            stages=[1, 2],
            reference_python=reference_python,
            served_model_name=recipe.id,
            recorder=recorder,
        )
        return {"state": "passed" if document["passed"] else "failed", "passed": document["passed"], "stages": [1, 2]}
    except HarnessError as error:
        return {"state": "failed", "error": str(error)}


def _pairs_path(recipe: Recipe, pairs_dir: str | Path | None) -> Path | None:
    """The recipe's pairs file: ``<pairs-dir>/<id>.jsonl``, falling back to ``default.jsonl``."""
    if pairs_dir is None:
        return None
    root = Path(pairs_dir)
    for candidate in (root / f"{recipe.id}.jsonl", root / "default.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def _record(recipe: Recipe, base_url: str, out: Path) -> dict[str, Any]:
    """The recorder's fixed request set, written under ``<out>/<engine>-<version>/<recipe-id>/``."""
    try:
        written = record_exchanges(recipe, base_url, out)
    except HarnessError as error:
        return {"state": "failed", "error": str(error)}
    return {"state": "passed", "files": [str(path) for path in written]}


def _engine_version(recipe: Recipe, vllm_cmd: str | None) -> str:
    """The engine version a recording is keyed by: the image's tag, or ``test-stub`` when a stub serves."""
    return "test-stub" if vllm_cmd else recipe.engine.image.rpartition(":")[2].removeprefix("v")


def _observe_corpus(
    run: _EngineRun,
    out: Path,
    pairs_dir: str | Path | None,
    *,
    vllm_cmd: str | None,
    port_base: int,
    restarted: list[_EngineRun],
    equivalence_exchanges: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """One observation corpus for the recipe over the request plan's rows (OBSERVATIONS-SPEC 1-6).

    The corpus lands at its keyed, immutable path (:func:`~rcp_ndcg_vllm.observe.corpus.corpus_path`:
    ``<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<recorded-at>/``), keyed by the one behaviour
    fingerprint (:func:`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`).  The ``restart`` closure stops the
    engine and starts it again on the same slot, so the ``after_restart`` pass runs against a new engine
    process with its own run id.  The engine block is probed (``nvidia-smi``, the engine environment's Python
    named by ``RCP_ENGINE_PYTHON``); in test mode a stub serves and no vLLM fact is claimed.  Returns the step
    document and the fingerprint.
    """
    from ..equivalence.fitting import load_pairs
    from ..fingerprint import behaviour_fingerprint
    from ..observe.corpus import corpus_path
    from ..observe.provenance import collector_facts, engine_facts
    from ..record import record_corpus

    recipe = run.recipe
    try:
        fingerprint = behaviour_fingerprint(recipe)
    except HarnessError as error:
        return {"state": "failed", "error": f"the behaviour fingerprint cannot be computed: {error}"}, None
    pairs_path = _pairs_path(recipe, pairs_dir)
    if pairs_path is None:
        return {"state": "skipped", "reason": "no pairs file; give --pairs-dir"}, fingerprint
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(load_pairs(pairs_path)):
        strata = row.get("_strata") or []
        rows.append(
            {
                **row,
                "request_id": str(row.get("request_id", f"pairs:{index}")),
                "stratum": str(strata[0]) if strata else "",
            }
        )
    slot = max(run.port - port_base, 0) if port_base else 0
    base_url = f"http://127.0.0.1:{run.port}"
    started = _now()
    version = _engine_version(recipe, vllm_cmd)
    directory = corpus_path(out, version, recipe.id, fingerprint, started)

    loading: list[dict[str, Any]] = []

    def restart() -> tuple[str, str] | None:
        run.stop()
        fresh = _start(recipe, run.gpus, slot, out, vllm_cmd, port_base, disk=run.disk)
        restarted.append(fresh)
        # OBSERVATIONS-SPEC section 1's readiness edge: one request while the engine is still loading (a stub
        # in test mode announces its port first; a refused connection is recorded as such).
        port = fresh.announced_port(30.0) if fresh.port == 0 else fresh.port
        if port:
            import httpx

            from ..record import bare_exchange

            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as http:
                loading.append(bare_exchange(http, "GET", "/v1/models", None))
        deadline = time.monotonic() + run.timeout_s
        while time.monotonic() < deadline:
            if fresh.exited():
                return None
            if fresh.ready():
                return f"http://127.0.0.1:{fresh.port}", f"{recipe.id}@{_now()}"
            time.sleep(_POLL_S)
        return None

    engine = engine_facts(
        image="test-stub (no vLLM behaviour is claimed)" if vllm_cmd else recipe.engine.image,
        serve_argv=list(run.status.get("serve_argv") or []),
        engine_python=None if vllm_cmd else os.environ.get("RCP_ENGINE_PYTHON"),
        environ=run.env,
        started=run.status.get("started"),
        ready_wait_s=run.status.get("ready_wait_s"),
    )
    engine["version"] = version
    collector = collector_facts(
        wave_id=os.environ.get("RCP_WAVE_ID") or out.name,
        job_id=os.environ.get("RCP_JOB_ID"),
        started=started,
        finished=None,
    )
    try:
        report = record_corpus(
            recipe,
            base_url,
            rows,
            directory,
            server_run_id=f"{recipe.id}@{run.status.get('started', started)}",
            engine_facts=engine,
            collector=collector,
            restart=restart,
            equivalence_exchanges=equivalence_exchanges,
            while_loading=loading,
        )
    except Exception as error:  # noqa: BLE001 - the corpus step fails this recipe, never the wave
        return {"state": "failed", "error": f"{type(error).__name__}: {error}"}, fingerprint
    return {"state": "passed" if report["passed"] else "failed", **report}, fingerprint


def _quality(
    run: _EngineRun,
    base_url: str,
    out: Path,
    reference_python: str | None,
    paper_numbers: str | Path | None,
    vllm_cmd: str | None,
) -> dict[str, Any]:
    """The T3 quality stage for one recipe (:func:`rcp_ndcg_vllm.quality.run_quality`) on its task-matrix tasks,
    written under ``<out>/<id>/quality/``.  The paper's stored per-subset numbers come from ``--paper-numbers``
    (``{recipe: {metric: {subset: value}}}``); the golden-replay corpus carries the wave's engine and collector
    blocks.  A recipe outside the task matrix, or any failing task, fails the step."""
    from .. import quality as t3
    from ..observe.provenance import collector_facts, engine_facts

    recipe = run.recipe
    if reference_python is None:
        return {"state": "failed", "error": "the quality stage needs --reference-python (the mteb reference)"}
    try:
        tasks = t3.tasks_for(recipe.id)
        paper = None
        if paper_numbers is not None:
            paper = json.loads(Path(paper_numbers).read_text(encoding="utf-8")).get(recipe.id)
        manifest = {
            "engine": engine_facts(
                image=recipe.engine.image,
                serve_argv=list(run.status.get("serve_argv") or []),
                engine_python=None if vllm_cmd else os.environ.get("RCP_ENGINE_PYTHON"),
                environ=run.env,
                started=run.status.get("started"),
                ready_wait_s=run.status.get("ready_wait_s"),
            ),
            "collector": collector_facts(
                wave_id=os.environ.get("RCP_WAVE_ID") or out.name,
                job_id=os.environ.get("RCP_JOB_ID"),
                started=_now(),
                finished=None,
            ),
        }
        document = t3.run_quality(
            recipe,
            engine_url=base_url,
            tasks=tasks,
            work_dir=out / recipe.id / "quality",
            reference_python=reference_python,
            paper=paper,
            golden_manifest=manifest,
        )
    except (HarnessError, OSError, ValueError) as error:
        return {"state": "failed", "error": str(error)}
    return {
        "state": "passed" if document["passed"] else "failed",
        "errors": document["errors"],
        "report": "quality.json",
    }


def _control_gates(
    recipe: Recipe, base_url: str, out_dir: Path, pairs_path: Path, reference_python: str
) -> dict[str, Any]:
    """Stages 1 and 2 (and a media recipe's media stage) for one control: the ordinary gates, which must fail
    it.  An error the served side raises (a garbled frame the client cannot decode) is the stage failing on that
    request, recorded with its text."""
    from rcp_ndcg.errors import RcpNdcgError

    try:
        document = run_equivalence(
            recipe,
            base_url=base_url,
            pairs_path=str(pairs_path),
            out_dir=str(out_dir),
            stages=[1, 2],
            reference_python=reference_python,
            served_model_name=recipe.id,
        )
    except (HarnessError, RcpNdcgError) as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "passed": bool(document["passed"]),
        "stage1": document.get("stage1", {}).get("passed"),
        "stage2": document.get("stage2", {}).get("passed"),
        "media": (document.get("media") or {}).get("passed"),
    }


def _controls(
    run: _EngineRun,
    out: Path,
    pairs_dir: str | Path | None,
    reference_python: str | None,
    *,
    vllm_cmd: str | None,
    port_base: int,
    restarted: list[_EngineRun],
) -> dict[str, Any]:
    """The negative controls (a)-(f) of one recipe (GPU-VALIDATION.md item 5), through the ordinary gates.

    Only after the recipe's own gates passed (a control "caught" by a gate that fails everything proves nothing).
    Wire controls run against the recipe's live engine with the request bodies patched; then the recipe's
    engines stop and each recipe variant is served on the slot in turn (one GPU owner at a time), its client and
    engine agreeing on the variant's id.  A variant whose engine does not come up is NOT counted as caught (an
    engine refusing its argv fails every gate and proves nothing): its row says so and the summary flags it.
    Returns the step: :func:`~rcp_ndcg_vllm.observe.controls.controls_summary`'s report and its state.
    """
    from ..equivalence.wire import patched_wire
    from ..observe.controls import control_variants, controls_summary

    recipe = run.recipe
    equivalence = run.status["steps"].get("equivalence") or {}
    if not equivalence.get("passed"):
        return {"state": "skipped", "reason": "the recipe's own gates did not pass: a control would prove nothing"}
    pairs_path = _pairs_path(recipe, pairs_dir)
    if pairs_path is None or reference_python is None:
        return {"state": "failed", "error": "the controls need the pairs file and --reference-python"}
    live = next((engine for engine in reversed(restarted) if not engine.exited()), run)
    live_url = f"http://127.0.0.1:{live.port}"
    work = out / recipe.id / "controls"
    rows: list[dict[str, Any]] = []
    variants = control_variants(recipe)
    for variant in variants:
        if variant["kind"] is None:
            rows.append({"control": variant["control"], "name": variant["name"], "reason": variant["reason"]})
        elif variant["kind"] == "wire":
            with patched_wire(variant["wire_patch"]):
                gates = _control_gates(recipe, live_url, work / variant["name"], pairs_path, reference_python)
            rows.append({"control": variant["control"], "name": variant["name"], "equivalence": gates})
    run.stop()
    for engine in restarted:
        engine.stop()
    slot = max(run.port - port_base, 0) if port_base else 0
    for variant in variants:
        if variant["kind"] != "recipe":
            continue
        engine = _start(variant["recipe"], run.gpus, slot, out, vllm_cmd, port_base, disk=run.disk)
        try:
            deadline = time.monotonic() + run.timeout_s
            while not engine.ready() and not engine.exited() and time.monotonic() < deadline:
                time.sleep(_POLL_S)
            if engine.ready():
                gates = _control_gates(
                    variant["recipe"], f"http://127.0.0.1:{engine.port}", work / variant["name"], pairs_path,
                    reference_python,
                )  # fmt: skip
            else:
                gates = {
                    "passed": None,
                    "error": "the variant's engine did not become ready: the control was not served",
                }
        finally:
            engine.stop()
        rows.append({"control": variant["control"], "name": variant["name"], "equivalence": gates})
    rows.sort(key=lambda row: str(row["control"]))
    summary = controls_summary(rows)
    return {"state": "passed" if summary["passed"] else "failed", **summary}


def _filter_changed(
    recipes: list[Recipe], index_path: Path, vllm_cmd: str | None
) -> tuple[list[Recipe], list[str], dict[str, Any]]:
    """OBSERVATIONS-SPEC section 7's re-record-changed-only: keep the recipes whose corpus key -- behaviour
    fingerprint and engine version -- differs from the previous ``wave.json``'s, and return the untouched ones'
    ids and the verdict (why each changed, which engine versions are new) for the wave document."""
    from ..observe.corpus import changed_since

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"--changed-since {index_path} is unreadable: {error}") from error
    try:
        verdict = changed_since(recipes, index, engine_version_of=lambda recipe: _engine_version(recipe, vllm_cmd))
    except HarnessError as error:
        raise HarnessError(f"--changed-since cannot fingerprint the wave's recipes: {error}") from error
    changed = set(verdict["changed"])
    return [recipe for recipe in recipes if recipe.id in changed], verdict["unchanged"], verdict


def _resolve_recipes(recipe_ids: list[str], recipes_root: str | Path | None) -> tuple[list[Recipe], dict[str, str]]:
    """The wave's recipes and its failures: the named ids under the root (every recipe there when the
    list is empty), loaded tolerantly — a recipe that fails validation lands in the failure map with its
    validation message and is marked failed by the wave, never a wave abort."""
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    recipes, failed = load_wave(recipe_ids, root)
    if not recipes and not failed:
        raise HarnessError(f"no recipes under {root}")
    return recipes, failed


def _upload(out: Path, uri: str) -> None:
    """Copy ``<out>``'s contents to ``uri``: gcloud, gsutil, then the product's own storage; failures
    only warn (the stock engine image ships neither CLI, and the client environment carries the
    product's gcsfs, so the third path is the node's usual one)."""
    if not any(out.iterdir()):
        return
    for argv in (
        ["gcloud", "storage", "cp", "-r", f"{out}/*", f"{uri}/"],
        ["gsutil", "-m", "cp", "-r", f"{out}/*", f"{uri}/"],
    ):
        try:
            completed = subprocess.run(argv, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        if completed.returncode == 0:
            return
    if _upload_storage(out, uri):
        return
    print(
        f"[wave] upload to {uri} failed (gcloud, gsutil and the python transfer); the wave continues", file=sys.stderr
    )


def _upload_storage(out: Path, uri: str) -> bool:
    """The product's own storage as the last fallback: every local file under ``out`` written to
    ``uri`` through :mod:`rcp_ndcg.storage` (the one home for gs:// paths; gcsfs via ADC)."""
    try:
        from rcp_ndcg import storage
    except ImportError:
        return False
    try:
        storage.makedirs(f"{uri.rstrip('/')}/")
        for path in sorted(out.rglob("*")):
            if path.is_file():
                storage.write_bytes(f"{uri.rstrip('/')}/{path.relative_to(out)}", path.read_bytes())
    except Exception as error:  # noqa: BLE001 - the upload warns, never fails the wave
        print(f"[wave] the python upload failed: {type(error).__name__}: {error}", file=sys.stderr)
        return False
    return True


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _status(recipe: Recipe, state: str, **fields: Any) -> dict[str, Any]:
    return {"recipe": recipe.id, "state": state, "gpus": recipe.resources.gpus, "started": _now(), **fields}


def _wave_document(
    gpus: int,
    results: dict[str, dict[str, Any]],
    *,
    skipped_unchanged: tuple[str, ...] | list[str] = (),
    change_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The wave summary: one row per recipe, the corpus keys (fingerprints and engine versions), the verdict."""
    rows = [results[recipe_id] for recipe_id in sorted(results)]
    fingerprints = {row["recipe"]: row["behaviour_fingerprint"] for row in rows if row.get("behaviour_fingerprint")}
    engine_versions = {row["recipe"]: row["engine_version"] for row in rows if row.get("behaviour_fingerprint")}
    return {
        "gpus": gpus,
        "recipes": rows,
        "skipped_unchanged": list(skipped_unchanged),
        "changes": (change_verdict or {}).get("changes", {}),
        "protocol_due": (change_verdict or {}).get("protocol_due", []),
        "fingerprints": fingerprints,
        "engine_versions": engine_versions,
        "control_blockers": {
            row["recipe"]: row["steps"]["controls"]["blockers"]
            for row in rows
            if (row.get("steps") or {}).get("controls", {}).get("blockers")
        },
        "passed": (bool(rows) or bool(skipped_unchanged)) and all(row["state"] == "verified" for row in rows),
        "finished": _now(),
    }


def _wave_markdown(document: dict[str, Any]) -> str:
    """The human-readable wave summary table."""
    lines = [
        "# Wave summary",
        "",
        f"- GPUs: {document['gpus']}",
        f"- finished: {document['finished']}",
        "",
        "| recipe | gpus | state | error |",
        "|---|---|---|---|",
    ]
    for row in document["recipes"]:
        # The error cell is one table line however the message wraps (pydantic's are multi-line).
        error = " ".join((row.get("error") or "").split()).replace("|", "\\|")
        gpus = row.get("gpus")
        lines.append(f"| {row['recipe']} | {gpus if gpus is not None else '-'} | {row['state']} | {error} |")
    for recipe_id, blockers in sorted((document.get("control_blockers") or {}).items()):
        for blocker in blockers:
            lines.append(f"\n- BLOCKER {recipe_id} control {blocker['control']} {blocker['name']}: {blocker['reason']}")
    lines += ["", f"Verdict: **{'PASS' if document['passed'] else 'FAIL'}**"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_vllm.jobs.run_wave``; exit 0 only when every recipe verified."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_vllm.jobs.run_wave",
        description="Run many serving recipes on one node's GPUs: serve, smoke, equivalence, record.",
    )
    parser.add_argument("--recipes", default="", help="comma-separated recipe ids, or @file with one id per line")
    parser.add_argument("--recipes-root", default=None, help="root of recipe directories (default: the package's)")
    parser.add_argument("--gpus", type=int, default=8, help="the node's GPU count")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--upload", default=None, help="URI to copy <out> to after each recipe")
    parser.add_argument("--record", action="store_true", help="record the engine request/response set per recipe")
    parser.add_argument(
        "--record-corpus",
        action="store_true",
        help="write one observation corpus per recipe (the request plan's rows twice in one process and "
        "once after an engine restart, plus the protocol probes and /tokenize)",
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="serve the negative controls (a)-(f) per recipe through the ordinary gates; a control that passes "
        "fails the recipe (GPU-VALIDATION.md item 5)",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="run the T3 quality stage per recipe (served path vs the mteb reference on its task-matrix tasks)",
    )
    parser.add_argument(
        "--paper-numbers",
        default=None,
        help="JSON {recipe: {metric: {subset: value}}}: the paper's stored per-subset numbers the T3 stage gates",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        help="a previous wave.json or corpus index: re-record only the recipes whose behaviour "
        "fingerprint changed (OBSERVATIONS-SPEC section 7)",
    )
    parser.add_argument("--pairs-dir", default=None, help="directory with <id>.jsonl (or default.jsonl) pairs files")
    parser.add_argument(
        "--reference-python",
        required=True,
        help="the python that runs the recipe's references (its environment carries torch/transformers); "
        "the reference subprocess runs after that recipe's smoke pass, while the engine is up",
    )
    parser.add_argument("--vllm-cmd", default=None, help="replace the 'vllm serve' launcher (tests: a stub engine)")
    parser.add_argument("--port-base", type=int, default=8100, help="first engine port (0: engines announce theirs)")
    parser.add_argument(
        "--failed-plugins",
        default=None,
        help="file with one plugin spec per line the bootstrap could not install; the recipes naming "
        "them fail early with the plugin's exact name, the rest of the wave runs",
    )
    args = parser.parse_args(argv)
    try:
        ids = parse_ids(args.recipes)
        failed_plugins = frozenset(parse_ids(f"@{args.failed_plugins}")) if args.failed_plugins else frozenset()
        document = run_wave(
            ids,
            args.recipes_root,
            gpus=args.gpus,
            out_dir=args.out,
            upload=args.upload,
            record=args.record,
            record_corpus=args.record_corpus,
            quality=args.quality,
            controls=args.controls,
            paper_numbers=args.paper_numbers,
            changed_since_index=args.changed_since,
            pairs_dir=args.pairs_dir,
            reference_python=args.reference_python,
            vllm_cmd=args.vllm_cmd,
            port_base=args.port_base,
            failed_plugins=failed_plugins,
        )
    except (HarnessError, RecipeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for row in document["recipes"]:
        print(f"{row['recipe']}: {row['state']}")
    print(f"wave: {'PASS' if document['passed'] else 'FAIL'}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
