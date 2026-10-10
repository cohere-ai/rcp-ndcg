"""The wave runner: many recipes on one node's GPUs, one failing recipe never stops the wave.

For every recipe it packs the engine onto ``resources.gpus`` GPUs (``--tensor-parallel-size`` follows the recipe),
starts one ``vllm serve`` per slot from :func:`~rcp_ndcg_vllm.recipe.serve_argv` with its own
``CUDA_VISIBLE_DEVICES``, port (``--port-base`` + slot; default 8100; ``--port-base 0`` gives every engine port 0),
``VLLM_PORT`` and ``TMPDIR`` (one home per slot: two engines cannot collide), waits for ``GET /v1/models`` within the
recipe's ``engine.startup_timeout_s`` (an engine that exits early fails that recipe only), then runs smoke,
equivalence (stages 1 and 2) and — with ``--record`` — the recorder, stops the engine's process group, and moves
on.  A recipe that cannot run at all — it fails validation when loaded, or the bootstrap recorded its
``serve.plugin`` among ``--failed-plugins`` (installed from the staged tree or wheelhouse only) — is a failed row
in the wave report with the validation message or the plugin's exact name; the wave runs the rest.

Every recipe's steps run in a worker thread of their own, once its engine is ready (GPU-E1: one stuck
request must not hold the other recipes' steps).  Every step carries a declared wall-clock budget --
the runner's formula from the recipe's request count (:data:`_STEP_BASE_S` plus :data:`_STEP_PER_REQUEST_S`
per request), which ``engine.step_budget_s`` in the recipe can only raise -- and the budget is enforced at
the requests through the :mod:`rcp_ndcg_test.stepwatch` seam: an overrunning step fails with
``step <name> exceeded <budget>s; in flight: <method path, request index>``, the engine stops, and the
other recipes continue.  The pod log gets one ``run_wave: <recipe> <step> start|passed|failed <secs>s``
line per step (no request bodies, no environment values), ``status.json`` is written atomically after
every step, and with ``--upload`` each finished recipe's directory is copied to the URI the moment the
recipe ends (a cancelled pod keeps the evidence of everything that finished; GPU-E1: results used to land
only at the end).  Every upload is verified against the destination and retried with backoff; the outcome
is recorded in the recipe's ``status.json`` row, the wave summary is written BEFORE the last upload so
``wave.json``/``WAVE.md`` reach the URI, and a wave with a failed upload does not pass (B1).  An engine
that dies mid-run fails only its recipe's ``serve`` step, with the engine's
last log lines in ``serve.log`` and a tail of them in the status; the others continue (GPU-E1: one
engine's CUDA fault took down the pod).  The reference subprocess gets a GPU of its own beside the
engine's (never the engine's GPU, which holds 90 % of its memory), pinned by ``CUDA_VISIBLE_DEVICES`` and
recorded in ``equivalence.json``; the packing reserves it (8 GPUs: at most 4 single-GPU recipes per pod
when each needs a reference GPU), and a recipe declaring ``reference.device: cuda`` refuses a CPU
reference run with the way out.  The harness's own requests (smoke, record, the corpus's bare probes) run
with the declared per-request timeout :data:`_REQUEST_TIMEOUT_S`, shorter than every step budget, and
reported in the step documents.

The pod has no persistent volume (node-runtime item 8): before each recipe the runner measures the free
disk and the model's Hub size and fails the recipe early when it measurably cannot fit (on a fresh pod
the cache does not exist yet, so the measurement lands on the nearest existing parent); after a recipe
whose model no later recipe reuses, the model's weights are evicted from the HF cache.  It writes
``<out>/<id>/{serve.log, equivalence.json, EQUIVALENCE.md, status.json}`` and a wave summary
(``wave.json`` and ``WAVE.md``: one row per recipe, the corpus fingerprints and the engine versions the
pods reported, and the verdict -- PASS, FAIL, or SKIPPED for an all-skipped ``--changed-since`` wave),
and with ``--upload`` copies ``<out>`` to the URI (``gcloud storage cp -r``, then a
``gsutil -m cp -r`` fallback, then the product's own :mod:`rcp_ndcg.storage` - the stock engine image
ships neither CLI).

The recording keys come from the pod, never the declared image: the corpus is keyed by the version the
running engine reports on its ``/version`` route (the engine environment's own ``vllm`` is the fallback
probe), and when neither answers the corpus step fails instead of keying by ``engine.image`` (B5).  The
staged plugin wheel's modules are hashed the same way the behaviour fingerprint hashes the plugin source
and a mismatch refuses the recording (item 9).

Test mode: ``--vllm-cmd "python tests/stub_engine.py"`` replaces the ``vllm serve`` launcher with that command
(the rest of the rendered argv is appended, so a stub engine receives the real flags and may ignore them), and
``--port-base 0`` gives every engine ``--port 0``; such an engine must announce its bound port by printing
``RCPS_STUB_PORT=<n>`` on stdout, which the runner reads instead of guessing a port.

Run it on the node with ``python -m rcp_ndcg_test.jobs.run_wave`` (the node bootstrap's wave mode does; see
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
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.patches import PATCHES_ENV, patches_env_value
from rcp_ndcg_vllm.recipe import Recipe, default_recipes_root, serve_argv

from rcp_ndcg_test.equivalence.reference import reference_of
from rcp_ndcg_test.errors import HarnessError, RecipeError
from rcp_ndcg_test.stepwatch import StepBudgetExceeded, StepWatch, current_watch, watched

from ..equivalence import run as run_equivalence
from ..record import record as record_exchanges
from .plugins import spec_of as _plugin_spec_of
from .wavelist import load_wave, parse_ids

__all__ = ["main", "run_wave"]

_POLL_S = 2.0
_ANNOUNCE_TIMEOUT_S = 60.0

_REQUEST_TIMEOUT_S = 120.0
"""The harness's own per-request timeout, seconds (smoke, record, the corpus's bare probes): declared
here, shorter than every step budget (GPU-E1 finding 7), and reported in the step documents."""

_STEP_BASE_S = 600.0
_STEP_PER_REQUEST_S = 60.0
"""The step budget's formula, seconds: a base that covers the reference subprocess's model load, plus one
allowance per request the step sends (from the recipe's request count); ``engine.step_budget_s`` in the
recipe raises it and never lowers it."""

_RECORD_REQUESTS = 5
"""The recorder's fixed request set (the provenance route, the role route, the two error probes, the
replied role request): the request count its step budget is computed from."""

_CORPUS_PASSES = 3
_CORPUS_PROBES = 8
"""The corpus step sends the plan's rows once per pass (two in-process, one after restart) plus the
standing protocol probes: the request count its step budget is computed from."""

_UPLOAD_ATTEMPTS = 3
_UPLOAD_BACKOFF_S = 2.0
_UPLOAD_TIMEOUT_S = 300.0
"""Every upload is verified and retried up to this many attempts, with exponential backoff (2s, 4s);
a transfer that reports success but does not leave the files at the destination is a failed attempt
(B1: the old upload was one fire-and-forget copy whose failure was a stderr line nobody read).  One CLI
attempt is bounded by :data:`_UPLOAD_TIMEOUT_S` (a hung gcloud/gsutil is a failed attempt, not a stuck
wave); the python fallback writes through the product's storage, whose fsspec layer has no transfer
timeout of its own -- a stalled fallback transfer is still a stall (pre-existing, recorded as an open
item; the CLI path is the one this bound closes)."""

_LOG_TAIL_LINES = 50
_LOG_TAIL_WIDTH = 300
"""A dead engine's evidence: its last log lines (GPU-E1), each clipped, in the serve step's document."""

_ZMQ_IPC_SUFFIX_CHARS = 37
"""One vLLM ZMQ IPC socket path under a slot's TMPDIR: ``/`` plus the 36-character uuid.  AF_UNIX's
``sun_path`` caps total paths at 107 characters, so a slot's TMPDIR must leave this much room
(``<slot tmpdir>`` + this <= 107)."""


def _slot_tmp_dir(slot: int, *, wave: str = "") -> Path:
    """One engine slot's TMPDIR: short, unique per wave and slot, outside the output tree.

    vLLM's ZMQ IPC sockets live under the slot's TMPDIR as ``<uuid>`` and AF_UNIX caps paths at 107
    characters - a TMPDIR of ``<out>/<recipe-id>/tmp`` blows the cap for long recipe ids (an engine
    died on exactly that path shape once).  The directory is ``<system temp>/rcp-s<pid>-<wave>-<slot>``
    (≈ 30 characters): whatever the recipe id and the state prefix are.  ``wave`` is the wave's own short
    token (one per :func:`run_wave` call), so two waves in one process never share a slot path: a previous
    wave's leftover engine or abandoned thread can neither remove nor reuse the TMPDIR the next wave's
    engine runs with (the shared ``rcp-s<pid>-<slot>`` path was exactly that hazard, and a test session
    runs many waves in one process).  The runner removes it with its engine (it is scratch).  Inputs: the
    slot index, the wave's token.  Output: the directory (not yet created).  Units: none.
    """
    token = f"-{wave}" if wave else ""
    return Path(tempfile.gettempdir()) / f"rcp-s{os.getpid()}{token}-{slot}"


def _log(message: str) -> None:
    """One pod-log line for the operator: ``run_wave: ...`` (no request bodies, no environment values)."""
    print(f"run_wave: {message}", flush=True)


def _step_budget_s(recipe: Recipe, requests: int) -> float:
    """One step's declared wall-clock budget, seconds: the base plus one allowance per request (the
    recipe's request count), never below the recipe's own floor (``engine.step_budget_s``)."""
    computed = _STEP_BASE_S + _STEP_PER_REQUEST_S * max(requests, 1)
    return float(max(recipe.engine.step_budget_s or 0, computed))


def _reference_needs_gpu(recipe: Recipe) -> bool:
    """Whether the recipe's reference runs as a subprocess (and so gets a GPU of its own when one is
    spare): every kind but ``stored_scores`` (whose scores need no model run)."""
    return recipe.reference is not None and recipe.reference.kind != "stored_scores"


def _reference_device(recipe: Recipe, reference_gpu: int | None) -> str:
    """The device the reference runs on: the recipe's declared device, else ``cuda`` when a GPU is
    reserved for it, else ``cpu``.  A judge recipe has no reference (its equivalence step is skipped by
    design, decision 15), so the device is ``cpu`` and never raises. Units: none."""
    if recipe.reference is None:
        return "cpu"
    return recipe.reference.device or ("cuda" if reference_gpu is not None else "cpu")


def _reference_python_for(
    recipe: Recipe, reference_python: str | None, reference_root: str | Path | None
) -> str | None:
    """The python that runs ``recipe``'s reference: the explicit ``--reference-python`` when given, else
    the family's environment under ``--reference-root`` (owner decision 35: one venv per family,
    ``<reference-root>/<family>/bin/python``).  ``None`` when neither is given (stage 2 then fails with
    the way out)."""
    if reference_python is not None:
        return reference_python
    if reference_root is None:
        return None
    directory = recipe._dir
    if directory is None:  # pragma: no cover - load_recipe sets it
        return None
    return str(Path(reference_root) / directory.name / "bin" / "python")


@dataclass
class _Wave:
    """One :func:`run_wave` call's identity: its slot-TMPDIR token and whether it has closed.

    The closing state is per wave, never one module-wide ``Event`` a later wave reopens for an earlier
    wave's abandoned thread: a step body the executor abandoned can outlive its wave (GPU-E1), and with a
    shared flag its late ``_start`` would be admitted into whichever wave happens to be open -- an engine
    started under a recipe no wave is tracking, on GPUs no wave booked, with a TMPDIR the earlier wave's
    cleanup removes.  :data:`_CURRENT_WAVE` names the wave that is open now; a start from any other (or a
    closed one) is refused.  ``token`` keeps every wave's slot TMPDIRs apart (:func:`_slot_tmp_dir`).
    """

    token: str
    closed: bool = False


_CURRENT_WAVE: list[_Wave | None] = [None]
"""The wave that is open now (``None`` between waves): what :func:`_start` checks a start against."""


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
    reference_root: str | Path | None = None,
    reference_store: str | Path | None = None,
    vllm_cmd: str | None = None,
    port_base: int = 8100,
    failed_plugins: Iterable[str] = (),
    plugin_wheel: str | Path | None = None,
) -> dict[str, Any]:
    """Run one wave: every recipe on the node's GPUs, as parallel as the GPUs allow.

    Inputs: the recipe ids (directories under ``recipes_root``; empty means every recipe there), the GPU count,
    the output directory, and the plugin specs the bootstrap could not install (the recipes that name one fail
    early, with its exact name).  Output: the wave document (also ``wave.json`` and ``WAVE.md`` under ``out_dir``);
    a recipe's own failure — including a recipe that fails validation at load — is recorded in its status and
    never raises.  Raises :class:`HarnessError` only for a bad wave request: a missing recipe root, or a wave
    list with no recipes at all (an unknown or invalid id is a failed row, not a wave abort).

    Each ready recipe's steps run in a worker thread of their own, under per-step budgets (GPU-E1), and each
    finished recipe's directory is uploaded the moment it ends.

    ``record_corpus`` writes one observation corpus per recipe under ``<out>/<id>/corpus/`` (the request
    plan's rows twice in one process and once after an engine restart -- the runner stops and restarts
    the engine between the passes).  ``--changed-since <index>`` re-records only the recipes whose
    behaviour fingerprint differs from the index (a previous ``wave.json`` or corpus index), listing
    the rest as ``skipped_unchanged`` (OBSERVATIONS-SPEC section 7).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The stored reference outputs (decision 35 item 3): default under the wave's own output, so they
    # upload with everything else; --reference-store reuses a previous wave's downloaded store.
    store = Path(reference_store) if reference_store is not None else out / "references"
    _MODEL_SIZES.clear()  # each wave asks the Hub for its models' sizes once
    wave = _Wave(token=os.urandom(3).hex())
    _CURRENT_WAVE[0] = wave  # this wave is the only one a start may join; the previous wave's is stale
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
        _retire(results, out, _failed_row(recipe_id, message), upload)
    for recipe in list(recipes):
        spec = _uninstalled_plugin(recipe, not_installed, root)
        if spec is not None:
            error = f"the recipe's plugin {spec} is not staged and not in the staged wheelhouse"
            _retire(
                results,
                out,
                _status(recipe, "failed", error=error, steps={"serve": {"state": "failed", "error": error}}),
                upload,
            )
            recipes.remove(recipe)
    used_gpus: set[int] = set()
    pending = list(recipes)
    running: list[_EngineRun] = []
    workers: list[_Worker] = []
    slot = 0
    try:
        while pending or running or workers:
            progressed = False
            for recipe in list(pending):
                engine_gpus = recipe.resources.gpus
                ref_needed = _reference_needs_gpu(recipe)
                # A recipe declaring reference.device: cpu gets no reference GPU: the reservation is
                # only for references that may run on one (GPU-E1: the runner gives each reference a GPU
                # of its own; the recipe's declared device wins).
                reserved = ref_needed and reference_of(recipe).device != "cpu" and engine_gpus + 1 <= gpus
                if ref_needed and not reserved and reference_of(recipe).device == "cuda":
                    error = (
                        f"recipe {recipe.id} declares reference.device: cuda, but the pod's {gpus} GPU(s) "
                        f"cannot give the reference one of its own beside the engine's {engine_gpus}; "
                        "pack fewer engines per pod or raise the pod's GPU count"
                    )
                    _retire(
                        results,
                        out,
                        _status(recipe, "failed", error=error, steps={"serve": {"state": "failed", "error": error}}),
                        upload,
                    )
                    pending.remove(recipe)
                    progressed = True
                    continue
                need = engine_gpus + (1 if reserved else 0)
                if need > gpus:
                    error = f"the engine needs {engine_gpus} GPUs, the wave has {gpus}"
                    _retire(
                        results,
                        out,
                        _status(recipe, "failed", error=error, steps={"serve": {"state": "failed", "error": error}}),
                        upload,
                    )
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
                        _retire(results, out, row, upload)
                        pending.remove(recipe)
                        progressed = True
                        continue
                    assigned = _lowest_free(used_gpus, need)
                    used_gpus.update(assigned)
                    try:
                        run = _start(
                            recipe, assigned[:engine_gpus], slot, out, vllm_cmd, port_base, disk=disk, wave=wave
                        )
                    except HarnessError as start_error:
                        # An engine that cannot even start (no vllm binary) fails that recipe only.
                        row = _status(recipe, "failed", error=str(start_error), steps={"serve": {"state": "failed"}})
                        _retire(results, out, row, upload)
                        used_gpus.difference_update(assigned)
                        pending.remove(recipe)
                        progressed = True
                        continue
                    # GPU-E1: the reference gets a GPU of its own beside the engine's, never the engine's.
                    run.reference_gpu = assigned[engine_gpus] if reserved else None
                    run.held_gpus = list(assigned)  # everything the recipe holds: engine + reference
                    running.append(run)
                    slot += 1
                    pending.remove(recipe)
                    progressed = True
            for run in list(running):
                error: str | None = None
                if run.exited():
                    error = f"the engine exited early (code {run.return_code()})"
                elif run.timed_out():
                    error = f"GET /v1/models not ready within {run.timeout_s:.0f}s"
                if run.exited() or run.timed_out() or run.ready():
                    running.remove(run)
                    # One home per concept (item 8): the model's weights stay while any other recipe in
                    # this wave still needs them (queued, served, or in its steps); otherwise they are
                    # evicted when the recipe's steps end.
                    reuse = any(other.model == run.recipe.model for other in pending) or any(
                        other.recipe.model == run.recipe.model
                        for other in running + [worker.run for worker in workers]
                        if other is not run
                    )
                    worker = _Worker(
                        run,
                        error,
                        out=out,
                        pairs_dir=pairs_dir,
                        record=record,
                        record_corpus=record_corpus,
                        quality=quality,
                        paper_numbers=paper_numbers,
                        controls=controls,
                        vllm_cmd=vllm_cmd,
                        port_base=port_base,
                        reference_python=_reference_python_for(run.recipe, reference_python, reference_root),
                        reference_root=reference_root,
                        reference_store=store,
                        reuse=reuse,
                        plugin_wheel=plugin_wheel,
                    )
                    workers.append(worker)
                    worker.start()
                    progressed = True
            for worker in list(workers):
                if worker.done():
                    worker.join(1.0)
                    workers.remove(worker)
                    used_gpus.difference_update(worker.run.held_gpus or worker.run.gpus)
                    row = worker.run.status
                    results[worker.run.recipe.id] = row
                    if upload is not None:
                        # GPU-E1: each finished recipe's directory lands the moment the recipe ends, so a
                        # cancelled or killed pod keeps the evidence of everything that finished.  The
                        # attempt is verified and retried, and its outcome is recorded in the row's
                        # status.json (B1: a failed upload used to be a stderr line nobody read).
                        row["upload"] = _upload_recipe(out, worker.run.recipe.id, upload)
                        _publish_status(out / worker.run.recipe.id / "status.json", row)
                    progressed = True
            if not progressed:
                time.sleep(_POLL_S)
    finally:
        # The wave leaves no engine behind, whatever happened to the runner: no engine may start once
        # the wave closes, and every engine still registered is stopped (an abandoned corpus body's
        # restart is caught here even after its worker's snapshot).
        wave.closed = True
        for run in list(running):
            run.stop()
        for worker in list(workers):
            worker.stop_engines()
        with _LIVE_LOCK:
            leftover = list(_LIVE_ENGINES)
        for engine in leftover:
            engine.stop()
    if _CURRENT_WAVE[0] is wave:
        _CURRENT_WAVE[0] = None  # between waves: no start joins a wave that has ended
    document = _wave_document(gpus, results, skipped_unchanged=skipped_unchanged, change_verdict=change_verdict)
    upload_failures = {
        recipe_id: row["upload"]
        for recipe_id, row in results.items()
        if isinstance(row.get("upload"), dict) and not row["upload"].get("ok")
    }
    if upload_failures:
        # B1: an upload that failed is the wave's failure too; the summary names it and main exits non-zero.
        document["upload_failures"] = upload_failures
        document["passed"] = False
        document["verdict"] = "failed"

    def write_summary() -> None:
        (out / "wave.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        (out / "WAVE.md").write_text(_wave_markdown(document), encoding="utf-8")

    # The summary is written BEFORE the last upload, so wave.json/WAVE.md reach the URI (B1: the old
    # order uploaded first and wrote the summary after, so the destination never held it).
    write_summary()
    if upload is not None:
        wave_upload = _upload(out, upload)
        document["upload"] = wave_upload
        if not wave_upload["ok"]:
            document["passed"] = False
            document["verdict"] = "failed"
        write_summary()
        if wave_upload["ok"] and not document["passed"]:
            # The destination holds the provisional summary (written before the upload); land the
            # corrected one that names the failed recipe uploads.  Best effort: its own outcome is
            # already recorded above.
            _upload(out, upload)
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
        wave: _Wave | None = None,
    ) -> None:
        self.recipe = recipe
        self.gpus = gpus
        self.port = port
        self.popen = popen
        self.log_path = log_path
        self.out_dir = out_dir
        self.tmpdir = Path(tmpdir) if tmpdir is not None else log_path.parent / "tmp"
        self.wave = wave if wave is not None else _Wave(token="unknown")
        self._stop_lock = threading.Lock()
        """Serializes this engine's teardown: the worker's ``_stop_after_failure`` and the wave's end can
        stop the same engine at once, and one teardown must not interleave with the other's signal."""
        self.disk: dict[str, Any] = disk or {}
        self.env: dict[str, str] = {}
        self.reference_gpu: int | None = None
        """The physical GPU the recipe's reference subprocess is pinned to (its own, never the engine's)."""
        self.held_gpus: list[int] = list(gpus)
        """Everything the recipe holds until its steps end: the engine's GPUs plus the reference's."""
        self.stopped_by_runner = False
        """Whether the runner stopped this engine on purpose (a step overrun, the recipe's end) -- as
        opposed to the engine dying on its own, which is the serve step's failure evidence."""
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
        """Stop the engine's whole process group: SIGTERM, then SIGKILL after a grace period.

        The engine runs in its own session (``start_new_session`` at start), so its process group IS its
        pid: the signal goes to ``popen.pid`` directly, never through ``os.getpgid`` -- that lookup is a
        second syscall whose answer can name another process group if the child's number was reused in
        between, and the stop is called from the worker and the wave's end at once, so the whole teardown
        is serialized on this engine's lock (a double stop signals once).  An engine's death never takes
        another one down (GPU-E1).  A deliberate stop is recorded as such: it is not the engine's own
        death.  Stopping removes the engine from the wave's registry (the wave's end sweeps whatever is
        left).
        """
        with self._stop_lock:
            with _LIVE_LOCK:
                _LIVE_ENGINES.discard(self)
            if self.popen.poll() is not None:
                return  # already gone: not the runner's doing, so the death evidence stays a death
            self.stopped_by_runner = True
            try:
                os.killpg(self.popen.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):  # pragma: no cover - the engine already died
                self.popen.kill()
            try:
                self.popen.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - a stuck engine
                try:
                    os.killpg(self.popen.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.popen.wait()
            self._thread.join(timeout=5)


class _Worker:
    """One recipe's steps in a thread of their own, each under its declared wall-clock budget.

    GPU-E1: the runner worked through the recipes' steps serially, so one stuck request blocked every
    other recipe's steps.  A ready recipe's engine is handed to a worker here; the steps run in the
    worker's thread, every step under its :class:`~rcp_ndcg_test.stepwatch.StepWatch` (the overrun fails
    the step with the in-flight request, the engine stops, the other recipes continue), ``status.json``
    is rewritten atomically after every step, and every step's start and end lands on the pod log.
    """

    def __init__(
        self,
        run: _EngineRun,
        serve_error: str | None,
        *,
        out: Path,
        pairs_dir: str | Path | None,
        record: bool,
        record_corpus: bool,
        quality: bool,
        paper_numbers: str | Path | None,
        controls: bool,
        vllm_cmd: str | None,
        port_base: int,
        reference_python: str | None,
        reference_root: str | Path | None,
        reference_store: str | Path | None,
        reuse: bool,
        plugin_wheel: str | Path | None,
    ) -> None:
        self.run = run
        self._serve_error = serve_error
        self.out = out
        self.pairs_dir = pairs_dir
        self.record = record
        self.record_corpus = record_corpus
        self.quality = quality
        self.paper_numbers = paper_numbers
        self.controls = controls
        self.vllm_cmd = vllm_cmd
        self.port_base = port_base
        self.reference_python = reference_python
        self.reference_root = reference_root
        self.reference_store = reference_store
        self.reuse = reuse
        self.plugin_wheel = plugin_wheel
        self.restarted: list[_EngineRun] = []
        self.corpus_fingerprint: str | None = None
        """The behaviour fingerprint of the corpus step's result (the step body's side channel: the step
        document goes through the ordinary ``_step`` machinery, the fingerprint into the row)."""
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True, name=f"run_wave:{run.recipe.id}")

    def start(self) -> None:
        """Start the worker's thread (one per recipe slot)."""
        self._thread.start()

    def done(self) -> bool:
        """Whether the recipe's steps have ended."""
        return self._done.is_set()

    def join(self, timeout_s: float) -> None:
        """Wait for the worker's thread (the reaper gives it a moment to finish its last write)."""
        self._thread.join(timeout_s)

    def stop_engines(self) -> None:
        """Stop this recipe's engines (its own and any the corpus step restarted); idempotent.  The
        snapshot matters: an abandoned corpus body may still append to ``restarted`` from its thread."""
        self.run.stop()
        for extra in list(self.restarted):
            extra.stop()

    # -- the steps -------------------------------------------------------------------------------------------

    def _work(self) -> None:
        """The recipe's steps, or the serve-level failure, then the finish; never raises."""
        run = self.run
        try:
            self._steps()
        except Exception as step_error:  # noqa: BLE001 - one recipe's failure never stops the wave
            run.status["state"] = "failed"
            run.status["error"] = f"{type(step_error).__name__}: {step_error}"
        finally:
            try:
                self._finish()
            except Exception as finish_error:  # noqa: BLE001 - the finish records, never raises
                run.status["state"] = "failed"
                run.status.setdefault("error", f"{type(finish_error).__name__}: {finish_error}")
            finally:
                self._done.set()

    def _steps(self) -> None:
        """The steps in order, each under its budget; an overrun or an engine death ends the recipe."""
        run = self.run
        error = self._serve_error
        run.status["ready_wait_s"] = round(time.monotonic() - run.started, 3)
        if error is None and run.port == 0:
            announced = run.announced_port(_ANNOUNCE_TIMEOUT_S)
            if announced is None:
                error = "the engine did not announce RCPS_STUB_PORT (test mode)"
            elif not run._models_ok(announced):
                error = f"GET /v1/models not ready on the announced port {announced}"
            else:
                run.port = announced
                run.status["port"] = announced
        if error is not None:
            run.status["state"] = "failed"
            run.status["error"] = error
            run.status["steps"]["serve"] = {"state": "failed", "error": error}
            return
        base_url = f"http://127.0.0.1:{run.port}"
        recipe = run.recipe
        # B5: the recording is keyed by the version the RUNNING engine reports (its /version route, then
        # the engine environment's own vllm), never the recipe's declared image.  A missing version is
        # recorded here and fails the corpus step; smoke and equivalence still run.
        version = _pod_engine_version(run, self.vllm_cmd)
        if version is None:
            run.status["engine_version_error"] = (
                "the engine's /version route and the engine environment's vLLM version both failed to "
                "answer; the recording cannot be keyed by the declared image"
            )
        else:
            run.status["engine_version"] = version
        rows = self._pair_rows()
        served: list[dict[str, Any]] = []  # the corpus step checks its replies against stage 2's exchanges
        self._step("smoke", _step_budget_s(recipe, 1), lambda: _smoke(recipe, base_url))
        if self._stop_after_failure("smoke"):
            return
        self._step(
            "equivalence",
            _step_budget_s(recipe, rows),
            lambda: _equivalence(
                recipe,
                base_url,
                self.out,
                self.pairs_dir,
                self.reference_python,
                device=_reference_device(run.recipe, run.reference_gpu),
                reference_gpu=run.reference_gpu,
                recorder=served if self.record_corpus else None,
                reference_store=self.reference_store,
                reference_environment=_environment_facts(recipe, self.reference_root),
            ),
        )
        if self._stop_after_failure("equivalence"):
            return
        if self.record and recipe.role != "judge":
            self._step("record", _step_budget_s(recipe, _RECORD_REQUESTS), lambda: _record(recipe, base_url, self.out))
            if self._stop_after_failure("record"):
                return
        if self.quality and recipe.role != "judge":
            self._step(
                "quality",
                _step_budget_s(recipe, max(rows, 1)),
                lambda: _quality(run, base_url, self.out, self.reference_python, self.paper_numbers, self.vllm_cmd),
            )
            if self._stop_after_failure("quality"):
                return
        if self.record_corpus and recipe.role != "judge":
            self._step(
                "observation_corpus",
                _step_budget_s(recipe, _CORPUS_PASSES * max(rows, 1) + _CORPUS_PROBES),
                lambda: self._observe_corpus(served),
            )
            run.status["behaviour_fingerprint"] = self.corpus_fingerprint
            self._write_status()
            if self._stop_after_failure("observation_corpus"):
                return
        if self.controls and recipe.role != "judge":
            # Last: the recipe-variant controls take the slot's GPUs one engine at a time (one owner).
            self._step(
                "controls",
                _step_budget_s(recipe, 2 * rows + 4),
                lambda: _controls(
                    run,
                    self.out,
                    self.pairs_dir,
                    self.reference_python,
                    vllm_cmd=self.vllm_cmd,
                    port_base=self.port_base,
                    restarted=self.restarted,
                ),
            )
            if self._stop_after_failure("controls"):
                return
        steps = run.status["steps"]
        run.status["state"] = _row_state(
            run.recipe,
            steps,
            record=self.record,
            record_corpus=self.record_corpus,
            quality=self.quality,
            controls=self.controls,
        )

    def _stop_after_failure(self, step: str) -> bool:
        """Whether a step's failure must end the recipe: a budget overrun or an engine DEATH stops the
        engine now (nothing waits silently); an ordinary step failure lets the next steps try, and an
        engine the runner stopped on purpose (the controls stop it after their variants) is no death."""
        run = self.run
        result = run.status["steps"].get(step) or {}
        overrun = "exceeded" in str(result.get("error") or "")
        if overrun or self._engine_death() is not None:
            run.status["state"] = "failed"
            if overrun:
                run.status.setdefault("error", f"{step}: {result.get('error')}")
            run.stop()
            for extra in list(self.restarted):
                extra.stop()
            return True
        return False

    def _step(self, name: str, budget_s: float, body: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """One step under its declared budget: the pod-log lines, the watch, the overrun failure, and the
        status rewritten the moment the step ends (GPU-E1: results used to land only at the end)."""
        run = self.run
        started = time.monotonic()
        run.status["steps"][name] = {"state": "running", "budget_s": round(budget_s)}
        self._write_status()
        self._log(f"{name} start")
        watch = StepWatch(name, budget_s)
        box: dict[str, Any] = {}

        def body_under_watch() -> None:
            with watched(watch):
                try:
                    box["result"] = body()
                except BaseException as error:  # noqa: BLE001 - the executor decides what it means
                    box["error"] = error

        helper = threading.Thread(target=body_under_watch, daemon=True, name=f"run_wave:{run.recipe.id}:{name}")
        helper.start()
        while helper.is_alive() and watch.elapsed_s() <= budget_s:
            if self._engine_death() is not None:
                break  # the engine died: fail the step now, never wait the budget out (GPU-E1)
            helper.join(0.5)
        death = self._engine_death() if helper.is_alive() else None
        overrun = helper.is_alive() and death is None
        if death is not None:
            # The product's transport parks a request while every replica is down (wait_on_outage_s);
            # with the engine dead that park never returns, so the abandoned helper is left to the wave's
            # end -- nothing it holds (no live engine, no GPU) can stall another recipe.
            helper.join(1.0)
        elif overrun:
            # The budget is out and the body is still going: the step fails HERE, whatever the body
            # eventually returns (a late pass never overrides the budget).  A short grace lets the
            # cooperative seams unwind (the transport cancels the in-flight request at the budget's
            # edge and names it); the wave's sweep owns whatever a body the watch cannot reach still
            # holds.
            helper.join(30.0)
        result: dict[str, Any]
        if death is not None:
            result = {"state": "failed", "error": death}
        elif overrun:
            # A cooperative seam's own message names the request that was in flight when the budget ran
            # out; without one, the watch's in-flight description is the best the runner has.
            late = box.get("error")
            error_text = (
                str(late)
                if isinstance(late, StepBudgetExceeded)
                else f"step {name} exceeded {budget_s:.0f}s; in flight: {watch.describe()}"
            )
            result = {"state": "failed", "error": error_text}
        elif "error" in box:
            error = box["error"]
            if isinstance(error, StepBudgetExceeded):
                result = {"state": "failed", "error": str(error)}
            elif isinstance(error, HarnessError):
                result = {"state": "failed", "error": str(error)}
            else:
                raise error
        else:
            result = box.get("result") or {"state": "failed", "error": f"step {name} produced no result"}
        secs = time.monotonic() - started
        result.setdefault("budget_s", round(budget_s))
        result["secs"] = round(secs, 3)
        self._log(f"{name} {result.get('state', 'failed')} {secs:.1f}s")
        run.status["steps"][name] = result
        self._write_status()
        return result

    def _log(self, message: str) -> None:
        """The pod-log line for one step boundary (finding 3: the pod log was silent)."""
        _log(f"{self.run.recipe.id} {message}")

    def _pair_rows(self) -> int:
        """The recipe's request count: its pairs file's rows (0 without one -- the budget falls back to
        the formula's base and the recipe's own floor)."""
        pairs_path = _pairs_path(self.run.recipe, self.pairs_dir)
        if pairs_path is None:
            return 0
        try:
            from ..equivalence.fitting import load_pairs

            return len(load_pairs(pairs_path))
        except (HarnessError, OSError, ValueError):
            return 0

    def _observe_corpus(self, equivalence_exchanges: list[dict[str, Any]]) -> dict[str, Any]:
        """The observation corpus step (the runner's own sequencing, in the worker's thread); the
        fingerprint lands in :attr:`corpus_fingerprint` for the row."""
        run = self.run
        step, fingerprint = _observe_corpus(
            run,
            self.out,
            self.pairs_dir,
            vllm_cmd=self.vllm_cmd,
            port_base=self.port_base,
            restarted=self.restarted,
            equivalence_exchanges=equivalence_exchanges or None,
            plugin_wheel=self.plugin_wheel,
        )
        self.corpus_fingerprint = fingerprint
        return step

    def _write_status(self) -> None:
        """The recipe's status file, atomically, after every step (GPU-E1: results used to land only at
        the end, so a killed pod left nothing)."""
        _publish_status(self.run.out_dir / "status.json", self.run.status)

    def _engine_death(self) -> str | None:
        """The engine's death while its steps ran, or ``None`` while it lives -- or after the runner
        stopped it on purpose (a step overrun, the recipe's end): a deliberate stop is no crash."""
        run = self.run
        if run.stopped_by_runner or not run.exited():
            return None
        return f"the engine exited (code {run.return_code()}) while its steps ran"

    def _finish(self) -> None:
        """The recipe's end state: the serve step's verdict, the row's error composition, the engines
        stopped, the scratch removed, the eviction recorded, the final status written."""
        run = self.run
        death = self._engine_death()
        if death is not None:
            # GPU-E4: an engine's death fails its recipe's serve step, with the engine's last log lines
            # in serve.log and a tail of them in the status; the other recipes continue.
            serve = run.status["steps"].get("serve") or {}
            run.status["steps"]["serve"] = {
                **serve,
                "state": "failed",
                "error": serve.get("error") or death,
                "log_tail": _log_tail(run.log_path),
            }
            run.status["state"] = "failed"
            run.status["error"] = run.status.get("error") or f"serve: {death}"
        # The serve step records ITS outcome: "the engine answered and was stopped cleanly" is a success,
        # whatever a later step's verdict is - a clean stop is not a failure; an engine that never became
        # ready (self._serve_error) failed it.
        _mark_serve_step(run, "failed" if (death is not None or self._serve_error) else "passed")
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
        self.stop_engines()
        shutil.rmtree(run.tmpdir, ignore_errors=True)  # the slot's scratch TMPDIR leaves with its engine
        run.status["disk"] = {**run.disk, **_evict(run.recipe, reuse=self.reuse)}
        run.status["finished"] = _now()
        self._write_status()


def _failed_row(recipe_id: str, error: str) -> dict[str, Any]:
    """A failed wave-report row for a recipe that never ran (no :class:`Recipe` could be loaded for it)."""
    return {"recipe": recipe_id, "state": "failed", "gpus": None, "started": _now(), "error": error, "steps": {}}


def _row_state(
    recipe: Recipe,
    steps: dict[str, Any],
    *,
    record: bool,
    record_corpus: bool,
    quality: bool,
    controls: bool,
) -> str:
    """A recipe's end state: ``verified`` when every step its role requires passed.

    A judge recipe's equivalence and recorder steps are skipped by design (decision 15: a judge has no
    reference and no observation corpus; its conformance is ``rcp-ndcg judge check`` and the T4 scenarios),
    so their skip counts as satisfied for a judge; for every other role a skipped equivalence or record
    step leaves the row unverified. Units: none.
    """
    judge = recipe.role == "judge"
    equivalence = steps.get("equivalence") or {}
    equivalence_ok = equivalence.get("state") == "skipped" if judge else bool(equivalence.get("passed"))
    record_ok = judge or not record or (steps.get("record") or {}).get("state") == "passed"
    controls_ok = not controls or (steps.get("controls") or {}).get("state") == "passed"
    corpus_ok = not record_corpus or (steps.get("observation_corpus") or {}).get("state") != "failed"
    quality_ok = not quality or (steps.get("quality") or {}).get("state") == "passed"
    verified = (
        (steps.get("smoke") or {}).get("state") == "passed"
        and equivalence_ok
        and record_ok
        and corpus_ok
        and quality_ok
        and controls_ok
    )
    return "verified" if verified else "failed"


def _uninstalled_plugin(recipe: Recipe, failed_plugins: frozenset[str], root: Path) -> str | None:
    """The recipe's plugin exact name when the bootstrap recorded it as not installable.

    The match is exactly the form ``jobs.plugins`` collects for THIS recipe (its staged file as
    ``<recipe-directory>/<file>``, else the bare spec), so one recipe's failed bare name never fails a recipe
    whose own collected form installed fine.  The row's message carries the exact name from the
    recipe.  Units: none.
    """
    spec = _plugin_spec_of(recipe, root)
    return recipe.serve.plugin if spec is not None and spec in failed_plugins else None


def _retire(results: dict[str, dict[str, Any]], out: Path, row: dict[str, Any], upload: str | None) -> None:
    """Record a recipe that finished without ever starting an engine: the results map, its status file,
    and (with ``--upload``) its directory, the moment it is finished; the upload's outcome is recorded
    in the row and its status file (B1)."""
    results[row["recipe"]] = row
    directory = out / row["recipe"]
    directory.mkdir(parents=True, exist_ok=True)
    _publish_status(directory / "status.json", row)
    if upload is not None:
        row["upload"] = _upload_recipe(out, row["recipe"], upload)
        _publish_status(directory / "status.json", row)


def _publish_status(path: Path, document: dict[str, Any]) -> None:
    """One recipe's status file, atomically (the product's ``storage.publish_bytes`` is the one home of
    the discipline): a reader sees the previous file or the complete new one, never a half-written one."""
    from rcp_ndcg import storage

    storage.publish_bytes(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))


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
    wave: _Wave,
) -> _EngineRun:
    """Start one engine on the given GPUs; the port is ``port_base + slot``, or 0 (announced) in test mode.

    Node-runtime item 7: every slot gets its own ``CUDA_VISIBLE_DEVICES``, HTTP port, ``VLLM_PORT`` (the
    engine's internal port) and ``TMPDIR``, so two engines on one node cannot collide on any of them.
    The engine runs in its own session and process group (GPU-E1: one engine's crash must never take
    another one down), teed into its own ``serve.log``.  ``wave`` is the wave starting it: a closed wave,
    or one that is no longer the open wave (an abandoned step body's late restart), may not start an
    engine -- it would run under a recipe no wave tracks and a TMPDIR its own wave's cleanup removes.
    """
    if wave.closed or wave is not _CURRENT_WAVE[0]:
        raise HarnessError(
            f"the wave {wave.token} is closed (or a later wave is open); the engine for {recipe.id} may not start"
        )
    port = port_base if port_base == 0 else port_base + slot
    argv = serve_argv(recipe, port=port, served_model_name=recipe.id)
    if vllm_cmd:
        argv = [*shlex.split(vllm_cmd), *argv[2:]]
    directory = out / recipe.id
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    # The recipe's declared patches are the engine's exact opt-in: every engine-start path renders them the
    # same way the serve console does, so the process that records a corpus runs the code the fingerprint
    # keys (a hand-set RCP_NDCG_VLLM_PATCHES is overridden, never silently added to).
    env[PATCHES_ENV] = patches_env_value(recipe.serve.patches)
    # One home per slot, kept SHORT and outside the output tree: the slot's TMPDIR carries vLLM's ZMQ
    # IPC sockets, whose paths must fit AF_UNIX's 107 characters whatever the recipe id is.  The wave's
    # token keeps two waves in one process off each other's paths.
    tmpdir = _slot_tmp_dir(slot, wave=wave.token)
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
        wave=wave,
    )
    with _LIVE_LOCK:
        _LIVE_ENGINES.add(run)
    run.status["serve_argv"] = argv
    run.env = env
    run.status["steps"]["serve"] = {"state": "running", "port": port, "gpus": gpus, "tmpdir": str(tmpdir)}
    # The serve step's boundary goes on the pod log and its running state on disk at once: an engine's
    # load can take minutes, and the log/status must not be silent while it does (GPU-E1 finding 3).
    _log(f"{recipe.id} serve start")
    _publish_status(directory / "status.json", run.status)
    return run


_LIVE_ENGINES: set[_EngineRun] = set()
_LIVE_LOCK = threading.Lock()
"""Every engine the wave started, and whether the wave is winding down (GPU-E1: an abandoned corpus
body can call :func:`_start` after its worker's snapshot, so the wave's end sweeps the registry and no
engine may start once the wave closes -- the wave leaves no engine behind)."""

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
    """Record the serve step's final state (its slot's TMPDIR, its error and log tail when it failed)
    and put its boundary on the pod log."""
    step = run.status["steps"].get("serve") or {}
    secs = time.monotonic() - run.started
    run.status["steps"]["serve"] = {
        "state": state,
        "port": run.port,
        "gpus": run.gpus,
        "tmpdir": str(run.tmpdir),
        "secs": round(secs, 3),
        **{key: step[key] for key in ("error", "log_tail") if step.get(key)},
    }
    _log(f"{run.recipe.id} serve {state} {secs:.1f}s")


def _log_tail(log_path: Path, lines: int = _LOG_TAIL_LINES, width: int = _LOG_TAIL_WIDTH) -> list[str]:
    """The engine log's last ``lines`` lines, each clipped to ``width`` characters (the crash evidence
    the status carries; the full log stays in ``serve.log``)."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line[-width:] for line in text.splitlines()[-lines:]]


def _smoke(recipe: Recipe, base_url: str) -> dict[str, Any]:
    """One minimal request per role: the engine serves, the route answers, the body parses.

    A multi-vector recipe whose pooling wire carries the role-prefixed token ids
    (``client.request_shape: token_ids``) is smoked with the recipe's own query render as ids
    (:func:`_smoke_query_ids`): a bare-text body is a request the recipe's client never sends, and
    such an engine's pooler refuses it by name -- the refusal kills the EngineCore and the recipe's
    serve step fails (the r2 wave's pplx-embed-v2-context failure).

    The harness's own request runs under the declared per-request timeout :data:`_REQUEST_TIMEOUT_S`
    (shorter than every step budget, GPU-E1 finding 7), which the step document reports.
    """
    import httpx

    routes: dict[str, tuple[str, dict[str, Any]]] = {
        "rerank": (
            "/rerank",
            {"model": recipe.id, "query": "smoke query", "documents": ["smoke document"], "top_n": 1},
        ),
        "embed": ("/v1/embeddings", {"model": recipe.id, "input": ["smoke text"]}),
        "multi_vector": ("/pooling", {"model": recipe.id, "input": ["smoke text"], "task": "token_embed"}),
        "judge": (
            "/v1/chat/completions",
            {"model": recipe.id, "messages": [{"role": "user", "content": "smoke"}], "max_tokens": 8},
        ),
    }
    route, body = routes[recipe.role]
    if recipe.role == "multi_vector" and recipe.client.get("request_shape") == "token_ids":
        body = {**body, "input": [_smoke_query_ids(recipe)]}
    try:
        root = base_url.rstrip("/")
        if root.endswith(("/v1", "/v2")):
            root = root.rsplit("/", 1)[0]
        with httpx.Client(base_url=root, timeout=_REQUEST_TIMEOUT_S) as http:
            watch = current_watch()
            if watch is not None:
                watch.begin("POST", route)
            try:
                reply = http.post(route, json=body)
            finally:
                if watch is not None:
                    watch.end()
        ok = reply.status_code == 200
    except httpx.HTTPError as error:
        return {"state": "failed", "error": str(error), "request_timeout_s": _REQUEST_TIMEOUT_S}
    return {"state": "passed" if ok else "failed", "request_timeout_s": _REQUEST_TIMEOUT_S}


def _smoke_query_ids(recipe: Recipe) -> list[int]:
    """The recipe's own query render as token ids (the ``token_ids`` wire's smoke body).

    The render and the ids come from the product's template and tokenizer -- the same pair the
    client's fit uses (R30), with the shape's declared ``add_special_tokens`` flag -- so the probe
    sends exactly the wire the recipe's client sends, never a re-derived one.
    """
    from rcp_ndcg_test.equivalence import fitting

    template = fitting.client_template(recipe)
    if template is None:  # pragma: no cover - the endpoint config requires a template for this wire
        raise HarnessError(f"recipe {recipe.id}: request_shape token_ids needs the client template")
    tokenizer = fitting.tokenizer_of(recipe)
    text = template.render("query", tokenizer, query="smoke query")
    return list(tokenizer.ids(text, add_special_tokens=template.adds_special_tokens("query")))


def _equivalence(
    recipe: Recipe,
    base_url: str,
    out: Path,
    pairs_dir: str | Path | None,
    reference_python: str | None,
    *,
    device: str,
    reference_gpu: int | None,
    recorder: list[dict[str, Any]] | None = None,
    reference_store: str | Path | None = None,
    reference_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stages 1 and 2 for one recipe, written to ``<out>/<id>/equivalence.json``; the reference runs on
    ``device`` (pinned to ``reference_gpu`` when the runner reserved one) and the report records both;
    ``recorder`` collects stage 2's captured exchanges (the corpus step checks its replies against them).
    ``reference_store`` reuses a stored reference output whose key is unchanged and stores the computed
    ones; ``reference_environment`` (the family's lock hash and freeze) is recorded in the report."""
    if recipe.role == "judge":
        return {
            "state": "skipped",
            "reason": (
                "a judge has no reference (decision 15); its conformance is `rcp-ndcg judge check` and the T4 scenarios"
            ),
        }
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
            device=device,
            reference_gpu=reference_gpu,
            reference_store=None if reference_store is None else str(reference_store),
            reference_environment=reference_environment,
        )
        return {
            "state": "passed" if document["passed"] else "failed",
            "passed": document["passed"],
            "stages": [1, 2],
            "reference_device": device,
            "reference_environment": document.get("reference_environment"),
            "reference_outputs": document.get("reference_outputs"),
            **({"reference_gpu": reference_gpu} if reference_gpu is not None else {}),
        }
    except HarnessError as error:
        return {"state": "failed", "error": str(error), "reference_device": device}


def _environment_facts(recipe: Recipe, reference_root: str | Path | None) -> dict[str, Any]:
    """The family's reference environment facts for one recipe (decision 35 item 5); ``{}`` without a
    family directory."""
    from .reference_env import environment_facts

    return environment_facts(recipe._dir, reference_root)


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
    """The recorder's fixed request set, written under ``<out>/<engine>-<version>/<recipe-id>/``; the
    harness's own requests run with the declared per-request timeout, reported in the step document."""
    try:
        written = record_exchanges(recipe, base_url, out, timeout_s=_REQUEST_TIMEOUT_S)
    except HarnessError as error:
        return {"state": "failed", "error": str(error), "request_timeout_s": _REQUEST_TIMEOUT_S}
    return {
        "state": "passed",
        "files": [str(path) for path in written],
        "request_timeout_s": _REQUEST_TIMEOUT_S,
    }


def _probe_engine_version(port: int) -> str | None:
    """The version the RUNNING engine reports on its ``/version`` route, or ``None`` when the route does
    not answer.  This is the recording key's source (B5): the declared image string is never it."""
    import httpx

    try:
        reply = httpx.get(f"http://127.0.0.1:{port}/version", timeout=10.0)
    except httpx.HTTPError:
        return None
    if reply.status_code != 200:
        return None
    try:
        version = str(reply.json().get("version") or "").strip()
    except ValueError:
        return None
    return version or None


def _engine_env_version() -> str | None:
    """The vLLM version the pod's engine environment reports (the provenance probe: ``import vllm`` in
    ``RCP_ENGINE_PYTHON``), or ``None``.  The pre-serve ``--changed-since`` selection uses it before any
    engine is up."""
    python = os.environ.get("RCP_ENGINE_PYTHON")
    if not python:
        return None
    try:
        completed = subprocess.run(
            [python, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _pod_engine_version(run: _EngineRun, vllm_cmd: str | None) -> str | None:
    """The engine version the running pod reports: the live engine's ``/version`` first, then the engine
    environment's own ``vllm`` (the provenance probe); ``test-stub`` in test mode.  ``None`` when neither
    answers -- the caller refuses to key a recording by the declared image (B5)."""
    if vllm_cmd:
        return "test-stub"
    version = _probe_engine_version(run.port)
    if version is not None:
        return version
    return _engine_env_version()


def _planned_engine_version(vllm_cmd: str | None) -> str | None:
    """The pod's engine version for the pre-serve ``--changed-since`` selection: the test stub, else the
    engine environment's own ``vllm`` (the same pod source the recording probes).  ``None`` when the pod's
    version cannot be determined before engines start -- the caller refuses rather than guessing the
    recipe's declared image."""
    if vllm_cmd:
        return "test-stub"
    return _engine_env_version()


def _staged_plugin_hashes(wheel: Path, modules: Iterable[str]) -> dict[str, str]:
    """The SHA-256 of each plugin module's source INSIDE the staged wheel, keyed by module name.

    The same canonicalisation as :func:`rcp_ndcg_test.fingerprint.plugin_module_hashes` (the module's
    source bytes through :func:`~rcp_ndcg_test.fingerprint.sha256_digest`), read from the wheel's zip
    members: ``<module>.py`` or, for a package, ``<module>/__init__.py``.  Raises :class:`HarnessError`
    when the wheel is not a readable zip or does not carry a module (the caller's step fails with the
    named reason, never an unhandled traceback).
    """
    from ..fingerprint import sha256_digest

    try:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            hashes: dict[str, str] = {}
            for module in modules:
                stem = module.replace(".", "/")
                for member in (f"{stem}.py", f"{stem}/__init__.py"):
                    if member in names:
                        hashes[module] = sha256_digest(archive.read(member))
                        break
                else:
                    raise HarnessError(f"the staged wheel {wheel} does not carry the plugin module {module}")
    except HarnessError:
        raise
    except Exception as error:  # noqa: BLE001 - any zip-layer failure is the named step refusal
        # BadZipFile, an unsupported compression method, an encrypted member, a truncated central
        # directory: every one is the wheel's failure, never an unhandled traceback that leaves the
        # step reading ``running``.
        raise HarnessError(
            f"the staged plugin wheel {wheel} cannot be read ({type(error).__name__}: {error})"
        ) from error
    return hashes


def _check_plugin_wheel(recipe: Recipe, wheel: Path) -> str | None:
    """Cross-check the staged wheel's plugin modules against the behaviour fingerprint's inputs.

    Inputs: the recipe and the staged wheel the engine environment installed.  Output: ``None`` when the
    recipe has no plugin or every ``plugin_sha256.<module>`` input matches the wheel's member; else the
    one-line mismatch naming the modules.  Item 9: the fingerprint keys the harness's resolved plugin
    source, the engine runs the staged wheel, and nothing else compares the two.
    """
    from ..fingerprint import plugin_module_hashes

    source_hashes = plugin_module_hashes(recipe)
    if not source_hashes:
        return None
    wheel_hashes = _staged_plugin_hashes(wheel, source_hashes)
    mismatches = sorted(module for module, digest in source_hashes.items() if wheel_hashes.get(module) != digest)
    if mismatches:
        return (
            f"the staged plugin wheel {wheel} does not carry the plugin code the behaviour fingerprint "
            f"keys (module(s): {', '.join(mismatches)}); the engine would run a different plugin build "
            "than the corpus records"
        )
    return None


def _observe_corpus(
    run: _EngineRun,
    out: Path,
    pairs_dir: str | Path | None,
    *,
    vllm_cmd: str | None,
    port_base: int,
    restarted: list[_EngineRun],
    equivalence_exchanges: list[dict[str, Any]] | None = None,
    plugin_wheel: str | Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    """One observation corpus for the recipe over the request plan's rows (OBSERVATIONS-SPEC 1-6).

    The corpus lands at its keyed, immutable path (:func:`~rcp_ndcg_test.observe.corpus.corpus_path`:
    ``<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<recorded-at>/``), keyed by the one behaviour
    fingerprint (:func:`rcp_ndcg_test.fingerprint.behaviour_fingerprint`).  The ``restart`` closure stops the
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
    if plugin_wheel is not None:
        wheel_path = Path(plugin_wheel)
        if not wheel_path.is_file():
            return {"state": "failed", "error": f"the staged plugin wheel {wheel_path} does not exist"}, fingerprint
        mismatch = _check_plugin_wheel(recipe, wheel_path)
        if mismatch is not None:
            # item 9: the fingerprint hashed one plugin build, the engine runs the wheel's; refuse to
            # record a corpus that would claim code the engine did not run.
            return {"state": "failed", "error": mismatch}, fingerprint
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
    version = run.status.get("engine_version")
    if not version:
        # B5: the corpus key is the pod's reported version; when neither probe answered, refuse to key by
        # the declared image (the caller's step fails; smoke and equivalence still ran).
        return {
            "state": "failed",
            "error": run.status.get("engine_version_error")
            or "the engine version was not probed; the corpus cannot be keyed by the declared image",
        }, fingerprint
    directory = corpus_path(out, version, recipe.id, fingerprint, started)

    loading: list[dict[str, Any]] = []

    def restart() -> tuple[str, str] | None:
        run.stop()
        fresh = _start(recipe, run.gpus, slot, out, vllm_cmd, port_base, disk=run.disk, wave=run.wave)
        restarted.append(fresh)
        # OBSERVATIONS-SPEC section 1's readiness edge: one request while the engine is still loading (a stub
        # in test mode announces its port first; a refused connection is recorded as such).
        port = fresh.announced_port(30.0) if fresh.port == 0 else fresh.port
        if port:
            import httpx

            from ..record import bare_exchange

            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=_REQUEST_TIMEOUT_S) as http:
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
        version=version,
    )
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
            timeout_s=_REQUEST_TIMEOUT_S,
            plugin_wheel=plugin_wheel,
        )
    except Exception as error:  # noqa: BLE001 - the corpus step fails this recipe, never the wave
        return {
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
            "request_timeout_s": _REQUEST_TIMEOUT_S,
        }, fingerprint
    return {
        "state": "passed" if report["passed"] else "failed",
        "request_timeout_s": _REQUEST_TIMEOUT_S,
        # item 9's cross-check is visible either way: the wheel that was hashed, or why it was not.
        "plugin_wheel": (
            str(plugin_wheel)
            if plugin_wheel is not None
            else "not given; the staged wheel was not cross-checked (give --plugin-wheel)"
            if recipe.serve.plugin is not None
            else "no plugin"
        ),
        **report,
    }, fingerprint


def _quality(
    run: _EngineRun,
    base_url: str,
    out: Path,
    reference_python: str | None,
    paper_numbers: str | Path | None,
    vllm_cmd: str | None,
) -> dict[str, Any]:
    """The T3 quality stage for one recipe (:func:`rcp_ndcg_test.quality.run_quality`) on its task-matrix tasks,
    written under ``<out>/<id>/quality/``.  The paper's stored per-subset numbers come from ``--paper-numbers``
    (``{recipe: {metric: {subset: value}}}``); the golden-replay corpus carries the wave's engine and collector
    blocks.  A recipe outside the task matrix, or any failing task, fails the step."""
    from .. import quality as t3
    from ..observe.provenance import collector_facts, engine_facts

    recipe = run.recipe
    if reference_python is None:
        return {"state": "failed", "error": "the quality stage needs --reference-python (the mteb reference)"}
    if not run.status.get("engine_version"):
        # F4: the quality manifest keys the engine block; without the pod's version it would fall back to
        # the declared image's tag.  Fail instead, with the same reason the corpus step uses.
        return {
            "state": "failed",
            "error": run.status.get("engine_version_error")
            or "the engine version was not probed; the quality stage cannot key by the declared image",
        }
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
                version=run.status.get("engine_version"),
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
    recipe: Recipe,
    base_url: str,
    out_dir: Path,
    pairs_path: Path,
    reference_python: str,
    *,
    device: str,
    reference_gpu: int | None,
) -> dict[str, Any]:
    """Stages 1 and 2 (and a media recipe's media stage) for one control: the ordinary gates, which must fail
    it.  An error the served side raises (a garbled frame the client cannot decode) is the stage failing on that
    request, recorded with its text.  The reference runs on the recipe's own device (and GPU), as its gates
    above did: a control judged on the wrong device would fail for the device, not the control."""
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
            device=device,
            reference_gpu=reference_gpu,
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
    """The negative controls (a)-(g) of one recipe (GPU-VALIDATION.md item 5), through the ordinary gates.

    Only after the recipe's own gates passed (a control "caught" by a gate that fails everything proves nothing).
    Wire controls run against the recipe's live engine with the request bodies patched; then the recipe's
    engines stop and each recipe variant is served on the slot in turn (one GPU owner at a time), its client and
    engine agreeing on the variant's id.  A variant whose engine does not come up is NOT counted as caught (an
    engine refusing its argv fails every gate and proves nothing): its row says so and the summary flags it.
    Returns the step: :func:`~rcp_ndcg_test.observe.controls.controls_summary`'s report and its state.
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
    if live.exited():
        # An earlier step ended the engine (a failed corpus step): a control run against a stopped
        # engine would record connection failures, not catch a breakage -- skip, said why.
        return {"state": "skipped", "reason": "the recipe's engine is stopped (an earlier step ended it)"}
    live_url = f"http://127.0.0.1:{live.port}"
    device = _reference_device(recipe, run.reference_gpu)
    work = out / recipe.id / "controls"
    rows: list[dict[str, Any]] = []
    variants = control_variants(recipe)
    for variant in variants:
        if variant["kind"] is None:
            rows.append({"control": variant["control"], "name": variant["name"], "reason": variant["reason"]})
        elif variant["kind"] == "unresolved":
            # Undecidable is never inapplicable: the row counts as a control the gates did not catch.
            gates = {"passed": None, "error": variant["reason"]}
            rows.append({"control": variant["control"], "name": variant["name"], "equivalence": gates})
        elif variant["kind"] == "wire":
            with patched_wire(variant["wire_patch"]):
                gates = _control_gates(
                    recipe,
                    live_url,
                    work / variant["name"],
                    pairs_path,
                    reference_python,
                    device=device,
                    reference_gpu=run.reference_gpu,
                )
            rows.append({"control": variant["control"], "name": variant["name"], "equivalence": gates})
    run.stop()
    for engine in restarted:
        engine.stop()
    slot = max(run.port - port_base, 0) if port_base else 0
    for variant in variants:
        if variant["kind"] != "recipe":
            continue
        engine = _start(variant["recipe"], run.gpus, slot, out, vllm_cmd, port_base, disk=run.disk, wave=run.wave)
        try:
            deadline = time.monotonic() + run.timeout_s
            while not engine.ready() and not engine.exited() and time.monotonic() < deadline:
                time.sleep(_POLL_S)
            if engine.ready():
                gates = _control_gates(
                    variant["recipe"],
                    f"http://127.0.0.1:{engine.port}",
                    work / variant["name"],
                    pairs_path,
                    reference_python,
                    device=device,
                    reference_gpu=run.reference_gpu,
                )
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
    ids and the verdict (why each changed, which engine versions are new) for the wave document.  The engine
    version is the pod's (the engine environment's own ``vllm``, the same source the recording probes), never
    the recipe's declared image; without it the selection refuses rather than guessing."""
    from ..observe.corpus import changed_since

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"--changed-since {index_path} is unreadable: {error}") from error
    planned = _planned_engine_version(vllm_cmd)
    if planned is None:
        raise HarnessError(
            "--changed-since needs the pod's engine version before engines start; set RCP_ENGINE_PYTHON "
            "(bootstrap exports the engine environment's python) or run with --vllm-cmd"
        )
    try:
        verdict = changed_since(recipes, index, engine_version_of=lambda recipe: planned)
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


def _upload(out: Path, uri: str) -> dict[str, Any]:
    """Copy ``<out>``'s contents to ``uri``, verified and retried with backoff (B1).

    Inputs: the local directory and the destination URI.  Output: the attempt record
    ``{"ok", "attempts", "files", "error"}`` -- never a raise; a failed upload is the wave's failure
    (``run_wave`` folds it into the verdict and ``main`` exits non-zero).  The transfer tries gcloud,
    gsutil, then the product's own storage (the stock engine image ships neither CLI, and the client
    environment carries the product's gcsfs, so the third path is the node's usual one); after a
    transfer reports success the destination is listed and every source file must be there at the same
    size, so a silent partial copy is a failure, not a pass.
    """
    if not any(out.iterdir()):
        return {"ok": True, "attempts": 0, "files": 0, "error": None}
    return _upload_verified(out, uri)


def _upload_recipe(out: Path, recipe_id: str, uri: str) -> dict[str, Any]:
    """One finished recipe's directory to ``uri`` (the moment the recipe ends; GPU-E1), verified and
    retried; the returned record is stored in the recipe's status row."""
    directory = out / recipe_id
    if not directory.is_dir():
        return {"ok": False, "attempts": 0, "files": 0, "error": f"no directory for {recipe_id} to upload"}
    return _upload_verified(directory, f"{uri.rstrip('/')}/{recipe_id}")


def _upload_verified(source: Path, uri: str) -> dict[str, Any]:
    """Upload ``source`` to ``uri`` with :data:`_UPLOAD_ATTEMPTS` verified attempts and backoff.

    Output: ``{"ok": bool, "attempts": int, "files": int, "error": str | None}`` -- the last error is
    kept when every attempt failed.  A transfer that reports success but leaves the destination without
    the source files (or with different sizes) is a failed attempt: the retry is the answer to a
    transient 5xx, the verification to a silent partial copy.
    """
    files = sum(1 for path in source.rglob("*") if path.is_file())
    result: dict[str, Any] = {"ok": False, "attempts": 0, "files": files, "error": None}
    for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
        result["attempts"] = attempt
        error = _upload_attempt(source, uri)
        if error is None:
            error = _verify_upload(source, uri)
        if error is None:
            result["ok"] = True
            result["error"] = None
            return result
        result["error"] = error
        if attempt < _UPLOAD_ATTEMPTS and _UPLOAD_BACKOFF_S:
            time.sleep(_UPLOAD_BACKOFF_S * (2 ** (attempt - 1)))
    return result


def _upload_attempt(source: Path, uri: str) -> str | None:
    """One transfer attempt through the first mechanism that answers; ``None`` on success, else the
    one-line error (the python path's own message when it was the one that failed)."""
    if _upload_cli(f"{source}/*", f"{uri.rstrip('/')}/"):
        return None
    return _upload_storage(source, uri)


def _verify_upload(source: Path, uri: str) -> str | None:
    """``None`` when every local file under ``source`` is at ``uri`` with the same size; else the
    mismatch.  The listing goes through the product's storage (the one home for a URI), so the check
    works for a local directory and a ``gs://``/``s3://`` destination alike."""
    expected = {
        str(path.relative_to(source)): path.stat().st_size for path in sorted(source.rglob("*")) if path.is_file()
    }
    if not expected:
        return None
    try:
        from rcp_ndcg import storage
    except ImportError:
        return "the product's storage is not importable, so the upload cannot be verified"
    try:
        listed = storage.ls(uri, recursive=True)
    except Exception as error:  # noqa: BLE001 - any listing failure is a failed verification
        return f"the destination {uri} could not be listed ({type(error).__name__}: {error})"
    seen: dict[str, int | None] = {}
    for entry in listed:
        try:
            seen[storage.relative(entry, uri)] = storage.info(entry).get("size")
        except Exception as error:  # noqa: BLE001 - a broken entry is a failed verification
            return f"the destination entry {entry} could not be read ({type(error).__name__}: {error})"
    missing = sorted(set(expected) - set(seen))
    wrong_size = sorted(name for name, size in expected.items() if name in seen and seen[name] != size)
    if missing or wrong_size:
        return (
            f"the destination {uri} does not hold the source files "
            f"(missing: {missing[:5]}, size mismatch: {wrong_size[:5]})"
        )
    return None


def _upload_cli(source: str, target: str) -> bool:
    """One ``cp -r`` through the first CLI that answers (the stock image ships neither); a CLI that does
    not finish within :data:`_UPLOAD_TIMEOUT_S` is a failed attempt, so a hung transfer cannot stall the
    wave."""
    for argv in (
        ["gcloud", "storage", "cp", "-r", source, target],
        ["gsutil", "-m", "cp", "-r", source, target],
    ):
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=_UPLOAD_TIMEOUT_S)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print(
                f"[wave] {argv[0]} did not finish within {_UPLOAD_TIMEOUT_S:.0f}s; trying the next transfer",
                file=sys.stderr,
            )
            continue
        if completed.returncode == 0:
            return True
    return False


def _upload_storage(source: Path, uri: str) -> str | None:
    """The product's own storage as the last fallback: every local file under ``source`` written to
    ``uri`` through :mod:`rcp_ndcg.storage` (the one home for gs:// paths; gcsfs via ADC).  Returns
    ``None`` on success, else the one-line error.  This path carries no transfer timeout of its own
    (fsspec's default); a stalled transfer here is a stall."""
    try:
        from rcp_ndcg import storage
    except ImportError:
        return "the product's storage is not importable"
    try:
        storage.makedirs(f"{uri.rstrip('/')}/")
        for path in sorted(source.rglob("*")):
            if path.is_file():
                storage.write_bytes(f"{uri.rstrip('/')}/{path.relative_to(source)}", path.read_bytes())
    except Exception as error:  # noqa: BLE001 - the caller records the error, never raises
        return f"the python upload failed: {type(error).__name__}: {error}"
    return None


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
    """The wave summary: one row per recipe, the corpus keys (fingerprints and engine versions), the verdict.

    The verdict is ``passed`` only when at least one recipe ran and every one verified; ``skipped`` when the
    wave ran nothing because ``--changed-since`` skipped every recipe (it verified nothing, so it is never a
    PASS); ``failed`` otherwise.  ``passed`` is ``verdict == "passed"``.
    """
    rows = [results[recipe_id] for recipe_id in sorted(results)]
    fingerprints = {row["recipe"]: row["behaviour_fingerprint"] for row in rows if row.get("behaviour_fingerprint")}
    engine_versions = {row["recipe"]: row["engine_version"] for row in rows if row.get("engine_version")}
    verified = bool(rows) and all(row["state"] == "verified" for row in rows)
    skipped = not rows and bool(skipped_unchanged)
    verdict = "passed" if verified else ("skipped" if skipped else "failed")
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
        "verdict": verdict,
        "passed": verdict == "passed",
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
    if not document["recipes"] and document.get("skipped_unchanged"):
        lines.append("")
        lines.append(f"- skipped (unchanged): {', '.join(document['skipped_unchanged'])}")
    for recipe_id, blockers in sorted((document.get("control_blockers") or {}).items()):
        for blocker in blockers:
            lines.append(f"\n- BLOCKER {recipe_id} control {blocker['control']} {blocker['name']}: {blocker['reason']}")
    for recipe_id, result in sorted((document.get("upload_failures") or {}).items()):
        lines.append(f"\n- UPLOAD FAILED {recipe_id}: {result.get('error')}")
    wave_upload = document.get("upload") or {}
    if wave_upload.get("ok") is False:
        lines.append(f"\n- UPLOAD FAILED the wave summary: {wave_upload.get('error')}")
    verdict = str(document.get("verdict") or ("passed" if document["passed"] else "failed")).upper()
    lines += ["", f"Verdict: **{verdict}**"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_test.jobs.run_wave``; exit 0 only when every recipe verified.  An
    all-skipped ``--changed-since`` wave prints ``wave: SKIPPED`` and exits 1: it verified nothing."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.jobs.run_wave",
        description="Run many serving recipes on one node's GPUs: serve, smoke, equivalence, record.",
    )
    parser.add_argument("--recipes", default="", help="comma-separated recipe ids, or @file with one id per line")
    parser.add_argument("--recipes-root", default=None, help="root of recipe directories (default: the package's)")
    parser.add_argument("--gpus", type=int, default=8, help="the node's GPU count")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--upload",
        default=None,
        help="URI to copy each finished recipe's directory to as it finishes, and the wave summary at the "
        "end; uploads are verified and retried, and a failed upload fails the wave",
    )
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
        help="serve the negative controls (a)-(g) per recipe through the ordinary gates; a control that passes "
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
        "fingerprint or pod-reported engine version changed; an all-skipped wave reports SKIPPED",
    )
    parser.add_argument("--pairs-dir", default=None, help="directory with <id>.jsonl (or default.jsonl) pairs files")
    parser.add_argument(
        "--reference-python",
        default=None,
        help="the python that runs the recipe's references (its environment carries torch/transformers); "
        "the reference subprocess runs after that recipe's smoke pass, while the engine is up, on a GPU "
        "of its own beside the engine's.  Overrides --reference-root (tests use it); a wave normally "
        "passes --reference-root and the family's venv is resolved per recipe",
    )
    parser.add_argument(
        "--reference-root",
        default=None,
        help="the parent of the per-family reference environments built by the bootstrap "
        "(<root>/<family>/bin/python, owner decision 35); one of --reference-root/--reference-python "
        "is required for stage 2",
    )
    parser.add_argument(
        "--reference-store",
        default=None,
        help="the stored reference outputs' directory: stage 2 reuses a stored output whose key inputs "
        "(reference hash, revision, pairs hash, environment lock hash, device, dtype) are unchanged and "
        "records the newly computed ones (default: <out>/references)",
    )
    parser.add_argument("--vllm-cmd", default=None, help="replace the 'vllm serve' launcher (tests: a stub engine)")
    parser.add_argument("--port-base", type=int, default=8100, help="first engine port (0: engines announce theirs)")
    parser.add_argument(
        "--failed-plugins",
        default=None,
        help="file with one plugin spec per line the bootstrap could not install; the recipes naming "
        "them fail early with the plugin's exact name, the rest of the wave runs",
    )
    parser.add_argument(
        "--plugin-wheel",
        default=None,
        help="the staged plugin wheel the engine environment installed; the wave hashes its modules "
        "against the behaviour fingerprint's plugin inputs and refuses to record when they differ "
        "(the bootstrap passes the staged rcp_ndcg_vllm wheel)",
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
            reference_root=args.reference_root,
            reference_store=args.reference_store,
            vllm_cmd=args.vllm_cmd,
            port_base=args.port_base,
            failed_plugins=failed_plugins,
            plugin_wheel=args.plugin_wheel,
        )
    except (HarnessError, RecipeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for row in document["recipes"]:
        print(f"{row['recipe']}: {row['state']}")
    for recipe_id in document.get("skipped_unchanged") or []:
        print(f"{recipe_id}: skipped (unchanged)")
    for recipe_id, result in sorted((document.get("upload_failures") or {}).items()):
        print(f"upload {recipe_id}: FAILED ({result.get('error')})")
    wave_upload = document.get("upload") or {}
    if wave_upload.get("ok") is False:
        print(f"upload wave: FAILED ({wave_upload.get('error')})")
    verdict = str(document.get("verdict") or ("passed" if document["passed"] else "failed"))
    print(f"wave: {verdict.upper()}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
