"""The wave runner: many recipes on one node's GPUs, one failing recipe never stops the wave.

For every recipe it packs the engine onto ``resources.gpus`` GPUs (``--tensor-parallel-size`` follows the recipe),
starts one ``vllm serve`` per slot from :func:`~rcp_ndcg_vllm.recipe.serve_argv` with its own
``CUDA_VISIBLE_DEVICES`` and port (``--port-base`` + slot; default 8100; ``--port-base 0`` gives every engine port 0), waits for
``GET /v1/models`` within the
recipe's ``engine.startup_timeout_s`` (an engine that exits early fails that recipe only), then runs smoke,
equivalence (stages 1 and 2) and — with ``--record`` — the recorder, stops the engine's process group, and moves
on.  It writes ``<out>/<id>/{serve.log, equivalence.json, EQUIVALENCE.md, status.json}``, a wave summary
(``wave.json`` and ``WAVE.md``), and with ``--upload`` copies ``<out>`` to the URI after each recipe
(``gcloud storage cp -r`` with a ``gsutil -m cp -r`` fallback).

Test mode: ``--vllm-cmd "python tests/stub_engine.py"`` replaces the ``vllm serve`` launcher with that command
(the rest of the rendered argv is appended, so a stub engine receives the real flags and may ignore them), and
``--port-base 0`` gives every engine ``--port 0``; such an engine must announce its bound port by printing
``RCPS_STUB_PORT=<n>`` on stdout, which the runner reads instead of guessing a port.

Run it on the node with ``python -m rcp_ndcg_vllm.jobs.run_wave`` (``bootstrap.sh`` does).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..equivalence import load_pairs
from ..equivalence import run as run_equivalence
from ..equivalence.client import EngineClient
from ..errors import HarnessError
from ..recipe import Recipe, RecipeError, default_recipes_root, iter_recipes, load_recipe, serve_argv
from ..record import record as record_exchanges

__all__ = ["main", "run_wave"]

_POLL_S = 2.0
_ANNOUNCE_TIMEOUT_S = 60.0


def run_wave(
    recipe_ids: list[str],
    recipes_root: str | Path | None = None,
    *,
    gpus: int,
    out_dir: str | Path,
    upload: str | None = None,
    record: bool = False,
    pairs_dir: str | Path | None = None,
    vllm_cmd: str | None = None,
    port_base: int = 8100,
) -> dict[str, Any]:
    """Run one wave: every recipe on the node's GPUs, as parallel as the GPUs allow.

    Inputs: the recipe ids (directories under ``recipes_root``; empty means every recipe there), the GPU count and
    the output directory.  Output: the wave document (also ``wave.json`` and ``WAVE.md`` under ``out_dir``); a
    recipe's own failure is recorded in its status and never raises.  Raises :class:`HarnessError` only for a bad
    wave request: an unknown recipe id or a missing recipe root.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recipes = _resolve_recipes(recipe_ids, recipes_root)
    results: dict[str, dict[str, Any]] = {}
    used_gpus: set[int] = set()
    pending = list(recipes)
    running: list[_EngineRun] = []
    slot = 0
    while pending or running:
        for recipe in list(pending):
            need = recipe.resources.gpus
            if need > gpus:
                results[recipe.id] = _status(
                    recipe, "failed", error=f"needs {need} GPUs, the wave has {gpus}", steps={}
                )
                directory = out / recipe.id
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "status.json").write_text(json.dumps(results[recipe.id], indent=2) + "\n", encoding="utf-8")
                pending.remove(recipe)
                continue
            if len(used_gpus) + need <= gpus:
                assigned = _lowest_free(used_gpus, need)
                used_gpus.update(assigned)
                running.append(_start(recipe, assigned, slot, out, vllm_cmd, port_base))
                slot += 1
                pending.remove(recipe)
        progressed = False
        for run in list(running):
            error: str | None = None
            if run.exited():
                error = f"the engine exited early (code {run.return_code()})"
            elif run.timed_out():
                error = f"GET /v1/models not ready within {run.timeout_s:.0f}s"
            if run.exited() or run.timed_out() or run.ready():
                _finalise(run, results, out, pairs_dir=pairs_dir, record=record, error=error)
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
    document = _wave_document(gpus, results)
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
    ) -> None:
        self.recipe = recipe
        self.gpus = gpus
        self.port = port
        self.popen = popen
        self.log_path = log_path
        self.out_dir = out_dir
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
        try:
            with EngineClient(self.recipe, f"http://127.0.0.1:{port}", served_model_name=self.recipe.id) as client:
                client.models()
        except Exception:  # noqa: BLE001 - a not-yet-ready engine is not an error
            return False
        return True

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


def _lowest_free(used: set[int], count: int) -> list[int]:
    """The lowest GPU indices not in use."""
    free: list[int] = []
    candidate = 0
    while len(free) < count:
        if candidate not in used:
            free.append(candidate)
        candidate += 1
    return free


def _start(recipe: Recipe, gpus: list[int], slot: int, out: Path, vllm_cmd: str | None, port_base: int) -> _EngineRun:
    """Start one engine on the given GPUs; the port is ``port_base + slot``, or 0 (the engine announces) in test mode."""
    port = port_base if port_base == 0 else port_base + slot
    argv = serve_argv(recipe, port=port, served_model_name=recipe.id)
    if vllm_cmd:
        argv = [*shlex.split(vllm_cmd), *argv[2:]]
    directory = out / recipe.id
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    run = _EngineRun(
        recipe,
        gpus,
        port,
        subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True),
        directory / "serve.log",
        directory,
    )
    run.status["serve_argv"] = argv
    run.status["steps"]["serve"] = {"state": "running", "port": port, "gpus": gpus}
    return run


def _finalise(
    run: _EngineRun,
    results: dict[str, dict[str, Any]],
    out: Path,
    *,
    pairs_dir: str | Path | None = None,
    record: bool = False,
    error: str | None = None,
) -> None:
    """Take one engine to its end state: run the steps, or record the failure, then stop it."""
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
            run.status["steps"]["equivalence"] = _equivalence(run.recipe, base_url, out, pairs_dir)
            if record:
                run.status["steps"]["record"] = _record(run.recipe, base_url, out)
            run.status["state"] = "verified" if run.status["steps"]["equivalence"].get("passed") else "failed"
        else:
            run.status["state"] = "failed"
            run.status["error"] = error
            run.status["steps"]["serve"] = {"state": "failed", "error": error}
    except HarnessError as step_error:
        run.status["state"] = "failed"
        run.status["error"] = str(step_error)
    finally:
        run.stop()
        run.status["finished"] = _now()
        _write_status(run)
        results[run.recipe.id] = run.status


def _write_status(run: _EngineRun) -> None:
    """One recipe's final status file."""
    path = run.out_dir / "status.json"
    path.write_text(json.dumps(run.status, indent=2) + "\n", encoding="utf-8")


def _smoke(recipe: Recipe, base_url: str) -> dict[str, Any]:
    """One minimal request per role: the engine serves, the route answers, the body parses."""
    try:
        with EngineClient(recipe, base_url, served_model_name=recipe.id) as client:
            if recipe.role == "rerank":
                scores = client.rerank("smoke query", ["smoke document"])
                ok = len(scores) == 1
            elif recipe.role == "embed":
                vectors = client.embeddings(["smoke text"])
                ok = len(vectors) == 1 and vectors[0].size > 0
            else:
                items = client.pooling(["smoke text"])
                ok = len(items) == 1
    except HarnessError as error:
        return {"state": "failed", "error": str(error)}
    return {"state": "passed" if ok else "failed"}


def _equivalence(recipe: Recipe, base_url: str, out: Path, pairs_dir: str | Path | None) -> dict[str, Any]:
    """Stages 1 and 2 for one recipe, written to ``<out>/<id>/equivalence.json``."""
    pairs_path = _pairs_path(recipe, pairs_dir)
    if pairs_path is None:
        return {"state": "skipped", "reason": "no pairs file; give --pairs-dir"}
    try:
        pairs = load_pairs(pairs_path)
        document = run_equivalence(
            recipe,
            base_url=base_url,
            pairs=pairs,
            out_dir=Path(out) / recipe.id,
            stages=[1, 2],
            served_model_name=recipe.id,
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
        written = record_exchanges(recipe, base_url, out, served_model_name=recipe.id)
    except HarnessError as error:
        return {"state": "failed", "error": str(error)}
    return {"state": "passed", "files": [str(path) for path in written]}


def _resolve_recipes(recipe_ids: list[str], recipes_root: str | Path | None) -> list[Recipe]:
    """The wave's recipes: the named ids under the root (every recipe there when the list is empty)."""
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    if recipe_ids:
        recipes = [_load_one(root, recipe_id) for recipe_id in recipe_ids]
    else:
        recipes = list(iter_recipes(root))
    if not recipes:
        raise HarnessError(f"no recipes under {root}")
    return recipes


def _load_one(root: Path, recipe_id: str) -> Recipe:
    """One recipe by id, with a clear wave-level error when the id does not resolve."""
    try:
        return load_recipe(root / recipe_id)
    except RecipeError as error:
        raise HarnessError(f"unknown recipe id {recipe_id!r} under {root}: {error}") from error


def _upload(out: Path, uri: str) -> None:
    """Copy ``<out>``'s contents to ``uri``: gcloud first, gsutil as the fallback; failures only warn."""
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
    print(f"[wave] upload to {uri} failed (gcloud and gsutil); the wave continues", file=sys.stderr)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _status(recipe: Recipe, state: str, **fields: Any) -> dict[str, Any]:
    return {"recipe": recipe.id, "state": state, "gpus": recipe.resources.gpus, "started": _now(), **fields}


def _wave_document(gpus: int, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The wave summary: one row per recipe, the wave's verdict last."""
    rows = [results[recipe_id] for recipe_id in sorted(results)]
    return {
        "gpus": gpus,
        "recipes": rows,
        "passed": bool(rows) and all(row["state"] == "verified" for row in rows),
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
        error = (row.get("error") or "").replace("|", "\\|")
        lines.append(f"| {row['recipe']} | {row['gpus']} | {row['state']} | {error} |")
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
    parser.add_argument("--pairs-dir", default=None, help="directory with <id>.jsonl (or default.jsonl) pairs files")
    parser.add_argument("--vllm-cmd", default=None, help="replace the 'vllm serve' launcher (tests: a stub engine)")
    parser.add_argument("--port-base", type=int, default=8100, help="first engine port (0: engines announce theirs)")
    args = parser.parse_args(argv)
    ids = _recipe_ids(args.recipes)
    try:
        document = run_wave(
            ids,
            args.recipes_root,
            gpus=args.gpus,
            out_dir=args.out,
            upload=args.upload,
            record=args.record,
            pairs_dir=args.pairs_dir,
            vllm_cmd=args.vllm_cmd,
            port_base=args.port_base,
        )
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for row in document["recipes"]:
        print(f"{row['recipe']}: {row['state']}")
    print(f"wave: {'PASS' if document['passed'] else 'FAIL'}")
    return 0 if document["passed"] else 1


def _recipe_ids(value: str) -> list[str]:
    """``a,b`` or ``@file.txt`` (one id per line, # comments allowed) into a list."""
    if value.startswith("@"):
        lines = Path(value[1:]).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
