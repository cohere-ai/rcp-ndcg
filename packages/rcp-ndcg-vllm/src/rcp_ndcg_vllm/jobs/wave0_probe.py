"""Wave 0's probes: the CPU-side work of the node test, one subcommand per report fragment.

``wave0.sh`` (the orchestrating shell script) calls this module with the interpreters it has at hand:
``host`` runs on the image's python3 before any environment exists (stdlib and ``nvidia-smi`` only);
everything else runs in the client environment the bootstrap built, through the product's own machinery
(:func:`rcp_ndcg.data.preprocess.fit`, :class:`~rcp_ndcg.inference.clients.EmbeddingClient`,
:mod:`rcp_ndcg.storage`, and the HF-cache helpers of :mod:`rcp_ndcg_vllm.jobs.weights`).

Every subcommand prints one JSON object (the report fragment) and exits non-zero on failure, so the
shell records it and fails fast with a one-line reason. No command prints a credential: the token is
read from the environment and never echoed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import HarnessError

__all__ = ["main"]

_ENGINE_TERM_S = 60.0
"""Seconds a stopped engine gets between ``SIGTERM`` and ``SIGKILL``."""


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_vllm.jobs.wave0_probe <subcommand>``; the fragment on stdout."""
    parser = argparse.ArgumentParser(prog="wave0_probe", description="Wave 0's step probes (one JSON fragment each).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gcs = sub.add_parser("gcs", help="the gs:// round-trip through rcp_ndcg.storage (write, list, read, delete)")
    p_gcs.add_argument("--out-uri", required=True, help="the wave's output prefix; the probe runs under it")
    p_gcs.add_argument("--report", required=True)

    p_start = sub.add_parser("engines-start", help="start the two slot engines and wait for both")
    p_start.add_argument("--spec", required=True, help="JSON: the slots' argv, ports, tmpdirs and log dirs")
    p_start.add_argument("--state", required=True, help="where the running engines' pids are recorded")
    p_start.add_argument("--report", required=True)

    p_stop = sub.add_parser("engines-stop", help="stop the engines and assert no engine process remains")
    p_stop.add_argument("--state", required=True, help="the file engines-start wrote")
    p_stop.add_argument("--report", required=True)

    p_embed = sub.add_parser("embed", help="fit the inputs under an explicit budget, embed, and /tokenize each")
    p_embed.add_argument("--base-url", required=True, help="the engine's root URL (http://127.0.0.1:<port>)")
    p_embed.add_argument("--served-model-name", required=True)
    p_embed.add_argument("--tokenizer", required=True, help="the tokenizer spec (repo@revision or a path)")
    p_embed.add_argument("--budget", type=int, required=True, help="the explicit text budget, in tokens")
    p_embed.add_argument("--count", type=int, default=20, help="how many inputs (default 20)")
    p_embed.add_argument("--over", type=int, default=5, help="how many of them over the budget (default 5)")
    p_embed.add_argument("--report", required=True)

    p_evict = sub.add_parser("evict", help="evict the wave's models from the HF cache, with the disk before/after")
    p_evict.add_argument("--model", action="append", required=True, help="model id (org/name), repeatable")
    p_evict.add_argument("--report", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "gcs":
            fragment = _gcs(args.out_uri, Path(args.report))
        elif args.command == "engines-start":
            fragment = _engines_start(Path(args.spec), Path(args.state), Path(args.report))
        elif args.command == "engines-stop":
            fragment = _engines_stop(Path(args.state), Path(args.report))
        elif args.command == "embed":
            fragment = _embed(args, Path(args.report))
        else:
            fragment = _evict(list(args.model), Path(args.report))
    except Exception as error:  # noqa: BLE001 - the fragment carries the failure, the shell fails fast
        fragment = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        _write(Path(args.report), fragment)
        print(json.dumps(fragment, indent=2))
        return 1
    print(json.dumps(fragment, indent=2))
    return 0 if fragment.get("passed", True) else 1


# --- (c) the GCS round-trip through the product's storage ----------------------------------------------


def _gcs(out_uri: str, report: Path) -> dict[str, Any]:
    """One gs:// object through :mod:`rcp_ndcg.storage`: write, list, read, delete - from the client env."""
    from rcp_ndcg import storage

    stamp = _now().replace(":", "").replace("-", "")
    prefix = f"{out_uri.rstrip('/')}/wave0/probe/{stamp}"
    path = f"{prefix}/roundtrip.txt"
    payload = f"wave-0 round-trip {stamp}\n".encode()
    storage.write_bytes(path, payload)
    listed = [entry for entry in storage.ls(prefix) if entry.endswith("roundtrip.txt")]
    read_back = storage.read_bytes(path)
    storage.filesystem(path).rm(path)  # the delete goes through the product's filesystem seam
    fragment = {
        "uri": prefix,
        "wrote_bytes": len(payload),
        "listed": len(listed) == 1,
        "read_equal": read_back == payload,
        "deleted": not storage.exists(path),
        "passed": len(listed) == 1 and read_back == payload and not storage.exists(path),
    }
    _write(report, fragment)
    return fragment


# --- (d) two engines, two slots, at the same time ------------------------------------------------------


def _engines_start(spec_path: Path, state_path: Path, report: Path) -> dict[str, Any]:
    """Start one engine per slot with the slot's own isolation, wait for both, record their facts."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    slots = spec["slots"]
    if len({slot["port"] for slot in slots}) != len(slots):
        raise HarnessError("two slots must not share an HTTP port")
    if len({slot["vllm_port"] for slot in slots}) != len(slots):
        raise HarnessError("two slots must not share VLLM_PORT")
    if len({slot["tmpdir"] for slot in slots}) != len(slots):
        raise HarnessError("two slots must not share TMPDIR")
    if len({slot["cuda_visible_devices"] for slot in slots}) != len(slots):
        raise HarnessError("two slots must not share CUDA_VISIBLE_DEVICES")

    handles: dict[int, subprocess.Popen[bytes]] = {}
    running: list[dict[str, Any]] = []
    started = time.monotonic()
    # The engine scan is scoped to what this job added: the pids that already match the engine pattern
    # before anything starts (another job's vLLM on a shared machine) are recorded and excluded later.
    baseline = {int(entry.split(":", 1)[0]) for entry in _engine_process_scan()}
    for slot in slots:
        directory = Path(slot["log_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        Path(slot["tmpdir"]).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=slot["cuda_visible_devices"],
            VLLM_PORT=str(slot["vllm_port"]),
            TMPDIR=slot["tmpdir"],
        )
        with (directory / "serve.log").open("ab") as log:
            try:
                process = subprocess.Popen(
                    slot["argv"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,  # its own process group, so the stop step can kill the whole engine
                )
            except OSError as error:
                raise HarnessError(f"cannot start the slot {slot['slot']} engine ({slot['model']}): {error}") from error
        handles[process.pid] = process
        running.append({"slot": slot["slot"], "model": slot["model"], "pid": process.pid, "port": slot["port"]})
    state = {"engines": running, "started": _now(), "baseline_pids": sorted(baseline)}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    deadline = time.monotonic() + float(spec.get("startup_timeout_s", 1800))
    ready = {engine["slot"]: False for engine in running}
    while time.monotonic() < deadline and not all(ready.values()):
        for engine in running:
            if ready[engine["slot"]]:
                continue
            process = handles[engine["pid"]]
            if process.poll() is not None:
                raise HarnessError(
                    f"the slot {engine['slot']} engine ({engine['model']}) exited with code {process.returncode}"
                )
            if _models_ok(engine["port"]):
                ready[engine["slot"]] = True
                engine["ready_after_s"] = round(time.monotonic() - started, 1)
        if not all(ready.values()):
            time.sleep(2.0)
    if not all(ready.values()):
        raise HarnessError(f"the engines were not ready within {spec.get('startup_timeout_s', 1800)}s")
    expected_by_slot = {slot["slot"]: slot["served_model_name"] for slot in slots}
    for engine in running:
        engine["models"] = _models(engine["port"])
        engine["engine_version"] = _engine_version(engine["port"])
        engine["served_model_name_expected"] = expected_by_slot[engine["slot"]]
    fragment = {
        "slots": running,
        "concurrent": len(running) == len(slots) and all(ready.values()),
        "isolation": {
            slot["slot"]: {
                "cuda_visible_devices": slot["cuda_visible_devices"],
                "port": slot["port"],
                "vllm_port": slot["vllm_port"],
                "tmpdir": slot["tmpdir"],
                "log_dir": slot["log_dir"],
            }
            for slot in slots
        },
        "passed": all(engine["models"] == [engine["served_model_name_expected"]] for engine in running),
    }
    _write(report, fragment)
    return fragment


def _engines_stop(state_path: Path, report: Path) -> dict[str, Any]:
    """Stop the engines' process groups, then assert that no engine process remains (step g).

    The assert is scoped to what this job added: every live process that looks like a vLLM engine and
    belongs to one of the engines' sessions (a survivor of what this job started), and the engines' own
    pids. A machine-wide scan of every vLLM-shaped cmdline is recorded (``scan_all``) for the operator,
    without gating: a shared machine runs other jobs' engines, and they are not this wave's. The free
    disk on the HF cache is re-measured here - the eviction's freed space shows once the engine that
    mapped the weights is gone.
    """

    state = json.loads(state_path.read_text(encoding="utf-8"))
    pids = [engine["pid"] for engine in state.get("engines", [])]
    for pid in pids:
        _stop_group(pid)
    leaked = _engine_leaks(set(pids))
    scan_all = _engine_process_scan(exclude_pids=set(pids))
    all_stopped = all(not _alive(pid) for pid in pids) and not leaked
    free_disk_after_stop = weights_free_disk()
    fragment = {
        "pids": pids,
        "leaked": leaked,
        "scan_found": leaked,
        "scan_all": scan_all,
        "all_stopped": all_stopped,
        "free_disk_after_stop_bytes": free_disk_after_stop,
        "passed": all_stopped,
    }
    _write(report, fragment)
    return fragment


def weights_free_disk() -> int | None:
    """Free bytes on the HF cache's filesystem (the eviction's authority), or ``None`` offline."""
    try:
        from . import weights

        return weights.disk_free_bytes(weights.hf_cache_root())
    except Exception:  # noqa: BLE001 - the measurement is a record, never a failure
        return None


def _stop_group(pid: int) -> None:
    """SIGTERM the process group, then SIGKILL what remains after the grace period."""
    import errno

    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise
    deadline = time.monotonic() + _ENGINE_TERM_S
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if _alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _engine_leaks(engine_pids: set[int]) -> list[str]:
    """The live processes that still belong to this job's engines: their own pids and every process of
    their sessions (each engine runs in its own session, so a surviving child shares its session id)."""
    found: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in engine_pids or _session_of(pid) in engine_pids:
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            except OSError:
                continue
            found.append(f"{pid}: {cmdline[:120]}")
    return found


def _session_of(pid: int) -> int:
    """The session id of ``pid`` (field 6 of ``/proc/<pid>/stat``), 0 when it cannot be read."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    tail = stat.rpartition(")")[2].split()
    try:
        return int(tail[3])  # the fields after the comm: state, ppid, pgrp, session
    except (IndexError, ValueError):
        return 0


def _engine_process_scan(*, exclude_pids: set[int] | None = None) -> list[str]:
    """The cmdlines of every live process that looks like a vLLM engine, machine-wide (recorded for the
    operator, never the assert: a shared machine's other jobs' engines are not this wave's)."""
    excluded = exclude_pids or set()
    found: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        if "vllm serve" in cmdline or "/vllm" in cmdline:
            found.append(f"{entry.name}: {cmdline[:120]}")
    return found


def _models_ok(port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=10.0).status_code == 200
    except httpx.HTTPError:
        return False


def _models(port: int) -> list[str]:
    import httpx

    reply = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=30.0)
    reply.raise_for_status()
    return [str(model["id"]) for model in reply.json().get("data", [])]


def _engine_version(port: int) -> str | None:
    import httpx

    try:
        reply = httpx.get(f"http://127.0.0.1:{port}/version", timeout=30.0)
        if reply.status_code == 200:
            return str(reply.json().get("version"))
    except httpx.HTTPError:
        pass
    return None


# --- (e) the product's budget, client and the engine's /tokenize ---------------------------------------


def _embed(args: argparse.Namespace, report: Path) -> dict[str, Any]:
    """The 20-text check: the product's ``fit`` under an explicit budget, the product's client, /tokenize."""
    import numpy as np

    from rcp_ndcg.data.preprocess import TextBudget, fit
    from rcp_ndcg.data.tokenizer import load_tokenizer
    from rcp_ndcg.inference import EmbeddingClient
    from rcp_ndcg.inference.adapters import embeddings as _shipped  # noqa: F401  # registers the adapters
    from rcp_ndcg.inference.adapters.base import get_adapter
    from rcp_ndcg.inference.config import EmbeddingEndpoint
    from rcp_ndcg.inference.types import Content, EmbedRequest, EncodeRole

    tokenizer = load_tokenizer(args.tokenizer)
    budget = TextBudget(tokenizer=args.tokenizer, max_tokens=args.budget, on_overflow="cut")
    inputs, kinds = _inputs(args.count, args.over, args.budget, tokenizer)
    fitted = fit(
        inputs,
        "query",
        budget,
        tokenizer,
        ids=[str(index) for index in range(len(inputs))],
        corpus="wave0",
    )
    texts = list(fitted.texts)
    cuts = {record.doc_id: True for record in fitted.cuts}
    endpoint_kwargs: dict[str, Any] = {
        "api": "openai_embeddings",
        "model": args.served_model_name,
        "base_url": f"{args.base_url.rstrip('/')}/v1",
        "tokenizer": args.tokenizer,
        "max_tokens": args.budget,
        "normalize": True,
    }
    embeddings = None
    wired = True
    try:
        # The wired client: the config carries the budget, the client fits the raw inputs itself.
        client = EmbeddingClient(EmbeddingEndpoint(**endpoint_kwargs))
        embeddings = client.encode([Content.from_text(text) for text in inputs], EncodeRole.DOCUMENT)
    except Exception as error:  # noqa: BLE001 - until the client wiring lands (clients-final), pre-fit
        if "max_tokens" not in str(error):
            raise
        wired = False
    if embeddings is None:
        # The unwired interim (the harness stage 2's own shape): the product's adapter and transport
        # send the fitted texts; the fit above already applied the explicit budget.
        from rcp_ndcg.inference.transport import Transport

        endpoint = EmbeddingEndpoint(**endpoint_kwargs)
        adapter = get_adapter(endpoint.api, role="embed")()
        request = EmbedRequest(contents=tuple(Content.from_text(text) for text in texts), role=EncodeRole.DOCUMENT)
        calls = adapter.calls(request, model=args.served_model_name)
        transport = Transport(endpoint)
        replies = list(transport.run(transport.send(calls)))
        transport.aclose()
        embeddings = adapter.interpret(request, replies)

    rows = []
    for index, (text, kind) in enumerate(zip(texts, kinds, strict=True)):
        engine_ids = _engine_tokenize(args.base_url, args.served_model_name, text)
        fit_ids = tokenizer.ids(text, add_special_tokens=True)
        rows.append(
            {
                "id": str(index),
                "kind": kind,
                "fit_tokens": len(fit_ids),
                "engine_tokens": len(engine_ids),
                "ids_equal": engine_ids == fit_ids,
                "cut": bool(cuts.get(str(index), False)),
                "text_head": text[:80],
            }
        )
    vectors = embeddings.as_matrix()
    ids_ok = all(row["ids_equal"] for row in rows)
    finite = bool(len(vectors) and np.isfinite(vectors).all())
    fragment = {
        "budget_tokens": args.budget,
        "n_inputs": len(inputs),
        "n_over_length": args.over,
        "cuts_recorded": len(fitted.cuts),
        "overhead_tokens": fitted.overhead,
        "client_budget_wired": wired,
        "client": {"n_vectors": int(vectors.shape[0]), "dim": int(vectors.shape[1]), "finite": finite},
        "tokenize_check": {"checked": len(rows), "passed": ids_ok, "rows": rows},
        "passed": ids_ok and int(vectors.shape[0]) == len(inputs) and finite,
    }
    _write(report, fragment)
    return fragment


def _inputs(count: int, over: int, budget: int, tokenizer: Any) -> tuple[list[str], list[str]]:
    """The wave's texts: ``count`` inputs, the last ``over`` of them padded past the budget."""
    if over >= count:
        raise HarnessError("over must be smaller than count")
    short = [
        "The quick brown fox jumps over the lazy dog.",
        "检索增强生成会把检索到的文档交给模型。",
        "Emoji smoke: 👍🏽 🚀 (and a ZWJ family: 👨‍👩‍👧‍👦).",
        "def embed(texts):\n    return model.encode(texts, normalize_embeddings=True)\n",
        "https://example.com/very/long/path?with=parameters&and=more#anchor",
        "     ",
        "Mixed ALPHA β γ δ Ελληνικά text.",
        "a",
        "Hello, world!",
        "Ein kleiner deutscher Satz mit Umlauten: äöü ß.",
        "Le cœur déçu mais l'âme plutôt naïve, Louÿs rêva de crapaüter en canoë.",
        "«Guillermo» y «niño» - español con puntuación.",
        "0123456789" * 3,
        "tab\tseparated\tvalues\nand a newline",
        "SELECT id, name FROM documents WHERE body LIKE '%query%' LIMIT 10;",
        "Two roads diverged in a yellow wood, and sorry I could not travel both.",
        '{"json": true, "nested": {"values": [1, 2, 3]}}',
        "The empty document problem: what should an engine do with nothing?",
        "Чистый дом — красивый дом; verba volant, scripta manent.",
        "@@placeholder-for-short-inputs@@",
    ]
    if count <= len(short):
        inputs = short[: count - over]
    else:
        inputs = short + [f"extra short text {index}" for index in range(count - over - len(short))]
    kinds = ["short"] * len(inputs)
    unit = "The model reads this sentence and embeds its meaning into one dense vector."
    one = max(tokenizer.count(unit), 1)
    for index in range(over):
        repetitions = max(budget // one * (2 + index), 2)
        inputs.append(" ".join([unit] * repetitions))
        kinds.append("over_length")
    return inputs, kinds


def _engine_tokenize(base_url: str, model: str, text: str) -> list[int]:
    """The engine's own ids of the fitted text (R29: the engine is the tokenization truth)."""
    import httpx

    from ..equivalence.stages import tokenize_url

    response = httpx.post(
        tokenize_url(base_url),
        json={"model": model, "prompt": text, "add_special_tokens": True},
        timeout=120.0,
    )
    if response.status_code != 200:
        raise HarnessError(f"the engine's /tokenize returned HTTP {response.status_code}: {response.text[:200]}")
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise HarnessError("the engine's /tokenize reply carries no 'tokens' list")
    return [int(value) for value in tokens]


# --- (f) the eviction ----------------------------------------------------------------------------------


def _evict(models: list[str], report: Path) -> dict[str, Any]:
    """Evict each wave model from the HF cache; the free disk before/after is the honest number."""
    from . import weights

    evictions = []
    for model in models:
        before = weights.disk_free_bytes(weights.hf_cache_root())
        held = weights.snapshot_bytes(model)
        eviction = weights.evict(model)
        after = weights.disk_free_bytes(weights.hf_cache_root())
        evictions.append(
            {
                "model": model,
                "cache_bytes": held,
                "removed": eviction.removed,
                "freed_bytes": eviction.freed_bytes,
                "free_disk_before_bytes": before,
                "free_disk_after_bytes": after,
                "error": eviction.error,
            }
        )
    fragment = {
        "models": evictions,
        "passed": all(entry["error"] is None for entry in evictions),
        "note": "freed_bytes is what the removed directory held; while an engine still maps the weights "
        "the filesystem may reclaim the space only after it closes (the stop step re-measures)",
    }
    _write(report, fragment)
    return fragment


# --- shared --------------------------------------------------------------------------------------------


def _write(report: Path, fragment: dict[str, Any]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
