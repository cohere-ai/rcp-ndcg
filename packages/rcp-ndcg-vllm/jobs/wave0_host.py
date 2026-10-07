#!/usr/bin/env python3
"""Wave 0 step (a): the node's facts, on the image's python3 before any environment exists.

Stdlib and ``nvidia-smi`` only (the client environment has not been built yet when this runs). Prints
one JSON object - the report's ``host`` fragment - and writes it to ``--report``. A fact the container
cannot measure (the image digest) is recorded as ``null`` with the reason; nothing is defaulted away.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["collect", "hub_metadata", "main"]

_SMI_TIMEOUT_S = 60


def collect(workdir: Path, transfer: str | None = None) -> dict[str, Any]:
    """The host facts: image (and digest when the operator resolved one), driver, GPUs, disk, /dev/shm,
    the image python's version."""
    return {
        "image": os.environ.get("RCP_IMAGE") or "vllm/vllm-openai:v0.31.0",
        "image_digest": os.environ.get("RCP_IMAGE_DIGEST"),
        "image_digest_note": (
            None
            if os.environ.get("RCP_IMAGE_DIGEST")
            else "RCP_IMAGE_DIGEST is unset: the operator resolves it before submitting (a container "
            "cannot see its own image digest); record it with the report"
        ),
        "driver": _nvidia_smi(["--query-gpu=driver_version"], first=True),
        "gpus": _gpu_rows(),
        "free_disk_bytes": _free_disk(workdir),
        "free_disk_path": str(workdir),
        "shm_bytes": _shm_bytes(),
        "shm_total_bytes": _shm_bytes(total=True),
        "python": {"engine": _python_version(sys.executable)},
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "collected": _now(),
        "transfer": transfer or "unknown",
        "passed": True,
    }


def _nvidia_smi(query: list[str], *, first: bool = False) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader"], capture_output=True, text=True, timeout=_SMI_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0] if first else ",".join(lines)


def hub_metadata(model: str, revision: str | None) -> dict[str, Any]:
    """One metadata call to the Hub with the token secret from the environment (step c).

    ``GET https://huggingface.co/api/models/<model>[/revision/<rev>]`` with ``Authorization: Bearer
    $HF_TOKEN``: the reply's ``sha`` is the commit the weights come from, and wave 0 checks it against
    the pinned revision. The token is read from the environment and never printed; the reply is a
    public metadata document.
    """
    import urllib.error
    import urllib.request

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("hub: HF_TOKEN is not set (the job's secret)")
    url = f"https://huggingface.co/api/models/{model}"
    if revision:
        url += f"/revision/{revision}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            status = reply.status
            body = json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return {
            "model": model,
            "revision": revision,
            "status": error.code,
            "error": f"the Hub answered HTTP {error.code} for {model}",
            "passed": False,
        }
    except (OSError, TimeoutError) as error:
        return {
            "model": model,
            "revision": revision,
            "status": None,
            "error": f"{type(error).__name__}: {error}",
            "passed": False,
        }
    sha = body.get("sha") if isinstance(body, dict) else None
    return {
        "model": model,
        "revision": revision,
        "status": status,
        "hub_sha": sha,
        "pinned_match": sha == revision if (sha and revision) else None,
        "passed": status == 200 and (sha == revision if revision else sha is not None),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _gpu_rows() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"index": int(parts[0]), "name": parts[1], "memory_total_mib": int(parts[2])})
    return rows


def _free_disk(path: Path) -> int | None:
    """Free bytes on ``path``'s filesystem, measured at its nearest existing parent - a fresh pod's
    work directory and cache do not exist yet (the rule of ``rcp_ndcg_vllm.jobs.weights``
    ``disk_free_bytes``, kept stdlib-only here: this script runs before any environment exists). A
    measurement never creates anything."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _shm_bytes(*, total: bool = False) -> int | None:
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stats.f_bsize * (stats.f_blocks if total else stats.f_bavail)


def _python_version(python: str) -> str:
    try:
        out = subprocess.run(
            [python, "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``host`` for the node facts (step a), ``hub`` for the Hub metadata call (step c); the
    fragment lands in ``--report`` and on stdout."""
    parser = argparse.ArgumentParser(
        prog="wave0_host.py", description="Wave 0's host facts and Hub probe (stdlib only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_host = sub.add_parser("host", help="the node facts (image, driver, GPUs, disk, /dev/shm, pythons)")
    p_host.add_argument("--report", required=True, help="fragment path (JSON)")
    p_host.add_argument("--workdir", required=True, help="the filesystem whose free disk is measured")
    p_host.add_argument("--transfer", default=None, help="which transfer path ran: gcloud, gsutil or python")
    p_hub = sub.add_parser("hub", help="one metadata call to the Hub with the token secret")
    p_hub.add_argument("--model", required=True)
    p_hub.add_argument("--revision", default=None, help="the pinned commit to compare the Hub's sha against")
    p_hub.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    if args.command == "host":
        fragment = collect(Path(args.workdir), args.transfer)
    else:
        fragment = hub_metadata(args.model, args.revision)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
    json.dump(fragment, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if fragment.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
