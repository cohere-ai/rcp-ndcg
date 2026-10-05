"""Stage 3: nDCG@10 per subset from rankings, scored by calling ``rcp-ndcg eval score`` as a subprocess.

The harness never imports ``rcp-ndcg``'s internals (the two packages meet over HTTP and at the command line, not
in the import graph).  The input is a rankings directory with, per subset:

- ``<subset>.served.jsonl`` — the served system's rankings, in rcp-ndcg's rankings format;
- ``<subset>.reference.jsonl`` — the reference (or stored) system's rankings, same format;
- ``<subset>.dataset.jsonl`` — the dataset rows (query, docs, qrels) the rankings were produced over.

The gate is the mean |delta nDCG@10| over subsets, default 2e-3 (the MTEB rerank tolerance on a 0-to-1 scale),
overridable with ``gates.metrics_max_abs``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..errors import HarnessError
from .gates import ResolvedGates

__all__ = ["stage3_metrics"]

_K = 10
_BODY_SNIPPET = 500


def stage3_metrics(rankings_dir: str | Path, gates: ResolvedGates, *, rcp_ndcg: str | None = None) -> dict[str, Any]:
    """nDCG@10 per subset from the rankings under ``rankings_dir``, served against reference.

    Output: a report dict with one entry per subset (both systems' nDCG@10 and the absolute delta), the mean
    |delta| and the gate row.  Raises :class:`HarnessError` with an install hint when the ``rcp-ndcg`` command is
    absent (the ``metrics`` extra provides it).
    """
    command = rcp_ndcg or shutil.which("rcp-ndcg") or (str(sys.executable) + " -m rcp_ndcg.cli")
    subsets = sorted(p.name[: -len(".served.jsonl")] for p in Path(rankings_dir).glob("*.served.jsonl"))
    if not subsets:
        raise HarnessError(f"no <subset>.served.jsonl rankings under {rankings_dir}")
    entries: list[dict[str, Any]] = []
    for subset in subsets:
        served = _score_one(command, rankings_dir, subset, "served")
        reference = _score_one(command, rankings_dir, subset, "reference")
        delta = abs(served - reference)
        entries.append({"subset": subset, "served_ndcg10": served, "reference_ndcg10": reference, "abs_delta": delta})
    mean_delta = float(sum(entry["abs_delta"] for entry in entries) / len(entries))
    gate_rows = [
        {
            "gate": "metrics_mean_abs_delta",
            "passed": bool(mean_delta <= gates.metrics_max_abs),
            "value": mean_delta,
            "bound": gates.metrics_max_abs,
            "referent": "mean |delta qrel-nDCG@10| over subsets",
        }
    ]
    return {
        "n_subsets": len(entries),
        "subsets": entries,
        "mean_abs_delta": mean_delta,
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows)),
    }


def _score_one(command: str, rankings_dir: str | Path, subset: str, system: str) -> float:
    """``rcp-ndcg eval score`` for one rankings file; returns that system's qrel-nDCG@10 mean."""
    path = Path(rankings_dir) / f"{subset}.{system}.jsonl"
    dataset = Path(rankings_dir) / f"{subset}.dataset.jsonl"
    if not path.is_file() or not dataset.is_file():
        raise HarnessError(f"stage 3 needs {path} and {dataset}: rankings and dataset rows in rcp-ndcg's formats")
    argv = [*command.split(), "eval", "score", "--rankings", str(path), "--dataset", f"jsonl:{dataset}",
            "--metrics", "qrel_ndcg", "--k", "10", "--bootstrap", "0", "--json"]  # fmt: skip
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise HarnessError(
            f"the rcp-ndcg command is not installed ({command}); install the metrics extra: "
            "pip install rcp-ndcg-vllm[metrics]"
        ) from error
    except subprocess.CalledProcessError as error:
        raise HarnessError(
            f"rcp-ndcg eval score failed for {subset}/{system}: {error.stderr[:_BODY_SNIPPET]}"
        ) from error
    document = json.loads(completed.stdout)
    value = _summary_value(document["data"]["summary"], subset, system)
    return value


def _summary_value(summary: list[dict[str, Any]], subset: str, system: str) -> float:
    """The qrel-nDCG@10 mean of one system from the eval score document; missing means an error, not zero."""
    rows = [row for row in summary if row["metric"] == "qrel_ndcg" and int(row["k"]) == 10 and row["system"] == system]
    if not rows:
        raise HarnessError(f"rcp-ndcg eval score returned no qrel_ndcg@10 row for {subset}/{system}")
    return float(rows[0]["value"])
