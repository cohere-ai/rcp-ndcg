"""The reference subprocess: the recipe's ``reference.py`` runs in its own environment, never in the harness's.

The reference implements the model's in-process scoring (torch, transformers, the checkpoint's trusted code),
so it never imports into the harness's process — the harness holds no torch and no transformers (a test
asserts that). :func:`run_reference` invokes it as a subprocess, with a declared interpreter:

    <reference-python> <recipe-dir>/<entry> --mode <mode> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

The modes, and the JSON each writes to ``--out``:

- ``render`` — stage 1's reference side.  Embedding roles: for each pairs-file row and declared shape, the
  exact prompt the engine reads (the reference's own anchor-preserving render: fixed segments reserved,
  content cut, template re-attached).  ``{"rows": [{"index", "shape", "text": str}]}``, one input per shape
  (the row's query for the query shape, its first document for the document shape).  Rerank: the spans the
  client ships — ``{"rows": [{"index", "shape": "pair", "query": str, "documents": [str, ...]}]}`` (the
  query as the recipe's instruction mode folds it; the frame is the engine's own template).
- ``score`` — stage 2 for a rerank recipe: one score per document per row, on the recipe's
  ``reference.score_scale``.  ``{"rows": [{"index", "scores": [...]}]}``.
- ``embed`` — stage 2 for the embedding roles: one vector per text for the row's query (role ``query``) and
  each document (role ``document``); a late-interaction reference returns one vector per token.
  ``{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[[...]]]}]}``.
- ``media`` -- the media stage's reference side (:mod:`rcp_ndcg_test.equivalence.media`), for the pairs rows
  that carry ``media``: per row and side that carries media (``query``, ``document <i>``), what the card's
  model consumes -- ``{"rows": [{"index", "side", "placement": ["image", "text", ...], "media": [{"kind",
  "width", "height", "tokens"} | {"kind": "video", "frames", "tokens"}]}]}``: the parts in the card's order,
  each image's size after the card's own resize and its prompt tokens (its vision markers included).  A side
  the card cannot consume is reported ``{"index", "side", "refused": "<why>"}`` (the media stage fails it,
  and the pairs generator prunes its row).

The instruction travels in the pairs file (per row); the reference folds it in the product's
``Task: <instruction>\nQuery: <text>`` format for ``instruction: fold``.  On the node the
engine comes up on the slot's GPUs first; the reference subprocess runs against the pairs file while
the engine is up and releases its memory when it exits (the wave runner sequences it after that recipe's
smoke pass).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.stepwatch import current_watch

__all__ = ["REFERENCE_MODES", "run_reference"]

REFERENCE_MODES = ("render", "score", "embed", "media")
"""The reference CLI's modes: ``render`` for stage 1's id comparison, ``score`` and ``embed`` for stage 2, ``media``
for the media stage."""

_TIMEOUT_S = 3600.0
_OUTPUT_SNIPPET = 500


def run_reference(
    reference_python: str,
    entry: str,
    *,
    mode: str,
    pairs_path: str | Path,
    out_path: str | Path,
    tokenizer_spec: str,
    device: str = "cpu",
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Run the recipe's reference module as a subprocess and return its parsed JSON output.

    Inputs: the declared reference interpreter (``--reference-python``, required; no default — the harness
    process never imports the reference), the recipe's ``reference.entry`` file, the mode
    (:data:`REFERENCE_MODES`), the pairs file, the output path, the tokenizer spec the reference tokenises
    with, the device its ``--device`` flag gets, and the physical GPU index it is pinned to via
    ``CUDA_VISIBLE_DEVICES`` (GPU-E1: the reference gets a GPU of its own beside the engine's, never the
    engine's GPU; ``None`` pins nothing).  The subprocess's timeout is the step watch's remaining budget
    when the thread runs inside one (the wave runner's step budget bounds the reference too), else one
    hour.  Raises :class:`~rcp_ndcg_test.errors.HarnessError` with the subprocess's own stderr when the
    reference fails, outlives its budget, or writes unparsable JSON.
    """
    if mode not in REFERENCE_MODES:
        raise HarnessError(f"reference mode {mode!r} must be one of {list(REFERENCE_MODES)}")
    argv = [
        reference_python,
        entry,
        "--mode",
        mode,
        "--pairs",
        str(pairs_path),
        "--out",
        str(out_path),
        "--tokenizer",
        tokenizer_spec,
        "--device",
        device,
    ]
    env = dict(os.environ)
    if cuda_visible_devices is not None:
        # The reference's own GPU: never the engine's, which holds most of its memory (GPU-E1).
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    watch = current_watch()
    timeout_s = watch.remaining_s() if watch is not None else _TIMEOUT_S
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as error:
        step = f"step {watch.step}" if watch is not None else "the reference subprocess"
        raise HarnessError(f"{step} exceeded its budget; in flight: the reference subprocess ({mode})") from error
    if completed.returncode != 0:
        raise HarnessError(
            f"the reference subprocess ({mode}) failed with exit code {completed.returncode}: "
            f"{completed.stderr[-_OUTPUT_SNIPPET:]}"
        )
    try:
        document = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise HarnessError(
            f"the reference subprocess wrote no parsable JSON to {out_path}: {error} "
            f"(stderr: {completed.stderr[-_OUTPUT_SNIPPET:]})"
        ) from error
    if not isinstance(document, dict):
        raise HarnessError(f"the reference's output at {out_path} must be a JSON object")
    return document
