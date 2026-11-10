#!/usr/bin/env python3
"""D4's probe: does the engine's own attention path serve ``embeddinggemma-2`` like the plugin's SDPA backend?

Runs on the node under the client mechanism (``sdpa_probe.sh``), one GPU, two engine starts of the same
recipe: the recipe as loaded (``patch``: ``serve.patches`` carries ``torch-sdpa-attention`` and
``--attention-backend TORCH_SDPA`` is pinned) and the recipe with that patch name and that flag pair removed
(``native``: the engine's own selection -- at ``v0.33.0`` the model's own config hook defaults an unrequested
backend to ``TRITON_ATTN``, ``vllm/model_executor/models/config.py:371-387``).  Same checkpoint and revision,
same ``serve`` fields, same prompts through the product's own role client
(:func:`rcp_ndcg_test.equivalence.wire.role_client`, full width), same request sequence.  Every text is one
client call, so the per-text attribution is exact and both arms send the same bytes.

The gate is the plan's, carried as data in the report (never re-typed at verdict time): pooled-output
``max_abs_diff <= 1e-5`` and ``cosine >= 0.99999`` (``/tmp/rcp-ndcg-vllm-033/PLAN.md`` v7 §3 W7, decision D4).
The run's own noise floor is **measured**, not assumed: each arm is requested a second time over the same
texts in the same call sequence (the null control), and the verdict reads that floor.  The decision rule, its
dispositions and the deletion's whole surface are in ``/tmp/rcp-ndcg-vllm-033/exec/DECISION-D4.md``, written
before this probe ran; the same table rides inside the report as ``decision_rule``.

    "$STATE/client" python sdpa_probe.py --out <dir> --recipe embeddinggemma-2 \
        --pairs <stage>/pairs/embeddinggemma-2.jsonl [--rows 3,4] [--port-base 8100] [--gpus 0]

    # the wrapper's plan, running nothing:
    SDPA_DRY=1 sdpa_probe.sh <RC_STAGE_URI> <OUT_URI>

What the report carries (``sdpa-report.json``, ``sdpa-report.md``, ``sdpa-vectors.json``): the gate with its
source, the two arms' ``argv``/``patches_env``/``state``/``serve_log_tail`` and the **engaged** backend read
out of the engine's own serve log (never inferred from the request), the per-text and aggregate comparison,
the null control per arm and its floor, the verdict, and the rule that produced it.  An arm that cannot serve
leaves the engine's own error in the report; a request that fails inside an arm leaves the transport's typed
error there too (the report is written either way, so the wrapper always uploads one); and a selection that
carries no text is ``nothing_checked``, never ``equivalent``.  Nothing is inferred from silence.
"""

from __future__ import annotations

__all__ = [
    "ARMS",
    "DECISION_RULE",
    "DEFAULT_GATE",
    "GATE_SOURCE",
    "PATCH_NAME",
    "arm_recipes",
    "arms",
    "build_parser",
    "build_report",
    "compare",
    "main",
    "verdict",
    "write_report",
]

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_vllm.patches import PATCHES_ENV, patches_env_value
from rcp_ndcg_vllm.recipe import Recipe, load_recipe, serve_argv

from rcp_ndcg.errors import ProviderError
from rcp_ndcg_test.errors import HarnessError

SCHEMA = "rcp-sdpa-probe/1"
"""The report's format version (the artifact the operator adjudicates on)."""

PATCH_NAME = "torch-sdpa-attention"
"""The patch this probe decides on: the recipe declares it, and the ``native`` arm drops it."""

ATTENTION_BACKEND_FLAG = "--attention-backend"
"""The flag the patch is useless without: the enum member's own value is ``""`` at ``v0.33.0`` ("this tag is
only used for ViT"), so a ``TORCH_SDPA`` pin without the registration raises at engine start.  The flag and
the patch name move together, in both directions."""

PATCHED_BACKEND = "TORCH_SDPA"
"""The only ``--attention-backend`` value this probe removes; any other value is refused, never dropped."""

ARMS = ("patch", "native")
"""The two engine starts: today's recipe, and the recipe with the patch and its flag removed."""

DEFAULT_GATE: dict[str, Any] = {
    "max_abs_diff": 1e-5,
    "cosine": 0.99999,
    "source": "PLAN v7 §3 W7 / decision D4: the bf16 noise floor, recorded here and in every report",
}
"""The plan's tolerance, carried as data: the verdict reads these numbers from the report, never from a
constant typed at verdict time.  The run's own floor is the null control's, measured."""

GATE_SOURCE = DEFAULT_GATE["source"]
"""The gate's provenance, exported so the tests and the report name one string."""

DECISION_RULE: dict[str, str] = {
    "equivalent": "delete",
    "near": "operator",
    "differs": "keep",
    "not_servable": "keep",
    "unmeasured": "operator",
}
"""The rule of DECISION-D4.md §3, one row per state: the report carries it so a reader of the uploaded
artifact needs no external document.  ``nothing_checked`` (a selection with no text) is deliberately not a
row: the pre-registered rule decides a comparison, and with none there is nothing to decide, so that state
takes the ``unmeasured`` row's disposition."""

_STOP_GRACE_S = 15.0
_READY_POLL_S = 2.0
_BACKEND_LINE = re.compile(
    r"EmbeddingGemma2: (?:using attention backend (?P<using>\S+)"
    r"|attention backend not specified; defaulting to (?P<default>\S+))"
)


# ---------------------------------------------------------------------------
# the two arms: one patch name and one flag pair apart, nothing else
# ---------------------------------------------------------------------------


def _without_attention_backend(extra_args: list[str]) -> list[str]:
    """``extra_args`` without the ``--attention-backend TORCH_SDPA`` pair (both spellings).

    Raises :class:`HarnessError` when the recipe pins another backend: this probe removes one declared
    value, it never silently drops a different pin.
    """
    kept: list[str] = []
    index = 0
    while index < len(extra_args):
        word = extra_args[index]
        if word == ATTENTION_BACKEND_FLAG or word.startswith(f"{ATTENTION_BACKEND_FLAG}="):
            value = extra_args[index + 1] if word == ATTENTION_BACKEND_FLAG else word.split("=", 1)[1]
            if value != PATCHED_BACKEND:
                raise HarnessError(
                    f"serve.extra_args pins {ATTENTION_BACKEND_FLAG} {value}: this probe removes only "
                    f"{PATCHED_BACKEND} (the value the {PATCH_NAME} patch registers)"
                )
            index += 2 if word == ATTENTION_BACKEND_FLAG else 1
            continue
        kept.append(word)
        index += 1
    return kept


def arm_recipes(recipe: Recipe) -> dict[str, Recipe]:
    """The two arms' recipes: ``patch`` is the recipe as loaded, ``native`` drops the patch and its flag.

    Inputs: the loaded recipe (the family's variant).  Outputs: ``{"patch": recipe, "native": copy}`` where
    the copy's ``serve.patches`` lacks :data:`PATCH_NAME` and its ``serve.extra_args`` lacks the
    ``--attention-backend TORCH_SDPA`` pair; every other field is the recipe's own.  Raises
    :class:`HarnessError` when the recipe no longer declares the patch (after a deletion there is nothing to
    compare) or pins another backend.  Units: none.
    """
    serve = recipe.serve
    if PATCH_NAME not in serve.patches:
        raise HarnessError(
            f"recipe {recipe.id} does not declare serve.patches: {PATCH_NAME}: this probe's question is "
            "answered (the patch is deleted or renamed) and there is nothing to compare"
        )
    remaining = serve.patches.__class__(name for name in serve.patches if name != PATCH_NAME)
    native_serve = serve.model_copy(
        update={"patches": remaining, "extra_args": _without_attention_backend(list(serve.extra_args))}
    )
    return {"patch": recipe, "native": recipe.model_copy(update={"serve": native_serve})}


def arms(recipe: Recipe, *, port_base: int = 8100, gpus: str = "0") -> dict[str, dict[str, Any]]:
    """The two engine starts' full plan: the rendered argv, the patch opt-in, the port and the env additions.

    Inputs: the loaded recipe, the port the engines serve on (both arms run sequentially on it, so the
    second start reuses it), and the GPU list for ``CUDA_VISIBLE_DEVICES``.  Outputs: ``{arm: {argv, port,
    patches_env, env}}`` -- the argv from :func:`rcp_ndcg_vllm.recipe.serve_argv` (the one renderer the serve
    console and the wave runner use too), the opt-in from
    :func:`rcp_ndcg_vllm.patches.patches_env_value`, and the environment the arm's process needs
    (``CUDA_VISIBLE_DEVICES``, :data:`PATCHES_ENV`, ``VLLM_PORT``).  Units: none.
    """
    plan: dict[str, dict[str, Any]] = {}
    for arm, arm_recipe in arm_recipes(recipe).items():
        opt_in = patches_env_value(arm_recipe.serve.patches)
        plan[arm] = {
            "arm": arm,
            "port": port_base,
            "patches_env": opt_in,
            "argv": serve_argv(arm_recipe, port=port_base, served_model_name=recipe.id),
            "env": {
                "CUDA_VISIBLE_DEVICES": gpus,
                PATCHES_ENV: opt_in,
                "VLLM_PORT": str(port_base + 1000),
            },
        }
    return plan


# ---------------------------------------------------------------------------
# the comparison: per text, then the worst text
# ---------------------------------------------------------------------------


def _sha(values: list[float]) -> str:
    """The vector's hash (float64 bytes), so equal vectors hash equal in the report."""
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()[:32]


def _diff(left: list[float], right: list[float]) -> dict[str, Any]:
    """One text's comparison: max/mean abs diff, cosine, both sides' hashes.

    Raises :class:`HarnessError` when the shapes differ: two widths are not a comparison, and reporting one
    as a large diff would read as a backend difference.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise HarnessError(f"vectors are not comparable: shapes {a.shape} and {b.shape}")
    diff = np.abs(a - b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cosine": float(np.dot(a, b) / (na * nb)) if na and nb else None,
        "left_sha256": _sha(left),
        "right_sha256": _sha(right),
    }


def compare(left: dict[str, list[float]], right: dict[str, list[float]]) -> dict[str, Any]:
    """Two keyed vector sets: every text's numbers, and the aggregate over the worst text.

    Inputs: ``{text key: vector}`` for each side, the same keys.  Outputs: ``{n_texts, max_abs_diff,
    mean_abs_diff, cosine, worst_text, per_text}`` where the aggregate takes the largest ``max_abs_diff`` and
    the smallest ``cosine`` (the worst text decides, as a gate does), and ``mean_abs_diff`` is the mean over
    texts of each text's mean.  Raises :class:`HarnessError` when the key sets differ.  Units: none.
    """
    keys = sorted(set(left) | set(right))
    missing = [key for key in keys if key not in left or key not in right]
    if missing:
        raise HarnessError(f"the two sides do not carry the same texts: {len(missing)} missing (first: {missing[:5]})")
    per_text = {key: _diff(left[key], right[key]) for key in keys}
    worst = max(per_text, key=lambda key: per_text[key]["max_abs_diff"]) if per_text else None
    cosines = [row["cosine"] for row in per_text.values() if row["cosine"] is not None]
    return {
        "n_texts": len(keys),
        "max_abs_diff": per_text[worst]["max_abs_diff"] if worst else 0.0,
        "mean_abs_diff": float(np.mean([row["mean_abs_diff"] for row in per_text.values()])) if per_text else 0.0,
        "cosine": min(cosines) if cosines else None,
        "worst_text": worst,
        "per_text": per_text,
    }


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """The D4 rule applied to a report's recorded numbers: ``{state, disposition, reason}``.

    A pure function of the report (the gate and the null control are read from it), so the same rule the
    operator reads in DECISION-D4.md §3 is the one the artifact carries:

    - ``equivalent``: inside the gate and the run's own noise floor is inside it too -- delete the patch;
    - ``near``: inside the gate, but the floor is not established (the null control is above the gate) --
      the operator decides;
    - ``differs``: outside the gate -- the patch stays;
    - ``not_servable``: the native arm served no vectors -- the patch stays, the engine's error is the reason;
    - ``unmeasured``: the patch arm failed, or nothing comparable was measured -- the operator decides;
    - ``nothing_checked``: the selection carries no text, so no vector was compared -- the operator decides,
      on the ``unmeasured`` row's disposition (the rule's rows decide a comparison, and there is none).
    """
    gate = report["gate"]
    comparison = report.get("comparison") or {}
    arm_states = {arm: (report.get("arms", {}).get(arm) or {}).get("state") for arm in ARMS}
    if arm_states["patch"] not in (None, "served"):
        return _verdict("unmeasured", f"the patch arm did not serve ({arm_states['patch']}); see its serve_log_tail")
    if arm_states["native"] not in (None, "served"):
        return _verdict(
            "not_servable",
            f"the native arm did not serve ({arm_states['native']}): the engine's own path cannot serve the "
            "checkpoint; see arms.native.serve_log_tail",
        )
    pair = comparison.get("patch_vs_native")
    if pair and not pair.get("n_texts"):
        # An empty comparison is not evidence: zero texts read max_abs_diff 0.0 and cosine None, which the
        # gate would call inside it -- a `--rows` selection that resolves to nothing would delete the patch.
        return _verdict(
            "nothing_checked",
            "the selection carries no text: no vector was compared, so nothing is evidence for a deletion",
            disposition=DECISION_RULE["unmeasured"],
        )
    if not pair or pair.get("max_abs_diff") is None:
        return _verdict("unmeasured", "no comparable pair of vectors was measured (see the report's arms)")
    within = pair["max_abs_diff"] <= gate["max_abs_diff"] and (
        pair["cosine"] is None or pair["cosine"] >= gate["cosine"]
    )
    if not within:
        return _verdict(
            "differs",
            f"max abs diff {pair['max_abs_diff']:.3g} / cosine {pair['cosine']} is outside the gate "
            f"{gate['max_abs_diff']:.3g} / {gate['cosine']}",
        )
    floor = (report.get("null_control") or {}).get("floor") or {}
    established = floor.get("max_abs_diff") is not None and floor["max_abs_diff"] <= gate["max_abs_diff"]
    if not established:
        return _verdict(
            "near",
            f"inside the gate (max abs diff {pair['max_abs_diff']:.3g}) but the run's own floor "
            f"{floor.get('max_abs_diff')} is not below it: the gate cannot certify equivalence",
        )
    return _verdict(
        "equivalent",
        f"max abs diff {pair['max_abs_diff']:.3g} and cosine {pair['cosine']} are inside the gate, and the "
        f"run's own floor {floor['max_abs_diff']:.3g} is below it",
    )


def _verdict(state: str, reason: str, *, disposition: str | None = None) -> dict[str, Any]:
    """One verdict row: the state, its disposition from :data:`DECISION_RULE`, and the reason.

    ``disposition`` overrides the table for the one state the rule has no row for: ``nothing_checked`` takes
    the ``unmeasured`` row's operator decision rather than adding a sixth row to a pre-registered rule.
    """
    return {"state": state, "disposition": disposition or DECISION_RULE[state], "reason": reason}


def build_report(
    recipe: Recipe,
    *,
    engine: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    served: dict[str, dict[str, list[float]]],
    null: dict[str, dict[str, list[float]]],
    texts: list[str],
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The report: the gate as data, the arms, the comparison, the measured floor and the verdict.

    Inputs: the recipe, the engine facts (image, version), the per-arm rows (``state``, ``argv``,
    ``patches_env``, ``backend``, ``serve_log_tail``), the served vectors per arm (``{text key: vector}``),
    the null control's vectors per arm (the same texts, a second pass), and the probed text keys.  ``gate``
    overrides :data:`DEFAULT_GATE` only for a test's synthetic report; the probe always passes the plan's.
    Outputs: the report document; ``verdict`` is :func:`verdict` applied to it, and ``decision_rule`` carries
    the rule.  Units: the diffs are in the vectors' own units (L2-normalised components), the cosine is
    dimensionless.
    """
    gate_block = dict(gate or DEFAULT_GATE)
    report: dict[str, Any] = {
        "probe": "sdpa-probe",
        "schema": SCHEMA,
        "recipe": {
            "id": recipe.id,
            "model": recipe.model,
            "revision": recipe.revision,
            "dtype": recipe.serve.dtype,
        },
        "engine": dict(engine),
        "gate": gate_block,
        "texts": {"n": len(texts), "keys": list(texts)},
        "arms": {arm: {**row, "vectors": len(served.get(arm) or {})} for arm, row in arms.items()},
        "comparison": {},
        "null_control": {},
        "decision_rule": dict(DECISION_RULE),
    }
    for arm in ARMS:
        if arm in served and arm in null:
            report["null_control"][arm] = compare(served[arm], null[arm])
    floors = [row["max_abs_diff"] for row in report["null_control"].values() if "max_abs_diff" in row]
    cosines = [row["cosine"] for row in report["null_control"].values() if row.get("cosine") is not None]
    floor = {
        "max_abs_diff": max(floors) if floors else None,
        "cosine": min(cosines) if cosines else None,
    }
    report["null_control"]["floor"] = floor
    report["null_control"]["floor_established"] = (
        floor["max_abs_diff"] is not None and floor["max_abs_diff"] <= gate_block["max_abs_diff"]
    )
    if "patch" in served and "native" in served:
        report["comparison"]["patch_vs_native"] = compare(served["patch"], served["native"])
        report["at_floor"] = (
            floor["max_abs_diff"] is not None
            and report["comparison"]["patch_vs_native"]["max_abs_diff"] <= floor["max_abs_diff"]
        )
    report["verdict"] = verdict(report)
    return report


def write_report(out: Path, report: dict[str, Any], *, vectors: dict[str, Any] | None = None) -> None:
    """Write ``sdpa-report.json``, ``sdpa-report.md`` (and ``sdpa-vectors.json`` when given) under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "sdpa-report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    if vectors is not None:
        (out / "sdpa-vectors.json").write_text(json.dumps(vectors, default=str) + "\n", encoding="utf-8")
    gate = report["gate"]
    pair = (report.get("comparison") or {}).get("patch_vs_native") or {}
    floor = (report.get("null_control") or {}).get("floor") or {}
    lines = [
        "# D4 probe: torch-sdpa-attention against the engine's own path",
        "",
        f"- recipe: `{report['recipe']['id']}` @ `{report['recipe']['revision']}` (`{report['recipe']['dtype']}`)",
        f"- engine: `{report['engine'].get('image')}`",
        f"- gate: max abs diff <= `{gate['max_abs_diff']}` and cosine >= `{gate['cosine']}` ({gate['source']})",
        f"- measured floor (null control, each arm against itself): `{floor.get('max_abs_diff')}` "
        f"(established: `{report['null_control'].get('floor_established')}`)",
        f"- **verdict: `{report['verdict']['state']}` -- {report['verdict']['disposition']}**",
        f"- reason: {report['verdict']['reason']}",
        "",
        "| arm | patches | engaged backend | state | max abs diff | cosine |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        row = report["arms"].get(arm) or {}
        lines.append(
            f"| {arm} | `{row.get('patches_env')}` | `{(row.get('backend') or {}).get('engaged')}` | "
            f"{row.get('state')} | {pair.get('max_abs_diff') if arm == 'native' else '-'} | "
            f"{pair.get('cosine') if arm == 'native' else '-'} |"
        )
    lines += [
        "",
        f"- texts probed: {report['texts']['n']}",
        "",
        "## raw",
        "",
        "```json",
        json.dumps(report, indent=1, default=str),
        "```",
        "",
    ]
    (out / "sdpa-report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# the node: two engine starts, one request sequence each
# ---------------------------------------------------------------------------


def _tail(log: Path, lines: int = 40, width: int = 300) -> list[str]:
    """The last ``lines`` of a serve log, each clipped: a dead engine's evidence."""
    if not log.is_file():
        return []
    return [line.strip()[:width] for line in log.read_text(errors="replace").splitlines()[-lines:]]


def engaged_backend(log: Path) -> dict[str, Any]:
    """The backend the engine actually ran, read out of its own serve log.

    The model's config hook logs either ``EmbeddingGemma2: using attention backend <X>`` (an explicit pin) or
    ``EmbeddingGemma2: attention backend not specified; defaulting to <X>``.  A request labelled by a flag is
    not evidence that the backend ran (a refused request can fall back inside vLLM), so the report records
    this line and ``None`` when the log carries none -- never an inferred name.
    """
    if not log.is_file():
        return {"engaged": None, "log_line": None}
    engaged: str | None = None
    line: str | None = None
    for text in log.read_text(errors="replace").splitlines():
        match = _BACKEND_LINE.search(text)
        if match:
            engaged = match.group("using") or match.group("default")
            line = text.strip()[:300]
    return {"engaged": engaged, "log_line": line}


def _wait_ready(base_url: str, popen: subprocess.Popen[bytes], timeout_s: float) -> bool:
    """Poll ``GET /v1/models`` until it answers 200, the process exits, or ``timeout_s`` runs out."""
    import httpx

    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=5.0) as http:
        while time.monotonic() < deadline:
            if popen.poll() is not None:
                return False
            try:
                if http.get(f"{base_url}/v1/models").status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(_READY_POLL_S)
    return False


def _serve(spec: dict[str, Any], *, out: Path, timeout_s: float) -> dict[str, Any]:
    """Start one arm's engine, wait for its route, and record what it ran (or why it did not)."""
    port = int(spec["port"])
    base_url = f"http://127.0.0.1:{port}"
    log = out / f"serve-{spec['arm']}.log"
    env = dict(os.environ)
    env.update(spec["env"])
    # One home per arm: vLLM's ZMQ IPC sockets live under TMPDIR, whose paths must fit AF_UNIX's 107 bytes.
    env["TMPDIR"] = str(out / f"tmp-{spec['arm']}")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "arm": spec["arm"],
        "argv": spec["argv"],
        "patches_env": spec["patches_env"],
        "base_url": base_url,
    }
    with log.open("wb") as handle:
        try:
            popen = subprocess.Popen(
                spec["argv"], stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        except OSError as error:
            return {
                **row,
                "state": "failed",
                "backend": {"engaged": None, "log_line": None},
                "serve_log_tail": [f"cannot start {spec['argv'][0]}: {error}"],
            }
    ready = _wait_ready(base_url, popen, timeout_s)
    return {
        **row,
        "state": "served" if ready else "failed",
        "backend": engaged_backend(log),
        "serve_log_tail": _tail(log),
        "popen": popen,
    }


def _stop(run: dict[str, Any]) -> None:
    """Stop one arm's engine: SIGTERM to its process group, then SIGKILL after the grace."""
    popen = run.get("popen")
    if popen is None or popen.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
    try:
        popen.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        popen.wait(timeout=_STOP_GRACE_S)


def texts_of(rows: list[dict[str, Any]]) -> list[tuple[str, Any, Any]]:
    """The probed texts of the pairs rows: ``(key, role, content)`` per query and per document.

    A media row goes through the harness's own ``side_contents`` (the content stage 2 sends); a text row is
    the product's ``Content.from_text``.  One entry per text, so each client call carries exactly one text.
    """
    from rcp_core.content import Content

    from rcp_ndcg.inference.types import EncodeRole
    from rcp_ndcg_test.equivalence.media import side_contents

    items: list[tuple[str, Any, Any]] = []
    for index, row in enumerate(rows):
        if row.get("media"):
            query, documents = side_contents(row)
        else:
            query = Content.from_text(str(row["query"]))
            documents = [Content.from_text(str(text)) for text in row["documents"]]
        items.append((f"{index}:query", EncodeRole.QUERY, query))
        for position, document in enumerate(documents):
            items.append((f"{index}:document:{position}", EncodeRole.DOCUMENT, document))
    return items


def encode_all(recipe: Recipe, base_url: str, items: list[tuple[str, Any, Any]]) -> dict[str, list[float]]:
    """One vector per text, through the product's role client (the recipe's own prompts and budget)."""
    from rcp_ndcg_test.equivalence.wire import role_client

    client, _capture = role_client(recipe, base_url, full_width=True)
    vectors: dict[str, list[float]] = {}
    for key, role, content in items:
        embeddings = client.encode([content], role)
        matrix = np.asarray(embeddings.vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise HarnessError(
                f"{key}: the client returned {matrix.shape} for one text; this probe compares one vector per text"
            )
        vectors[key] = [float(value) for value in matrix[0]]
    return vectors


def _rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The pairs rows to probe: every row, or the ``--rows`` indices (comma-separated)."""
    from rcp_ndcg_test.equivalence.fitting import load_pairs

    rows = load_pairs(args.pairs)
    if not args.rows:
        return rows
    wanted = [int(value) for value in args.rows.split(",") if value.strip()]
    out = []
    for index in wanted:
        if not 0 <= index < len(rows):
            raise HarnessError(f"--rows {index}: the pairs file carries {len(rows)} rows")
        out.append(rows[index])
    return out


def build_parser() -> argparse.ArgumentParser:
    """The probe's parser: the wrapper builds its argv by hand, so the flags must match here."""
    parser = argparse.ArgumentParser(description="D4's probe: the plugin's SDPA backend against the engine's own")
    parser.add_argument("--out", required=True, help="where the report and the serve logs land")
    parser.add_argument("--recipe", default="embeddinggemma-2", help="the variant id to probe")
    parser.add_argument("--pairs", required=True, help="the recipe's pairs file (every row is probed)")
    parser.add_argument("--rows", default="", help="comma-separated row indices (default: every row)")
    parser.add_argument("--port-base", type=int, default=8100, help="the port both arms serve on")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES for the engine")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0, help="an engine's load budget")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write the report; 0 when a verdict was reached, 1 when nothing was measured.

    A request that fails inside an arm is recorded in that arm's row (the transport's own typed error) and
    the report is still written, so the wrapper always has an artifact to upload; ``unmeasured`` and
    ``nothing_checked`` are the two states that mean no comparison was made, and both exit 1.
    """
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe(args.recipe)
    if recipe.role != "embed":
        raise SystemExit(f"sdpa_probe: {recipe.id} is a {recipe.role} recipe; this probe compares embeddings")
    items = texts_of(_rows(args))
    plan = arms(recipe, port_base=args.port_base, gpus=args.gpus)
    arm_rows: dict[str, dict[str, Any]] = {}
    served: dict[str, dict[str, list[float]]] = {}
    null: dict[str, dict[str, list[float]]] = {}
    for arm in ARMS:
        run = _serve(plan[arm], out=out, timeout_s=args.startup_timeout_s)
        row = {key: value for key, value in run.items() if key != "popen"}
        arm_rows[arm] = row
        if run["state"] != "served":
            print(f"sdpa_probe: the {arm} arm did not serve; its log is {out / f'serve-{arm}.log'}", file=sys.stderr)
            continue
        try:
            served[arm] = encode_all(recipe, run["base_url"], items)
            null[arm] = encode_all(recipe, run["base_url"], items)
        except (HarnessError, ProviderError) as error:
            # The harness's own refusals and the transport's typed request failures (a 404, an exhausted
            # retry budget, a rejected or empty answer): the arm row carries the error and the verdict reads
            # it, so the report the wrapper uploads is written even when a request fails mid-arm.
            arm_rows[arm] = {**row, "state": "failed", "error": str(error)}
        finally:
            _stop(run)
    report = build_report(
        recipe,
        engine={"image": recipe.engine.image, "min_version": recipe.engine.min_version},
        arms=arm_rows,
        served=served,
        null=null,
        texts=[key for key, _role, _content in items],
    )
    write_report(out, report, vectors={"served": served, "null_control": null})
    print(f"sdpa_probe: verdict {report['verdict']['state']} ({report['verdict']['disposition']})")
    return 0 if report["verdict"]["state"] not in ("unmeasured", "nothing_checked") else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main() on the node
    raise SystemExit(main())
