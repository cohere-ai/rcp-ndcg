"""The GPU validation ledger and the recipes' status blocks, regenerated from the waves' own evidence.

``docs/reference/gpu-validation.md`` is the release's record of the GPU work: one row per shipped variant
(the release candidate and the wave whose evidence the row records, the date that wave finished, the
behaviour fingerprint the tree computes today and the state) and the dispositions the owner holds.  Every
family's ``status`` block carries the same facts as data, and the README catalog repeats the state.  The
rows are a wave OUTPUT, so they are regenerated from the waves' evidence -- never written by hand (the plan
forbids writing ``verified`` before a wave ran, and a hand-edited row is exactly how that happens).

This module is that one writer::

    python -m rcp_ndcg_test.jobs.ledger --evidence <mirror>            # a dry run: prints the table
    python -m rcp_ndcg_test.jobs.ledger --evidence <mirror> --write    # the rows, the status blocks, ...

``--evidence`` is a local mirror of the waves' evidence root (the release's own store; the layout is
``<wave>/<image-slug>/`` as :mod:`rcp_ndcg_test.jobs.run_wave` uploads it, with ``judgecheck/`` beside the
waves).  Nothing here reads the network: the mirror is fetched by the operator.

**A row's key.**  A wave's evidence must carry the key the tree computes today, or it is not evidence.  The
equivalence-only waves and the judge probe record no behaviour fingerprint (only a recording wave writes an
``observation_corpus`` step, and that step is what records the key), so the key is DERIVED at the wave's RC
commit -- the commit every ``WAVE.md``/``wave.json`` records -- by checking that commit's tree out of this
repository (``git archive``) and running the tree's own
:func:`rcp_ndcg_test.fingerprint.behaviour_fingerprint` on it, in a subprocess, with that tree's sources on
``PYTHONPATH``.  A wave whose RC commit's tree does not compute the tree's current key is REFUSED by name
(:class:`LedgerEvidenceError`): the wave measured another behaviour, and the fix is a re-run, not a
tolerance.  Where a wave did record a key (the recording waves' ``fingerprints`` map and the corpus step's
``behaviour_fingerprint``), it must equal today's key too -- the cross-check of the derivation.
Recording the key inside every wave is the 0.0.2 follow-up that retires the derivation.

**States.**  ``verified`` only for a recipe whose own validation wave passed, whose row says ``verified``,
whose ``status.json`` and ``equivalence.json`` agree, whose engine reported the version its pinned image
carries, and whose key (derived and, where recorded, recorded) is the tree's.  ``failed`` only for a recipe
the owner's disposition list names (:data:`DISPOSED_FAILED`) beside a wave that ran and failed.  Everything
else -- no wave, a wave that failed for a recipe the owner did not dispose, a wave at another key -- reads
``unverified``: the pin's own rule ("no wave is claimed that did not run"), never a silent downgrade.

**Writes.**  The family ``status`` blocks (textually: the files are hand-formatted, so a YAML round-trip
would destroy their comments), the goldens' resolved ``recipe.status`` (the sanctioned writer's bytes:
``json.dumps(..., indent=1, sort_keys=True)`` plus a newline, the status field only), the README catalog's
state cell and the ledger page's rows.  The write is idempotent: the same evidence writes the same bytes.
A family whose variants disagree keeps its family-level block for the ``unverified`` ones and declares the
others per variant (``Variant.status``: "the family's until a variant declares its own").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rcp_ndcg_vllm.recipe import (
    Recipe,
    RecipeFamily,
    StatusSpec,
    default_recipes_root,
    iter_recipes,
    load_family,
)

from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.fingerprint import behaviour_fingerprint

__all__ = [
    "DISPOSED_FAILED",
    "EVIDENCE_URI",
    "LedgerEvidenceError",
    "Row",
    "Wave",
    "apply",
    "current_keys",
    "main",
    "plan",
    "rc_keys",
    "read_evidence",
]

State = Literal["verified", "failed", "unverified"]
"""The three states a ``status`` block records (``StatusSpec.state``'s own set)."""

DISPOSED_FAILED = frozenset({"topk-embed-v1-small", "topk-embed-v1-xsmall"})
"""The owner's disposition list (AGENTS.md: a ``failed`` variant ships only when it is named here).  A
recipe outside it never reads ``failed``, whatever its wave says: an undisposed failure is a pending row."""

EVIDENCE_URI = "gs://YOUR-BUCKET/rcp-ndcg-rfc/waves"
"""The evidence root the status blocks' ``report`` pointer names, in the public placeholder form the tree's
"public names only" rule requires (the local mirror ``--evidence`` points at has this layout)."""

LEDGER = Path("docs/reference/gpu-validation.md")
"""The ledger page, relative to the checkout root."""

GOLDENS = Path("rcp-ndcg-test/tests/recipes/golden")
"""The per-variant goldens, relative to the checkout root."""

CATALOG = Path("rcp-ndcg-vllm/README.md")
"""The README's rendered catalog, relative to the checkout root."""

RECIPES = Path("rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes")
"""The shipped recipes' root, relative to the checkout root."""

VALIDATION_KINDS = ("equivalence", "record", "gate")
"""The wave kinds whose evidence is a recipe's own validation (``equivalence-<family>``,
``record-<family>``, ``gate-<family>``).  Every other kind -- ``parity-*``, ``*-probe``, ``all-retrieval``,
``e2e`` -- is a probe and never writes a row."""

_KEYS_SCRIPT = """
import json, sys
from pathlib import Path
from rcp_ndcg_vllm.recipe import iter_recipes
from rcp_ndcg_test.fingerprint import behaviour_fingerprint

tree = Path(sys.argv[1])
golden = tree / "rcp-ndcg-test" / "tests" / "recipes" / "golden"
keys, errors = {}, {}
for recipe in iter_recipes(tree / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"):
    try:
        inputs = json.loads((golden / f"{recipe.id}.json").read_text(encoding="utf-8"))["fingerprint"]["inputs"]
        keys[recipe.id] = behaviour_fingerprint(recipe, tokenizer_sha256_value=inputs["tokenizer_sha256"])
    except Exception as error:  # a missing golden or a recipe the old tree cannot resolve
        errors[recipe.id] = f"{type(error).__name__}: {error}"
print(json.dumps({"keys": keys, "errors": errors}))
"""
"""The derivation run inside the RC commit's tree (see :func:`rc_keys`): the tree's own recipes, its own
goldens (the tokenizer SHA-256 is the fingerprint's one file input) and its own fingerprint rule."""

_LEDGER_ROW = re.compile(
    r"^\| `(?P<recipe>[^`]+)` \| (?P<role>[^|]+?) \| (?P<image>[^|]+?) \| (?P<rc>[^|]+?) \| (?P<date>[^|]+?) "
    r"\| `(?P<fingerprint>[0-9a-f]+)…` \| (?P<state>[a-z]+) \|$"
)
_TESTED_RC = re.compile(
    r"^- tested RC: (?P<name>.+?) at commit (?P<commit>\S+) \(manifest sha256 (?P<manifest>\S+)\)$",
    re.MULTILINE,
)
_FINISHED = re.compile(r"^- finished: (?P<finished>\S+)$", re.MULTILINE)
_VERDICT = re.compile(r"^Verdict: \*\*(?P<verdict>[A-Z]+)\*\*$", re.MULTILINE)
_SUMMARY_ROW = re.compile(r"^\| (?P<recipe>[^ |]+) \| (?P<gpus>[^|]*) \| (?P<state>[a-z]+) \|", re.MULTILINE)


class LedgerEvidenceError(HarnessError):
    """The waves' evidence cannot support the row it would write: a wave's own documents disagree with each
    other or are missing, a wave recorded another behaviour key, or the tree at the wave's RC commit does not
    compute the tree's current key.  The message names the recipe, the wave and the fix (a re-run on the
    current tree); the tool writes nothing rather than a row it cannot stand behind."""


@dataclass(frozen=True)
class Wave:
    """One wave's evidence, as the wave wrote it.

    Attributes:
        name: The wave directory's name (``record-octen-embedding``, ``judgecheck``, ``parity-031``).
        kind: ``equivalence``, ``record``, ``gate``, ``judgecheck`` or ``probe`` (see
            :data:`VALIDATION_KINDS`).
        path: The wave's directory under the evidence root.
        rc_name: The release candidate the wave tested (``rc2``), or ``None`` when it recorded none.
        commit: The full commit the wave tested -- the commit whose tree derives the row's key.
        finished: The ISO timestamp the wave finished (the judge probe records its start: it writes no
            finish time).
        verdict: ``passed`` or ``failed`` (the wave's own word).
        rows: ``{recipe id: the wave's row}`` -- the wave summary's per-recipe entry.
        fingerprints: ``{recipe id: key}`` -- the behaviour keys the wave itself recorded (recording waves
            only; the equivalence-only waves and the judge probe record none).
    """

    name: str
    kind: str
    path: Path
    rc_name: str | None
    commit: str | None
    finished: str | None
    verdict: str
    rows: dict[str, dict[str, Any]]
    fingerprints: dict[str, str]


@dataclass(frozen=True)
class Row:
    """One ledger row: what the evidence supports for one shipped recipe.

    Attributes:
        recipe: The variant id (one row per shipped recipe).
        role: The recipe's role (``embed``, ``multi_vector``, ``rerank``, ``judge``).
        image: The engine image the recipe pins today.
        rc: The release candidate the row's wave tested, or ``None`` (no wave claims the row).
        date: The ISO date the row's wave finished, or ``None``.
        fingerprint: The behaviour fingerprint the tree computes today (always: the page's declared meaning).
        state: ``verified``, ``failed`` or ``unverified``.
        wave: The wave directory's name, or ``None``.
        report: The evidence pointer the recipe's status block records, or ``None``.
    """

    recipe: str
    role: str
    image: str
    rc: str | None
    date: str | None
    fingerprint: str
    state: State
    wave: str | None
    report: str | None


def _kind(name: str) -> str:
    """The wave kind a wave directory's name declares (the committed wave lists' naming: one prefix per
    kind).  Inputs: the directory name.  Outputs: one of :data:`VALIDATION_KINDS`, ``judgecheck`` or
    ``probe``.  Units: none."""
    if name == "judgecheck":
        return "judgecheck"
    prefix = name.split("-", 1)[0]
    return prefix if prefix in VALIDATION_KINDS else "probe"


def _json(path: Path, wave: str) -> dict[str, Any]:
    """Read one JSON document of a wave's evidence; a missing or unreadable one is refused by name."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LedgerEvidenceError(
            f"wave {wave}: {path} cannot be read ({error}); the wave's evidence is incomplete"
        ) from error
    if not isinstance(document, dict):
        raise LedgerEvidenceError(f"wave {wave}: {path} is not a JSON object")
    return document


def _check_summary(wave: Wave) -> None:
    """Cross-check the wave's human summary against its data document (the two are written from one
    document, so a disagreement means one of them is another wave's).  Refuses by name."""
    summary = wave.path / "WAVE.md"
    if not summary.is_file():
        return
    text = summary.read_text(encoding="utf-8")
    rc, verdict, finished = _TESTED_RC.search(text), _VERDICT.search(text), _FINISHED.search(text)
    if wave.commit is not None:
        if rc is None:
            raise LedgerEvidenceError(
                f"wave {wave.name}: WAVE.md names no tested RC while wave.json names {wave.commit}"
            )
        if rc["commit"] != wave.commit or rc["name"] != (wave.rc_name or ""):
            raise LedgerEvidenceError(
                f"wave {wave.name}: WAVE.md tested {rc['name']} at {rc['commit']}, wave.json "
                f"{wave.rc_name} at {wave.commit}: the two documents are not one wave's"
            )
    if verdict is not None and verdict["verdict"].lower() != wave.verdict:
        raise LedgerEvidenceError(f"wave {wave.name}: WAVE.md says {verdict['verdict']}, wave.json says {wave.verdict}")
    if finished is not None and wave.finished is not None and finished["finished"] != wave.finished:
        raise LedgerEvidenceError(
            f"wave {wave.name}: WAVE.md finished at {finished['finished']}, wave.json at {wave.finished}"
        )
    stated = {match["recipe"]: match["state"] for match in _SUMMARY_ROW.finditer(text)}
    for recipe_id, row in wave.rows.items():
        if recipe_id in stated and stated[recipe_id] != str(row.get("state")):
            raise LedgerEvidenceError(
                f"wave {wave.name}: WAVE.md lists {recipe_id} as {stated[recipe_id]}, wave.json as {row.get('state')}"
            )


def read_evidence(root: str | Path) -> dict[str, Wave]:
    """Every wave under an evidence mirror, by wave name.

    Inputs: the mirror root (a local copy of the evidence store's ``waves/`` directory: one directory per
    wave).  Outputs: ``{wave name: Wave}``; a wave's documents are cross-checked (its ``WAVE.md`` against its
    ``wave.json``).  Raises :class:`LedgerEvidenceError` when the root is not a directory or a wave's two
    summary documents disagree.  Units: none.
    """
    evidence = Path(root)
    if not evidence.is_dir():
        raise LedgerEvidenceError(f"no evidence mirror at {evidence}: point --evidence at a local copy of the waves")
    waves: dict[str, Wave] = {}
    for directory in sorted(path for path in evidence.iterdir() if path.is_dir()):
        kind = _kind(directory.name)
        if kind == "judgecheck":
            waves[directory.name] = _judge_wave(directory)
            continue
        summaries = [directory / "wave.json", *(p / "wave.json" for p in directory.iterdir() if p.is_dir())]
        found = [path for path in summaries if path.is_file()]
        if not found:
            continue  # a directory with no wave summary is not a wave
        if len(found) > 1:
            raise LedgerEvidenceError(f"wave {directory.name}: {len(found)} wave.json documents under it")
        document = _json(found[0], directory.name)
        rc = document.get("rc") or {}
        verdict = str(document.get("verdict") or ("passed" if document.get("passed") else "failed"))
        wave = Wave(
            name=directory.name,
            kind=kind,
            path=found[0].parent,
            rc_name=rc.get("name"),
            commit=rc.get("commit"),
            finished=document.get("finished"),
            verdict=verdict,
            rows={str(row.get("recipe")): row for row in document.get("recipes") or []},
            fingerprints={str(key): str(value) for key, value in (document.get("fingerprints") or {}).items()},
        )
        _check_summary(wave)
        waves[wave.name] = wave
    if not waves:
        raise LedgerEvidenceError(f"no wave under {evidence}: the mirror holds no wave.json or judgecheck-report.json")
    return waves


def _judge_wave(directory: Path) -> Wave:
    """The judge probe's wave: ``judgecheck-report.json`` (no ``wave.json``/``WAVE.md`` -- it is not a
    ``run_wave`` wave; its bootstrap report names the RC and its ``recipes`` map the per-recipe state)."""
    report = _json(directory / "judgecheck-report.json", directory.name)
    bootstrap = report.get("bootstrap") or {}
    rc = bootstrap.get("rc") or {}
    rows = {
        str(recipe_id): {"state": (entry or {}).get("state")}
        for recipe_id, entry in (report.get("recipes") or {}).items()
    }
    return Wave(
        name=directory.name,
        kind="judgecheck",
        path=directory,
        rc_name=rc.get("name"),
        commit=rc.get("commit"),
        finished=report.get("finished") or report.get("started"),
        verdict="passed" if report.get("passed") else "failed",
        rows=rows,
        fingerprints={},
    )


def current_keys(repo_root: str | Path, recipes: Mapping[str, Recipe]) -> dict[str, str]:
    """The behaviour key the checkout computes today, per recipe.

    Inputs: the checkout root and the recipes to key (loaded from that checkout).  Outputs: ``{recipe id:
    64-hex key}`` -- :func:`rcp_ndcg_test.fingerprint.behaviour_fingerprint` over the resolved recipe, with
    the tokenizer SHA-256 the recipe's golden records (the key's one input the tree does not carry).
    Raises :class:`LedgerEvidenceError` when a recipe has no golden to take it from.  Units: none.
    """
    keys: dict[str, str] = {}
    for recipe_id, recipe in recipes.items():
        path = Path(repo_root) / GOLDENS / f"{recipe_id}.json"
        if not path.is_file():
            raise LedgerEvidenceError(f"no golden at {path}: the key's tokenizer input cannot be resolved")
        try:
            inputs = json.loads(path.read_text(encoding="utf-8"))["fingerprint"]["inputs"]
            tokenizer_sha = str(inputs["tokenizer_sha256"])
        except (OSError, ValueError, KeyError) as error:
            raise LedgerEvidenceError(f"golden {path} carries no fingerprint tokenizer input ({error})") from error
        keys[recipe_id] = behaviour_fingerprint(recipe, tokenizer_sha256_value=tokenizer_sha)
    return keys


def rc_keys(commit: str, repo_root: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """The behaviour keys the tree at one commit computes: the derivation that makes a wave's row its own.

    Inputs: the commit (as a wave's ``WAVE.md``/``wave.json`` records it) and the checkout to read it from.
    Outputs: ``(keys, errors)`` -- ``{recipe id: key}`` for every recipe that commit's tree resolves, and
    ``{recipe id: message}`` for the ones it cannot (a golden the commit predates, a family it cannot load).
    Raises :class:`LedgerEvidenceError` when the commit is not in the checkout, or the derivation itself
    fails.  Units: none.
    """
    with tempfile.TemporaryDirectory(prefix="rcp-ledger-") as temporary:
        tree = Path(temporary) / "tree"
        tree.mkdir()
        archive = subprocess.run(["git", "-C", str(repo_root), "archive", commit], capture_output=True)
        if archive.returncode != 0:
            raise LedgerEvidenceError(
                f"the RC commit {commit} is not in {repo_root} ({archive.stderr.decode(errors='replace').strip()}); "
                "the wave's key cannot be derived -- fetch the history or re-run the wave"
            )
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
        sources = [tree / "rcp-ndcg-core" / "src", tree / "rcp-ndcg" / "src"]
        sources += [tree / "rcp-ndcg-vllm" / "src", tree / "rcp-ndcg-test" / "src"]
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in sources)}
        run = subprocess.run(
            [sys.executable, "-c", _KEYS_SCRIPT, str(tree)], capture_output=True, text=True, env=environment
        )
        if run.returncode != 0:
            raise LedgerEvidenceError(
                f"the tree at {commit} cannot compute its keys: {run.stderr.strip().splitlines()[-1:] or run.stdout}"
            )
        payload = json.loads(run.stdout)
    return dict(payload["keys"]), dict(payload["errors"])


def _evidence_for(recipe: Recipe, waves: Mapping[str, Wave]) -> Wave | None:
    """The newest wave that is ``recipe``'s own validation (a later failure beats an older pass).

    Inputs: the recipe and every wave under the mirror.  Outputs: the wave, or ``None`` when none names the
    recipe (a judge recipe is validated by the judge probe, every other role by an equivalence, recording or
    gate wave).  Units: none.
    """
    kinds = ("judgecheck",) if recipe.role == "judge" else VALIDATION_KINDS
    candidates = [wave for wave in waves.values() if wave.kind in kinds and recipe.id in wave.rows]
    if not candidates:
        return None
    return max(candidates, key=lambda wave: wave.finished or "")


def _image_version(image: str) -> str:
    """The engine version an image tag names (``vllm/vllm-openai:v0.33.0`` -> ``0.33.0``): the string the
    pod reports on ``/version`` and the ``status.json`` of a wave records."""
    tag = image.rsplit(":", 1)[-1]
    return tag.removeprefix("v") if re.fullmatch(r"v?\d+(\.\d+)*", tag) else tag


def _status_of(wave: Wave, recipe: Recipe) -> dict[str, Any]:
    """One recipe's ``status.json`` from a wave: the serve/equivalence steps' own record."""
    return _json(wave.path / recipe.id / "status.json", wave.name)


def _keyed(recipe: Recipe, wave: Wave, key: str, keys_at: Mapping[str, str], errors_at: Mapping[str, str]) -> None:
    """The key check every claimed row rests on: the wave's RC commit exists, its tree keys the recipe, and the
    key it computes is the tree's today.  Raises :class:`LedgerEvidenceError` naming the two keys."""
    if wave.commit is None:
        raise LedgerEvidenceError(f"{recipe.id}: wave {wave.name} records no RC commit, so its key cannot be derived")
    if recipe.id in errors_at:
        raise LedgerEvidenceError(
            f"{recipe.id}: the tree at {wave.commit} (wave {wave.name}) cannot key it: {errors_at[recipe.id]}"
        )
    if keys_at[recipe.id] != key:
        raise LedgerEvidenceError(
            f"{recipe.id}: the tree at {wave.commit} (wave {wave.name}) computes {keys_at[recipe.id][:12]}…, the tree "
            f"today {key[:12]}…: the wave measured another behaviour -- re-run it on the current tree"
        )


def _verified(recipe: Recipe, wave: Wave, key: str, keys_at: Mapping[str, str], errors_at: Mapping[str, str]) -> str:
    """Check one passing wave's evidence against the recipe; return the row's date.

    Inputs: the recipe, its wave, the tree's key, and the keys the wave's RC commit's tree computes (with
    the recipes that tree could not key).  Outputs: the ISO date of the wave's finish.  Raises
    :class:`LedgerEvidenceError` naming the recipe and what disagrees: a missing or failing ``status.json``/
    ``equivalence.json`` (a retrieval recipe) or judge probe record, a pod-reported engine version other than
    the pinned image's, a recorded key other than the tree's, or an RC commit whose tree keys another
    behaviour.  Units: none.
    """
    _keyed(recipe, wave, key, keys_at, errors_at)
    if wave.kind == "judgecheck":
        _judge_evidence(recipe, wave)
    else:
        _equivalence_evidence(recipe, wave, key)
    return str(wave.finished)[:10]


def _failed(recipe: Recipe, wave: Wave, key: str, keys_at: Mapping[str, str], errors_at: Mapping[str, str]) -> str:
    """Check one failing wave's evidence for an owner-disposed recipe; return the row's date.

    The disposition list names the recipe; the wave must have run on the tree's behaviour (the key check) and
    its own documents must agree that it failed.  Raises :class:`LedgerEvidenceError` otherwise.  Units: none.
    """
    _keyed(recipe, wave, key, keys_at, errors_at)
    if wave.verdict == "passed" or str(wave.rows[recipe.id].get("state")) != "failed":
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name} did not fail it ({wave.verdict}, {wave.rows[recipe.id].get('state')!r}); "
            "the owner's disposition records a measured failure, not a pending row"
        )
    if wave.kind != "judgecheck":
        status = _status_of(wave, recipe)
        if str(status.get("state")) != "failed":
            raise LedgerEvidenceError(
                f"{recipe.id}: wave {wave.name}'s summary row says failed, its status.json {status.get('state')!r}"
            )
        if str(status.get("engine_version")) != _image_version(recipe.engine.image):
            raise LedgerEvidenceError(
                f"{recipe.id}: wave {wave.name}'s engine reported {status.get('engine_version')!r} while the recipe "
                f"pins {recipe.engine.image!r}"
            )
    return str(wave.finished)[:10]


def _judge_evidence(recipe: Recipe, wave: Wave) -> None:
    """The judge probe's record for one judge recipe: the report passed, the recipe's own state verified, every
    mode it ran ok and the media check ok or not applicable, on the engine version the recipe pins."""
    report = _json(wave.path / "judgecheck-report.json", wave.name)
    entry = (report.get("recipes") or {}).get(recipe.id) or {}
    if str(entry.get("state")) != "verified":
        raise LedgerEvidenceError(f"{recipe.id}: wave {wave.name} records {entry.get('state')!r} for it, not verified")
    failed = sorted(str(mode) for mode, result in (entry.get("modes") or {}).items() if not (result or {}).get("ok"))
    if failed:
        raise LedgerEvidenceError(f"{recipe.id}: wave {wave.name}'s judge probe failed the modes {failed}")
    media = entry.get("media") or {}
    if media.get("checked") and not media.get("ok"):
        raise LedgerEvidenceError(f"{recipe.id}: wave {wave.name}'s media probe failed: {media.get('detail')!r}")
    version = str((report.get("engine") or {}).get("vllm"))
    if version != _image_version(recipe.engine.image):
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name}'s engine reported {version!r} while the recipe pins "
            f"{recipe.engine.image!r}: a recipe is verified only on the image it pins"
        )


def _equivalence_evidence(recipe: Recipe, wave: Wave, key: str) -> None:
    """A retrieval recipe's record: its ``status.json`` and ``equivalence.json``, its pod-reported engine
    version and the behaviour key the recording step wrote (when it ran)."""
    status = _status_of(wave, recipe)
    if str(status.get("recipe")) != recipe.id:
        raise LedgerEvidenceError(f"{recipe.id}: wave {wave.name} carries {status.get('recipe')!r}'s status.json")
    if str(status.get("state")) != "verified":
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name}'s summary row says verified, its status.json {status.get('state')!r}"
        )
    if str(status.get("engine_version")) != _image_version(recipe.engine.image):
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name}'s engine reported {status.get('engine_version')!r} while the recipe pins "
            f"{recipe.engine.image!r}: a recipe is verified only on the image it pins"
        )
    equivalence = _json(wave.path / recipe.id / "equivalence.json", wave.name)
    if str(equivalence.get("recipe")) != recipe.id:
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name} carries {equivalence.get('recipe')!r}'s equivalence.json"
        )
    if equivalence.get("passed") is not True:
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name}'s summary row says verified, its equivalence.json did not pass"
        )
    if str(equivalence.get("image")) != recipe.engine.image:
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name} declares image {equivalence.get('image')!r}, the recipe pins "
            f"{recipe.engine.image!r}"
        )
    recorded = wave.fingerprints.get(recipe.id) or status.get("behaviour_fingerprint")
    if recorded is not None and str(recorded) != key:
        raise LedgerEvidenceError(
            f"{recipe.id}: wave {wave.name} recorded the key {str(recorded)[:12]}…, the tree computes {key[:12]}…"
        )


def plan(evidence: str | Path, repo_root: str | Path) -> list[Row]:
    """The ledger's rows, decided from the waves' evidence and the checkout's recipes.

    Inputs: the evidence mirror and the checkout (its recipes and goldens are the rows' recipe set and keys).
    Outputs: one :class:`Row` per shipped recipe, in recipe-id order -- ``verified`` only where the wave's
    evidence is complete and keys the tree's behaviour, ``failed`` only for the owner's disposition list
    beside a wave that failed, ``unverified`` otherwise.  Raises :class:`LedgerEvidenceError` (see
    :func:`_verified`) or :class:`rcp_ndcg_vllm.errors.RecipeError` for a recipe set that does not load.
    Units: none.
    """
    root = Path(repo_root)
    recipes = {recipe.id: recipe for recipe in iter_recipes(root / RECIPES)}
    if not recipes:
        raise LedgerEvidenceError(f"no recipe under {root / RECIPES}")
    keys = current_keys(root, recipes)
    waves = read_evidence(evidence)
    derived: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    rows: list[Row] = []
    for recipe_id in sorted(recipes):
        recipe = recipes[recipe_id]
        wave = _evidence_for(recipe, waves)
        row = Row(
            recipe=recipe_id,
            role=recipe.role,
            image=recipe.engine.image,
            rc=None,
            date=None,
            fingerprint=keys[recipe_id],
            state="unverified",
            wave=None,
            report=None,
        )
        if wave is None:
            rows.append(row)
            continue
        report = f"{EVIDENCE_URI}/{wave.path.relative_to(Path(evidence)).as_posix()}/"
        passed = wave.verdict == "passed" and str(wave.rows[recipe_id].get("state")) == "verified"
        if passed:
            date = _verified(recipe, wave, keys[recipe_id], *_at(wave, root, derived))
            rows.append(_row(row, "verified", wave, date, report))
            continue
        if recipe_id in DISPOSED_FAILED:
            # The disposition's own row: the wave ran and failed, and its key must still be the tree's.
            date = _failed(recipe, wave, keys[recipe_id], *_at(wave, root, derived))
            rows.append(_row(row, "failed", wave, date, report))
            continue
        rows.append(row)
    return rows


def _at(wave: Wave, repo_root: Path, cache: dict[str, tuple[dict[str, str], dict[str, str]]]):
    """The keys of ``wave``'s RC commit's tree, derived once per commit per run."""
    commit = str(wave.commit)
    if commit not in cache:
        cache[commit] = rc_keys(commit, repo_root)
    return cache[commit]


def _row(row: Row, state: State, wave: Wave, date: str, report: str) -> Row:
    """The row with its claim filled in from the wave."""
    return Row(
        recipe=row.recipe,
        role=row.role,
        image=row.image,
        rc=wave.rc_name,
        date=date or None,
        fingerprint=row.fingerprint,
        state=state,
        wave=wave.name,
        report=report,
    )


_STATUS_KEYS = ("state", "image", "date", "report")
"""The ``status`` block's fields, in the order the family files write them."""


def _fields(spec: StatusSpec) -> tuple[str | None, ...]:
    """A status block's values as one comparable tuple (a pydantic model is not hashable)."""
    return tuple(getattr(spec, key) for key in _STATUS_KEYS)


def _scalar(key: str, value: str | None) -> str:
    """One status value as the family files write it (``state`` bare, the rest double-quoted, ``null``)."""
    if value is None:
        return "null"
    return value if key == "state" else json.dumps(value)


def _value(text: str) -> str | None:
    """One status value as the family files write it, read back (``null``, a quoted string or a bare one)."""
    if text == "null":
        return None
    return json.loads(text) if text.startswith('"') else text


def _block(spec: StatusSpec, indent: int) -> list[str]:
    """The YAML lines of a ``status:`` block at ``indent`` spaces."""
    pad = " " * indent
    lines = [f"{pad}status:\n"]
    for key in ("state", "image", "date", "report"):
        lines.append(f"{pad}  {key}: {_scalar(key, getattr(spec, key))}\n")
    return lines


def _replace_block(lines: list[str], start: int, indent: int, spec: StatusSpec) -> list[str]:
    """Replace the ``status:`` block whose key line is ``lines[start]`` with ``spec``'s lines."""
    pad = " " * indent
    end = start + 1
    while end < len(lines) and lines[end].startswith(pad + "  "):
        end += 1
    return lines[:start] + _block(spec, indent) + lines[end:]


def _family_status(lines: list[str]) -> tuple[int, StatusSpec | None]:
    """The family-level ``status:`` key line's index and the block it declares (``None`` when absent)."""
    for index, line in enumerate(lines):
        if line.rstrip("\n") == "status:":
            values = {}
            for inner in lines[index + 1 :]:
                if not inner.startswith("  ") or inner.strip() == "":
                    break
                key, _, value = inner.strip().partition(":")
                values[key.strip()] = _value(value.strip())
            return index, StatusSpec.model_validate(values)
    return -1, None


def _set_variant_status(lines: list[str], variant_id: str, spec: StatusSpec) -> list[str]:
    """Declare ``spec`` on one variant: replace its own ``status:`` block, or insert one after its ``id:``."""
    marker = f"  - id: {variant_id}\n"
    start = lines.index(marker)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("  - ") or (lines[index].strip() and not lines[index].startswith("    ")):
            break
        if lines[index].startswith("    status:"):
            return _replace_block(lines, index, 4, spec)
    return lines[: start + 1] + _block(spec, 4) + lines[start + 1 :]


def _write_families(repo_root: Path, rows: Mapping[str, Row]) -> list[Path]:
    """Write every family's ``status`` block(s) from the rows; return the files that changed.

    A ``verified``/``failed`` row writes the wave's block (its state, the image the recipe pins, the wave's
    date and its evidence pointer).  An ``unverified`` row claims nothing, so its block stays the one it
    already declares -- the image it was last verified against.  A family-level block stands for the block
    most of its variants share; a variant that differs declares its own (``Variant.status``: "the family's
    until a variant declares its own").  Every new text is computed before anything is written, and every
    written file is reloaded and its resolved statuses compared with the rows: a textual edit that does not
    read back is refused, never left in the tree.
    """
    planned: list[tuple[Path, str, dict[str, StatusSpec]]] = []
    for directory in sorted(path for path in (repo_root / RECIPES).iterdir() if (path / "family.yaml").is_file()):
        family: RecipeFamily = load_family(directory)
        path = directory / "family.yaml"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        index, _ = _family_status(lines)
        if index < 0:
            raise LedgerEvidenceError(f"{path}: the family declares no status block")
        wanted = {
            variant.id: (
                variant.status or family.status
                if rows[variant.id].state == "unverified"
                else StatusSpec(
                    state=rows[variant.id].state,
                    image=rows[variant.id].image,
                    date=rows[variant.id].date,
                    report=rows[variant.id].report,
                )
            )
            for variant in family.variants
        }
        order = [_fields(wanted[variant.id]) for variant in family.variants]
        shared = max(set(order), key=lambda value: (order.count(value), -order.index(value)))
        default = next(spec for spec in wanted.values() if _fields(spec) == shared)
        updated = _replace_block(lines, index, 0, default)
        declared = {variant_id for variant_id, spec in wanted.items() if _fields(spec) == shared}
        for variant_id in sorted(set(wanted) - declared):
            updated = _set_variant_status(updated, variant_id, wanted[variant_id])
        planned.append((path, "".join(updated), wanted))
    written: list[Path] = []
    for path, text, _ in planned:
        if text != path.read_text(encoding="utf-8"):
            path.write_text(text, encoding="utf-8")
            written.append(path)
    for path, _, wanted in planned:
        reloaded = load_family(path.parent)
        for variant in reloaded.variants:
            resolved = variant.status or reloaded.status
            if resolved != wanted[variant.id]:
                raise LedgerEvidenceError(
                    f"{path}: {variant.id}'s written status {resolved} is not the row's {wanted[variant.id]}"
                )
    return written


def _write_goldens(repo_root: Path, recipes: Mapping[str, Recipe]) -> list[Path]:
    """Write each golden's resolved ``recipe.status`` in the sanctioned writer's bytes; return the changed."""
    written: list[Path] = []
    for recipe_id, recipe in recipes.items():
        path = repo_root / GOLDENS / f"{recipe_id}.json"
        if not path.is_file():
            raise LedgerEvidenceError(f"no golden at {path}: the status has nowhere to land")
        original = path.read_text(encoding="utf-8")
        document = json.loads(original)
        document["recipe"]["status"] = recipe.status.model_dump(mode="json")
        text = json.dumps(document, indent=1, sort_keys=True) + "\n"
        if text != original:
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


def _write_catalog(repo_root: Path, recipes: Mapping[str, Recipe]) -> list[Path]:
    """Write the README catalog's state cell from the resolved recipes; return the files that changed."""
    path = repo_root / CATALOG
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    for index, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) < 4:
            continue
        recipe_id = cells[2].strip().strip("`")
        if recipe_id not in recipes:
            continue
        cells[-2] = f" {recipes[recipe_id].status.state} "
        lines[index] = "|".join(cells)
        seen.add(recipe_id)
    missing = sorted(set(recipes) - seen)
    if missing:
        raise LedgerEvidenceError(f"{path}: no catalog row for {missing}")
    text = "".join(lines)
    if text == path.read_text(encoding="utf-8"):
        return []
    path.write_text(text, encoding="utf-8")
    return [path]


def _ledger_line(row: Row) -> str:
    """One ledger row, as the page renders it (``—`` where no wave claims the cell)."""
    rc = f"`{row.rc}`" if row.rc else "—"
    return (
        f"| `{row.recipe}` | {row.role} | `{row.image}` | {rc} | {row.date or '—'} "
        f"| `{row.fingerprint[:12]}…` | {row.state} |"
    )


def _write_ledger(repo_root: Path, rows: Iterable[Row]) -> list[Path]:
    """Write the ledger page's rows (one per recipe, in place); return the page when it changed."""
    path = repo_root / LEDGER
    targets = {row.recipe: row for row in rows}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    for index, line in enumerate(lines):
        match = _LEDGER_ROW.match(line.rstrip("\n"))
        if match is None or match["recipe"] not in targets:
            continue
        lines[index] = _ledger_line(targets[match["recipe"]]) + "\n"
        seen.add(match["recipe"])
    missing = sorted(set(targets) - seen)
    if missing:
        raise LedgerEvidenceError(f"{path}: no ledger row for {missing}")
    text = "".join(lines)
    if text == path.read_text(encoding="utf-8"):
        return []
    path.write_text(text, encoding="utf-8")
    return [path]


def apply(repo_root: str | Path, rows: Iterable[Row]) -> list[Path]:
    """Write every artifact the rows decide: the status blocks, the goldens, the catalog and the ledger.

    Inputs: the checkout root and the rows :func:`plan` returned.  Outputs: the paths that changed (empty on
    a second run of the same evidence -- the write is idempotent).  Raises :class:`LedgerEvidenceError` when
    a written status does not read back as the row's, or an artifact has no place for a row.  Units: none.
    """
    root = Path(repo_root)
    ordered = list(rows)
    targets = {row.recipe: row for row in ordered}
    written = _write_families(root, targets)
    recipes = {recipe.id: recipe for recipe in iter_recipes(root / RECIPES)}
    if set(recipes) != set(targets):
        raise LedgerEvidenceError(
            f"the written tree resolves {sorted(set(recipes) - set(targets))} / misses "
            f"{sorted(set(targets) - set(recipes))}"
        )
    written += _write_goldens(root, recipes)
    written += _write_catalog(root, recipes)
    written += _write_ledger(root, ordered)
    return written


def _table(rows: Iterable[Row]) -> list[str]:
    """The dry run's table: one line per recipe, then the totals."""
    lines = [f"{'recipe':45s} {'state':11s} {'rc':5s} {'date':11s} {'key':14s} wave"]
    ordered = list(rows)
    for row in ordered:
        lines.append(
            f"{row.recipe:45s} {row.state:11s} {row.rc or '-':5s} {row.date or '-':11s} "
            f"{row.fingerprint[:12]}… {row.wave or '-'}"
        )
    states = {state: sum(1 for row in ordered if row.state == state) for state in ("verified", "failed", "unverified")}
    lines.append(
        f"{len(ordered)} rows: {states['verified']} verified, {states['failed']} failed, "
        f"{states['unverified']} unverified"
    )
    return lines


def default_repo_root() -> Path:
    """The checkout the installed recipes live in (the tool's default read/write root).

    Walks up from the recipe package's own directory until a directory holds both workspace members the
    ledger spans (``rcp-ndcg-vllm`` and ``rcp-ndcg-test``).  Raises :class:`LedgerEvidenceError` when no such
    directory exists (an installed wheel with no checkout: pass ``--repo-root``).  Units: none.
    """
    for parent in Path(default_recipes_root()).resolve().parents:
        if (parent / "rcp-ndcg-vllm").is_dir() and (parent / "rcp-ndcg-test").is_dir():
            return parent
    raise LedgerEvidenceError("no checkout around the installed recipes: pass --repo-root")


def main(argv: list[str] | None = None) -> int:
    """The CLI: a dry run prints the table; ``--write`` performs it.  Exit 0 on success, 2 on a refusal."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.jobs.ledger",
        description="Regenerate the GPU ledger, the status blocks, the goldens and the catalog from the waves.",
    )
    parser.add_argument("--evidence", required=True, help="a local mirror of the waves' evidence root")
    parser.add_argument(
        "--repo-root", default=None, help="the checkout to read and write (default: the installed tree's)"
    )
    parser.add_argument("--write", action="store_true", help="write the artifacts (default: a dry run)")
    args = parser.parse_args(argv)
    try:
        root = Path(args.repo_root) if args.repo_root is not None else default_repo_root()
        rows = plan(Path(args.evidence), root)
        for line in _table(rows):
            print(line)
        if args.write:
            for path in apply(root, rows):
                print(f"wrote {path.relative_to(root)}")
    except HarnessError as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
