"""The repo-wide hygiene guards over the shipped tree.

A shipped recipe is package data of a public distribution: every line must read as a self-contained public
statement. Internal process shorthand (a lane or sweep name, a private review id), a private work directory or
a local machine path is not a source a reader of the wheel can resolve, and the standing rule forbids them
(the ctxl family's own guard is the pattern; this module is its one repo-wide home, so a family that ships a
new file cannot pass unnoticed). The same rule covers the rest of the shipped tree: no tracked file outside
the scaffolding (``handover/``, ``experiments/``) cites the internal design document or a lane branch -- the
docs wording test forbids those ids in Markdown and the gate's public-names step forbids operator paths, so
this scan is their one home everywhere else. Offline by construction: it reads the tracked tree, nothing else.
"""

from __future__ import annotations

import gzip
import re
import subprocess
from pathlib import Path

from rcp_ndcg_vllm.recipe import default_recipes_root

INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl|vl|late|zerank|qwen3)"
    r"|\bsweep|lanes' base|audit-synth"
    r"|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb|qwen3vl-emb|qwen3vl-rer|topk|pplx|zerank[12]?|qwen3-rer)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|\bG[1-5]\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d|shake"
    r"|\bworkstream\b|\brecipe-fix\b|\bhandover/"
    r"|(?:^|[\s\"'(=])/(?:root|home|Users|private|tmp)/"
)
"""The internal-label and private-path pattern every shipped recipe file is scanned with."""

#: The one legal ``gs://`` host in a shipped recipe file: the declared placeholder (AGENTS.md's
#: public-names rule). A wave report lives in operator storage, so any other bucket host is a private name
#: a reader of the wheel cannot resolve -- the placeholder is what a shipped file cites. A bare scheme
#: mention (``a ``gs://`` object``) carries no host and is not a bucket.
PLACEHOLDER_BUCKET = "YOUR-BUCKET"
_GS_HOST = re.compile(r"gs://([^/\s`\"')\\]+)")

#: The internal design-document and lane-branch citations no shipped file may carry: the design document is
#: operator scaffolding and a lane branch is unknown outside this workspace. Case-insensitive: a lowercased
#: citation of the same design id is the same id. The scan skips :data:`SCAFFOLDING` (see the module docstring).
DESIGN_AND_LANE_IDS = re.compile(r"\bRFC[- ]?\d{3,4}\b|\blane/", re.IGNORECASE)

#: The retired internal validation spec's id: the file is gone, its content folded into the public pages
#: (``docs/reference/gpu-validation.md``, ``docs/how-to/validate-a-recipe.md`` and the fake-engines and
#: release-candidate pages), so a citation of it is a source a reader cannot resolve. Case-sensitive on
#: purpose: the live public page's lowercase ``gpu-validation.md`` name is not the retired id. Split so this
#: module's own source does not carry the literal it forbids.
INTERNAL_SPEC_IDS = re.compile(r"GPU-" r"VALIDATION")

#: The internal design-id shorthand the docs wording guard forbids in Markdown, in the non-Markdown homes
#: the release publishes: the shipped recipe files (package data of ``rcp-ndcg-vllm``) and the generated
#: schemas (``rcp-ndcg-vllm/schema/`` and the root ``schemas/``). ``tests/docs/test_wording.py`` owns the
#: Markdown half; this is the same rule for the data files it cannot read. The last alternative is a plan
#: step id (``the release plan's A1b``): an internal step label a reader of the wheel cannot resolve.
DESIGN_IDS = re.compile(
    r"owner decisions? \d+|\bdecision \d+|\blane L\d+|\blayout-move item \d+|\bplan'?s? (?:step )?[A-Z]\d+[a-z]?\b",
    re.IGNORECASE,
)

#: The scaffolding directories the shipped-tree scan skips: working notes and paper reproduction, deleted at
#: release.
SCAFFOLDING = ("handover", "experiments")

#: The committed corpora: the repository subset of every recorded observation (a manifest with the pod's
#: environment and argv, a subset ``index.json`` naming where the full corpus lives, the records).
CORPORA = "rcp-ndcg-test/corpora"

#: The private-infrastructure shapes no committed corpus may carry: a private-range address (10/8, 172.16/12
#: or 192.168/16 -- a pod's Service endpoint), a Kubernetes Service-link name (``*_SERVICE_HOST`` /
#: ``*_SERVICE_PORT`` and the ``VLLM_SERVICE_`` prefix the Services ride under, which the pod's injection puts
#: beside the engine's own ``VLLM_*`` knobs) and the pod's staging paths. A committed corpus records the pod,
#: so this is where a Service address or a staging path lands first.
PRIVATE_INFRASTRUCTURE = re.compile(
    r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    r"|\b192\.168\.\d{1,3}\.\d{1,3}\b"
    r"|\b172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}\b"
    r"|_SERVICE_HOST\b|_SERVICE_PORT\b|VLLM_SERVICE_"
)

#: A machine path in a corpus: the pod's staging directory or an operator checkout. The serve argv records
#: the paths the pod served from, so a staging root reaches a manifest as directly as an address does.
MACHINE_PATH = re.compile(r"(?:^|[\s\"'(=])/(?:root|home|Users|private|tmp)/")

#: The files that legitimately *name* the Service-link shape: the filter that drops it out of a new recording,
#: this module's own pattern, and the filter's test. Every other tracked byte must not carry it.
SERVICE_LINK_CARRIERS = frozenset(
    {
        "rcp-ndcg-test/src/rcp_ndcg_test/observe/provenance.py",
        "rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py",
        "rcp-ndcg-test/tests/test_observe_corpus.py",
    }
)

#: The Service-link shape on its own. Unlike a bare private-range address (a public document may quote one) or
#: a ``gs://`` host (the docs show ``gs://bucket`` in URI examples), this shape names a Kubernetes Service's
#: injected link variables and nothing else, so it can be held across the whole tracked tree.
SERVICE_LINK = re.compile(r"_SERVICE_HOST\b|_SERVICE_PORT\b|VLLM_SERVICE_")

#: The real cluster subnet and the real Service-host assignment shape, split into fragments so this module's
#: own source -- which must name the shape -- does not carry the literal it forbids.
_REAL_CLUSTER_SUBNET = re.compile(r"10\." r"156\.")
_REAL_SERVICE_HOST = re.compile(r"(?:VLLM_SERVICE_[A-Z0-9_]+|KUBERNETES)_SERVICE_HOST=10\.")

#: The trees that hold tests: a fixture is as published as the code it exercises.
TEST_TREES = ("tests", "rcp-ndcg-test/tests", "rcp-ndcg-vllm/tests")


#: The per-column media fields the harness retired in favour of one ``media`` field (W2). Every family's
#: ``reference.py`` refuses them, because a row carrying one would otherwise be encoded or scored as text --
#: a different prompt, never compared. The refusal is five copies **by design** (a reference subprocess
#: imports nothing from the product, so it cannot share the product's helper); this module is the one place
#: that names the list, so a sixth retired column cannot be added to four of the five.
RETIRED_MEDIA_COLUMNS = ("query_image", "query_video", "documents_images", "documents_videos")

_COLUMN_TUPLE = re.compile(r"for key in \(([^)]*)\) if row\.get\(key\)")


def _repository_root() -> Path:
    """The checkout root: this module lives three directories below it."""
    return Path(__file__).resolve().parents[3]


def _shipped_files(root: Path) -> list[Path]:
    """Every tracked file under ``root`` except the scaffolding (see :data:`SCAFFOLDING`).

    The tracked tree is what a release ships; an untracked file is not part of it.
    """
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [
        root / name for name in listed.decode("utf-8").split("\0") if name and name.split("/", 1)[0] not in SCAFFOLDING
    ]


def test_no_shipped_file_cites_an_internal_design_or_lane_id() -> None:
    """No tracked file outside the scaffolding cites the internal design document, a lane branch or
    the retired validation spec.

    ``handover/`` and ``experiments/`` are scaffolding, deleted at release; the docs wording test covers
    Markdown and the gate's public-names step operator paths, so this scan is the one home for the rest of the
    shipped tree (the product's code and tests, the test package's cases and goldens).
    """
    root = _repository_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in _shipped_files(root)
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if DESIGN_AND_LANE_IDS.search(line) or INTERNAL_SPEC_IDS.search(line)
    ]
    assert not hits, "internal design or lane ids in the shipped tree:\n" + "\n".join(hits)


def test_every_reference_refuses_the_same_retired_media_columns() -> None:
    """The five copies of the retired-column refusal name the same columns (see the constant above)."""
    root = default_recipes_root()
    found: dict[str, tuple[str, ...]] = {}
    for path in sorted(root.glob("*/reference.py")):
        text = path.read_text(encoding="utf-8")
        if "_refuse_old_media_columns" not in text:
            continue
        match = _COLUMN_TUPLE.search(text)
        assert match is not None, f"{path.parent.name}: the refusal no longer names its columns in the pinned shape"
        found[path.parent.name] = tuple(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert found, "no family's reference refuses the retired media columns"
    assert len(set(found.values())) == 1, f"the retired-column copies disagree: {found}"
    assert set(next(iter(found.values()))) == set(RETIRED_MEDIA_COLUMNS), found


def test_the_design_id_scan_catches_a_plan_step_id() -> None:
    """The shape that shipped: an internal plan step id ("the release plan's A1b") is the same class as
    a numbered decision -- a reader of the wheel cannot resolve it, so the scan must catch it wherever it
    cites the step. The recipe files carry none (the two citations are reworded to the constraint itself).
    """
    assert DESIGN_IDS.search("the release plan's A1b re-verification runs on the final RC")
    assert DESIGN_IDS.search("the plan's step B2")
    assert DESIGN_IDS.search("the release plan's a1b")  # a lowercased citation is the same id
    assert not DESIGN_IDS.search("the plan is to run the gate on the final candidate")
    assert not DESIGN_IDS.search("the engine's own default clip budget would decide")


def test_no_shipped_recipe_file_names_a_real_bucket() -> None:
    """Every ``gs://`` a shipped recipe file carries is the ``gs://YOUR-BUCKET/`` placeholder.

    The wave reports are operator storage; a real bucket or an operator prefix is not a source a reader of
    the wheel can resolve, and the repo's public-names rule declares the placeholder the public form. The
    scan covers the same shipped-file set as the internal-label scan below, so the placeholder cannot
    regress into a private bucket unnoticed. The corpus subset ``index.json`` files name the full corpus's
    storage path and are covered by :func:`test_no_committed_corpus_names_private_infrastructure` (a
    whole-tree scan would read the CHANGELOG's own ``gs://bucket/`` example as a host).
    """
    root = default_recipes_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if (match := _GS_HOST.search(line)) and match.group(1) != PLACEHOLDER_BUCKET
    ]
    assert not hits, "real buckets in shipped recipe files:\n" + "\n".join(hits)


def _corpus_text(path: Path) -> str:
    """One committed corpus file's text: a ``.gz`` record file is decompressed first, so the scan reads the
    recorded exchanges as well as the manifests (the credential scan reads the subset's records the same way)."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def test_the_service_link_pattern_matches_the_injected_shape() -> None:
    """The tree-wide guard's pattern still matches what it exists to catch, so it cannot rot unnoticed."""
    assert SERVICE_LINK.search("VLLM_SERVICE_EXAMPLE_EMB_SERVICE_HOST=192.0.2.10")  # a synthetic address
    assert SERVICE_LINK.search("VLLM_SERVICE_EXAMPLE_EMB_SERVICE_PORT")
    assert SERVICE_LINK.search("KUBERNETES_SERVICE_HOST=192.0.2.1")  # the pod's own API-server link
    assert not SERVICE_LINK.search("VLLM_BUILD_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6")
    assert not SERVICE_LINK.search("RCP_NDCG_VLLM_PATCHES=pooling-full-context")


def test_no_test_file_names_a_real_cluster_host() -> None:
    """No test file names a real cluster host: the cluster's ``10.<subnet>.`` addresses, or a
    ``VLLM_SERVICE_*_SERVICE_HOST`` bound to a private address.

    The guard's own fixture was the leak: :func:`test_the_service_link_pattern_matches_the_injected_shape` used
    the pod's real Service names and addresses as the shape to match (review finding security-2), and a test
    fixture is as published as the code it exercises. The scan covers every tracked test file, this module
    included; the patterns are split so this module's own source does not carry the literal it forbids.
    """
    root = _repository_root()
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    hits = [
        f"{name}:{number}: {line.strip()[:140]}"
        for name in listed.decode("utf-8").split("\0")
        if name and any(name == tree or name.startswith(f"{tree}/") for tree in TEST_TREES)
        for number, line in enumerate((root / name).read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if _REAL_CLUSTER_SUBNET.search(line) or _REAL_SERVICE_HOST.search(line)
    ]
    assert not hits, "a real cluster host literal in the test tree:\n" + "\n".join(hits)


def test_no_tracked_file_outside_the_corpora_names_a_service_link() -> None:
    """No tracked file outside the carriers names a Kubernetes Service link.

    The corpora scan below covers the private-range addresses, the buckets and the machine paths -- shapes with
    legitimate public occurrences (a public document may quote a private address; the docs show ``gs://bucket``
    in URI examples), which is why that scan is scoped to the corpora, where the leak landed. This shape is
    unambiguous, so it is held tree-wide instead: a manifest-like artifact committed anywhere -- a fixture, a
    log, a wave report -- cannot carry the pod's Service links either. The corpora are skipped here because the
    scan below already reads every byte of them.
    """
    root = _repository_root()
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    hits = [
        f"{name}:{number}: {line.strip()[:140]}"
        for name in listed.decode("utf-8").split("\0")
        if name and name not in SERVICE_LINK_CARRIERS and not name.startswith(CORPORA)
        for number, line in enumerate(_corpus_text(root / name).splitlines(), start=1)
        if SERVICE_LINK.search(line)
    ]
    assert not hits, (
        "a Kubernetes Service link is named in a tracked file outside the three carriers (the pod's Service "
        "links are private infrastructure):\n" + "\n".join(hits)
    )


def test_no_committed_corpus_names_private_infrastructure() -> None:
    """No committed corpus carries a private-range address, a Kubernetes Service-link name, a real bucket or
    a machine path.

    A corpus manifest records the pod's environment and the exact serve argv, so the pod's Service links
    (internal addresses, and internal or unreleased Service names under the ``VLLM_`` prefix the engine's own
    knobs use) and the pod's staging paths land in a tracked file first; each subset ``index.json`` names the
    full corpus's storage path. The scan covers every tracked corpus byte (the records included) and is the
    corpus half of the hygiene guards: what the engine-environment filter in
    :mod:`rcp_ndcg_test.observe.provenance` keeps out of a new recording, this holds out of the committed one.
    """
    root = _repository_root()
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", CORPORA], check=True, capture_output=True
    ).stdout
    hits = [
        f"{name}:{number}: {line.strip()[:140]}"
        for name in listed.decode("utf-8").split("\0")
        if name
        for number, line in enumerate(_corpus_text(root / name).splitlines(), start=1)
        if PRIVATE_INFRASTRUCTURE.search(line)
        or MACHINE_PATH.search(line)
        or ((match := _GS_HOST.search(line)) and match.group(1) != PLACEHOLDER_BUCKET)
    ]
    assert not hits, "private infrastructure in the committed corpora:\n" + "\n".join(hits)


def test_every_shipped_recipe_file_reads_as_a_public_statement() -> None:
    """Every shipped recipe file and template reads as a public statement, judge families included.

    The scan covers the internal labels and private paths (:data:`INTERNAL_LABELS`) and the design-id
    shorthand (:data:`DESIGN_IDS`) the Markdown wording guard forbids -- the recipe files are package data
    of the published wheel, so a design id there is the same leak as one in a page.
    """
    root = default_recipes_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if INTERNAL_LABELS.search(line) or DESIGN_IDS.search(line)
    ]
    assert not hits, "internal labels in shipped recipe files:\n" + "\n".join(hits)


def test_every_generated_schema_reads_as_a_public_statement() -> None:
    """The generated schemas carry no internal design ids either.

    ``rcp-ndcg-vllm/schema/*.schema.json`` is exported public surface of the vllm distribution (grafted
    into its sdist) and the root ``schemas/`` is the product's exported surface; both are generated from
    docstrings, so a design id in a docstring ships in a JSON file the wording guard cannot read.
    """
    root = _repository_root()
    schema_files = sorted((root / "schemas").glob("*.json")) + sorted(
        (root / "rcp-ndcg-vllm" / "schema").glob("*.json")
    )
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in schema_files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if DESIGN_IDS.search(line)
    ]
    assert not hits, "internal design ids in generated schemas:\n" + "\n".join(hits)
