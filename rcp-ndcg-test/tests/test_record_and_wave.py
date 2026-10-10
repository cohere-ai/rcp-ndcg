"""The recorder and the wave runner: the product's wire path observed through the recording transport."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from rcp_ndcg_test.jobs import run_wave as run_wave_module
from rcp_ndcg_test.jobs import weights
from rcp_ndcg_test.jobs.run_wave import _ZMQ_IPC_SUFFIX_CHARS, _slot_tmp_dir, run_wave
from rcp_ndcg_test.record import record
from rcp_ndcg_vllm import load_recipe

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub

REFERENCE_PYTHON = sys.executable
VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"
REFERENCE_PY = sys.executable


def test_record_writes_exchanges_per_route(tmp_path: Path) -> None:
    """The recorded set: models, the role route, /score, the over-length 400 and the unknown-field 400."""

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in written]
    routes = [(document["route"], document["status"]) for document in documents]
    assert ("http://engine/v1/models", 200) in routes
    assert ("http://engine/v1/embeddings", 200) in routes  # the role route, the product's own request
    # the over-length and unknown-field probes record the engine's 400 bodies on the role route, not 200s
    statuses_400 = [document for document in documents if document["status"] == 400]
    assert statuses_400, "the over-length and unknown-field probes must record the engine's 400"
    assert {document["route"] for document in statuses_400} == {"http://engine/v1/embeddings"}
    for path in written:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert "http://engine" in document["route"]
        assert "127.0.0.1" not in json.dumps(document)


def test_record_over_length_probe_crosses_the_engine_cap(tmp_path: Path) -> None:
    """The over-length probe is sized from serve.max_model_len and records the engine's over-length 400."""

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in written]
    over_length = next(
        document
        for document in documents
        if document["status"] == 400 and document["route"] == "http://engine/v1/embeddings"
    )
    message = over_length["body"]["error"]["message"]
    assert str(recipe.serve.max_model_len) in message  # the engine refused the over-length prompt


def test_record_exchanges_carry_the_product_shape(tmp_path: Path) -> None:
    """The recorded requests match the product's adapter (the wire path, not a second path)."""

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in written]
    role_route = next(
        document
        for document in documents
        if document["route"] == "http://engine/v1/embeddings" and document["status"] == 200
    )
    body = role_route["body"]
    assert "embedding" in body["data"][0]
    # the role request is the product's shape: model = the recipe id, float encoding
    request = role_route["request"]["body"]
    assert request["model"] == "fixture-embed"
    assert request["encoding_format"] == "float"


def test_wave_runs_a_recipe_end_to_end(tmp_path: Path) -> None:
    """One recipe on one slot with the stub engine: serve, smoke, equivalence."""
    out = tmp_path / "wave"
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "fixture-embed.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8"
    )
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=out,
        pairs_dir=pairs_dir,
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "verified"
    assert (out / "fixture-embed" / "serve.log").is_file()
    assert (out / "fixture-embed" / "status.json").is_file()
    assert (out / "wave.json").is_file()
    equivalence = json.loads((out / "fixture-embed" / "equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["reference_outputs"]["state"] == "computed"  # the store is under <out>/references
    assert equivalence["reference_environment"]["family"] == "fixture-embed"
    assert list((out / "references").iterdir()), "the computed reference output is stored"


def test_wave_resolves_the_family_reference_environment(tmp_path: Path) -> None:
    """Owner decision 35: with --reference-root, each recipe's reference runs from its family's venv
    (``<root>/<family>/bin/python``), not from one pod-wide interpreter."""
    reference_root = tmp_path / "reference"
    family_bin = reference_root / "fixture-embed" / "bin"
    family_bin.mkdir(parents=True)
    (family_bin / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    (family_bin / "python").chmod(0o755)
    out = tmp_path / "wave"
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "fixture-embed.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8"
    )
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=out,
        pairs_dir=pairs_dir,
        reference_root=reference_root,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "verified"


def test_reference_python_for_prefers_the_explicit_override(tmp_path: Path) -> None:
    """The resolver: an explicit --reference-python wins; otherwise the family venv under the root."""
    recipe = load_recipe(RECIPES / "fixture-embed")
    root = tmp_path / "reference"
    explicit = tmp_path / "explicit" / "python"
    assert run_wave_module._reference_python_for(recipe, str(explicit), str(root)) == str(explicit)
    assert run_wave_module._reference_python_for(recipe, None, str(root)) == str(
        root / "fixture-embed" / "bin" / "python"
    )
    assert run_wave_module._reference_python_for(recipe, None, None) is None


def test_environment_facts_record_the_lock_and_freeze(tmp_path: Path) -> None:
    """Decision 35 item 5: the report names the family, the lock's SHA-256 (the environment identity)
    and, when the bootstrap recorded it, the venv's freeze."""
    from rcp_ndcg_test.jobs.reference_env import environment_facts
    from rcp_ndcg_vllm.recipe import resolve_recipe

    recipe = resolve_recipe("zerank-1-reranker")
    root = tmp_path / "reference"
    (root / "zerank").mkdir(parents=True)
    (root / "zerank" / "freeze.txt").write_text("torch==2.13.0+cu130\ntransformers==4.57.6\n", encoding="utf-8")
    facts = environment_facts(recipe._dir, root)
    assert facts["family"] == "zerank"
    assert len(str(facts["lock_sha256"])) == 64
    assert facts["freeze"] == ["torch==2.13.0+cu130", "transformers==4.57.6"]
    assert environment_facts(recipe._dir, None)["family"] == "zerank"


def test_wave_records_disk_and_evicts_after_the_last_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node-runtime item 8 in the wave runner: the disk record per recipe, the eviction once.

    Two recipes serving the same model run in parallel on two slots; the model's cache is evicted only
    after the last of them stopped, and every recipe's status carries the disk before/after.
    """
    cache = tmp_path / "hub"
    model_dir = cache / "models--fixtures--DenseEmbedder"
    (model_dir / "snapshots" / "0123456789abcdef0123456789abcdef01234567").mkdir(parents=True)
    (model_dir / "snapshots" / "0123456789abcdef0123456789abcdef01234567" / "model.safetensors").write_bytes(
        b"0" * (3 << 20)
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    # The fixtures sit two levels above a recipe dir (../../tokenizer.json, ../../deterministic.py).
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed-cls" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("model: fixtures/ClsEmbedder", "model: fixtures/DenseEmbedder"),
        encoding="utf-8",
    )
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        recipes_root,
        gpus=2,
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed", "fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert all(row["state"] == "verified" for row in document["recipes"]), by_id
    disks = {row["recipe"]: row["disk"] for row in document["recipes"]}
    assert disks["fixture-embed"]["free_disk_bytes"] > 0
    assert disks["fixture-embed"]["model_bytes"] is None  # offline: the size is unknown, recorded
    evictions = [disks[row["recipe"]].get("evicted") for row in document["recipes"]]
    assert evictions.count(True) == 1, evictions  # one model, one eviction, after the last recipe
    assert not model_dir.exists()


def test_smoke_sends_a_judge_recipe_a_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every recipe role has a smoke route: a judge (role 'judge') POSTs /v1/chat/completions. The rc0
    judges wave failed all ten judges with ``KeyError: 'judge'`` because the route table had no entry."""
    import httpx
    from rcp_ndcg_vllm.recipe import resolve_recipe

    recipe = resolve_recipe("gemma-4-12b-it")
    seen: dict[str, object] = {}

    class Reply:
        status_code = 200

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def post(self, route: str, json: dict | None = None) -> Reply:
            seen["route"] = route
            seen["json"] = json
            return Reply()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    report = run_wave_module._smoke(recipe, "http://127.0.0.1:8100/v1")
    assert report["state"] == "passed", report
    assert seen["route"] == "/v1/chat/completions", seen
    body = seen["json"]
    assert isinstance(body, dict) and body.get("messages"), body


def test_equivalence_skips_a_judge_with_the_decision_15_reason() -> None:
    """A judge has no reference (decision 15): the equivalence step is skipped by design, never failed."""
    from rcp_ndcg_vllm.recipe import resolve_recipe

    report = run_wave_module._equivalence(
        resolve_recipe("gemma-4-12b-it"),
        "http://127.0.0.1:1",
        "/tmp/unused",
        "rcp-ndcg-test/pairs",
        None,
        device="cpu",
        reference_gpu=None,
    )
    assert report["state"] == "skipped", report
    assert "judge check" in report["reason"], report


def test_a_judges_skipped_equivalence_and_record_count_as_satisfied() -> None:
    """A judge's T0 evidence (serve + smoke) verifies it: its equivalence and recorder steps are skipped
    by design. An embed recipe's skipped equivalence does NOT verify it (the rc0 judges wave failed all
    ten judges on KeyError('judge') from the recorder and on the equivalence refusal)."""
    from rcp_ndcg_vllm.recipe import resolve_recipe

    judge_steps = {
        "serve": {"state": "passed"},
        "smoke": {"state": "passed"},
        "equivalence": {"state": "skipped"},
        "record": {"state": "skipped"},
    }
    assert (
        run_wave_module._row_state(
            resolve_recipe("gemma-4-12b-it"),
            judge_steps,
            record=True,
            record_corpus=False,
            quality=False,
            controls=False,
        )
        == "verified"
    )
    assert (
        run_wave_module._row_state(
            resolve_recipe("jina-embeddings-v5-text-nano"),
            {"smoke": {"state": "passed"}, "equivalence": {"state": "skipped"}, "record": {"state": "passed"}},
            record=True,
            record_corpus=False,
            quality=False,
            controls=False,
        )
        == "failed"
    )


def test_reference_device_is_cpu_for_a_judge() -> None:
    """A judge has no reference: the device helper returns cpu instead of raising, so the equivalence
    step reaches its skip (the rc0 judges wave failed on `reference_of` before the skip could run)."""
    from rcp_ndcg_vllm.recipe import resolve_recipe

    assert run_wave_module._reference_device(resolve_recipe("gemma-4-12b-it"), 3) == "cpu"
    assert run_wave_module._reference_device(resolve_recipe("jina-embeddings-v5-text-nano"), None) == "cpu"
    assert run_wave_module._reference_device(resolve_recipe("jina-embeddings-v5-text-nano"), 1) == "cuda"


def test_smoke_sends_a_token_ids_recipe_its_own_query_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-vector recipe whose pooling wire carries the role-prefixed token ids
    (``client.request_shape: token_ids``, the pplx-embed-v2-context plugin's contract) is smoked with the
    recipe's own query render as ids.  The bare-text body the other roles get is a request the recipe's
    client never sends, and that engine's pooler refuses it by name -- the refusal kills the EngineCore
    (the r2 wave's pplx-context serve failure, `got [3445, 4587, 1414]` for the literal smoke text)."""
    import httpx
    import yaml
    from rcp_ndcg_test.equivalence import fitting

    copied = tmp_path / "recipes" / "smoke-token-ids"
    shutil.copytree(RECIPES / "fixture-multi-vector", copied)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["id"] = "smoke-token-ids"
    data["variants"][0]["id"] = "smoke-token-ids"
    data["client"]["tokenizer"] = str(TOKENIZER)
    data["client"]["request_shape"] = "token_ids"
    data["client"]["template"]["query"] = [{"fixed": "Q: "}, {"content": "query"}]
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(copied)
    template = fitting.client_template(recipe)
    assert template is not None
    tokenizer = fitting.tokenizer_of(recipe)
    expected = list(
        tokenizer.ids(
            template.render("query", tokenizer, query="smoke query"),
            add_special_tokens=template.adds_special_tokens("query"),
        )
    )
    seen: dict[str, object] = {}

    class Reply:
        status_code = 200

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def post(self, route: str, json: dict | None = None) -> Reply:
            seen["route"] = route
            seen["json"] = json
            return Reply()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    report = run_wave_module._smoke(recipe, "http://127.0.0.1:8100")
    assert report["state"] == "passed", report
    assert seen["route"] == "/pooling", seen
    body = seen["json"]
    assert isinstance(body, dict) and body["input"] == [expected], body
    assert body["input"] != ["smoke text"], "the bare-text body is what the plugin refuses"


def test_the_post_serve_steps_of_two_recipes_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C4/scale F2: a ready recipe's steps run in a worker of their own, so one recipe's slow post-serve
    step never serializes the other recipes through the scheduler loop (the pre-harness-fix runner called
    ``_finalise`` inline and every other ready engine idled while it ran)."""
    real_smoke = run_wave_module._smoke
    lock = threading.Lock()
    gate = threading.Event()
    started: list[str] = []

    def slow_smoke(recipe, base_url):
        with lock:
            started.append(recipe.id)
            if len(started) == 2:
                gate.set()
        assert gate.wait(30), f"the other recipe's smoke never started (the steps serialized): {started}"
        return real_smoke(recipe, base_url)

    monkeypatch.setattr(run_wave_module, "_smoke", slow_smoke)
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        RECIPES,
        gpus=4,  # each recipe holds its engine's GPU plus the reference's
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed", "fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert sorted(started) == ["fixture-embed", "fixture-embed-cls"]
    assert all(row["state"] == "verified" for row in document["recipes"])


def test_wave_fails_a_recipe_that_measurably_cannot_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose size cannot fit the free disk fails before its engine started (one line)."""
    monkeypatch.setattr(weights, "model_disk_bytes", lambda model, revision=None: 1 << 40)  # 1 TiB of weights
    monkeypatch.setattr(weights, "disk_free_bytes", lambda path: 2 << 30)  # 2 GiB free: it cannot fit
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "free" in (row["error"] or "") and "GiB" in (row["error"] or "")
    assert row["disk"]["model_bytes"] == 1 << 40
    assert row["steps"]["serve"]["state"] == "failed"
    assert not (tmp_path / "wave" / "fixture-embed" / "serve.log").exists()  # no engine ever started


def test_wave_runs_on_a_fresh_pod_before_the_cache_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a fresh pod the HF cache does not exist before the first download.  The disk check
    must measure the nearest existing parent (never `FileNotFoundError: .../huggingface/hub`), and the
    post-recipe eviction of an empty cache is a recorded no-op."""
    cache = tmp_path / "fresh" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert not cache.exists()
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "verified", row
    assert row["disk"]["free_disk_bytes"] > 0
    assert row["disk"]["evicted"] is False  # nothing was cached yet; recorded, not an error
    assert not cache.exists()  # even the eviction did not create the cache directory


def test_wave_gives_each_slot_its_own_short_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node-runtime item 7 (an engine died when its ZMQ path blew the cap): vLLM's ZMQ IPC sockets live under the
    slot's TMPDIR as ``<uuid>``, and AF_UNIX caps paths at 107 characters -- so each slot's TMPDIR is a
    short per-slot dir OUTSIDE the output tree (recipe ids and the state prefix never reach it), one
    per slot, and it is removed with its engine."""
    slots_root = tmp_path / "slots"
    slots_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(slots_root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        RECIPES,
        gpus=2,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed", "fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert all(row["state"] == "verified" for row in document["recipes"])
    tmpdirs = {}
    for row in document["recipes"]:
        tmpdir = row["steps"]["serve"]["tmpdir"]  # recorded, per slot
        assert row["recipe"] not in tmpdir  # the recipe id never lengthens the path
        assert Path(tmpdir).is_relative_to(slots_root)
        assert not Path(tmpdir).exists()  # removed with its engine
        tmpdirs[row["recipe"]] = tmpdir
    assert len(set(tmpdirs.values())) == 2  # one home per slot


def test_slot_tmp_dirs_fit_vllms_zmq_ipc_with_the_longest_recipe_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 107-character AF_UNIX budget with the bootstrap's real prefix as the temp root and the
    longest recipe ids in play: ``<slot tmpdir>/<uuid>`` must fit whatever the recipe id is."""
    monkeypatch.setattr(tempfile, "tempdir", "/tmp/rcp-bootstrap.XXXXXX")  # the bootstrap's real prefix
    longest = max((directory.name for directory in RECIPES.iterdir() if directory.is_dir()), key=len)
    for slot in range(3):
        tmpdir = _slot_tmp_dir(slot)
        assert longest not in str(tmpdir) and "ctxl" not in str(tmpdir)  # no id in the path: any length
        assert len(str(tmpdir)) + _ZMQ_IPC_SUFFIX_CHARS <= 107  # sun_path, including the uuid + its slash


def _pairs_dir(tmp_path: Path, recipe_ids: set[str]) -> Path:
    """A pairs directory with one row per named recipe, so each recipe's equivalence stage runs."""
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    for recipe_id in recipe_ids:
        (pairs / f"{recipe_id}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8"
        )
    return pairs


def test_wave_survives_an_engine_that_dies_mid_run(tmp_path: Path) -> None:
    """GPU-E1: one engine's crash took down the pod.  An engine that dies (SIGABRT) while its recipe's
    steps run fails ONLY that recipe's serve step -- with the engine's last log lines in serve.log and a
    tail of them in the step document -- and the wave's other recipes still verify."""
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        RECIPES,
        gpus=4,  # engine + reference GPU per recipe: both recipes run
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed", "fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --fault abort --fault-only fixture-embed --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "failed"
    serve = by_id["fixture-embed"]["steps"]["serve"]
    assert serve["state"] == "failed"  # the dead engine's serve step carries the evidence:
    assert serve.get("log_tail"), "the status records the engine's last log lines"
    assert any("the abort fault begins here" in line for line in serve["log_tail"])
    assert "exited" in (by_id["fixture-embed"].get("error") or "")
    assert (out / "fixture-embed" / "serve.log").read_text(encoding="utf-8").count("the abort fault begins here") >= 1
    assert by_id["fixture-embed-cls"]["state"] == "verified", by_id["fixture-embed-cls"]  # the others continue
    assert document["passed"] is False


def test_wave_starts_each_engine_in_its_own_session(tmp_path: Path) -> None:
    """GPU-E1: each engine runs in its own session/process group, so one engine's group can be signalled
    (or die) without reaching the runner or another engine."""
    assert "start_new_session=True" in Path(run_wave_module.__file__).read_text(encoding="utf-8")
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert document["recipes"][0]["state"] == "verified"


def test_wave_gives_the_reference_a_gpu_of_its_own_and_records_it(tmp_path: Path) -> None:
    """GPU-E1: every reference ran on the pod's CPU.  The runner reserves one GPU per recipe for its
    reference (never the engine's), pins the reference subprocess to it, and records the device and the
    GPU in the step document and equivalence.json."""
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=2,
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "verified", row
    step = row["steps"]["equivalence"]
    assert step["reference_device"] == "cuda"
    assert step["reference_gpu"] == 1  # the reference's own GPU, beside the engine's GPU 0
    report = json.loads((out / "fixture-embed" / "equivalence.json").read_text(encoding="utf-8"))
    assert report["device"] == "cuda" and report["reference_gpu"] == 1


def test_wave_packs_engines_so_reference_gpus_remain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """8 GPUs and single-GPU recipes with a reference GPU each: at most 4 recipes run at once (the packing
    holds engine + reference GPUs until the recipe's steps end, so the fifth waits for a slot)."""
    lock = threading.Lock()
    alive: set[int] = set()
    most = 0
    real_start = run_wave_module._start
    real_stop = run_wave_module._EngineRun.stop

    def counting_start(recipe, gpus, slot, out, vllm_cmd, port_base, **kwargs):
        nonlocal most
        run = real_start(recipe, gpus, slot, out, vllm_cmd, port_base, **kwargs)
        with lock:
            alive.add(id(run))
            most = max(most, len(alive))
        return run

    def counting_stop(self) -> None:
        real_stop(self)
        with lock:
            alive.discard(id(self))

    monkeypatch.setattr(run_wave_module, "_start", counting_start)
    monkeypatch.setattr(run_wave_module._EngineRun, "stop", counting_stop)
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    ids = []
    for index in range(5):
        recipe_dir = recipes_root / f"packed-{index}"
        shutil.copytree(RECIPES / "fixture-embed", recipe_dir)
        text = (recipe_dir / "family.yaml").read_text(encoding="utf-8")
        (recipe_dir / "family.yaml").write_text(
            text.replace("id: fixture-embed", f"id: packed-{index}"), encoding="utf-8"
        )
        ids.append(f"packed-{index}")
    document = run_wave(
        ids,
        recipes_root,
        gpus=8,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, set(ids)),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert all(row["state"] == "verified" for row in document["recipes"]), document["recipes"]
    assert most == 4, most  # 8 GPUs / (1 engine + 1 reference) = 4 concurrent recipes, never 5


def test_wave_honours_a_recipe_declaring_a_cpu_reference(tmp_path: Path) -> None:
    """``reference.device: cpu`` means the reference must NOT take a GPU: the runner reserves none (the
    declared device wins over the runner's own choice) and records cpu with no reference GPU, on a pod
    that has a spare one."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("entry: reference.py", "entry: reference.py\n  device: cpu"),
        encoding="utf-8",
    )
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed"],
        recipes_root,
        gpus=2,
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "verified", row
    step = row["steps"]["equivalence"]
    assert step["reference_device"] == "cpu"
    assert "reference_gpu" not in step  # no GPU was reserved for a CPU reference
    report = json.loads((out / "fixture-embed" / "equivalence.json").read_text(encoding="utf-8"))
    assert report["device"] == "cpu" and "reference_gpu" not in report


def test_wave_refuses_a_cuda_reference_recipe_that_cannot_get_its_own_gpu(tmp_path: Path) -> None:
    """A recipe that requires reference.device: cuda on a pod that cannot give the reference a GPU of
    its own fails early, with the way out -- never a silent CPU reference run."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("entry: reference.py", "entry: reference.py\n  device: cuda"),
        encoding="utf-8",
    )
    document = run_wave(
        ["fixture-embed"],
        recipes_root,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "reference.device: cuda" in (row.get("error") or "")
    assert "pack fewer engines" in (row.get("error") or "")
    assert not (tmp_path / "wave" / "fixture-embed" / "serve.log").exists()  # no engine was ever started


def test_wave_fails_a_stuck_request_at_its_step_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU-E1: one stuck request held a node for hours.  A request that never answers is cancelled at its
    step's declared budget and the step fails with
    `step <name> exceeded <budget>s; in flight: <method path, request index>`; the engine stops and the
    other recipes continue.  (The formula's constants are scaled down so the overrun is test-scale.)"""
    monkeypatch.setattr(run_wave_module, "_STEP_BASE_S", 8.0)
    monkeypatch.setattr(run_wave_module, "_STEP_PER_REQUEST_S", 2.0)
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        RECIPES,
        gpus=4,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed", "fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --fault hang --fault-only fixture-embed --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    failed = by_id["fixture-embed"]
    assert failed["state"] == "failed"
    equivalence = failed["steps"]["equivalence"]
    assert equivalence["state"] == "failed"
    assert "step equivalence exceeded" in (equivalence.get("error") or ""), equivalence
    assert "in flight: POST /v1/embeddings (request" in (equivalence.get("error") or ""), equivalence
    # The request was CANCELLED at the budget's edge (the transport's wait_for), not left to the product
    # client's 600 s timeout: the step ends within a few seconds of its budget, far below the executor's
    # 30 s abandon grace.  Removing the wait_for bound in wire.py pushes secs past budget + 30.
    assert equivalence["secs"] < equivalence["budget_s"] + 15, equivalence
    assert not failed["steps"]["serve"]["state"] == "failed"  # the engine answered; the step budget stopped it
    assert by_id["fixture-embed-cls"]["state"] == "verified"  # the other recipes continue


def test_wave_fails_a_step_whose_body_ignores_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A step body the watch cannot reach (a subprocess the harness drove, like the quality stage) still
    fails at its budget: the executor abandons it and names the step.  The recipe's own
    ``engine.step_budget_s`` floor only raises the budget: 0.5s computed, 3s declared -> 3s."""
    monkeypatch.setattr(run_wave_module, "_STEP_BASE_S", 0.5)
    monkeypatch.setattr(run_wave_module, "_STEP_PER_REQUEST_S", 0.5)

    def deaf_smoke(recipe, base_url):
        time.sleep(10.0)
        return {"state": "passed"}

    monkeypatch.setattr(run_wave_module, "_smoke", deaf_smoke)
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace(
            "startup_timeout_s: 60", "startup_timeout_s: 60, step_budget_s: 3"
        ),
        encoding="utf-8",
    )
    document = run_wave(
        ["fixture-embed"],
        recipes_root,
        gpus=2,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "failed"
    smoke = row["steps"]["smoke"]
    assert smoke["state"] == "failed"
    assert smoke["budget_s"] == 3  # the recipe's floor raised the formula's 1s
    assert "step smoke exceeded 3s" in (smoke.get("error") or ""), smoke
    assert "in flight:" in (smoke.get("error") or "")


def test_wave_writes_status_after_every_step_and_uploads_each_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU-E1: results used to land only at the end.  status.json is rewritten (atomically, through the
    product's publish) after every step -- a slow step's running state is on disk while it runs -- and
    with --upload each finished recipe's directory lands the moment the recipe ends."""
    real_smoke = run_wave_module._smoke

    def slow_smoke(recipe, base_url):
        time.sleep(1.0)
        return real_smoke(recipe, base_url)

    monkeypatch.setattr(run_wave_module, "_smoke", slow_smoke)
    out = tmp_path / "wave"
    bucket = tmp_path / "bucket"
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=2,
        out_dir=out,
        upload=str(bucket),
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert document["recipes"][0]["state"] == "verified"
    status = out / "fixture-embed" / "status.json"
    # The running state was on disk while the step ran, and the finished state is there now.
    final = json.loads(status.read_text(encoding="utf-8"))
    assert final["state"] == "verified"
    # Each finished recipe's directory was uploaded the moment the recipe ended (before wave.json exists).
    uploaded = json.loads((bucket / "fixture-embed" / "status.json").read_text(encoding="utf-8"))
    assert uploaded["state"] == "verified"
    assert (bucket / "fixture-embed" / "equivalence.json").is_file()
    assert (bucket / "fixture-embed" / "serve.log").is_file()


def test_wave_logs_one_line_per_step_start_and_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """GPU-E1: the pod log was silent.  Every step's start and end lands on the log as
    `run_wave: <recipe> <step> start|passed|failed <secs>s` -- no request bodies, no environment values."""
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=2,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert document["recipes"][0]["state"] == "verified"
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run_wave: ")]
    assert "run_wave: fixture-embed smoke start" in lines
    passed = [line for line in lines if line.startswith("run_wave: fixture-embed smoke passed ")]
    assert passed and passed[0].rstrip().endswith("s") and "secs" not in passed[0]
    assert "run_wave: fixture-embed equivalence start" in lines
    assert any(line.startswith("run_wave: fixture-embed equivalence passed ") for line in lines)
    assert not any("http" in line.lower() and "127.0.0.1" in line for line in lines)  # no request bodies


def test_wave_verifies_a_recipe_whose_controls_all_pass(tmp_path: Path) -> None:
    """A ``--controls`` wave: the controls stop the recipe's engine on purpose after their variants, which
    is not an engine death.  A recipe whose every applicable control the gates catch stays verified (the
    deliberate stop used to be read as a death, failing the row with an empty error)."""
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --model-pooling LAST --tokenizer {TOKENIZER}",
        port_base=0,
        controls=True,
    )
    row = document["recipes"][0]
    assert row["state"] == "verified", row
    assert row["steps"]["controls"]["state"] == "passed", row["steps"]["controls"]
    assert document["passed"] is True


def test_the_controls_run_the_reference_on_the_recipe_s_device(tmp_path: Path) -> None:
    """The controls' gates run the reference on the recipe's own device (and GPU), as its own gates did: a
    cuda-declared recipe's controls used to run on CPU and fail on the refusal, not on the control."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("entry: reference.py", "entry: reference.py\n  device: cuda"),
        encoding="utf-8",
    )
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed"],
        recipes_root,
        gpus=2,
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --model-pooling LAST --tokenizer {TOKENIZER}",
        port_base=0,
        controls=True,
    )
    row = document["recipes"][0]
    assert row["steps"]["controls"]["state"] == "passed", row["steps"]["controls"]
    reports = sorted((out / "fixture-embed" / "controls").glob("*/equivalence.json"))
    assert reports, "the controls wrote no gate reports"
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["device"] == "cuda", report_path
        assert report.get("reference_gpu") == 1, report_path


def test_no_engine_starts_once_the_wave_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wave's end closes ITS wave and sweeps the live-engine registry: an abandoned corpus body that
    calls ``_start`` after its worker's snapshot fails loudly instead of leaking a GPU-holding engine past
    the wave -- and a *later* wave opening never reopens an earlier wave's flag (the shared-Event bug: a
    stale body's late start was admitted into whichever wave happened to be open, on a TMPDIR its own
    wave's cleanup removes)."""
    from rcp_ndcg_test.errors import HarnessError

    recipe = load_recipe(RECIPES / "fixture-embed")
    monkeypatch.setattr(run_wave_module, "_CURRENT_WAVE", [run_wave_module._Wave(token="live")])
    closed = run_wave_module._Wave(token="dead", closed=True)
    with pytest.raises(HarnessError, match="closed"):
        run_wave_module._start(recipe, [0], 0, tmp_path, VLLM_CMD, 0, wave=closed)
    stale = run_wave_module._Wave(token="stale")  # open, but not the wave that is current
    with pytest.raises(HarnessError, match="closed"):
        run_wave_module._start(recipe, [0], 0, tmp_path, VLLM_CMD, 0, wave=stale)
    assert run_wave_module._LIVE_ENGINES == set()  # nothing registered by the refused starts


def test_every_wave_gets_its_own_slot_tmpdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two waves in one process (a test session runs many) never share a slot TMPDIR: the path carries the
    wave's own token, so a previous wave's leftover engine or abandoned thread can neither remove nor
    reuse the TMPDIR the next wave's engine runs with (the shared ``rcp-s<pid>-<slot>`` path was exactly
    that hazard)."""
    monkeypatch.setattr(run_wave_module.tempfile, "tempdir", "/tmp/rcp-slot-test")
    first = run_wave_module._slot_tmp_dir(0, wave="aaaaaa")
    second = run_wave_module._slot_tmp_dir(0, wave="bbbbbb")
    assert first != second and first.name != second.name
    assert "aaaaaa" in first.name and "bbbbbb" in second.name
    assert first.parent == second.parent  # both under the system temp dir, short and outside the output
    assert run_wave_module._slot_tmp_dir(0) != run_wave_module._slot_tmp_dir(1)  # per slot within a wave


def test_stopping_an_engine_signals_its_own_session_by_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The engine runs in its own session (``start_new_session``), so its process group IS its pid: the
    stop signals ``popen.pid`` directly -- never ``os.getpgid``, whose lookup is a second syscall that can
    name another process group if the child's number was reused between the two calls."""
    import signal as signal_module

    calls: list[tuple[int, int]] = []

    class _FakePopen:
        pid = 4242

        def __init__(self) -> None:
            self.stdout = __import__("io").BytesIO(b"")
            self.gone = False

        def poll(self) -> int | None:
            return 0 if self.gone else None

        def wait(self, timeout: float | None = None) -> int:
            self.gone = True
            return 0

    monkeypatch.setattr(run_wave_module.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    monkeypatch.setattr(
        run_wave_module.os, "getpgid", lambda pid: pytest.fail("the stop must not look the group up again")
    )
    recipe = load_recipe(RECIPES / "fixture-embed")
    run = run_wave_module._EngineRun(
        recipe,
        [0],
        1234,
        _FakePopen(),
        tmp_path / "serve.log",
        tmp_path,
        wave=run_wave_module._Wave(token="t"),
    )
    run.stop()
    assert calls == [(4242, signal_module.SIGTERM)]
    assert run.stopped_by_runner is True
    run.stop()  # idempotent: the engine is gone, nothing more is signalled
    assert calls == [(4242, signal_module.SIGTERM)]


def test_wave_logs_the_serve_boundaries_and_writes_its_running_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The serve step is on the pod log and on disk from its start: an engine's load can take minutes and
    the log/status must show it (GPU-E1 finding 3)."""
    snapshots: list[dict] = []
    real_publish = run_wave_module._publish_status

    def recording(path, document):
        snapshots.append(json.loads(json.dumps(document)))
        real_publish(path, document)

    monkeypatch.setattr(run_wave_module, "_publish_status", recording)
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    assert document["recipes"][0]["state"] == "verified"
    # The FIRST write is the serve start alone: no other step has run yet (deleting the serve
    # status write leaves the first snapshot the smoke step's, which this pins).
    assert set(snapshots[0]["steps"]) == {"serve"}, snapshots[0]["steps"]
    assert snapshots[0]["steps"]["serve"]["state"] == "running"
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run_wave: ")]
    assert "run_wave: fixture-embed serve start" in lines
    assert any(line.startswith("run_wave: fixture-embed serve passed ") for line in lines)


def test_wave_reports_the_declared_request_timeout(tmp_path: Path) -> None:
    """GPU-E1 finding 7: the harness's own requests (smoke, record) run with one declared per-request
    timeout, shorter than every step budget, reported in the step documents."""
    from rcp_ndcg_test.jobs.run_wave import _REQUEST_TIMEOUT_S, _STEP_BASE_S

    assert _REQUEST_TIMEOUT_S < _STEP_BASE_S  # shorter than every step budget
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=2,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        port_base=0,
        record=True,
    )
    row = document["recipes"][0]
    assert row["steps"]["smoke"]["request_timeout_s"] == _REQUEST_TIMEOUT_S
    assert row["steps"]["record"]["request_timeout_s"] == _REQUEST_TIMEOUT_S


def test_wave_records_the_serve_step_success_and_a_clean_stop(tmp_path: Path) -> None:
    """With smoke and record passed and equivalence deliberately skipped (no pairs
    yet), the serve step records ITS success and a clean stop is not a failure -- and a row that does
    not verify says why, never bare."""
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
        record=True,
    )
    row = document["recipes"][0]
    assert row["steps"]["smoke"]["state"] == "passed"
    assert row["steps"]["record"]["state"] == "passed"
    assert row["steps"]["equivalence"]["state"] == "skipped"
    assert row["steps"]["serve"]["state"] == "passed"  # serving succeeded; a clean stop is no failure
    assert row["state"] == "failed"  # unverified (equivalence skipped), with the reason recorded:
    assert "verification incomplete" in (row.get("error") or "") and "no pairs file" in (row.get("error") or "")


def test_the_all_recipes_wave_list_isolates_a_broken_family(tmp_path: Path) -> None:
    """An empty wave list ("every recipe") tolerates a broken family: it is a failed entry, never an abort.

    The regression: the empty-ids branch expanded the families outside the tolerant loop, so one broken
    ``family.yaml`` raised ``RecipeError`` out of ``load_wave`` and the wave never started -- contradicting
    ``load_wave``'s own contract and the wave runner's "one failing recipe never stops the wave".
    """
    from rcp_ndcg_test.jobs.wavelist import load_wave

    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    broken = recipes_root / "broken-recipe"
    broken.mkdir()
    broken_text = (RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8")
    (broken / "family.yaml").write_text(
        broken_text.replace("id: fixture-embed", "id: broken-recipe") + "bogus-field: true\n", encoding="utf-8"
    )
    recipes, failed = load_wave([], recipes_root)
    assert failed.get("broken-recipe"), failed
    assert "bogus-field" in failed["broken-recipe"]
    assert {recipe.id for recipe in recipes} >= {"fixture-embed", "fixture-embed-cls"}


def test_wave_marks_an_invalid_recipe_failed_with_the_validation_message(tmp_path: Path) -> None:
    """One failing recipe never stops the wave, end to end: a recipe that fails validation is a failed
    row in the wave report (with the validation message), and the wave runs the rest."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    broken = recipes_root / "broken-recipe"
    broken.mkdir()
    broken_text = (RECIPES / "fixture-embed" / "family.yaml").read_text(encoding="utf-8")
    (broken / "family.yaml").write_text(
        broken_text.replace("id: fixture-embed", "id: broken-recipe") + "bogus-field: true\n", encoding="utf-8"
    )
    out = tmp_path / "wave"
    document = run_wave(
        ["fixture-embed", "broken-recipe"],
        recipes_root,
        gpus=1,
        out_dir=out,
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "verified", by_id
    assert by_id["broken-recipe"]["state"] == "failed"
    assert "bogus-field" in (by_id["broken-recipe"].get("error") or "")  # the validation message
    assert document["passed"] is False
    assert (out / "broken-recipe" / "status.json").is_file()
    wave_md = (out / "WAVE.md").read_text(encoding="utf-8")
    assert "broken-recipe" in wave_md and "failed" in wave_md
    pipe_rows = [line for line in wave_md.splitlines() if line.startswith("|")]
    assert len(pipe_rows) == 2 + 2  # header, separator and ONE row per recipe (errors are one line)
    assert all(line.rstrip().endswith("|") for line in pipe_rows)  # no cell split across lines


def test_wave_fails_the_recipes_of_a_plugin_the_bootstrap_could_not_install(tmp_path: Path) -> None:
    """A plugin found nowhere (the bootstrap records it) fails exactly the recipes that name it, with
    the plugin's exact name in the message -- the rest of the wave runs."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace("  plugin: null\n", "  plugin: Private-Plugin.Name==1.2.3\n"),
        encoding="utf-8",
    )
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        recipes_root,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed-cls"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
        failed_plugins={"Private-Plugin.Name==1.2.3"},
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed-cls"]["state"] == "verified", by_id
    assert by_id["fixture-embed"]["state"] == "failed"
    assert "Private-Plugin.Name==1.2.3" in (by_id["fixture-embed"].get("error") or "")  # the exact name


def test_wave_names_a_failed_step_in_the_row_error_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row never fails bare: a failed step (with its own error) is named in the row's error as well
    (the WAVE.md error column previously stayed empty while the step carried the cause)."""
    monkeypatch.setattr(
        run_wave_module,
        "_smoke",
        lambda _recipe, _base_url: {"state": "failed", "error": "smoke blew up"},
    )
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert row["steps"]["smoke"]["state"] == "failed"
    assert "smoke blew up" in (row.get("error") or "")


def test_wave_cleans_the_slot_tmpdir_when_the_engine_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed engine start leaves no scratch TMPDIR behind (one per slot must not accumulate)."""
    slots_root = tmp_path / "slots"
    slots_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(slots_root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        reference_python=REFERENCE_PYTHON,
        vllm_cmd="/nonexistent/binary",
        port_base=0,
    )
    assert document["recipes"][0]["state"] == "failed"
    assert list(slots_root.iterdir()) == []  # RED: the slot dir leaks without the fix


def test_wave_fails_only_the_recipes_whose_collected_plugin_form_failed(tmp_path: Path) -> None:
    """The failed-plugin match is exactly the form `collect` emits for that recipe: a recipe whose
    staged wheel file (collected as <recipe-directory>/<file>) installed fine is never failed because another
    recipe's bare <file> spec failed (the exact name still appears in the failing recipe's row)."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    spec = "plugin_wheel-1.0.0-py3-none-any.whl"
    (recipes_root / "fixture-embed" / spec).write_bytes(b"stub wheel bytes")  # staged in A's dir only
    for recipe_id in ("fixture-embed", "fixture-embed-cls"):
        recipe_yaml = recipes_root / recipe_id / "family.yaml"
        recipe_yaml.write_text(
            recipe_yaml.read_text(encoding="utf-8").replace(
                "  plugin: null\n",
                f"  plugin: {spec}\n  plugin_architectures: [FixturePluginModel]\n",
            ),
            encoding="utf-8",
        )
    document = run_wave(
        ["fixture-embed", "fixture-embed-cls"],
        recipes_root,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=_pairs_dir(tmp_path, {"fixture-embed"}),
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
        failed_plugins={spec},  # the BARE form failed (as the bootstrap records B's failure)
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "verified", by_id  # A's <id>/<file> form installed fine
    assert by_id["fixture-embed-cls"]["state"] == "failed"
    assert spec in (by_id["fixture-embed-cls"].get("error") or "")


def test_the_wave_start_renders_the_recipes_patches_into_the_engine_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine that records a corpus runs exactly the recipe's declared patches: the wave start renders
    them into ``RCP_NDCG_VLLM_PATCHES`` (overriding an inherited value), as the serve console does -- the
    behaviour fingerprint keys the patch module, so the process that records must run it."""
    import io

    # the engine start is called directly here, not through run_wave: give it an open wave of its own (the
    # per-wave state a start joins, and the live-engine registry its sweep walks) so its fake engine never
    # leaks -- a start from any wave that is not the current one is refused (the cross-wave guard)
    wave = run_wave_module._Wave(token="patches-test")
    monkeypatch.setattr(run_wave_module, "_CURRENT_WAVE", [wave])
    monkeypatch.setattr(run_wave_module, "_LIVE_ENGINES", set())  # its fake engine never reaches a later sweep

    from rcp_ndcg_vllm.patches import PATCHES_ENV

    slots_root = tmp_path / "slots"
    slots_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(slots_root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    recipe_yaml = recipes_root / "fixture-embed" / "family.yaml"
    recipe_yaml.write_text(
        recipe_yaml.read_text(encoding="utf-8").replace(
            "  plugin: null\n",
            "  plugin: rcp-ndcg-vllm\n  plugin_architectures: [PplxContextualModel]\n"
            "  patches: [pooling-full-context]\n",
        ),
        encoding="utf-8",
    )
    recipe = load_recipe(recipes_root / "fixture-embed")
    started: list[dict] = []

    class _Popen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            self.argv = argv
            self.stdout = io.BytesIO(b"")
            started.append(dict(kwargs))

        def poll(self) -> None:
            return None

    monkeypatch.setattr(run_wave_module.subprocess, "Popen", _Popen)
    monkeypatch.setenv(PATCHES_ENV, "some-other-patch")
    run = run_wave_module._start(recipe, [0], 0, tmp_path / "out", None, 0, wave=wave)
    assert run.env[PATCHES_ENV] == "pooling-full-context"
    assert started and started[0]["env"][PATCHES_ENV] == "pooling-full-context"  # type: ignore[index]
    # The fake engine went into the wave's live-engine registry; leave the process-global registry as
    # this test found it, or a later wave's final sweep stops the fake (whose Popen has no pid).
    run_wave_module._LIVE_ENGINES.discard(run)


def test_wave_recipe_cannot_start_fails_only_itself(tmp_path: Path) -> None:
    """An engine that cannot start fails that recipe only."""
    out = tmp_path / "wave"
    document = run_wave(["fixture-embed"], RECIPES, gpus=1, out_dir=out,
                        reference_python=REFERENCE_PYTHON, vllm_cmd="/nonexistent/binary", port_base=0)  # fmt: skip
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "cannot start the engine" in (row.get("error") or "")


def test_the_bare_exchange_is_public_and_the_wave_uses_it(tmp_path: Path) -> None:
    """One bare probe as a captured exchange is a public accessor of the recorder: the wave runner's readiness
    edge records through it, never through a private helper; a refused connection is recorded, not raised."""
    import httpx
    from rcp_ndcg_test import record as record_module

    assert "bare_exchange" in record_module.__all__
    with httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5) as http:
        exchange = record_module.bare_exchange(http, "GET", "/v1/models", None)
    assert exchange["status"] is None and exchange["url"].endswith("/v1/models")
    assert "_bare_exchange" not in Path(run_wave_module.__file__).read_text(encoding="utf-8")
