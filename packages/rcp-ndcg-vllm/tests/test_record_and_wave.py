"""The recorder and the wave runner: the product's wire path observed through the recording transport."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence.metrics import stage3_metrics
from rcp_ndcg_vllm.jobs import run_wave as run_wave_module
from rcp_ndcg_vllm.jobs import weights
from rcp_ndcg_vllm.jobs.run_wave import _ZMQ_IPC_SUFFIX_CHARS, _slot_tmp_dir, run_wave
from rcp_ndcg_vllm.record import record

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub

REFERENCE_PYTHON = sys.executable
VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"
REFERENCE_PY = sys.executable


@pytest.fixture(autouse=True)
def _hub_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in tests: every wave's Hub question (the model's bytes) answers "unknown" here;
    the tests that need a size monkeypatch their own value on top of this."""
    monkeypatch.setattr(weights, "model_disk_bytes", lambda model, revision=None: None)


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


def test_stage3_metrics_compares_served_against_reference(tmp_path: Path) -> None:
    """Stage 3 shells out to `rcp-ndcg eval score`; identical rankings give delta 0 and a pass."""
    pytest.importorskip("rcp_ndcg")
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates

    from rcp_ndcg.data import Rankings

    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    for system in ("served", "reference"):
        Rankings.from_orders({"q1": ["a", "b", "c"], "q2": ["b", "a", "c"]}, system=system).save(
            rankings_dir / f"toy.{system}.jsonl"
        )
    dataset = [
        {
            "query_id": "q1",
            "query": "q1",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 1.0, "b": 0.5, "c": 0.1},
        },  # fmt: skip
        {
            "query_id": "q2",
            "query": "q2",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 0.2, "b": 0.9, "c": 0.0},
        },  # fmt: skip
    ]
    (rankings_dir / "toy.dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8"
    )
    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    document = stage3_metrics(rankings_dir, gates)
    assert document["passed"] is True


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
    recipe_yaml = recipes_root / "fixture-embed-cls" / "recipe.yaml"
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


def test_wave_marks_an_invalid_recipe_failed_with_the_validation_message(tmp_path: Path) -> None:
    """One failing recipe never stops the wave, end to end: a recipe that fails validation is a failed
    row in the wave report (with the validation message), and the wave runs the rest."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    broken = recipes_root / "broken-recipe"
    broken.mkdir()
    broken_text = (RECIPES / "fixture-embed" / "recipe.yaml").read_text(encoding="utf-8")
    (broken / "recipe.yaml").write_text(
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
    recipe_yaml = recipes_root / "fixture-embed" / "recipe.yaml"
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
    staged wheel file (collected as <recipe-id>/<file>) installed fine is never failed because another
    recipe's bare <file> spec failed (the exact name still appears in the failing recipe's row)."""
    recipes_root = tmp_path / "recipes"
    shutil.copytree(RECIPES, recipes_root)
    shutil.copy2(RECIPES.parent / "tokenizer.json", recipes_root.parent / "tokenizer.json")
    shutil.copy2(RECIPES.parent / "deterministic.py", recipes_root.parent / "deterministic.py")
    spec = "plugin_wheel-1.0.0-py3-none-any.whl"
    (recipes_root / "fixture-embed" / spec).write_bytes(b"stub wheel bytes")  # staged in A's dir only
    for recipe_id in ("fixture-embed", "fixture-embed-cls"):
        recipe_yaml = recipes_root / recipe_id / "recipe.yaml"
        recipe_yaml.write_text(
            recipe_yaml.read_text(encoding="utf-8").replace("  plugin: null\n", f"  plugin: {spec}\n"),
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


def test_wave_recipe_cannot_start_fails_only_itself(tmp_path: Path) -> None:
    """An engine that cannot start fails that recipe only."""
    out = tmp_path / "wave"
    document = run_wave(["fixture-embed"], RECIPES, gpus=1, out_dir=out,
                        reference_python=REFERENCE_PYTHON, vllm_cmd="/nonexistent/binary", port_base=0)  # fmt: skip
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "cannot start the engine" in (row.get("error") or "")
