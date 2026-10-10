"""The per-family reference lock tool (owner decision 35): the input parser, the image-stack constraint,
the lock's identity header, the offline check and the committed family locks.

The resolver tests run ``uv pip compile`` against a hand-built local wheelhouse only (``--no-index``),
so the suite stays offline; the committed-lock guard re-reads every family's ``reference.in`` and
``reference.lock`` and checks the header hashes against the committed image stack.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.jobs import reference_lock as rl

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"
IMAGE_STACK = ROOT / "rcp-ndcg-vllm" / "reference-image-v0.31.0.txt"

FREEZE = """\
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
triton==3.7.1
nvidia-cublas==13.1.1.3
transformers==5.17.0
vllm==0.31.0
"""


def _tiny_wheel(directory: Path, name: str = "tiny", version: str = "1.0") -> Path:
    """A minimal valid pure-Python wheel in ``directory`` (metadata uv can read offline)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}/__init__.py", "")
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{name}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{name}-{version}.dist-info/RECORD", "")
    return path


# --- parsing ---------------------------------------------------------------------------------


def test_parse_reference_in_reads_directives_and_strips_comments() -> None:
    parsed = rl.parse_reference_in(
        "# a comment\n"
        "# own-torch: true\n"
        "# own-torch-evidence: the card pins torch (README.md:12)\n"
        "torch==2.9.1\n"
        "transformers>=4.51  # a trailing comment\n"
        "\n",
        family="demo",
    )
    assert parsed.own_torch is True
    assert parsed.evidence == "the card pins torch (README.md:12)"
    assert parsed.requirements == ("torch==2.9.1", "transformers>=4.51")


def test_own_torch_without_evidence_is_refused() -> None:
    with pytest.raises(HarnessError, match="own-torch-evidence"):
        rl.parse_reference_in("# own-torch: true\ntorch==2.9.1\n", family="demo")


def test_image_stack_keeps_only_the_torch_cuda_stack() -> None:
    stack = rl.image_stack(FREEZE)
    assert stack["torch"] == "2.13.0+cu130"
    assert stack["nvidia-cublas"] == "13.1.1.3"
    assert "transformers" not in stack and "vllm" not in stack


# --- generation ------------------------------------------------------------------------------


def _fake_compiler(requirements: list[str], **kwargs: object) -> str:
    return "".join(f"{rl.requirement_name(spec)}==9.9.9\n" for spec in requirements)


def test_build_lock_drops_the_image_floor_and_records_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    lock = rl.build_lock(
        "# own-torch: false\ntorch>=2.0\ntransformers>=4.51\n",
        FREEZE,
        family="demo",
        image="vllm/vllm-openai:v0.31.0",
        index_url=None,
    )
    assert "# own-torch: false" in lock
    assert "# image-constraint: torch==2.13.0+cu130 (family floor: torch>=2.0)" in lock
    assert "transformers==9.9.9" in lock
    assert "\ntorch==" not in lock  # the image's torch is never installed


def test_a_stack_pin_without_own_torch_is_refused() -> None:
    with pytest.raises(HarnessError, match="own-torch"):
        rl.build_lock("nvidia-not-in-the-image==1.0\n", FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0")


def test_a_family_floor_the_image_cannot_satisfy_is_refused() -> None:
    with pytest.raises(HarnessError, match="does not satisfy the family floor"):
        rl.build_lock("torch==2.9.1\n", FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0")


def test_own_torch_keeps_the_exact_stack_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    lock = rl.build_lock(
        "# own-torch: true\n# own-torch-evidence: flash-attn 2.8.3 has a wheel only for torch 2.9.1\n"
        "torch==2.9.1\nflash-attn==2.8.3\n",
        FREEZE,
        family="demo",
        image="vllm/vllm-openai:v0.31.0",
        index_url=None,
    )
    assert "# own-torch: true" in lock
    assert "torch==9.9.9" in lock  # the compiled pin; the resolver saw torch==2.9.1
    assert "flash-attn==9.9.9" in lock
    assert "image-constraint" not in lock


def test_own_torch_needs_an_exact_stack_pin() -> None:
    with pytest.raises(HarnessError, match="must be exact"):
        rl.build_lock(
            "# own-torch: true\n# own-torch-evidence: because\ntorch>=2.9\n",
            FREEZE,
            family="demo",
            image="vllm/vllm-openai:v0.31.0",
        )


def test_a_workspace_range_is_refused() -> None:
    with pytest.raises(HarnessError, match="workspace pin"):
        rl.build_lock("rcp-ndcg>=0.0.1\n", FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0")


def test_compile_requirements_resolves_offline_from_a_wheelhouse(tmp_path: Path) -> None:
    """The resolver runs uv against a local wheelhouse only: an exact pin, hashed, no index."""
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - the repository runs everything through uv
        pytest.skip("uv is not on PATH")
    wheelhouse = tmp_path / "wheelhouse"
    _tiny_wheel(wheelhouse)
    text = rl.compile_requirements(["tiny==1.0"], find_links=(str(wheelhouse),), index_url=None, uv=uv)
    assert "tiny==1.0" in text
    assert not any(line.startswith("#") for line in text.splitlines())


def test_compile_requirements_reports_uvs_failure() -> None:
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover
        pytest.skip("uv is not on PATH")
    with pytest.raises(HarnessError, match="uv pip compile failed"):
        rl.compile_requirements(["definitely-not-a-package==1.0"], find_links=(), index_url=None, uv=uv)


# --- the offline check -----------------------------------------------------------------------


def test_check_lock_accepts_a_valid_lock_and_names_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    reference_in = "torch>=2.0\ntransformers>=4.51\n"
    lock = rl.build_lock(reference_in, FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0", index_url=None)
    assert rl.check_lock(lock, reference_in, FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0") == []
    problems = rl.check_lock(lock, reference_in + "numpy\n", FREEZE, family="demo")
    assert any("reference-in-sha256" in problem for problem in problems)
    assert any("numpy" in problem for problem in problems)
    problems = rl.check_lock(lock, reference_in, FREEZE.replace("2.13.0", "2.14.0"), family="demo")
    assert any("image-freeze-sha256" in problem for problem in problems)


def test_check_lock_refuses_a_hand_edited_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    reference_in = "transformers>=4.51\n"
    lock = rl.build_lock(reference_in, FREEZE, family="demo", image="vllm/vllm-openai:v0.31.0", index_url=None)
    tampered = lock.replace("transformers==9.9.9", "transformers==1.0")
    problems = rl.check_lock(tampered, reference_in, FREEZE, family="demo")
    assert any("does not satisfy" in problem for problem in problems), problems


# --- the committed family locks --------------------------------------------------------------


def _reference_families() -> list[Path]:
    """The families with a reference (a judge recipe has none, decision 15): they ship the lock."""
    from rcp_ndcg_vllm.recipe import load_family, load_recipes_of

    families = []
    for path in sorted(RECIPES.glob("*/family.yaml")):
        recipes = load_recipes_of(load_family(path.parent), path.parent)
        if any(recipe.reference is not None for recipe in recipes):
            families.append(path.parent)
    return families


def test_every_family_ships_a_reference_in_and_lock() -> None:
    families = _reference_families()
    assert families, "no recipe families found"
    for family in families:
        assert (family / "reference.in").is_file(), family
        assert (family / "reference.lock").is_file(), family
        assert not (family / "requirements-reference.txt").exists(), f"{family}: the old file must be migrated"


def test_every_committed_lock_matches_its_inputs() -> None:
    """The CPU guard: every committed lock's header hashes its committed reference.in and the committed
    image stack, every family requirement is pinned or image-constrained, and the own-torch declaration
    matches.  Regeneration is deliberate; a hand edit fails here."""
    stack_text = IMAGE_STACK.read_text(encoding="utf-8")
    for family in _reference_families():
        reference_in = (family / "reference.in").read_text(encoding="utf-8")
        lock = (family / "reference.lock").read_text(encoding="utf-8")
        problems = rl.check_lock(lock, reference_in, stack_text, family=family.name)
        assert problems == [], f"{family.name}: " + "; ".join(problems)
        # The lock is hashed: every pin resolved from the index carries its wheel hashes; only the
        # project's own workspace pin (the staged wheelhouse's built wheel) stays verbatim.
        for line in lock.splitlines():
            if "==" not in line or line.startswith("#") or line.strip().startswith("--hash"):
                continue
            name = line.split("==", 1)[0].strip()
            if name in rl.WORKSPACE_NAMES:
                continue
            assert line.rstrip().endswith("\\"), f"{family.name}: {line!r} carries no hash continuation"


def test_every_lock_names_its_familys_engine_image() -> None:
    """A lock's ``# image:`` header is the family's resolved ``engine.image`` (so the reference venv is
    built over the same image the job runs), including a digest-pinned nightly."""
    from rcp_ndcg_vllm.recipe import load_family, load_recipes_of

    for family_dir in _reference_families():
        lock = (family_dir / "reference.lock").read_text(encoding="utf-8")
        image = rl._lock_header(lock).get("image")
        recipes = load_recipes_of(load_family(family_dir), family_dir)
        assert image == recipes[0].engine.image, family_dir.name


def test_a_lock_on_another_image_records_an_uncommitted_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """A digest-pinned nightly whose stack is not committed: the lock names the freeze's source image and
    records the hash as uncommitted; the offline check accepts it against that source stack."""
    monkeypatch.setattr(rl, "compile_requirements", _fake_compiler)
    reference_in = "torch>=2.0\n"
    lock = rl.build_lock(
        reference_in,
        FREEZE,
        family="demo",
        image="registry.example.com/nightly@sha256:" + "a" * 64,
        index_url=None,
        image_freeze_source="vllm/vllm-openai:v0.31.0",
    )
    assert "# image-freeze-sha256: uncommitted" in lock
    assert "# image-freeze-source: vllm/vllm-openai:v0.31.0" in lock
    assert rl.check_lock(lock, reference_in, FREEZE, family="demo") == []


def test_the_tool_cli_builds_and_checks_a_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI end to end over a local wheelhouse: build writes the lock, check passes, a mutation fails."""
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover
        pytest.skip("uv is not on PATH")
    wheelhouse = tmp_path / "wheelhouse"
    _tiny_wheel(wheelhouse)
    reference_in = tmp_path / "reference.in"
    reference_in.write_text("tiny==1.0\ntorch>=2.0\n", encoding="utf-8")
    freeze = tmp_path / "freeze.txt"
    freeze.write_text(FREEZE, encoding="utf-8")
    lock = tmp_path / "reference.lock"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "rcp_ndcg_test.jobs.reference_lock",
            "build",
            "--family",
            "demo",
            "--in",
            str(reference_in),
            "--out",
            str(lock),
            "--image-freeze",
            str(freeze),
            "--image",
            "registry.example.com/other:1",
            "--image-freeze-source",
            "vllm/vllm-openai:v0.31.0",
            "--find-links",
            str(wheelhouse),
            "--uv",
            uv,
        ],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    assert "# image-freeze-sha256: uncommitted" in lock.read_text(encoding="utf-8")
    assert "# image-freeze-source: vllm/vllm-openai:v0.31.0" in lock.read_text(encoding="utf-8")
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "rcp_ndcg_test.jobs.reference_lock",
            "check",
            "--family",
            "demo",
            "--in",
            str(reference_in),
            "--lock",
            str(lock),
            "--image-freeze",
            str(freeze),
        ],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    reference_in.write_text("tiny==1.0\ntorch>=2.0\nnumpy\n", encoding="utf-8")
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "rcp_ndcg_test.jobs.reference_lock",
            "check",
            "--family",
            "demo",
            "--in",
            str(reference_in),
            "--lock",
            str(lock),
            "--image-freeze",
            str(freeze),
        ],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 1
    assert "numpy" in checked.stdout
