"""The engine-image grouping (owner decisions 38 and 35): one GPU job per container image.

The real-catalog test is the brief's "two images in one wave list": the committed all-retrieval list
carries the released image's variants and ``embeddinggemma-2``'s digest-pinned nightly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.jobs import wavegroups as wg

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "rcp-ndcg-test" / "tests" / "fixtures" / "recipes"
WAVE_LIST = ROOT / "rcp-ndcg-test" / "wave-lists" / "all-retrieval.txt"


def test_slug_of_is_stable_and_digest_specific() -> None:
    assert wg.slug_of("vllm/vllm-openai:v0.31.0") == "vllm-vllm-openai-v0-31-0"
    first = wg.slug_of("vllm/vllm-openai:nightly-abc@sha256:" + "a" * 64)
    second = wg.slug_of("vllm/vllm-openai:nightly-abc@sha256:" + "b" * 64)
    assert first != second and first.endswith("-" + "a" * 12)


def test_group_ids_keeps_the_wave_order_and_first_appearance(tmp_path: Path) -> None:
    """Two families on two images: one group per image, ids in wave order; an unknown id raises."""
    for family, image in (("family-a", "registry.example.com/a:1"), ("family-b", "registry.example.com/b:2")):
        shutil.copytree(FIXTURES / "fixture-embed", tmp_path / family)
        yaml = tmp_path / family / "family.yaml"
        text = yaml.read_text(encoding="utf-8")
        text = text.replace("id: fixture-embed", f"id: {family}")
        text = text.replace('image: "vllm/vllm-openai:v0.31.0"', f'image: "{image}"')
        yaml.write_text(text, encoding="utf-8")
    groups = wg.group_ids(["family-a", "family-b", "family-a"], tmp_path)
    assert groups == {"registry.example.com/a:1": ["family-a", "family-a"], "registry.example.com/b:2": ["family-b"]}
    with pytest.raises(HarnessError, match="does not load"):
        wg.group_ids(["no-such-recipe"], tmp_path)


def test_write_groups_writes_one_list_per_image(tmp_path: Path) -> None:
    groups = {"registry.example.com/b:2": ["b"], "registry.example.com/a:1": ["a"]}
    mapping = wg.write_groups(tmp_path / "wave-lists", "all", groups)
    assert sorted(mapping) == ["registry.example.com/a:1", "registry.example.com/b:2"]
    for image, name in mapping.items():
        assert (tmp_path / "wave-lists" / name).read_text(encoding="utf-8") == groups[image][0] + "\n"


def test_the_real_catalog_has_exactly_two_engine_images() -> None:
    """The brief's two-image wave: the released image plus embeddinggemma-2's digest-pinned nightly."""
    ids = [
        line.strip()
        for line in WAVE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    groups = wg.group_ids(ids)
    assert len(groups) == 2, groups
    nightly = next(image for image in groups if "@sha256:" in image)
    assert groups[nightly] == ["embeddinggemma-2"]
    released = next(image for image in groups if "@sha256:" not in image)
    assert "embeddinggemma-2" not in groups[released]
    assert sum(len(group) for group in groups.values()) == len(ids)


def test_the_cli_writes_the_lists_and_the_stage_map(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "wave-lists").mkdir(parents=True)
    ids = [
        line.strip()
        for line in WAVE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    list_file = tmp_path / "all-retrieval.txt"
    list_file.write_text("".join(f"{recipe_id}\n" for recipe_id in ids), encoding="utf-8")
    assert (
        wg.main(
            [
                "--wave",
                "all-retrieval",
                "--recipes",
                f"@{list_file}",
                "--out",
                str(stage / "wave-lists"),
                "--map-out",
                str(stage),
            ]
        )
        == 0
    )
    document = json.loads((stage / wg.GROUP_MAP).read_text(encoding="utf-8"))
    assert set(document["all-retrieval"]) == set(wg.group_ids(ids))
    for name in document["all-retrieval"].values():
        assert (stage / "wave-lists" / name).is_file()
