"""The judge recipes' sampling: each sends its checkpoint's own generation defaults.

Owner decision 2026-10-10 (superseding decision 4.3's paper sampling): every judge recipe reads its
checkpoint's ``generation_config.json`` at the pinned revision and sends exactly those sampling
parameters -- ``temperature``/``top_p``/``top_k`` as the config declares; a config without sampling
parameters is greedy, so the recipe sends ``temperature: 0.0`` and no ``top_p``/``top_k`` -- keeping
the recipe's own ``max_output_tokens``. The table below is the fetched evidence (the Hub's raw
``generation_config.json`` at each pin, fetched 2026-10-10); this test pins the declarations to it, so
a checkpoint's sampling cannot drift silently and a new judge variant must declare its values. The
sampling fields are CONTENT fields of the judgement family, so the decision re-keys every judge family
(the owner accepted the re-key).

OFFLINE: the recipe data is package data and the expected values are the fetched bytes restated here;
no network, no torch.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_vllm.recipe import iter_recipes

#: Every shipped judge variant's sampling, from its checkpoint's own generation_config.json at the
#: pinned revision (fetched 2026-10-10). ``top_p``/``top_k`` are None where the config declares no
#: sampling parameters: the recipe then sends temperature 0.0 (greedy) and no top_p/top_k.
JUDGE_SAMPLING: dict[str, dict[str, object]] = {
    "gpt-oss-120b": {
        "model": "openai/gpt-oss-120b",
        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "temperature": 0.0,  # the config declares no sampling parameters: greedy
        "top_p": None,
        "top_k": None,
        "max_output_tokens": 8192,
    },
    "gemma-4-12b-it": {
        "model": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,
    },
    "gemma-4-26b-a4b-it": {
        "model": "google/gemma-4-26B-A4B-it",
        "revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,
    },
    "gemma-4-26b-a4b-nvfp4": {
        "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "revision": "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,
    },
    "gemma-4-31b-it-nvfp4": {
        "model": "nvidia/Gemma-4-31B-IT-NVFP4",
        "revision": "4135a98a9b728a548947683219633b25682223ac",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,
    },
    "qwen3.5-397b-a17b-nvfp4": {
        "model": "nvidia/Qwen3.5-397B-A17B-NVFP4",
        "revision": "0368c1b3233414cd4a617b8ff9515e25752dc16c",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 16384,
    },
    "qwen3.6-27b-fp8": {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "revision": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 16384,
    },
    "qwen3.8-27b-fp8": {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 16384,
    },
    "qwen3.8-flash-next-nvfp4": {
        "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 16384,
    },
    "qwen3.8-flash-next-fp8": {
        "model": "Qwen/Qwen3.8-Flash-Next-FP8",
        "revision": "236dfdf285828023ca3bcd3f37366c58a3469b13",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 16384,
    },
}

JUDGE_VARIANTS = [recipe.id for recipe in iter_recipes() if recipe.role == "judge"]


def test_the_sampling_table_covers_every_judge_variant() -> None:
    """A new judge variant must declare its sampling here, from its own generation_config.json."""
    assert sorted(JUDGE_VARIANTS) == sorted(JUDGE_SAMPLING), (
        "the judge sampling table is out of step with the shipped judge recipes: "
        f"missing {sorted(set(JUDGE_VARIANTS) - set(JUDGE_SAMPLING))}, "
        f"extra {sorted(set(JUDGE_SAMPLING) - set(JUDGE_VARIANTS))}"
    )


@pytest.mark.parametrize("variant_id", sorted(JUDGE_SAMPLING))
def test_the_recipe_sends_its_checkpoints_generation_defaults(variant_id: str) -> None:
    """temperature/top_p/top_k equal the checkpoint's generation_config.json; max_output_tokens stays."""
    expected = JUDGE_SAMPLING[variant_id]
    recipe = next(item for item in iter_recipes() if item.id == variant_id)
    assert recipe.role == "judge"
    assert recipe.client["temperature"] == expected["temperature"], (
        f"{variant_id}: the recipe must send the checkpoint's own temperature "
        f"({expected['model']}@{expected['revision']} declares {expected['temperature']!r})"
    )
    extra_body = dict(recipe.client.get("extra_body") or {})
    top_p, top_k = expected["top_p"], expected["top_k"]
    if top_p is None and top_k is None:
        assert "top_p" not in extra_body and "top_k" not in extra_body, (
            f"{variant_id}: the checkpoint's generation_config.json declares no sampling parameters, so the "
            f"recipe must send no top_p/top_k; it declares {extra_body!r}"
        )
    else:
        assert extra_body.get("top_p") == top_p and extra_body.get("top_k") == top_k, (
            f"{variant_id}: extra_body must carry the checkpoint's own top_p/top_k "
            f"({expected['model']}@{expected['revision']} declares top_p {top_p!r}, top_k {top_k!r}); "
            f"it declares {extra_body!r}"
        )
    assert recipe.client["max_output_tokens"] == expected["max_output_tokens"], (
        f"{variant_id}: max_output_tokens is not part of the sampling decision and must stay "
        f"{expected['max_output_tokens']}"
    )
