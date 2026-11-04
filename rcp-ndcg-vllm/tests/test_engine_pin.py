"""The engine pin: every shipped recipe and the plugin's guard name one vLLM release.

A recipe declares the engine image it was validated against and the guard refuses any engine outside the
tested range; a bump that moves one and not the other either serves a recipe on an engine the plugin
refuses, or registers the plugin on an engine no recipe was measured on.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_vllm.models import version_guard
from rcp_ndcg_vllm.recipe import (
    _PINNED_VLLM_REF,
    PINNED_POOLER_CONFIG_FIELDS,
    RecipeFamily,
    iter_families,
)

PINNED_IMAGE = "vllm/vllm-openai:v0.33.0"
PINNED_VERSION = "0.33.0"


@pytest.mark.parametrize("family", list(iter_families()), ids=lambda family: family.id)
def test_every_shipped_recipe_names_the_pinned_engine(family: RecipeFamily) -> None:
    """The image the recipe is served on and the version it declares are the pin (0.33)."""
    assert family.engine.image == PINNED_IMAGE, family.id
    assert family.engine.min_version == PINNED_VERSION, family.id


def test_the_guard_and_the_pooler_fields_name_the_pinned_engine() -> None:
    """The tested range, both spellings, the pooler field list's tag and the field that tag adds are 0.33."""
    assert version_guard.TESTED_VLLM_MIN == (0, 33)
    assert version_guard.TESTED_VLLM_MAX == (0, 34)
    assert version_guard.SUPPORTED_VLLM_MIN == version_guard.TESTED_VLLM_MIN
    assert version_guard.SUPPORTED_VLLM_MAX == version_guard.TESTED_VLLM_MAX
    assert _PINNED_VLLM_REF == "v0.33.0"
    assert "enable_flash_late_interaction" in PINNED_POOLER_CONFIG_FIELDS
