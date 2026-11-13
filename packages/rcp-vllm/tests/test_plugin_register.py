"""``register()``: a folded plugin's registration failure is loud and names the package.

The entry point is one ``vllm.general_plugins`` callable for both folded distributions (layout-move item
3), and vLLM's loader catches whatever it raises and only logs the traceback
(``rcp_vllm.models.version_guard``'s docstring documents the swallowing). A failure in the second
package therefore used to leave the first registered and the second not, with nothing but a traceback in
the engine's log saying which half was missing. ``register()`` prints the refusal with the package's name
and re-raises.
"""

from __future__ import annotations

from typing import Any

import pytest
import rcp_vllm.models as models
import rcp_vllm.models.pplx as pplx
import rcp_vllm.models.topk.plugin as topk_plugin
import rcp_vllm.models.version_guard as version_guard
import rcp_vllm.patches as patches


@pytest.fixture(autouse=True)
def _engine_less(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard and the patch applier answered without a vLLM, so the test drives the registration only."""

    def guard() -> tuple[int, int]:
        return (0, 33)

    monkeypatch.setattr(version_guard, "require_vllm_version", guard)
    monkeypatch.setattr(patches, "apply_opted_in_patches", lambda: None)


def test_a_broken_folded_plugin_fails_loudly_naming_its_package(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second package's registration raising must reach the caller with the package named on stderr:
    the first package is already registered by then, so a silent (or traceback-only) failure is an engine
    that starts half-registered and dies later on an opaque architecture error."""
    registered: list[str] = []

    def good() -> None:
        registered.append("topk")

    def broken() -> None:
        raise ImportError("cannot import name 'MODELS_CONFIG_MAP' from vllm.model_executor.models.config")

    monkeypatch.setattr(topk_plugin, "register_topk", good)
    monkeypatch.setattr(pplx, "register_pplx", broken)

    with pytest.raises(ImportError, match="MODELS_CONFIG_MAP"):
        models.register()

    assert registered == ["topk"], "the first package registered before the second failed"
    assert "rcp_vllm.models.pplx" in capsys.readouterr().err


def test_the_first_folded_plugin_failing_is_named_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> None:
        raise RuntimeError("the pinned image's registry has no TopkEmbedModel hook")

    monkeypatch.setattr(topk_plugin, "register_topk", broken)

    def unreachable() -> Any:
        raise AssertionError("the second package must not register after the first failed")

    monkeypatch.setattr(pplx, "register_pplx", unreachable)

    with pytest.raises(RuntimeError, match="TopkEmbedModel"):
        models.register()

    assert "rcp_vllm.models.topk" in capsys.readouterr().err
