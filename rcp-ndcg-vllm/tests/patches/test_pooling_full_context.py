"""CPU tests for the opt-in pooling-full-context scheduler patch; a fake Scheduler stands in for vLLM's.

The patch imports vLLM lazily (the package imports clean without it), so these tests run in the lean
package venv: they hand the apply seam a fresh fake class shaped like
``vllm.v1.core.sched.scheduler.Scheduler`` -- the config it reads, the one attribute the wrapper writes --
and pin the opt-in, the logging and the two runners' behaviour.
"""

from __future__ import annotations

import logging
import types

import pytest
from rcp_ndcg_vllm.patches import (
    PATCHES_ENV,
    apply_opted_in_patches,
    opted_in_patch_names,
    pooling_full_context,
)

LOGGER = "rcp_ndcg_vllm.patches.pooling_full_context"


def _config(runner_type: str) -> types.SimpleNamespace:
    """The ``vllm_config`` shape the wrapper reads: ``model_config.runner_type``."""
    return types.SimpleNamespace(model_config=types.SimpleNamespace(runner_type=runner_type))


def _scheduler_class(initial: int = 1) -> type:
    """A fresh fake Scheduler class per test: vLLM v0.31.0's ``__init__`` reserves one sampled slot.

    ``calls`` counts the original constructor's invocations, so a double wrap (the idempotence bug) is
    visible as ``calls == 2`` for one instantiation.
    """

    class Scheduler:
        calls = 0

        def __init__(self, vllm_config: object, **kwargs: object) -> None:
            type(self).calls += 1
            self.vllm_config = vllm_config
            self.num_sampled_tokens_per_step = initial

    return Scheduler


@pytest.fixture(autouse=True)
def _no_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PATCHES_ENV, raising=False)


def test_opted_in_names_parse_the_comma_separated_variable() -> None:
    assert opted_in_patch_names({PATCHES_ENV: " pooling-full-context , other ,pooling-full-context "}) == (
        "pooling-full-context",
        "other",
    )
    assert opted_in_patch_names({}) == ()
    assert opted_in_patch_names({PATCHES_ENV: ""}) == ()
    assert opted_in_patch_names({PATCHES_ENV: " , "}) == ()


def test_patch_applies_to_a_pooling_scheduler(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")
    scheduler_cls = _scheduler_class()
    assert apply_opted_in_patches(scheduler_cls=scheduler_cls) == ("pooling-full-context",)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        scheduler = scheduler_cls(_config("pooling"))

    assert scheduler.num_sampled_tokens_per_step == 0
    assert [r.getMessage() for r in caplog.records if "applied" in r.getMessage()] != [], (
        "the patch logs one line when it applies"
    )
    assert len(caplog.records) == 1


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")
    scheduler_cls = _scheduler_class()
    assert pooling_full_context.apply(scheduler_cls=scheduler_cls) is True
    assert pooling_full_context.apply(scheduler_cls=scheduler_cls) is False, "the second apply is a no-op"

    # The registry path is idempotent too.
    assert apply_opted_in_patches(scheduler_cls=scheduler_cls) == ("pooling-full-context",)
    scheduler = scheduler_cls(_config("pooling"))
    assert scheduler_cls.calls == 1, "a second apply must not nest a second wrapper around the original"
    assert scheduler.num_sampled_tokens_per_step == 0


def test_patch_is_inert_when_the_engine_already_sets_zero(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A vLLM release that carries vllm-project/vllm#48039 already stores 0; the patch only says so."""
    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")
    scheduler_cls = _scheduler_class(initial=0)
    apply_opted_in_patches(scheduler_cls=scheduler_cls)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        scheduler = scheduler_cls(_config("pooling"))

    assert scheduler.num_sampled_tokens_per_step == 0
    messages = [r.getMessage() for r in caplog.records]
    assert any("inert" in message for message in messages), "the already-fixed release gets its one inert line"
    assert not any("applied" in message for message in messages)


def test_patch_never_touches_a_generate_runner(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")
    scheduler_cls = _scheduler_class()
    apply_opted_in_patches(scheduler_cls=scheduler_cls)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        scheduler = scheduler_cls(_config("generate"))

    assert scheduler.num_sampled_tokens_per_step == 1
    assert caplog.records == [], "a generate runner is not the patch's business"


def test_patch_is_not_applied_without_the_opt_in(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler_cls = _scheduler_class()
    assert apply_opted_in_patches(scheduler_cls=scheduler_cls) == ()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        scheduler = scheduler_cls(_config("pooling"))

    assert scheduler.num_sampled_tokens_per_step == 1
    assert caplog.records == []


def test_an_unknown_opted_in_name_warns_and_applies_nothing(caplog: pytest.LogCaptureFixture) -> None:
    scheduler_cls = _scheduler_class()
    with caplog.at_level(logging.WARNING, logger="rcp_ndcg_vllm.patches"):
        applied = apply_opted_in_patches(environ={PATCHES_ENV: "not-a-patch"}, scheduler_cls=scheduler_cls)

    assert applied == ()
    assert any("not-a-patch" in r.getMessage() for r in caplog.records)
    scheduler = scheduler_cls(_config("pooling"))
    assert scheduler.num_sampled_tokens_per_step == 1


def test_the_entry_point_seam_applies_the_opted_in_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rcp_ndcg_vllm.models.register`` -- the one ``vllm.general_plugins`` callable -- applies the opt-in.

    The registry modules are stubbed (they import vLLM themselves) and a fake ``vllm`` package is placed in
    ``sys.modules`` so the seam's lazy ``from vllm.v1.core.sched.scheduler import Scheduler`` resolves here.
    """
    import sys

    from rcp_ndcg_vllm import models

    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")
    monkeypatch.setattr("rcp_ndcg_vllm.models.version_guard.require_vllm_version", lambda: (0, 31))
    monkeypatch.setattr("rcp_ndcg_vllm.models.topk.plugin.register_topk", lambda: None)
    monkeypatch.setattr("rcp_ndcg_vllm.models.pplx.register_pplx", lambda: None)

    scheduler_cls = _scheduler_class()
    fake = {
        name: types.ModuleType(name)
        for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched", "vllm.v1.core.sched.scheduler")
    }
    fake["vllm.v1.core.sched.scheduler"].Scheduler = scheduler_cls
    fake["vllm.v1.core.sched"].scheduler = fake["vllm.v1.core.sched.scheduler"]
    fake["vllm.v1.core"].sched = fake["vllm.v1.core.sched"]
    fake["vllm.v1"].core = fake["vllm.v1.core"]
    fake["vllm"].v1 = fake["vllm.v1"]
    for name, module in fake.items():
        monkeypatch.setitem(sys.modules, name, module)

    models.register()

    assert scheduler_cls(_config("pooling")).num_sampled_tokens_per_step == 0
