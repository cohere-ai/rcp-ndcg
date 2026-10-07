"""The HF weight helpers: measure, decide, evict (node-runtime item 8) — offline, on a fake cache."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from rcp_ndcg_vllm.jobs import weights


@pytest.fixture()
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An HF cache root in tmp_path, selected the way the engines select it ($HF_HUB_CACHE)."""
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


def _model(cache_root: Path, model: str, *, megabytes: int) -> Path:
    """A fake downloaded model: blobs, snapshots and refs, the layout huggingface_hub writes."""
    directory = cache_root / f"models--{model.replace('/', '--')}"
    (directory / "refs").mkdir(parents=True)
    (directory / "blobs").mkdir()
    snapshots = directory / "snapshots" / "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    snapshots.mkdir(parents=True)
    (snapshots / "model.safetensors").write_bytes(b"0" * (megabytes * 1 << 20))
    return directory


def test_hf_cache_root_honours_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """$HF_HUB_CACHE wins over $HF_HOME, and neither gives the home default."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert weights.hf_cache_root() == tmp_path / "hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert weights.hf_cache_root() == tmp_path / "home" / "hub"
    monkeypatch.delenv("HF_HOME")
    assert weights.hf_cache_root() == Path.home() / ".cache" / "huggingface" / "hub"


def test_snapshot_bytes_measures_a_cached_model(cache_root: Path) -> None:
    """The cache's own bytes: 4 MiB here, 0 for a model that was never served."""
    _model(cache_root, "org/model", megabytes=4)
    assert weights.snapshot_bytes("org/model") >= 4 << 20
    assert weights.snapshot_bytes("org/absent") == 0


def test_evict_removes_only_the_model_directory(cache_root: Path) -> None:
    """One eviction frees the model's bytes and touches nothing else in the cache root."""
    _model(cache_root, "org/model", megabytes=8)
    keep = cache_root / "models--org--other"
    keep.mkdir()
    eviction = weights.evict("org/model")
    assert eviction.removed
    assert eviction.freed_bytes >= 8 << 20
    assert eviction.error is None
    assert not (cache_root / "models--org--model").exists()
    assert keep.is_dir()


def test_evict_of_an_absent_model_is_a_recorded_noop(cache_root: Path) -> None:
    """A model that was never served: removed=False, no error, the wave goes on."""
    eviction = weights.evict("org/absent")
    assert eviction.removed is False
    assert eviction.freed_bytes == 0
    assert eviction.error is None


def test_evict_frees_readonly_blobs(cache_root: Path) -> None:
    """HF marks blobs read-only; an eviction still removes them."""
    directory = _model(cache_root, "org/model", megabytes=1)
    os.chmod(directory / "snapshots" / "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" / "model.safetensors", 0o444)
    assert weights.evict("org/model").removed
    assert not directory.exists()


def test_disk_free_bytes_measures_a_not_yet_created_cache_at_its_parent(tmp_path: Path) -> None:
    """A fresh pod's HF cache does not exist before the first download: the measurement lands on the
    nearest existing parent (the same filesystem), never a FileNotFoundError crash."""
    missing = tmp_path / "deep" / "not" / "created" / "hub"
    assert not missing.exists()
    assert weights.disk_free_bytes(missing) == weights.disk_free_bytes(tmp_path)


def test_will_fit_refuses_a_model_that_measurably_does_not_fit() -> None:
    """The one-line early failure: the model's size plus headroom against the free disk."""
    free = 2 << 30
    ok, reason = weights.will_fit(free, (4 << 30))
    assert not ok
    assert "GiB" in reason and "free" in reason
    ok, reason = weights.will_fit(free, (1 << 30))
    assert ok and reason == ""
    ok, reason = weights.will_fit(free, None)
    assert ok and reason == ""  # unknown size: the caller records it and decides
