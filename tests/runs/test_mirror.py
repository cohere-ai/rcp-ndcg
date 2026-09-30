"""The run mirror: append-only JSONL in immutable, contiguous parts; everything else whole; restore from it."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import pytest

from rcp_ndcg import storage
from rcp_ndcg.errors import DataError
from rcp_ndcg.runs.mirror import Mirror, read_state, restore

REMOTE = "memory://mirror/run"


@pytest.fixture(autouse=True)
def _empty_memory_fs():
    memory = storage.filesystem("memory://")
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]
    yield
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]


def _parts(relative: str) -> list[str]:
    return sorted(uri.rpartition("/")[2] for uri in storage.ls(f"{REMOTE}/{relative}.parts"))


def test_appended_lines_go_up_in_immutable_contiguous_parts(tmp_path: Path) -> None:
    store = tmp_path / "judgements" / "rubric.jsonl"
    store.parent.mkdir()
    store.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    mirror = Mirror(tmp_path, REMOTE)

    mirror.flush()
    first = storage.read_bytes(f"{REMOTE}/judgements/rubric.jsonl.parts/000000000000-000000000018")
    mirror.flush()  # nothing new: no new part
    with store.open("ab") as handle:
        handle.write(b'{"a": 3}\n')
    mirror.flush()

    assert _parts("judgements/rubric.jsonl") == ["000000000000-000000000018", "000000000018-000000000027"]
    assert storage.read_bytes(f"{REMOTE}/judgements/rubric.jsonl.parts/000000000000-000000000018") == first
    assert read_state(tmp_path / ".mirror.json").last_upload_at is not None


def test_a_torn_last_line_waits_until_it_is_complete(tmp_path: Path) -> None:
    store = tmp_path / "tournament.jsonl"
    store.write_bytes(b'{"a": 1}\n{"a": ')
    mirror = Mirror(tmp_path, REMOTE)

    mirror.flush()
    assert _parts("tournament.jsonl") == ["000000000000-000000000009"]
    with store.open("ab") as handle:
        handle.write(b"2}\n")
    mirror.flush()
    assert _parts("tournament.jsonl") == ["000000000000-000000000009", "000000000009-000000000018"]


def test_other_files_go_up_whole_and_a_deleted_directory_comes_back(tmp_path: Path) -> None:
    local = tmp_path / "run"
    (local / "judgements").mkdir(parents=True)
    (local / "judgements" / "rubric.jsonl").write_bytes(b'{"a": 1}\n{"a": ')  # the torn tail stays local
    (local / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    (local / "work").mkdir()
    (local / "work" / "index.bin").write_bytes(b"scratch")
    mirror = Mirror(local, REMOTE)
    mirror.flush()
    (local / "manifest.json").write_text('{"v": 22}', encoding="utf-8")
    mirror.flush()
    assert storage.read_text(f"{REMOTE}/manifest.json") == '{"v": 22}'
    assert not storage.exists(f"{REMOTE}/work/index.bin")

    fresh = tmp_path / "elsewhere"
    assert sorted(restore(fresh, REMOTE)) == ["judgements/rubric.jsonl", "manifest.json"]
    assert (fresh / "judgements" / "rubric.jsonl").read_bytes() == b'{"a": 1}\n'
    assert (fresh / "manifest.json").read_text(encoding="utf-8") == '{"v": 22}'
    assert restore(fresh, REMOTE) == []  # nothing is missing any more


def test_a_gap_between_parts_is_refused(tmp_path: Path) -> None:
    storage.write_bytes(f"{REMOTE}/rubric.jsonl.parts/000000000000-000000000009", b'{"a": 1}\n')
    storage.write_bytes(f"{REMOTE}/rubric.jsonl.parts/000000000012-000000000021", b'{"a": 2}\n')
    with pytest.raises(DataError, match="not contiguous"):
        restore(tmp_path, REMOTE)


def test_a_rewritten_store_is_refused_rather_than_mirrored_over(tmp_path: Path) -> None:
    store = tmp_path / "rubric.jsonl"
    store.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    Mirror(tmp_path, REMOTE).flush()
    store.write_bytes(b'{"b": 1}\n{"b": 2}\n{"b": 3}\n')
    with pytest.raises(DataError, match="no longer extends"):
        Mirror(tmp_path, REMOTE).flush()


def test_a_run_whose_node_vanished_resumes_from_its_mirror_asking_only_the_missing_windows(
    data: Path, tmp_path: Path
) -> None:
    import json
    import shutil

    from click.testing import CliRunner

    import rcp_ndcg
    from rcp_ndcg.cli.main import cli
    from tests.runs.conftest import tiny_config

    price = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}
    judge = {"base_url": "fake://seed/0", "model": "fake", "price": price}
    config = tiny_config(data, judge=judge, steps=["tournament"], mirror=REMOTE, budget_usd=0.03)

    stopped = rcp_ndcg.run(config, runs_dir=str(tmp_path / "runs"))  # the budget stops it partway
    store = Path(stopped.dir) / "judgements" / "tournament.jsonl"
    judged = len(store.read_text(encoding="utf-8").splitlines())
    assert stopped.manifest.status.value == "partial" and 0 < judged
    assert stopped.status().mirror.last_upload_at is not None

    shutil.rmtree(stopped.dir)  # the node is gone; only the mirror is left
    args = ["run", "resume", "--run", stopped.dir, "--mirror", REMOTE, "--budget-usd", "10", "--json"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output

    resumed = rcp_ndcg.runs.run.Run(stopped.dir)
    total = len(store.read_text(encoding="utf-8").splitlines())
    asked = resumed.manifest.step("tournament").usage.requests
    assert resumed.manifest.status.value == "completed"
    assert asked == total - judged  # the restored windows were not asked again
    assert json.loads(result.stdout)["data"]["state"]["mirror"]["remote"] == REMOTE


def test_a_remote_runs_directory_is_refused(data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.runs import Pipeline
    from tests.runs.conftest import tiny_config

    monkeypatch.chdir(tmp_path)  # a regression must not write a run into the checkout
    with pytest.raises(ConfigError, match="local") as refused:
        Pipeline(tiny_config(data), runs_dir="gs://bucket/runs")
    assert "--mirror" in (refused.value.hint or "")


def test_judge_mirrors_its_store(data: Path, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from rcp_ndcg.cli.main import cli

    out = tmp_path / "store"
    args = ["judge", "rubric", "--dataset", f"jsonl:{data}", "--judge", "fake", "--set", "schedule.window=4"]
    result = CliRunner().invoke(cli, [*args, "--out", str(out), "--mirror", REMOTE, "--json"])
    assert result.exit_code == 0, result.output
    assert b"".join(storage.read_bytes(uri) for uri in storage.ls(f"{REMOTE}/rubric.jsonl.parts")) == (
        (out / "rubric.jsonl").read_bytes()
    )
    assert storage.exists(f"{REMOTE}/identity.json")


def test_a_mirror_on_an_unavailable_filesystem_names_what_is_missing(tmp_path: Path) -> None:
    from rcp_ndcg.errors import DependencyError

    with pytest.raises(DependencyError, match="no fsspec filesystem is registered for nosuchstore://") as unknown:
        Mirror(tmp_path, "nosuchstore://bucket/runs")
    assert "register_implementation" in (unknown.value.hint or "")
    if importlib.util.find_spec("s3fs") is None:
        with pytest.raises(DependencyError, match="s3fs") as missing:
            Mirror(tmp_path, "s3://bucket/runs")
        assert missing.value.hint == 'pip install "rcp-ndcg[s3]"'


def test_a_custom_filesystem_registered_with_fsspec_is_a_mirror(tmp_path: Path) -> None:
    import fsspec
    from fsspec import AbstractFileSystem

    class Objects(AbstractFileSystem):
        """Three operations over a dict: all a mirror needs."""

        protocol = "objects"
        store: dict[str, bytes] = {}

        def pipe_file(self, path, value, **kwargs):
            self.store[self._strip_protocol(path)] = bytes(value)

        def cat_file(self, path, start=None, end=None, **kwargs):
            return self.store[self._strip_protocol(path)]

        def ls(self, path, detail=True, **kwargs):
            prefix = self._strip_protocol(path).rstrip("/") + "/"
            names = {prefix + key[len(prefix) :].split("/")[0] for key in self.store if key.startswith(prefix)}
            if not names:
                raise FileNotFoundError(path)
            entries = [{"name": n, "type": "file" if n in self.store else "directory", "size": 0} for n in names]
            return entries if detail else sorted(names)

    fsspec.register_implementation("objects", Objects, clobber=True)
    (tmp_path / "run" / "judgements").mkdir(parents=True)
    (tmp_path / "run" / "judgements" / "rubric.jsonl").write_bytes(b'{"a": 1}\n')
    (tmp_path / "run" / "manifest.json").write_text("{}", encoding="utf-8")
    Mirror(tmp_path / "run", "objects://bucket/run").flush()
    assert sorted(restore(tmp_path / "copy", "objects://bucket/run")) == ["judgements/rubric.jsonl", "manifest.json"]
    assert (tmp_path / "copy" / "judgements" / "rubric.jsonl").read_bytes() == b'{"a": 1}\n'


def test_a_hub_mirror_warns_that_it_is_for_publishing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from rcp_ndcg.errors import DependencyError

    with caplog.at_level("WARNING"), contextlib.suppress(DependencyError):  # without huggingface_hub installed
        Mirror(tmp_path, "hf://datasets/someone/runs")
    assert "every write to the Hugging Face Hub is a commit" in caplog.text
