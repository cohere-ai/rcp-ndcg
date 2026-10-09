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
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil

    from click.testing import CliRunner

    import rcp_ndcg
    from rcp_ndcg.cli.main import cli
    from rcp_ndcg.judging import JudgeClient
    from rcp_ndcg.judging.client import BackendUnavailableError
    from tests.runs.conftest import tiny_config

    complete = JudgeClient.complete
    answered = []

    async def down_after_three(self, request):
        if len(answered) >= 3:
            raise BackendUnavailableError("the endpoint went away")
        answered.append(request)
        return await complete(self, request)

    monkeypatch.setattr(JudgeClient, "complete", down_after_three)
    config = tiny_config(data, steps=["tournament"], mirror=REMOTE)
    with pytest.raises(BackendUnavailableError):  # the endpoint stops the run partway
        rcp_ndcg.run(config, runs_dir=str(tmp_path / "runs"))
    monkeypatch.setattr(JudgeClient, "complete", complete)
    (run_dir,) = (tmp_path / "runs").iterdir()
    stopped = rcp_ndcg.runs.run.Run(str(run_dir))
    store = Path(stopped.dir) / "judgements" / "tournament.jsonl"
    judged = len(store.read_text(encoding="utf-8").splitlines())
    assert stopped.manifest.status.value == "failed" and 0 < judged
    assert stopped.status().mirror.last_upload_at is not None

    shutil.rmtree(stopped.dir)  # the node is gone; only the mirror is left
    args = ["run", "resume", "--run", stopped.dir, "--mirror", REMOTE, "--json"]
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


@pytest.mark.parametrize("scheme", ["file://", ""])
def test_a_local_or_shared_path_is_a_mirror_and_restores(data: Path, tmp_path: Path, scheme: str) -> None:
    """A file:// or plain-path mirror failed: its parent directories were never created."""
    import shutil

    import rcp_ndcg
    from tests.runs.conftest import tiny_config

    remote = f"{scheme}{tmp_path / 'shared' / 'mirror'}"
    run = rcp_ndcg.run(tiny_config(data, mirror=remote), runs_dir=str(tmp_path / "runs"))
    assert run.manifest.status.value == "completed"
    assert (tmp_path / "shared" / "mirror" / "calibration" / "coverage.json").is_file()
    state = run.status().mirror
    assert state is not None and state.last_error is None and state.last_upload_at is not None
    original = {path.relative_to(run.dir): path.read_bytes() for path in Path(run.dir).rglob("*") if path.is_file()}

    shutil.rmtree(run.dir)
    assert "manifest.json" in restore(run.dir, remote)
    for relative, payload in original.items():
        if relative.parts[0] != "logs":
            assert (Path(run.dir) / relative).read_bytes() == payload, relative


def test_a_local_directory_behind_its_mirror_takes_the_mirrors_newer_files(tmp_path: Path) -> None:
    """A run that went on elsewhere (a Kubernetes pod): the local manifest stayed 'submitted' after a restore."""
    import json

    elsewhere = tmp_path / "pod"
    (elsewhere / "metrics").mkdir(parents=True)
    (elsewhere / "manifest.json").write_text(json.dumps({"status": "completed", "updated_at": "2026-09-30T12:00:00Z"}))
    (elsewhere / "run.yaml").write_text("same\n")
    (elsewhere / "metrics" / "report.json").write_text("{}")
    Mirror(elsewhere, REMOTE).flush()

    here = tmp_path / "here"
    here.mkdir()
    (here / "manifest.json").write_text(json.dumps({"status": "submitted", "updated_at": "2026-09-30T11:00:00Z"}))
    (here / "run.yaml").write_text("same\n")
    (here / "notes.txt").write_text("only here")
    assert restore(here, REMOTE) == ["manifest.json", "metrics/report.json"]
    assert json.loads((here / "manifest.json").read_text())["status"] == "completed"
    assert (here / "notes.txt").read_text() == "only here"

    # A local copy ahead of its mirror keeps what it holds.
    (here / "manifest.json").write_text(json.dumps({"status": "running", "updated_at": "2026-09-30T13:00:00Z"}))
    assert restore(here, REMOTE) == []
    assert json.loads((here / "manifest.json").read_text())["status"] == "running"


def test_a_final_flush_that_fails_is_recorded_for_run_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job failed on its last upload while run status showed a completed run with no mirror state at all."""
    from rcp_ndcg.runs.mirror import mirrored

    state_file = tmp_path / "logs" / "mirror.json"
    (tmp_path / "manifest.json").write_text("{}")

    def refuse(self, relative, payload):
        raise PermissionError("the bucket refused the write")

    monkeypatch.setattr("rcp_ndcg.runs.mirror._Target.write", refuse)
    with pytest.raises(PermissionError), mirrored(tmp_path, REMOTE, interval_s=3600, state_file=state_file):
        pass
    state = read_state(state_file)
    assert state is not None and "PermissionError: the bucket refused the write" in (state.last_error or "")


def test_a_torn_state_file_reads_as_never_ran_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A reader racing a flush (or a killed writer) sees a partial JSON: the state is treated as absent, as the
    store treats a torn identity, and the parse is never allowed to crash ``run status``."""
    state = tmp_path / ".mirror.json"
    state.write_text('{"remote": "memory://mirror/run", "last_upload_at": "2026-10-0', encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert read_state(state) is None
    assert "torn" in caplog.text


def test_run_status_survives_a_torn_state_file(
    finished: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import shutil

    from rcp_ndcg.runs.run import Run

    run_dir = Path(shutil.copytree(finished, tmp_path / "torn"))
    (run_dir / "logs" / "mirror.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        state = Run(run_dir).status()
    assert state.mirror is None
    assert "torn" in caplog.text


def test_the_state_file_is_published_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The state file goes through the storage helper (temp file + rename), never a plain ``write_text``."""
    import json

    published: list[str] = []
    real = storage.publish_bytes

    def record(target, payload):
        published.append(str(target))
        real(target, payload)

    monkeypatch.setattr(storage, "publish_bytes", record)
    Mirror(tmp_path, REMOTE).flush()
    assert published == [str(tmp_path / ".mirror.json")]
    assert json.loads((tmp_path / ".mirror.json").read_text(encoding="utf-8"))["remote"] == REMOTE


def test_a_local_mirror_publishes_whole_files_through_the_storage_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local or shared mirror must not ``pipe_file`` whole files in place: a concurrent ``restore()`` could
    read a partial ``manifest.json``. ``storage.publish_bytes`` writes the temp file and renames it."""
    import json

    local = tmp_path / "run"
    local.mkdir()
    (local / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    remote = tmp_path / "shared" / "mirror"
    published: list[str] = []
    real = storage.publish_bytes

    def record(target, payload):
        published.append(str(target))
        real(target, payload)

    monkeypatch.setattr(storage, "publish_bytes", record)
    Mirror(local, str(remote)).flush()
    assert json.loads((remote / "manifest.json").read_text(encoding="utf-8")) == {"v": 1}
    assert str(remote / "manifest.json") in published


def test_a_dry_run_refuses_a_mirror_the_real_run_would_refuse_and_the_run_leaves_no_directory(
    data: Path, tmp_path: Path
) -> None:
    import json

    import yaml
    from click.testing import CliRunner

    from rcp_ndcg.cli.main import cli
    from tests.runs.conftest import tiny_config

    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")
    runs = tmp_path / "runs"
    for extra in ("--dry-run", "--estimate", None):
        args = ["run", "start", str(config), "--runs-dir", str(runs), "--mirror", "nosuchstore://b/x", "--json"]
        result = CliRunner().invoke(cli, [*args, *([extra] if extra else [])])
        assert result.exit_code == 10, (extra, result.output)
        assert "nosuchstore" in json.loads(result.stdout)["error"]["message"]
    assert not runs.exists()
