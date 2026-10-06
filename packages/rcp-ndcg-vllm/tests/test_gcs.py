"""The GCS transfer: gcs.py's copy/list/remove against a fake filesystem, and the shell dispatch.

No test touches the network: the transfer functions take the filesystem as an argument (the tests pass
a fake with the fsspec API), and the shell dispatch is exercised with fake tools on PATH.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
GCS_PY = JOBS / "gcs.py"
GCS_SH = JOBS / "gcs.sh"

PY = sys.executable


def _gcs_module() -> ModuleType:
    """The helper imported by path: a standalone script (stdlib plus gcsfs), not a package module -
    it runs on the image's python3 before any environment exists."""
    spec = importlib.util.spec_from_file_location("wave0_gcs", GCS_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gcs = _gcs_module()


class FakeGcsFs:
    """The fsspec API surface the helper uses, over two dicts: one local-side mirror, one gs:// side."""

    def __init__(self) -> None:
        self.remote: dict[str, bytes] = {}
        self.remote_dirs: set[str] = set()

    def _remote_path(self, path: str) -> str:
        return path[len("gs://") :] if path.startswith("gs://") else path

    def isdir(self, path: str) -> bool:
        prefix = path.rstrip("/") + "/"
        return any(key.startswith(prefix) for key in self.remote)

    def find(self, prefix: str) -> list[str]:
        return [key for key in sorted(self.remote) if key.startswith(prefix)]

    def put_file(self, src: str, dst: str) -> None:
        self.remote[self._remote_path(dst)] = Path(src).read_bytes()

    def get_file(self, src: str, dst: str) -> None:
        Path(dst).write_bytes(self.remote[self._remote_path(src)])

    def rm(self, path: str) -> None:
        self.remote.pop(self._remote_path(path))


def test_gcs_copy_local_to_remote_and_back(tmp_path: Path) -> None:
    fs = FakeGcsFs()
    (tmp_path / "one.txt").write_bytes(b"one")
    assert gcs.gcs_copy(str(tmp_path / "one.txt"), "gs://YOUR-BUCKET/rc0/one.txt", fs=fs) == [
        "gs://YOUR-BUCKET/rc0/one.txt"
    ]
    assert fs.remote["YOUR-BUCKET/rc0/one.txt"] == b"one"
    (tmp_path / "back").mkdir()
    assert gcs.gcs_copy("gs://YOUR-BUCKET/rc0/one.txt", str(tmp_path / "back" / "one.txt"), fs=fs) == [
        str(tmp_path / "back" / "one.txt")
    ]
    assert (tmp_path / "back" / "one.txt").read_bytes() == b"one"


def test_gcs_copy_of_a_directory_is_recursive(tmp_path: Path) -> None:
    """A directory source copies every file, same relative layout, decided by the source's kind."""
    fs = FakeGcsFs()
    (tmp_path / "wh" / "sub").mkdir(parents=True)
    (tmp_path / "wh" / "a.whl").write_bytes(b"a")
    (tmp_path / "wh" / "sub" / "b.whl").write_bytes(b"b")
    written = gcs.gcs_copy(str(tmp_path / "wh"), "gs://YOUR-BUCKET/rc0/wheelhouse", fs=fs)
    assert sorted(written) == ["gs://YOUR-BUCKET/rc0/wheelhouse/a.whl", "gs://YOUR-BUCKET/rc0/wheelhouse/sub/b.whl"]
    back = tmp_path / "back"
    written = gcs.gcs_copy("gs://YOUR-BUCKET/rc0/wheelhouse", str(back), fs=fs)
    assert (back / "a.whl").read_bytes() == b"a"
    assert (back / "sub" / "b.whl").read_bytes() == b"b"
    assert len(written) == 2


def test_gcs_copy_a_file_into_a_remote_directory(tmp_path: Path) -> None:
    """A file into a gs:// directory (or a trailing slash) keeps its basename."""
    fs = FakeGcsFs()
    fs.remote["YOUR-BUCKET/out/"] = b""  # an object that makes the prefix look like a directory
    (tmp_path / "report.json").write_bytes(b"{}")
    written = gcs.gcs_copy(str(tmp_path / "report.json"), "gs://YOUR-BUCKET/out", fs=fs)
    assert written == ["gs://YOUR-BUCKET/out/report.json"]


def test_gcs_list_and_remove(tmp_path: Path) -> None:
    fs = FakeGcsFs()
    fs.remote["YOUR-BUCKET/rc0/wheelhouse/a-1.0-py3-none-any.whl"] = b"a"
    fs.remote["YOUR-BUCKET/rc0/wheelhouse/b-1.0-py3-none-any.whl"] = b"b"
    assert gcs.gcs_list("gs://YOUR-BUCKET/rc0", fs=fs) == [
        "gs://YOUR-BUCKET/rc0/wheelhouse/a-1.0-py3-none-any.whl",
        "gs://YOUR-BUCKET/rc0/wheelhouse/b-1.0-py3-none-any.whl",
    ]
    gcs.gcs_remove("gs://YOUR-BUCKET/rc0/wheelhouse/a-1.0-py3-none-any.whl", fs=fs)
    assert gcs.gcs_list("gs://YOUR-BUCKET/rc0/wheelhouse", fs=fs) == [
        "gs://YOUR-BUCKET/rc0/wheelhouse/b-1.0-py3-none-any.whl"
    ]


def test_gcs_cli_cp_prints_the_written_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI the node scripts call, with the filesystem injected (no network)."""
    fs = FakeGcsFs()
    (tmp_path / "r.json").write_bytes(b"{}")
    monkeypatch.setattr(gcs, "make_filesystem", lambda: fs)
    assert gcs.main(["cp", str(tmp_path / "r.json"), "gs://YOUR-BUCKET/w/r.json"]) == 0
    assert fs.remote["YOUR-BUCKET/w/r.json"] == b"{}"


def test_gcs_cli_fails_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transfer that fails exits 1 with a one-line reason (no traceback) - and never the network.

    The filesystem is injected: the real ``make_filesystem`` opens a connection on a machine with
    credentials (this test once reached the service and got "Invalid bucket name" from it), so the
    fake raises like a real failure would and the CLI's contract is checked offline.
    """

    def failing_fs() -> Any:
        raise OSError("no network in tests: the transfer would fail here")

    monkeypatch.setattr(gcs, "make_filesystem", failing_fs)
    assert gcs.main(["cp", str(tmp_path / "absent.txt"), "gs://YOUR-BUCKET/x"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("gcs:") and "Traceback" not in captured.err


# --- the shell dispatch: the auth script runs first, then gcloud | gsutil | the python path -----------


def test_gcs_cp_remote_directory_download_through_the_real_dispatch(tmp_path: Path) -> None:
    """A gs:// prefix source takes the contents form on the CLI path (the stage download's shape).

    The dispatch probes the listing: a prefix with children goes to `cp -r SRC/* DST` (contents), a
    single object to `cp SRC DST`. The fake gcloud records the argv, so the regression this test pins
    (a gs:// source silently losing the recursive form, which broke the stage download) cannot return.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gcloud-dir.log"
    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >>"{log}"\n'
        'if [[ "$2" == "ls" ]]; then echo "${3%/}/"; exit 0; fi\n'  # a prefix listing: the prefix itself
        "exit 0\n",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    dst = tmp_path / "state" / "stage"
    dst.mkdir(parents=True)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && export GCS_PY="$(command -v python3)" GCS_TOOLS_DIR="$2" '
            'GCS_HELPER_PY="$3" GCS_WHEELHOUSE="" && gcs_cp "$4" "$5/" dir',
            "bash",
            str(GCS_SH),
            str(tmp_path / "gcs-tools"),
            str(GCS_PY),
            "gs://YOUR-BUCKET/rc0",
            str(dst),
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8")
    assert "storage cp -r" in calls  # the remote directory went through the recursive contents form
    assert "gs://YOUR-BUCKET/rc0/*" in calls  # the contents wildcard, expanded by the service


def test_gcs_cp_remote_single_object_takes_the_file_branch(tmp_path: Path) -> None:
    """A gs:// source that lists as exactly one object copies as a file (no -r, no wildcard)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gcloud-file.log"
    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >>"{log}"\n'
        'if [[ "$2" == "ls" ]]; then echo "${3%/}"; exit 0; fi\n'  # the object itself, no trailing slash
        "exit 0\n",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && export GCS_PY="$(command -v python3)" GCS_TOOLS_DIR="$2" '
            'GCS_HELPER_PY="$3" GCS_WHEELHOUSE="" && gcs_cp "$4" "$5" file',
            "bash",
            str(GCS_SH),
            str(tmp_path / "gcs-tools"),
            str(GCS_PY),
            "gs://YOUR-BUCKET/rc0/manifest.json",
            str(tmp_path / "out" / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8")
    assert "storage cp -r" not in calls  # the file branch, no -r
    assert "storage cp " in calls


def _fake_tools(tmp_path: Path) -> dict[str, Path]:
    """A bin dir whose python3 logs its argv (the simulated python transfer path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >>"{log}"\n'
        'if [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 0; fi\n'  # the tools install "succeeds"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text('#!/usr/bin/env bash\nfor i in 0 1 2 3 4 5 6 7; do echo "GPU $i"; done\n', encoding="utf-8")
    fake_smi.chmod(0o755)
    return {"bin": bin_dir, "log": log}


def test_gcs_transfer_detect_prefers_gcloud_then_gsutil_then_python(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    for on_path, expected in (("gcloud", "gcloud"), ("gsutil", "gsutil")):
        (tools["bin"] / on_path).write_text("#!/usr/bin/env true\n")
        (tools["bin"] / on_path).chmod(0o755)
        completed = subprocess.run(
            ["bash", "-c", f'source "{GCS_SH}" && gcs_transfer_detect'],
            capture_output=True,
            text=True,
            env={"PATH": f"{tools['bin']}:/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        assert completed.stdout.strip() == expected
        (tools["bin"] / on_path).unlink()
    completed = subprocess.run(
        ["bash", "-c", 'source "$1" && gcs_transfer_detect', "bash", str(GCS_SH)],
        capture_output=True,
        text=True,
        env={"PATH": f"{tools['bin']}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.stdout.strip() == "python"


def test_gcs_cp_without_the_clis_runs_the_python_helper(tmp_path: Path) -> None:
    """Neither CLI on PATH: the copy goes through gcs.py with the tools dir, engine env untouched."""
    tools = _fake_tools(tmp_path)
    source = tmp_path / "report.json"
    source.write_bytes(b"{}")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && export GCS_PY="$(command -v python3)" GCS_TOOLS_DIR="$2" '
            'GCS_HELPER_PY="$3" GCS_WHEELHOUSE="" && gcs_cp "$4" "$5"',
            "bash",
            str(GCS_SH),
            str(tools["bin"] / ".." / "gcs-tools"),
            str(GCS_PY),
            str(source),
            "gs://YOUR-BUCKET/w/report.json",
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{tools['bin']}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
    calls = tools["log"].read_text(encoding="utf-8")
    assert "gcs.py cp" in calls  # the helper ran through the tools python
    assert "-m pip" in calls  # the tools install ran, in its own --target directory
