"""The error hierarchy and :func:`rcp_ndcg.errors.classify`: every failure maps to one typed, actionable error."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

import rcp_ndcg
from rcp_ndcg import errors
from rcp_ndcg.errors import DataError, ExitCode, RcpNdcgError, RcpNdcgWarning, classify


def _validation_error() -> ValidationError:
    class Model(BaseModel):
        k: int

    try:
        Model(k="x")  # type: ignore[arg-type]
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def _yaml_error() -> yaml.YAMLError:
    try:
        yaml.safe_load("a: [")
    except yaml.YAMLError as exc:
        return exc
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (FileNotFoundError(2, "No such file", "runs/x"), ExitCode.MISSING_INPUT),
        (ModuleNotFoundError("No module named 'torch'", name="torch"), ExitCode.DEPENDENCY),
        (ImportError("reading the released data needs pyarrow: pip install 'rcp-ndcg[data]'"), ExitCode.DEPENDENCY),
        (_validation_error(), ExitCode.CONFIG),
        (_yaml_error(), ExitCode.CONFIG),
        (json.JSONDecodeError("Expecting value", "x", 0), ExitCode.DATA),
        (ConnectionError("refused"), ExitCode.PROVIDER),
        (TimeoutError("timed out"), ExitCode.PROVIDER),
        (KeyboardInterrupt(), ExitCode.INTERRUPTED),
        # A bare ValueError or TypeError is not a statement about the user's config: code that refuses input
        # deliberately raises a typed error, so an untyped one is a bug (exit 1), not "fix your config" (exit 3).
        (ValueError("operands could not be broadcast together"), ExitCode.INTERNAL),
        (TypeError("f() got an unexpected keyword argument 'x'"), ExitCode.INTERNAL),
        (ModuleNotFoundError("No module named 'nonesuch'", name="nonesuch"), ExitCode.INTERNAL),
        (RuntimeError("boom"), ExitCode.INTERNAL),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value),
)
def test_classify_maps_foreign_exceptions(exc: BaseException, code: ExitCode) -> None:
    error = classify(exc)

    assert isinstance(error, RcpNdcgError)
    assert error.exit_code is code
    assert error.hint


def test_a_missing_extra_names_the_install_command() -> None:
    error = classify(ModuleNotFoundError("No module named 'torch'", name="torch"))

    assert error.hint == 'pip install "rcp-ndcg[calibrate]"'
    assert error.details == {"module": "torch", "extra": "calibrate"}


def test_typed_errors_pass_through_unchanged() -> None:
    original = errors.IdentityError("config changed", details={"differing": ["judge.model"]})

    assert classify(original) is original


def test_the_error_object_has_the_envelope_shape() -> None:
    payload = errors.CapabilityError(
        "the endpoint refused the answer schema", hint="set decoding: free", details={"status": 400}
    ).to_dict()

    assert payload == {
        "code": "CAPABILITY",
        "exit_code": 8,
        "message": "the endpoint refused the answer schema",
        "hint": "set decoding: free",
        "retryable": False,
        "details": {"status": 400},
    }
    assert errors.ProviderError("503").retryable is True
    assert errors.ProviderError("400", retryable=False).retryable is False


def test_every_exit_code_has_exactly_one_error_class() -> None:
    classes = [obj for obj in vars(errors).values() if isinstance(obj, type) and issubclass(obj, RcpNdcgError)]
    owners: dict[ExitCode, list[str]] = {}
    for cls in classes:
        owners.setdefault(cls.exit_code, []).append(cls.__name__)

    assert set(owners) == set(ExitCode) - {ExitCode.SUCCESS}
    assert all(len(names) == 1 for names in owners.values()), owners


def test_a_retired_exit_code_is_never_reused() -> None:
    """7 was the spend ceiling's code; it is retired and the codes after it keep their numbers."""
    assert 7 not in {int(code) for code in ExitCode}
    assert (ExitCode.PROVIDER, ExitCode.CAPABILITY, ExitCode.DATA) == (6, 8, 12)


def test_a_warning_code_comes_from_the_closed_list() -> None:
    assert RcpNdcgWarning("UNREADABLE_RUN", "x").to_dict() == {"code": "UNREADABLE_RUN", "message": "x"}
    with pytest.raises(ValueError, match="unknown warning code"):
        RcpNdcgWarning("SOMETHING_ELSE", "x")  # type: ignore[arg-type]


def test_a_provider_failure_names_the_concurrency_setting_that_exists() -> None:
    hint = errors.classify(ConnectionError("refused")).hint
    assert "--concurrency" not in hint and "judge.concurrency" in hint


def test_sigterm_stops_a_command_with_the_interrupted_exit_code(tmp_path) -> None:
    """A scheduler stops a job with SIGTERM; the command reports INTERRUPTED (9), as it does for Ctrl-C."""
    import json
    import signal
    import subprocess
    import sys
    import time

    ready = tmp_path / "ready"
    code = (
        "import platform, sys, time\n"
        "from pathlib import Path\n"
        "from rcp_ndcg.cli.main import cli\n"
        f"platform.python_version = lambda: (Path({str(ready)!r}).touch(), time.sleep(60))[1]\n"
        "sys.argv = ['rcp-ndcg', 'doctor', '--json']\n"
        "cli.main()\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while not ready.exists():
        assert time.monotonic() < deadline and process.poll() is None
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=60)
    assert process.returncode == 9
    assert json.loads(out)["error"]["code"] == "INTERRUPTED"


_PYTHON_WORDING = re.compile(r"(?<![\w.\-])[a-z_]+=(?!=)|\w\((\.\.\.|\))")


def test_every_python_worded_hint_has_a_command_line_one() -> None:
    """A hint that names keyword arguments or calls (``gains=``, ``force=True``, ``index()``) also says it in flags."""
    offending = []
    for path in sorted((Path(rcp_ndcg.__file__).parent).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            if "hint" not in keywords or "cli_hint" in keywords:
                continue
            value = keywords["hint"]
            parts = [value] if isinstance(value, ast.Constant) else getattr(value, "values", [])
            text = "".join(
                part.value for part in parts if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if _PYTHON_WORDING.search(text):
                offending.append(f"{path.name}:{node.lineno}: {text}")
    assert not offending


def test_the_cli_shows_the_command_line_hint_and_python_keeps_its_own(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from rcp_ndcg.calibration import calibrate, read_judgements
    from rcp_ndcg.cli.main import cli
    from rcp_ndcg.llm import judge
    from rcp_ndcg.testing import TINY_TOURNAMENT, FakeJudge, tiny_rows

    rows, _ = tiny_rows()
    judge(rows, None, FakeJudge(), stage="tournament", out=tmp_path / "store", schedule=TINY_TOURNAMENT)
    with pytest.raises(DataError) as python:
        calibrate(read_judgements(tmp_path / "store"))
    result = CliRunner().invoke(cli, ["calibration", "fit", "--judgements", str(tmp_path / "store"), "--out",
                                      str(tmp_path / "cal"), "--json"])  # fmt: skip

    assert "stage='rubric'" in (python.value.hint or "")
    assert result.exit_code == 12
    assert json.loads(result.stdout)["error"]["hint"] == "judge the candidates with `rcp-ndcg judge rubric` first"
