"""The uploads ordering: a failed engine-logs upload reaches the uploaded report artifact.

Verbatim extraction of wave0.sh's record_upload/upload/upload_artifacts (the test_auth_invocation
pattern), with a fake gcs_cp that fails only for the logs upload: the failing attempt must appear in
the uploaded report's "uploads" section (the engine-logs incident), and the stdout emit carries the
complete record.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "wave0.sh"
REPORT_PY = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_test" / "jobs" / "report.py"


def _function(name: str) -> str:
    text = WAVE0_SH.read_text(encoding="utf-8")
    match = re.search(rf"^{name}\(\) \{{.*?^\}}\n", text, flags=re.S | re.M)
    assert match, f"wave0.sh defines {name}()"
    return match.group(0)


def test_a_failed_logs_upload_reaches_the_uploaded_report(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "logs" / "slot-0").mkdir(parents=True)
    (work / "logs" / "slot-0" / "serve.log").write_text("log line", encoding="utf-8")
    report = work / "wave0-report.json"
    bucket_out = tmp_path / "bucket-out"
    bucket_stage = tmp_path / "bucket-stage"
    subprocess.run(
        [
            Path(__import__("sys").executable),
            str(REPORT_PY),
            "init",
            "--file",
            str(report),
            "--schema",
            "rcp-ndcg.wave0-report.v1",
        ],
        check=True,
        capture_output=True,
    )

    program = (
        f'WORK="{work}"\n'
        f'REPORT_PY="{REPORT_PY}"\n'
        f'REPORT="{report}"\n'
        'OUT_URI="gs://YOUR-BUCKET/waves/wave0"\n'
        'RC_STAGE_URI="gs://YOUR-BUCKET/rc0"\n'
        'STAMP="00000000T000000Z"\n'
        "# The fake dispatch: the logs upload fails, the report copies land in their buckets.\n"
        "gcs_cp() {\n"
        '  if [[ "$1" == "$WORK/logs" ]]; then echo "fake: service unavailable" >&2; return 1; fi\n'
        '  local dst="$2" bucket="' + str(bucket_out) + '"\n'
        '  if [[ "$dst" == "$RC_STAGE_URI"* ]]; then bucket="' + str(bucket_stage) + '"; fi\n'
        '  local rel="${dst#gs://YOUR-BUCKET}"\n'
        '  mkdir -p "$bucket$(dirname "$rel")"\n'
        '  cp "$1" "$bucket$rel"\n'
        "}\n"
        + _function("record_upload")
        + _function("upload")
        + _function("upload_artifacts").replace("upload() {", "upload() { # verbatim", 1)
        + "upload_artifacts\n"
    )
    program_file = tmp_path / "program.sh"
    program_file.write_text(program, encoding="utf-8")
    completed = subprocess.run(["bash", str(program_file)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, (completed.stdout, completed.stderr, program)

    assert "fake: service unavailable" in completed.stderr  # the failure was visible on stderr
    # The two uploaded report copies: the second carries the logs failure + the first copy's attempt.
    out_copy = json.loads((bucket_out / "waves/wave0/wave0-report.json").read_text(encoding="utf-8"))
    assert out_copy["uploads"][0]["ok"] is False
    assert "service unavailable" in out_copy["uploads"][0]["error"]
    assert out_copy["uploads"][0]["destination"] == "gs://YOUR-BUCKET/waves/wave0/logs/"
    stage_copy = json.loads(
        next((bucket_stage / "rc0" / "reports").glob("wave0-report-*.json")).read_text(encoding="utf-8")
    )
    assert len(stage_copy["uploads"]) == 2  # the logs failure + the first copy's attempt
    # The emitted (local) report carries the complete record.
    emitted = json.loads(report.read_text(encoding="utf-8"))["uploads"]
    assert len(emitted) == 3 and all(entry["error"] == "" for entry in emitted[1:])
