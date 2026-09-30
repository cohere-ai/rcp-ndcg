"""The whole pipeline offline: judge (tournament and rubric), calibrate and evaluate, with the fake judge.

`tiny` is a run config that ships with the package: `judge: fake`, a deterministic judge that answers both stages
through the real prompts and parsers, on the tiny example dataset. Every artifact of a real run is written to
`runs/<run_id>/`. Replace the judge with a judge config to judge for real; `rcp.run(..., estimate=True)` counts a
config's calls and tokens first.

    python examples/03_pipeline_with_fake_judge.py
    rcp-ndcg run start tiny            # the same from the command line, from any directory
"""

import rcp_ndcg as rcp
from rcp_ndcg.examples import tiny

estimate = rcp.run("tiny", estimate=True)
print(f"estimate: {estimate.calls} calls, ~{estimate.input_tokens} input tokens")

run = rcp.run("tiny")
status = run.status()
print(f"run {run.dir}: {status.status}, {status.requests} judge requests")
for step in status.steps:
    print(f"  {step.name:10s} {step.status}")

# Score two systems' rankings against the calibration the run fitted: no further judge calls.
calibration = rcp.Calibration.load(run.artifacts()["calibration"])
report = rcp.evaluate(
    rcp.load_rankings(tiny() / "systems.jsonl"),
    dataset=rcp.load_dataset(f"jsonl:{tiny() / 'rows.jsonl'}"),
    gains=calibration,
    k=5,
    bootstrap=0,
)
print(report.leaderboard(k=5).round(3).to_string())
