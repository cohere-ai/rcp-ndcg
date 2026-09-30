"""Pool two judges who answered the same rubric into one calibration.

Both judges answer the rubric (C1-C5) on the same pools; the first also runs the tournament. The pooled fit shares
one set of item parameters and gives each judge a severity offset on the logit scale: a lenient judge gets a
negative one. Each judge writes its own judgement store, because judgements of two judges are two families.

    python examples/06_two_judges.py
"""

import html
import json

import rcp_ndcg as rcp
from rcp_ndcg.examples import tiny
from rcp_ndcg.testing import FakeJudge

TINY = tiny()  # the example dataset that ships with the package
dataset = rcp.load_dataset(f"jsonl:{TINY / 'rows.jsonl'}")
thetas = json.loads((TINY / "released.json").read_text())["thetas"]
rows = [json.loads(line) for line in (TINY / "rows.jsonl").read_text().splitlines()]
ability = {text: thetas[r["query_id"]][doc] for r in rows for doc, text in zip(r["doc_ids"], r["docs"], strict=True)}


def fake(name: str, severity: float) -> FakeJudge:
    return FakeJudge(lambda text: ability[html.unescape(text)], name=name, severity=severity, seed=0)


strict, lenient = fake("strict", 0.3), fake("lenient", -0.3)
tournament = rcp.TournamentSchedule(window=4, random_placements=2.0, stratified_placements=1.0,
                                    adaptive_window=4, adaptive_batches=1, adaptive_placements=1.0)  # fmt: skip
rubric = rcp.RubricSchedule(window=4, placements_per_doc=4.0, random_share=0.5)

judgements = [
    rcp.judge(dataset, None, strict, stage="tournament", out="strict", schedule=tournament),
    rcp.judge(dataset, None, strict, stage="rubric", out="strict", schedule=rubric),
    rcp.judge(dataset, None, lenient, stage="rubric", out="lenient", schedule=rubric),
]
pooled = rcp.calibrate(judgements, judges="pooled")
print("judge severity (logits):", {judge: round(s, 2) for judge, s in pooled.judge_severity.items()})
print("criteria difficulties (beta):", [round(b, 2) for b in pooled.items.beta])

single = rcp.calibrate(judgements[:2])  # the first judge alone, for comparison
print("first judge alone (beta):    ", [round(b, 2) for b in single.items.beta])
