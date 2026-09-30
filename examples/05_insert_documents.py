"""Insert a new document into a tournament calibration without refitting it.

The pool of q1 grew by one document after the calibration was published. The new document is judged in Stage A
windows against documents the calibration already ranks, and `insert_documents` estimates its ability with every
existing ability held fixed. The anchor report checks that no published gain moved.

    python examples/05_insert_documents.py
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
judge = FakeJudge(lambda text: ability[html.unescape(text)], seed=0)

new = "q1-d1"
pools = {q: [d for d in docs if d != new] for q, docs in dataset.candidates.items()}
tournament = rcp.TournamentSchedule(window=4, random_windows=4, stratified_windows=2, adaptive_window=4,
                                    adaptive_batches=1, adaptive_windows_per_batch=2)  # fmt: skip
rubric = rcp.RubricSchedule(window=4, windows_per_query=8, random_windows=4)
fitted = rcp.judge(dataset, pools, judge, stage="tournament", out="judgements", schedule=tournament)
answered = rcp.judge(dataset, pools, judge, stage="rubric", out="judgements", schedule=rubric)
calibration = rcp.calibrate([fitted, answered])

# Two windows of the schedule's size (four), each holding the new document and three ranked opponents; in a real
# pool `rcp_ndcg.calibration.select_opponents(calibration, "q1", new, n=..., window=4)` picks them. Exactly these
# windows are judged (each also reversed: the schedule mirrors), into the calibration's own store.
windows = [[new, *pools["q1"][:3]], [new, *pools["q1"][3:6]]]
judged = rcp.judge(dataset, None, judge, stage="tournament", out="judgements", schedule=tournament,
                   windows={"q1": windows})  # fmt: skip

# A pool of eight has few opponents to offer: accept a standard error of 1.5 logits (the default asks for 0.5,
# which a real pool reaches with more opponents: `n=36` in windows of ten is four windows, eight calls).
extension = rcp.insert_documents(calibration, judged, se_target=1.5)
print("anchor report:", extension.anchor_report.ok, "largest gain shift", extension.anchor_report.max_abs_gain_shift)
(record,) = extension.records
print(f"{record.doc_id}: theta {record.estimate.theta:+.2f} (se {record.estimate.se:.2f})")
order = sorted(calibration.extended(extension).gains()["q1"].items(), key=lambda kv: -kv[1])
print("q1 by gain:", [doc for doc, _ in order])
