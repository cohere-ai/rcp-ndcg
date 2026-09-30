"""Re-judge a subset of documents and score them without refitting the calibration.

Two documents of q2 were missing when the pools were judged and calibrated (a failed window, a document added
later). Judging only them on the rubric appends their windows to the same store; `score_documents` then gives each
one a calibrated ability with the calibration's item parameters frozen, so no published gain moves.

    python examples/04_rejudge_subset.py
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
# An offline judge whose hidden abilities are the released thetas; a real one is rcp.JudgeConfig.load(<config>).
judge = FakeJudge(lambda text: ability[html.unescape(text)], seed=0)

missing = ["q2-d6", "q2-d7"]
pools = {q: [d for d in docs if d not in missing] for q, docs in dataset.candidates.items()}
tournament = rcp.TournamentSchedule(window=4, random_windows=4, stratified_windows=2, adaptive_window=4,
                                    adaptive_batches=1, adaptive_windows_per_batch=2)  # fmt: skip
rubric = rcp.RubricSchedule(window=4, windows_per_query=8, random_windows=4)

rcp.judge(dataset, pools, judge, stage="tournament", out="judgements", schedule=tournament)
judged = rcp.judge(dataset, pools, judge, stage="rubric", out="judgements", schedule=rubric)
calibration = rcp.calibrate([judged])

# Only the missing documents: the store is append-only, so only their new windows are asked.
rejudged = rcp.judge(dataset, None, judge, stage="rubric", out="judgements", schedule=rubric, docs={"q2": missing})
extension = rcp.score_documents(calibration, rejudged)  # the items are frozen: no published ability moves
for record in extension.records:
    print(f"{record.doc_id}: theta {record.estimate.theta:+.2f} (se {record.estimate.se:.2f})")

extended = calibration.extended(extension)
print({doc: round(gain, 3) for doc, gain in extended.gains()["q2"].items() if doc in missing})
