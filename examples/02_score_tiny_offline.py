"""Score two systems with RCP-nDCG and qrel-nDCG, offline: the tiny dataset and its released-style calibration.

The released datasets carry, for every judged pool document, its calibrated ability `theta` and the judge's item
parameters. This example does the same with a three-query dataset that ships with the package (`rcp-ndcg data
fetch --dataset tiny --out DIR` copies it): it turns the thetas into gains, scores two systems' rankings, and
compares them.

    python examples/02_score_tiny_offline.py
"""

import json

import rcp_ndcg as rcp
from rcp_ndcg.examples import tiny

TINY = tiny()  # the example dataset that ships with the package

dataset = rcp.load_dataset(f"jsonl:{TINY / 'rows.jsonl'}")  # queries, documents, qrels, candidate pools
released = json.loads((TINY / "released.json").read_text())
gains = {
    query_id: {doc_id: rcp.gain(theta, released["items"]) for doc_id, theta in thetas.items()}
    for query_id, thetas in released["thetas"].items()
}
rankings = rcp.load_rankings(TINY / "systems.jsonl")  # two systems: bm25 and reranker

report = rcp.evaluate(rankings, dataset=dataset, gains=gains, k=5, bootstrap=200)
for system in rankings.systems:
    rcp_ndcg, qrel_ndcg = report.value(system, "rcp_ndcg"), report.value(system, "qrel_ndcg")
    print(f"{system:10s} RCP-nDCG@5 {rcp_ndcg:.3f}   qrel-nDCG@5 {qrel_ndcg:.3f}")

comparison = rcp.compare(report, baseline="bm25", bootstrap=200)
print(comparison.to_pandas()[["system_a", "system_b", "delta", "p_value"]].to_string(index=False))
