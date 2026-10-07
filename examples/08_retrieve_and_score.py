"""Produce a rankings file with BM25 and score it with RCP-nDCG, offline: retrieve, write, evaluate, compare.

python examples/08_retrieve_and_score.py
"""

import json

import rcp_ndcg as rcp
from rcp_ndcg.examples import tiny
from rcp_ndcg.retrieval import validate_retriever

TINY = tiny()  # the example dataset that ships with the package
dataset = rcp.load_dataset(f"jsonl:{TINY / 'rows.jsonl'}")
released = json.loads((TINY / "released.json").read_text())
gains = {
    query_id: {doc_id: rcp.gain(theta, released["items"]) for doc_id, theta in thetas.items()}
    for query_id, thetas in released["thetas"].items()
}

# First stage: BM25 over the corpus, then a rankings file every scoring protocol reads.
bm25 = rcp.retrieve(dataset, validate_retriever({"kind": "bm25", "stemmer": "english"}), depth=5, out="index")
bm25.save("rankings.jsonl")  # columns query_id, doc_id, score (system: bm25)
print("rankings.jsonl:", bm25.systems, "rows")

report = rcp.evaluate(bm25, dataset=dataset, gains=gains, k=5, bootstrap=200)
print(report.to_pandas())

# Compare two released systems (bm25 and reranker) on the same gains.
shipped = rcp.load_rankings(TINY / "systems.jsonl")
report = rcp.evaluate(shipped, dataset=dataset, gains=gains, k=5, bootstrap=200)
comparison = rcp.compare(report, baseline="bm25", bootstrap=200)
print(comparison.to_pandas()[["system_a", "system_b", "delta", "p_value"]].to_string(index=False))
