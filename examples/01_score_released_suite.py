"""Score rerankers on a released RCP-nDCG dataset, without an LLM (needs the network and the [hf] extra).

Every released dataset carries the calibrated RCP gains of its judged pools, and under `provenance/runs/` the
stored rankings of the 14 rerankers the paper compares. This example scores those rankings on one NanoBEIR task
with the paper's NanoBEIR protocol. Your own system works the same way: a Parquet, CSV, TREC or JSONL file with
query ids, document ids and scores (`rcp.load_rankings`), or `rcp-ndcg eval score --rankings FILE --suite nanobeir`.

    pip install "rcp-ndcg[hf]"
    python examples/01_score_released_suite.py
"""

import rcp_ndcg as rcp

REPO = "fabianschmidt-cohere/rcp-ndcg-nanobeir"
TASK = "NanoFiQA2018Retrieval"

dataset = rcp.load_dataset(f"hf://{REPO}/{TASK}")  # qrels with gains, judged pools, excluded ids; protocol nanobeir
rankings = rcp.load_rankings(f"hf://datasets/{REPO}/provenance/runs/{TASK}.parquet")
rerankers = [s for s in rankings.systems if "judge" not in s]  # the file also holds the judge's own orders

report = rcp.evaluate(rankings, dataset=dataset, k=10, bootstrap=0)
print(f"{TASK} under the {dataset.protocol} protocol")
print(f"{'reranker':42s} {'RCP-nDCG@10':>12s} {'qrel-nDCG@10':>13s}")
for system in sorted(rerankers, key=lambda s: -report.value(s, "rcp_ndcg")):
    print(f"{system:42s} {report.value(system, 'rcp_ndcg'):12.4f} {report.value(system, 'qrel_ndcg'):13.4f}")
