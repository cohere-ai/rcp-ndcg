"""Evaluate an embedding model on a released suite through MTEB (needs the network and the [mteb] extra).

`rcp_ndcg.eval.mteb.get_tasks` returns the released suites as mteb tasks. Each reports mteb's usual metrics on the
integer qrels plus `ndcg_float_at_10`, nDCG over the continuous RCP gains with group-mean ties; by default a model
reranks each query's judged pool. The same tasks run in stock mteb from the `rcp_ndcg_tasks.py` of each dataset.

    pip install "rcp-ndcg[mteb]"
    python examples/07_mteb.py
"""

import mteb

from rcp_ndcg.eval.mteb import get_tasks

tasks = get_tasks("nanobeir", ["NanoFiQA2018Retrieval"])
model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
results = mteb.evaluate(model, tasks=tasks)

for task_result in results:
    scores = next(iter(task_result.scores.values()))[0]  # the task's one evaluation split
    print(f"{task_result.task_name}: ndcg_float_at_10 {scores['ndcg_float_at_10']:.4f} (RCP gains), "
          f"ndcg_at_10 {scores['ndcg_at_10']:.4f} (integer qrels)")  # fmt: skip
