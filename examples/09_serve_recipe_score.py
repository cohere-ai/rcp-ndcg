"""Serve a retrieval model from a recipe and score it (needs an engine: set RCP_NDCG_ENGINE_URL).

The engine side is `rcp-ndcg-vllm serve qwen3-embedding-0.6b` on the stock vLLM image
(`python3 -m pip install --no-deps rcp-ndcg-vllm`). This example prints the engine argv the recipe builds and
then retrieves with the recipe's client block against the running engine.

    python examples/09_serve_recipe_score.py
"""

import json
import os

import rcp_ndcg as rcp
from rcp_ndcg.examples import tiny
from rcp_ndcg.retrieval import validate_retriever

url = os.environ.get("RCP_NDCG_ENGINE_URL")  # e.g. http://127.0.0.1:8000/v1
if not url:
    print("skipped: start an engine with `rcp-ndcg-vllm serve qwen3-embedding-0.6b` and set RCP_NDCG_ENGINE_URL")
    raise SystemExit(0)

try:
    from rcp_ndcg_vllm.recipe import iter_recipes, serve_argv
except ImportError:
    print("skipped: install the serving package first (python3 -m pip install --no-deps rcp-ndcg-vllm)")
    raise SystemExit(0) from None

recipe = next(r for r in iter_recipes() if r.id == "qwen3-embedding-0.6b")
print("engine argv:", serve_argv(recipe, port=8000, served_model_name=recipe.id))
print("(rcp-ndcg-vllm serve builds the same argv; --dry-run prints it without running it)")

TINY = tiny()
dataset = rcp.load_dataset(f"jsonl:{TINY / 'rows.jsonl'}")
released = json.loads((TINY / "released.json").read_text())
gains = {
    query_id: {doc_id: rcp.gain(theta, released["items"]) for doc_id, theta in thetas.items()}
    for query_id, thetas in released["thetas"].items()
}

# recipe: takes the whole client block from the recipe; base_url (run time) stays ours.
retriever = validate_retriever({"kind": "dense", "encoder": {"recipe": "qwen3-embedding-0.6b", "base_url": url}})
rankings = rcp.retrieve(dataset, retriever, depth=5, out="index")
rankings.save("rankings-recipe.parquet")
report = rcp.evaluate(rankings, dataset=dataset, gains=gains, k=5, bootstrap=200)
print(report.to_pandas())
