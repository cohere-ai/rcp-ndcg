"""The fixture reference, run as a subprocess in its own environment (never imported by the harness).

The reference uses the product's :func:`rcp_ndcg.data.preprocess.fit` for the anchor-preserving render -- the
same call the served path makes -- and the model's in-process scoring with the same deterministic numbers as
the stub engine (tests/fixtures/deterministic.py).  It reads the pairs file and writes the mode's JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_TOKENS = 160
QUERY_MAX_TOKENS = 48
HEAD = "SYSTEM: Judge whether the Document answers the Query. Answer yes or no. USER: Query: "
BETWEEN = " Document: "
TAIL = " ASSISTANT"


def main() -> int:
    from rcp_ndcg.data.preprocess import TextBudget, fit
    from rcp_ndcg.data.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(description="the fixture rerank reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from deterministic import fold, score

    recipe = yaml.safe_load((Path(__file__).parent / "recipe.yaml").read_text(encoding="utf-8"))
    client = recipe["client"]
    tokenizer = load_tokenizer(args.tokenizer)
    budget = TextBudget(
        tokenizer=args.tokenizer,
        max_tokens=client["max_tokens"],
        query_max_tokens=client.get("query_max_tokens"),
        template=client.get("template"),
        on_overflow=client.get("on_overflow", "cut"),
    )
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.mode == "render":
        rows = []
        for index, row in enumerate(pairs):
            folded_query = (
                f"Task: {row.get('instruction')}\nQuery: {row['query']}" if row.get("instruction") else row["query"]
            )
            result = fit([(folded_query, row["documents"][0])], "pair", budget, tokenizer, ids=["0"])
            rows.append({"index": index, "shape": "pair", "text": result.texts[0]})
        output_result = {"rows": rows}
    else:
        rows = []
        for index, row in enumerate(pairs):
            folded = fold(row["query"], row.get("instruction"))
            rows.append({"index": index, "scores": [score(folded, document) for document in row["documents"]]})
        output_result = {"rows": rows}
    Path(args.out).write_text(json.dumps(output_result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
