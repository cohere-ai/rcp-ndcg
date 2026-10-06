"""The fixture reference, run as a subprocess in its own environment (never imported by the harness).

The reference uses the product's own :func:`rcp_ndcg.data.preprocess.fit` for the anchor-preserving render --
the same call the served path makes -- and the embedding model's vectors with the same deterministic numbers
as the stub engine (tests/fixtures/deterministic.py).  It reads the pairs file and writes the mode's JSON to
--out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_TOKENS = 128


def main() -> int:
    from rcp_ndcg.data.preprocess import TextBudget, fit
    from rcp_ndcg.data.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(description="the fixture embed reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from deterministic import fold, vector  # noqa: F401

    recipe = yaml.safe_load((Path(__file__).parent / "recipe.yaml").read_text(encoding="utf-8"))
    client = recipe["client"]
    tokenizer = load_tokenizer(args.tokenizer)
    budget = TextBudget(
        tokenizer=args.tokenizer,
        max_tokens=client["max_tokens"],
        template=client.get("template"),
        on_overflow=client.get("on_overflow", "cut"),
    )
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    def fit_one(shape: str, query: str, document: str, instruction: str | None) -> str:
        """The product's fit render for one input."""
        inputs = [(query, document)] if shape == "pair" else [query if shape == "query" else document]
        result = fit(inputs, shape, budget, tokenizer, ids=["0"])
        return result.texts[0]

    if args.mode == "render":
        rows = []
        for index, row in enumerate(pairs):
            shapes = set(
                shape for shape in ("query", "document", "pair") if shape in (client.get("template") or {})
            ) or {"document"}
            for shape in shapes:
                rendered = fit_one(shape, row["query"], row["documents"][0], row.get("instruction"))
                rows.append({"index": index, "shape": shape, "text": rendered})
        output_result = {"rows": rows}
    else:
        from deterministic import token_vectors

        shapes = set(shape for shape in ("query", "document", "pair") if shape in (client.get("template") or {})) or {
            "document"
        }
        rows = []
        for index, row in enumerate(pairs):
            row_result: dict = {"index": index}
            for shape in shapes:
                texts_in = (
                    [(row["query"], document) for document in row["documents"]]
                    if shape == "pair"
                    else [row["query"] if shape == "query" else document for document in row["documents"]]
                )
                fitted = [fit([inp], shape, budget, tokenizer, ids=[str(i)]).texts[0] for i, inp in enumerate(texts_in)]

                vectors = [
                    [[float(x) for x in row] for row in token_vectors(text, "tok").astype("float32")] for text in fitted
                ]
                if shape == "query":
                    row_result["query_vectors"] = vectors
                else:
                    row_result["document_vectors"] = vectors
            rows.append(row_result)
        output_result = {"rows": rows}
    Path(args.out).write_text(json.dumps(output_result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
