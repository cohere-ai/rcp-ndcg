"""The fixture reference, run as a subprocess in its own environment (never imported by the harness).

The reference is deliberately independent: it folds the query and renders the spans from its own constants --
never through the product's :func:`rcp_ndcg.data.preprocess.fit`, so stage 1's render check compares two
implementations, not a function with itself.  The scores come from the same deterministic numbers as the stub
engine (``tests/fixtures/deterministic.py``).  It reads the pairs file and writes the mode's JSON to ``--out``.

Render contract (the rerank wire): per pairs-file row, ``{"index", "shape": "pair", "query": str,
"documents": [str, ...]}`` -- the spans the engine receives (the query as the client folds it per the recipe's
instruction mode, the documents as the client ships them; the frame is the engine's, from its own template).
The harness compares them against the captured request bodies, and the settle-once query is one span per row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INSTRUCTION_MODE = "fold"
"""The recipe's instruction mode: ``fold`` folds the raw query, ``field``/``system`` send it as a request
field (the query span stays raw)."""


def fold(query: str, instruction: str | None) -> str:
    """The reference's own fold render: ``Task: <instruction>\nQuery: <query>`` when the mode folds."""
    if INSTRUCTION_MODE != "fold" or not instruction:
        return query
    return f"Task: {instruction}\nQuery: {query}"


def fold_span(row: dict) -> str:
    """The query span one pairs-file row's request carries (folded when the mode folds)."""
    return fold(str(row["query"]), row.get("instruction"))


def main() -> int:
    from rcp_ndcg.data.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(description="the fixture rerank reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--recipe", required=True, help="the resolved recipe JSON the harness passed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from deterministic import score

    load_tokenizer(args.tokenizer)  # loaded like the client's own budget resolves it: one tokenizer, everywhere
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.mode == "render":
        rows = []
        for index, row in enumerate(pairs):
            folded = fold_span(row)
            rows.append(
                {
                    "index": index,
                    "shape": "pair",
                    "query": folded,
                    "documents": [str(document) for document in row["documents"]],
                }
            )
        output_result = {"rows": rows}
    else:
        rows = []
        for index, row in enumerate(pairs):
            folded = fold_span(row)
            rows.append({"index": index, "scores": [score(folded, document) for document in row["documents"]]})
        output_result = {"rows": rows}
    Path(args.out).write_text(json.dumps(output_result, indent=1) + "\n", encoding="utf-8")
    return 0


def score(query: str, document: str) -> float:
    """The reference's own scoring: the stub engine's deterministic pair score."""
    from deterministic import score as deterministic_score

    return deterministic_score(query, document)


if __name__ == "__main__":
    raise SystemExit(main())
