"""The Qwen3-Reranker family's one reference (the paper's exact in-process scorer, per variant).

Derived with unchanged behaviour from the paper-exact module
``experiments/paper/rerankers/reference/qwen3.py`` (``QwenOGRerank``, itself the released
Qwen3-Reranker model-card recipe, Apache-2.0) — the paper's numbers rest on it. The harness's
subprocess interface is added, and the constructor arguments the paper's factory passed are fixed
here (max_seq_len 8192 = ``MAX_SEQ_LENGTH``, batch_size per variant from the paper configs, bfloat16
= the paper pipeline's ``DTYPE``). One file serves every variant of the family (decision 34): the
variant travels with the invocation, in the resolved recipe the harness passes.

Run as the harness's reference subprocess (never imported by the harness, which holds no torch)::

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <spec> \
        --recipe <resolved-recipe.json> [--device <d>]

``--recipe`` is the resolved recipe the harness loaded (a JSON dump of the family-expanded
``Recipe``): the variant's id selects the paper batch size (:data:`BATCH_SIZES` — a variant outside
the table is refused, never defaulted), and its ``model``/``revision`` pin the checkpoint the score
mode loads.  ``--tokenizer`` is the recipe's own ``client.tokenizer`` spec (the same checkpoint's
tokenizer at the same revision).

Scoring: the prompt is ``prefix + <Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}
+ suffix`` and the score is ``softmax([no_logit, yes_logit])[yes]`` at the last position, a
probability in [0, 1]. The pair (query and document together) is truncated ``longest_first`` to
``8192 - len(prefix_tokens) - len(suffix_tokens)`` tokens — the pair only: the instruction and the
assistant suffix always survive, and the suffix (the ``im_end``/assistant/think frame the served
template re-attaches) is the anchor this model reads its score from. This is the paper's own cut,
NEVER the client's (the rerank client settles an over-share query at its share where the paper keeps
it whole inside the pair cut), so ``reference.known_deviations`` declares only
``over_cap_cut_differs``: the harness reports over-cap pairs non-gating and gates under-cap rows
exactly. Per-row instructions in the pairs file are ignored: the paper code scores with its one
fixed default instruction, and the recipe declares ``instruction: none``.

``render`` mode is tokenizer-only (the ``tokenizers`` library over the same ``tokenizer.json``; the
ids and the character offsets are the backend's own) so stage 1 runs on CPU without weights. The
kept text is cut at the RAW character offsets of the last kept token — never a ``decode(encode())``
round trip: this checkpoint's normalizer maps non-NFC text to NFC, keeping the token ids equal but
not the characters.

The chat-template markers are composed with ``chr()``, so this file quotes no chat-template special
token literally. Reference environment: ``requirements-reference.txt`` beside this file (the paper
pins of ``experiments/paper/rerankers/reference/requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_SEQ_LENGTH = 8192
"""The paper's pair budget (``MAX_SEQ_LENGTH``, the pre-unified-inference retrieval constant)."""

BATCH_SIZES = {
    "qwen3-reranker-0.6b": 16,
    "qwen3-reranker-4b": 8,
    "qwen3-reranker-8b": 4,
}
"""Documents per forward pass, per variant: the paper configs
(experiments/paper/rerankers/qwen3_reranker_{0_6b,4b,8b}.yaml). Throughput only — per-pair scores are
invariant to the batch composition (causal attention, left padding, the last position's logits) — but
the paper's per-model value is kept so stage 2 runs the paper's own numbers."""

DTYPE_NAME = "bfloat16"
"""The paper pipeline's dtype; the class default would be float16 (research report section 3)."""

JUDGE_TEXT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# --- chat-template markers, composed, never typed literally -----------------
IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/" + "think" + chr(62)

PREFIX = f"{IM_START}system\n{JUDGE_TEXT}{IM_END}\n{IM_START}user\n"
SUFFIX = f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

TOKEN_TRUE_ID = 9693  # "yes"
TOKEN_FALSE_ID = 2152  # "no"


def _load_recipe(path: str) -> dict:
    """The resolved recipe the harness passed (``--recipe``): the variant's facts and its batch."""
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    variant_id = recipe["id"]
    if variant_id not in BATCH_SIZES:
        raise SystemExit(
            f"recipe {variant_id!r} is not in this family reference's BATCH_SIZES "
            f"({', '.join(sorted(BATCH_SIZES))}): add the variant's paper batch size deliberately, never a default"
        )
    return recipe


def _fast_from_spec(spec: str):
    """The ``tokenizers.Tokenizer`` for the spec, from the same ``tokenizer.json`` the engine reads."""
    from tokenizers import Tokenizer

    if spec.endswith(".json") or spec.startswith(("/", "./", "../", "~")):
        path = Path(spec).expanduser()
        return Tokenizer.from_str((path / "tokenizer.json" if path.is_dir() else path).read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    repo, _, revision = spec.partition("@")
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=revision or None))


_FAST_TOKENIZER = None
"""The render-mode tokenizer, loaded once per process."""


def kept_pair_spans(tok, query: str, doc: str, budget: int) -> tuple[str, str]:
    """The paper's kept ``(query, document)`` content spans for one pair.

    The pair string is truncated ``longest_first`` at ``budget`` tokens (the paper's
    ``_process_inputs`` on a single sequence right-cuts it) at the RAW character offsets of
    the last kept token — never a ``decode(encode())`` round trip: this checkpoint's
    normalizer maps non-NFC text to NFC, keeping the token ids equal but not the characters.
    The kept spans are the verbatim prefixes of the query and document pieces of the kept
    pair text — the instruction and the label text live in the served template's fixed frame
    and are not part of the spans. This is the paper's cut, never the client's: the query is
    not settled at any share.
    """
    header = f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: "
    mid = "\n<Document>: "
    pair = header + query + mid + doc
    encoded = tok.encode(pair, add_special_tokens=True)
    kept = pair if len(encoded.ids) <= budget else pair[: encoded.offsets[budget - 1][1]]
    query_start, query_end = len(header), len(header) + len(query)
    document_start = query_end + len(mid)
    query_span = kept[query_start : min(query_end, len(kept))] if len(kept) > query_start else ""
    document_span = kept[document_start:] if len(kept) > document_start else ""
    return query_span, document_span


def render_rows(rows: list[dict], tokenizer_spec: str) -> list[dict]:
    """The wire's cut content spans per pairs row (``--mode render``): the paper's own cut.

    For every row: one query span (identical for the row's pairs — the paper's right cut
    keeps the same query prefix for each of them) and one document span per document, through
    :func:`kept_pair_spans` under the pair budget ``8192 - len(prefix) - len(suffix)``. The
    spans are raw character prefixes, so a non-NFC input compares byte-identical to the
    client's shipped spans (which are cut the same verbatim way).
    """
    global _FAST_TOKENIZER
    if _FAST_TOKENIZER is None:
        _FAST_TOKENIZER = _fast_from_spec(tokenizer_spec)
    tok = _FAST_TOKENIZER
    prefix_ids = tok.encode(PREFIX, add_special_tokens=False).ids
    suffix_ids = tok.encode(SUFFIX, add_special_tokens=False).ids
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(suffix_ids)
    out: list[dict] = []
    for index, row in enumerate(rows):
        query = str(row["query"])
        documents = [str(document) for document in row["documents"]]
        spans = [kept_pair_spans(tok, query, document, budget) for document in documents]
        query_span = spans[0][0] if spans else kept_pair_spans(tok, query, "", budget)[0]
        if any(span != query_span for span, _ in spans):
            raise SystemExit(f"row {index}: the paper's pair cut kept different query spans for one shared query")
        out.append(
            {
                "index": index,
                "shape": str(row.get("shape") or "pair"),
                "query": query_span,
                "documents": [document for _, document in spans],
            }
        )
    return out


def score_rows(rows: list[dict], recipe: dict, tokenizer_spec: str, device: str) -> list[dict]:
    """One yes-probability per document per row (``--mode score``), the paper's ``QwenOGRerank.predict``.

    Loads the checkpoint exactly as the paper code does (``AutoTokenizer`` with
    ``padding_side="left"``, ``AutoModelForCausalLM`` at the paper's bfloat16, flash
    attention 2 on CUDA), batches at the variant's paper batch size, pads left to the batch
    max and reads ``log_softmax([no, yes])[yes].exp()`` at the last position.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The checkpoint loads from the resolved recipe's model at its revision (the paper code's
    # model_name_or_path + revision constructor args, now carried by the family-expanded recipe).
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(tokenizer_spec), padding_side="left", revision=_tokenizer_revision(tokenizer_spec)
    )
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    assert token_true_id == TOKEN_TRUE_ID and token_false_id == TOKEN_FALSE_ID, (
        f"the checkpoint's yes/no token ids drifted from the pinned recipe ({TOKEN_FALSE_ID}/{TOKEN_TRUE_ID})"
    )

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(suffix_ids)
    batch_size = BATCH_SIZES[recipe["id"]]

    attn = "flash_attention_2" if str(device).startswith("cuda") else None
    model_kwargs: dict = {"dtype": torch.bfloat16, "revision": recipe["revision"]}
    if attn is not None:
        model_kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(recipe["model"], **model_kwargs)
    model.eval()
    model.to(device)

    def score_batch(pairs: list[str]) -> list[float]:
        """One batch of pairs, the paper's ``_process_inputs`` + ``_compute_scores`` verbatim."""
        try:
            return _compute_scores(
                model, tokenizer, token_false_id, token_true_id, pairs, prefix_ids, suffix_ids, budget
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise
            mid = len(pairs) // 2
            return score_batch(pairs[:mid]) + score_batch(pairs[mid:])

    rows_out: list[dict] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            # The instruction is the reference's fixed default; row instructions are not read.
            pairs = [format_instruction(None, str(row["query"]), str(document)) for document in row["documents"]]
            flat: list[float] = []
            for start in range(0, len(pairs), batch_size):
                flat.extend(score_batch(pairs[start : start + batch_size]))
            if len(flat) != len(row["documents"]):
                raise RuntimeError(f"expected {len(row['documents'])} scores, got {len(flat)}")
            rows_out.append({"index": index, "scores": flat})
    return rows_out


def _tokenizer_source(spec: str) -> str:
    """The tokenizer spec as AutoTokenizer's repository argument (``repo@revision`` or a local path)."""
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json"):
        return spec
    return spec.partition("@")[0]


def _tokenizer_revision(spec: str) -> str | None:
    """The tokenizer spec's ``@revision``, or ``None`` for a local path."""
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json"):
        return None
    return spec.partition("@")[2] or None


def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    """The pair text, exactly as ``QwenOGRerank.predict`` builds it (the default instruction folded in)."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _compute_scores(
    model,
    tokenizer,
    token_false_id: int,
    token_true_id: int,
    pairs: list[str],
    prefix_ids: list[int],
    suffix_ids: list[int],
    budget: int,
) -> list[float]:
    """The paper's ``_process_inputs`` + ``_compute_scores``, unchanged in behaviour."""
    import torch

    inputs = tokenizer(pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=budget)
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_ids + ele + suffix_ids
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits[:, -1, :]
    true_vector = logits[:, token_true_id]
    false_vector = logits[:, token_false_id]
    stacked = torch.stack([false_vector, true_vector], dim=1)
    scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
    return scores.float().tolist()


def main() -> int:
    """The harness's reference CLI: ``--mode render|score --pairs --out --tokenizer --recipe [--device]``."""
    parser = argparse.ArgumentParser(description="the qwen3-reranker family reference (the paper's QwenOGRerank)")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    recipe = _load_recipe(args.recipe)
    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        output = {"rows": render_rows(rows, args.tokenizer)}
    else:
        output = {"rows": score_rows(rows, recipe, args.tokenizer, args.device)}
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
