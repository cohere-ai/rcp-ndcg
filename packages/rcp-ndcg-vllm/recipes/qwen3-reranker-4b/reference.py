"""Reference implementation for Qwen/Qwen3-Reranker-4B: the paper's exact in-process scorer.

Derived with unchanged behaviour from the paper-exact module
``experiments/paper/rerankers/reference/qwen3.py`` (``QwenOGRerank``, itself the released
Qwen3-Reranker model-card recipe, Apache-2.0) — see that file's docstring for its provenance.
The paper's numbers rest on it; only the harness's subprocess interface
(``--mode render|score --pairs --out --tokenizer --device``) is added, and the
constructor arguments the paper's factory passed are fixed here (max_seq_len 8192 =
``MAX_SEQ_LENGTH``, batch_size 8 = the 4B paper config, bfloat16 = the paper pipeline's
``DTYPE``).

Reference environment (declared per the recipe contract; the harness documents it, never
installs it): the package's ``requirements-reference.txt`` base plus the paper pins in
``experiments/paper/rerankers/reference/requirements.txt`` — torch 2.9.1,
transformers 4.57.6, flash-attn 2.8.3 (the paper's attention backend on CUDA; CPU loads
fall back to the model's default attention, a load-environment accommodation that changes
no behaviour). ``score`` mode needs the weights and torch; ``render`` mode is
tokenizer-only (the fast tokenizer behind the paper's ``AutoTokenizer``, the same
``tokenizer.json`` backend) so stage 1 runs on CPU without weights.

Scoring: the prompt is ``prefix + <Instruct>: {instruction}\\n<Query>: {query}\\n<Document>:
{doc} + suffix`` and the score is ``softmax([no_logit, yes_logit])[yes]`` at the last
position, a probability in [0, 1]. The pair (query and document together) is truncated
``longest_first`` to ``8192 - len(prefix_tokens) - len(suffix_tokens)`` tokens — the pair
only: the instruction and the assistant suffix always survive, and the suffix (the
``im_end``/assistant/think frame the served template re-attaches) is the anchor this
model reads its score from. This is the paper's own cut, NEVER the client's (the rerank
client settles an over-share query at its share where the paper keeps it whole inside the
pair cut), so ``reference.known_deviations`` declares only ``over_cap_cut_differs``: the
harness reports over-cap pairs non-gating and gates under-cap rows exactly. Per-row
instructions in the pairs file are ignored: the paper code scores with its one fixed
default instruction, and the recipe declares ``instruction: none``.

The chat-template markers are composed with ``chr()`` and resolved from the tokenizer's
added tokens where possible, so this file quotes no chat-template special token literally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_SEQ_LENGTH = 8192
"""The paper's pair budget (``MAX_SEQ_LENGTH``, rcp-ndcg's retrieval constant)."""

BATCH_SIZE = 8
"""Documents per forward pass: the 4B paper config (experiments/paper/rerankers/qwen3_reranker_4b.yaml)."""

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


def _prefix() -> str:
    """The fixed prefix: the system turn and the user-turn opener, verbatim from the paper code."""
    return f"{IM_START}system\n{JUDGE_TEXT}{IM_END}\n{IM_START}user\n"


def _suffix() -> str:
    """The fixed suffix: the assistant turn with the empty think block, the model's read-out anchor."""
    return f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"


def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    """The pair text, exactly as ``QwenOGRerank.predict`` builds it (the default instruction folded in)."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _repo_and_revision(spec: str) -> tuple[str, str | None]:
    """A tokenizer spec (``repo@revision`` or a local path) as AutoTokenizer's two arguments."""
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json"):
        return spec, None
    repo, _, revision = spec.partition("@")
    return repo, revision or None


def _load_fast_tokenizer(spec: str):
    """The fast tokenizer behind the paper's ``AutoTokenizer``, from the same tokenizer.json.

    ``render`` mode is tokenizer-only: it runs in the reference environment without weights
    and without transformers, on the same backend ``AutoTokenizer`` wraps for this checkpoint
    (the ids and the decode are the backend's own).
    """
    if spec.endswith(".json") or spec.startswith(("/", "./", "../", "~")):
        from tokenizers import Tokenizer

        path = Path(spec).expanduser()
        return Tokenizer.from_str((path / "tokenizer.json" if path.is_dir() else path).read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    repo, _, revision = spec.partition("@")
    return hf_hub_download(repo, "tokenizer.json", revision=revision or None)


def _fast_from_spec(spec: str):
    """The ``tokenizers.Tokenizer`` for the spec, cached per process (render mode)."""
    global _FAST_TOKENIZER
    if _FAST_TOKENIZER is None:
        from tokenizers import Tokenizer

        loaded = _load_fast_tokenizer(spec)
        _FAST_TOKENIZER = loaded if isinstance(loaded, Tokenizer) else Tokenizer.from_file(loaded)
    return _FAST_TOKENIZER


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
    and are not part of the spans.
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
    tok = _fast_from_spec(tokenizer_spec)
    prefix_ids = tok.encode(_prefix(), add_special_tokens=False).ids
    suffix_ids = tok.encode(_suffix(), add_special_tokens=False).ids
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(suffix_ids)
    out: list[dict] = []
    for index, row in enumerate(rows):
        query = str(row.get("query", ""))
        documents = [str(document) for document in row.get("documents", [])]
        pairs = [kept_pair_spans(tok, query, document, budget) for document in documents]
        query_span = pairs[0][0] if pairs else kept_pair_spans(tok, query, "", budget)[0]
        if any(span != query_span for span, _ in pairs):
            raise RuntimeError(
                "the paper's pair cut kept different query spans for one shared query; "
                "stage 1 renders one query span per row"
            )
        out.append(
            {
                "index": index,
                "shape": str(row.get("shape") or "pair"),
                "query": query_span,
                "documents": [document for _, document in pairs],
            }
        )
    return out


def score_rows(rows: list[dict], tokenizer_spec: str, device: str) -> list[list[float]]:
    """One yes-probability per document per row (``--mode score``), the paper's ``_score_pairs``.

    Loads the checkpoint exactly as the paper code does (``AutoTokenizer`` with
    ``padding_side="left"``, ``AutoModelForCausalLM`` at the paper's bfloat16, flash
    attention 2 on CUDA), batches at the paper's batch_size, pads left to the batch max and
    reads ``log_softmax([no, yes])[yes].exp()`` at the last position.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The tokenizer spec is the recipe's <repo>@<revision>; the checkpoint loads from the same
    # repo at the same revision (the paper code's model_name_or_path + revision constructor args).
    repo, revision = _repo_and_revision(tokenizer_spec)
    tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left", revision=revision)
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    prefix_ids = tokenizer.encode(_prefix(), add_special_tokens=False)
    suffix_ids = tokenizer.encode(_suffix(), add_special_tokens=False)
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(suffix_ids)

    attn = "flash_attention_2" if str(device).startswith("cuda") else None
    model_kwargs: dict = {"dtype": torch.bfloat16, "revision": revision}
    if attn is not None:
        model_kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
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

    scores: list[list[float]] = []
    with torch.no_grad():
        for row in rows:
            # The instruction is the reference's fixed default; row instructions are not read.
            pairs = [format_instruction(None, str(row["query"]), str(document)) for document in row["documents"]]
            flat: list[float] = []
            for start in range(0, len(pairs), BATCH_SIZE):
                flat.extend(score_batch(pairs[start : start + BATCH_SIZE]))
            if len(flat) != len(row["documents"]):
                raise RuntimeError(f"expected {len(row['documents'])} scores, got {len(flat)}")
            scores.append(flat)
    return scores


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
    """The harness's reference CLI: ``--mode render|score --pairs --out --tokenizer [--device]``."""
    parser = argparse.ArgumentParser(description="the qwen3-reranker-4b reference (the paper's QwenOGRerank)")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        output = {"rows": render_rows(rows, args.tokenizer)}
    else:
        scores = score_rows(rows, args.tokenizer, args.device)
        rows_out = [{"index": index, "scores": scores} for index, scores in enumerate(scores)]
        output = {"rows": rows_out}
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
