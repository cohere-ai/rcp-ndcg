"""The zerank-1-reranker reference: the paper's exact in-process implementation, run as a subprocess.

Derived from ``experiments/paper/rerankers/reference/zerank.py`` (the ``ZerankRerank`` class the
package carried when the paper's runs were made, itself moved unchanged out of
``src/rcp_ndcg/retrieval/external_rerankers.py``). The scoring behaviour is unchanged: per pair, the
chat template of the checkpoint's own tokenizer renders ``[{system: query.strip()},
{user: doc.strip()}]`` with ``add_generation_prompt=True``, the whole rendered prompt is tokenized
with ``truncation=True, max_length=8192`` (right side), the logit of "Yes" (single token 9454) at the
last non-pad position is divided by 5 and sigmoided (``scipy.special.expit``): probabilities in
(0, 1). Batching (length-descending permutation, 15,000-character buckets, right padding) and the
OOM backoff are the paper's; per-pair scores are invariant to both (causal attention, masked right
padding, the last non-pad position).

Two harness modes (see ``rcp_ndcg_vllm.equivalence.reference`` for the contract):

- ``--mode render`` (stage 1; tokenizer only, no torch): ``{"rows": [{"index", "shape": "pair",
  "query": str, "documents": [str, ...]}]}`` -- the content spans in the harness's rerank format,
  under the PAPER's cut: the stripped query and document as the paper's prompt carries them,
  right-cut with the whole rendered prompt at 8192 tokens (raw character offsets, never a
  ``decode(encode())`` round trip). Under the cap the spans equal the client's byte for byte
  (stage 1 gates them exactly); over it the paper's cut drops the anchor, the recipe's declared
  ``anchor_drop_over_cap``, and the harness reports those rows non-gating. The reference never
  ports the client's cut (its query share, its settle rule, its anchor reservation).
- ``--mode score`` (stage 2): the paper's ``predict`` per row, on the recipe's
  ``reference.score_scale`` (probability). The recipe takes no instruction (``instruction: none``): a
  row's ``instruction`` field is ignored, exactly as the paper's path ignored it.

The anchor drop is never copied anywhere: the paper's whole-prompt right cut at 8192 tokens drops
the trailing assistant header (the last-token anchor) on over-cap pairs; the recipe declares that as
``reference.known_deviations: [anchor_drop_over_cap]`` and stage 2 gates under-cap pairs only.

Runs in its OWN python (``--reference-python``), never inside the harness: its environment is pinned
by ``requirements-reference.txt`` beside this file (render mode needs only ``tokenizers`` and
``huggingface_hub``; score mode adds torch and transformers for the paper's code). The model is
loaded from the recipe's ``model`` at the recipe's ``revision`` (read from the ``recipe.yaml`` beside
this file), bfloat16, on ``--device``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "zeroentropy/zerank-1-reranker"
REVISION = "d03c467e29e29c0a16a130a86ce3b62d30116a2c"
MAX_SEQ_LENGTH = 8192
"""The paper's whole-prompt token cap and the recipe's ``client.max_tokens`` pair budget."""
BATCH_SIZE_TOKENS = 15_000
"""The paper's character budget per forward batch (throughput only; scores are per pair)."""
YES_TOKEN_ID = 9454
"""The "Yes" logit the score head reads: a single token of the pinned tokenizer (measured)."""
ADD_SPECIAL_TOKENS = True
"""The scoring route tokenizes with the post-processor's tokens (measured a no-op for this
tokenizer, so the frame is the whole fixed cost either way). Declared, not assumed."""
TOKENIZER_FILE = "tokenizer.json"


def _recipe_dir() -> Path:
    """The directory this reference and its ``recipe.yaml`` live in."""
    return Path(__file__).resolve().parent


def _model_spec() -> tuple[str, str]:
    """The recipe's ``(model, revision)`` - the same checkpoint the serve block pins."""
    import yaml

    recipe = yaml.safe_load((_recipe_dir() / "recipe.yaml").read_text(encoding="utf-8"))
    return str(recipe["model"]), str(recipe["revision"])


def _tokenizer_source(spec: str) -> tuple[str, str | None]:
    """A tokenizer spec into (source, revision) for ``AutoTokenizer``: a repo id splits at ``@``; a local
    ``tokenizer.json`` file becomes its directory (transformers loads a directory, not a bare file)."""
    path = Path(spec).expanduser()
    if path.exists():
        return (str(path.parent) if path.is_file() else str(path)), None
    repo, _, revision = spec.partition("@")
    return repo, revision or None


def load(tokenizer_spec: str, device: str):
    """The paper's load: right-padded tokenizer (pad -> eos fallback) and the causal LM in bfloat16,
    moved to ``device``. Returns ``(tokenizer, model, yes_token_id, device)`` for score mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source, revision = _tokenizer_source(tokenizer_spec)
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="right", revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_name, model_revision = _model_spec()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, revision=model_revision)
    model.eval()
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    assert yes_token_id == YES_TOKEN_ID, yes_token_id
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, yes_token_id, device


def score_pair(model, tokenizer, yes_token_id: int, device: str, query: str, documents: list[str]) -> list[float]:
    """The paper's ``predict`` for one query and its documents (probabilities, aligned with ``docs``).

    The prompt per pair is the paper's ``ZerankRerank._format_inputs``, unchanged: the query and the
    document are stripped (the declared normalisation), the checkpoint's own chat template renders
    them as the system and user turns, and the generation prompt (the assistant header) is appended.
    """
    from scipy.special import expit

    texts: list[str] = []
    for document in documents:
        messages = [
            {"role": "system", "content": query.strip()},
            {"role": "user", "content": document.strip()},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Length-descending permutation for tight token buckets (the paper's batch composition).
    permutation = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    sorted_texts = [texts[i] for i in permutation]

    batches: list[list[str]] = []
    max_length = 0
    for text in sorted_texts:
        text_len = len(text)
        if not batches or (len(batches[-1]) + 1) * max(max_length, text_len) > BATCH_SIZE_TOKENS:
            batches.append([])
            max_length = 0
        batches[-1].append(text)
        max_length = max(max_length, text_len)

    all_logits: list[float] = []
    for batch in batches:
        all_logits.extend(_batch_logits(model, tokenizer, yes_token_id, device, batch))

    scores = expit(all_logits).tolist()
    result = [0.0] * len(documents)
    for orig_idx, score in zip(permutation, scores, strict=True):
        result[orig_idx] = score
    return result


def _batch_logits(model, tokenizer, yes_token_id: int, device: str, batch: list[str]) -> list[float]:
    """The paper's ``_batch_logits``: right-padded whole-prompt tokenization capped at 8192, the
    "Yes" logit at the last non-pad position, divided by 5; the OOM backoff halves the batch."""
    import torch

    try:
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, use_cache=False)
            attention_mask = inputs["attention_mask"]
            last_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(outputs.logits.shape[0], device=device)
            last_logits = outputs.logits[batch_indices, last_positions]
            yes_logits = last_logits[:, yes_token_id]
            return [float(value) / 5.0 for value in yes_logits]
    except torch.cuda.OutOfMemoryError:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        return _batch_logits(model, tokenizer, yes_token_id, device, batch[:mid]) + _batch_logits(
            model, tokenizer, yes_token_id, device, batch[mid:]
        )


# -------------------------------------------------------------------------------------------
# The tokenizer and the frame for render mode: the `tokenizers` library over the recipe's
# tokenizer.json (the same file the harness and the engine read); specials resolved by name.
# -------------------------------------------------------------------------------------------


def _split_spec(spec: str) -> tuple[str, str | None]:
    """``org/model@<revision>`` -> (repo, revision); a local path passes through with no revision.

    The same resolution rule as the product's ``rcp_ndcg.data.tokenizer`` path handling: a path is
    anything that exists, is absolute or relative, or ends in ``.json``.
    """
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json") or Path(spec).exists():
        return spec, None
    repo, _, revision = spec.partition("@")
    return repo, revision or None


def _tokenizer_file(spec: str) -> Path:
    """The tokenizer.json file ``spec`` names: a local path as given, else downloaded from the Hub."""
    path, revision = _split_spec(spec)
    candidate = Path(path).expanduser()
    file = candidate / TOKENIZER_FILE if candidate.is_dir() else candidate
    if file.is_file():
        return file
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(path, TOKENIZER_FILE, revision=revision))


def _load_backend_tokenizer(spec: str):
    """The reference's counting tokenizer: the `tokenizers` library over the recipe's tokenizer.json."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_tokenizer_file(spec)))


def _added_tokens(tokenizer) -> dict[str, str]:
    """The added vocabulary: the special's bare name (and its literal form) to its literal text."""
    tokens: dict[str, str] = {}
    for token in tokenizer.get_added_tokens_decoder().values():
        content = token.content
        name = content[2:-2] if content.startswith("<|") and content.endswith("|>") else content
        tokens.setdefault(name, content)
        tokens.setdefault(content, content)
    return tokens


def _special_text(tokenizer, name: str) -> str:
    """The literal text of the added token named ``name`` (as the recipe's ``{special:<name>}`` resolves)."""
    tokens = _added_tokens(tokenizer)
    try:
        return tokens[name]
    except KeyError:
        known = sorted({key for key in tokens if not key.startswith("<|")})
        raise SystemExit(f"the tokenizer has no added token named {name!r}; its added tokens are {known}") from None


def _frame(tokenizer) -> tuple[str, str, str]:
    """The pair frame: (head, middle, tail) around the query and document spans -- the paper's prompt
    (the checkpoint's chat template with the generation prompt, measured equal), the recipe's
    declared pair shape and the template file's frame. Specials resolved from the tokenizer."""
    im_start = _special_text(tokenizer, "im_start")
    im_end = _special_text(tokenizer, "im_end")
    head = f"{im_start}system\n"
    mid = f"{im_end}\n{im_start}user\n"
    tail = f"{im_end}\n{im_start}assistant\n"
    return head, mid, tail


# -------------------------------------------------------------------------------------------
# render mode (stage 1): the paper's own cut, written as the content spans the harness compares.
# The format is the harness's rerank render contract; the cut is the paper's ZerankRerank, never
# the client's (no query share, no settle rule, no anchor reservation).
# -------------------------------------------------------------------------------------------


def paper_spans(tokenizer, query: str, document: str) -> tuple[str, str]:
    """The paper's kept ``(query, document)`` content spans for one pair.

    The paper's ``ZerankRerank._format_inputs`` renders ``[{system: query.strip()}, {user:
    doc.strip()}]`` through the checkpoint's chat template with the generation prompt (that render
    is exactly this recipe's frame, ``head + query + mid + document + tail``: measured on the
    pinned checkpoint's own template), and ``_batch_logits`` tokenizes the WHOLE prompt with
    ``truncation=True, max_length=MAX_SEQ_LENGTH`` -- a right cut.  The kept text is located at the raw
    character offset of the last kept token (never a ``decode(encode())`` round trip); the spans
    are the kept parts of the stripped query and document.  Under the cap they are the stripped
    texts, byte for byte; over it the cut drops the tail first -- the assistant header, the anchor
    the score is read from -- then the document's end (and, for a query that fills the cap alone,
    the query's): the recipe's declared ``anchor_drop_over_cap``.
    """
    head, mid, tail = _frame(tokenizer)
    query, document = query.strip(), document.strip()
    prompt = head + query + mid + document + tail
    encoded = tokenizer.encode(prompt, add_special_tokens=ADD_SPECIAL_TOKENS)
    if len(encoded.ids) <= MAX_SEQ_LENGTH:
        return query, document
    kept = prompt[: encoded.offsets[MAX_SEQ_LENGTH - 1][1]]
    query_start = len(head)
    document_start = query_start + len(query) + len(mid)
    return kept[query_start : query_start + len(query)], kept[document_start : document_start + len(document)]


def render_rows(tokenizer, pairs: list[dict]) -> list[dict]:
    """``--mode render``: per pairs row, ``{"index", "shape": "pair", "query", "documents"}`` under
    the paper's cut (:func:`paper_spans`).  The recipe declares ``instruction: none``: a row's
    instruction is not part of this model's input.  The paper's right cut lands at one token
    frontier of the shared query, so the kept query span is the same for every document of a row;
    a disagreement is refused rather than papered over."""
    rows = []
    for index, row in enumerate(pairs):
        shape = str(row.get("shape") or "pair")
        if shape != "pair":
            raise SystemExit(f"row {index}: shape {shape!r} is not declared; this recipe declares the pair shape")
        query = str(row["query"])
        spans = [paper_spans(tokenizer, query, str(document)) for document in row["documents"]]
        query_span = spans[0][0] if spans else paper_spans(tokenizer, query, "")[0]
        if any(span != query_span for span, _ in spans):
            raise SystemExit(f"row {index}: the paper's cut kept different query spans for one shared query")
        rows.append({"index": index, "shape": shape, "query": query_span, "documents": [d for _, d in spans]})
    return rows


def main() -> int:
    """The reference CLI: ``--mode render|score``, a pairs file in, the mode's JSON out."""
    parser = argparse.ArgumentParser(description="the zerank-1-reranker reference (the paper's implementation)")
    parser.add_argument("--mode", required=True, choices=["render", "score", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision or a path)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "embed":
        raise SystemExit("zerank-1-reranker is a reranker: the reference speaks render and score, not embed")

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.mode == "score":
        tokenizer, model, yes_token_id, device = load(args.tokenizer, args.device)
        rows = []
        for index, row in enumerate(pairs):
            scores = score_pair(model, tokenizer, yes_token_id, device, str(row["query"]), list(row["documents"]))
            rows.append({"index": index, "scores": scores})
        del model  # release the weights while the engine on the slot keeps serving
        output = {"rows": rows}
    else:
        tokenizer = _load_backend_tokenizer(args.tokenizer)
        rows = render_rows(tokenizer, pairs)
        output = {"rows": rows}

    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
