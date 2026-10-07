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

- ``--mode render`` (stage 1): ``{"rows": [{"index", "shape", "pair", "query": str,
  "documents": [str, ...]}]}`` -- the cut content spans the served wire carries (the rerank reference
  contract; the frame is the engine's own template). Implemented as an offset-based port of the
  product's settle rule for this recipe's frame (the query settles at its declared share whenever it
  exceeds it, then through fit's probe pair; each document gets what remains; every content span is
  cut at a token boundary as a verbatim prefix -- never a decode(encode()) round trip), because the
  reference environment holds no rcp-ndcg and the render check compares two implementations. The
  recipe tests pin the spans byte-identical to the product's own fit on sampled pairs. The declared
  normalisation (``normalize: [strip]``, the paper's ``query.strip()``/``doc.strip()``) applies to the
  content spans before anything is measured, exactly as fit applies it.
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
QUERY_MAX_TOKENS = 4096
"""The query's declared share (the recipe's ``client.query_max_tokens``): the client settles the
shared query at this share whenever the query exceeds it."""
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
# The wire's content spans: an offset-based port of the product's settle rule for this recipe's
# frame (the reference environment holds no rcp-ndcg, and the render check compares two
# implementations of one policy).  Pipeline, exactly the merged rerank client's: (1) the shared
# query settles ONCE per request, at its declared share whenever the raw query exceeds it (the
# share's count reads the raw text, as the client's count does), (2) the declared normalisation
# (``normalize: [strip]``, the paper's strip) runs on the spans before anything is measured (fit
# applies it), (3) the settled query keeps room for the frame even with an empty document (fit's
# probe pair), and (4) each document gets what remains.  Cuts are verbatim prefixes located at
# token boundaries via the tokenizer's offset mapping.  The recipe tests pin these spans
# byte-identical to the product's own fit.
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


def _count(text: str, tokenizer, *, add_special_tokens: bool = False) -> int:
    """The number of tokens of ``text`` as the engine counts it (the post-processor included when
    ``add_special_tokens``)."""
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


def _offsets(text: str, tokenizer) -> list[tuple[int, int]]:
    """``(start, end)`` character offsets of each token of ``text``, in order."""
    return [(offset[0], offset[1]) for offset in tokenizer.encode(text, add_special_tokens=False).offsets]


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
    """The served pair frame: (head, middle, tail) around the query and document spans, the recipe's
    declared pair shape -- the template file's frame. Specials resolved from the tokenizer."""
    im_start = _special_text(tokenizer, "im_start")
    im_end = _special_text(tokenizer, "im_end")
    head = f"{im_start}system\n"
    mid = f"{im_end}\n{im_start}user\n"
    tail = f"{im_end}\n{im_start}assistant\n"
    return head, mid, tail


def _token_prefix(
    text: str,
    max_tokens: int,
    tokenizer,
    *,
    rendered=None,
    add_special_tokens: bool = False,
) -> str:
    """The longest prefix of ``text`` that ends at one of its first ``max_tokens`` token boundaries
    and counts at most ``max_tokens`` as the engine reads it (``rendered(piece)`` when given).

    The cut is located with the tokenizer's offset mapping on the original text, so the result is a
    verbatim prefix of ``text``; a candidate is counted as the engine reads it (the assembled
    render), because a cut word can re-tokenize longer. Port of
    ``rcp_ndcg.data.preprocess.token_prefix``: the same galloping-then-binary search over token
    boundaries, the same counting.
    """

    def count(piece: str) -> int:
        shown = rendered(piece) if rendered is not None else piece
        return _count(shown, tokenizer, add_special_tokens=add_special_tokens)

    if count(text) <= max_tokens:
        return text
    offsets = _offsets(text, tokenizer)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return count(prefix(tokens)) <= max_tokens

    over = min(max_tokens, len(offsets))
    if fits(over):
        return prefix(over)
    fitting, step = over - 1, 1
    while fitting > 0 and not fits(fitting):
        over, fitting, step = fitting, max(fitting - step, 0), step * 2
    while over - fitting > 1:
        middle = (over + fitting) // 2
        fitting, over = (middle, over) if fits(middle) else (fitting, middle)
    return prefix(fitting)


def served_spans(tokenizer, query: str, document: str) -> tuple[str, str]:
    """The cut content spans (query, document) the served wire carries for one pair.

    Port of the product's role client for this recipe's declaration. The query settles once per
    request -- to its declared share (``QUERY_MAX_TOKENS``) whenever the RAW query exceeds it, then
    through fit's probe pair (the query with an empty document) -- and each document gets what
    remains. Both spans carry the declared normalisation (``strip``; measured against the raw input
    first, as fit does). Every cut is a verbatim prefix at a token boundary of the content; the
    frame and its trailing anchor are re-assembled by the engine around these spans. A query that
    fills the budget and leaves the document nothing raises, never cut undeclared.
    """
    head, mid, tail = _frame(tokenizer)

    def assemble(q: str, d: str) -> str:
        return head + q + mid + d + tail

    cap = MAX_SEQ_LENGTH
    overhead = _count(assemble("", ""), tokenizer, add_special_tokens=ADD_SPECIAL_TOKENS)
    if overhead > cap:
        raise SystemExit(f"the frame's fixed overhead alone is {overhead} tokens, over the budget of {cap}")
    # 1. the client settles the shared query at its declared share whenever the query exceeds it:
    q_final = query
    if _count(q_final, tokenizer) > QUERY_MAX_TOKENS:
        q_final = _token_prefix(q_final, QUERY_MAX_TOKENS, tokenizer)
    # 2. the declared normalisation (fit strips the content spans before it measures anything):
    q_final = q_final.strip()
    d_final = document.strip()
    # 3. fit's probe pair (the query with an empty document): the query keeps the frame room.
    q_final = _token_prefix(
        q_final, cap, tokenizer, rendered=lambda piece: assemble(piece, ""), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    q_min = _count(assemble(q_final, ""), tokenizer, add_special_tokens=ADD_SPECIAL_TOKENS)
    if q_min >= cap and _count(d_final, tokenizer) > 0:
        raise SystemExit(
            f"the query fills the pair budget of {cap} tokens and leaves the document nothing; "
            "lower QUERY_MAX_TOKENS (or raise MAX_SEQ_LENGTH), so the document keeps a share"
        )
    # 4. the document gets what remains after the settled query and the frame:
    d_final = _token_prefix(
        d_final, cap, tokenizer, rendered=lambda piece: assemble(q_final, piece), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    return q_final, d_final


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
        rows = []
        for index, row in enumerate(pairs):
            shape = str(row.get("shape") or "pair")
            if shape != "pair":
                raise SystemExit(f"row {index}: shape {shape!r} is not declared; this recipe declares the pair shape")
            # The recipe declares instruction: none: the instruction (if a row carries one) is not
            # part of this model's input, on the served path or here. The settled query spans one
            # row's requests; each document span is cut to what remains after it.
            query_span, _ = served_spans(tokenizer, str(row["query"]), str(row["documents"][0]))
            document_spans = []
            for document in row["documents"]:
                _, document_span = served_spans(tokenizer, str(row["query"]), str(document))
                document_spans.append(document_span)
            rows.append({"index": index, "shape": shape, "query": query_span, "documents": document_spans})
        output = {"rows": rows}

    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
