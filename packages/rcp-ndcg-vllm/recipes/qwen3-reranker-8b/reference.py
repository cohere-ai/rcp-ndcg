"""Reference implementation for Qwen/Qwen3-Reranker-8B: the paper's exact in-process recipe.

Derives from ``experiments/paper/rerankers/reference/qwen3.py`` (``QwenOGRerank``, the paper's
in-process code, extracted unchanged in behaviour from the former
``src/rcp_ndcg/retrieval/external_rerankers.py``), which reproduces the model card's usage
recipe: a causal LM prompted with the ``<Instruct>/<Query>/<Document>`` chat frame, scored as
``softmax([no_logit, yes_logit])[yes]`` at the final position.

Runs as a subprocess in the reference environment (its own python; the harness imports no
torch), under the harness's reference contract:

    reference.py --mode render|score --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

Modes (the JSON each writes to ``--out``):

- ``render`` (stage 1): for every pairs-file row, the content spans in the harness's rerank
  format -- ``{"index", "shape": "pair", "query": str, "documents": [str, ...]}`` -- under the
  paper's own cut: the pair string right-truncated at ``8192 - 39 - 9 = 8144`` tokens (the
  paper's ``longest_first`` on the pair only; the 39-token system prefix and the 9-token
  assistant suffix always re-attached, so no anchor drops), located at raw character offsets.
  Under the cap the spans are the raw texts; over it the paper's cut differs from the client's
  (which settles an over-share query at its 4096-token share), declared as
  ``reference.known_deviations: [over_cap_cut_differs]``.  It never ports the client's cut.
- ``score`` (stage 2): one probability per document per row, ``softmax([no, yes])[yes]`` -- the
  recipe's ``reference.score_scale: probability``.  The instruction is the paper's fixed
  default (the recipe declares ``instruction: none``); a pairs row's ``instruction`` field is
  ignored, exactly as the served client ignores it.

Reference environment (this file's own python; ``requirements-reference.txt`` beside the
recipe pins it): torch, transformers (the paper ran torch 2.9.1 / transformers 4.57.6),
accelerate, flash-attn on Linux.  ``render`` needs only ``tokenizers`` (and ``huggingface_hub``
for a Hub spec), so stage 1 runs without torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_SEQ_LENGTH = 8192  # the paper's budget (MAX_SEQ_LENGTH of the pre-unified-inference retrieval/cross_encoder.py)
BATCH_SIZE = 4  # the paper's 8B config (experiments/paper/rerankers/qwen3_reranker_8b.yaml)
DTYPE = "bfloat16"  # the paper pipeline's DTYPE (the class default would be float16)

# --- chat-template markers, built without typing them literally ----------------------------
IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/" + "think" + chr(62)

JUDGE_TEXT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

#: The default instruction (model card config_sentence_transformers.json prompts["query"];
#: identical in QwenOGRerank.__init__ and the vLLM template fallback, modulo a trailing period
#: in SGLang's variant).  The paper's runs never vary it.
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

PREFIX = f"{IM_START}system\n{JUDGE_TEXT}{IM_END}\n{IM_START}user\n"
SUFFIX = f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

TOKEN_TRUE_ID = 9693  # "yes" (asserted against the loaded tokenizer at run time)
TOKEN_FALSE_ID = 2152  # "no"


def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    """The pair text, exactly as the model card and QwenOGRerank build it."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _load_tokenizer(spec: str, padding_side: str = "left"):
    """The tokenizer ``spec`` names: a local directory (a fast tokenizer) or ``repo@revision``.

    ``padding_side="left"`` is exactly ``QwenOGRerank.__init__``; it matters for the score
    path's padding, not for the render path's counting.
    """
    from transformers import AutoTokenizer

    if Path(spec).exists():
        return AutoTokenizer.from_pretrained(spec, padding_side=padding_side)
    repo, _, revision = spec.partition("@")
    kwargs = {"padding_side": padding_side}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(repo, **kwargs)


def _raw_tokenizer(spec: str):
    """The same tokenizer for the render path, without importing torch: the ``tokenizers``
    library over the same ``tokenizer.json`` (the same Rust engine the HF fast tokenizer wraps)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(spec)
    if path.is_dir():
        return Tokenizer.from_file(str(path / "tokenizer.json"))
    if path.is_file():
        return Tokenizer.from_file(str(path))
    repo, _, revision = spec.partition("@")
    file = hf_hub_download(repo, "tokenizer.json", revision=revision or None)
    return Tokenizer.from_file(file)


# ---------------------------------------------------------------------------
# render mode (stage 1): the paper's own cut, as the wire's content spans; no torch.
# ---------------------------------------------------------------------------


def kept_pair_spans(tok, query: str, doc: str, budget: int) -> tuple[str, str]:
    """The paper's kept ``(query, document)`` content spans for one pair.

    ``QwenOGRerank._process_inputs`` truncates the pair string ``longest_first`` at ``budget``
    tokens (on a single sequence: a right cut of the pair's ids) and re-attaches the prefix and
    suffix ids, so no anchor is ever dropped.  The kept text is located at the RAW character offset
    of the last kept token -- never a ``decode(encode())`` round trip, which is not the identity
    for this checkpoint (its normalizer maps non-NFC text to NFC, keeping the ids equal but not
    the characters).  The spans are the verbatim prefixes of the query and document pieces of the
    kept pair text; the instruction and the label text belong to the served template's fixed
    frame.  This is the paper's cut, never the client's: the query is not settled at any share.
    """
    header = f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: "
    mid = "\n<Document>: "
    pair = header + query + mid + doc
    encoded = tok.encode(pair, add_special_tokens=True)  # the paper's call; Qwen's post-processor adds none
    kept = pair if len(encoded.ids) <= budget else pair[: encoded.offsets[budget - 1][1]]
    query_start, query_end = len(header), len(header) + len(query)
    document_start = query_end + len(mid)
    query_span = kept[query_start : min(query_end, len(kept))] if len(kept) > query_start else ""
    document_span = kept[document_start:] if len(kept) > document_start else ""
    return query_span, document_span


def render_rows(pairs: list[dict], tokenizer_spec: str) -> list[dict]:
    """The wire's content spans per pairs row under the paper's own cut (``--mode render``).

    The format is the harness's rerank render contract (``{"index", "shape": "pair", "query",
    "documents"}``); the cut is :func:`kept_pair_spans` under the paper's pair budget
    ``MAX_SEQ_LENGTH - len(prefix) - len(suffix)`` (8144 tokens on the pinned tokenizer).  Under
    the cap the spans are the raw query and documents, byte for byte; over it they are the
    paper's kept prefixes, which the recipe declares as ``over_cap_cut_differs`` (the client
    cuts its own way; those rows are reported, not gated).  The paper's right cut lands at one
    pair-token frontier, so the kept query span is the same for every document of a row; a
    disagreement is refused rather than papered over.
    """
    tok = _raw_tokenizer(tokenizer_spec)
    prefix_ids = tok.encode(PREFIX, add_special_tokens=False).ids
    suffix_ids = tok.encode(SUFFIX, add_special_tokens=False).ids
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(suffix_ids)
    rows = []
    for index, row in enumerate(pairs):
        query = str(row["query"])
        documents = [str(document) for document in row["documents"]]
        spans = [kept_pair_spans(tok, query, document, budget) for document in documents]
        query_span = spans[0][0] if spans else kept_pair_spans(tok, query, "", budget)[0]
        if any(span != query_span for span, _ in spans):
            raise SystemExit(f"row {index}: the paper's pair cut kept different query spans for one shared query")
        rows.append(
            {"index": index, "shape": "pair", "query": query_span, "documents": [document for _, document in spans]}
        )
    return rows


# ---------------------------------------------------------------------------
# score mode (stage 2): QwenOGRerank, paper-exact, torch + transformers.
# ---------------------------------------------------------------------------


class QwenOGRerank:
    """The paper's ``QwenOGRerank`` (``experiments/paper/rerankers/reference/qwen3.py``),
    unchanged in behaviour: same prompts, truncation, padding side, dtype, scoring and OOM
    backoff.  The multi-GPU ``AccelState`` sharding the paper's runs did not use is not
    carried (single GPU per model)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_spec: str,
        revision: str | None,
        max_seq_len: int = MAX_SEQ_LENGTH,
        batch_size: int = BATCH_SIZE,
        instruction: str = DEFAULT_INSTRUCTION,
        dtype: str = DTYPE,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.tokenizer = _load_tokenizer(tokenizer_spec)
        self.max_length = max_seq_len
        self.batch_size = batch_size
        self.instruction = instruction

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        assert self.token_true_id == TOKEN_TRUE_ID and self.token_false_id == TOKEN_FALSE_ID, (
            "the checkpoint's yes/no token ids drifted from the pinned recipe (9693/2152)"
        )

        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

        # flash_attention_2 is the paper's default; it exists only on CUDA (the paper scored
        # on GPU).  A CPU run is a smoke path, not a paper path, and takes the HF default.
        attn = "flash_attention_2" if torch.cuda.is_available() else None
        kwargs: dict = {"dtype": dtype, "revision": revision}
        if attn is not None:
            kwargs["attn_implementation"] = attn
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self._torch = torch

    def _process_inputs(self, pairs: list[str]) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_scores(self, inputs: dict) -> list[float]:
        import torch

        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return scores.float().tolist()

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        import torch

        try:
            return self._compute_scores(self._process_inputs(pairs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise
            mid = len(pairs) // 2
            return self._score_pairs(pairs[:mid]) + self._score_pairs(pairs[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        import torch

        pairs = [format_instruction(self.instruction, query, doc) for doc in docs]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                scores.extend(self._score_pairs(pairs[i : i + self.batch_size]))
        if len(scores) != len(docs):
            raise RuntimeError(f"QwenOGRerank: expected {len(docs)} scores, got {len(scores)}")
        return scores


def score_rows(pairs: list[dict], tokenizer_spec: str, device: str) -> list[dict]:
    """One score per document per row, on the paper's probability scale."""
    repo, _, revision = tokenizer_spec.partition("@")
    if Path(tokenizer_spec).exists():
        repo, revision = tokenizer_spec, None  # a local checkout carries no revision split
    reranker = QwenOGRerank(repo, tokenizer_spec=tokenizer_spec, revision=revision, device=device)
    rows = []
    for index, row in enumerate(pairs):
        scores = reranker.predict(str(row["query"]), [str(doc) for doc in row["documents"]])
        rows.append({"index": index, "scores": scores})
    return rows


# ---------------------------------------------------------------------------
# the harness's subprocess CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="the Qwen3-Reranker-8B paper reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        rows = render_rows(pairs, args.tokenizer)
    else:
        rows = score_rows(pairs, args.tokenizer, args.device)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
