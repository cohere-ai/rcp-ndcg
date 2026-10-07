"""Reference implementation for zeroentropy/zerank-1-small-reranker (== zeroentropy/zerank-1-small, one repository).

Ported unchanged in behaviour from the paper's in-process code, the ``ZerankRerank`` framework
(``experiments/paper/rerankers/reference/zerank.py`` on the unified-inference line -- this recipe's
base still carries the same code as ``ZerankRerank`` in ``src/rcp_ndcg/retrieval/external_rerankers.py``;
Apache-2.0, adapted from the vendor's zerank-1 code), as the paper config instantiated it:
max_seq_len=8192, bfloat16, right padding. Where the model card and the paper code disagree, the
paper code wins:

* score scale: ``sigmoid(yes_logit / 5)`` (probability in [0, 1]).
* context budget: the whole rendered prompt truncates at 8192 tokens from the right; the card
  advertises 32768. 8192 wins (it is what the paper ran). NOTE: this whole-prompt right cut
  drops the assistant header -- the anchor the score head reads -- on over-cap pairs; the SERVED
  path reserves the anchor and cuts the content instead. The recipe declares this as
  ``reference.known_deviations: [anchor_drop_over_cap]``; stage 2 reports over-cap pairs outside
  the gates. It is deliberately NOT copied here from the served path: this file is the paper's
  scoring, and the deviation is the paper's.
* padding side: right (set explicitly; also the tokenizer default; the pad token ``...`` is already
  set in the checkpoint's tokenizer_config.json).
* trust_remote_code: not needed.

Checkpoint: zeroentropy/zerank-1-small-reranker @ a65fd51c450e9b47fdddab98e31166ecad21af8d.
Qwen3-1.7B causal LM (hidden 2048, 28 layers, tie_word_embeddings=true, bf16; the card's
"Base Model: Qwen3-4B" row is wrong), scored on the last position's logit of the token "Yes"
(id 9454, a single token of this tokenizer: measured on the pinned tokenizer.json). Score =
``sigmoid(l_Yes / 5)``.

This file runs as a SUBPROCESS in the reference environment -- never inside the harness, which
imports no torch and no transformers. The reference environment is pinned in
``requirements-reference.txt`` beside this file (torch, transformers, scipy; tokenizers and
huggingface_hub come with transformers and are all stage 1 needs). Because that environment has
no rcp-ndcg, the recipe's anchor-preserving render (the harness's stage-1 ``--mode render``
contract: fixed segments reserved, content cut, template re-attached) is implemented here
directly, as a faithful port of ``rcp_ndcg.data.preprocess`` (``token_prefix`` and ``fit``'s pair
path) for this recipe's frame alone; the recipe tests pin it byte-identical to the product's own
``fit`` on sampled pairs, including over-budget ones. The served frame itself is built from the
tokenizer's added tokens by name (``im_start``, ``im_end``), never typed literally.

Subprocess contract (the harness invokes: ``<python> reference.py --mode <mode> --pairs <file>
--out <file> --tokenizer <spec> [--device <d>]``):

- ``--mode render``  -> ``{"rows": [{"index", "shape": "pair", "query": str,
  "documents": [str, ...]}]}``: the cut content spans the served wire carries per pairs row (the
  query settled once per row, the document spans cut to what remains -- the anchor-preserving
  cut, as a faithful port of ``rcp_ndcg.data.preprocess`` (``token_prefix`` and ``fit``'s pair
  path) for this recipe's frame alone; the recipe tests pin it byte-identical to the product's
  own spans on sampled pairs, including over-budget ones).  Stage 1; tokenizer only, no torch.
- ``--mode score``   -> ``{"rows": [{"index", "scores": [float, ...]}]}`` on the probability
  scale (stage 2; needs torch and the ~3.4 GB weights).
- ``--mode embed``   -> refused: a reranker has no embed mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "zeroentropy/zerank-1-small-reranker"
REVISION = "a65fd51c450e9b47fdddab98e31166ecad21af8d"
MAX_SEQ_LEN = 8192  # the paper's MAX_SEQ_LENGTH: the whole-pair budget, the recipe's client.max_tokens
QUERY_MAX_TOKENS = 4096  # the served path's query share, the recipe's client.query_max_tokens
BATCH_SIZE_TOKENS = 15_000  # the paper's char-length batching budget (class default)
DTYPE = "bfloat16"  # the paper's dtype; the checkpoint config agrees
YES_TOKEN = "Yes"
YES_TOKEN_ID = 9454  # a single token of this tokenizer: measured on the pinned tokenizer.json
# ("Yes" -> [9454], convert_tokens_to_ids agrees); the recipe tests pin it

TOKENIZER_FILE = "tokenizer.json"
#: Whether the scoring route counts the tokenizer's post-processor tokens. The engine adds them
#: (add_special_tokens=True); for this tokenizer they are measured to add none (plain strings and the
#: empty render alike), so the frame overhead below is the whole fixed cost either way. Declared, not
#: assumed.
ADD_SPECIAL_TOKENS = True


# -------------------------------------------------------------------------------------------
# The tokenizer backend: the `tokenizers` library (also stage 1's; the same tokenizer.json the
# harness and the engine read). A Hub spec downloads that one file with huggingface_hub.
# -------------------------------------------------------------------------------------------


def _split_spec(spec: str) -> tuple[str, str | None]:
    """``org/model@<revision>`` -> (repo, revision); a local path passes through with no revision.

    The same resolution rule as the product's ``rcp_ndcg.data.tokenizer._local_path``: a path is
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
    declared pair shape -- the template file's zerank branch. Specials resolved from the tokenizer."""
    im_start = _special_text(tokenizer, "im_start")
    im_end = _special_text(tokenizer, "im_end")
    head = f"{im_start}system\n"
    mid = f"{im_end}\n{im_start}user\n"
    tail = f"{im_end}\n{im_start}assistant\n"
    return head, mid, tail


# -------------------------------------------------------------------------------------------
# The anchor-preserving render: a faithful port of the product's mechanism for this recipe's
# frame (rcp_ndcg.data.preprocess.token_prefix and fit's pair path). The reference environment
# has no rcp-ndcg, so the port lives here; the recipe tests pin it byte-identical to the
# product's fit on sampled pairs. The anchor (the trailing assistant header) is reserved: only
# the content spans are cut, and the frame is re-attached after the cut.
# -------------------------------------------------------------------------------------------


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

    Port of the product's role client for this recipe's declaration: the query's span settles once
    per request -- to its declared share (``QUERY_MAX_TOKENS``) whenever it exceeds it (the settle
    rule the wire carries; a bare pair fit would bind the share on overflow only), then through the
    pair fit's probe pair (the query with an empty document) that guarantees the frame and the
    anchor fit even alone -- and each document gets what remains. Every cut is a verbatim prefix at
    a token boundary of the content, the frame and its trailing anchor always re-attached by the
    engine around these spans. A query that fills the budget and leaves the document nothing is
    refused, never cut undeclared.
    """
    head, mid, tail = _frame(tokenizer)

    def assemble(q: str, d: str) -> str:
        return head + q + mid + d + tail

    cap = MAX_SEQ_LEN
    overhead = _count(assemble("", ""), tokenizer, add_special_tokens=ADD_SPECIAL_TOKENS)
    if overhead > cap:
        raise SystemExit(f"the frame's fixed overhead alone is {overhead} tokens, over the budget of {cap}")
    # 1. settle at the declared share whenever the query exceeds it:
    if _count(query, tokenizer) > QUERY_MAX_TOKENS:
        q_final = _token_prefix(query, QUERY_MAX_TOKENS, tokenizer)
    else:
        q_final = query
    # 2. fit's probe pair (the query with an empty document): the query keeps the frame room.
    q_final = _token_prefix(
        q_final, cap, tokenizer, rendered=lambda piece: assemble(piece, ""), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    q_min = _count(assemble(q_final, ""), tokenizer, add_special_tokens=ADD_SPECIAL_TOKENS)
    if q_min >= cap and _count(document, tokenizer) > 0:
        raise SystemExit(
            f"the query fills the pair budget of {cap} tokens and leaves the document nothing; "
            "lower QUERY_MAX_TOKENS (or raise MAX_SEQ_LEN), so the document keeps a share"
        )
    # 3. the document gets what remains after the settled query and the frame:
    d_final = _token_prefix(
        document, cap, tokenizer, rendered=lambda piece: assemble(q_final, piece), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    return q_final, d_final


def served_render(tokenizer, query: str, document: str) -> str:
    """The served pair render for one (query, document): the frame with the content spans fitted.

    The frame-assembled view of :func:`served_spans` (the engine assembles the template around the
    wire's cut spans; the anchor -- the trailing assistant header -- is always re-attached).
    """
    head, mid, tail = _frame(tokenizer)
    q_final, d_final = served_spans(tokenizer, query, document)
    return head + q_final + mid + d_final + tail


# -------------------------------------------------------------------------------------------
# The paper's in-process reference (experiments/paper/rerankers/reference/zerank.py, unchanged
# in behaviour; the interface the harness documents: load(device), render(...), score(...)).
# -------------------------------------------------------------------------------------------


class ZerankReference:
    """``load(device)`` / ``render(query, doc, instruction)`` / ``score(query, docs, instruction)``.

    The paper's ``ZerankRerank``: chat template with system=query and user=document (both
    stripped), the assistant generation prompt appended, the "Yes"-token logit read at the last
    non-pad position, divided by 5 and sigmoided. The whole rendered prompt truncates at
    ``MAX_SEQ_LEN`` from the right at forward time -- the known ``anchor_drop_over_cap`` deviation
    the recipe declares, never copied into the served path.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = None
        self.yes_token_id: int | None = None

    # ------------------------------------------------------------------ load
    def load(self, device: str | None = None, *, tokenizer_spec: str = f"{REPO}@{REVISION}") -> ZerankReference:
        """Load the checkpoint the way the paper's ``ZerankRerank.__init__`` does (torch + transformers)."""
        torch, automodel, autotokenizer = _reference_stack()
        repo, revision = _split_spec(tokenizer_spec)
        self.tokenizer = autotokenizer.from_pretrained(repo, padding_side="right", revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = automodel.from_pretrained(repo, dtype=torch.bfloat16, revision=revision)
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
        assert self.yes_token_id == YES_TOKEN_ID, self.yes_token_id

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        return self

    # ----------------------------------------------------------------- render
    def render(self, query: str, doc: str, instruction: str | None = None) -> str:
        """The paper's prompt TEXT for one pair: the checkpoint chat template over
        [system=query.strip(), user=document.strip()] with the generation prompt appended.

        The family takes no instruction parameter (the caller folds any into *query*); this
        reference ignores one, exactly like the served client with ``instruction: none``.
        Truncation is NOT applied here -- the paper truncates the whole rendered prompt at forward
        time (``_batch_logits``, right side, ``MAX_SEQ_LEN``); ``score`` reproduces that.
        """
        if self.tokenizer is None:
            raise RuntimeError("call load() before render()")
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": query.strip()},
                {"role": "user", "content": doc.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------ score
    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """``sigmoid(yes_logit / 5)`` per document, aligned with *docs*.

        Port of the paper's ``predict`` + ``_batch_logits``: length-descending permutation for
        tight token buckets, right padding, the score read at the last non-pad position, the whole
        rendered prompt truncated at ``MAX_SEQ_LEN`` from the right.
        """
        import numpy as np
        from scipy.special import expit

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("call load() before score()")
        if instruction is not None:
            # The family takes no instruction; the paper pipeline never passed one. Dropping it is
            # what the served client (instruction: none) does too.
            instruction = None
        texts = [self.render(query, doc, instruction) for doc in docs]

        # Length-descending permutation for tight token buckets (paper, char-length budget).
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
            all_logits.extend(self._batch_logits(batch))

        scores = expit(np.asarray(all_logits)).tolist()
        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, scores, strict=True):
            result[orig_idx] = score
        return result

    def _batch_logits(self, batch: list[str]) -> list[float]:
        """One padded forward; on CUDA OOM, halve and recurse (the paper's backoff)."""
        import torch

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs, use_cache=False)
                attention_mask = inputs["attention_mask"]
                last_positions = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(outputs.logits.shape[0], device=self.device)
                last_logits = outputs.logits[batch_indices, last_positions]
                yes_logits = last_logits[:, self.yes_token_id]
                return [float(value) / 5.0 for value in yes_logits]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._batch_logits(batch[:mid]) + self._batch_logits(batch[mid:])

    def embed(self, texts: list[str], role: str) -> None:  # pragma: no cover - not applicable
        """A reranker has no embed mode."""
        raise NotImplementedError("zerank-1-small is a reranker, not an embedder")


def _reference_stack():
    """torch and transformers, imported only where they are needed (the reference environment)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise SystemExit(
            "score mode needs the reference environment (torch + transformers + the ~3.4 GB checkpoint "
            "weights); stage 2 runs it on the GPU wave. Pin recipes/zerank-1-small-reranker/"
            "requirements-reference.txt in that environment."
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------------------------------------------------
# The subprocess CLI the harness invokes.
# -------------------------------------------------------------------------------------------


def main() -> int:
    """The reference CLI: ``--mode render|score`` over the pairs file, the mode's JSON to ``--out``."""
    parser = argparse.ArgumentParser(description="the zerank-1-small-reranker reference (the paper's ZerankRerank)")
    parser.add_argument("--mode", required=True, choices=["render", "score", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision or a path)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "embed":
        raise SystemExit("zerank-1-small is a reranker: the reference speaks render and score, not embed")

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        tokenizer = _load_backend_tokenizer(args.tokenizer)
        rows = []
        for index, row in enumerate(pairs):
            shape = str(row.get("shape") or "pair")
            if shape != "pair":
                raise SystemExit(f"row {index}: shape {shape!r} is not declared; this recipe declares the pair shape")
            # The recipe declares instruction: none: the instruction (if a row carries one) is not
            # part of this model's input, on the served path or here. The settled query spans one
            # row's requests; each document span is cut to what remains after it.
            query_span, _ = served_spans(tokenizer, row["query"], str(row["documents"][0]))
            document_spans = []
            for document in row["documents"]:
                _, document_span = served_spans(tokenizer, row["query"], str(document))
                document_spans.append(document_span)
            rows.append({"index": index, "shape": shape, "query": query_span, "documents": document_spans})
        output = {"rows": rows}
    else:
        model = ZerankReference().load(args.device, tokenizer_spec=args.tokenizer)
        rows = []
        for index, row in enumerate(pairs):
            scores = model.score(row["query"], list(row["documents"]), row.get("instruction"))
            rows.append({"index": index, "scores": scores})
        output = {"rows": rows}
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
