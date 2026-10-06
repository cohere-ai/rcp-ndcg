"""The zerank-1-reranker reference: the paper's exact in-process implementation, run as a subprocess.

Derived from ``experiments/paper/rerankers/reference/zerank.py`` (the ``ZerankRerank`` class the
package carried when the paper's runs were made, itself moved unchanged out of
``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py``). The scoring behaviour is unchanged: per pair, the
chat template of the checkpoint's own tokenizer renders ``[{system: query.strip()},
{user: doc.strip()}]`` with ``add_generation_prompt=True``, the whole rendered prompt is tokenized
with ``truncation=True, max_length=8192`` (right side), the logit of "Yes" (single token 9454) at the
last non-pad position is divided by 5 and sigmoided (``scipy.special.expit``): probabilities in
(0, 1). Batching (length-descending permutation, 15,000-character buckets, right padding) and the
OOM backoff are the paper's; per-pair scores are invariant to both (causal attention, masked right
padding, the last non-pad position).

Two harness modes (see ``rcp_ndcg_vllm.equivalence.reference`` for the contract):

- ``--mode render``: the anchor-preserving render, per row. For a pair whose paper prompt fits
  ``client.max_tokens`` this is the paper's own construction (the fidelity stage 1 asserts: the
  declared shape in the recipe must reproduce the checkpoint's chat template byte for byte). For an
  over-cap pair the paper's right cut would drop the trailing assistant header - the last-token
  anchor - so the render is the anchor-preserving one instead: the product's ``fit`` (the same
  mechanism the served client runs) reserves the fixed frame, cuts the content spans and re-attaches
  the template. The anchor drop is never copied into the served path; the recipe declares it as
  ``reference.known_deviations: [anchor_drop_over_cap]`` and stage 2 gates under-cap pairs only.
- ``--mode score``: the paper's ``predict`` per row, on the recipe's ``reference.score_scale``
  (probability). The recipe takes no instruction (``instruction: none``): a row's ``instruction``
  field is ignored, exactly as the paper's path ignored it.

Runs in its OWN python (``--reference-python``), never inside the harness: its environment is pinned
by ``requirements-reference.txt`` beside this file (torch, transformers, scipy for the paper's code,
plus the ``rcp-ndcg`` release whose ``fit`` renders the anchor-preserving shape). The model is loaded
from the recipe's ``model`` at the recipe's ``revision`` (read from the ``recipe.yaml`` beside this
file), bfloat16, on ``--device``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_SEQ_LENGTH = 8192
"""The paper's whole-prompt token cap (``MAX_SEQ_LENGTH`` of the old served client)."""
BATCH_SIZE_TOKENS = 15_000
"""The paper's character budget per forward batch (throughput only; scores are per pair)."""


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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, yes_token_id, device


def paper_prompt(tokenizer, query: str, document: str) -> str:
    """The paper's prompt for one pair: ``ZerankRerank._format_inputs``, unchanged.

    The query and document are stripped, the checkpoint's own chat template renders them as the
    system and user turns, and the generation prompt (the assistant header) is appended.
    """
    messages = [
        {"role": "system", "content": query.strip()},
        {"role": "user", "content": document.strip()},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str):
        raise TypeError("apply_chat_template(tokenize=False) returned a non-string")
    return text


def score_pair(model, tokenizer, yes_token_id: int, device: str, query: str, documents: list[str]) -> list[float]:
    """The paper's ``predict`` for one query and its documents (probabilities, aligned with ``docs``)."""
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


def _anchor_preserving_render(tokenizer_spec: str, query: str, document: str, instruction: str | None) -> str:
    """The anchor-preserving render of one pair, by the product's own budget mechanism.

    The served client runs ``rcp_ndcg.data.preprocess.fit`` with the recipe's declared template and
    budget; this is the same call, so the reference's render is the render the engine will read: the
    fixed frame reserved, the content spans cut at token boundaries, the assistant header (the
    last-token anchor) always re-attached. The instruction rides beside the fit (the recipe declares
    ``instruction: none``, so the query is the bare query either way).
    """
    import yaml

    from rcp_ndcg.data.preprocess import TextBudget, fit
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = yaml.safe_load((_recipe_dir() / "recipe.yaml").read_text(encoding="utf-8"))
    client = recipe["client"]
    tokenizer = load_tokenizer(tokenizer_spec)
    budget = TextBudget(
        tokenizer=tokenizer_spec,
        max_tokens=client["max_tokens"],
        query_max_tokens=client.get("query_max_tokens"),
        template=client.get("template"),
        on_overflow=client.get("on_overflow", "cut"),
    )
    result = fit([(query, document)], "pair", budget, tokenizer, ids=["0"], instruction=instruction)
    return result.texts[0]


def main() -> int:
    """The reference CLI: ``--mode render|score``, a pairs file in, the mode's JSON out."""
    parser = argparse.ArgumentParser(description="the zerank-1-reranker reference (the paper's implementation)")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

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
        from rcp_ndcg.data.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)
        from transformers import AutoTokenizer

        source, revision = _tokenizer_source(args.tokenizer)
        hf_tokenizer = AutoTokenizer.from_pretrained(source, padding_side="right", revision=revision)
        rows = []
        for index, row in enumerate(pairs):
            query, document = str(row["query"]), str(row["documents"][0])
            prompt = paper_prompt(hf_tokenizer, query, document)
            if tokenizer.count(prompt, add_special_tokens=True) <= _max_tokens():
                text = prompt
            else:
                text = _anchor_preserving_render(args.tokenizer, query, document, row.get("instruction"))
            rows.append({"index": index, "shape": "pair", "text": text})
        output = {"rows": rows}

    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


def _max_tokens() -> int:
    """The recipe's declared pair budget (``client.max_tokens``)."""
    import yaml

    recipe = yaml.safe_load((_recipe_dir() / "recipe.yaml").read_text(encoding="utf-8"))
    return int(recipe["client"]["max_tokens"])


if __name__ == "__main__":
    raise SystemExit(main())
