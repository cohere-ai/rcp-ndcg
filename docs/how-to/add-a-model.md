# Add a model: serving recipes for vLLM

A recipe is one declarative description of how a model is served with vLLM and how `rcp-ndcg` reads it back. This
guide shows the format, how to check a served recipe against its reference implementation, and how a wave of
recipes is submitted to the GPU host. The package lives at `packages/rcp-ndcg-vllm/` (outside the root uv
workspace; it is installed into the engine image, which carries its own vLLM and torch).

## The recipe directory

One directory per model, `packages/rcp-ndcg-vllm/recipes/<id>/`, with three files:

```text
recipes/<id>/
  recipe.yaml      # the recipe (the Recipe schema; every field is listed in schema/recipe.schema.json)
  template.jinja   # the chat template given to vllm serve --chat-template (only when the model needs one)
  reference.py     # the reference implementation, run as a subprocess (see the reference interface)
```

The `id` equals the directory name, matches `^[a-z0-9][a-z0-9.-]*$`, and is also the `--served-model-name` the
engine serves. The schema is closed (`extra="forbid"`) and role-aware: a field that only makes sense for one role
is refused for the others, so a typo cannot silently change what is served. Validate a recipe without an engine (run from the package directory, so `recipes/<id>` resolves):

```bash
cd packages/rcp-ndcg-vllm
python -m rcp_ndcg_vllm.equivalence --recipe recipes/<id> --pairs pairs.jsonl --out /tmp/equiv --stages 1
```

## The template block, the anchors and the explicit budget

Three research findings shape the `serve` and `client` blocks, and the schema enforces them:

- **Anchors.** A model reads its output from fixed positions of its template — the *anchors* (a last-token
  pooler's end token, a CLS head's leading special, a marker id, a reranker's assistant suffix). Every cut
  applies to the content spans only, inside a budget computed after reserving every fixed template token, and
  the template is re-attached after the cut. Engine-side truncation of a rendered prompt cannot honour this in
  either direction, so the schema has no engine-truncation field at all: a recipe whose reference deliberately
  drops anchors declares `reference.known_deviations: [anchor_drop_over_cap]` and the harness reports those
  pairs separately, outside the gates. The declared shape's `anchor` is `last`, `first`, `mean` or `marker`
  (with `anchor_markers`), and stage 1 samples over-length inputs (at least 20 per shape) and asserts every
  anchor survived — reported as `anchor_check`, separately from token-id mismatches.
- **Segments: `fixed` and `content`, nothing else.** A shape is an ordered list of segments with exactly two
  kinds: `fixed` — ordinary template text, which may name special tokens as `{special:<name>}` placeholders
  that the recipe tokenizer's added tokens resolve at run time — and `content`, the cuttable span
  (`query` or `document`). The schema is closed, so `{"special": ...}`, `{"text": ...}` or `{"ids": ...}`
  segments are refused. Every declared shape needs at least one content span; an `anchor: last` shape must end
  with a fixed segment (or pin `add_special_tokens: true`, declaring the tokenizer's end token as the anchor).
  When `serve.chat_template` is set, stage 1 also proves the declared shapes render to the same token ids as
  the template file.
- **Explicit budgets.** Every recipe declares `client.tokenizer` and `client.max_tokens` — there is no implicit
  default. `on_overflow` is `cut` by default (content spans only, at token boundaries, anchors reserved;
  `chunk` and `fail` are the alternatives) and `aggregation: max` is the only chunk aggregation. The engine's
  `max_model_len` must fit the client's budget. `serve.pooler_config` keys are validated against the pinned
  engine's `PoolerConfig` fields, because the engine rejects unknown keys.
- **The engine is the tokenization truth (R29).** With an engine URL, stage 1 sends `fit`'s rendered prompts
  to the engine's `/tokenize` (same `add_special_tokens` as the route) and requires the ids and counts to
  equal `fit`'s; mismatches are reported per shape in `equivalence.json` (`engine_tokenize_check`) and fail
  the stage. Without an engine (CPU), the check is reported as `not_run`, never as passed.
- **The reference runs as a subprocess.** Stage 2 runs the recipe's `reference.py` as a subprocess
  (`--reference-python <path>`, required when stage 2 runs; no default) that reads the pairs file and writes
  scores or vectors to a file the harness compares. The harness process imports no torch or transformers; the
  reference environment is documented in `packages/rcp-ndcg-vllm/requirements-reference.txt`. The engine comes
  up on the slot's GPUs first; the reference subprocess runs against the pairs file while the engine is up and
  releases its memory when it exits.

Recipe YAML at a glance (abridged; the schema's docstrings define every field):

```yaml
id: qwen3-reranker-0.6b
model: Qwen/Qwen3-Reranker-0.6B
revision: <40-hex commit>        # quoted: a bare commit can read as a number
role: rerank                     # embed | multi_vector | rerank
input: [text]                    # subset of [text, image, video]
scoring: pointwise               # rerank only: pointwise | listwise
licence: apache-2.0              # SPDX id or see-model-card
engine: {name: vllm, image: "vllm/vllm-openai:v0.31.0", min_version: "0.31.0"}
resources: {gpus: 1}             # tensor_parallel_size = gpus
serve:                           # everything rendered into `vllm serve` argv; nothing implicit
  runner: pooling
  hf_overrides: {"architectures": [...], "classifier_from_token": ["no", "yes"]}   # always quote strings!
  chat_template: template.jinja  # a file in this directory, or null
  pooler_config: {use_activation: true}   # keys must be PoolerConfig fields at the pinned engine
  max_model_len: 8192
  dtype: bfloat16
  extra_args: []                 # further flags, verbatim (one argv element per item)
client:                          # the rcp-ndcg endpoint fields this recipe implies
  api: rerank                    # openai_embeddings | vllm_pooling | rerank
  request_shape: text            # text | messages | token_ids
  add_special_tokens: null       # bool|null; true declares the post-processor end token as the anchor
  template:                      # the request shapes as data; specials by name, never typed
    pair:
      - {special: im_start}      # resolved from the tokenizer's added tokens
      - {text: "system\nJudge whether the document answers the query."}
      - {special: im_end}
      - {content: query}         # the cuttable span; fold mode puts the instruction text here
      - {content: document}
      - {text: "assistant-suffix-readonly"}   # sketch: a fixed tail the model reads - a real recipe
                                              # writes the measured text or ids here
    anchor: last
    query_max_tokens: 1024       # the query's share of the pair budget; the document span gets the rest
  instruction: fold              # rerank only: none | field | fold | system
  default_instruction: "Judge whether the document answers the query."
  tokenizer: "<repo>@<40-hex commit>"
  max_tokens: 8192               # explicit; there is no implicit budget
  on_overflow: cut               # cut (default) | chunk | fail; cuts apply to content spans only
  aggregation: null              # only with on_overflow: chunk; max is the only supported aggregation
  empty_doc: omit_zero           # omit_zero | send | send_text
  normalize: null                # embed and multi_vector only
  embed_dtype: null              # multi_vector only; float16 by default, sent explicitly (engine default float32)
reference:
  kind: transformers             # transformers | sentence_transformers | remote_code | stored_scores
  score_scale: probability       # probability | logit | cosine; vectors compare per vector
  entry: reference.py
  known_deviations: []           # e.g. [anchor_drop_over_cap]: stage 2 gates under-cap pairs only
gates: {}                        # overrides of the stage-2 defaults for this score_scale
status: {state: unverified, image: null, date: null, report: null}
sources: []                      # URLs and path:line references the recipe rests on
notes: ""
```

Two YAML footguns, both caught in review and by the golden tests: always quote string tokens that YAML reads as
booleans (`"no"`, `"yes"`, `"on"`), and always quote the 40-hex revision.

## The reference interface

`reference.py` runs as a subprocess CLI, not an import: stage 2 launches it with the recipe's
`--reference-python` and a mode, and reads back a JSON file. It may import torch, transformers or the
checkpoint's remote code — the harness process never does. The contract (enforced by
`equivalence/reference.py`'s runner, which all fixture references implement):

```text
reference.py --mode <render|score|embed> --pairs <file> --out <file> \
             --tokenizer "<repo>@<revision>|path/to/tokenizer.json" --device <cpu|cuda:0>
```

- `--mode render` — stage 1's reference side: `{"rows": [{"index", "shape", "text": str}]}`, the exact prompt
  text the reference expects the engine to see for that pair's declared shape (the anchor-preserving render:
  fixed segments reserved, content cut, template re-attached).
- `--mode score` — rerank: `{"rows": [{"index", "scores": [float, ...]}]}` on the recipe's
  `reference.score_scale` (probability | logit | cosine).
- `--mode embed` — embedding roles: `{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[...]]}]}`
  (dense: one vector per side; late interaction: one per token).
- `requirements-reference.txt` in the recipe directory pins the reference environment (torch, transformers,
  sentence-transformers as needed); it is documented, not installed, by the harness.

## Choosing how vLLM serves a model

The serving path is chosen per model — flags alone, a chat template, pooler settings, or a plugin — and the
decision tree lives in this section once the survey of model families lands; for now, a recipe's `serve` section
renders verbatim into `vllm serve` argv, and `serve.plugin` is reserved for a `vllm.general_plugins` package when
no flag can express the model's scoring.

## Worked example: a last-token-pooling embedder (CPU)

A last-token-pooling embedder's anchor is its trailing end token: the wrapper renders `prefix + text + suffix`
and the naive fix — truncating the whole string on the right — drops the token the model was trained to read
out of. The correct cut reserves the suffix, cuts only the text, and re-attaches the template. The mechanism in
ten dependency-free lines (the real implementation is the product's `fit` in `rcp_ndcg.data.preprocess`, checked
against your reference in stage 1):

```python
# The template block of a last-token-pooling embedder, as segments:
#   [{text: "doc: "}, {content: document}, {text: "<end>"}], anchor: last.
prefix_ids = [1, 2]          # token ids of the "doc: " fixed head, measured with the recipe tokenizer
suffix_ids = [9]             # the end token: the anchor the model pools from
content_ids = [3, 4, 5, 6, 7, 8]  # the document, already tokenised

max_tokens = 5
budget = max_tokens - len(prefix_ids) - len(suffix_ids)
cut = content_ids[:budget]   # cut the content span only ...
prompt_ids = prefix_ids + cut + suffix_ids   # ... and re-attach the template
assert prompt_ids == [1, 2, 3, 4, 9]
assert prompt_ids[-len(suffix_ids):] == suffix_ids  # the anchor survived the cut
assert len(prompt_ids) == max_tokens
```

The same assertion in code lives in the package's tests (`test_stage1_passes_for_every_anchor_kind...`): for an
over-length input, every declared anchor id sits at its declared position in the assembled ids.

## The three equivalence stages

Stage 1 (CPU, zero tolerance) renders every sampled prompt the way the client renders it — a declared shape
assembled from segments, or the `template.jinja` file under a jinja2 environment with `trim_blocks` and
`lstrip_blocks` on, a stripped trailing newline and undefined variables refused (the instruction variable
carrying the pairs row's instruction, empty when the row has none — in `fold` mode the query text already
carries it) — tokenises with the recipe's tokenizer and requires exact equality with `reference.render`. Each
declared shape is sampled on its own (at least 20 over-length inputs per shape, padded in that shape's own
content span), and the audit asserts every anchor survived the cut on the served render of every shape. The
result is reported as `anchor_check`, separately from the token-id mismatches; when `serve.chat_template` is
set, stage 1 also proves the declared shapes render to the same token ids as the template file.

Stage 2 scores or embeds the same pairs against the served engine (plain `httpx` to `/rerank`, `/v1/embeddings`,
`/pooling`) and applies the gates: probability |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all; logit |Δ| ≤
0.05·(1 + |s|); cosine scores |Δ| ≤ 0.01; vectors cosine ≥ 1 − 1e-3 per vector (per token, after the same
float16 cast); median per-query Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these. With
`reference.known_deviations: [anchor_drop_over_cap]`, pairs whose uncut prompt exceeds `client.max_tokens` are
reported in a separate, non-gating table and the gates run on the under-cap pairs only.

Stage 3 (optional) scores rankings per subset with `rcp-ndcg eval score` as a subprocess (the package depends
on `rcp-ndcg`, so the command is always available) and requires the mean |Δ nDCG@10| over subsets ≤ 2e-3.

```bash
python -m rcp_ndcg_vllm.equivalence --recipe recipes/<id> --base-url http://127.0.0.1:8100 \
    --pairs pairs.jsonl --out /tmp/equiv            # stages 1 and 2 against a running engine
```

The exit code is 0 only when every gate passes; `equivalence.json` carries every number with its referent
(per document, per query, per subset) and `EQUIVALENCE.md` is the short section for the recipe's report.

## Submitting a wave

A wave runs on a node against a staged release candidate: [rc_build.sh](release-candidates.md#build-a-release-candidate)
stages the wheels, the wheelhouse, the recipes and the wave lists; `submit.sh` submits the wave's job against
that stage, and the node's `bootstrap.sh` builds the three environments and runs the wave runner:

```bash
export RCP_KJOBS_CONFIG=/path/to/jobs-config.yaml    # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # passed as a kjobs secret, never read or echoed
packages/rcp-ndcg-vllm/jobs/submit.sh gs://YOUR-BUCKET/stage/rc0 gs://YOUR-BUCKET/waves <wave-name>
```

The three variables are required — the script refuses to run without them, because no tracked file may name a
machine's paths or a token. The recipe list is the staged `<RC>/wave-lists/<wave>.txt`; `--priority dev-high`,
`--max-jobs N`, `--script wave0` (the node test) and `KJOBS=echo` (print the plan instead of submitting) are
optional. The full procedure — the candidate, the wheelhouse, the three environments and wave 0 — is in
[Release candidates and the GPU waves](release-candidates.md). The wave packs recipes onto the node's GPUs,
serves one engine per slot with its own GPU slice, port, `VLLM_PORT` and `TMPDIR`, checks the free disk and
evicts each model's weights after its last use, runs smoke, equivalence and the recorder, and writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary.
