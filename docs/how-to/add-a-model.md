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
  reference.py     # the in-process reference implementation the equivalence harness compares against
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
- **Special tokens by name.** Template segments never contain special-token literals: `{"special": "im_end"}`
  is resolved through the recipe tokenizer's added tokens at run time, `{"text": ...}` is ordinary template
  text, `{"ids": [...]}` is a measured sequence, and `{"content": query | document}` marks the cuttable span.
  Every declared shape needs at least one content span; an `anchor: last` shape must end with a fixed segment
  (or pin `add_special_tokens: true`, declaring the tokenizer's end token as the anchor). When
  `serve.chat_template` is set, stage 1 also proves the declared shapes render to the same token ids as the
  template file.
- **Explicit budgets.** Every recipe declares `client.tokenizer` and `client.max_tokens` — there is no implicit
  default. `on_overflow` is `cut` by default (content spans only, at token boundaries, anchors reserved;
  `chunk` and `fail` are the alternatives) and `aggregation: max` is the only chunk aggregation. The engine's
  `max_model_len` must fit the client's budget. `serve.pooler_config` keys are validated against the pinned
  engine's `PoolerConfig` fields, because the engine rejects unknown keys.

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

`reference.py` is loaded straight from the recipe directory; it may import torch, transformers or the checkpoint's
remote code. It defines exactly:

```python
def load(device: str) -> object:
    """Load the model once onto device ("cpu" or "cuda:0"); keep it module-global."""

def score(query: str, documents: list[str], instruction: str | None) -> list[float]:
    """Rerank: one score per document, on reference.score_scale (probability | logit | cosine)."""

def embed(texts: list[str], role: str) -> "list[numpy.ndarray]":
    """Embedding roles: one array per text, (dim,) for dense, (n_tokens, dim) for late interaction.
    role is "query" or "document"; the reference composes its own prompts."""

def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1, default shape: the token ids of the exact prompt the engine must see for that pair.
    The default shape is the pair shape for a rerank recipe and the document shape for an embedding recipe
    (the recipe's declared shapes assembled with the anchor-preserving cuts, or the prompted text when no
    template block is declared)."""

def render_shape(shape: str, query: str, document: str, instruction: str | None) -> list[int]:
    """Optional: the same for the other declared shapes, so the anchor audit checks the reference's render
    of every shape; without the hook, only the default shape's reference-side audit is possible."""
```

Optionally `tokenizer()`, an object with `encode(text) -> list[int]` and `id_to_token(id) -> str`, so stage 1 can
run on CPU without a Hub download (tests and CI); production references omit it and stage 1 loads the recipe's
`client.tokenizer` with transformers.

## Choosing how vLLM serves a model

The serving path is chosen per model — flags alone, a chat template, pooler settings, or a plugin — and the
decision tree lives in this section once the survey of model families lands; for now, a recipe's `serve` section
renders verbatim into `vllm serve` argv, and `serve.plugin` is reserved for a `vllm.general_plugins` package when
no flag can express the model's scoring.

## Worked example: a last-token-pooling embedder (CPU)

A last-token-pooling embedder's anchor is its trailing end token: the wrapper renders `prefix + text + suffix`
and the naive fix — truncating the whole string on the right — drops the token the model was trained to read
out of. The correct cut reserves the suffix, cuts only the text, and re-attaches the template. The mechanism in
ten dependency-free lines (the real implementation is the harness's `assemble_shape_text`, checked against your
reference in stage 1):

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

The same assertion in code lives in the tests (`test_zembed`-shaped): for an over-length input, every declared
anchor id sits at its declared position in the assembled ids.

## The three equivalence stages

Stage 1 (CPU, zero tolerance) renders every sampled prompt the way the client renders it — a declared shape
assembled from segments, or the `template.jinja` file under a jinja2 environment with `trim_blocks` and
`lstrip_blocks` on, a stripped trailing newline and undefined variables refused (with the instruction variable
carrying the recipe's `default_instruction` in `field` mode, and empty in `fold` mode, where the query text
already carries it) — tokenises with the recipe's tokenizer and requires exact equality with
`reference.render`. Each declared shape is sampled on its own (at least 20 over-length
inputs per shape, padded in that shape's own content span), and the audit asserts every anchor survived the cut
on the served render and — when the reference provides the optional `render_shape(shape, query, document,
instruction)` hook — on the reference's render of the same shape. The result is reported as `anchor_check`,
separately from the token-id mismatches; when `serve.chat_template` is set, stage 1 also proves the declared
shapes render to the same token ids as the template file.

Stage 2 scores or embeds the same pairs against the served engine (plain `httpx` to `/rerank`, `/v1/embeddings`,
`/pooling`) and applies the gates: probability |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all; logit |Δ| ≤
0.05·(1 + |s|); cosine scores |Δ| ≤ 0.01; vectors cosine ≥ 1 − 1e-3 per vector (per token, after the same
float16 cast); median per-query Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these. With
`reference.known_deviations: [anchor_drop_over_cap]`, pairs whose uncut prompt exceeds `client.max_tokens` are
reported in a separate, non-gating table and the gates run on the under-cap pairs only.

Stage 3 (optional, needs the `metrics` extra) scores rankings per subset with `rcp-ndcg eval score` as a
subprocess and requires the mean |Δ nDCG@10| over subsets ≤ 2e-3.

```bash
python -m rcp_ndcg_vllm.equivalence --recipe recipes/<id> --base-url http://127.0.0.1:8100 \
    --pairs pairs.jsonl --out /tmp/equiv            # stages 1 and 2 against a running engine
```

The exit code is 0 only when every gate passes; `equivalence.json` carries every number with its referent
(per document, per query, per subset) and `EQUIVALENCE.md` is the short section for the recipe's report.

## Submitting a wave

```bash
export RCP_KJOBS_CONFIG=/path/to/jobs-config.yaml    # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
packages/rcp-ndcg-vllm/jobs/submit.sh <wave-name> <recipes-file> gs://YOUR-BUCKET/stage gs://YOUR-BUCKET/waves
```

`RCP_KJOBS_CONFIG` and `RCP_GCS_AUTH_FILE` are required — the script refuses to run without them, because no
tracked file may name a machine's paths. `EXTRA_DIRS` (space-separated directories staged into the tarball) and
`KJOBS=echo` (print the commands instead of running them) are optional.

The script stages a tarball of the current commit (plus any directories in `EXTRA_DIRS`, and the recipes file),
uploads it to `<stage-prefix>/<wave-name>/code.tar.gz`, and submits the job `rcp-<wave-name>`; on the node,
`bootstrap.sh` authenticates, installs the packages from the tarball into the image's Python (`--no-deps`, so the
image's vLLM and torch are never touched), and runs `run_wave.py` with `--record` and `--upload`. The wave packs
recipes onto the node's GPUs, serves one engine per slot, runs smoke, equivalence and the recorder, and writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary. `KJOBS=echo` prints the commands instead
of running them.
