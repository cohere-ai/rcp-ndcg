# Add a model: serving recipes for vLLM

A recipe is one declarative description of how a model is served with vLLM and how `rcp-ndcg` reads it back. This
guide shows the format, how to check a served recipe against its reference implementation, and how a wave of
recipes is submitted to the GPU host. The package lives at `rcp-ndcg-vllm/` (outside the root uv
workspace; it is installed into the engine image, which carries its own vLLM and torch).

## The recipe directory

One directory per model, `rcp-ndcg-vllm/recipes/<id>/`, with these four files (a recipe may also ship
a vendored card script that its reference runs verbatim, byte-identical to the Hub file and hash-pinned by the
recipe's test):

```text
recipes/<id>/
  recipe.yaml                  # the recipe (the Recipe schema; every field is listed in schema/recipe.schema.json)
  template.jinja               # the chat template given to vllm serve --chat-template (only when the model needs one)
  reference.py                 # the reference implementation, run as a subprocess (see the reference interface)
  requirements-reference.txt   # optional: the reference's environment (the node's bootstrap installs it)
```

The `id` equals the directory name, matches `^[a-z0-9][a-z0-9.-]*$`, and is also the `--served-model-name` the
engine serves. The schema is closed (`extra="forbid"`) and role-aware: a field that only makes sense for one role
is refused for the others, so a typo cannot silently change what is served. Validate a recipe without an engine (run from the package directory, so `recipes/<id>` resolves):

```bash
cd rcp-ndcg-vllm
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
  reference environment is documented in `rcp-ndcg-vllm/requirements-reference.txt`. The engine comes
  up on the slot's GPUs first; the reference subprocess runs against the pairs file while the engine is up and
  releases its memory when it exits.

Recipe YAML at a glance (a complete, loadable recipe — `tests/docs` runs `load_recipe` on it; the schema's
docstrings define every field):

```yaml
id: example-reranker-0-6b
model: example-org/example-reranker
revision: "0123456789abcdef0123456789abcdef01234567"   # quoted: a bare commit can read as a number
role: rerank                     # embed | multi_vector | rerank
input: [text]                    # subset of [text, image, video]
scoring: pointwise               # rerank only: pointwise | listwise
licence: apache-2.0              # SPDX id or see-model-card
engine: {name: vllm, image: "vllm/vllm-openai:v0.31.0", min_version: "0.31.0"}
resources: {gpus: 1}             # tensor_parallel_size = gpus
serve:                           # everything rendered into `vllm serve` argv; nothing implicit
  runner: pooling
  convert: null
  hf_overrides: {"architectures": ["ExampleForSequenceClassification"], "classifier_from_token": ["no", "yes"]}
                                 # always quote strings YAML reads as booleans ("no", "yes", "on")!
  chat_template: template.jinja  # a file in this directory, or null
  pooler_config: {use_activation: true}   # keys must be PoolerConfig fields at the pinned engine
  trust_remote_code: false
  max_model_len: 8192
  dtype: bfloat16
  plugin: null
  extra_args: []                 # further flags, verbatim (one argv element per item)
client:                          # the product's endpoint config for the role; the product validates it at load
  api: rerank                    # the role's wire: openai_embeddings | vllm_pooling | rerank
  tokenizer: "example-org/example-reranker@0123456789abcdef0123456789abcdef01234567"
  max_tokens: 8192               # explicit; there is no implicit budget
  query_max_tokens: 1024         # the query's share of the pair budget; the document span gets the rest
  template:                      # the request shapes as data: fixed and content segments only
    pair:
      - {fixed: "SYSTEM: Judge whether the Document answers the Query. USER: Query: "}
      - {content: query}         # the cuttable span; fold mode puts the instruction text here
      - {fixed: " Document: "}
      - {content: document}
      - {fixed: "{special:im_end} ASSISTANT"}   # specials by name, resolved from the tokenizer
    anchor: last
  instruction: fold              # rerank only: none | field | fold | system (fold is the default)
  use_activation: true           # a served rerank wire must set it: the score's scale is content
  on_overflow: cut               # cut (default) | chunk | fail; cuts apply to content spans only
  empty_doc: send                # omit_zero | send | send_text
reference:
  kind: transformers             # transformers | sentence_transformers | remote_code | stored_scores
  score_scale: probability       # probability | logit | cosine; vectors compare per vector
  entry: reference.py
  known_deviations: []           # e.g. [anchor_drop_over_cap]: the over-cap pairs are reported non-gating
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

- `--mode render` — stage 1's reference side. Embedding roles: `{"rows": [{"index", "shape", "text": str}]}` —
  the exact prompt the engine reads for that pair's declared shape (the fixed frame around the content; the
  post-processor's tokens are the engine's). Rerank: `{"rows": [{"index", "shape": "pair", "query": str,
  "documents": [str, ...]}]}` — the spans the client ships (the query as the recipe's instruction mode folds
  it, the documents as shipped; the frame is the engine's own template). One input per shape is rendered (the
  row's query for the query shape, its first document for the document shape).
- `--mode score` — rerank: `{"rows": [{"index", "scores": [float, ...]}]}` on the recipe's
  `reference.score_scale` (probability | logit | cosine), one score per document in the order the pairs file
  gives them.
- `--mode embed` — embedding roles: `{"rows": [{"index", "query_vectors": [...], "document_vectors": [...]}]}` —
  per text: one vector for a dense embedder, one per-token matrix for a late-interaction model (the same
  nesting for query and document sides, for every text of the row).
- The reference environment: `rcp-ndcg-vllm/requirements-reference.txt` pins it for every recipe
  (torch, transformers, sentence-transformers as needed); a recipe may ship its own
  `recipes/<id>/requirements-reference.txt`, which the node's bootstrap installs for that recipe instead of
  the shared one. The harness documents both and installs neither.

## Choosing how vLLM serves a model

The serving path is chosen per model — flags alone, a chat template, pooler settings, or a plugin — and the
decision tree lives in this section once the survey of model families lands; for now, a recipe's `serve` section
renders verbatim into `vllm serve` argv, and `serve.plugin` is reserved for a `vllm.general_plugins` package when
no flag can express the model's scoring (the first one ships: `rcp-ndcg-vllm-pplx`, which registers
perplexity-ai/pplx-embed-v2-context-9b-preview's per-chunk pooling head on the stock image; its README carries
the token-id client contract).

## Worked example: a last-token-pooling embedder (CPU)

A last-token-pooling embedder's anchor is its trailing end token: the wrapper renders `prefix + text + suffix`
and the naive fix — truncating the whole string on the right — drops the token the model was trained to read
out of. The correct cut reserves the suffix, cuts only the text, and re-attaches the template. The mechanism in
ten dependency-free lines (the real implementation is the product's `fit` in `rcp_ndcg.data.preprocess` — the
same call the role clients make, with the recipe's real budget — audited against the captured wire in stage 1):

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

Stage 1 probes the recipe's role client for every sampled input through the product's injection point (the
engine when `--base-url` is given, the product's offline fake otherwise) and audits what the client actually
sends — the harness re-derives no render, no cut and no settlement. Each declared shape is sampled on its own
(at least 20 over-length inputs per shape, padded in that shape's own content span), and:

- `anchor_check` asserts every anchor survived the client's cut, on the captured requests of every shape —
  the rendered prompt's edge for the embed roles, and for a reranker the settle-once query (one settled span
  per row, identical across the row's pointwise requests, within its declared `query_max_tokens`, and no cut
  on an in-budget pair);
- `render_check` compares the reference subprocess's `render` output against the captured texts, zero
  tolerance — every declared shape of every pairs-file row (a row carrying the per-row `shape` field is
  compared too; the injected over-length samples are audited, not compared). Under a declared
  `reference.known_deviations: [anchor_drop_over_cap]`, the rows the client had to cut are reported in a
  separate non-gating table here as well (the reference renders them its own way by declaration);
- `engine_tokenize_check` (R29, needs the engine) requires the engine's `/tokenize` ids and counts of every
  captured text to equal the recipe tokenizer's; reported `not_run` without an engine, never as passed;
- `template_render_check`, when `serve.chat_template` is set: the template file's jinja2 render (the engine's
  settings) against the declared template's render, for every declared shape.

Stage 2 sends the same pairs through the product's role clients (`EmbeddingClient`, `PoolingClient`,
`RerankClient`) built from the recipe's real budget — the client prompts, fits and settles exactly as the
served path does (for a reranker, the shared query span settles once per call) — and applies the gates:
probability |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all; logit |Δ| ≤ 0.05·(1 + |s|); cosine scores
|Δ| ≤ 0.01; vectors cosine ≥ 1 − 1e-3 per vector (per token, after the same float16 cast); median per-query
Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these. With
`reference.known_deviations: [anchor_drop_over_cap]`, the inputs the client had to cut (decided on the
client's own census) are reported in a separate, non-gating table — in stage 2 for every role — and the gates
run on the under-cap pairs only.

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
rcp-ndcg-vllm/jobs/submit.sh gs://YOUR-BUCKET/stage/rc0 gs://YOUR-BUCKET/waves <wave-name>
```

The three variables are required — the script refuses to run without them, because no tracked file may name a
machine's paths or a token. The recipe list is the staged `<RC>/wave-lists/<wave>.txt`; `--priority dev-high`,
`--max-jobs N`, `--script wave0` (the node test) and `KJOBS=echo` (print the plan instead of submitting) are
optional. The full procedure — the candidate, the wheelhouse, the three environments and wave 0 — is in
[Release candidates and the GPU waves](release-candidates.md). The wave packs recipes onto the node's GPUs,
serves one engine per slot with its own GPU slice, port, `VLLM_PORT` and `TMPDIR`, checks the free disk and
evicts each model's weights after its last use, runs smoke, equivalence and the recorder, and writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary.
