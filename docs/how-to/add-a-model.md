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
  pooler_config: {}
  max_model_len: 8192
  dtype: bfloat16
  extra_args: []                 # further flags, verbatim (one argv element per item)
client:                          # the rcp-ndcg endpoint fields this recipe implies
  api: rerank                    # openai_embeddings | vllm_pooling | rerank
  instruction: fold              # rerank only: none | field | fold
  default_instruction: "Judge whether the document answers the query."
  tokenizer: "<repo>@<40-hex commit>"
  max_tokens: 8192
  normalize: null                # embed and multi_vector only
  embed_dtype: null              # multi_vector only; float16 (default) or float32
reference:
  kind: transformers             # transformers | sentence_transformers | remote_code | stored_scores
  score_scale: probability       # probability | logit | cosine; vectors compare per vector
  entry: reference.py
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
    """Stage 1: the token ids of the exact prompt the engine must see for that pair.
    For an embedding recipe: the ids of doc_prompt + document."""
```

Optionally `tokenizer()`, an object with `encode(text) -> list[int]` and `id_to_token(id) -> str`, so stage 1 can
run on CPU without a Hub download (tests and CI); production references omit it and stage 1 loads the recipe's
`client.tokenizer` with transformers.

## Choosing how vLLM serves a model

The serving path is chosen per model — flags alone, a chat template, pooler settings, or a plugin — and the
decision tree lives in this section once the survey of model families lands; for now, a recipe's `serve` section
renders verbatim into `vllm serve` argv, and `serve.plugin` is reserved for a `vllm.general_plugins` package when
no flag can express the model's scoring.

## The three equivalence stages

Stage 1 (CPU, zero tolerance) renders the recipe's `template.jinja` the way the engine renders chat templates — a
sandboxed jinja2 environment with `trim_blocks` and `lstrip_blocks` on, a stripped template trailing newline, and
undefined variables refused — with the context `{query, document, instruction}` (the instruction variable carries
the recipe's `default_instruction` in `field` mode, and is empty in `fold` mode, where the query text already
carries it), tokenises with the recipe's tokenizer, and requires exact equality with `reference.render`.

Stage 2 scores or embeds the same pairs against the served engine (plain `httpx` to `/rerank`, `/v1/embeddings`,
`/pooling`) and applies the gates: probability |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all; logit |Δ| ≤
0.05·(1 + |s|); cosine scores |Δ| ≤ 0.01; vectors cosine ≥ 1 − 1e-3 per vector (per token, after the same float16
cast); median per-query Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these.

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
packages/rcp-ndcg-vllm/jobs/submit.sh <wave-name> <recipes-file> gs://YOUR-BUCKET/stage gs://YOUR-BUCKET/waves
```

The script stages a tarball of the current commit (plus any directories in `EXTRA_DIRS`, and the recipes file),
uploads it to `<stage-prefix>/<wave-name>/code.tar.gz`, and submits the job `rcp-<wave-name>`; on the node,
`bootstrap.sh` authenticates, installs the packages from the tarball into the image's Python (`--no-deps`, so the
image's vLLM and torch are never touched), and runs `run_wave.py` with `--record` and `--upload`. The wave packs
recipes onto the node's GPUs, serves one engine per slot, runs smoke, equivalence and the recorder, and writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary. `KJOBS=echo` prints the commands instead
of running them.
