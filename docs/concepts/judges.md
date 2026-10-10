# Judges, the judgement store and estimates

rcp-ndcg's contract with a judge is one OpenAI-compatible URL. Serve it with any engine and image you choose
(vLLM, a gateway in front of several workers) or use a hosted API, describe it in a judge config, and run the
judging steps here or through a job runner on SLURM or Kubernetes ([runs and job runners](runs.md)). For your own
engines and your own judge the package never builds an image, never pins an engine and never translates engine
flags. For the shipped retrieval models `rcp-ndcg-vllm serve <recipe-id>` composes the engine command from the
GPU-validated recipe ([serve a retrieval model](../how-to/serve-a-model.md)); anything else is your command,
verbatim.

## The judge config

`rcp_ndcg.judging.JudgeConfig` describes one judge:

| Field | Meaning |
|---|---|
| `base_url`, `model` | the OpenAI-compatible endpoint (`.../v1`), or a list of replica URLs of the same model, and the served model name |
| `api` | the wire adapter that speaks the endpoint's protocol, resolved within the judge role's registry; unset (the default) is the judge's `openai_chat` wire (`POST {base_url}/chat/completions` over the shared transport; [the inference layer](inference.md)). A third-party adapter from the `rcp_ndcg.adapters` entry-point group (entries named `judge.<name>`) enters the identity: it decides what is computed |
| `revision` | the checkpoint commit; recorded in every judgement |
| `title` | how a document's title reaches the model: `join` (the default, MTEB's `(title + " " + body).strip()`) or `separate` (the title as its own leading text part). Content: the judge reads a different string, so it enters the judgement family and the record ids |
| `temperature` | the sampling temperature; `None` (the default) sends none, so the server's default applies |
| `max_output_tokens`, `extra_body` | the completion cap (reasoning included) and further request fields, e.g. `reasoning_effort` |
| `context_tokens` | the model's context window; sets the per-window text budget, counted with the judge's `tokenizer` ([preprocessing](preprocessing.md)) |
| `tokenizer` | the model's Hugging Face repo id (optionally `@revision`) or a `tokenizer.json` path; text limits, the window budget and estimates count its tokens ([preprocessing](preprocessing.md)); without one nothing is cut and estimates approximate |
| `concurrency`, `timeout_s`, `connect_timeout_s`, `max_retries` | the transport ([the inference layer](inference.md)); `concurrency` is shared by all replicas |
| `api_key_env` | the environment variable that holds the API key; the key itself is never written anywhere |
| `headers_env` | extra headers (e.g. a gateway key), each value read from its environment variable at send time, never from a config, and never logged ([the inference layer](inference.md)) |
| `decoding` | `json_schema`: each request carries the stage's answer schema as `response_format`, and the family records `decoding: json_schema`; an endpoint that refuses the schema fails the pass with exit code 8. `free` (the default): the judge answers in free text |
| `max_images`, `max_videos` | what the served model accepts per request; 0 (the default) means it reads none |
| `image_processor` | the model's image processor family (`qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`); the client sizes every image as it does ([preprocessing](preprocessing.md)) |
| `allow_floating_model` | accept an undated model alias on the OpenAI API (`gpt-5`); by default only a dated snapshot (`gpt-5-2025-08-07`) is accepted, since an alias moves between snapshots and its judgements are not reproducible |
| `wait_on_outage_s` | how long a request waits while every replica is down, counted from its first failed send (time queued behind `concurrency` never counts); the default is 1800 s (an engine restart plus a large model's load), and `None` waits indefinitely, except in a job that starts the judge's engine, where the wait is that engine's `outage_timeout_s` |

Only the content fields (model, revision, the title rule, the sampling settings, context, tokenizer, image processor, and the offline judge's seed) enter the judgement
identity. The text-formatting rule (`TEXT_FORMATTING_VERSION`) enters too: it is code, and a changed join or instruction frame is a new instrument. The transport, the URLs included, can be retuned between runs, and a store still resumes.

### Three ways to name a judge

Every way ends in the same `JudgeConfig`; only the source of the `client` block differs.

1. **A shipped judge recipe** — `--judge gpt-oss-120b`, or `--judge recipe:gpt-oss-120b`, or `judge:
   recipe:gpt-oss-120b` in a run config. The recipes are `rcp-ndcg-vllm`'s package data: the ten judges of the
   catalog (the six paper/owner judges plus the four Gemma 4 judges), one family per model family, every
   variant its own id. The recipe's `client` block is the judge config (validated here, never a second model),
   its `serve` block is the engine argv (`rcp-ndcg-vllm serve <id>`), and its `base_url` stays unset until the
   run supplies it (`--set judge.base_url=...`, a run's `serve:` block, or `RCP_NDCG_ENGINES`). The `recipe`
   pointer in the judgement identity is the recipe's id, so judgements record the server-side settings they
   came from. `rcp-ndcg-vllm serve <id> --dry-run` prints the exact argv, and the recipe's notes carry the
   memory arithmetic for one B200 and one H100 (the default serving environments).
2. **Your own recipe directory** — `--judge recipe:./my-family`, `recipe:../my-family/family.yaml` or
   `recipe:/abs/path`: a family directory of your own loads through the same schema (families included),
   marked unshipped with `status: unverified` and identified by the content hash of its resolved form, so two
   runs whose files differ never share a run identity. The judge route reads a directory with exactly one
   variant (a multi-variant family is refused by name: `--variant <id>` selects a size for
   `rcp-ndcg-vllm serve`, which the judging commands do not take); name the shipped variant id for a
   multi-variant family of the catalog.
3. **A plain judge config file, or a hosted vendor profile** — `--judge ./my-judge.yaml` for any
   OpenAI-compatible endpoint, and `--judge gpt5_hosted` for the shipped OpenAI profile. The self-hosted
   presets are gone (decision 15: they became recipes); `gpt5_hosted` stays a vendor profile.

The shipped recipes, as a catalog (the ids are the ones `--judge` and `rcp-ndcg-vllm serve` take):

| Judge recipe | Weights | Reasoning parser | Context | GPUs (one B200) |
|---|---|---|---|---|
| `qwen3.5-397b-a17b-nvfp4` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | `qwen3` | 262144 | 2 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | `openai_gptoss` | 131072 | 1 |
| `qwen3.6-27b-fp8` | `Qwen/Qwen3.6-27B-FP8` | `qwen3` | the model's | 1 |
| `qwen3.8-27b-fp8` | `Qwen/Qwen3.8-27B-FP8` | `qwen3` | 131072 | 1 |
| `qwen3.8-flash-next-nvfp4` | `nvidia/Qwen3.8-Flash-Next-NVFP4` | `qwen3` | 131072 | 1 |
| `qwen3.8-flash-next-fp8` | `Qwen/Qwen3.8-Flash-Next-FP8` | `qwen3` | 131072 | 2 |
| `gemma-4-12b-it` | `google/gemma-4-12B-it` | `gemma4` | 131072 | 1 |
| `gemma-4-26b-a4b-it` | `google/gemma-4-26B-A4B-it` | `gemma4` | 131072 | 1 |
| `gemma-4-26b-a4b-nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `gemma4` | 131072 | 1 |
| `gemma-4-31b-it-nvfp4` | `nvidia/Gemma-4-31B-IT-NVFP4` | `gemma4` | 131072 | 1 |

A recipe whose weights do not fit one GPU declares the smallest tensor parallel size that fits one B200 with a
useful cache, and documents its H100 shape as a `serve --set resources.gpus=<n>` override in the recipe notes
(owner decision 41: one B200 or one H100 is the default serving environment; throughput scales by replicas).
The table's "the model's" context is the paper's TREC-DL preset, which declared no `context_tokens`: documents
are sent whole and the engine's own context bounds the window.

The shipped vendor profile `gpt5_hosted` sends no temperature and constrains answers with the OpenAI API's
`response_format`; it needs `OPENAI_API_KEY` in the environment (the key is never written anywhere).

```python
from rcp_ndcg.judging import JudgeConfig

judge_cfg = JudgeConfig(base_url="http://127.0.0.1:8000/v1", model="my-model", context_tokens=131072)
assert judge_cfg.temperature is None  # no temperature is sent: the server's default sampling
replicas = JudgeConfig(base_url=["http://node1:8000/v1", "http://node2:8000/v1"], model="my-model")
assert replicas.urls == ("http://node1:8000/v1", "http://node2:8000/v1")
recipe_cfg = JudgeConfig.load("recipe:gpt-oss-120b")  # the recipe's client block; base_url arrives at runtime
assert recipe_cfg.recipe == "gpt-oss-120b"
```

`JudgeConfig.load(name_or_path)` reads a recipe (`<id>` or `recipe:<id-or-path>`), a shipped profile by name,
or a YAML config by path, and `JudgeConfig.fake(seed=0)` gives the offline fake judge (`fake://`, model `fake`,
answered by the fake chat route below the transport, [the inference layer](inference.md)). The seed decides every
draw of the fake route, so it is content: it enters the judgement family and the identity, and two seeds never
share a store (a real judge's model name keeps its judgements from pooling with the fake's).

## Serving a judge

Any engine works that serves the OpenAI chat-completions API under the judge's `model` name. Pin the image tag, or
the engine version, you serve with: correct engine behaviour comes from a correctly versioned engine. Three settings
matter to the judge:

- **The served model name** must be the judge config's `model`.
- **The reasoning parser.** A reasoning judge with `decoding: json_schema` must be served with its reasoning parser,
  so the reasoning goes to its own channel and the answer schema constrains only the answer.
- **The per-request media limit** of a judge that reads page images or video must allow its `max_images` and
  `max_videos`. The client sizes every image itself, so no pixel or processor flag is needed
  ([preprocessing](preprocessing.md)).

For the shipped judges, with vLLM, the serve block is the recipe's own — `rcp-ndcg-vllm serve <id>` renders it,
`--dry-run` prints it, and each recipe's notes state the flag citations and the memory arithmetic:

```bash
# the recipe's exact argv (dry-run prints it; a real serve execs it)
rcp-ndcg-vllm serve gpt-oss-120b --dry-run
rcp-ndcg-vllm serve qwen3.8-27b-fp8 --dry-run

# the same engine by hand, for the second judge
vllm serve openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b --reasoning-parser openai_gptoss --max-model-len 131072 --port 8000
```

A recipe's flags (the quantisation, the KV-cache dtype, the reasoning parser, the media limit) live in its
`serve.extra_args` / `serve.limit_mm_per_prompt`; `serve --set` names only the deployment fields (the GPU count,
the port, the scheduling knobs), and `--set resources.gpus=<n>` is the documented H100 shape where a recipe's
notes name one. The client side still declares `max_images`/`max_videos` (`--set judge.max_images=10`), which the
engine's `--limit-mm-per-prompt` must allow. A `video_url` corpus, whose containers the engine decodes, also
counts videos in the limit (`{"video": 1}`), and its policy refuses to run unless the judging config declares
`engine_video_pinning: true` -- the engine must be pinned to the video policy's own sampling: a uniform
`num_frames` (`--media-io-kwargs '{"video": {"num_frames": 8}}'` on vLLM) or the engine's rate, `fps`
(`--media-io-kwargs '{"video": {"fps": 2}}'`; the Qwen3-VL video backend samples by fps and ignores
`num_frames`). Frame-directory corpora need neither, because their frames are sent as images.

Inside one node, use the engine's own data parallelism for one URL per node (vLLM `--data-parallel-size`); across
nodes, run independent replicas and list their URLs
(below). The paper's judges ran on SGLang; the paper's submission code is the record of those engine commands, and
this release serves the same checkpoints on vLLM v0.31.0.

### Checking an endpoint before a long run

`rcp-ndcg judge check --judge <recipe|config|fake>` probes an endpoint with two fixed windows (the shipped
tournament and rubric prompts over one query and two documents) and reports, per stage: whether the answer schema
was accepted, whether the answer parsed with the stage's own parser, and whether the endpoint reported a
reasoning channel beside the answer (`separated`) or none (`absent` — with `decoding: json_schema` that can mean
the engine runs without the model's reasoning parser, or that the model does not reason). It writes nothing and
calls the judge twice.

```bash
rcp-ndcg judge check --judge recipe:gpt-oss-120b --set judge.base_url=http://127.0.0.1:8000/v1 --json
```

The report's `ok` is true when both windows were answered and parsed; a failed check prints the refusal or the
parse failure, and the command's exit code stays 0 (like `rcp-ndcg doctor`), so a script reads the report's
`ok` -- under `--json` that is `data.ok` (the envelope's own `ok` says the command ran, not that the probe
passed). A `--set`
of anything but `judge.<field>` is refused.

### What the client checks at run time

- **A refused media count.** An HTTP 400 or 422 that says a request carries too many images or videos stops the
  pass with a `CapabilityError` (exit code 8) whose hint names the server's per-request media limit.
- **A missing reasoning parser.** With `decoding: json_schema`, when the first 8 answers carry no reasoning, the
  client logs one warning that the server probably runs without the model's reasoning parser. A model that does not
  reason, or an API that does not return its reasoning, can ignore it.
- **What the endpoint serves.** At the start of every judging pass the client asks each replica `GET /models` and
  records the served model id, `max_model_len` when reported, the `server` and any version header, and the first
  answer's `system_fingerprint` (vLLM puts its version there), in the judgement store and in the run manifest's
  judging steps. A server that does not list the judge's `model` is named in a warning. Nothing engine-specific is
  asked, and none of it enters an identity.

### Several replicas

`base_url` takes a list of replica URLs of the same served model. Each request goes to the live replica with the
fewest requests in flight from this client; the client is normally the only sender, so its counts are exact. A
gateway or a Kubernetes Service is simply a list of one.
The same routing, retries and parking are the shared transport's, described in [the inference layer](inference.md);
the judge is one of its roles.

- **A replica that fails** (a connection error, a timeout, HTTP 408, 429 or 5xx after retries) is set aside for a
  backoff that doubles while it keeps failing, from 5 to 60 seconds, and its request moves at once to another live
  replica. A replica that answers again is used again.
- **When every replica is down,** requests wait and are re-sent with backoff until one answers, or until
  `wait_on_outage_s` passes (`BackendUnavailableError`; the default is 1800 s, and `null` waits indefinitely). A
  run against dead servers therefore parks instead of turning the outage into missing judgements, and a run
  step's `step_budget_s` bounds the whole step on top (`StepBudgetExceededError`, the store resumable). A job
  that starts the judge's engine (`serve:`) bounds the wait to that engine's `outage_timeout_s` (900 s by
  default, carried to the step in `RCP_NDCG_ENGINES`) and then fails, since its engine will not come back on
  its own.
- **A request that keeps failing on a replica that answers other requests** is that request's failure: it is
  refused and recorded like any refused window.

## How the client behaves

- **Refusals.** HTTP 401 or 403 stops the pass with `CredentialsError` (exit code 5), and HTTP 404 (no such route or
  model) with `ProviderError`: both concern every request, not one window. Any other refusal of one request, such
  as HTTP 400 for an over-long prompt, is that window's: it is asked again up to three attempts, then recorded as
  an invalid judgement, and a resumed pass asks it again. When an earlier attempt carried an answer and a later one
  was refused, the record keeps the answer's text (and its parse failure, so the store never loses what the judge
  said and `reparse` can read it again). Every request goes over `httpx` through the shared
  transport (the package ships no second HTTP stack), so the refusals, the retries and their delays are the
  transport's, described in [the inference layer](inference.md).
- **Answers.** The client records the endpoint's `finish_reason` as it comes, any string or none; only the answer's
  text is parsed.
- **Usage.** The client counts its requests, their failures and the tokens each answer reports, for the run
  manifest and the CLI's judge command. The wire reports tokens and calls only: the endpoint's cached-input
  token detail (`prompt_tokens_details.cached_tokens`) is not tracked.
- **Window budget.** Each document's text is cut to the tokens its window leaves it, and every cut is recorded
  ([preprocessing](preprocessing.md)).
- **Media.** A prompt with images or video is refused (`CapabilityError`) unless the judge config declares that the
  model reads them.

## The judgement store

`rcp_ndcg.judging.judge` writes one append-only store per judging pass: `tournament.jsonl` and `rubric.jsonl`, one
`Judgement` record per window, plus `identity.json`, `preprocessing.jsonl` and `prompts/<sha256>.txt`, the text of
every prompt the store was judged with under the hash its judgement family records. A custom prompt thus stays
reproducible from the store after its file moves or changes.

- **Record ids.** Every window has a stable `record_id`, a hash of the judgement family, the query, the stage, the
  dataset's identity key (the digest of the store identity's dataset entry below, so two corpora that share query
  and document ids never share a record id), the window's sequence number and its placements (a planned window of
  an insertion plan carries no sequence number: it is keyed by its placements and the digest of the schedule it was
  asked under). Calling `judge` again over the same store asks only for the missing windows. Resuming and
  re-judging a subset of documents (`docs=`) are therefore the same call. A planned window is rendered at its own
  size's text budget, so the same window in any plan shows the same text and reuses its record. A pass with no
  candidates for a query is refused: `docs` naming no documents, a Stage A pool of fewer than two documents, and a
  duplicated query id are all errors, never a silently unjudged query.
- **Identity.** `identity.json` records what produced the store: the judgement family (judge model and revision,
  prompt hash, criteria, parse version, decoding, preprocessing, tokenizer hash, the title rule and the
  text-formatting version, the offline judge's seed, and the judge's declared
  temperature, output and context budgets, extra body and wire adapter when it declared any), the judge's content
  fields, the schedule and the dataset. A row-sequence input (no dataset object) names its rows by their digest. It holds content only: the prompt and the tokenizer enter by their SHA-256, and a local
  dataset by its path absolute and normalised, so the same file named from another directory (`./rows.jsonl`) is
  one identity -- the same bytes copied under another directory are not; the names the pass
  was given are kept beside it (`sources`). Judging into a store of another identity raises `IdentityError` and names
  the differing fields.
  `force=True` moves the old records aside instead. Beside the identity, each stage's entry lists what the judge's
  endpoints said they serve (`engines`: the served model id, `max_model_len`, `owned_by`, the `server` and any
  version header, and the first answer's `system_fingerprint`). This is runtime information: it never enters the
  identity, so a store resumed against another engine version adds a report instead of being refused. The dataset
  enters the identity by its name and URI and, for a Hub dataset, by the commit its revision resolved to, so a moved
  upstream is refused. Local files have no such pin: editing the documents of a local dataset after judging it is
  your responsibility, since the stored answers would then be paired with texts the judge never saw. Adding
  documents is fine: only windows that show them are new.
- **Concurrent writers.** The store is reader-safe by design and safe to overlap: two passes claiming two
  stages, or appending records of the same stage, serialize on an advisory lock on the store directory, and the
  identity file and each prompt's text are published through the one atomic temp-file-and-rename helper
  (`rcp_ndcg.storage.publish`), so a killed writer leaves no torn file a later pass cannot read (a torn last record
  or census row is skipped with a warning and asked or recorded again). A resumed pass that re-asks a refused
  window refits under the new answer: the later-phase windows its first fit selected are retired with an
  appended ``superseded`` tombstone and asked again, so the fit never reads two generations of one query's
  schedule (and the stage file stays append-only, as the mirror's immutable parts require).
- **Reparse.** Every record keeps the judge's raw answer. `rcp_ndcg.judging.reparse(store, out)`, or
  `rcp-ndcg judge reparse --judgements DIR --out DIR`, reads the stored answers again with the current parser and
  writes a new store under the current parse version, with its own family key and record ids. It never calls the
  judge and never writes into the source store. A source already at the current parse version has nothing to
  re-parse, and a source written by a newer checkout cannot be downgraded: both are refused before anything is
  written. The command reports per stage how many windows were recovered, stayed
  invalid (by category), were unchanged or changed (`rcp_ndcg.judging.reparse` itself returns the new store's
  :class:`~rcp_ndcg_core.schemas.JudgementSet`).

## Estimating a pass

`rcp_ndcg.judging.estimate(dataset, candidates, judge, stages=...)`, and `--estimate` on the command line, report the
calls, input and output tokens and wall time of a judging pass before the judge is called. The estimate counts the
same query text the pass sends: the judge's `title` rule and the dataset's query-side task instruction included.
Input tokens are
counted exactly with the judge's tokenizer, or approximated at 2.0 characters per token without one;
`input_token_count` says which. `requests_min`..`requests_max` is the request range: every window asks once, and an
unparseable answer or a refused request is retried up to three attempts, each re-sending the prompt, so requests
(and tokens) can reach three times the one-attempt count. For a judge without an `image_processor`, images are
approximated at 1,000 tokens
each, stated in the assumptions and warned about (`APPROXIMATE_IMAGE_TOKENS`). The assumptions name only the stages
estimated. A run whose candidates come from retrieval can be estimated before it retrieves: each query's pool is then
assumed to hold `candidates.depth` documents, and the assumptions say so.

