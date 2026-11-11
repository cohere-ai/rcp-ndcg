# Judges, the judgement store and estimates

rcp's contract with a judge is one OpenAI-compatible URL. You serve it with any engine and image you
choose: vLLM, or a gateway in front of several workers. You can also use a hosted API. Describe the judge in a
judge config. Run the judging steps here, or through a job runner on SLURM or Kubernetes ([runs and job runners](runs.md)). For your own engines and your own judge, the package never builds an image. It never pins
an engine and never translates engine flags. For the shipped retrieval models,
`rcp-vllm serve <recipe-id>` composes the engine command from the GPU-validated recipe ([serve a retrieval model](../how-to/serve-a-model.md)). Anything else is your command, verbatim.

## The judge config

`rcp.judging.JudgeConfig` describes one judge:

| Field | Meaning |
|---|---|
| `base_url`, `model` | the OpenAI-compatible endpoint (`.../v1`), or a list of replica URLs of the same model, and the served model name |
| `api` | the wire adapter that speaks the endpoint's protocol, resolved within the judge role's registry; unset (the default) is the judge's `openai_chat` wire (`POST {base_url}/chat/completions` over the shared transport; [the inference layer](inference.md)). A third-party adapter from the `rcp.adapters` entry-point group (entries named `judge.<name>`) enters the identity: it decides what is computed |
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
| `image_processor` | the model's image processor family (`qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `gemma4`); the client sizes every image as it does ([preprocessing](preprocessing.md)) |
| `allow_floating_model` | accept an undated model alias on the OpenAI API (`gpt-5`); by default only a dated snapshot (`gpt-5-2025-08-07`) is accepted, since an alias moves between snapshots and its judgements are not reproducible |
| `wait_on_outage_s` | how long a request waits while every replica is down, counted from its first failed send (time queued behind `concurrency` never counts); the default is 1800 s (an engine restart plus a large model's load), and `None` waits indefinitely, except in a job that starts the judge's engine, where the wait is that engine's `outage_timeout_s` |

Only the content fields enter the judgement identity. They are model, revision, the title rule, the sampling
settings, context, tokenizer, image processor, and the offline judge's seed. The text-formatting rule
(`TEXT_FORMATTING_VERSION`) enters too. It is code, and a changed join or instruction frame is a new
instrument. The transport, the URLs included, can be retuned between runs, and a store still resumes.

### Three ways to name a judge

Every way ends in the same `JudgeConfig`. Only the source of the `client` block differs. A mapping that names
a recipe (`recipe: <id>`) takes its whole client block from the recipe. A CONTENT field set explicitly must
equal the recipe's, or the config is refused naming both values. Two run-level fields are the exceptions. The
first is `model`, the served model name that the engine answers under. The second is the
`chat_template_kwargs.enable_thinking` switch in `extra_body`. The recipe's own `extra_body` must travel with
the switch.

1. **A shipped judge recipe**: `--judge gpt-oss-120b`, `--judge recipe:gpt-oss-120b`, or `judge: recipe:gpt-oss-120b` in
   a run config. The recipes are `rcp-vllm`'s package data. They are the ten judges of the catalog: the six
   paper/owner judges plus the four Gemma 4 judges. There is one family per model family, and every variant has its own
   id. The recipe's `client` block is the judge config, validated here and never a second model. Its `serve` block is
   the engine argv (`rcp-vllm serve <id>`). Its `base_url` stays unset until the run supplies it:
   `--set judge.base_url=...`, a run's `serve:` block, or `RCP_NDCG_ENGINES`. The `recipe` pointer in the judgement
   identity is the recipe's id. Judgements therefore record the server-side settings that they came from.
   `rcp-vllm serve <id> --dry-run` prints the exact argv. The recipe's notes carry the memory arithmetic for one
   B200 and one H100, the default serving environments. The recipes are an optional extra of this package.
   `pip install "rcp[vllm]"` installs the sibling `rcp-vllm`. A `recipe:` config on an install without it is
   refused, and the refusal names that install line. The import is lazy, so nothing else needs the extra.
2. **Your own recipe directory**: `--judge recipe:./my-family`, `recipe:../my-family/family.yaml` or `recipe:/abs/path`.
   A family directory of your own loads through the same schema, families included. It is marked unshipped with
   `status: unverified`, and it is identified by the content hash of its resolved form. Two runs whose files differ
   therefore never share a run identity. The judge route reads a directory with exactly one variant. A multi-variant
   family is refused by name. `--variant <id>` selects a size for `rcp-vllm serve`, and the judging commands do not
   take it. Name the shipped variant id for a multi-variant family of the catalog.
3. **A plain judge config file, or a hosted vendor profile**: `--judge ./my-judge.yaml` for any OpenAI-compatible
   endpoint. The `--judge gpt5_hosted` option names the shipped OpenAI profile. The self-hosted presets are gone,
   because they became recipes. `gpt5_hosted` stays a vendor profile.

The shipped recipes, as a catalog (the ids are the ones `--judge` and `rcp-vllm serve` take):

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

Some weights do not fit one GPU. Such a recipe declares the smallest tensor parallel size that fits one B200
with a useful cache. It documents its H100 shape as a `serve --set resources.gpus=<n>` override in the recipe
notes. One B200 or one H100 is the default serving environment, and throughput scales by replicas. The table's
"the model's" context is the paper's TREC-DL preset, which declared no `context_tokens`. Documents are sent
whole there, and the engine's own context bounds the window. The pass warns that no window budget is counted.

Every shipped judge recipe sends its checkpoint's own generation defaults. The `client` block's `temperature`
and `extra_body` (`top_p`/`top_k`) are the values that the checkpoint's `generation_config.json` declares at
the recipe's pinned revision. A config without sampling parameters is greedy, so that recipe sends
`temperature: 0.0` and no `top_p` /`top_k` (gpt-oss-120b). `max_output_tokens` is the recipe's own declared
cap, not the config's. The sampling settings are content fields of the judgement family. A judgement made
under a changed sampling setting therefore never pools with one made before it.

The shipped vendor profile `gpt5_hosted` sends no temperature. It constrains answers with the OpenAI API's
`response_format`. It needs `OPENAI_API_KEY` in the environment, and the key is never written anywhere.

```python
from rcp.judging import JudgeConfig

judge_cfg = JudgeConfig(base_url="http://127.0.0.1:8000/v1", model="my-model", context_tokens=131072)
assert judge_cfg.temperature is None  # no temperature is sent: the server's default sampling
replicas = JudgeConfig(base_url=["http://node1:8000/v1", "http://node2:8000/v1"], model="my-model")
assert replicas.urls == ("http://node1:8000/v1", "http://node2:8000/v1")
recipe_cfg = JudgeConfig.load("recipe:gpt-oss-120b")  # the recipe's client block; base_url arrives at runtime
assert recipe_cfg.recipe == "gpt-oss-120b"
```

`JudgeConfig.load(name_or_path)` reads a recipe (`<id>` or `recipe:<id-or-path>`), a shipped profile by name,
or a YAML config by path. `JudgeConfig.fake(seed=0)` gives the offline fake judge (`fake://`, model `fake`).
The fake chat route below the transport answers it ([the inference layer](inference.md)). The seed decides
every draw of the fake route, so it is content. It enters the judgement family and the identity, and two seeds
never share a store. A real judge's model name also keeps its judgements from pooling with the fake's.

## Serving a judge

Any engine works that serves the OpenAI chat-completions API under the judge's `model` name. Pin the image tag
or the engine version that you serve with. Correct engine behaviour comes from a correctly versioned engine.
Three settings matter to the judge:

- **The served model name** must be the judge config's `model`.
- **The reasoning parser.** A reasoning judge with `decoding: json_schema` must be served with its reasoning parser.
  The reasoning then goes to its own channel, and the answer schema constrains only the answer.
- **The per-request media limit.** A judge that reads page images or video must allow its `max_images` and
  `max_videos`. The client sizes every image itself, so no pixel or processor flag is needed
  ([preprocessing](preprocessing.md)).

For the shipped judges with vLLM, the serve block is the recipe's own. `rcp-vllm serve <id>` renders it,
and `--dry-run` prints it. Each recipe's notes state the flag citations and the memory arithmetic:

```bash
# the recipe's exact argv (dry-run prints it; a real serve execs it)
rcp-vllm serve gpt-oss-120b --dry-run
rcp-vllm serve qwen3.8-27b-fp8 --dry-run

# the same engine by hand, for the second judge
vllm serve openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b --reasoning-parser openai_gptoss --max-model-len 131072 --port 8000
```

A recipe's flags live in its `serve.extra_args` / `serve.limit_mm_per_prompt`. They are the quantisation, the
KV-cache dtype, the reasoning parser and the media limit. `serve --set` names only the deployment fields: the
GPU count, the port and the scheduling knobs. `--set resources.gpus=<n>` is the documented H100 shape where a
recipe's notes name one. The client side still declares `max_images` /`max_videos` (`--set judge.max_images=10`), and the engine's `--limit-mm-per-prompt` must allow them. A `video_url` corpus also
counts videos in the limit, because the engine decodes its containers. The limit must admit a whole window.
The shipped video schedules put 5 documents in a request, and the tournament's adaptive window does too. Both
the client's `max_videos` and the engine's `--limit-mm-per-prompt` therefore need at least 5 (`{"video": 5}`).
The video policy refuses to run unless the judging config declares `engine_video_pinning: true`. The engine
must be pinned to the video policy's own sampling. That is a uniform `num_frames` (`--media-io-kwargs '{"video": {"num_frames": 8}}'` on vLLM), or the engine's rate, `fps` (`--media-io-kwargs '{"video": {"fps": 2}}'`). The Qwen3-VL video backend samples by fps and ignores `num_frames`. Frame-directory corpora need
neither, because their frames are sent as images.

Inside one node, use the engine's own data parallelism for one URL per node (vLLM `--data-parallel-size`).
Across nodes, run independent replicas and list their URLs (below). The paper's judges ran on SGLang. The
paper's submission code is the record of those engine commands. This release serves the same checkpoints on
vLLM v0.33.0.

### Checking an endpoint before a long run

`rcp judge check --judge <recipe|config|fake>` probes an endpoint with two fixed windows. Those windows
are the shipped tournament and rubric prompts over one query and two documents. The command reports three
facts per stage. It reports whether the answer schema was accepted, and whether the answer parsed with the
stage's own parser. It reports whether the endpoint gave a reasoning channel beside the answer (`separated`)
or none (`absent`). With `decoding: json_schema`, `absent` can mean that the engine runs without the model's
reasoning parser. It can also mean that the model does not reason. The command writes nothing and calls the
judge twice.

```bash
rcp judge check --judge recipe:gpt-oss-120b --set judge.base_url=http://127.0.0.1:8000/v1 --json
```

The report's `ok` is true when both windows were answered and parsed. A failed check prints the refusal or the
parse failure, and the command's exit code stays 0, like `rcp doctor`. A script therefore reads the
report's `ok`, which is `data.ok` under `--json`. The envelope's own `ok` says that the command ran, not
that the probe passed. A `--set` of anything but `judge.<field>` is refused.

### What the client checks at run time

- **A refused media count.** An HTTP 400 or 422 says that a request carries too many images or videos. It stops the pass.
  It raises a `CapabilityError` (exit code 8) whose hint names the server's per-request media limit.
- **A missing reasoning parser**. With `decoding: json_schema`, the first 8 answers may carry no reasoning. The client
  then logs one warning that the server probably runs without the model's reasoning parser. A model that does not
  reason, or an API that does not return its reasoning, can ignore it.
- **What the endpoint serves.** At the start of every judging pass the client asks each replica `GET /models`. It
  records the served model id and `max_model_len` when reported. It records the `server` and any version header, and
  the first answer's `system_fingerprint`, where vLLM puts its version. All of it goes into the judgement store and
  into the run manifest's judging steps. A server that does not list the judge's `model` is named in a warning.
  Nothing engine-specific is asked, and none of it enters an identity.

### Several replicas

`base_url` takes a list of replica URLs of the same served model. Each request goes to the live replica with
the fewest requests in flight from this client. The client is normally the only sender, so its counts are
exact. A gateway or a Kubernetes Service is simply a list of one. The same routing, retries and parking are
the shared transport's, described in [the inference layer](inference.md). The judge is one of its roles.

- **A replica that fails** is set aside for a backoff. The backoff doubles while it keeps failing, from 5 to 60 seconds. The
  failure is a connection error, a timeout, or HTTP 408, 429 or 5xx after retries. Its request moves at once to
  another live replica. A replica that answers again is used again.
- **When every replica is down,** requests wait and are re-sent with backoff until one answers, or until
  `wait_on_outage_s` passes. That raises `BackendUnavailableError`. The default is 1800 s, and `null` waits
  indefinitely. A run against dead servers therefore parks, instead of turning the outage into missing judgements. A run
  step's `step_budget_s` bounds the whole step on top (`StepBudgetExceededError`), and the store stays resumable. A job
  that starts the judge's engine (`serve:`) bounds the wait to that engine's `outage_timeout_s`. That value is 900 s by
  default, carried to the step in `RCP_NDCG_ENGINES`. The job then fails, since its engine will not come back on its
  own.
- **A request that keeps failing on a replica that answers other requests** is that request's failure. It is refused and
  recorded like any refused window. A 408 or 429 never is. It is the endpoint's load, so it stays on the outage path.
  The replica is set aside for the server's `Retry-After` when it gives a usable one. A usable value is a finite,
  positive number of seconds. A zero or unparseable header is no wait, so the backoff applies. The request parks and
  retries.

## How the client behaves

- **Refusals.** HTTP 401 or 403 stops the pass with `CredentialsError` (exit code 5). HTTP 404 (no such route or model)
  stops it with `ProviderError`, because both concern every request, not one window. Any other refusal of one request is
  that window's failure. It is asked again up to three attempts, then recorded as an invalid judgement. A resumed pass
  asks it again. One example is HTTP 400 for an over-long prompt. When an earlier attempt carried an answer and a later
  one was refused, the record keeps the answer's text. It also keeps its parse failure, so the store never loses what
  the judge said. `reparse` can read it again. Every request goes over `httpx` through the shared transport; the package
  ships no second HTTP stack. The refusals, the retries and their delays are therefore the transport's, described in
  [the inference layer](inference.md).
- **Answers.** The client records the endpoint's `finish_reason` as it comes: any string or none. Only the answer's
  text is parsed.
- **Usage.** The client counts its requests, their failures and the tokens each answer reports. Those counts go to
  the run manifest and the CLI's judge command. The wire reports tokens and calls only, so the endpoint's cached-input
  token detail (`prompt_tokens_details.cached_tokens`) is not tracked.
- **Window budget.** Each document's text is cut to the tokens its window leaves it, and every cut is recorded
  ([preprocessing](preprocessing.md)).
- **Media.** A prompt with images or video is refused (`CapabilityError`) unless the judge config declares that the
  model reads them.

## The judgement store

`rcp.judging.judge` writes one append-only store per judging pass: `tournament.jsonl` and `rubric.jsonl`
. Each holds one `Judgement` record per window. The store also holds `identity.json`, `preprocessing.jsonl`
and `prompts/<sha256>.txt`. The last holds the text of every prompt that the store was judged with. Each text
sits under the hash that its judgement family records. A custom prompt thus stays reproducible from the store
after its file moves or changes.

- **Record ids.** Every window has a stable `record_id`. That id is a hash of the judgement family, the query, the stage
  and the dataset's identity key. That key is the digest of the store identity's dataset entry below. Two corpora that
  share query and document ids therefore never share a record id. The hash also covers the window's sequence number and
  its placements. A planned window of an insertion plan carries no sequence number. It is keyed by its placements and
  the digest of the schedule it was asked under. Calling `judge` again over the same store asks only for the missing
  windows. Resuming and re-judging a subset of documents (`docs=`) are therefore the same call. A planned window is
  rendered at its own size's text budget. The same window in any plan therefore shows the same text, and it reuses its
  record. A pass with no candidates for a query is refused. `docs` naming no documents is an error. A Stage A pool of
  fewer than two documents is an error. A duplicated query id is an error. None is a silently unjudged query.
- **Identity.** `identity.json` records what produced the store. It records the judgement family. That family holds the
  judge model and revision, prompt hash, criteria, parse version, decoding, preprocessing and tokenizer hash. It also
  holds the title rule, the text-formatting version and the offline judge's seed. It records the judge's declared
  temperature, output and context budgets, extra body and wire adapter, when it declared any. It records the judge's
  content fields, the schedule and the dataset. A row-sequence input has no dataset object, so it names its rows by
  their digest. The identity holds content only. The prompt and the tokenizer enter by their SHA-256. A local dataset
  enters by its path, absolute and normalised. The same file named from another directory (`./rows.jsonl`) is therefore
  one identity. The same bytes copied under another directory are not. The names that the pass was given are kept beside
  it (`sources`). Judging into a store of another identity raises `IdentityError`, and it names the differing fields.
  `force=True` moves the old records aside instead. The file's tag is `rcp.judgement-store.v1`. A store written
  before the id moved keeps `rcp-ndcg.judgement-store.v1`, because a store keeps the tag that it was created with.
  A resumed pass therefore writes the identity back under that tag. The code that wrote the store still opens it.
  Beside the identity, each stage's entry lists what the judge's
  endpoints said they serve (`engines`). That covers the served model id, `max_model_len`, `owned_by`, the `server` and
  any version header, and the first answer's `system_fingerprint`. This is runtime information, and it never enters the
  identity. A store resumed against another engine version therefore adds a report instead of being refused. The dataset
  enters the identity by its name and URI. For a Hub dataset it also enters by the commit that its revision resolved to.
  A moved upstream is therefore refused. A local source has no commit, so it enters by the digest of the files that its
  reader reads. That digest is a file's bytes, or a directory's sorted listing of names, sizes and mtimes. That is
  `rcp.data.revisions.dataset_content_digest`, the same digest that the run steps' identity carries. Editing the
  documents of a local dataset after judging it is therefore refused as another dataset. It never pairs the stored
  answers with texts that the judge never saw. Adding documents to a local source is therefore a new dataset too,
  because its bytes changed. Growing a pool through `candidates=` instead redraws the whole schedule. `windows_for` and
  the balanced draws depend on the pool size. Every random, stratified and adaptive window therefore gets a new record
  id. It is not only the windows that show the new documents. The fit reads the old pool's windows beside the new one's.
  The supported route for a new document is an insertion plan: `select_opponents` with `judge(windows=...)`, then
  `insert_documents`. A fresh store is the other route.
- **Concurrent writers.** The store is reader-safe by design, and it is safe to overlap. Two passes can claim two
  stages, or append records of the same stage. They serialize on an advisory lock on the store directory. The identity
  file and each prompt's text are published through the one atomic temp-file-and-rename helper
  (`rcp.storage.publish`). A killed writer therefore leaves no torn file that a later pass cannot read. A torn last
  record or census row is skipped with a warning, and it is asked or recorded again. A resumed pass that re-asks a
  refused window refits under the new answer. When the answer arrives, the later-phase windows that its first fit
  selected are retired with an appended ``superseded`` tombstone. They are asked again. The fit therefore never reads
  two generations of one query's schedule. The stage file stays append-only, as the mirror's immutable parts require. A
  window that the endpoint keeps refusing leaves the later phases alone, so a resume that changes nothing is idempotent.
  Records are appended as answers arrive. A pass at the judge's `concurrency` therefore appends in arrival order. So
  `read()` returns a stage's judgements in that order, not in the schedule's order.
  The retirement is scoped to the pass's own generation. That is its pool's generation key, a digest of the units that
  the pass judges. It also holds its schedule's window sequences and window sizes. The tournament's mirrors are in the
  sequences, per call. A full pass therefore never retires a subset pass's windows, and a subset pass never retires the
  full pass's. A planned pass with `windows=` runs no schedule, and it retires nothing. The calibration's coverage does
  not count a tombstone as an invalid window, and `run status` counts the live records.
- **Reparse.** Every record keeps the judge's raw answer. `rcp.judging.reparse(store, out)`, or `rcp judge reparse --judgements DIR --out DIR`, reads the stored answers again with the current parser. It writes a new store
  under the current parse version, with its own family key and record ids. It never calls the judge, and it never
  writes into the source store. A source already at the current parse version has nothing to re-parse. A source
  written by a newer checkout cannot be downgraded. Both are refused before anything is written. The command reports
  per stage how many windows were recovered. It reports how many stayed invalid, by category, and how many were
  unchanged or changed. `rcp.judging.reparse` itself returns the new store's
  :class:`~rcp_core.schemas.JudgementSet`.

### A store of your own: `CacheStore`

The judgement store is this package's resumable per-example store. A downstream may need the same
resumption for artifacts of its own. It writes example 4,001, and it resumes at 4,002 after a crash. Such a
downstream implements `rcp.storage.cache.CacheStore` instead of writing a store beside the package. The
protocol has four methods over an opaque string key.

| Method | The contract |
|---|---|
| `get(key)` | the bytes stored under the key, or `None` for a key the store does not hold. A miss is an answer, never an exception. |
| `put(key, value)` | store the value under the key, replacing what the key held. |
| `keys(prefix)` | every key that starts with `prefix` (`""` lists all of them). The order is not part of the contract. |
| `delete(key)` | remove the key: a later `get` never returns it, and deleting a key that is not there is not an error. |

Two implementations ship, and a downstream's own store is the third. `MemoryCacheStore` is a dictionary.
It persists nothing and it validates nothing. `UriCacheStore` is the URI cache itself as a store. It holds the
same entry files that `cache()` populates, under the same publication lock and the same freshness rule. A
reader's downloads are therefore listable (`keys("gs://YOUR-BUCKET/run-1/rows/")`) and deletable:

- a **hit** is an entry whose recorded identity still equals the source's: `get` returns its bytes.
- a **miss** is a key that the cache does not hold, or one whose source has moved. `get` never fetches. A miss
  means that the caller reads through `cache()`, which downloads and publishes the entry that the next `get`
  hits.
- `put` writes the value into the entry, and it records the identity that the source has now. The value is
  therefore served until the source moves. The source object is never written. A key whose source does not
  exist is a `MissingInputError`. An entry with no identity to be fresh for would never be served.
- `delete` removes the payload and both sidecars under the publication lock, so the invalidated bytes are
  never served again. `cache()` fetches them from the source again.

A key that carries a credential is served but never listed. A presigned URL's query string is the example.
The digest names that entry, and the cache directory is shared.

`lookup(store, key)` reads any store defensively. A store that raises is a miss: a backend outage, or an
expired credential. The miss is logged with the store's class and the key. The key is redacted because it may
be a presigned URL. It is never a failed run.

```python
from rcp.storage.cache import MemoryCacheStore, cache_key, lookup

store = MemoryCacheStore()
row = cache_key("run-1", "rows", "004001")  # "run-1/rows/004001"
store.put(row, b'{"query_id": "q1"}')
assert lookup(store, row) == b'{"query_id": "q1"}'  # a hit
assert lookup(store, cache_key("run-1", "rows", "004002")) is None  # a miss
store.delete(row)
assert lookup(store, row) is None  # an invalidated entry is never returned
assert store.keys("run-1/rows/") == []  # ... and it is gone from the prefix
```

A key is opaque. `cache_key` is the package's convention: `/`-separated segments, none of them empty and
none of them holding a `/`. A prefix scan therefore names exactly one namespace.
`keys(cache_key("run-1", "rows") + "/")` lists the run's rows, and not `run-1/rows-2/1`. The conformance test
is `assert_cache_store_contract` in `tests/storage/test_cache_store.py`. It is one contract, run against both
shipped implementations. A downstream's own store therefore has a model to check itself against, with its own
keys.

## Estimating a pass

`rcp.judging.estimate(dataset, candidates, judge, stages=...)` reports the calls, input and output tokens
and wall time of a judging pass. `--estimate` does the same on the command line. Both report before the judge
is called. The estimate counts the same query text that the pass sends. That includes the judge's `title` rule
and the dataset's query-side task instruction. Input tokens are counted exactly with the judge's tokenizer, or
approximated at 2.0 characters per token without one. `input_token_count` says which. `requests_min`
..`requests_max` is the request range. Every window asks once, and an unparseable answer or a refused request
is retried up to three attempts. Each attempt re-sends the prompt, so requests and tokens can reach three
times the one-attempt count. A judge without an `image_processor` has its images approximated at 1,000 tokens
each. The assumptions state that, and the pass warns about it (`APPROXIMATE_IMAGE_TOKENS`). The wall time is a
**floor**. It is a per-call constant at the judge's concurrency, with no term for what a call carries. The
assumptions say so. The CHANGELOG records the measured 150-image window that took 18.9x the projection. The
estimate refuses a judged window that carries more images than the judge's `max_images`. That is exit 8, the
pass's own pre-send gate, because its first real call would be refused. The refusal names the cap, the window
and `--set judge.max_images=...`. A planned pass counts its windows exactly. A scheduled pass counts the most
images that any window of that size could carry. The assumptions name only the stages estimated. A run whose
candidates come from retrieval can be estimated before it retrieves. Each query's pool is then assumed to hold
`candidates.depth` documents, and the assumptions say so.
