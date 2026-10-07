# Judges, the judgement store and estimates

rcp-ndcg's contract with a judge is one OpenAI-compatible URL. Serve it with any engine and image you choose
(vLLM, SGLang, a gateway in front of several workers) or use a hosted API, describe it in a judge config, and run the
judging steps here or through a job runner on SLURM or Kubernetes ([runs and job runners](runs.md)). For your own
engines and your own judge the package never builds an image, never pins an engine and never translates engine
flags. For the shipped retrieval models `rcp-ndcg-vllm serve <recipe-id>` composes the engine command from the
GPU-validated recipe ([serve a retrieval model](../how-to/serve-a-model.md)); anything else is your command,
verbatim.

## The judge config

`rcp_ndcg.llm.JudgeConfig` describes one judge:

| Field | Meaning |
|---|---|
| `base_url`, `model` | the OpenAI-compatible endpoint (`.../v1`), or a list of replica URLs of the same model, and the served model name |
| `api` | the wire adapter that speaks the endpoint's protocol, resolved within the judge role's registry; unset (the default) is the judge's `openai_chat` wire (`POST {base_url}/chat/completions` over the shared transport; [the inference layer](inference.md)). A third-party adapter from the `rcp_ndcg.adapters` entry-point group (entries named `judge.<name>`) enters the identity: it decides what is computed |
| `revision` | the checkpoint commit; recorded in every judgement |
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
| `wait_on_outage_s` | how long a request waits while every replica is down, counted from its first failed send (time queued behind `concurrency` never counts); `None` waits indefinitely, except in a job that starts the judge's engine, where the wait is that engine's `outage_timeout_s` |

Only the content fields (model, revision, sampling settings, context, tokenizer, image processor) enter the judgement
identity. The transport, the URLs included, can be retuned between runs, and a store still resumes. The shipped
configs ship inside the package (`rcp_ndcg/llm/judges/`) and load by name, from any directory: `qwen35_397b_nvfp4` is
the paper's primary judge (Qwen3.5-397B), `qwen35_397b_fp8` the same model in its FP8 release, `gpt_oss_120b` its
second judge, `qwen36_27b_fp8` its TREC-DL judge (Qwen3.6-27B), and `gpt5_hosted` a hosted judge through the OpenAI
API. The shipped configs send no temperature, as the paper's runs did, and name no engine.

```python
from rcp_ndcg.llm import JudgeConfig

judge_cfg = JudgeConfig(base_url="http://127.0.0.1:8000/v1", model="my-model", context_tokens=131072)
assert judge_cfg.temperature is None  # no temperature is sent: the server's default sampling
replicas = JudgeConfig(base_url=["http://node1:8000/v1", "http://node2:8000/v1"], model="my-model")
assert replicas.urls == ("http://node1:8000/v1", "http://node2:8000/v1")
```

`JudgeConfig.load(name_or_path)` reads a shipped config by name or a YAML config by path, and
`JudgeConfig.fake(seed=0)` gives the offline fake judge (`fake://`, model `fake`, answered by the fake chat route
below the transport, [the inference layer](inference.md)), recorded as model `fake` so that its
judgements never pool with a real judge's.

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

For the shipped judges, with vLLM and with SGLang:

| Judge config | Weights | Served name | Reasoning parser (vLLM / SGLang) | Context |
|---|---|---|---|---|
| `qwen35_397b_nvfp4` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | `qwen3.5-397b` | `qwen3` / `qwen3` | 262144 |
| `qwen35_397b_fp8` | `Qwen/Qwen3.5-397B-A17B-FP8` | `qwen3.5-397b-fp8` | `qwen3` / `qwen3` | 262144 |
| `gpt_oss_120b` | `openai/gpt-oss-120b` | `gpt-oss-120b` | `openai_gptoss` / `gpt-oss` | 131072 |
| `qwen36_27b_fp8` | `Qwen/Qwen3.6-27B-FP8` | `qwen3.6-27b-fp8` | `qwen3` / `qwen3` | the model's |

```bash
# vLLM (the vllm/vllm-openai image runs `vllm serve`)
vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --reasoning-parser openai_gptoss \
  --max-model-len 131072 --tensor-parallel-size 4 --port 8000

# SGLang (the lmsysorg/sglang image)
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --served-model-name gpt-oss-120b \
  --reasoning-parser gpt-oss --context-length 131072 --tp 4 --port 8000

# A Qwen3.5 judge of page images: ten pages per window
vllm serve nvidia/Qwen3.5-397B-A17B-NVFP4 --revision 0368c1b3233414cd4a617b8ff9515e25752dc16c \
  --served-model-name qwen3.5-397b --reasoning-parser qwen3 --max-model-len 262144 \
  --limit-mm-per-prompt '{"image": 10}' --tensor-parallel-size 4 --data-parallel-size 2
python3 -m sglang.launch_server --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 --served-model-name qwen3.5-397b \
  --reasoning-parser qwen3 --context-length 262144 --limit-mm-data-per-request '{"image": 10}' --tp 4
```

and `--set judge.max_images=10` on the judging side. A `video_url` corpus, whose containers the engine decodes, also
counts videos in the limit (`{"video": 1}`), and its policy refuses to run unless the judging config declares
`engine_video_pinning: true` -- the engine must be pinned to the video policy's `num_frames`:
`--media-io-kwargs '{"video": {"num_frames": 8}}'` on vLLM, `--mm-process-config '{"video": {"nframes": 8}}'` on
SGLang. Frame-directory corpora need neither, because their frames are sent as images.

Inside one node, use the engine's own data parallelism for one URL per node (vLLM `--data-parallel-size`, SGLang
`python3 -m sglang_router.launch_server --dp-size`); across nodes, run independent replicas and list their URLs
(below). The paper's exact engine commands, with their images pinned, are in `experiments/paper/serve/` of the
repository.

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
  `wait_on_outage_s` passes (`BackendUnavailableError`). A run against dead servers therefore parks instead of
  turning the outage into missing judgements. A job that starts the judge's engine (`serve:`) bounds the wait to
  that engine's `outage_timeout_s` (900 s by default, carried to the step in `RCP_NDCG_ENGINES`) and then fails,
  since its engine will not come back on its own.
- **A request that keeps failing on a replica that answers other requests** is that request's failure: it is
  refused and recorded like any refused window.

## How the client behaves

- **Refusals.** HTTP 401 or 403 stops the pass with `CredentialsError` (exit code 5), and HTTP 404 (no such route or
  model) with `ProviderError`: both concern every request, not one window. Any other refusal of one request, such
  as HTTP 400 for an over-long prompt, is that window's: it is asked again up to three attempts, then recorded as
  an invalid judgement, and a resumed pass asks it again. Every request goes over `httpx` through the shared
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

`rcp_ndcg.llm.judge` writes one append-only store per judging pass: `tournament.jsonl` and `rubric.jsonl`, one
`Judgement` record per window, plus `identity.json`, `preprocessing.jsonl` and `prompts/<sha256>.txt`, the text of
every prompt the store was judged with under the hash its judgement family records. A custom prompt thus stays
reproducible from the store after its file moves or changes.

- **Record ids.** Every window has a stable `record_id`, a hash of the judgement family, the query, the stage, the
  window's sequence number and its placements (a planned window of an insertion plan carries no sequence number:
  it is keyed by its placements and the digest of the schedule it was asked under). Calling `judge` again over the same store asks only for the missing
  windows. Resuming and re-judging a subset of documents (`docs=`) are therefore the same call.
- **Identity.** `identity.json` records what produced the store: the judgement family (judge model and revision,
  prompt hash, criteria, parse version, decoding, preprocessing, tokenizer hash), the judge's content fields, the
  schedule and the dataset. It holds content only: the prompt and the tokenizer enter by their SHA-256, and a local
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
- **Reparse.** Every record keeps the judge's raw answer. `rcp_ndcg.llm.reparse(store, out)`, or
  `rcp-ndcg judge reparse --judgements DIR --out DIR`, reads the stored answers again with the current parser and
  writes a new store under the current parse version, with its own family key and record ids. It never calls the
  judge and never writes into the source store, and it reports per stage how many windows were recovered, stayed
  invalid (by category), were unchanged or changed.

## Estimating a pass

`rcp_ndcg.llm.estimate(dataset, candidates, judge, stages=...)`, and `--estimate` on the command line, report the
calls, input and output tokens and wall time of a judging pass before the judge is called. Input tokens are
counted exactly with the judge's tokenizer, or approximated at 2.0 characters per token without one;
`input_token_count` says which. For a judge without an `image_processor`, images are approximated at 1,000 tokens
each, stated in the assumptions and warned about (`APPROXIMATE_IMAGE_TOKENS`). The assumptions name only the stages
estimated. A run whose candidates come from retrieval can be estimated before it retrieves: each query's pool is then
assumed to hold `candidates.depth` documents, and the assumptions say so.

