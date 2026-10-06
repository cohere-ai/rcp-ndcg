# Judges, serving and runners

rcp-ndcg's contract with a model is one OpenAI-compatible URL. Serve the judge with any engine and image you choose
(vLLM, SGLang, a gateway in front of several workers) or use a hosted API, describe it in a judge config, and run the
judging steps here or through a job runner on SLURM or Kubernetes. The package never builds an image, never pins an
engine and never translates engine flags. A run config's `serve:` section (below) names one engine per role — the
judge, the retrieval encoder, the reranker — and starts your image with your command, verbatim, in the phases of the
run that use it (on SLURM without a container runtime, your command on the node).

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
counts videos in the limit (`{"video": 1}`), and pins the engine's frame count to the video policy's `num_frames`:
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
  window's sequence number and its placements. Calling `judge` again over the same store asks only for the missing
  windows. Resuming and re-judging a subset of documents (`docs=`) are therefore the same call.
- **Identity.** `identity.json` records what produced the store: the judgement family (judge model and revision,
  prompt hash, criteria, parse version, decoding, preprocessing, tokenizer hash), the judge's content fields, the
  schedule and the dataset. It holds content only: the prompt and the tokenizer enter by their SHA-256, and a local
  dataset by its absolute path, so the same file under another name or path is the same identity; the names the pass
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

## Runs and job runners

A run config names the dataset, the candidates, the judge and the steps (`retrieve`, `rerank`, `tournament`,
`rubric`, `calibrate`, `evaluate`). `rcp-ndcg run start my_run.yaml` runs them into `runs/<run_id>/`, whose
`manifest.json` records every step's identity, inputs, outputs and usage (judge calls and tokens). Resuming (`rcp-ndcg run resume --run
runs/<run_id>`) reuses every step whose identity is unchanged, and `rcp-ndcg run status` shows the state of each
step. Estimate a run with `--estimate` before you start it.

`run resume --set KEY=VALUE` (a YAML literal) changes the run's config, and the run keeps the change only if the
resume succeeds. A resume that fails or is refused leaves `run.yaml`, the recorded config and
the run's status as they were; a step it re-ran before failing is redone by the next resume. Once a judging step has
started a stage under the changed config, its store holds judgements of that config, so the change stays and the
run is `failed` until a resume finishes it. `run resume --only STEP` runs just those steps now and never changes the
run's recorded `steps`.

The steps run on this machine by default. `rcp-ndcg run start my_run.yaml --runner slurm` (or `kubernetes`) submits
the run as one job instead, which runs `rcp-ndcg run resume` on the scheduler (`run resume` always runs the run in
the process that calls it); `run status`, `run logs` and `run cancel` follow it, from any process. `--detach` makes
`run start` return at once and leaves the run to be followed with `run status`; without `--runner`, the run becomes a
background job of the local runner. The MCP tool `run_start` always detaches: it returns the run directory, and the
agent polls `run_status`. `rcp-ndcg run start my_run.yaml --runner slurm --dry-run` prints the step plan and the
sbatch script (or, for `kubernetes`, the manifests) that would be submitted, and writes nothing.

The runner's options go in the run config. Each runner validates them with its own model (`LocalOptions`,
`SlurmOptions`, `KubernetesOptions` in `rcp_ndcg.runners`), when the config is read and before anything is written,
so a typo leaves no run directory behind. Three of them describe the job rather than the runner: `resources`
(`gpus`, `cpus`, `memory_gb`, `time_limit_s`), `image` and `env`:

```yaml
runner:
  name: slurm
  options:
    partition: gpu
    account: my-project
    log_dir: logs/slurm
    setup: ["source .venv/bin/activate"]
    resources: {cpus: 8, memory_gb: 64, time_limit_s: 86400}
    env: {HF_HOME: /shared/hf}
```

- **SLURM.** Resources become `--gres=gpu:N`, `--cpus-per-task`, `--mem` and `--time`, and `sbatch_args` passes
  anything else through. With `serve:`, a phase holds its engine replica(s) and, on the phase's first node, the
  coordinator: a node's GPUs are the sum of what runs on it (the coordinator's own request, plus each engine's —
  the largest engine shares the coordinator's node), CPUs and memory add up, and an engine whose `memory_gb` is
  unstated gets the node's whole memory (`--mem=0`), whatever the coordinator asks ([partitioning](#gpus-are-partitioned-per-node-not-shared)). The job runs in the node's
  environment (`setup`) or in a container
  (`container_runtime: apptainer` or `pyxis`, with `container_mounts`); a container runs the stock coordinator image
  below unless `image` names another. Runs, stores and caches live on the cluster's shared filesystem. A node without
  internet access needs the weights and data staged beforehand: set `HF_HOME` to a shared cache and
  `HF_HUB_OFFLINE=1`. The shared cache is filled by the online runs themselves: a dataset run resolves its
  revision online once and records the commit behind the branch (`refs/<ref>` in the cache), so an offline run
  without `--revision` reads from the same cache. A cache with no recorded ref (staged by a download pinned to a
  commit) needs `--revision <full sha>`; until then the run warns (`UNPINNED_REVISION`, on stderr and in the
  `--json` envelope's `warnings`). A file the online run never fetched is an error naming the fix, not an
  empty table.
- **Kubernetes.** Each run is a `batch/v1` Job, applied with the `kubectl` on your `PATH` (and `context`, if set).
  Resources become the containers' requests and limits (`nvidia.com/gpu`, `cpu`, `memory`) and the Job's
  `activeDeadlineSeconds`. A phase container asks for the sum of its engines' GPU requests plus the coordinator's,
  and for the engines' CPUs and memory plus the coordinator's; a CPU or memory amount the engine leaves unstated is
  left unlimited, whatever the coordinator asks, so the coordinator's share never caps the engine; `secrets` are exposed to every container as environment (an HF token, the mirror's
  credentials). The pod's disk is scratch, an `emptyDir` at `/scratch`, so a Kubernetes run needs a `mirror:`
  ([durability](#durability-local-runs-and-a-mirror)). The pod does not see the submitting host's files either,
  so every input must be a URI it can read (`hf://`, `s3://`, `gs://`, `https://`): a run that names a local dataset,
  rankings file, evaluation system, judge config file or prompt file is refused before anything is written, naming
  them. A judge or prompt goes by its shipped name, or inline in the run config. The packaged `tiny` example reads
  local files, so it is not submittable to Kubernetes as it is.

### The coordinator installs itself

The coordinator, the process that runs `rcp-ndcg run resume`, needs no GPU and no image of its own by default:
the job's `resources.gpus` is 0 unless a config declares some, and the runners then reserve exactly what the
config declares (on SLURM a `--gres` on the phase's first node, on Kubernetes a container request —
[partitioned](#gpus-are-partitioned-per-node-not-shared) ahead of the engines' slices). In a container
it runs a stock image with uv and Python 3.12 (`ghcr.io/astral-sh/uv:python3.12-trixie-slim` unless the runner's
`image` names another), which installs the same release as the submitting host when the job starts:

```bash
uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==<version>' \
  --constraints https://github.com/cohere-ai/rcp-ndcg/releases/download/v<version>/requirements-constraints.txt \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  rcp-ndcg run resume --run <run_dir> --mirror <uri>
```

The constraints file, attached to each GitHub release, is the release's `uv.lock` exported, so every dependency is
the version the release was tested with. `unsafe-best-match` lets uv consider the PyTorch CPU index next to PyPI for
every package, so torch resolves to its CPU build at the locked version; without it uv stops at the first index that
has a package and fails. On Kubernetes uv's cache lives on the scratch volume. Without PyPI, install from the
repository instead: `--from 'rcp-ndcg[calibrate,hf,s3,azure] @ git+https://github.com/cohere-ai/rcp-ndcg@v<version>'`
in the larger `ghcr.io/astral-sh/uv:python3.12-trixie` image, which has git.

**Before the release is on PyPI** (a release candidate, an RC wave on a cluster), and on a node without any network
access, the coordinator installs from a **wheelhouse** instead of PyPI: set the runner's `wheelhouse` and
`constraints` options (`runner.options.wheelhouse`, `runner.options.constraints`; the generic `--set
runner.options.wheelhouse=...` overrides them from the command line). The wheelhouse is a directory or an
`http(s)://`/`gs://` URL of staged wheels, readable on the node (on Kubernetes, mounted in or a URL); the rendered
`uvx` then takes everything from the wheelhouse and asks no index, and the constraints file you name replaces the
release's:

```yaml
runner:
  name: kubernetes
  options:
    wheelhouse: gs://my-bucket/wheelhouse/0.0.1rc1    # or /shared/wheelhouse on a shared filesystem
    constraints: gs://bucket/wheelhouse/0.0.1rc1/requirements-constraints.txt
```

The local runner runs the coordinator in this host's environment and installs nothing (a wheelhouse there is
refused); on SLURM the option applies with a container runtime — with `container_runtime: none` the node's own
environment provides the release, and a wheelhouse is refused.

`jobs/rc_build.sh` in `rcp-ndcg-vllm` does all of this in one command (build, checks, wheelhouse, stage,
manifest) — see [Release candidates and the GPU waves](../how-to/release-candidates.md). Built by hand, the
same commands are:

```bash
# the release wheels, as release.yml builds them (the three pyproject.toml versions must match)
uv build --out-dir /shared/wheelhouse/0.0.1rc1
uv build --out-dir /shared/wheelhouse/0.0.1rc1 packages/rcp-ndcg-vllm   # built from its own directory
# the locked dependencies, pinned exactly (the command the committed requirements-constraints.txt records)
uv export --frozen --no-hashes --no-emit-workspace --no-dev --extra calibrate --extra hf --extra s3 --extra azure \
  -o /shared/wheelhouse/0.0.1rc1/requirements-constraints.txt
pip download -r /shared/wheelhouse/0.0.1rc1/requirements-constraints.txt \
  --dest /shared/wheelhouse/0.0.1rc1 --only-binary :all: \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

`uv export --frozen ...` writes the same file the release attaches (the command in its header); `pip download`
resolves it for the node's platform and pulls torch from the cpu index first. Stage the directory to the URL the
nodes read (and mount it for containers), and the job installs exactly the release, whatever PyPI serves that
night.

### Starting the engines with the run

Without `serve:`, the job runs the coordinator only, and each role config's `base_url` is used as it is: your own
engines, gateways or hosted APIs. With `serve:`, the run names one engine per role and the `slurm` and `kubernetes`
runners start each engine inside the run's own job, so the engines and the coordinator are scheduled together and
end together. `judge` serves the judge, `encoder` the retrieval config's encoder, and `reranker` its reranker; the
served model name is that config's `model`, and a role config whose engine is served names no `base_url` — the
job's URLs for it reach the step at runtime, never the config. Setting both is refused rather than silently
overridden:

```yaml
judge: gpt_oss_120b
candidates:
  from: retrieval
  retrieval: {kind: dense, encoder: {api: openai_embeddings, model: octen-embedding-8b}}
  rerank: {api: rerank, model: qwen3-reranker-8b}
steps: [retrieve, rerank, tournament, rubric, calibrate, evaluate]
serve:
  encoder:
    image: vllm/vllm-openai:v0.30.0                # your engine and your tag
    command: [vllm, serve, Octen/Octen-Embedding-8B, --runner, pooling, --served-model-name, octen-embedding-8b,
              --host, 0.0.0.0, --port, "8000"]
    env: {HF_HOME: /models}
    resources: {gpus: 1}                           # per replica
  judge:
    image: vllm/vllm-openai:v0.30.0
    command: [vllm, serve, openai/gpt-oss-120b, --served-model-name, gpt-oss-120b, --reasoning-parser, openai_gptoss,
              --max-model-len, "131072", --tensor-parallel-size, "4", --data-parallel-size, "2",
              --host, 0.0.0.0, --port, "8000"]
    resources: {gpus: 8}
    outage_timeout_s: 900                          # fail the run if every replica stops answering this long
runner:
  name: kubernetes
  options: {namespace: eval, secrets: [hf-token]}
mirror: s3://my-bucket/runs/nano-gpt-oss
```

Every engine has the same fields (`image`, `command`, `env`, `resources`, `replicas`, `port`,
`readiness_path`, `startup_timeout_s`, `outage_timeout_s`), whatever role it serves. The `command` is yours,
verbatim (an argv list or one string): it must serve the role config's `model` name on `port`, on all interfaces
when there are several replicas. `nodes_per_replica` must be 1: a replica that spans several nodes is not
implemented, and another value fails when the config is read. A served encoder or reranker runs one replica (this
release's retrieval clients address one URL); the judge may run several. A hosted model (Cohere, Voyage, Gemini) is
not served by a job's engine at all — drop the role. The `image` is
required on Kubernetes and with the SLURM runner's `container_runtime: apptainer` or `pyxis`. With
`container_runtime: none` (the SLURM default) the command runs on the node itself, so the role's `image` is refused
there: set a container runtime to run the engine in its image, or drop `image` to run the command on the node. The
local runner, and a run in this process, start no engine and refuse phases that would start one: start the
engines yourself and pass their URLs with `run resume --engine <role>=<url>[,<url>]`.

#### Phases

The run's steps run in phases, so the job's GPUs are the maximum over phases instead of the sum over engines.
Consecutive steps that call the same served engines share a phase, and steps that call no served engine (BM25
retrieval, hosted APIs, `calibrate`, `evaluate`) form phases without one. The run above becomes four phases, each
starting only its engines, waiting for readiness, running `rcp-ndcg run resume --run <dir> --only <steps>`, and
stopping them before the next phase starts:

1. `retrieve` — the encoder (1 GPU);
2. `rerank` — the reranker;
3. `tournament`, `rubric` — the judge (8 GPUs);
4. `calibrate`, `evaluate` — no engine.

The engines' URLs reach each phase's coordinator through `RCP_NDCG_ENGINES`, as JSON
`{"judge": {"urls": ["http://node1:8000/v1"], "wait_on_outage_s": 900}, ...}`. The coordinator applies them to the
role configs at runtime only: they are never written into `run.yaml` and never enter a step identity, so a run
resumed by hand is byte-identical with or without the variable. `run resume --engine role=url[,url]` (repeatable)
is the same overlay on the command line, for engines you started yourself. A job that starts an encoder or reranker
engine starts one replica for it (this release's retrieval clients address one replica URL, and more is refused).

#### GPUs are partitioned per node, not shared

A vLLM-class engine pre-allocates most of each GPU it sees (`--gpu-memory-utilization`), so two co-located engines
that both see every GPU fail with out-of-memory. The runner therefore treats a phase's GPUs as a fixed pool that
the engines **partition**, per node:

- **The request.** A node's GPU request is the **sum** of what runs on it: the coordinator's own `resources.gpus`
  (the coordinator runs on the phase's first node, or in the phase's container) plus each engine's `resources.gpus`
  times its replicas there. The job asks for the maximum of that over the phases (SLURM's `--gres` is per node;
  Kubernetes' limit is the container's).
- **The devices.** Every co-located engine process gets a disjoint `CUDA_VISIBLE_DEVICES` slice: with a 4-GPU
  judge and a 1-GPU encoder in one phase container, the container asks for 5 and the judge runs with
  `0,1,2,3`, the encoder with `4`. The slices continue across a role's replicas when they share a GPU set
  (two replicas of a 2-GPU engine: `0,1` and `2,3`, with distinct ports). An engine that declares no GPUs gets
  the empty slice — it sees no device, never all of it.
- **SLURM.** Each role's replicas are pinned to a disjoint slice of the allocation's nodes (one replica per node),
  so no two engine processes share a node; a step's `--gres` is its own engine's count, and SLURM's per-step
  `CUDA_VISIBLE_DEVICES` — set per step with unique devices (gres.html, "GPU Management") — could still overlap
  across steps, because the engine steps run under `srun --overlap`, which srun(1) documents as allowing
  steps to "share all resources (CPUs, memory, and GRES) with all other steps" (SLURM 26.05). The
  coordinator's task therefore claims its own `--gres` (its reservation, with its `CUDA_VISIBLE_DEVICES`
  prefix on the phase's first node), the node pinning keeps the engine steps apart, and a node's `--gres`
  carries the sum of what runs on it; a cluster that constrains devices per step (`ConstrainDevices=yes`)
  should be checked against these slices.

What the runners submit:

| | One replica per engine | Several replicas of an engine |
|---|---|---|
| **Kubernetes** | each engine phase is an init container whose engines run in one container of the (single) engine's `image` (a phase's engines share one image and, if several, need distinct ports), the last phase the main container; the container's command is the supervision script below, which starts the phase's engines and its coordinator side by side and talks to them on `localhost`; `restartPolicy: Never` fails the pod on the first phase that exits non-zero, and the run directory on the pod's `emptyDir` is mounted by every container | as before, plus one StatefulSet of engine pods (`podManagementPolicy: Parallel`) behind a headless Service per role with several replicas, both owned by the Job, so `run cancel` or the Job's TTL deletes them; such engines are **run-scoped** — they live for the whole run, not one phase — and the phases that use them wait for the pods' stable names, at most `startup_timeout_s` until one replica answers; a phase whose engines are all StatefulSet replicas waits in the coordinator's image |
| **SLURM** | one sbatch asking for the maximum nodes and GPUs over the phases, one supervision block per engine phase: each role's engines are one background step (`srun --overlap`, one replica per node, pinned to their slice of the allocation's nodes), the URLs built from the node list, the coordinator on the first node | the same, with the several-replica roles' steps on their slices; the whole allocation holds the largest phase |

### When the engine fails, the job fails

A job that starts its engine never outlives it. Holding an allocation for a dead or hung engine costs more than
starting again: the judgement stores are append-only, a mirror keeps them, and `run resume` asks only for the windows
they lack. The supervision script of each phase, the same on SLURM and in a Kubernetes container, therefore:

- starts each of the phase's engines once, in the background, and never restarts them;
- waits until every role has a replica answering `readiness_path`, for at most `startup_timeout_s` (1800 s by
  default); an engine that exits before it answers fails the job at once, with its status and its own output in the
  job's log;
- then runs the phase's coordinator in the background and ends the phase with whichever ends first: when an engine
  exits, the coordinator is stopped (`SIGTERM`, then `SIGKILL` 20 s later) and the job exits 1 with a message naming
  the engine; when the coordinator exits, the phase's engines are stopped — and reaped — and, only if the
  coordinator exited 0, the next phase starts. A failing phase ends the job with the coordinator's status;
- stops the phase's engines and the coordinator when the scheduler cancels or preempts the job (`SIGTERM` or
  `SIGINT`), so no engine keeps running.

On SLURM, an engine step runs with `srun --kill-on-bad-exit=1 --wait=10`: one replica that fails ends the whole step
at once (one that exits with status 0 ends it 10 s later), and with it the job. On Kubernetes with several
replicas, the StatefulSet restarts an engine pod that dies, and the run relies on the judge instead: a job that
starts engines hands each role an outage wait of `outage_timeout_s` through `RCP_NDCG_ENGINES`, so a judge that
finds no replica answering for that long (900 s by default) stops with `BackendUnavailableError` and a non-zero
exit. The same bound applies to every job with `serve:`, so an engine that hangs without exiting fails the run too.
A judge whose engine the job does not start keeps its own `wait_on_outage_s`.

A failed job is not retried by default. To run it again, engine included, submit the run again with
`rcp-ndcg run resume --run <dir> --runner slurm` (or `kubernetes`): it takes the runner options of the run's last
job and its `serve:` section, restores the directory from the mirror first when the run has one, and the new job
resumes the run where it stopped, asking only for the windows its stores lack. `run resume` without `--runner`
resumes in this process, which starts no engine. On Kubernetes, `backoff_limit` (0 by default) lets the Job retry a
failed pod by itself, and a retried pod resumes the run from its mirror.

The pod of one replica needs nothing from the cluster but an image and a command, so any launcher that takes those
two runs it, and no Kubernetes feature beyond a plain Job is used. The engine image needs `bash` 4.3 or later,
`python3` and `pip`; the stock vLLM and SGLang images have them. The readiness probe uses Python's standard library,
and the coordinator runs in the engine image through `uvx`: an image without uv gets it first with
`python3 -m pip install --target`, which needs `pip` and access to PyPI (an image that has uv needs no `pip`). The
job checks these before it starts the engine. An image without `bash` fails to start the container; one without a
recent enough `bash`, without `python3`, or without both uv and `pip` stops the job at once with a message naming
what is missing. On SLURM, the node that runs the batch script is checked for `bash` 4.3 and `python3` the same way.

### Deployment modes

A suite (`dataset: suite:<name>`) is one run config and one coordinator: the judge's concurrency interleaves the
queries of all its datasets. So the recipe depends on the platform and the number of nodes, not on how many
datasets are judged:

| | One node | Several nodes |
|---|---|---|
| **One dataset or a suite, Kubernetes** | `serve: {judge: {replicas: 1}}`: one engine phase — the engine and the coordinator in one container of the Job's pod, engine-native data parallelism inside the node | `serve: {judge: {replicas: N}}`: N run-scoped engine pods, one coordinator balancing over them |
| **One dataset or a suite, SLURM** | `serve: {judge: {replicas: 1}}`: one sbatch, the engine as a background step | `serve: {judge: {replicas: N}}`: one N-node sbatch, one engine per node |
| **A shared or long-lived engine** | no `serve:`: point `judge.base_url` at it (below) | no `serve:`: list the replicas in `judge.base_url`, or point it at their gateway |

Replicas are independent engines. Do not span one engine's data parallelism over nodes for a mixture-of-experts
judge (both paper judges are): its expert layers then synchronise every forward pass across the nodes.

### The job interface

`rcp_ndcg.runners.get_runner(name, **options)` returns the `local`, `slurm` or `kubernetes` runner, or a runner
that another installed package registers under the `rcp_ndcg.runners` entry-point group. A job is one command line
with its image, resources and environment (`JobSpec.argv`), or the phases to run in order (`JobSpec.phases`) —
exactly one of the two: a job without phases runs `argv`, a phased job's commands are its phases' `argv`.
`render` shows what would be submitted without submitting anything:

```python
from rcp_ndcg.runners import JobPhase, JobSpec, Resources, ServeConfig, get_runner

engine = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command="vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --reasoning-parser openai_gptoss",
    resources=Resources(gpus=8),
)
resume = ("rcp-ndcg", "run", "resume", "--run", "/shared/runs/nano-nfcorpus")
job = JobSpec(
    name="nano-nfcorpus",
    resources=Resources(cpus=8, memory_gb=32, time_limit_s=86400),
    env={"HF_HOME": "/shared/hf"},
    phases=(  # a phased job takes no argv: its commands are its phases' argv
        JobPhase(engines={"judge": engine}, argv=(*resume, "--only", "tournament", "--only", "rubric")),
        JobPhase(argv=(*resume, "--only", "calibrate", "--only", "evaluate")),
    ),
)

slurm = get_runner("slurm", partition="gpu", account="my-project", log_dir="logs/slurm",
                   container_runtime="pyxis")  # the engine runs in the engine's image
print(slurm.render([job])["nano-nfcorpus"])  # the sbatch script

kubernetes = get_runner("kubernetes", namespace="eval", secrets=["hf-token"])
print(kubernetes.render([job])["nano-nfcorpus"])  # the Job: engine phases as init containers, the last phase
# the main container; several-replica engines as run-scoped StatefulSets owned by the Job
```

`runner.submit([job])` submits and returns handles; `runner.status(handle)`, `runner.logs(handle, tail=100)` and
`runner.cancel(handle)` follow them.

## Durability: local runs and a mirror

A run directory and a judgement store (`judge --out`) are written on a local or shared filesystem, never straight to
a bucket: a remote runs directory or `--out` is refused. Inputs are still read from anywhere (`hf://`, `s3://`,
`https://`).

A mirror copies what a run writes to any fsspec URI while it runs, so a job that is preempted loses at most one
interval of work. Set it with `--mirror s3://bucket/runs/<name>` on `run start`, `run resume` and
`judge tournament|rubric`, or with `mirror:` (and `mirror_interval_s:`, 60 by default) in the run config; a job a
runner starts carries it on its `run resume` command line. On Kubernetes the mirror is how the run reaches its pod:
the pod's disk is scratch, so the prepared run directory is uploaded to the mirror before the Job is submitted, and
the pod restores it into `/scratch/runs/<run_id>`. A Kubernetes run without a mirror is refused before anything is
written.

- Every `mirror_interval_s`, and once more when the run ends, also on `SIGTERM` or `SIGINT`, the judgement stores and
  the text census go up as immutable parts: `<file>.parts/<start>-<end>` holds the bytes appended since the last
  upload, up to the last complete line. Every other file (the manifest, `run.yaml`, `identity.json`, the calibration,
  the reports, the log) goes up whole when it changed. `work/` is scratch space and is not mirrored.
- `run resume --run <dir> --mirror <uri>` on a node that lacks the run directory, or holds a shorter store, first
  rebuilds it from the mirror (the parts in offset order; a gap or an overlap is refused with exit 12), and then
  resumes as usual: only the windows the store lacks are asked. A directory whose manifest is older than the
  mirror's (the run went on in a job elsewhere) also takes the mirror's newer whole files. The restore writes into
  the run directory before anything else, also with `--dry-run` or `--estimate`.
- A local or shared path (`/shared/mirrors/nano` or `file:///shared/mirrors/nano`) is a mirror too; its directories
  are created as the mirror writes.
- `run status` shows the mirror's last upload and its lag (`data.mirror.last_upload_at`, `data.mirror.lag_s`).

**Any fsspec filesystem is a mirror target**, because the mirror uses exactly three of its operations: write an
object (`pipe_file`), read an object (`cat_file`) and list a prefix (`ls`). It never asks whether an object exists,
renames or appends. GCS works as installed (`gcsfs` is a dependency), S3 needs `s3fs` (`pip install
"rcp-ndcg[s3]"`) and Azure `adlfs` (`[azure]`); a protocol with no filesystem installed or registered stops the run at
start with exit 10, naming what is missing.
`hf://` works, but every write to the Hub is a commit and its rate limits apply: publish a finished run there, and
mirror a running one to an object store.

Your own storage is a small `fsspec.AbstractFileSystem` with those three methods, registered under a protocol
with `fsspec.register_implementation`, or from your package's `fsspec.specs` entry point:

```python
import fsspec
from fsspec import AbstractFileSystem


class MyStore(AbstractFileSystem):
    """Objects in a dict; a real one would call its service's put, get and list."""

    protocol = "mystore"
    objects: dict[str, bytes] = {}

    def pipe_file(self, path, value, **kwargs):
        self.objects[self._strip_protocol(path)] = bytes(value)

    def cat_file(self, path, start=None, end=None, **kwargs):
        return self.objects[self._strip_protocol(path)]

    def ls(self, path, detail=True, **kwargs):
        prefix = self._strip_protocol(path).rstrip("/") + "/"
        names = {prefix + key[len(prefix) :].split("/")[0] for key in self.objects if key.startswith(prefix)}
        if not names:
            raise FileNotFoundError(path)
        entries = [{"name": n, "type": "file" if n in self.objects else "directory", "size": 0} for n in names]
        return entries if detail else sorted(names)


fsspec.register_implementation("mystore", MyStore, clobber=True)
```

Then `--mirror mystore://team/runs/nano` (or `mirror: mystore://team/runs/nano`) mirrors to it. The same class
serves `rcp_ndcg.runs.Mirror` directly:

```python
from pathlib import Path

from rcp_ndcg.runs import Mirror, restore

Path("run/judgements").mkdir(parents=True)
Path("run/judgements/rubric.jsonl").write_text('{"window": 1}\n', encoding="utf-8")
Mirror("run", "mystore://team/runs/nano").flush()
assert restore("elsewhere", "mystore://team/runs/nano") == ["judgements/rubric.jsonl"]
```

## One engine, many judge jobs

For many runs against one long-lived engine, serve it once as a Deployment behind a Service, and run GPU-less judge
jobs against it. The manifest below serves `gpt-oss-120b` with the stock vLLM image. The judge config of the run
points `base_url` at the Service (`http://judge-engine:8000/v1`); nothing else changes. Engine health comes from
`/health`, and the engine's Prometheus `/metrics` shows running and queued requests and KV-cache use.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: judge-engine
spec:
  replicas: 1
  selector:
    matchLabels: {app: judge-engine}
  template:
    metadata:
      labels: {app: judge-engine}
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.30.0         # pin the tag you tested
          command: [vllm, serve]
          args: ["openai/gpt-oss-120b", "--served-model-name", "gpt-oss-120b", "--reasoning-parser", "openai_gptoss",
                 "--tensor-parallel-size", "4", "--max-model-len", "131072", "--port", "8000"]
          ports: [{containerPort: 8000}]
          envFrom: [{secretRef: {name: hf-token}}]
          resources:
            limits: {nvidia.com/gpu: 4}
          startupProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
            failureThreshold: 360                 # up to an hour to load the weights
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
          volumeMounts: [{name: dshm, mountPath: /dev/shm}]
      volumes:
        - name: dshm
          emptyDir: {medium: Memory}
---
apiVersion: v1
kind: Service
metadata:
  name: judge-engine
spec:
  selector: {app: judge-engine}
  ports: [{port: 8000, targetPort: 8000}]
```

A restarted judge job resumes its run from its mirror, and the judgement store asks the judge only for the missing
windows.
