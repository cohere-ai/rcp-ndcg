<!-- Handover copy of the operator's working note `GPU-VALIDATION.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# GPU validation plan for 0.0.1 (8x B200, `vllm/vllm-openai:v0.31.0`)

## Principle: the out-of-the-box image, separate environments
Every wave runs on the unmodified `vllm/vllm-openai:v0.31.0` image; no custom image, no build. Three environments on the node, never mixed (owner, 21:11):
1. **Engine environment** = the image's own Python, which runs `vllm serve`. Untouched, with one exception: a recipe's
   pure-Python plugin wheel (vLLM must import it through `vllm.general_plugins`), installed with `--no-deps` only. A
   `pip freeze` before and after must differ by exactly that wheel, or the wave fails before any engine starts.
2. **Client environment** = a separate venv (`uv venv /opt/rcp/client --python 3.12`), holding the staged wheels
   (`rcp-ndcg[hf]`, `rcp-ndcg-core`, `rcp-ndcg-vllm`) installed with the release's constraints file. It runs the
   harness, the recorder, the rcp-ndcg CLI and the wave runner, and talks to the engine over HTTP only; it needs no
   torch.
3. **Reference environment(s)** for the in-process reference implementations of the equivalence and quality checks:
   a venv with `--system-site-packages`, so it reads the image's torch and CUDA without modifying them and installs only
   what is missing into the venv. A reference that needs other versions (the paper's code pins torch 2.9.1 and
   transformers 4.57.6) gets its own fully separate venv from its `requirements.txt`. Each run records which
   environment and versions were used.
`uv` comes from `pip install --target /opt/rcp/uv uv` (or a downloaded standalone binary) and is never installed into
the engine environment; `UV_CACHE_DIR` lives on the node's local disk; install time and the resulting versions are
recorded per wave. The engine and the client are separate processes, the reference runs in its own process.

## Node runtime: isolation and resources (corrections after the owner's review, 21:20)
The miss was one class: treating the node as one shared environment and trusting runtime facts I had not checked. Every
item below is binding for rc-build, gpu-quality, gpu-e2e, fake-engines, the recipe and plugin lanes.
1. **Reuse the product's own isolation.** rcp-ndcg already runs next to an engine without touching it:
   `rcp_ndcg.runners.script.install_argv` runs every `rcp-ndcg` command through `uvx` (an isolated tool environment) and
   bootstraps `uv` with `pip install --target`. The node's client environment uses that same mechanism (one home per
   concept), not a second installer; the waves therefore also exercise the code users will run.
2. **Installing a release candidate before it is on PyPI.** `install_argv` pins `rcp-ndcg==<version>` from PyPI and the
   constraints file from the GitHub release, neither of which exists before the tag. Product change (lane
   fix-review-2): an install source option (a wheelhouse directory or URL passed as `--find-links`, and a local
   constraints file) usable by every runner, documented for pre-release and air-gapped use; the waves install from the
   staged wheelhouse through it.
3. **A wheelhouse, not live PyPI.** rc-build stages every wheel the client and reference environments need
   (`uv pip download` / `uv export` against the lock) next to the release wheels, so a node install is exact, fast and
   independent of PyPI's state that night; the manifest lists their hashes.
4. **References never run inside the client.** The equivalence and quality references (torch, transformers) run as a
   separate process in the reference environment and write their outputs to files the harness then compares; the
   harness's in-process `reference.py` call becomes a subprocess call with a declared environment (gpu-quality).
5. **One GPU, one owner at a time.** vLLM pre-allocates most of a GPU's memory, so a reference on the same GPU would run
   out of memory. Per slot: run the reference first and store its outputs, release the GPU, then start the engine (or
   give references their own GPU slots); record which.
6. **The engine's tokenizer is the truth, not ours.** The client counts tokens with the `tokenizers` version in
   rcp-ndcg's lock and renders templates with its own template-as-data; vLLM tokenizes with the image's transformers and
   renders chat templates with its own jinja. Every recipe's T0/T2 calls vLLM's `/tokenize` (with the same messages or
   prompt) for the observed inputs and requires the client's `fit()` token counts and ids to equal the engine's;
   a mismatch fails the recipe. The observations record those `/tokenize` replies (OBSERVATIONS-SPEC section 1).
7. **Several engines in one pod.** Each slot gets its own `CUDA_VISIBLE_DEVICES`, HTTP port, `VLLM_PORT` (and any other
   internal port vLLM opens), `TMPDIR`, and log directory; engines with tensor parallelism get distinct distributed
   ports. `/dev/shm` is 20 GiB by default: submit with `worker.shared_memory` sized for 8 engines (e.g. 128Gi), and
   record it.
8. **Disk.** The pod has no persistent volume: weights land on the container filesystem. The wave runner measures free
   disk before each recipe, fails early with a clear message if the model will not fit, and evicts that model's weights
   from the HF cache after the recipe (unless a later recipe in the same wave reuses them); free disk is recorded.
9. **Network and credentials on the node.** The node needs the Hub (weights, with the HF token secret) and GCS (the
   mounted auth script); PyPI only as a fallback to the wheelhouse. T0 checks each and fails fast with the reason.
   T0 also round-trips a small file through `rcp_ndcg.storage` on `gs://` (write, list, read, delete under the run
   prefix) from the client environment: the lock moved `oauthlib` 3.3.1 -> 4.0.0 inside the GCS OAuth chain, which
   no offline test exercises.
10. **Process boundaries in the end-to-end runs.** The rendered phased script starts `vllm serve` from the engine
    environment and runs the coordinator through the client mechanism of item 1; gpu-e2e asserts the coordinator never
    imports from the engine environment (it records `sys.prefix` and the versions it saw).
11. **Plugins are self-contained.** A plugin wheel (public or private) vendors its model code and imports only what the
    engine environment ships (torch, vllm, transformers); it must never import the internal training repositories, and
    its installation must leave `pip freeze` unchanged except for itself.
12. **Shared checkouts in the operator's own setup.** Lanes never run commands in another lane's worktree or in the main
    checkout's venv; research lanes that must read code use a detached worktree at a named commit (as the QA lanes do),
    because the main checkout changes under them at every merge.

## Principle: test the release, not a branch
- **Release candidates.** RC0 = `rfc-0001` once the adapters, budget mechanism, wiring, judge port, phases, recipes and
  plugins are merged (finds bugs). RC1 = the final candidate (version 0.0.1, CHANGELOG folded) after RC0's fixes.
- **Artifacts, not sources.** For each RC, build the three distributions exactly as `release.yml` does (same commands,
  same constraints file), stage the wheels plus recipes and plugins to GCS, and let the node install those wheels. The
  tag `v0.0.1` goes on the RC1 commit whose wheels passed.
- **Rerun rule after RC1.** Any change to `inference/`, `data/preprocess|resolution|prepare`, `llm/client`, a recipe or
  a plugin reruns T0-T2 for the affected recipes; any change to `runs/`, `runners/`, `support/serve` reruns T4. A
  docs-only change reruns nothing.

## Tiers (each recipe runs T0-T3; T4 is per scenario)
| Tier | What | Pass criterion | Output |
|---|---|---|---|
| T0 smoke | boot vLLM from `serve_argv(recipe)`; `GET /v1/models`; one request per route the role uses; record engine version, `max_model_len`, dtype | engine up within `startup_timeout_s`; every route 2xx; shapes as the adapter expects | `status.json` |
| T1 recordings (RFC L7) | the recorder's 10 exchanges on one small model per route (dense, multi-vector, pointwise rerank, listwise rerank, VL embed, VL rerank), incl. over-length 400 and float16 base64/bytes pooling | fixtures written, no host names | `tests/contract/engines/vllm-0.31.0/` (committed) |
| T2 equivalence (RFC 6.3) | stage 1 prompt ids incl. >= 20 over-length inputs per shape (anchor check); stage 2 served vs in-process reference on pairs sampled from NanoBEIR, BRIGHT (long documents) and, for VL models, ViDoRe v3; stage 3 subset metrics | gates per score scale (harness); anchors present; paper models also against the stored provenance scores (pools from `top_ranked`), over-cap pairs reported separately where the paper code dropped anchors | `equivalence.json`, `EQUIVALENCE.md` |
| T3 quality ("MTEB across modalities") | the model through **rcp-ndcg's own served path** on the suites as MTEB tasks: embedders in the retrieval view (full corpus), rerankers in the reranking view (released pools, the paper protocol); the same tasks once through the reference implementation (`mteb` with the HF/sentence-transformers model) | served vs reference nDCG@10 within 0.5 points per subset (RCP-nDCG@10 and qrel-nDCG@10); paper rerankers vs the paper's stored per-subset numbers within 0.5 points; published numbers (model cards, MTEB leaderboard) as a sanity column with every deviation explained | `QUALITY.md` per model + one matrix |
| T4 end to end | full `rcp-ndcg run` with `serve:` by role, phases rendered by the runner and executed in the pod (the one-container Kubernetes rule: the supervision script runs engines and coordinator) | every step completes; manifests record each phase's engines; resume after a killed engine parks and recovers; changing engine URLs re-runs nothing; identities and outputs equal between two identical runs | `E2E.md`, run dirs |

## Task matrix (T3)
| Family | Models | Suites and view |
|---|---|---|
| text embedders | zembed-1, pplx-embed-v2-context-9b-preview, Qwen3-Embedding-0.6B, jina-embeddings-v5-text-small, Octen-Embedding-8B | NanoBEIR (13 subsets) retrieval view; BRIGHT (all subsets; long documents stress the text budget) retrieval view; TREC DL 2019/2020 retrieval view |
| text rerankers | zerank-1, zerank-1-small, zerank-2, Qwen3-Reranker 0.6B/4B/8B, ctxl-rerank v2 1B/2B/6B, jina-reranker-v3 | NanoBEIR, BRIGHT, TREC DL reranking view (the paper's pools) |
| visual documents | Qwen3-VL-Embedding-2B (retrieval view), Qwen3-VL-Reranker-2B (reranking view), topk-embed-v1-small (late interaction, float16 `/pooling`; retrieval view) | ViDoRe v3 (all subsets; multilingual) |
| late interaction, text | topk-embed-v1-small (the same model on text documents; the public multi-vector vehicle) | NanoBEIR and BRIGHT retrieval view |
| hosted (operator, CPU, not on the node) | Cohere embed-v4.0 + rerank-v4.0-fast, Voyage | minimal correctness checks only |

## E2E scenarios (T4)
1. **Text, four phases**: NanoBEIR one subset, served encoder (Qwen3-Embedding-0.6B) -> served reranker
   (Qwen3-Reranker-0.6B) -> served judge (vLLM) tournament + rubric -> calibrate + evaluate; limit ~10 queries, depth 30.
2. **Outage**: scenario 1 with the judge engine killed mid-tournament; the run parks, the engine restarts, the run
   finishes; `wait_on_outage_s` expiry path once.
3. **Identity**: rerun scenario 1 with different engine ports; nothing recomputes.
4. **Visual documents**: one ViDoRe v3 subset, served VL encoder -> served VL judge, media budgets on.

## Packing (8 GPUs, one node per wave)
- Wave A, T0+T1+T2 for every public and paper recipe (~20 recipes, 1 GPU each, ~20 min each): ~1 h.
- Wave B, T3 for every model (rerankers on pools are fast; embedders index full corpora; ViDoRe pages dominate): ~2-3 h.
- Wave C, T4 scenarios (the judge takes several GPUs; encoder and reranker phases reuse them): ~1-2 h.
About 6-8 node-hours per RC; RC1 reruns everything once more.

## The GPU run is also the test suite's audit (owner, 20:52 and 20:56)
GPU time produces **observations**; the test suite runs on CPU only. No test needs a GPU.

1. **Observations.** For every recipe, the waves record a corpus of real exchanges with vLLM v0.31.0: requests sampled
   from the suites (short, long and over-length inputs, empty documents, CJK and emoji text, every request shape, media
   for VL models) and the engine's exact replies (status, headers that matter, body; `/pooling` in float, base64 and
   bytes framing; usage fields), plus the error bodies (over-length 400, unknown field, media-count refusal). Each is
   recorded twice; volatile fields are stripped and any non-determinism between the two is measured and recorded.
2. **Verified fake engines** (`rcp_ndcg.testing.engines`, CPU only). One emulator per engine version and recipe,
   selected as `fake://vllm-0.31.0/<recipe>` (the existing hash-based `fake://` stays for tests that need no realism):
   - the **protocol** is emulated, not replayed: routes, request validation, error statuses and bodies, the result
     ordering and framing, usage counts, and token counting with the recipe's real tokenizer files (CPU) — so an
     over-length request fails exactly where the engine fails, and the template the engine renders is rendered the same
     way (the recipe's template);
   - the **model outputs** are replayed for observed inputs, and for unseen inputs come from a declared surrogate
     (deterministic, clearly marked in the reply's metadata), so a test that asserts numbers can only use observed
     inputs.
3. **Verification, offline.** A conformance suite in the root tests replays every recorded exchange against its
   emulator and requires identical status and body (numbers within the recorded non-determinism). An emulator is
   valid only for the engine version and recipe revision it was verified against; it records both, and a changed recipe
   or a new vLLM version needs a new recording wave and a re-verification. The conformance suite runs in CI on every
   push.
4. **What the emulators replace.** Every canned mock body in the adapter, client and pipeline tests that
   `qa-tests` flags as tautological or unrealistic is rewritten against an emulator. The golden replay: one NanoBEIR and
   one ViDoRe subset run through rcp-ndcg's full retrieval and rerank path against the emulators reproduce the GPU
   run's nDCG@10 and RCP-nDCG@10 (to 1e-9, since every input is observed). The phased job script's supervision runs
   with emulators as its engines.
5. **The GPU checks stay harness scripts** on the node (T0-T4, the equivalence gates, the quality tables), and every
   GPU check proves it can fail: each wave serves deliberately broken variants (the chat template removed; an
   engine-side right cut of the rendered prompt; `use_activation` flipped; the wrong pooling; float32 decoded as
   float16; an unpinned `max_pixels`) and requires the gates to fail them; a control that passes is a blocker.
6. **Failure to test.** Every GPU failure is classified and, wherever reproducible, gets an offline test against the
   emulators before it is fixed. After wave A, `qa-tests` re-runs its integration-gap section against the emulators.
7. **Size.** Recordings in the repository stay compact (compressed JSONL; vectors as float16 `.npy`; at most 2 MB per
   recipe, 30 MB in total; the corpus for the golden replay is a sampled subset that fits). Anything larger stays in the
   GCS results, not in the repository.

8. **Forward compatibility when model implementations change** (new checkpoint revision, changed template or pooler
   settings, a new vLLM version, a changed plugin model class such as the private ones):
   - **Two layers, keyed separately.** The *protocol* layer (routes, validation, errors, framing, usage) is keyed by the
     engine and its version (`vllm-0.31.0`); the *model* layer (outputs per input) by a **behaviour fingerprint** of the
     recipe: the SHA-256 of everything that can change what the model returns -- model id and revision, the `serve` block
     (overrides, pooler config, dtype, plugin name and version, `mm_processor_kwargs`), the template file, the tokenizer's
     SHA-256 and the client fields that shape the request. rcp-ndcg's own internals are not in it: the emulators speak the
     wire, so refactoring the package never needs a re-recording.
   - **Staleness is a failure, never a silent pass.** The conformance suite recomputes each recipe's fingerprint from the
     repository and fails, naming the changed inputs, when it no longer matches the corpus's; the only way past is a new
     recording wave or an explicit, dated entry in a waiver file that the release checklist requires to be empty.
   - **Versions coexist.** Corpora live at `tests/contract/engines/<engine>-<version>/<recipe>/<fingerprint>/` with a
     versioned corpus schema (`schema_version`, migrations for older ones); the emulator registry resolves by engine,
     version and fingerprint, so an old and a new implementation can be verified side by side during a migration, and
     old ones are removed deliberately.
   - **Diff-driven re-recording and a behaviour diff.** One command in the wave runner re-records only the recipes whose
     fingerprint changed and writes a behaviour diff against the previous corpus (protocol changes, per-input score or
     vector deltas, changed refusals) for review; the same equivalence gates decide whether the change is accepted.
     registry (an entry-point group), with their corpora in their own repository, verified the same way.

## Lanes this needs
- `gpu-quality`: the T3 stage in `rcp-ndcg-vllm` (served run via rcp-ndcg's CLI, reference run via mteb, comparison
  table, published-number column), the task matrix as data.
- `gpu-e2e`: the T4 scenario configs and driver (render the phased job script, run it in the pod, kill/resume, identity
  re-run, report).
- `rc-build`: the RC procedure as a script (build wheels as `release.yml`, stage to GCS, submit waves from wheels).
- No extra late-interaction models: topk-embed-v1-small (multimodal late interaction) is the public multi-vector vehicle.

## Judges for T4 (owner decision)
- `Qwen/Qwen3.8-27B-FP8` (scenarios 2 and 3, and the text run), and a Flash-Next judge for the main four-phase run:
  `nvidia/Qwen3.8-Flash-Next-NVFP4` if vLLM v0.31.0 serves it well on B200 (the `gpu-e2e` lane establishes that from
  the vLLM source and a T0 smoke), otherwise `Qwen/Qwen3.8-Flash-Next-FP8`. Both served by vLLM; SGLang is covered by
  the same OpenAI-compatible judge path and gets no separate run.

## Secrets on the node (never read into any context)
  script, never echoed); the script redirects the job CLI's output to a file and prints only the job name and status
  lines. The node exposes it as `HF_TOKEN` to vLLM and the reference runs; no script prints the environment, and the
  wave runner's logs never include it (a test greps a wave's logs for the token's shape).
- The operator never cats, greps or reads either file; nothing about them goes into a lane brief except their paths.
