<!-- Handover copy of the operator's working note `shake/FINDINGS.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Shakedown findings (operator-classified; read the rows for your recipes)

shake1 = int-recipes + rfc-0001 (rc-build); waves shake-a1/shake-a2: T0 smoke + recorder (no pairs yet; equivalence follows).
Rows appear here as the waves report. "none yet" means the wave has not reported your recipe.

## shake1 (20:42): both waves failed in the bootstrap's plugin step before any engine started
- HARNESS (rc-build follow-up, not a family lane): `rcp_ndcg_vllm.jobs.plugins collect` loads every recipe of the wave
  list up front and one recipe that fails validation kills the whole job; the runner promises "one failing recipe never
  stops the wave". Fix: collect skips (and reports) an invalid recipe; the wave runner marks it failed with the message.
- octen-embedding-8b, qwen3-embedding-0.6b: `{fixed: ""}` refused (TemplateSpec). Fixed by recipe-common (content-final
  shape + add_special_tokens: true); shake1b carries that fix.
- qwen3-vl-embedding-2b, qwen3-vl-reranker-2b, topk-embed-v1-small: `recipe.input declares image but the client config
  carries max_images: 0` (and `max_videos: 0` for qwen3-vl-embedding) -> sweep-recipes #7: declare the client media
  policy (fam-vl, fam-late). shake1b uses a SHAKEDOWN-ONLY `max_images: 1` (+ `max_videos: 1`) to test engine start.
- topk-embed-v1-small: `serve.plugin: topk-embed-vllm` names no built wheel; the distribution is `rcp-ndcg-vllm-topk`
  (FOLLOWUP-1 item 2; fam-late). shake1b uses that name.
- zerank-1-reranker: `serve.convert (classify) serves an embed or classify endpoint, not a reranker; a rerank recipe
  declares the checkpoint's scorer through engine.hf_overrides` (fam-zerank). Not in shake1b.
- HARNESS: bootstrap installs a not-staged-as-file plugin by name from an index (pip: no matching distribution) although its wheel is in the staged wheelhouse; shake1b patch: `--no-index --find-links $STAGE_DIR/wheelhouse`.
- HARNESS (BLOCKER for every reference): the reference venv (`--system-site-packages`) got `torch 2.14.0+cpu` from the
  wheelhouse over the image's CUDA torch 2.13 (pip: vllm requires torch==2.13.0) — references would run on CPU torch.
  shake1c patch: constrain the reference install to the image's torch/torchvision/torchaudio/triton pins.
- HARNESS: run_wave `_disk_check` crashes (`FileNotFoundError: /root/.cache/huggingface/hub`) on a fresh pod: the
  cache does not exist before the first download. shake1c patch: measure the nearest existing parent.
- HARNESS: with the torch pin, pip then fails resolving the image torch's declared deps (`nvidia-nccl-cu13==2.29.7` not registered in the image): the reference install must not resolve the image stack at all. shake1c patch: `--no-deps` under the image's full freeze as constraints + an import/CUDA check that fails loudly.
- HARNESS: sentence-transformers needs scikit-learn (+scipy, joblib, threadpoolctl), absent from the image but in the wheelhouse; with --no-deps nothing pulls them. shake1c patch: complete the venv's OWN dists' missing deps to a fixed point (--no-deps each; the image's dists are never completed — that would shadow its CUDA stack).

## shake1c (21:06, 17 recipes on 2 x 8 GPUs; results in <operator-notes>/shake/shake1c/<wave>/<id>/)
12 recipes: engine up, smoke PASSED, recorder PASSED (the first real observation corpus: shake/shake1c/*/vllm-0.31.0/),
equivalence skipped (no pairs files yet). Failures:
- HARNESS (BLOCKER, ctxl 1b/2b/6b): the engine exits: `zmq.error.ZMQError: ipc path "/tmp/rcp-bootstrap.<x>/wave/<id>/tmp/<uuid>"
  is longer than 107 characters (sizeof(sockaddr_un.sun_path))` — the per-slot TMPDIR must be short (e.g. a short
  per-slot dir under /tmp), any recipe id length; test it with the longest recipe id.
- HARNESS: every recipe's `steps.serve.state` is `failed` even when the engine served, smoke and record passed, and the
  engine was stopped cleanly: the serve step never records success (or treats the stop's exit code as failure).
- fam-late, topk-embed-v1-small: the plugin model fails to load: `ValueError: Following weights were not initialized
  from checkpoint: {'custom_text_proj.bias'}` (serve.log in shake/shake1c/shake-a1/topk-embed-v1-small/).
- fam-late, pplx-embed-v2-context-9b-preview: `ValueError: Unsupported dtype torch.float32, expected bfloat16 or
  float16` — the recipe's dtype is not servable by vLLM v0.31.0's model; decide the served dtype against the reference.
- PRODUCT (perf, offline fake): `rcp_ndcg.inference.fake` multi-vector pooling draws one sha256 per scalar (`_unit_vector` -> `fake_uniform`): 2048 dims x 16k tokens = 33.5M hashes per text -> recipe stage-1 tests hung for tens of minutes. Fix: one seeded draw per vector (numpy), keeping determinism; any pinned fake values move deliberately. Owner: lane fake-engines (it owns the fakes).

## OPERATOR DECISION for the recipe families (09:2x) — binding, read before your next round
References stay the paper's or the model card's: a reference must never port the product client's cut (settle rule,
query share) to make over-cap rows pass. recipe-common's commit f1d1e0a did exactly that in
`qwen3-reranker-8b/reference.py` and `zerank-1-small-reranker/reference.py` ("Port of the product's rerank client"):
fam-qwen3-rerank and fam-zerank restore the paper's cut there (keep the span OUTPUT format the harness now expects:
`{"index","shape":"pair","query","documents"}` with the paper's cut applied). Where the paper's over-cap cut differs from
the client's while the anchors are kept, declare `reference.known_deviations: [over_cap_cut_differs]` (new; harness
change in lane/recipe-sweep 9b6bfb5 — merge `lane/recipe-sweep` to get it); `anchor_drop_over_cap` only when the
reference really drops an anchor. Under-cap rows still gate exactly.

- G1: `request_shape: token_ids` bodies (`{"input": [[ids]]}`) crash `stages._anchor_check` (`TypeError:
  TextInputSequence must be str`, via `tokenizer.ids` on a list of ints): the audit must read the SENT IDS directly.
- G2: `messages`-route bodies extract as an empty input list -> the anchor audit checks nothing and PASSES vacuously
  (`checked: 0`): extract the texts (and media placeholders) from messages; a check that checked 0 inputs is a failure.
- G3: the `anchor: first` head-edge audit compares the fixed head's STANDALONE ids verbatim against the joined render's
  head; the head's trailing join-space merges into the first content token on mergey tokenizers -> false failures.
  Compare on the product's own overhead accounting (the join-merge it already documents).
- Plus (recipe-common found it): `stages._over_length` re-tokenizes a growing 65k-token text at every step (quadratic).
  today's workaround: copy the wheel into `<RC>/wheelhouse/`). -> rc-fix.
