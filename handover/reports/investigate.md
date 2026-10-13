# Lane investigate — embeddinggemma-2's row-specific gap (CPU diagnosis) and topk-embed-v1's controlled GPU plan

**Lane:** `lane/investigate` (worktree outside the operator's checkout). **Base:** `rfc-0001` at
`c668c439`; **merged:** `rfc-0001` at `973a5f36` (which is the orchestrator's merge of this lane's
first state, so the merge brings in this lane's own earlier commits and no upstream drift). **Gated:**
`37dcf799` (the merged tip).
**Tree:** no product, harness or recipe code touched; this lane's committed artefacts are this report,
the harness (`topk_probe.py` / `topk_probe.sh`) and the fixtures under
`handover/reports/investigate/`. **No GPU, no node submission, nothing pushed, `rfc-0001` not merged
into.**

**Revision note.** Two adversarial reviews (GLM, DeepSeek) landed after the first report and found real
flaws; this revision applies every correction they named: E1 now compares each preprocessing variant
against the engine's pixels; `_first_divergent` reads the shared-tensor feed; E4/E5 compute what their
rows claim; E7 is replaced (this checkpoint has no sliding window); the harness's reference side and the
wrapper's pin read are fixed; a one-row control ties the probe's engine vectors to the wave's; the
rows-31/32 residual is now a proven geometry difference; and Part A's video mechanism is corrected
(the image runs v0.31.0's generic video loader, not the nightly `EmbeddingGemma2VideoBackend`).

## 1. Status

| Part | Item | Outcome |
|---|---|---|
| A | the final RC's `embeddinggemma-2` failures | **diagnosed, proven on CPU** — three independent causes, all named with file/line; **no fix landed** (two need a GPU wave or an owner decision; the third has a small harness fix available and was left for the owner because it changes a check's referent) |
| A1 | stage 1 `template_render_check` (query, row 65) | the client's declared render joins a message's text parts with `TEXT_JOIN` (`"\n"`); the checkpoint's own `chat_template.jinja` joins them with the empty string. The engine's render and the reference's render both have no newline; the client's budget render does. Reproduced offline, same failure, same heads; the engine's recorded `usage.prompt_tokens` (274) is the no-newline arithmetic |
| A2 | stage 2, row 36 document 0 (the video) | the engine shows the model **4 video frames**, the client and reference **32**. The engine's own media count (492 = 4 × 123) proves it; the reference passes the declared pin to the processor's sampling call (32). The document's render strings are identical on both sides. The image's video loader is v0.31.0's generic `VideoBackend` (the pin reaches the decoder only); the nightly `EmbeddingGemma2VideoBackend` would force `do_sample_frames=False` and cannot produce the measured 501 |
| A3 | stage 2, rows 31/32 (0.994–0.996) | **proven input difference**: the client sends the Gemma 4 fixed point (288×2160 → 272 tokens; 13344×48 → 280) and the engine's media deltas match it; the reference's single pass over the raw bytes lands on 288×2112 (266) and 9792×48 (206). All four bad rows are input differences |
| A4 | the media stage's other failures (rows 33, 35) | **a check-arithmetic artefact**, not an engine difference: the engine check's subtraction baseline loses a tokenization merge when a text part follows the media placeholder (+1 token). Proven with the product tokenizer and the recorded usages (274, 290) |
| B | `topk-embed-v1`'s image divergence | **GPU plan + runnable harness** — E0–E7 (E7 replaced: this checkpoint has no sliding window), each isolating one variable, with the exact node job command, the submit wiring (including the job-test updates) and a one-row control against the wave's recorded vectors |

## 2. Part A — embeddinggemma-2: the proven chain

### 2.1 The evidence read (the final RC's wave)

Read-only from the RC's staged wave artefacts: the recorded observation corpus (the client's exact
request bodies and the engine's replies), the reference store (`output.json`), `equivalence.json` and
`status.json`. The blocks this report cites are committed verbatim, host-free, in
`handover/reports/investigate/evidence-embeddinggemma-2.json`.

The two failures, verbatim from `equivalence.json`:

```
stage1.template_render_check: template google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b:chat_template.jinja
  sha256 4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e
  checked 112, passed False, failures 1:
  {"shape": "query", "row": 65,
   "engine_head":   "task: search result | query: which page shows the quarterly chart",
   "declared_head": "task: search result | query: \nwhich page shows the quarterly chart"}

stage2: n_vectors 95, n_comparisons 475, passed False
  min_cosine (full width) 0.9343192260626695   min_cosine (k=128) 0.9593554811550121
  min_cosine (k=256)      0.9463366672212329   min_cosine (k=512) 0.9355271114266014
  min_cosine (k=768)      0.9343192260626695
```

The brief's "10 of 475 vectors" is **20** in the artefact: four document rows (31, 32, 36, 37), five
widths each. The *gate minima* are all row 36 document 0, exactly the four numbers the brief quotes.
Every comparison outside those four rows is ≥ 0.999; every query vector is ≥ 0.9997.

| referent | cosines (full, 128, 256, 512, 768) | what the row is |
|---|---|---|
| row 31 document 0 | 0.9956, 0.9980, 0.9964, 0.9957, 0.9956 | `media:image:tall_receipt` (576×4096) |
| row 32 document 0 | 0.9941, 0.9963, 0.9947, 0.9941, 0.9941 | `media:image:extreme_aspect` (3000×20) |
| **row 36 document 0** | **0.9343, 0.9594, 0.9463, 0.9355, 0.9343** | `media:video:icon` (64×64, 4.0 s, 8 fps, 32 frames) |
| row 37 document 0 | 0.9656, 0.9755, 0.9691, 0.9663, 0.9656 | `media:video:page` (224×224 + text) |

The media stage also failed, which the brief's summary did not mention:

```
media.engine_check.failures:
  row 33 document 0: engine_media_tokens 269  vs client_media_tokens 268
  row 35 query:      engine_media_tokens 259  vs client_media_tokens 258
  row 36 document 0: engine_media_tokens 492  vs client_media_tokens 3936
  row 37 document 0: engine_media_tokens 493  vs client_media_tokens 3936
```

### 2.2 Which rows the two failures are

- **Stage 2 row 36** is the pairs file's row 36 (`rcp-ndcg-test/pairs/embeddinggemma-2.jsonl`,
  0-based): stratum `media:video:icon`, query `"which page shows the quarterly chart"`, one document
  whose text is empty and whose media is a 64×64, 4.0 s, 8 fps AVI with 32 native frames. The
  observation corpus keys it `pairs:36` (`inputs.request_id`), which pins the index.
- **Stage 1 row 65** is *not* a pairs-file row: the stage-1 probe's row list is the 25 text rows + 20
  over-length samples per declared shape (65 rows), then one media-template probe per declared shape.
  Row 65 is the first appended probe — the **query shape's media-template probe** — and it is the
  pairs file's row 35 (`media:query_image`): the query text `"which page shows the quarterly chart"`
  with the 64×64 icon PNG on the query side. Row 66 is the document probe (pairs row 25).
- The two failures **do not share a pairs row**. They share the query string (every media row uses
  it) and the "icon" asset: row 35's query image is the same 64×64 PNG as row 26's document
  (`sha256 0ddb3b11…`); row 36's document is a 64×64 video (`sha256 758780a5…`).

### 2.3 The `template_render_check` failure, reproduced on CPU

Reproduction (offline; the harness's own functions, the product's own recipe/tokenizer/template):

```
$ uv run --no-sync python <a scratch probe outside the repository>
text rows 25  media rows 13  sampled 65  media probes 2
probe row 65: shape=query media-list position 10 (pairs row 35) strata=['media:image', 'media:query_image']

parts in order: [('text', 'task: search result | query: '), 'image_url', ('text', 'which page shows the quarterly chart')]
engine render   : 'task: search result | query: which page shows the quarterly chart'
declared render : 'task: search result | query: \nwhich page shows the quarterly chart'
engine media    : 'task: search result | query: <TOKEN:258880>which page shows the quarterly chart'
reference render: 'task: search result | query: <TOKEN:258880>which page shows the quarterly chart'
first char difference at 29: engine ...'rch result | query: which page shows the'... declared ...'rch result | query: \nwhich page shows th'...
token ids: engine 15 [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 837, 3104, 3831, 506, 42688, 9535, 1]
           declared 17 [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 236743, 107, 7650, 3104, 3831, 506, 42688, 9535, 1]
ids equal: False
```

The same offline `stage1_prompts` run reports the RC's exact failure (`checked 112`, one failure, the
same `engine_head`/`declared_head`), so the check is fully reproducible without a GPU. An independent
discriminator from the recorded reply: the engine's `usage.prompt_tokens` for `pairs:35` query is
**274** — the no-newline arithmetic `17 − 1 + 258 = 274`; the newline variant would be 275.

**The divergence, named.** The client sent one user message with three parts: the task-prompt text,
the image, the query text. The engine's route (`/v1/embeddings` with `messages`, vLLM's
`content_format == "openai"` → `wrap_dicts=True`) hands the template the **parts list**; the
checkpoint's `chat_template.jinja` concatenates a message's text parts with the **empty string**
(its loop renders each text part and each media placeholder back to back; the `join` filter it also
computes is called with no separator). So the engine renders `prompt + image-placeholder + text` —
no newline. The **client's declared render** (the budget render, the product's one home
`rcp_ndcg.data.text_budget.rendered_request` over `Content.text`) joins the text parts with
`TEXT_JOIN` = `"\n"` (`rcp-ndcg-core/src/rcp_ndcg_core/content.py:33`, `:216`), because
`rcp_ndcg_test.equivalence.wire.Capture.texts` flattens a `messages` body's text parts with
`TEXT_JOIN` (`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/wire.py:213-214`, its comment "as the engine
joins them" — true for vLLM's flattened path, not for the `openai` content-format path where the
template does the joining). The client's own fit runs over the same joined string
(`rcp-ndcg/src/rcp_ndcg/inference/clients/_base.py:1144`; the `messages` lowering in
`clients/embed.py:303-310`), so the check does not invent a join the budget does not also use. The
declared render therefore carries two extra ids (`236743, 107`) where the newline sits.

The **reference** renders the same parts the same way: its `side_input` builds a user turn with the
parts in order and the card's prompt as a system message, and the checkpoint template renders system
content then the user parts → `prompt + placeholder + text` (the transcript above prints it). So this
failure is **not** an engine-vs-reference input difference: it is the client's budget render inserting
a newline the model never reads (≤2 tokens per interleaved media row).

Not a small provable bug in our tree: the join is the *checkpoint template's* rule, and the product
has no knob to declare it; changing `TEXT_JOIN` would move a product-wide convention (cuts, chunking,
`split_text_across_parts`) and changing the check would stop it comparing the engine's frame against
the client's budget render. The candidates are an owner decision (see §2.8).

### 2.4 Row 36's document: the renders agree, the frames do not

The same probe, over the recorded request bytes for `pairs:36` document 0:

```
engine render (media) : 'title: none | text: <TOKEN:258884>'
engine render (text)  : 'title: none | text: '
declared render       : 'title: none | text: '
reference render      : 'title: none | text: <TOKEN:258884>'
engine text == declared: True
engine media == reference: True
usage.prompt_tokens (engine): 501
```

- The **client's fit render** is `"title: none | text: "` (the empty document under the doc prompt).
- The **template's render** of the sent conversation is the same string plus the video placeholder
  token (id 258884) — and the **reference's** own conversation shape (the prompt as a system message,
  a video-only user turn) renders to exactly the same string. The text/render inputs are identical.
- The **frame count differs**, and the engine's own report proves it:

```
row 36 video entry: 64x64, num_frames 32, duration_s 4.0, fps 8.0
declared pin fps=60 max_frames=32 -> frames 32
declared pin fps=1  max_frames=32 -> frames 4
checkpoint geometry: fixed point 528x528 -> 1089 patches -> 123 tokens/frame
engine media tokens 492 / 123 tokens-per-frame = 4.0 frames
client media tokens 3936 / 123 = 32.0 frames
row 36 document 0 cosine (recomputed): 0.934319
```

`492 = 4 × 123` and `3936 = 32 × 123`; `123` is the checkpoint's own geometry for a 64×64 frame at
`max_soft_tokens` 140 (`reference.gemma4_fixed_point(64, 64, 140)` → 528×528 → 1089 patches →
1089/9 + 2). The engine's `usage.prompt_tokens` is `501 = 9 (the text render) + 492`.

**The mechanism, with file:line.**

1. The recipe pins the engine's video sampling with
   `serve.extra_args --media-io-kwargs {"video": {"fps": 60, "max_frames": 32}}`
   (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/family.yaml`). The engine log confirms
   the argv (`media_io_kwargs: {'video': {'fps': 60, 'max_frames': 32}}`) and the engine version
   (`0.31.0`).
2. **The image's video loader is v0.31.0's generic `VideoBackend`, not the nightly
   `EmbeddingGemma2VideoBackend`.** The pinned image is a v0.31.0-based build (the recipe's own
   E2-round-4 note; the wave's freeze carries `vllm-0.31.0-…whl`; the serve log reports 0.31.0), and
   v0.31.0's `vllm/multimodal/video.py` has no EmbeddingGemma2 backend. The generic path decodes the
   pinned 32 indices (`video.py:202-250`: `num_frames_to_sample = min(32, 32)`, then
   `min(32, floor(4.0 × 60))` → all 32 frames) and its `create_hf_metadata` (`video.py:181-194`, the
   `do_sample_frames` line at `:193`) sets `do_sample_frames = (len(valid_frame_indices) ==
   total_frames_num)` — **True**, because the pin sampled the whole 32-frame clip. (The nightly's
   `EmbeddingGemma2VideoBackend.create_hf_metadata` — `video.py:581-592` at the nightly commit —
   overrides it to **False**, which would leave all 32 decoded frames in place and report
   `32 × 123 + 9 = 3945`, not the measured 501: that subclass cannot be the image's loader. The
   recipe's own "Media" note attributes the nightly class to the image; this report corrects that,
   and the fix lane should not follow the note's class attribution without a direct read of the
   pinned image.)
3. The **model's** HF video-processor call gets only the recipe's `mm_processor_kwargs` (`{}`). The
   plugin ships the nightly model file (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/embedding_gemma2/
   fold/embedding_gemma2.py:718-727`): `video_mm_kwargs = dict(hf_kwargs)` (empty) plus
   `do_sample_frames` from the metadata (True), then `call_hf_processor(get_hf_processor(**kwargs),
   ...)`. The processor's own defaults are `fps = 1`, `max_frames = 32`, `do_sample_frames = True`
   (the checkpoint's `processor_config.json`; transformers 5.19.0
   `video_processing_embedding_gemma2.py:183,191-192`), so
   `EmbeddingGemma2VideoProcessor.sample_frames` resamples the 32 decoded frames to
   `int(4.0 × 1) = 4` frames `[0, 8, 16, 24]`. The model therefore sees 4 frames — the card's own
   "1 fps by default" policy, not the declared 32.
4. The **reference** passes the pin into the processor's call kwargs
   (`reference.py::video_processing_kwargs`, used as `processing_kwargs` in `mode_embed`), and
   transformers' `BaseVideoProcessor.preprocess` sets each valid kwarg from the call
   (`sample_indices_fn = partial(self.sample_frames, **kwargs)`), so `fps=60` reaches
   `sample_frames` and yields 32 frames.
5. The **client** counts 32 frames (`client.video_policy.num_frames`), so the media stage's engine
   check compares 3936 against the engine's 492 and fails.

The vector gap follows from the input gap: the engine mean-pools over 4 frames, the reference over
32, and the cosine drops to 0.9343 (row 36) and 0.9656 (row 37). The pooled-vector comparison on the
one row says the same thing from the other side: row 36 document 0 is 0.9343 while the *same query
text* on the same row is 0.99997 and every media row's query is ≥ 0.9997 — the same text, frame and
head path agrees; only the video document's frames differ. The brief's fallback (a numeric gap, then
the SDPA backend and the sliding window) therefore does not apply to this row: the input difference
is measured, and no engine-vs-reference numeric comparison on row 36 can be read until the frame
counts match. (The fix11ref `text_vectors` path applies to text-only sides; a media side keeps the
card's `encode` and its processor, which is the render compared above.)

### 2.5 The media stage's +1-token artefact (rows 33 and 35)

Rows 33 and 35 fail the media engine check by exactly one token, and that one is the check's own
arithmetic, not the engine:

```
pairs:35 query:    recorded usage.prompt_tokens 274
  image expansion 258 + 1 merge -> delta 259; 274 - 259 = 15 (the media-stripped text render)
pairs:33 document: recorded usage.prompt_tokens 290
  image expansion 268 + 1 merge -> delta 269; 290 - 269 = 21 (the media-stripped text render)
```

The engine's real prompt replaces the placeholder token with the image expansion
(`boi + n + eoi`, `n = 256` for row 35 and 266 for row 33) and the engine's media-stripped baseline
joins the text parts with `""` (`media.py::_without_media` strips media but keeps every text part),
so the measured delta carries the 258/268 image tokens **plus one** for the tokenization merge the
placeholder's presence breaks (`"query: which"` tokenizes differently when the placeholder splits
it). The client's 258/268 is the image expansion alone, and the checkpoint's geometry gives exactly
`n + 2`. Rows 25–32 (a single text part, no body text after the media) pass because there is no merge
to lose. This is a check finding, not a product divergence: the check's referent ("the engine's
`usage.prompt_tokens` minus the same request without its media parts") is not tokenization-neutral
when a text part follows the media. The video rows' 492/3936 is the real divergence of §2.4.

### 2.6 Rows 31/32: the fixed-point geometry, proven

The first report called rows 31/32 an unproven pixel-level residual. The product's own code and the
reference's own code prove a concrete **input** difference instead:

```
row 31 tall_receipt: raw 576x4096
  client (fixed point) 288x2160 -> 272 tokens
  reference (single pass over the raw bytes) 288x2112 -> 266 tokens
  engine's recorded media delta 272 tokens (the fixed point's count)
row 32 extreme_aspect: raw 3000x20
  client (fixed point) 13344x48 -> 280 tokens
  reference (single pass over the raw bytes) 9792x48 -> 206 tokens
  engine's recorded media delta 280 tokens (the fixed point's count)
```

- The client prepares the **fixed point** of the Gemma 4 resize (`gemma4_fixed_point`,
  `rcp-ndcg/src/rcp_ndcg/data/resolution.py:568-613`, iterating `gemma4_resize` at `:513-565`) and
  sends those bytes; the engine's single-pass processor is the identity on them. The wave's own
  usages confirm it: `pairs:31` = 281 = 9 + **272**, `pairs:32` = 289 = 9 + **280** — the fixed
  point's token counts.
- The **reference** is handed the pairs file's raw bytes (`side_input`, `reference.py:336-366`) and
  transformers' Gemma 4 processor is a **single pass** (`get_aspect_ratio_preserving_size`), landing
  on 288×2112 (266 tokens) and 9792×48 (206 tokens) — different grids from the engine's.
- The reference's own two modes disagree: `mode_media` reports the fixed point
  (`reference.py::media_side` calls `gemma4_fixed_point`), while `mode_embed` feeds the raw bytes
  through the single-pass processor. The media stage's client-vs-reference check therefore passes
  (both sides' *declared* counts are the fixed point) while the reference's actual `encode` sees the
  single-pass grid.

This explains the 0.9956/0.9941 cosines as an input difference of exactly the class §2.4 names, and
it is **code-provable on CPU**. All rows whose first pass is already a fixed point (25–30, 33, 34)
agree in geometry and sit at ≥ 0.999, so all four bad rows are now input differences (36/37 frames;
31/32 grid) and no engine-vs-reference numeric comparison is needed for this wave. The named fix
candidate: the reference's media path should prepare the image to the fixed point (as the client
does) before handing it to the processor — a reference-side alignment the fix lane can test on the
GPU wave.

### 2.7 The commands behind every claim

All of these ran on this CPU node, offline, with `uv run --no-sync`; the probe files live in a scratch
directory outside the repository and are not committed.

| claim | command (shape) | output (shape) |
|---|---|---|
| the RC's failures | read the wave's `equivalence.json`/`status.json`; the committed excerpt | §2.1's blocks (`evidence-embeddinggemma-2.json`) |
| the offline reproduction | `stage1_prompts(recipe, pairs, None, over_length_per_shape=20)` | `checked 112`, the same single failure |
| row 65's renders and ids | the checkpoint template + the reference's conversation shape + the recipe tokenizer | §2.3's block (the reference render printed) |
| the 274 discriminator | the recorded `pairs:35` reply's `usage.prompt_tokens` | 274 = 17 − 1 + 258 |
| row 36's renders | the recorded `pairs:36` body + the checkpoint template | §2.4's block |
| the frame counts | `reference._sampled_frames(entry, 60, 32)` / `(entry, 1, 32)` + `gemma4_fixed_point` | 32 / 4; 123 tokens/frame |
| the cosine | the recorded reply's embedding vs the reference store's vector | 0.934319 |
| rows 33/35 | the product tokenizer + the recorded usages | 274/290; deltas 259/269 |
| rows 31/32 | `reference.gemma4_fixed_point` / `gemma4_resize` + the recorded usages | 272/280 vs 266/206 |

### 2.8 Verdict and the named next measurements

**No fix landed; every cause has a named fix candidate.** Two need the owner or a GPU wave, one is a
small harness change:

1. **The video frame count (the vector gap).** The engine's declared pin does not reach the model's
   frame sampling; the reference and the client use the pin. Two candidates: (a) move the pin into
   the processor's own kwargs (`mm_processor_kwargs: {"fps": 60, "max_frames": 32}` beside the image
   keys) and re-run one video row — the engine's `usage.prompt_tokens` must become ≈ 3936 + text,
   not 501; (b) align the reference/client to the engine's actual sampling (the card's 1 fps → 4
   frames for this clip). The frame-count rule itself is CPU-testable and a sibling lane is writing
   that test; the end-to-end confirmation is the wave. The per-row pass criterion: the media stage's
   engine count must equal the client's, and stage 2's `min_cosine` on rows 36/37 must clear 0.999.
2. **The reference's fixed-point geometry (rows 31/32).** The reference must process the
   client-prepared fixed point (or `mode_media` must stop reporting a grid `mode_embed` does not
   consume); then rows 31/32 can be expected to clear the same bound. Until then, "the wave passes
   stage 2" does not follow from the video pin alone.
3. **The render join (the template check).** The client's budget render joins interleaved text parts
   with `TEXT_JOIN`; this checkpoint's template joins them with `""`. Candidates: (a) a declared
   per-recipe join (a product/recipe design change), (b) rendering the media-template probe's
   declared side from the conversation's parts (a harness-check change), or (c) accepting the
   ≤2-token over-reserve and relaxing the check for multi-text-part media rows. All three are
   decisions, not repairs.
4. **The media check's +1 artefact (rows 33, 35).** A small harness fix *is* available (compare the
   engine's own tokenization of the with-media render, or make the baseline keep the placeholder as a
   token); it was left for the owner because it changes a check's referent, not a product number.

## 3. Part B — topk-embed-v1: the controlled GPU experiments

### 3.1 The starting facts (the final RC's own artefacts)

`topk-embed-v1-small` on the final RC: stage 1 passes, the media stage passes, stage 2 fails at every
declared width.

```
min_cosine (full 2048) -0.11344692890060686   min_cosine (k=64)  -0.35537964710105385
min_cosine (k=128)     -0.2848102523636812    min_cosine (k=256) -0.1478045449199848
min_cosine (k=512)     -0.14398048332058655   min_cosine (k=1024) -0.12270265446270309
```

Per-referent sub-0.9 token vectors (209,188 comparisons): **11,296**, of which the nine image
documents hold **10,880** (96%). The per-row **minimum per-token cosine** (the calibration numbers,
committed in `wave-cosines-topk-small.json`): row 20 (the 16×16 tiny, 8×8 merged) **0.9094** (0 of
448 sub-0.9 — the negative control); row 21 (64×64 icon) 0.5473 (58/448); row 22 (a4 72 dpi) 0.0052
(631/3,458); row 23 (a4 150 dpi) −0.3554 (2,232/8,820); row 24 (a4 300 dpi) −0.1512 (2,304/8,820);
row 25 (slide) −0.0939 (2,099/8,554); row 26 (576×4096 tall receipt) −0.3270 (3,309/8,645); row 27
(3000×20) 0.0337 (189/658); row 28's image document (document 1, the 64×64 icon in the mixed batch)
0.5473 (58/448; the row's text document 0 has 0 of 84). The long text documents (rows 17/18) hold
416 sub-0.9 of 51,562/57,302 (0.45%/0.32%, minima 0.677/0.736) and every query is clean. (The brief
quotes the earlier round's 1,583-of-1,637 shape; the final RC's artefact is the table above — the
same conclusion: image documents dominate, degrading with the grid.) Rows 17/18 are *text*
documents and are not part of the image sweep (the probe selects image documents; their 0.45%/0.32%
sub-0.9 rate is a separate, smaller text-path signal the plan does not chase). This family declares no
video and no query-side media: every selected row is an image document, and the probe selects exactly
those (`_pairs_rows`). Excluded already, per the fix11topk lane: permutation/shift/flip, the head
mapping, the keep-rule, the geometry, the M-RoPE positions, the text path. One real recorded
difference remains unproven: the checkpoint's `_image_row` resizes with torchvision antialias BICUBIC
and normalises in bfloat16 (`topk_embed_st.py:110-134`, `_patchify`), while the engine's preprocessing
goes through the HF processor (PIL BICUBIC, float32 affine). For this family the client itself
prepares with PIL BICUBIC (`prepare.py:377-383`), so E1's raw-bytes comparison is the right isolation
for the model-side stacks.

### 3.2 The experiment table

Each experiment isolates one variable; the harness runs all of them for every selected image document
row and writes the numbers. "Signal" is what a pass/fail looks like; every artefact lands in
`probe-report.json` and in the Markdown table. The wave's magnitude is the numeric per-row minimum
per-token cosine of §3.1 (the fixture), never a phrase.

| # | experiment | hypothesis | exact inputs | expected signal | artefact |
|---|---|---|---|---|---|
| **E0** | the control (the reviews' most dangerous assumption) | the probe's in-process engine is the wave's engine | the probe's E6 per-token vectors for one row vs the wave's recorded vectors for the same row (`--control-vectors`) | per-token cosine ≈ 1 on every token; a failure means the probe measures a different machine and every E-number is void. `not_run` (loud) when no control file is given — never a silent pass | `control`, `E0_control` |
| **E1** | preprocessing isolation | the engine's pixels differ from the checkpoint's (filter, affine dtype, or both) | one image; the engine's HF image processor at the pinned budget (the same call vLLM's overlay makes from the recipe's nested `images_kwargs`) vs the checkpoint's `_image_row`; three variants: torchvision antialias BICUBIC + fp32 affine, PIL BICUBIC + fp32 affine, PIL BICUBIC + bf16 affine | each variant's `vs_engine` (the decisive comparison: the variant closest to the engine's pixels names the engine's preprocessing) and `vs_checkpoint`; the calibration is the row's wave minimum (§3.1) and the probe's own E6 cosine | `E1_pixel_values`, `E1_variants[*].vs_engine`, `E1_variants[*].vs_checkpoint` |
| **E2** | the same tensors into the engine tower | with preprocessing removed as a variable, the engine tower either agrees with the reference tower (preprocessing was the cause) or diverges (model math) | the engine's `pixel_values` and the reference's, each through the engine's tower | per-tensor diffs for both feeds; a shared-tensor diff ≫ the in-run bf16 floor implicates the tower | `E2_tower_on_engine_pixels`, `E2_tower_on_reference_pixels` |
| **E3** | layer-by-layer inside the vision tower | the first tensor whose max-abs-diff jumps names the layer | forward hooks on `patch_embed`, every `blocks[i]`, `merger` on the engine tower, both feeds | `first_divergent_layer` reads the **shared-tensor feed** (`E2_tower_on_reference_pixels`); the conflated feed is reported beside it; the threshold is `max(1e-2, 10 × bf16_floor)` with the floor measured in-run | `first_divergent_layer`, `first_divergent_layer_engine_pixels`, `bf16_floor` |
| **E4** | the image-token embeddings | the merged output is what the text model receives | the **engine's** merger output on the engine's own feed vs the reference's | max-abs-diff/cosine; a jump at the merger but not at the blocks separates merger math from per-block math | `E4_image_token_embeddings` |
| **E5** | cross-feed | separates the vision path from the text path | the reference tower on the **engine's** pixels (dumped to `engine-input.pt`, re-run under the reference venv) and the engine tower on the reference's pixels | the 2×2 matrix: same-side feeds vs cross feeds; if the cross-feed matches the same-side feed, the tower is input-independent | `E5_cross_feed` (three comparisons + the engine pixel shape) |
| **E6** | the grid sweep | the loss grows with the grid (a position/geometry effect) | the nine image document rows, one run each, on the checkpoint's own image-only chat render (the recipe's declared image-document render; a row whose image is not the document's only part is skipped loudly) | per-row grid, token count, min per-token cosine next to the wave minimum; a monotone relation with the grid names the geometry; row 20 (min 0.9094, 0 sub-0.9) is the negative control | rows table, `E6_end_to_end` |
| **E7** | the text path, layer by layer | the divergence may be in the 24 text layers (18 linear-attention + 6 full-attention, non-causal), not the vision tower | the engine's text-layer hooks during `llm.encode` vs the reference's `PackedQwen3_5TextModel` layer dump, same row | the first text layer whose diff crosses the floor; a jump at a full-attention layer implicates the non-causal/mask handling. **This checkpoint has no sliding window** (`layer_types` = 18 `linear_attention` + 6 `full_attention`, `is_causal: false`) and vLLM builds the decoder layers from `layer_types` at init, so the brief's window experiment is invalid here (a post-init rewrite is a no-op and would only produce a false exclusion); no config is mutated | `E7_text_layers` |

**Decision rule.** E0 must pass before any other number is trusted. E1 says whether the input pixels
are the whole story (the variant whose `vs_engine` is ~0 identifies the engine's preprocessing; the
row's wave minimum is the magnitude to beat). E2/E3/E4 name the first divergent tensor if the shared
feed is not clean; E5 says which side of the splice owns it; E6 says whether the effect is positional
(row 20 is the control, rows 22–27 the gradient); E7 covers the text path. The first divergent layer
with a shared-tensor feed is the layer to fix; the checkpoint's own code is the definition.

### 3.3 The harness (runnable, in this repository)

| file | what it is |
|---|---|
| `handover/reports/investigate/topk_probe.py` | the probe. Engine side: the engine's HF processor, vLLM's own model loaded in process (`gpu_memory_utilization` pinned), forward hooks on the vision and text towers, `llm.encode`; reference side: the same file re-executed under the family's reference venv on its own GPU, loading the card's `MultiVectorEncoder`, moving every tensor to the model's device, and dumping its tensors (own pixels, variants, vision layers, the cross-feed on the engine's pixels, text layers, document vectors) to a `.pt`. Writes `probe-report.json` + `probe-report.md`. A failed row records its traceback and the sweep continues |
| `handover/reports/investigate/topk_probe.sh` | the node wrapper (the `judgecheck.sh` pattern): `bootstrap.sh envs` builds the engine's plugin wheel + the family's reference venv from the staged RC, reads the checkpoint pin from the staged `family.yaml` **with the variant named** (`recipe.model` / `recipe.revision`, the HF repo id), runs the probe, and uploads the report — a failed upload fails the job |
| `handover/reports/investigate/wave-cosines-topk-small.json` | the wave's per-row minimum per-token cosines (the calibration numbers of §3.1) |
| `handover/reports/investigate/evidence-embeddinggemma-2.json` | Part A's cited report blocks, verbatim and host-free |

Both scripts are inert for the product (no import, no test reads them); `ruff check`,
`ruff format --check`, `shellcheck`, `bash -n` and the dry run are clean.

The **control fixture** (E0) is extracted from the wave's observation corpus (the recording pass must
cover the family; the final RC's recording pass did not, so the control is `not_run` until a wave
records it):

```python
# on a machine that can read the wave's observation corpus (records.jsonl):
import base64, json
import numpy as np
records = [json.loads(line) for line in open("records.jsonl")]
record = next(r for r in records
              if r["inputs"].get("request_id") == "pairs:26" and r["inputs"].get("side") == "document")
frame = base64.b64decode(record["response"]["body_parsed"]["data"][0]["data"])
vectors = np.frombuffer(frame, dtype="<f2").reshape(-1, 2048).astype(float).tolist()
json.dump({"row": 26, "vectors": vectors}, open("control-row26.json", "w"))
# then pass --control-vectors control-row26.json (or RCP_TOPK_CONTROL_VECTORS)
```

The direct in-pod command (an operator with a shell in the pod):

```
python3 /etc/rcp/files/topkprobe/topk_probe.py --out /tmp/topk-probe \
  --model topk-io/topk-embed-v1-small --revision e54485ebab921f2c18c4d092b3f4c40dcca26781 \
  --pairs <stage>/pairs/topk-embed-v1-small.jsonl --recipe-dir <stage>/recipes/topk-embed-v1 \
  --reference-python <state>/reference/topk-embed-v1/bin/python \
  --device cuda:0 --reference-device cuda:1 --gpu-memory-fraction 0.45 \
  --wave-cosines <mounted>/wave-cosines-topk-small.json --control-vectors <mounted>/control-row26.json \
  [--rows 26]
```

Smoke-test the first row with `--rows 26` (the worst row, 3,309 sub-0.9 of 8,645 — the fastest strong
signal; row 21 would be the cheap weak-signal row and is not the first choice).

### 3.4 The exact node job command (one command, once the wiring lands)

`submit.sh`'s `--script` set is closed and asserted by tests, so the plan includes the one-time
wiring, modelled exactly on the `judgecheck` branch:

1. Move `topk_probe.py`, `topk_probe.sh`, `wave-cosines-topk-small.json` and
   `evidence-embeddinggemma-2.json` into `rcp-ndcg-test/src/rcp_ndcg_test/jobs/`.
2. In `submit.sh`: add `topk-probe` to the `--script` validation and usage; resolve
   `TOPK_PROBE_SH="$HERE/topk_probe.sh"` and `TOPK_PROBE_PY="$HERE/topk_probe.py"` beside the
   `JUDGECHECK_*` lines; add the existence check; add the branch (which also forwards the operator's
   knobs, so `RCP_TOPK_*` reach the pod instead of being reachable in-pod only):

   ```bash
   elif [[ "$SCRIPT_NAME" == "topk-probe" ]]; then
     args+=(
       "worker.command=${HF_TOKEN_PREFIX}/bin/bash /etc/rcp/files/topkprobe/topk_probe.sh $RC_STAGE_URI $OUT_URI"
       "files.topkprobe.from_file=$TOPK_PROBE_SH" "files.topkprobe.mount_path=/etc/rcp/files/topkprobe/topk_probe.sh"
       "files.topkprobepy.from_file=$TOPK_PROBE_PY" "files.topkprobepy.mount_path=/etc/rcp/files/topkprobe/topk_probe.py"
       "files.topkprobecosines.from_file=$HERE/wave-cosines-topk-small.json" \
         "files.topkprobecosines.mount_path=/etc/rcp/files/topkprobe/wave-cosines-topk-small.json"
       "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
     )
     for knob in RCP_TOPK_VARIANT RCP_TOPK_ROWS RCP_TOPK_DEVICE RCP_TOPK_REFERENCE_DEVICE \
                 RCP_TOPK_GPU_FRACTION RCP_TOPK_CONTROL_VECTORS; do
       [[ -n "${!knob:-}" ]] && args+=("env.$knob=${!knob}")
     done
   ```
3. In `rcp-ndcg-test/tests/test_jobs_scripts.py`: update the usage assertion in
   `test_submit_refuses_an_unknown_script` (`:1172-1175`) to the new usage string
   (`bootstrap, wave0, e2e, casefill, judgecheck or topk-probe`), and add a mount test mirroring
   `test_submit_judgecheck_mounts_the_judgecheck_script` (`:2094-2120`): the `worker.command=` line
   names `/etc/rcp/files/topkprobe/topk_probe.sh` with the stage and `OUT_URI`, the
   `files.topkprobe*` mounts are present, and no token value appears in the output.

Then the one command (operator-side, from the checkout, exactly as the other scripts are submitted):

```
RCP_SUBMIT_DIR=<a directory for the job CLI's output> \
RCP_KJOBS_CONFIG=<the job CLI config> RCP_GCS_AUTH_FILE=<the mounted auth script> \
RCP_HF_TOKEN_FILE=<the HF token file> \
RCP_TOPK_CONTROL_VECTORS=<the control-row26.json path> \
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script topk-probe --priority dev-high \
  <RC_STAGE_URI> <OUT_PREFIX> topk-probe
```

The wrapper defaults to `topk-embed-v1-small` (override with `RCP_TOPK_VARIANT`), every image
document row (override with `RCP_TOPK_ROWS=20,26,27`), the engine on `cuda:0` and the reference on
`cuda:1` (`RCP_TOPK_DEVICE` / `RCP_TOPK_REFERENCE_DEVICE`), `gpu_memory_utilization=0.45`
(`RCP_TOPK_GPU_FRACTION`), and `TOPK_DRY=1` prints the whole plan without running anything. The
report lands at `<OUT_PREFIX>/topk-probe/topk-probe/`.

## 4. Process and verification

- The lane started on `rfc-0001`'s tip `c668c439` (the branch already contained it; nothing to merge,
  no merge commit).
- **No product code changed**: the tree's only committed delta from `rfc-0001` is this report, the
  two harness files and the two fixtures (the gate's ruff/format checks cover the Python; the shell
  wrapper is shellcheck-clean). No product test was added because no fix landed; the CPU
  reproductions above are the evidence, and the Part B harness is a plan artefact.
- Checks run on the tree: `uv run --no-sync ruff check .` and `ruff format --check .` after the
  harness rewrite; `shellcheck` and `bash -n` on the wrapper; `TOPK_DRY=1` plan; the probe's
  `--help` plus offline exercises of its row selection, report writer, `_first_divergent` (both
  feeds), control summary, bf16-safe `sha256`/`stats`/`bf16_floor` and wave-cosine lookup.
- `bin/gate lane/investigate` on `37dcf799` (the merged tip; `rfc-0001`'s new tip `973a5f36` merged
  first): **GATE: PASS**, every step exit 0 — ruff-check/format, basedpyright 0 errors, root pytest
  `4145 passed, 155 skipped`, contract-docs `413 passed, 107 skipped`, mkdocs built, test-pkg
  `1193 passed, 229 skipped`, recipes clean, vllm-pkg `50 passed`, vllm-models `110 passed, 8 skipped`,
  run_all (1022/987/35/0, 67/67, 82/82), public-names clean, clean. The first attempt on the same
  revision died at `test-pkg` with `exit=139` (SIGSEGV) at
  `test_wavegroups.py::test_the_real_catalog_has_exactly_two_engine_images` under load average 84;
  that test passes in isolation (`1 passed in 4.59s`) and the retry passed, so the crash is
  environmental (the known load flake), not a tree failure. The first report revision's gate
  (`54d75bc2`) had the same flake at a different test.

## 5. Open questions / what could not be proven

1. **The pinned engine image itself** was not read directly (not present locally): the claim that its
   video loader is v0.31.0's generic `VideoBackend` rests on the recipe's own E2-round-4 note, the
   wave's freeze (`vllm-0.31.0-…whl`), the serve log version and the measured 492 — strong but
   indirect. A direct read of the pinned image would settle it; the fix lane must not follow the
   recipe's "Media" note's nightly-class attribution without that read.
2. **The reference's 32 frames and the reference-side fixed-point difference** are proven from the
   transformers 5.19.0 source behaviour and the reference's own ports; the reference venv could not
   be built on this CPU node (the checkpoint's `fla` path needs a GPU).
3. **The Part B harness has not been executed** (no GPU, and the engine image is not available here):
   it parses, lints, dry-runs and its offline paths are exercised, but its vLLM API calls are the
   first thing the executing lane should smoke-test with `--rows 26`.
4. **The E0 control is `not_run` for the final RC** (its recording pass did not cover the family):
   the fixture extraction command is in §3.3, and the probe records `not_run` loudly until a wave
   records the row.
5. **The brief's counts** differ from the artefacts in two places (the 10-vs-20 stage-2 entries; the
   1,583-of-1,637 topk split): the artefacts' own numbers are in §2.1 and §3.1, committed as
   fixtures.

## 6. For the next lanes

- **embeddinggemma-2**: three fixes, in order — the video pin (one GPU wave, one row; criterion: the
  media stage's engine count = the client's, then stage-2 `min_cosine` ≥ 0.999 on rows 36/37); the
  reference's fixed-point geometry (rows 31/32 cannot clear the bound until the reference consumes
  the client-prepared grid); the render join and the media check's +1 artefact (owner decisions,
  §2.8). Do not touch the SDPA backend: all four bad rows are input differences.
- **topk-embed-v1**: run Part B's harness on `--rows 26` first; E0 must pass before any other number
  is trusted (extract the control from a recorded wave; if none, the report says `not_run`), E1's
  `vs_engine` decides whether the recorded resize difference is the cause, and E3's shared-feed
  first-divergent layer (or E7's text layers) decides the rest. The recipe must not gain a
  `known_deviations` entry for anti-correlated vectors. The fix's confirmation is the wave's media
  stage + stage 2; E6's in-process min cosine is the cheap pre-wave proxy.
- **The harness's home**: if a later lane wants the probe as a first-class job, move the four files
  into `jobs/` and apply the `submit.sh` + test wiring of §3.4; until then they are the plan's
  runnable copy.
