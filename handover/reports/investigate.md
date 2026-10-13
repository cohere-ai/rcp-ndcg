# Lane investigate — embeddinggemma-2's row-specific gap (CPU diagnosis) and topk-embed-v1's controlled GPU plan

**Lane:** `lane/investigate` (worktree outside the operator's checkout). **Base and merged tip:**
`rfc-0001` at `c668c439` (the branch tip; no merge commit was needed — the lane started on it).
**Tree:** unchanged (no product, harness or recipe code touched; this lane's committed artefacts are
this report and the two harness files under `handover/reports/investigate/`). **No GPU, no node
submission, nothing pushed, `rfc-0001` not merged into.**

## 1. Status

| Part | Item | Outcome |
|---|---|---|
| A | the final RC's `embeddinggemma-2` failures | **diagnosed, proven on CPU** — two independent causes, both named with file/line; **no fix landed** (neither cause is a small CPU-provable bug in our tree; the fix candidates need a GPU wave or an owner decision) |
| A1 | stage 1 `template_render_check` (query, row 65) | the client's declared render joins a message's text parts with `TEXT_JOIN` (`"\n"`); the checkpoint's own `chat_template.jinja` joins them with the empty string. The engine's render (and the reference's) has no newline; the client's budget render does. Reproduced offline, same failure, same heads |
| A2 | stage 2, row 36 document 0 (the video) | the engine shows the model **4 video frames**, the client and reference **32**. The engine's own media count (492 = 4 × 123) proves it; the reference passes the declared pin to the processor's sampling call (32). The document's render strings are identical on both sides |
| A3 | the media stage's other failures (rows 33, 35) | **a check-arithmetic artefact**, not an engine difference: the engine check's subtraction baseline loses a tokenization merge when a text part follows the media placeholder (+1 token). Proven with the product tokenizer |
| B | `topk-embed-v1`'s image divergence | **GPU plan + runnable harness** — `handover/reports/investigate/topk_probe.py` and `topk_probe.sh`, with the exact node job command and the submit wiring diff; seven staged experiments (E1–E7), each isolating one variable |

## 2. Part A — embeddinggemma-2: the proven chain

### 2.1 The evidence read (the final RC's wave)

Read-only from the RC's staged wave artefacts: the recorded observation corpus (the client's exact
request bodies and the engine's replies), the reference store (`output.json`), `equivalence.json` and
`status.json`. No bucket or host is named here; the artefacts are the RC's own wave outputs.

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
Every other comparison is ≥ 0.994; every query vector is ≥ 0.9997.

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
engine render  : 'task: search result | query: which page shows the quarterly chart'
declared render: 'task: search result | query: \nwhich page shows the quarterly chart'
engine media   : 'task: search result | query: <TOKEN:258880>which page shows the quarterly chart'
first char difference at 29: engine ...'rch result | query: which page shows the'... declared ...'rch result | query: \nwhich page shows th'...
token ids: engine 15 [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 837, 3104, 3831, 506, 42688, 9535, 1]
           declared 17 [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 236743, 107, 7650, 3104, 3831, 506, 42688, 9535, 1]
ids equal: False
```

The same offline `stage1_prompts` run reports the RC's exact failure (`checked 112`, one failure, the
same `engine_head`/`declared_head`), so the check is fully reproducible without a GPU.

**The divergence, named.** The client sent one user message with three parts: the task-prompt text,
the image, the query text. The engine's route (`/v1/embeddings` with `messages`, vLLM's
`content_format == "openai"` → `wrap_dicts=True`) hands the template the **parts list**; the
checkpoint's `chat_template.jinja` concatenates a message's text parts with the **empty string**
(its loop renders each text part and each media placeholder back to back; the `join` filter it also
computes is called with no separator). So the engine renders `prompt + image-placeholder + text` —
no newline. The **client's declared render** (the budget render, the product's one home
`rcp_ndcg.data.text_budget.rendered_request` over `Content.text`) joins the text parts with
`TEXT_JOIN` = `"\n"` (`rcp-ndcg-core/src/rcp_ndcg_core/content.py:32`, `:216`), because
`rcp_ndcg_test.equivalence.wire.Capture.texts` flattens a `messages` body's text parts with
`TEXT_JOIN` (`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/wire.py:213-214`, its comment "as the engine
joins them" — true for vLLM's flattened path, not for the `openai` content-format path where the
template does the joining). The declared render therefore carries two extra ids
(`236743, 107`) where the newline sits.

The *reference* agrees with the engine here: for a media side the reference's `side_input` builds a
user turn with the parts in order and the card's prompt as a system message; the same checkpoint
template renders system content then the user parts → `prompt + placeholder + text`, no newline. So
this failure is **not** an engine-vs-reference input difference: it is the client's budget render
inserting a newline the model never reads (≤2 tokens per interleaved media row).

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
   the argv (`media_io_kwargs: {'video': {'fps': 60, 'max_frames': 32}}`).
2. That pin reaches the **decoder** only:
   `EmbeddingGemma2VideoBackend.load_bytes` (vLLM nightly `vllm/multimodal/video.py:528-575`) passes
   `fps` into `compute_frames_index_to_sample` and decodes the 32 pinned indices. The loader's
   metadata then says `do_sample_frames = len(valid_frame_indices) == total_frames_num`
   (`vllm/multimodal/video.py:181-194`) — **True**, because the pin sampled the whole 32-frame clip.
3. The **model's** HF video-processor call gets only the recipe's `mm_processor_kwargs` (`{}`):
   `vllm/model_executor/models/embedding_gemma2.py:707-721` builds `video_mm_kwargs = dict(hf_kwargs)`
   (empty) plus `do_sample_frames=True`, and calls the processor. The processor's own defaults are
   `fps = 1`, `max_frames = 32` (the checkpoint's `processor_config.json`), so
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
row 35 query with media:    17 tokens   (the engine's render, image placeholder as one token)
row 35 query without media: 15 tokens
engine delta (product tokenizer): 2      (the placeholder token + one boundary merge)
wave's engine_check: engine 259 vs client 258
```

The engine's real prompt replaces the placeholder token with the image expansion
`boi + 256 + eoi = 258` ids, so the engine's count is `17 − 1 + 258 = 274` against a baseline of 15:
delta 259 = the 258 image tokens **+ 1** for the tokenization merge the placeholder's presence
changes (`"query: which"` tokenizes differently when the placeholder splits it). The client's 258 is
the image expansion alone, and the checkpoint's geometry gives exactly 256 + 2. Rows 25–32 (a single
text part, no body text after the media) pass because there is no merge to lose. This is a check
finding, not a product divergence: the check's referent ("the engine's `usage.prompt_tokens` minus
the same request without its media parts") is not tokenization-neutral when a text part follows the
media. The video rows' 492/3936 is the real divergence of §2.4.

### 2.6 What is excluded, and what remains

- The **head, the L2 normalisation and the text path**: every text row and every query vector is
  ≥ 0.994 (mostly 0.9998+), so the head mapping, the MRL cuts and the pooling agree; the fp16/bf16
  floor cannot produce 0.93.
- The **document render** for row 36: proven identical (§2.4) — the divergence is the frame count.
- Rows 31 and 32 (0.994–0.996): these are *image* documents whose engine media count **matches** the
  client's, so their token counts agree; the residual is pixel-level. They are the extreme-aspect
  images where the client prepares the image with PIL and the processor's own resize is not
  guaranteed to be the identity on the prepared bytes (the same class of filter/affine difference the
  topk lane recorded). Not proven here; the named measurement is a CPU/GPU pixel comparison
  (client-prepared bytes vs `Gemma4ImageProcessor`'s resize) — the Part B harness's E1 shape applies
  to this family too.

### 2.7 The commands behind every claim

All of these ran on this CPU node, offline, with `uv run --no-sync`; the probe file lives in a scratch
directory outside the repository and is not committed.

| claim | command (shape) | output (shape) |
|---|---|---|
| the RC's failures | read `equivalence.json`/`status.json` from the wave's artefacts | §2.1's blocks |
| the offline reproduction | `stage1_prompts(recipe, pairs, None, over_length_per_shape=20)` | `checked 112`, the same single failure |
| row 65's renders and ids | `render_chat`/`_jinja_environment` + the recipe tokenizer | §2.3's block |
| row 36's renders | the recorded `pairs:36` body + the checkpoint template | §2.4's block |
| the frame counts | `reference._sampled_frames(entry, 60, 32)` / `(entry, 1, 32)` + `gemma4_fixed_point` | 32 / 4; 123 tokens/frame |
| the cosine | the recorded reply's embedding vs the reference store's vector | 0.934319 |
| the media artefact | the product tokenizer on the engine's renders | 17 vs 15; delta 2 |

### 2.8 Verdict and the named next measurements

**No fix landed.** The two causes are not small CPU-provable bugs in this tree:

1. **The video frame count (the vector gap).** The engine's declared pin does not reach the model's
   frame sampling; the reference and the client use the pin. The next measurement is one GPU wave
   with the pin moved into the processor's own kwargs — add
   `mm_processor_kwargs: {"fps": 60, "max_frames": 32}` (merged beside the image keys) to the recipe
   and re-run one video row: the engine's `usage.prompt_tokens` must become ≈ 3936 + text, not 501.
   If the engine's processor path refuses or ignores the keys, the alternative is the owner's call:
   align the reference/client to the engine's actual sampling (the card's 1 fps → 4 frames for this
   clip) instead of pinning 32. Either way the recipe's `engine_video_pinning` claim needs the wave's
   own engine count to hold.
2. **The render join (the template check).** The client's budget render joins interleaved text parts
   with `TEXT_JOIN`; this checkpoint's template joins them with `""`. The candidates are (a) a
   declared per-recipe join (a product/recipe design change), (b) rendering the media-template probe's
   declared side from the conversation's parts (a harness-check change), or (c) accepting the
   ≤2-token over-reserve and relaxing the check for multi-text-part media rows. All three are
   decisions, not repairs; no test-first patch was warranted.
3. **The media check's +1 artefact (rows 33, 35).** A small harness fix *is* available (compare the
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
documents hold **10,880** (96%) — row 26 (the 95×13 receipt) 3,309 of 8,645; rows 23/24 (a4
150/300 dpi) 2,232/2,304 of 8,820; row 25 (slide) 2,099 of 8,554; row 22 (a4 72 dpi) 631 of 3,458;
row 27 (the 42×30 extreme aspect) 189 of 658; rows 21/28 (the 64×64 icon) 58 of 448 each; row 20 (the
8×8-merged tiny) 0. The long text documents (rows 17/18) hold 416 (0.3–0.5% of their tokens) and
every query is clean. (The brief quotes the earlier round's 1,583-of-1,637 shape; the final RC's
artefact is the table above — the same conclusion: image documents dominate, degrading with the
grid.) Excluded already, per the fix11topk lane: permutation/shift/flip, the head mapping, the
keep-rule, the geometry, the M-RoPE positions, the text path. One real recorded difference remains
unproven: the checkpoint's `_image_row` resizes with torchvision antialias BICUBIC and normalises in
bfloat16 (`topk_embed_st.py:117-126`, `_patchify`), while the engine's preprocessing goes through the
HF processor (PIL BICUBIC, float32 affine).

### 3.2 The experiment table

Each experiment isolates one variable; the harness runs all of them for every selected image row and
writes the numbers. "Signal" is what a pass/fail looks like; every artefact lands in
`probe-report.json` and in the Markdown table.

| # | experiment | hypothesis | exact inputs | expected signal | artefact |
|---|---|---|---|---|---|
| **E1** | preprocessing isolation | the engine's pixels differ from the checkpoint's (filter, affine dtype, or both) | one image; the engine's HF image processor at the pinned budget vs the checkpoint's `_image_row`; then three variants: torchvision antialias BICUBIC + fp32 affine, PIL BICUBIC + fp32 affine, PIL BICUBIC + bf16 affine | `pixel_values` shape/hash/max-abs-diff per variant; if the torchvision-vs-PIL variant alone moves the diff to the wave's magnitude, the filter is the cause; if only the fp32-vs-bf16 affine moves it, the dtype is | `E1_pixel_values`, `E1_variants` |
| **E2** | the same tensors into both models | with the preprocessing removed as a variable, the towers either agree (preprocessing was the cause) or diverge (model math) | the engine's `pixel_values` and the reference's into **both** towers | per-tensor diffs for each feed; a shared-tensor diff ≫ the bf16 floor implicates the tower | `E2_tower_on_engine_pixels`, `E2_tower_on_reference_pixels` |
| **E3** | layer-by-layer inside the vision tower | the first tensor whose max-abs-diff jumps names the layer | forward hooks on `patch_embed`, every `blocks[i]`, `merger` on both towers | `first_divergent_layer`: a jump at `patch_embed` (input/patchify), at a block (attention/rope), or at `merger` (pooling/projection) | `first_divergent_layer`, per-layer table |
| **E4** | the image-token embeddings | the merged output is what the text model receives | the `merger` output, engine vs reference | max-abs-diff/cosine; a jump at the merger but not at the blocks separates merger math from per-block math | `E4_image_token_embeddings` |
| **E5** | cross-feed | separates the vision path from the text path | the engine tower on the reference's pixels and the reference tower on the engine's pixels (already computed in E2), plus each side's end-to-end vectors | if the cross-feed matches the same-side feed, the tower is input-independent and the text path is implicated; if not, the vision path is | `E2_tower_on_*`, `E6_end_to_end` |
| **E6** | the grid sweep | the loss grows with the grid (a position/geometry effect) | the nine image rows (16×16 … 2480×3508), one run each | per-row grid, token count, first divergent layer and min per-token cosine; a monotone relation with the grid names the geometry | rows table |
| **E7** | the attention window | the text side's sliding window (the checkpoint's 512 + the every-6th-layer full attention) changes the divergence | one row re-encoded with the window disabled on the engine side (the loaded text config's `sliding_window` cleared and every layer type forced to full attention) | the min cosine before/after; if it moves, the window/mask semantics are implicated; if it does not, they are excluded | `E7_window` |

The end-to-end per-token cosine (E6) is the wave's own signal, computed in process: the engine's
`llm.encode(..., pooling_task="token_embed")` for the checkpoint's image-only chat render against the
reference's `encode_document` vectors, kept positions aligned.

**Decision rule.** E1 says whether the input pixels are the whole story; E2/E3/E4 name the first
divergent tensor if they are not; E5 says which side of the splice owns it; E6 says whether the
effect is positional; E7 rules the sliding window in or out. The first divergent layer with a
shared-tensor feed is the layer to fix; the checkpoint's own code is the definition.

### 3.3 The harness (runnable, in this repository)

| file | what it is |
|---|---|
| `handover/reports/investigate/topk_probe.py` | the probe. Engine side: the engine's HF processor, vLLM's own model loaded in process, forward hooks, `llm.encode`; reference side: the same file re-executed under the family's reference venv, loading the card's `MultiVectorEncoder` and dumping its tensors to a `.pt`. Writes `probe-report.json` + `probe-report.md` (per row: grid, E1 pixel diff, per-layer table, E4, E6 min cosine, E7 before/after). A failed row records its traceback and the sweep continues |
| `handover/reports/investigate/topk_probe.sh` | the node wrapper (the `judgecheck.sh` pattern): `bootstrap.sh envs` builds the engine's plugin wheel + the family's reference venv from the staged RC, then runs the probe with the image's own python and uploads the report |

Both are inert for the product (no import, no test reads them); `ruff check`, `ruff format --check`
and `shellcheck` are clean on them. The direct in-pod command (an operator with a shell in the pod):

```
python3 /etc/rcp/files/topkprobe/topk_probe.py --out /tmp/topk-probe \
  --model topk-io/topk-embed-v1-small --revision e54485ebab921f2c18c4d092b3f4c40dcca26781 \
  --pairs <stage>/pairs/topk-embed-v1-small.jsonl --recipe-dir <stage>/recipes/topk-embed-v1 \
  --reference-python <state>/reference/topk-embed-v1/bin/python --device cuda:0 [--rows 20,26,27]
```

### 3.4 The exact node job command (one command, once the wiring lands)

`submit.sh`'s `--script` set is closed, so the plan includes its one-time wiring, modelled exactly on
the `judgecheck` branch:

1. Move `topk_probe.py` / `topk_probe.sh` into `rcp-ndcg-test/src/rcp_ndcg_test/jobs/`.
2. In `submit.sh`: add `topk-probe` to the `--script` validation and usage; resolve
   `TOPK_PROBE_SH="$HERE/topk_probe.sh"` and `TOPK_PROBE_PY="$HERE/topk_probe.py"` beside the
   `JUDGECHECK_*` lines; add the existence check; add the branch

   ```bash
   elif [[ "$SCRIPT_NAME" == "topk-probe" ]]; then
     args+=(
       "worker.command=${HF_TOKEN_PREFIX}/bin/bash /etc/rcp/files/topkprobe/topk_probe.sh $RC_STAGE_URI $OUT_URI"
       "files.topkprobe.from_file=$TOPK_PROBE_SH" "files.topkprobe.mount_path=/etc/rcp/files/topkprobe/topk_probe.sh"
       "files.topkprobepy.from_file=$TOPK_PROBE_PY" "files.topkprobepy.mount_path=/etc/rcp/files/topkprobe/topk_probe.py"
       "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
     )
   ```

Then the one command (operator-side, from the checkout, exactly as the other scripts are submitted):

```
RCP_SUBMIT_DIR=$PWD/shake/ops/submit-<rc>-topkprobe \
RCP_KJOBS_CONFIG=<the job CLI config> RCP_GCS_AUTH_FILE=<the mounted auth script> \
RCP_HF_TOKEN_FILE=<the HF token file> \
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script topk-probe --priority dev-high \
  <RC_STAGE_URI> <OUT_PREFIX> topk-probe
```

The wrapper defaults to `topk-embed-v1-small` (override with `RCP_TOPK_VARIANT=xsmall`), every media
row (override with `RCP_TOPK_ROWS=20,26,27`), one GPU (`--device cuda:0`), and reads the checkpoint
pin from the staged `family.yaml`; `TOPK_DRY=1` prints the whole plan without running anything. The
report lands at `<OUT_PREFIX>/topk-probe/topk-probe/`.

## 4. Process and verification

- The lane started on `rfc-0001`'s tip `c668c439` (the branch already contained it; nothing to merge,
  no merge commit).
- **No code changed**: `git status` is clean except this report and the two harness files (which the
  gate's ruff/format checks cover). No test was added because no fix landed; the CPU reproductions
  above are the evidence, and the Part B harness is a plan artefact.
- Checks run on the tree: `uv run --no-sync ruff check .` and `ruff format --check .` after the
  harness files were added; `shellcheck` on the wrapper; `bash -n` on the wrapper; `python
  topk_probe.py --help` and an offline exercise of its row selection and report writer.
- `bin/gate lane/investigate` on `54d75bc2` (the report + harness commit; `rfc-0001` was already its
  tip): **GATE: PASS**, every step exit 0 — ruff-check/format, basedpyright 0 errors, root pytest
  `4145 passed, 155 skipped`, contract-docs `413 passed, 107 skipped`, mkdocs built, test-pkg
  `1193 passed, 229 skipped`, recipes clean, vllm-pkg `50 passed`, vllm-models `110 passed, 8 skipped`,
  run_all (1022/987/35/0, 67/67, 82/82), public-names clean, clean. The first attempt on the same
  revision died at `test-pkg` with `exit=139` (SIGSEGV) at
  `test_record_and_wave.py::test_wave_packs_engines_so_reference_gpus_remain` under load average 79;
  that test passes in isolation (`1 passed in 14.78s`) and the retry passed, so the crash is
  environmental (the known load flake), not a tree failure.

## 5. Open questions / what could not be proven

1. **The engine's video frames on the node** are proven from the engine's own `usage.prompt_tokens`
   (492) plus the model's geometry (123/frame); no GPU was available to dump the processed frames
   directly. The `do_sample_frames=True` path is read from the pinned nightly's source.
2. **The reference's 32 frames** are proven from the transformers 5.19.0 `BaseVideoProcessor`
   behaviour (call kwargs reach `sample_frames`) plus the reference's own declared pin; the reference
   venv could not be built on this CPU node.
3. **Rows 31/32's residual (0.994–0.996)** is not proven: the token counts match, so it is
   pixel-level; the named measurement is a pixel comparison (E1's shape) for this family.
4. **The brief's counts** differ from the artefacts in two places (the 10-vs-20 stage-2 entries; the
   1,583-of-1,637 topk split): the artefacts' own numbers are in §2.1 and §3.1.
5. **The Part B harness has not been executed** (no GPU, and the engine image is not available here):
   it parses, lints, dry-runs and its report writer is exercised offline, but its vLLM API calls are
   the first thing the executing lane should smoke-test with `--rows 21` before the full sweep.

## 6. For the next lanes

- **embeddinggemma-2**: the video pin is the first fix to try (one GPU wave, one row); the render
  join and the media check's +1 artefact are the two smaller items, both listed with candidates in
  §2.8. Do not touch the SDPA backend for this family: the render and the head are excluded, and the
  video frame count explains the row-36 gap.
- **topk-embed-v1**: run Part B's harness on one row first (`--rows 26`, the worst); E1 decides
  whether the recorded resize difference is the cause, E3's first divergent layer decides the rest.
  The recipe must not gain a `known_deviations` entry for anti-correlated vectors.
- **The harness's home**: if a later lane wants the probe as a first-class job, move the two files
  into `jobs/` and add the `submit.sh` branch of §3.4; until then they are the plan's runnable copy.
