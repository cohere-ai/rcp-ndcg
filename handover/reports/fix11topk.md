# Lane fix11topk — topk-embed-v1's anti-correlated vectors: localized to the engine's image path, not fixed

**Lane:** `lane/fix11topk` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `74aa544e` (the
fix10 merges). **Merged:** `rfc-0001` at `9224b700` (the cuda reference-case fills). **Gate:** `timeout 7200
bin/gate lane/fix11topk` — see §6.

**Outcome: NOT FIXED, and the recipe stays `failed`.** No code change landed and **no deviation is declared**
(the brief's rule: anti-correlated vectors are a real divergence, never a declared one). The lane spent its
timebox localizing the divergence with the wave's own evidence; this report records the exact layer, what was
excluded, and the one measurement that localizes it further.

## 1. What the wave says (the evidence this lane worked from)

The round-10 wave `rc0-202610130245` (r4, image `vllm/vllm-openai:v0.31.0`, recipe
`topk-embed-v1-small` at `e54485eb...`) fills the reference and runs both stages. Stage 1 passes; stage 2 fails
with a negative cosine at every declared `k`:

| gate | value | bound |
|---|---|---|
| `min_cosine` (full width) | -0.1087 | 0.999 |
| `min_cosine` (k=64) | -0.3588 | 0.999 |
| `min_cosine` (k=128) | -0.2845 | 0.999 |
| `min_cosine` (k=256) | -0.1480 | 0.999 |

`topk-embed-v1-xsmall` is the same shape (full width -0.2406, k=64 -0.3401).

**The divergence is image-only.** Parsing the run's `equivalence.json` per row and side:

| rows | kind | full-width min cosine | mean | vectors < 0.9 |
|---|---|---|---|---|
| 0-16, 19 (text pairs) | text query + text documents | 0.8317 (one query token), 0.9056 (one long document; all other rows >= 0.95) | 0.998-0.9999 | 1 of 21,600 |
| 17, 18 | 11k/12k-character text documents | 0.7534, 0.7830 | 0.9965 | 28, 25 of 7,366/8,186 |
| 20-28 | the nine **media (image) rows** | **-0.1087** (row 26) | 0.815 (row 26) | **1,583 of 1,637 bad are here** |

Every media row's *query* vectors are 0.9998. The text path therefore agrees at the fp32-head-vs-bf16-head
floor the recipe already declares, and the image path is the only one diverging. Inside the image rows the bad
vectors are scattered through the kept image-patch run (row 26: 964 of 1,235 vectors below 0.99, 481 below 0.9,
indices 8, 15, 24, 34, 38, 41, ...), and the degradation grows with the grid: the 8x8-merged images (rows 20,
21, 28) sit at 0.91-0.99 with 0-8 bad vectors, while the 95x13 and 42x30 grids lose a quarter to a third.

## 2. The mechanism class (measured, not guessed)

The engine's per-token vectors are **not** a permutation of the reference's: the reference's own image-patch
vectors for row 26 are mutually dissimilar (random pairs: median 0.197, 8.3% negative), so a misalignment
would drag the mean down to ~0.2-0.4. Measured, the engine-vs-reference mean is **0.815** — i.e. most positions
carry the reference's value and a subset (roughly a third of the row) is perturbed. The candidate permutations
were computed against the reference's own vectors and all are excluded:

| permutation of the reference's row-26 vectors | mean | < 0.9 |
|---|---|---|
| observed engine-vs-reference | 0.815 | 0.389 |
| shift by +-1, +-13 | 0.288 / 0.378 | 0.95-0.99 |
| grid transpose (95x13 -> 13x95) | 0.256 | 0.991 |
| flip rows / flip columns | 0.290 / 0.326 | 0.92-0.98 |

So the engine's hidden states are *value-wrong at some image positions*, position-dependently, with the error
growing with the grid — the signature of the image tokens' positional/rope treatment or of the vision tower's
per-patch features, not of an ordering bug and not of the head.

The geometry is right: decoding the nine pairs images and running the checkpoint's own `smart_resize` (factor
32, min 65,536, max 1,310,720 px) reproduces every row's kept count exactly (64, 64, 494, 1,260, 1,260, 1,222,
1,235, 94, 64), which is also what the wave's media stage and its engine-count check passed.

## 3. What was checked and excluded (CPU, code-level)

Read against the engine's source at the shipped tag (`v0.31.0` = `db9527a468`) and the checkpoint's own code at
the pinned revision (`topk_embed_st.py`, `hf_backbone.py`, `modeling_topk_embed.py`, `config.json`):

- **The head mapping, normalization and dtype.** `head.` -> `custom_text_proj.` is the text path's own head;
  the text rows match at 0.9995-0.9999, so the mapping, the fp32 head arithmetic, the L2 normalisation and the
  fp16 transfer cannot be the image-only divergence.
- **The keep-rule.** `document_keep_token_ids: [248056]` keeps exactly the positions whose id is the
  checkpoint's `image_token_id`; the kept count equals the reference's per row (above), and the engine's
  `usage.prompt_tokens` media count passed. The rule is not dropping or adding positions.
- **The render.** Stage 1's frame check and the media stage passed; the reference's image prompt is the
  template's `[im_start, "user", newline, vision_start] + N image tokens + [vision_end, im_end, newline]`
  (`_image_prefix`/`_image_suffix`), which is the same frame vLLM renders through the checkpoint's
  `chat_template.jinja`.
- **The 3-axis (M-RoPE) positions.** The checkpoint's `TopkEmbedTransformer._position_ids` and vLLM's
  `Qwen3VLForConditionalGeneration._get_mrope_input_positions` were compared term by term for one image
  document: image tokens get `(prefix, prefix + row, prefix + col)` on both sides (`np.indices((1, h, w)) +
  text_len` vs `prefix + j // w`, `prefix + j % w`), and the suffix continues at `max(image positions) + 1` on
  both sides (the checkpoint's `delta = amax - h*w` produces exactly that). The interleaved section layout
  agrees too: transformers' `apply_interleaved_mrope` puts H at `i % 3 == 1, i < 3*section[1]` and W at
  `i % 3 == 2, i < 3*section[2]`, which is vLLM's `apply_interleaved_rope` and the fused kernel's
  `is_h`/`is_w`. (`MRotaryEmbeddingInterleaved`, the greedy-layout class, is only selected for
  `scaling_type == "openpangu"`; the checkpoint's `rope_type: "default"` takes the plain `MRotaryEmbedding`
  with `mrope_interleaved=True`.)
- **The vision tower, on paper.** The processor's patch order equals the checkpoint's `_patchify` order
  (`(gh//merge, gw//merge, merge, merge, C, T, P, P)`); the patch embed (`Conv3d` vs the checkpoint's
  `F.linear(weight.view(out, -1))`), the bilinear position-embedding interpolation (vLLM's triton/native
  kernels and the checkpoint's `_grid_inputs` produce the same corner indices and weights in the same
  merged order, `num_grid_per_side = int(sqrt(2304)) = 48` on both), the vision rotary (`rot_pos_ids` and the
  checkpoint's `row`/`col` are the same merged-order arrays; `Qwen3_5VisionRotaryEmbedding(head_dim // 2)`
  matches `get_rope(head_size=head_dim, partial_rotary_factor=0.5)`), the merger (`LayerNorm(hidden)` then
  `view(-1, 4*hidden)`), the GELU (`gelu_pytorch_tanh` -> GELU(tanh) in both) and the vision attention
  structure all match line for line.

## 4. The divergence that IS visible in the two code paths (and what it is worth)

One real, verified engine/reference difference remains in the image input itself:

- the checkpoint's own `_image_row` resizes with **torchvision** `resize(..., InterpolationMode.BICUBIC,
  antialias=True)` and then normalises in **bfloat16** (`_patchify`); the engine goes through the checkpoint's
  `processor_config.json`, whose `resample: 3` is **PIL BICUBIC**, and normalises in float32 before the bf16
  cast. The two resize filters are not the same kernel, and for the downscaling cases (the a4/slide/receipt
  images, 2-3x reductions) they differ most.

This is a real difference between the reference (the checkpoint's own code, the definition) and the engine's
preprocessing, and it is worth recording. It is **not** proven to be the whole divergence: the 16x16 -> 256x256
upscale cases have near-identical pixels under either filter yet still lose up to 9% of cosine, which a pure
input-filter difference does not obviously explain, and the amplitude from the input to a 24-layer vision tower
plus 24 LLM layers is unmeasured.

## 5. Why no fix landed

The remaining candidates (the vision tower's per-patch features vs the checkpoint's own, the image tokens'
positional treatment, the resize/normalisation path above) can only be separated by comparing the engine's
**own hidden states** against the checkpoint's on the same document — the engine's vision output and its
per-layer text hidden states for one image row. That needs the GPU (the engine's GDN/fla path and vLLM do not
run here) and it was not reachable in this lane's timebox. A speculative plugin-side change (e.g. re-plumbing
the image preprocessing) would have been an unapproved product decision with no red-first test, so none was
made: the recipe stays `failed` and no `known_deviations` entry was added.

**The exact next measurement** (one GPU job, one image row, e.g. pairs row 26):

1. Run the engine with a debug hook that dumps, for one media request: the engine's `pixel_values` hash, the
   engine's `image_grid_thw`, the vision tower's merged output (`self.visual(...)` before the splice), the
   input embeddings' image positions, and the hidden states after each text layer.
2. Run the reference (`--mode embed`) for the same row and dump the same tensors from
   `PackedQwen3_5VisionModel.forward` and `PackedQwen3_5TextModel.forward` (the checkpoint's own code).
3. Compare in order: pixel_values (this settles the resize/normalisation difference of §4), vision merged
   output (the vision tower), the image-token input embeddings, then the per-layer hidden states (the text
   path). The first tensor that disagrees names the layer to fix; the checkpoint's own code is the definition.

## 6. Process and verification

- Merged the current `rfc-0001` tip (`9224b700`, the cuda reference-case fills) into this lane before gating.
- `timeout 7200 bin/gate lane/fix11topk` on the merged revision: see the gate summary for this lane's sha.
- No test was added: every fixable layer this lane could reach on CPU (the head mapping, the keep-rule, the
  pooling math) is already pinned by `rcp-ndcg-vllm/tests/models/topk/`, and the divergence is only
  reproducible with the engine's own forward pass (GPU). Adding a CPU test that re-states the two position
  computations would test a transcription, not the engine.

## CHANGELOG entry

None: no public change landed (no code, schema, CLI or `__all__` change; the recipe stays `failed`).

## Public surface changes

None.

## Files outside scope

None. The lane's only committed artefact is this report.

## For the next lanes

- **The divergence is image-only and position-dependent.** Any topk GPU job should dump the engine's vision
  output and per-layer hidden states (§5) before touching the plugin: the head, the keep-rule, the render and
  the mrope positions are already excluded, and the recipe's `known_deviations` must not gain an entry for
  anti-correlated vectors.
- **The resize/normalisation difference is real and independent of the outcome** (torchvision antialias
  BICUBIC + bf16 affine vs the HF processor's PIL BICUBIC + fp32 affine). If the hidden-state comparison
  localizes the divergence to the vision tower, this difference is the first thing to quantify on the
  pixel tensors themselves.
- The other lanes' results for this RC (`fix11ctx`, `fix11ref`) are independent of this one.
