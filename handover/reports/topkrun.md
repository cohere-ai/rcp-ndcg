# Lane topkrun — the topk probe becomes a job, runs on a node, and names the divergence

**Lane:** `lane/topkrun` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `ac7c4a5d`
(already the lane's tip: nothing to merge, no merge commit). **Gated:** `60cb5615` — `GATE: PASS` (every
step exit 0); the report commit below is docs-only.
**Tree:** the wiring in `submit.sh`, the probe's three files moved into the job package, the harness fixes
the node runs forced, the dtype knobs for the precision experiment, the tests, the docs and the CHANGELOG.
**Eleven node jobs run; nothing pushed; `rfc-0001` merged at `63078410`.**

## 1. What landed

| # | Item | Outcome |
|---|---|---|
| 1 | `submit.sh --script topk-probe` | **wired** — mounts `topk_probe.sh`, `topk_probe.py`, the wave-cosines fixture and `bootstrap.sh`; the probe's own argv (`<RC_STAGE_URI> <OUT_URI>`); the operator's `RCP_TOPK_*` knobs forwarded only when set |
| 2 | the probe's home | `handover/reports/investigate/{topk_probe.py,topk_probe.sh,wave-cosines-topk-small.json}` moved into `rcp-ndcg-test/src/rcp_ndcg_test/jobs/` (the report's §3.4 step 1; the Part A evidence fixture stayed) |
| 3 | tests | the usage string, the mounts, the forwarded knobs, the wrapper's dry plan, the recorder, the squeeze, the engine check — **red first**, then green |
| 4 | the first node run | **failed** on a real harness bug (`engine_model`: `llm.llm_engine.named_modules()` — the model lives in the EngineCore process by default) |
| 5 | the fix + the dry engine check | the probe forces `VLLM_ENABLE_V1_MULTIPROCESSING=0` before vLLM is imported, walks the documented in-process paths with a loud typed error, and `--check-engine` (seconds, no weights) checks the image's vLLM API before the model loads |
| 6 | the second node run | **ran all nine rows and uploaded** — but its E2/E3 tables held only the `merger`: `nn.Module` forward hooks never fire on vLLM's `@support_torch_compile`-decorated modules |
| 7 | the recorder fix | the probe wraps each target module's bound `forward` (works for decorated and plain modules), records the keys a recording missed, and `stats()` squeezes singleton dimensions (vLLM keeps a leading 1 on every block) — the two traps that had made "the merger" a false first divergent layer |
| 8 | the third node run | **the bf16 verdict** (section 4): all 27 tower tensors and 24 text layers recorded; the first divergent tensor is the vision tower's `block_0`, at bf16-rounding scale, amplified by the tower's own blow-up |
| 9 | the dtype knobs | `--dtype` / `--reference-dtype` on the probe (both sides), forwarded as `RCP_TOPK_DTYPE` / `RCP_TOPK_REFERENCE_DTYPE`; the wrapper's argv is parsed by the module's own parser in a test (a node run died on `unrecognized arguments`) and the reference subprocess gets its own dtype (the first fp32 run compared two precisions) |
| 10 | the precision run (both sides fp32) | **the precision verdict** (section 6): every row's tower output clears 0.999 in fp32, and the engine's fp32 *serve* is refused by vLLM's GatedDeltaNet kernels — the dtype route is closed, the mechanism is proven |
| 11 | the fp16 cells (both cells, image rows) | **fp16 fails the gate on every image row** (E6 minima 0.095-0.997), and it does not overflow: its tower is closer to the reference than bf16's, and the "later first-divergent block" is the wrong-dtype floor artefact |
| 12 | the text-only cells (`--text-rows` / `--query-rows`) | **the text verdict** (section 7): the wave's majority failure is text-side, byte-identical inputs, dtype-independent from `text_layer_0`; the 7-8k-token documents fail in both dtypes, and the divergence is unchanged with the vendored FLA (`triton`) or `cutedsl` prefill engaged (lane topk-gdn) |
| 13 | the calibration fixture | `wave-cosines-topk-small.json`'s `_source` was wrong (its values are each row's minimum over all mrl widths, not the full-width minima): relabelled with the wave's own full-width numbers, which the top-level keys now hold |

## 2. The wiring (tests first)

`submit.sh` gains the `topk-probe` value in the usage, the option message, the validation and the
branch (the `judgecheck` pattern): `TOPK_PROBE_SH` / `TOPK_PROBE_PY` / `TOPK_PROBE_COSINES` resolved
beside `JUDGECHECK_*` (both `jobs/` and the package-root invocation), existence checks, and the job spec

```
worker.command=<token wrapper> /bin/bash /etc/rcp/files/topkprobe/topk_probe.sh <RC_STAGE_URI> <OUT_URI>
files.topkprobe{,.mount_path}          topk_probe.sh
files.topkprobepy{,.mount_path}        topk_probe.py
files.topkprobecosines{,.mount_path}   wave-cosines-topk-small.json
files.bootstrap{,.mount_path}          bootstrap.sh
env.RCP_TOPK_{VARIANT,ROWS,DEVICE,REFERENCE_DEVICE,GPU_FRACTION,CONTROL_VECTORS,MODEL,REVISION}
```

`RCP_TOPK_PROBE_PY` / `RCP_TOPK_WAVE_COSINES` are the mounted helpers' own node paths and are never
forwarded from the operator's environment (a host path in the pod would be meaningless); the GCS helper
files are already mounted for every script.

The probe module joins `report.py` and `reference_deps.py` in `basedpyright`'s exclusion list (a node
script, never imported by the suite; the one error it produced in the checked scope was `PIL.Image.BICUBIC`).

Red-first evidence (the tree before the wiring):

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q -k 'unknown_script or topk_probe'
4 failed, 86 deselected
  test_submit_refuses_an_unknown_script: assert 'bootstrap, wave0, e2e, casefill, judgecheck or topk-probe' in
    'submit.sh: --script must be bootstrap, wave0, e2e, casefill or judgecheck, got deploy\n'
  test_submit_topk_probe_mounts_the_probe_script: AssertionError: submit.sh: --script must be ... got topk-probe
  test_the_topk_probe_dry_plan_names_the_probe_and_the_upload_prefix:
    AssertionError: bash: .../jobs/topk_probe.sh: No such file or directory (exit 127)
```

## 3. The node runs

Every job ran against the same RC (`rc0-<stamp>`, the topk recipe, the plugin wheel, the reference
lock and the pairs are staged) on the pinned image `vllm/vllm-openai:v0.31.0`, priority `dev-high`, each
with its own out prefix. `TOPK_DRY=1` printed the plan first, and the `submit.sh` plan was rendered with
`KJOBS=echo` before the first submission.

| run | job | result |
|---|---|---|
| 1 | the wiring's first job | failed: `AttributeError: 'LLMEngine' object has no attribute 'named_modules'` in `engine_model` |
| 2 | after the single-process/check fix | exit 0, all nine rows, report uploaded — but the E2/E3 tables held only `merger` (the hook bug) |
| 3 | after the recorder fix | exit 0, report uploaded — the blocks recorded but not compared (the leading-1 shape) |
| 4 | after the squeeze fix | exit 0, report uploaded — **the bf16 verdict below** |
| 5 | the first fp32 attempt | failed on a node: `unrecognized arguments: --dtype float32` (the wrapper's flags were not in the module's parser) |
| 6 | the fp32 retry | exit 0, uploaded — but the reference subprocess argv never carried `--reference-dtype`: the engine ran fp32 against a bf16 reference (its rows all read `ValueError: Unsupported dtype torch.float32…` at the end-to-end step) |
| 7 | both sides fp32 | exit 0, uploaded — **the precision verdict below** |
| 8 | fp16 / fp16 (image rows) | exit 0, uploaded — E6 minima 0.095-0.997: **fp16 fails the gate on every image row** |
| 9 | fp16 / fp32 (image rows) | exit 0, uploaded — E6 minima -0.054-0.997: the mixed cell fails too |
| 10 | the text-only rows (bf16 / bf16) | exit 0, uploaded — **the text verdict below**: rows 17/18 at 0.8017 / 0.7041, the probe reproducing the wave's own text-row failure |
| 11 | the text-only rows (fp16 / fp16) | exit 0, uploaded — 0.8739 / 0.8771, still under the bound; the 466-token query clears at 0.99927 |

The wrapper runs `topk_probe.py --check-engine` before the probe; in the pod it printed
`{"vllm": "0.31.0", "single_process_switch": "VLLM_ENABLE_V1_MULTIPROCESSING=0", "problems": []}` and the
probe continued (run 2 onward). The same check passes locally against the image whose digest the submitter
resolves (`sha256:c1c9f6fd…`), so the pin, the plan and the check agree.

## 4. The probe's findings (run 4)

**E0, the control: `not_run`** — no `--control-vectors` file, because the family's observation corpus was
never recorded (the last recording pass did not cover it); the report says so in its header and its JSON,
and every number below is therefore *uncalibrated against the wave's own engine*. Honest, not green.
This is still true in run 7 (fp32): `control: not_run`.

**E1, preprocessing: the engine is exonerated.** On every row the engine's `pixel_values` match the
`pixel_values_torchvision_fp32_affine` / `pixel_values_pil_fp32_affine` variants at 1.19e-07, while the
checkpoint's own `_image_row` (bf16 affine) differs by 3.6e-03 max (cos 0.9999979). The grids agree on
every row. The only input difference is the affine's dtype: the client and the engine's HF processor
compute it in fp32, the checkpoint's `_patchify` in bf16.

**E2/E3, the vision tower on the shared feed (bit-identical inputs).** Row 26 (the worst; the same shape
on the other rows):

| tensor | max-abs | mean-abs | cosine |
|---|---|---|---|
| `patch_embed` | **0.0** | 0.0 | 1.0 |
| `block_0` | 0.094 | 6.6e-04 | 0.9999945 |
| `block_9` | 1.016 | 4.7e-03 | 0.99937 |
| `block_15` | 82.9 | 0.0204 | 0.99224 |
| `block_22` | 92.2 | 0.0781 | 0.98726 |
| `block_23` | 5082 | 1.95 | 0.97395 |
| `merger` / `merged` | 1.703 | 0.0166 | 0.95653 |

- `patch_embed` is **bit-identical** — the pixels, the patch order and the patch embedding are the same.
- The first tensor that differs is **`block_0`**, at 0.2% of the tensor's own mean magnitude (0.094 max on
  absmax 5.4) — bf16 kernel rounding, not a semantic difference: no layer shows a step, the difference
  grows *smoothly* with depth (a wrong mask, position or reshape changes a tensor materially at one layer).
- Both implementations' states **blow up** at the end of the tower: the reference's own `block_23` reaches
  absmax 6176 (its own cross-feed run: 6240), the engine's 6240 — so the tower itself is numerically
  unstable there, and a rounding-level difference at `block_0` becomes a material one by `block_23`.
- The merger's LayerNorm collapses the scale, leaving the tower output at cos 0.9565 — which is where the
  first run's hook bug had put the "first divergent layer".

**E5, cross-feed.** The engine's own tower on the two pixel feeds differs by max 2.05 (mean 0.0163) and
the reference's by max 2.12 (mean 0.0165): a 3.6e-03 pixel difference alone produces the same magnitude of
output difference as engine-vs-reference. The tower amplifies input differences ~15x (pixels → output) and
~130x (a block-0 rounding difference → the merger), so the merger-level comparison is noise-limited at
this depth.

**E7, the text path.** `text_layer_0` differs at cos 0.335 (mean 0.072 on the reference's own absmean
0.075 — the two implementations' first-layer states are essentially uncorrelated), and the difference
grows to cos -0.14 by `text_layer_23`. The text model's input is the image-token block (1235 of 1242
positions) whose embeddings differ at cos 0.949; one recurrent (linear-attention) layer over 1235 positions
amplifies that, which is what the numbers show. Two candidate mechanisms were checked and **ruled out** by
reading the code, not guessed: the render (the checkpoint's `_image_row` builds the image document from
exactly the probe's chat-template render — `_image_prefix` / `_image_suffix` from
`apply_chat_template([{"role": "user", "content": [{"type": "image"}]}])`; `document_prompt` is the *text*
document's), and the position scheme (the checkpoint's `_position_ids` gives the image block
`(prefix, prefix + row, prefix + col)` and continues after it at `prefix + max(h, w)`; vLLM's
`Qwen3VLForConditionalGeneration._get_mrope_input_positions` builds `np.indices((1, h, w))` plus the text
length and continues at `max + 1` — the same scheme).

**E6, end to end (run 4).** Against the wave's *full-width* minima, re-derived from the wave's own
`equivalence.json` (`stage2.per_vector`, `mrl_dim null`): row 20 0.7356 vs 0.9111, row 21 0.5491 vs
0.5514, row 22 0.0389 vs 0.0390, row 23 -0.0697 vs -0.0682, row 24 -0.0830 vs -0.0808, row 25 -0.0496
vs -0.0348, row 26 -0.1155 vs -0.1134, row 27 0.0579 vs 0.0577, row 28 0.5491 vs 0.5514 — agreement to
0.002-0.015 on every row **except row 20 (Δ 0.175)**. Two review findings are folded in here, both
re-derived from the wave's artefact and both true: (a) the fixture `wave-cosines-topk-small.json`
carried each row's minimum over *all* mrl widths, not the full-width minimum its `_source` claimed (row
23's -0.35538 is the k=64 minimum; the full-width one is -0.068241) — lane topkrun relabelled the
fixture: the top-level keys now hold the full-width minima the probe's E6 computes and
`_minima_over_all_widths` keeps the old values; (b) the probe's E6 is **not** the gate's statistic: it
lacks the gate's two-sided fp16 cast (client `embed_dtype: float16` and the reference's `_fp16_lists`;
the induced cosine error is ~1e-5, which cannot explain the minima but can move tokens sitting near the
bound), its engine is `enforce_eager` and single-process where the wave's serve is not, and row 20's
Δ 0.175 is **unexplained** — the control that would settle it (E0) is `not_run`, because the family's
corpus was never recorded. Every number in this report is therefore a *proxy* for the gate's, read for
its mechanism and ordering, never as the gate's own value.

## 5. The bf16 verdict (run 4)

**The first divergent tensor is the vision tower's `block_0`, at bf16-rounding scale; the divergence is
numerical amplification, not a mapping, geometry, weight or render bug.** The evidence:

- `patch_embed` is bit-identical, the grids agree, E1 exonerates the preprocessing, the merger's code and
  shapes are the stock vLLM ones the checkpoint's own code mirrors, and the model loads (the engine's
  in-process object is reachable through vLLM's own documented route, which is what the probe now uses).
- No layer steps: the difference grows smoothly from 0.2% at `block_0` to 16% at `block_23`.
- The *reference's own* tower blows up the same way (absmax 6176 vs the engine's 6240), and its own two
  runs differ by as much as engine-vs-reference (E5: 2.12 vs 1.70).
- The seed is bf16: two implementations of the same architecture (vLLM's kernels vs the checkpoint's
  packed HF path), plus the declared fp32-vs-bf16 affine difference in the pixels the client and the
  reference prepare.

**Therefore no bounded, CPU-provable fix exists in this lane**: the product's plugin is faithful (its
mapping, keep-rule and geometry are the ones the reference defines), and "fixing" the engine's bf16 kernels
to match the reference's bit-for-bit is not a fix — it is the family's *comparison* that cannot hold at
this precision. The candidates, in order, are decisions for the owner and a follow-up lane:

1. **Compare this family's image documents at a coarser level** (the pooled document vector, or the
   per-token cosine with a declared tolerance derived from the measured amplification), or declare the
   image-document per-token comparison a `known_deviation` with the numbers above as its evidence — the
   recipe must not claim a bound the family cannot meet. The text rows are unaffected (0.45%/0.32% sub-0.9).
2. **If the per-token comparison must stand**, the cheapest decisive experiment is a *precision* one: run
   the reference's tower in fp32 (the reference venv can; the engine cannot — its serve is bf16) and, in
   the same run, the engine's tower in fp32 through `LLM(dtype="float32")` on one row; if the same-feed
   difference collapses to the fp32 floor, the amplification reading is confirmed and the bound is
   unreachable for a bf16 serve.
3. **The affine dtype is a second, smaller seed** (E1): the client's pixels are fp32-affine, the
   checkpoint's `_image_row` bf16-affine. Aligning them is a *reference-side* question (the reference is
   the definition; bending it to the engine is not allowed) — the same class as the embeddinggemma-2
   A3 finding, and it belongs in the recipe's notes either way.
4. **The text path needs no separate fix** while (1)-(2) stand: its layer-0 difference is the propagated
   tower difference, and the two alternative mechanisms (render, positions) are ruled out above. If a
   later lane wants it separated, the probe should dump the text model's *input* embeddings (the merged
   image features as the engine scatters them) beside the vision output — one more field in E4.

## 6. The precision experiment (run 7, both sides fp32) and the verdict

Run 7 ran with `RCP_TOPK_DTYPE=float32 RCP_TOPK_REFERENCE_DTYPE=float32`, so the engine's `LLM` **and**
the reference subprocess's checkpoint load are fp32. Side by side, the same-feed (bit-identical inputs)
vision-tower comparison, row by row — the merger (the tower's output) and `block_0`:

| pairs row | bf16 merger max / mean / cosine | fp32 merger max / mean / cosine | fp32 `block_0` cosine | E1 (engine vs checkpoint pixels) |
|---|---|---|---|---|
| 20 | 0.742 / 0.00739 / 0.99467 | 0.0366 / 0.000523 / 0.999971 | 0.99999996 | 1.19e-07 |
| 21 | 0.359 / 0.00714 / 0.99540 | 0.0282 / 0.000529 / 0.999977 | 0.99999995 | 1.19e-07 |
| 22 | 1.355 / 0.01160 / 0.98518 | 0.2317 / 0.000955 / 0.999898 | 0.99999994 | 1.19e-07 |
| 23 | 2.859 / 0.01409 / 0.97742 | 0.3750 / 0.001106 / 0.999865 | 0.99999996 | 1.19e-07 |
| 24 | 1.844 / 0.01487 / 0.97777 | 0.6124 / 0.001182 / 0.999807 | 0.99999997 | 1.19e-07 |
| 25 | 2.121 / 0.01499 / 0.97283 | 0.2421 / 0.000851 / 0.999896 | 0.99999990 | 1.19e-07 |
| 26 | 1.703 / 0.01661 / 0.95653 | 0.1782 / 0.000828 / 0.999880 | 0.99999996 | 1.19e-07 |
| 27 | 1.301 / 0.01104 / 0.98108 | 0.0547 / 0.000570 / 0.999950 | 0.99999996 | 1.19e-07 |
| 28 | 0.359 / 0.00714 / 0.99540 | 0.0282 / 0.000529 / 0.999977 | 0.99999995 | 1.19e-07 |

- **Every row's tower output clears 0.999 in fp32** (0.99981-0.99998) where bf16 gives 0.956-0.995, and the
  mean difference drops 15-20x. The pixel seed disappears with it: E1 is 1.19e-07 on every row (the
  checkpoint's fp32 `_patchify` and the engine's fp32 processor agree bit-for-bit up to float rounding),
  where bf16 leaves 3.58e-03.
- **The amplification is still the tower's**, not a bug: even in fp32 `block_23`'s max-abs difference is
  752 (mean 0.085) on identical inputs — both implementations' states blow up to ~6000 there — and the
  merger's LayerNorm collapses it to 0.18 max / 8.3e-04 mean. bf16 rounding is simply ~20x more seed for
  the same amplifier.
- **The fp32 fix is not available to this family**: the engine loads fp32 and its tower runs, but the
  end-to-end encode refuses fp32 on every row with `ValueError: Unsupported dtype torch.float32, expected
  bfloat16 or float16` — vLLM's GatedDeltaNet prefill kernel (flashinfer's `_cutlass_io_dtype`; the
  Triton/FLA fallback refuses fp32 too), the exact blocker the `pplx-embed-v2-context` recipe already
  documents in its notes for this image. So `serve.dtype: float32` cannot be pinned, and the per-token
  E6/E7 comparison cannot be measured in fp32 either.

**The fp16 cells (runs 8 and 9).** fp16 is the other dtype the GDN kernels accept, and it is a serve
dtype the recipe could pin. `engine fp16 / reference fp16` (run 8) and `engine fp16 / reference fp32`
(run 9) both ran end to end; the image rows' E6 minima are 0.095-0.997 (run 8) and -0.054-0.997 (run 9)
— **fp16 fails the gate on every image row**, and its tower is nonetheless *closer* to the reference
than bf16's (row 26's merger cosine 0.99853 in fp16 vs 0.95653 in bf16; `block_0` max-abs 0.0098 vs
0.094; no inf/nan anywhere in the report, so nothing overflows — the tower's states stay well inside
fp16's range). The "first divergent layer: block_10" a reader will see in the fp16 reports is a
**threshold artefact, not a location**: the probe computes `bf16_floor` whatever `--dtype` says, so the
fp16 run's cutoff is 10 x 0.015625 = 0.156 and `block_9`'s 0.152 sits just under it (the same artefact
hides `block_0`/`block_1` in the fp32 runs). The floor must follow the engine's dtype in a follow-up.

**Verdict.** The bf16 gap is precision-driven and its mechanism is named: bf16 rounding (plus the
bf16-affine pixel seed, 3.58e-03) amplified by a vision tower whose own hidden states reach ~6000 at
`block_23` in both implementations. In fp32 the tower output clears 0.999 on every row; the engine cannot
serve fp32; therefore the family's per-token bound on image documents **cannot be met by a bf16 serve on
the stock image**, and no code fix in this lane can change that. The disposition is an owner decision on
the comparison, with the numbers above as its evidence:

1. **A multi-vector score-level gate**: keep the existing cosine-scale bound on the *MaxSim* score
   (the score the family is actually used with) and report the per-token vectors as non-gating
   diagnostics — the `zerank` precedent. The evidence for it: the per-token statistic is dominated by
   the text-side divergence (section 7), which is present with the reference's own chunked-scan
   algorithm engaged in the engine (lane topk-gdn: `--gdn-prefill-backend triton` and `cutedsl` leave
   every minimum unchanged), while the pooled/score-level signal is what the downstream product
   consumes.
2. **A declared deviation** naming the per-token comparison unachievable at the engine's available
   precision (bf16 and fp16 both measured end to end, fp32 unrunnable), with the tables in sections 4-7
   as its evidence.

**Never a lowered `vec_min_cosine`**: the minima are negative (row 26 -0.1134 in the wave, -0.1145 in
the probe) and lowering the bound over anti-correlated vectors would ship a false green. The decision
between the two options is the **owner's**, not this lane's; no recipe or bound was changed here (the
only recipe-adjacent change is the fixture's `_source`, a factual relabel with the wave's own numbers
cited).

One caveat, recorded rather than glossed: run 7 could not measure E6/E7, because the engine's fp32
encode is refused before the text path runs. The per-token fp32 numbers the first branch of this
experiment would need therefore do not exist; the tower-level numbers are what an fp32 run can show, and
they are decisive about the mechanism (and the refusal itself is the second finding).

## 7. The text-only cells (C-T1): the majority of the failing comparisons

The wave's own record (re-derived here from its `equivalence.json`, full width, `mrl_dim null`): the
**text rows 0-19 carry 6,481 of the 11,819 sub-0.999 comparisons (54.8% -- 6,481 text + 5,338 image)** and the image rows 20-28
carry 5,338 (49.4% of their own 5,841 — every image document row fails, while all nine of their queries
pass at min 0.999847). Rows 17 and 18 are text-only documents (7,366 / 8,186 tokens) at min 0.7534 /
0.7830 with 2,667 / 3,158 below the bound; row 12's query is 0.9339 with 208 of 466 below. Stage 1
passes, so those inputs are **byte-identical on both sides**: the text-side divergence has no input
seed, no vision tower and no pixel-affine question, and the image sweep could not see it
(`_pairs_rows` selects image-carrying rows). The probe now measures it: `--text-rows` / `--query-rows`
render the checkpoint's own `format_document` / `format_query`, encode it in process, and compare
per-token against the reference's own `encode_document` / `encode_query` (its keep mask travels in the
dump so both sides' kept positions match), with the same text-layer recording as E7.

| row | cell | tokens | min cosine | mean | below 0.9 | `text_layer_0` cosine | `text_layer_23` cosine |
|---|---|---|---|---|---|---|---|
| 17 document | bf16 / bf16 | 7,366 | 0.8017 | 0.99618 | 37 | 0.1836 | -0.1571 |
| 17 document | fp16 / fp16 | 7,366 | 0.8739 | 0.99936 | 3 | 0.1835 | -0.1567 |
| 18 document | bf16 / bf16 | 8,186 | 0.7041 | 0.99573 | 36 | 0.1847 | -0.1545 |
| 18 document | fp16 / fp16 | 8,186 | 0.8771 | 0.99931 | 1 | 0.1845 | -0.1558 |
| 12 query | bf16 / bf16 | 466 | 0.9472 | 0.99785 | 0 | 0.2547 | 0.0272 |
| 12 query | fp16 / fp16 | 466 | 0.99927 | 0.99996 | 0 | 0.2553 | 0.0279 |
| 26 image document | bf16 / bf16 | 1,235 | -0.1145 | 0.81529 | 481 | 0.3349 | -0.1445 |
| 26 image document | fp16 / fp16 | 1,235 | 0.1528 | 0.97845 | 58 | 0.3530 | -0.1118 |

- **The text rows fail the gate in bf16 and the probe reproduces the wave's own text-row failure**
  (probe 0.8017 / 0.7041 against the wave's full-width 0.7534 / 0.7830 — the closest calibration the
  probe has, and the only one on rows whose inputs are identical).
- **fp16 improves them and still does not clear 0.999** (0.874 / 0.877 on the 7-8k-token documents;
  the 466-token query clears at 0.99927). The sequence-length ordering is the mechanism's signature.
- **The text model's hidden states diverge at its first layer, dtype-independently** (`text_layer_0`
  cosine 0.1836/0.1847 for the two documents in bf16, 0.1835/0.1845 in fp16 — the same number), and
  the layers stay uncorrelated to the last. With byte-identical inputs, no tower and no pixels in the
  loop, this is the text model's own computation — but **not the GDN prefill kernel** (measured by
  lane topk-gdn): the image vendors flash-linear-attention
  (`vllm/third_party/flash_linear_attention/ops/chunk.py`) and the tag's own `--gdn-prefill-backend`
  selects it, so the engine *can* run the reference's algorithm; with `triton` (the vendored FLA
  chunked scan) and `cutedsl` engaged on the same RC and rows, every per-token minimum is unchanged
  within the probe's run-to-run spread (rows 17/18 0.7814/0.7042 triton, 0.7643/0.7154 cutedsl,
  against 0.7863/0.7125 flashinfer; row 26 -0.1140/-0.1134 against -0.1145), and the per-layer
  cosines agree to 4 decimals across the three kernels (`text_layer_0` 0.1835/0.1836/0.1838). The
  matched-kernel experiment is therefore done, not a follow-up — and it does not fix the gate.

**So the dtype ladder is exhausted, measured end to end**: bf16 fails everywhere; fp16 (the only other
dtype the GDN kernels accept) fails the documents in both the text and image paths; fp32 clears the
*tower* (cos 0.99981-0.99998) and cannot be served. Nothing in the engine's plugin can turn the gate
green; the residual is not rounding (the text-side divergence is identical under all three GDN prefill
kernels, so it is not the kernel algorithm either — lane topk-gdn), and its mechanism inside the text
model is not localised.

## 8. Verification

- `uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q` — **100 passed** (15 of them this
  lane's).
- `ruff format --check . && ruff check .` — clean; `basedpyright` — 0 errors.
- `shellcheck` and `bash -n` on `submit.sh` and `topk_probe.sh` — clean.
- The engine check and the hook mechanism were proven **inside the pinned image on CPU** (the digest the
  submitter resolves): `--check-engine` prints `problems: []`; `Qwen3_VisionPatchEmbed.__call__` and
  `Qwen3_VisionBlock.__call__` are *not* `nn.Module.__call__` while the merger's is; `Qwen3_VisionTransformer.forward`
  calls `patch_embed`, `blocks`, `merger`.
- `timeout 7200 bin/gate lane/topkrun`:
  - `50d6c9a4` — **GATE: PASS** (the fp32 knobs and the parser test)
  - `60cb5615` — **GATE: PASS** (ruff-check/format, basedpyright, the root suite 4145 passed/155 skipped,
    contract+docs 413/107, mkdocs, test-pkg 1206/229, recipes, vllm-pkg 50, vllm-models 110/8, run_all
    1022/987/35/0 + 67/67 + 82/82, public-names clean, clean).
  - `0a9648a0` — FAIL at `test-pkg` with `exit=139` (SIGSEGV) at
    `test_wave_plugin_wheel.py::test_a_wave_without_a_plugin_wheel_records_the_fingerprint_hashes_unchang`;
    the third occurrence of the same load flake in the shared slot (see below), and the retry passed.
  - `5035ced5` — FAIL at `test-pkg` with `exit=139` (SIGSEGV) at
    `test_wave_corpus.py::test_changed_since_skips_the_unchanged_and_records_the_changed`; the retry
    (`50d6c9a4`) passed the same step.
  - `e2cef0d7` — FAIL at `test-pkg` with `exit=139` (SIGSEGV) at
    `test_wavegroups.py::test_the_cli_writes_the_lists_and_the_stage_map`; the same class of load flake the
    investigate lane recorded (its first attempt died the same way at a different test in the same file);
    the retry on `a9bbfb11` passed the same step, and the failure is not in this lane's tree.
  - `d542eca1` (the wiring alone) — **GATE: PASS**.
- Red-first for every fix is in sections 2 and 7.

## 9. The harness fixes the node forced (each with its red test)

| fix | red evidence | what the fix does |
|---|---|---|
| the in-process model accessor | run 1's traceback (`'LLMEngine' object has no attribute 'named_modules'`) | `force_single_process_engine()` sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` before vLLM is imported; `engine_model` walks `ENGINE_MODEL_PATHS` (vLLM's own `_get_driver_model_for_cleanup` route first) and raises a typed error naming every path it tried |
| the image-side API check | `test_the_probe_engine_check_names_the_api_the_probe_depends_on` and the wrapper's `test_..._checks_the_engine_api_before_loading_weights` fail on the unfixed tree (`AttributeError: module ... has no attribute 'check_engine'`; the wrapper ran the probe without the check) | `--check-engine`: the single-process switch is honoured, `LLMEngine.__init__` still assigns `model_executor` from the in-process engine core, `LLM.encode`/`apply_model` exist; the wrapper runs it before the probe and stops the job there |
| the recorder | `test_the_probe_records_a_forward_that_a_decorated_call_bypasses` fails on the unfixed tree (`AttributeError: ... no attribute '_run_recorded'`); run 2's report held only `merger` | `_record` wraps the bound `forward` (vLLM's decorator replaces `__call__`, so hooks never fire on `Qwen3_VisionPatchEmbed`/`Qwen3_VisionBlock`), `_missing_recorded` records what a recording missed in the row and in the Markdown header |
| the singleton-dimension comparison | `test_the_probe_compares_tensors_that_differ_only_by_a_singleton_dimension` fails on the unfixed tree (`KeyError: 'max_abs_diff'`) | `stats()` squeezes singleton dimensions, records that it did, and still refuses a genuinely different shape |
| the wrapper's argv parses | run 5's node error (`topk_probe.py: error: unrecognized arguments: --dtype float32 --reference-dtype float32`); `test_the_topk_probe_wrapper_forwards_the_dtypes` fails on the unfixed tree (`AttributeError: ... no attribute 'build_parser'`) | `build_parser()` exposes the parser, the test parses the argv the wrapper actually built with it, and the `--dtype` / `--reference-dtype` flags exist |
| the reference subprocess' dtype | run 6's rows (`E1` unchanged at 3.58e-03, the tower compared across two precisions); `test_the_reference_subprocess_gets_its_own_dtype` fails on the unfixed tree | `_reference_command` builds the subprocess argv once and carries `--reference-dtype` |

## 10. Docs, CHANGELOG, public surface

- `docs/how-to/release-candidates.md` — the `--script` list gains `topk-probe`; a new section "The topk
  probe job (`--script topk-probe`)" documents the command, the check, the E0-E7 experiments, the knobs and
  the outputs.
- `handover/reports/investigate.md` — the harness's paths point at its new home, the wiring is marked
  landed.
- `CHANGELOG.md` — one bullet under `## 0.0.1` / `### Changed` (the submittable probe job, its check and
  the single-process engine).
- **Public surface: none.** No `__all__`, CLI command or flag, exit code or schema moved: `submit.sh`'s
  `--script` gains one value (its own argv validation), and the probe is node tooling in the unpublished
  `rcp-ndcg-test`.

## 11. For the next lanes

- **The owner's decision** (section 6) is the only thing between this family and a green stage 2; the
  numbers to decide on are in sections 4 and 6, and the recipe's notes should cite them.
- **The dtype route is closed** (section 6): the engine cannot serve fp32 (vLLM's GatedDeltaNet kernels
  refuse it, the same blocker the `pplx-embed-v2-context` recipe documents), so a follow-up must not
  re-try `serve.dtype: float32`; the fp32 probe run is the evidence.
- **The matched-kernel cell was run (lane topk-gdn) and is negative**: the engine's vendored
  flash-linear-attention prefill (`--gdn-prefill-backend triton`) and `cutedsl` engage and leave every
  per-token minimum unchanged, so the residual is not the GDN kernel algorithm and the disposition
  cannot be narrowed to a kernel pin. The text model's own remaining difference (inputs, positions,
  layer alignment, the 6 full-attention layers) is the next thing to localise.
- **The `bf16_floor` fix**: the floor must be the engine's own dtype round trip (`--dtype`), else every
  non-bf16 run's "first divergent layer" is a threshold artefact (the fp16 and fp32 reports both are).
- **E0 is still `not_run`** for this family: a recording pass over `topk-embed-v1-small` (the wave's own
  `RCP_RECORD_CORPUS=1`) makes the control real, and the probe then ties its engine to the wave's vectors.
- **The `known_deviation` route**: the recipe's `known_deviations` must never be used to paper over a
  vector anticorrelation without the measured amplification beside it — the investigate report's rule
  stands.
