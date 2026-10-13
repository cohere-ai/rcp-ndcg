# Lane topkrun — the topk probe becomes a job, runs on a node, and names the divergence

**Lane:** `lane/topkrun` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `ac7c4a5d`
(already the lane's tip: nothing to merge, no merge commit). **Gated:** `60cb5615` — `GATE: PASS` (every
step exit 0); the report commit below is docs-only.
**Tree:** the wiring in `submit.sh`, the probe's three files moved into the job package, the harness fixes
the node runs forced, the tests, the docs and the CHANGELOG. **Four node jobs run; nothing pushed;
`rfc-0001` not merged into.**

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
| 8 | the third node run | **the verdict** (section 3): all 27 tower tensors and 24 text layers recorded; the first divergent tensor is the vision tower's `block_0`, at bf16-rounding scale, amplified by the tower's own blow-up |

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
| 4 | after the squeeze fix | exit 0, report uploaded — **the verdict below** |

The wrapper runs `topk_probe.py --check-engine` before the probe; in the pod it printed
`{"vllm": "0.31.0", "single_process_switch": "VLLM_ENABLE_V1_MULTIPROCESSING=0", "problems": []}` and the
probe continued (run 2 onward). The same check passes locally against the image whose digest the submitter
resolves (`sha256:c1c9f6fd…`), so the pin, the plan and the check agree.

## 4. The probe's findings (run 4)

**E0, the control: `not_run`** — no `--control-vectors` file, because the family's observation corpus was
never recorded (the last recording pass did not cover it); the report says so in its header and its JSON,
and every number below is therefore *uncalibrated against the wave's own engine*. Honest, not green.

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

**E6, end to end.** The per-row minima reproduce the wave's pattern and its magnitude: row 21 0.5466 (wave
0.5473), row 22 0.0411 (0.0052), row 26 -0.1145 (-0.3270), row 20 (the negative control) 0.7356 (0.9094);
row 28's image document 0.5466 (0.5473). The in-process engine is the wave's engine.

## 5. Verdict and the proposed fix (not bounded in this lane)

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

## 6. Verification

- `uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q` — **98 passed** (13 of them this
  lane's).
- `ruff format --check . && ruff check .` — clean; `basedpyright` — 0 errors.
- `shellcheck` and `bash -n` on `submit.sh` and `topk_probe.sh` — clean.
- The engine check and the hook mechanism were proven **inside the pinned image on CPU** (the digest the
  submitter resolves): `--check-engine` prints `problems: []`; `Qwen3_VisionPatchEmbed.__call__` and
  `Qwen3_VisionBlock.__call__` are *not* `nn.Module.__call__` while the merger's is; `Qwen3_VisionTransformer.forward`
  calls `patch_embed`, `blocks`, `merger`.
- `timeout 7200 bin/gate lane/topkrun`:
  - `60cb5615` — **GATE: PASS** (ruff-check/format, basedpyright, the root suite 4145 passed/155 skipped,
    contract+docs 413/107, mkdocs, test-pkg 1206/229, recipes, vllm-pkg 50, vllm-models 110/8, run_all
    1022/987/35/0 + 67/67 + 82/82, public-names clean, clean).
  - `e2cef0d7` — FAIL at `test-pkg` with `exit=139` (SIGSEGV) at
    `test_wavegroups.py::test_the_cli_writes_the_lists_and_the_stage_map`; the same class of load flake the
    investigate lane recorded (its first attempt died the same way at a different test in the same file);
    the retry on `a9bbfb11` passed the same step, and the failure is not in this lane's tree.
  - `d542eca1` (the wiring alone) — **GATE: PASS**.
- Red-first for every fix is in sections 2 and 7.

## 7. The harness fixes the node forced (each with its red test)

| fix | red evidence | what the fix does |
|---|---|---|
| the in-process model accessor | run 1's traceback (`'LLMEngine' object has no attribute 'named_modules'`) | `force_single_process_engine()` sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` before vLLM is imported; `engine_model` walks `ENGINE_MODEL_PATHS` (vLLM's own `_get_driver_model_for_cleanup` route first) and raises a typed error naming every path it tried |
| the image-side API check | `test_the_probe_engine_check_names_the_api_the_probe_depends_on` and the wrapper's `test_..._checks_the_engine_api_before_loading_weights` fail on the unfixed tree (`AttributeError: module ... has no attribute 'check_engine'`; the wrapper ran the probe without the check) | `--check-engine`: the single-process switch is honoured, `LLMEngine.__init__` still assigns `model_executor` from the in-process engine core, `LLM.encode`/`apply_model` exist; the wrapper runs it before the probe and stops the job there |
| the recorder | `test_the_probe_records_a_forward_that_a_decorated_call_bypasses` fails on the unfixed tree (`AttributeError: ... no attribute '_run_recorded'`); run 2's report held only `merger` | `_record` wraps the bound `forward` (vLLM's decorator replaces `__call__`, so hooks never fire on `Qwen3_VisionPatchEmbed`/`Qwen3_VisionBlock`), `_missing_recorded` records what a recording missed in the row and in the Markdown header |
| the singleton-dimension comparison | `test_the_probe_compares_tensors_that_differ_only_by_a_singleton_dimension` fails on the unfixed tree (`KeyError: 'max_abs_diff'`) | `stats()` squeezes singleton dimensions, records that it did, and still refuses a genuinely different shape |

## 8. Docs, CHANGELOG, public surface

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

## 9. For the next lanes

- **The owner's decision** (section 5) is the only thing between this family and a green stage 2; the
  numbers to decide on are in section 4, and the recipe's notes should cite them.
- **If the comparison stays per-token**, the precision experiment (section 5, item 2) is one job: the same
  probe with the engine in fp32 (`RCP_TOPK_GPU_FRACTION` unchanged) — but note the probe's `LLM(...)` call
  hardcodes `dtype="bfloat16"`, so a follow-up must add the knob.
- **E0 is still `not_run`** for this family: a recording pass over `topk-embed-v1-small` (the wave's own
  `RCP_RECORD_CORPUS=1`) makes the control real, and the probe then ties its engine to the wave's vectors.
- **The `known_deviation` route**: the recipe's `known_deviations` must never be used to paper over a
  vector anticorrelation without the measured amplification beside it — the investigate report's rule
  stands.
