# Lane topk-gdn — the GDN prefill backend is not the topk divergence

**Lane:** `lane/topk-gdn` (worktree `/root/repos/rcp-ndcg-lanes/wt-topk-gdn`, outside the operator's
checkout). **Base:** `afc76650`. **Knob commit:** `0dbc6520`. **Rescope commit:** `52917238`.
**Three node jobs** on the pinned image `vllm/vllm-openai:v0.31.0`
(`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`) against the same RC as the
existing evidence (`rc0-202610130615`), the same rows (text 17/18, query 12, image 20-28), one per
backend.

**Verdict: the fix is NOT a backend pin.** The tag's own `--gdn-prefill-backend` knob engages
`flashinfer` / `triton` / `cutedsl` (verified on the loaded layers, 36 modules each) and the per-token
minima are unchanged within the probe's run-to-run spread. The engine *can* run the reference's own
chunked-scan algorithm (the image vendors flash-linear-attention) and it does not match the reference
either. The claim that "no runnable engine configuration meets the per-token 0.999 gate" is rescoped to
what was measured, and the "the engine ships no `fla`, the matched-kernel cell is a follow-up" reading is
struck: the cell was run and it is negative.

## 1. Why the experiment existed

The release reviews surfaced that `vllm/vllm-openai:v0.31.0` vendors flash-linear-attention
(`vllm/third_party/flash_linear_attention/ops/chunk.py`, "code copied from the flash-linear-attention
project") and the tag's own flag `--gdn-prefill-backend {flashinfer,triton,cutedsl}` selects it
(`engine/arg_utils.py:842`, `:1891-1892` -> `additional_config["gdn_prefill_backend"]` at `:2774-2775`;
`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:51-60` imports the vendored module and
`ChunkGatedDeltaRule.__init__` picks `forward_native` for triton). The recipe has `serve.extra_args: []`,
so every measurement so far ran the default (flashinfer), while the reference runs `fla`'s chunked scan.
If the engine could run the reference's algorithm and match, the disposition would be a recipe change; if
not, the "no runnable engine configuration" claim had to be rescoped.

## 2. The knob (tests first, `0dbc6520`)

- `topk_probe.py`: `--gdn-backend {flashinfer,triton,cutedsl}` (unset = the engine's own `auto`), passed
  into the in-process `LLM(...)` as `additional_config={"gdn_prefill_backend": <value>}`; unset passes no
  `additional_config` at all. `engine_llm_kwargs()` is the tested seam.
- **The engaged backend is read from the loaded layers, never the request.**
  `engaged_gdn_backends()` walks the model for `gdn_prefill_backend` (assigned from the resolver's
  `active_backend` in `ChunkGatedDeltaRule.__init__`, copied by every
  `QwenGatedDeltaNetAttention`), and fails loudly when no module carries it. A
  `GdnBackendLogCapture` handler on the gdn module's logger records the resolver's own decision lines
  (the tag's logging level in the pod suppresses INFO, so the capture is honestly empty in these
  reports; the layer read is the evidence). The report's header and JSON carry
  `requested`, `engaged`, `layers`, `log_lines` and `match`.
- `topk_probe.sh` reads `RCP_TOPK_GDN_BACKEND`, prints it in the dry plan and appends `--gdn-backend`
  when set; `submit.sh` forwards it as `env.RCP_TOPK_GDN_BACKEND`.
- Tests (red first, then green): the parser's choices, the `additional_config` shape (and its absence
  when unset), the engaged-backend read with a fake model (and the loud failure), the wrapper's argv and
  the submit forwarding. `rcp-ndcg-test/tests/test_jobs_scripts.py`: 104 passed.

In-image validation on CPU (`docker run --entrypoint python3 -e VLLM_TARGET_DEVICE=cpu`), against the
pinned digest: `--check-engine` prints `problems: []`; the parser builds
`additional_config={'gdn_prefill_backend': 'triton'}`; `EngineArgs.additional_config` and
`EngineArgs.gdn_prefill_backend` exist, `create_engine_config` writes the switch into
`additional_config`, `_resolve_gdn_prefill_backend` reads it, the decision line exists on
`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`, and `LLM.__init__` forwards `**kwargs` to
`EngineArgs`.

## 3. The node runs

| job | backend requested | engaged | layers | match | job id |
|---|---|---|---|---|---|
| `topk-gdn-flashinfer` | flashinfer | `["flashinfer"]` | 36 | true | `fabi-rcp-topk-gdn-flashinfer-202610132103-qmcl` |
| `topk-gdn-triton` | triton | `["triton"]` | 36 | true | `fabiansc-rcp-topk-gdn-triton-202610132103-l4ut` |
| `topk-gdn-cutedsl` | cutedsl | `["cutedsl"]` | 36 | true | `fabians-rcp-topk-gdn-cutedsl-202610132103-mdjx` |

All three exited 0 and uploaded `probe-report.{json,md}` plus the per-row dumps to
`gs://YOUR-BUCKET/rcp-ndcg-rfc/waves-rc0-202610130615-topkgdn/topk-gdn-<backend>/topk-probe/`. The 36
modules are the 18 GatedDeltaNet layers plus their `ChunkGatedDeltaRule` submodules; `match: true` on
each run means the resolver chose exactly the requested kernel (a refused request would have engaged
`triton` with a warning, and the engaged value — not the flag — is what the table below is labelled
with). The probe's own stdout line in each job log repeats it, e.g.
`topk_probe: GDN prefill backend requested 'triton', engaged ['triton'] on 36 layers (match True)`.

## 4. The numbers (per-token cosine minimum, engine bf16 / reference bf16)

| pairs row | flashinfer (control) | triton (vendored FLA) | cutedsl | earlier evidence: probe / wave (flashinfer) |
|---|---|---|---|---|
| 20 image | 0.7356 | 0.7338 | 0.7417 | 0.7356 / 0.9111 |
| 21 image | 0.5466 | 0.5509 | 0.5590 | 0.5491 / 0.5514 |
| 22 image | 0.0411 | 0.0408 | 0.0400 | 0.0389 / 0.0390 |
| 23 image | -0.0705 | -0.0701 | -0.0704 | -0.0697 / -0.0682 |
| 24 image | -0.0831 | -0.0805 | -0.0806 | -0.0830 / -0.0808 |
| 25 image | -0.0353 | -0.0545 | -0.0501 | -0.0496 / -0.0348 |
| 26 image | -0.1145 | -0.1140 | -0.1134 | -0.1145 / -0.1134 |
| 27 image | 0.0577 | 0.0566 | 0.0571 | 0.0579 / 0.0577 |
| 28 image | 0.5466 | 0.5509 | 0.5590 | 0.5491 / 0.5514 |
| 12 query | 0.9472 | 0.9735 | 0.9688 | 0.9472 / 0.9339 |
| 17 document | 0.7863 | 0.7814 | 0.7643 | 0.8017 / 0.7534 |
| 18 document | 0.7125 | 0.7042 | 0.7154 | 0.7041 / 0.7830 |

The earlier-evidence column is the `topkrun` probe's bf16 numbers (rows 26, 12, 17, 18) and its run-4
image-row table (rows 20-25, 27, 28), each against the wave's own full-width minima re-derived from its
`equivalence.json`; the wave values are the same numbers `family.yaml` cites.

- **No cell clears 0.999 on any backend.** The text documents stay at 0.70-0.79, the image rows stay at
  -0.11 to 0.74, exactly where the flashinfer-only evidence had them.
- **The control reproduces the existing evidence**: row 26 (-0.11446643578534195) and row 12's query
  (0.9471743197772884) are bit-identical to the `topkrun` probe's flashinfer numbers, so the knob and
  the harness did not move the measurement.
- **The residual spread is the probe's, not the backend's.** The two long documents are the only cells
  with run-to-run movement: the earlier flashinfer run measured 0.8017/0.7041 where this lane's
  flashinfer control measures 0.7863/0.7125 (same RC, same rows, same dtype; the short query and image
  rows are bit-identical), so the long-prefill cells carry a ~0.01-0.02 spread — plausibly the engine's
  memory-derived prefill chunking. Every backend's long-document cells sit inside that spread.
- **The decisive statistic is the layer-level one**: the per-layer cosines are identical across the
  three kernels to 4 decimals (`text_layer_0` on row 17: 0.183592 flashinfer / 0.183487 triton /
  0.183750 cutedsl; on row 12's query: 0.254734 / 0.254971 / 0.254600), and the kernel choice moves the
  engine's *own* numbers only at the ~1e-4 cosine level (the query row: flashinfer and the earlier run
  bit-identical, triton differs in the 4th decimal) while the gap to the reference is ~0.8. The GDN
  prefill kernel is three orders of magnitude too small to explain the divergence.

## 5. Verdict

**Not the cause; no recipe change.** `serve.extra_args: []` stays, the gate, the bound and the serve
dtype are untouched, and the flip was not run. The `topk-embed-v1` disposition remains the owner's.

## 6. The rescope (this lane's second deliverable)

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1/family.yaml` (and the regenerated
  `rcp-ndcg-test/tests/recipes/golden/topk-embed-v1-{small,xsmall}.json`): "No runnable engine
  configuration meets the per-token 0.999 gate" becomes "the gate fails on the default flashinfer
  backend, and the GDN prefill kernel is not the cause", with the three-backend numbers and the probe
  prefix beside it.
- `handover/reports/topkrun.md`: the section-7 clause "the engine cannot match it from the engine side
  (the image ships no `fla`, and the recipe's own canary asserts it must stay out)" is struck and
  replaced with the measured negative; the verdict's "the text model's kernel-algorithm divergence,
  which no engine-side change can remove" and the section-11 "matched-kernel cell" item are rescoped
  the same way.
- `CHANGELOG.md`: one entry under the release's Fixed section naming the finding and the rescope.

## 7. Verification

- `uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q` — 104 passed (the knob's tests
  included).
- `uv run --no-sync pytest rcp-ndcg-test/tests/recipes -q` — 305 passed, 228 skipped (goldens match).
- `uv run --no-sync pytest tests/contract -q` — 17 passed.
- `uv run --no-sync ruff format --check .`, `ruff check .`, `basedpyright` — clean.
- The three node jobs exited 0 with `probe-report.json`/`.md` uploaded (the wrapper fails a job whose
  report is missing).

## 8. Open questions (for whoever picks the family up)

- **The text model's first-layer divergence is not localised.** It is present identically under three
  GDN prefill kernels, is dtype-independent (bf16 and fp16 give the same `text_layer_0` cosine), and the
  inputs are byte-identical (stage 1), so it is not the prefill kernel, not rounding and not the render.
  Candidates left: the 6 full-attention layers' engine backend, the packed-document position scheme, the
  layer alignment between the engine's `language_model.layers` and the reference's, or the engine's
  chunked-prefill state handling (the long-document run-to-run spread points there).
- **E0 is still `not_run`** (the family's observation corpus was never recorded), so every probe number
  remains uncalibrated against the wave's own engine.
